//! person2vec offline corpus pipeline.
//!
//! Run once (needs network the first time): downloads public-domain books from
//! Project Gutenberg, strips boilerplate, chunks them into ~200-word passages,
//! embeds every passage locally with all-MiniLM-L6-v2 via `fastembed`, projects
//! the vectors to 2D with a small power-iteration PCA, holds out a "mystery" set,
//! and writes `web/assets/person2vec.{json,bin}` for the web app to consume.
//!
//!   cargo run -p corpus --release

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::Duration;

use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use regex::Regex;
use serde::Deserialize;
use shared::{dot, l2_normalize, AuthorMeta, Bundle, Meta, ModelInfo, Passage, Results};

/// Target words per passage.
const PASSAGE_WORDS: usize = 200;
/// Minimum words to keep a passage (drops ragged tail/heading chunks).
const MIN_PASSAGE_WORDS: usize = 130;
/// Cap passages per author so classes stay balanced and the bundle stays small.
const MAX_PER_AUTHOR: usize = 130;
/// Held-out passages per author (the "new work" test set for the prediction reel).
const MYSTERY_PER_AUTHOR: usize = 4;
/// Polite delay between mirror requests.
const FETCH_DELAY: Duration = Duration::from_secs(1);
const USER_AGENT: &str =
    "person2vec-corpus/0.1 (educational authorship-embedding demo; low volume)";
/// Project Gutenberg mirrors (www.gutenberg.org rate-limits/blocks bulk robots, so
/// we use the standard mirror file tree over HTTP).
const MIRRORS: &[&str] = &[
    "http://aleph.gutenberg.org",
    "http://gutenberg.pglaf.org",
    "http://mirror.csclub.uwaterloo.ca/gutenberg",
];

#[derive(Deserialize)]
struct Config {
    authors: Vec<AuthorCfg>,
}
#[derive(Deserialize)]
struct AuthorCfg {
    name: String,
    #[serde(default)]
    gender: String,
    #[serde(default)]
    country: String,
    books: Vec<BookCfg>,
}
#[derive(Deserialize)]
struct BookCfg {
    id: u32,
    title: String,
    /// Optional series name; books sharing a series are held out together.
    #[serde(default)]
    series: Option<String>,
}

fn main() {
    let root = manifest_dir();
    let cache_dir = root.join(".cache");
    let _ = fs::create_dir_all(&cache_dir);

    let cfg: Config = toml::from_str(
        &fs::read_to_string(root.join("authors.toml")).expect("read authors.toml"),
    )
    .expect("parse authors.toml");

    let start_re =
        Regex::new(r"(?im)^\*\*\*\s*START OF TH.*?PROJECT GUTENBERG.*?\*\*\*\s*$").unwrap();
    let end_re = Regex::new(r"(?im)^\*\*\*\s*END OF TH.*?PROJECT GUTENBERG.*?\*\*\*\s*$").unwrap();

    // ---- 1..4: fetch, strip, chunk, sample per author --------------------
    let mut authors: Vec<AuthorMeta> = Vec::new();
    let mut passages: Vec<Passage> = Vec::new();
    let mut texts: Vec<String> = Vec::new(); // parallel to `passages`, fed to the embedder

    for (author_id, a) in cfg.authors.iter().enumerate() {
        authors.push(AuthorMeta {
            id: author_id,
            name: a.name.clone(),
            color: author_color(author_id),
            gender: a.gender.clone(),
            birth_country: a.country.clone(),
        });

        // Gather (book_title, series, passage) across all of this author's books.
        let mut collected: Vec<(String, String, String)> = Vec::new();
        for book in &a.books {
            let Some(raw) = fetch_book(book.id, &cache_dir) else {
                eprintln!("  !! skipping {} (id {}): fetch failed", book.title, book.id);
                continue;
            };
            let body = strip_boilerplate(&raw, &start_re, &end_re);
            let chunks = chunk_passages(&body);
            println!("  {} — {}: {} passages", a.name, book.title, chunks.len());
            let series = book.series.clone().unwrap_or_else(|| book.title.clone());
            for c in chunks {
                collected.push((book.title.clone(), series.clone(), c));
            }
        }

        // Evenly sample down to the per-author cap (spreads across books/plot).
        let sampled = sample_even(collected, MAX_PER_AUTHOR);
        let n = sampled.len();
        for (idx, (title, series, text)) in sampled.into_iter().enumerate() {
            // Mark spread-out passages per author as held-out "new work".
            let positions = [n / 5, 2 * n / 5, 3 * n / 5, 4 * n / 5];
            let is_mystery = n > 8
                && positions.contains(&idx)
                && count_mystery(&passages, author_id) < MYSTERY_PER_AUTHOR;
            texts.push(text.clone());
            passages.push(Passage {
                author_id,
                book_title: title,
                series,
                text,
                x: 0.0,
                y: 0.0,
                is_mystery,
            });
        }
        println!("  => {} total: {} passages", a.name, n);
    }

    assert!(!passages.is_empty(), "no passages collected; check network/IDs");
    println!("\nCollected {} passages. Embedding with each model...", passages.len());

    // ---- 5..8: embed the SAME corpus with each model, project, serialize --
    struct ModelSpec {
        key: &'static str,
        model: EmbeddingModel,
        name: &'static str,
        size: &'static str,
    }
    // A little size ladder: same-size-different-model (MiniLM vs BGE-small) plus a
    // bigger model (BGE-base, 768-d). Add BGELargeENV15 (1024-d, ~1.3GB) for more.
    let specs = [
        ModelSpec { key: "minilm", model: EmbeddingModel::AllMiniLML6V2, name: "all-MiniLM-L6-v2", size: "tiny · 23M params" },
    ];

    let mut model_infos: Vec<ModelInfo> = Vec::new();
    for spec in specs {
        println!("\n=== {} ===", spec.name);
        let mut embedder = match TextEmbedding::try_new(
            TextInitOptions::new(spec.model).with_show_download_progress(true),
        ) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("  !! skipping {}: {e}", spec.name);
                continue;
            }
        };
        let embeddings = match embedder.embed(texts.clone(), None) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("  !! embed failed for {}: {e}", spec.name);
                continue;
            }
        };
        let dim = embeddings[0].len();
        let mut vectors = Vec::with_capacity(embeddings.len() * dim);
        for e in &embeddings {
            vectors.extend_from_slice(e);
        }

        // Per-model PCA to 2D for the map.
        let coords = pca_2d(&vectors, passages.len(), dim);
        let mut passages_m = passages.clone();
        for (p, (x, y)) in passages_m.iter_mut().zip(coords) {
            p.x = x;
            p.y = y;
        }

        let mut bundle = Bundle {
            meta: Meta {
                dim,
                authors: authors.clone(),
                passages: passages_m,
                results: Results::default(),
            },
            vectors,
        };
        // Precompute the whole familiarity ladder so the browser just displays it.
        let results = shared::compute_results(&bundle);
        println!("  dim {}  ·  familiarity ladder:", dim);
        for rung in &results.rungs {
            println!("    {:<24} {:.1}%", rung.label, rung.accuracy * 100.0);
        }
        bundle.meta.results = results;
        write_model(spec.key, &bundle);
        model_infos.push(ModelInfo {
            key: spec.key.to_string(),
            name: spec.name.to_string(),
            dim,
            size: spec.size.to_string(),
        });
    }

    assert!(!model_infos.is_empty(), "no models embedded successfully");
    write_manifest(&model_infos);
}

// ---------------------------------------------------------------------------
// Fetching + cleaning
// ---------------------------------------------------------------------------

fn fetch_book(id: u32, cache_dir: &Path) -> Option<String> {
    let cache_file = cache_dir.join(format!("{id}.txt"));
    if let Ok(s) = fs::read_to_string(&cache_file) {
        if s.len() > 1000 {
            return Some(s);
        }
    }
    // Standard Gutenberg mirror file-tree path: digits of the id (except the last)
    // become directories, then the id itself, then `{id}{suffix}`.
    // Prefer the UTF-8 `-0.txt`, fall back to `.txt`, then latin-1 `-8.txt`.
    let dir = mirror_dir(id);
    for base in MIRRORS {
        for suffix in ["-0.txt", ".txt", "-8.txt"] {
            let url = format!("{base}/{dir}/{id}{suffix}");
            println!("  fetching {url}");
            let body = ureq::get(&url)
                .set("User-Agent", USER_AGENT)
                .call()
                .ok()
                .and_then(|resp| {
                    let mut bytes = Vec::new();
                    resp.into_reader()
                        .take(20 * 1024 * 1024)
                        .read_to_end(&mut bytes)
                        .ok()
                        .map(|_| decode(bytes))
                });
            sleep(FETCH_DELAY);
            if let Some(body) = body {
                if body.len() > 1000 && !body.contains("<title>404") {
                    let _ = fs::write(&cache_file, &body);
                    return Some(body);
                }
            }
        }
    }
    None
}

/// Mirror directory for an ebook id, e.g. 1342 -> "1/3/4/1342", 98 -> "9/98".
fn mirror_dir(id: u32) -> String {
    let s = id.to_string();
    if s.len() == 1 {
        return format!("0/{s}");
    }
    let mut dir = String::new();
    for ch in s[..s.len() - 1].chars() {
        dir.push(ch);
        dir.push('/');
    }
    dir.push_str(&s);
    dir
}

/// Decode bytes as UTF-8, falling back to latin-1 for older `-8.txt` files.
fn decode(bytes: Vec<u8>) -> String {
    String::from_utf8(bytes).unwrap_or_else(|e| e.into_bytes().iter().map(|&b| b as char).collect())
}

fn strip_boilerplate(text: &str, start_re: &Regex, end_re: &Regex) -> String {
    let start = start_re.find(text).map(|m| m.end()).unwrap_or(0);
    let end = end_re.find(text).map(|m| m.start()).unwrap_or(text.len());
    if end > start {
        text[start..end].to_string()
    } else {
        text.to_string()
    }
}

/// Split prose into ~PASSAGE_WORDS-word passages on paragraph boundaries.
fn chunk_passages(text: &str) -> Vec<String> {
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    let mut passages = Vec::new();
    let mut buf: Vec<String> = Vec::new();
    let mut words = 0usize;

    for para in normalized.split("\n\n") {
        // Collapse intra-paragraph line breaks into spaces.
        let para: String = para.split_whitespace().collect::<Vec<_>>().join(" ");
        if para.is_empty() {
            continue;
        }
        words += para.split_whitespace().count();
        buf.push(para);
        if words >= PASSAGE_WORDS {
            passages.push(buf.join(" "));
            buf.clear();
            words = 0;
        }
    }
    if words >= MIN_PASSAGE_WORDS {
        passages.push(buf.join(" "));
    }

    passages
        .into_iter()
        .filter(|p| p.split_whitespace().count() >= MIN_PASSAGE_WORDS)
        .collect()
}

/// Evenly sample `items` down to at most `max`, preserving order.
fn sample_even<T: Clone>(items: Vec<T>, max: usize) -> Vec<T> {
    if items.len() <= max {
        return items;
    }
    let step = items.len() as f64 / max as f64;
    (0..max)
        .map(|i| items[((i as f64) * step) as usize].clone())
        .collect()
}

fn count_mystery(passages: &[Passage], author_id: usize) -> usize {
    passages
        .iter()
        .filter(|p| p.author_id == author_id && p.is_mystery)
        .count()
}

// ---------------------------------------------------------------------------
// PCA (power iteration with deflation) — offline only, no linalg dependency
// ---------------------------------------------------------------------------

fn pca_2d(vectors: &[f32], n: usize, dim: usize) -> Vec<(f32, f32)> {
    // Mean-center a working copy.
    let mut mean = vec![0f32; dim];
    for i in 0..n {
        let row = &vectors[i * dim..(i + 1) * dim];
        for k in 0..dim {
            mean[k] += row[k];
        }
    }
    for m in mean.iter_mut() {
        *m /= n as f32;
    }
    let mut centered = vec![0f32; n * dim];
    for i in 0..n {
        for k in 0..dim {
            centered[i * dim + k] = vectors[i * dim + k] - mean[k];
        }
    }

    let pcs = top_principal_components(&centered, n, dim, 2, 150);
    (0..n)
        .map(|i| {
            let row = &centered[i * dim..(i + 1) * dim];
            (dot(row, &pcs[0]), dot(row, &pcs[1]))
        })
        .collect()
}

fn top_principal_components(
    centered: &[f32],
    n: usize,
    dim: usize,
    k: usize,
    iters: usize,
) -> Vec<Vec<f32>> {
    let mut pcs: Vec<Vec<f32>> = Vec::new();
    for _ in 0..k {
        // Deterministic non-degenerate seed.
        let mut v: Vec<f32> = (0..dim).map(|i| ((i as f32 * 0.7).sin()).abs() + 0.1).collect();
        gram_schmidt(&mut v, &pcs);
        l2_normalize(&mut v);
        for _ in 0..iters {
            // Cᵀv without forming the covariance: Cᵀv = Σ_i row_i (row_i · v)
            let mut cv = vec![0f32; dim];
            for i in 0..n {
                let row = &centered[i * dim..(i + 1) * dim];
                let proj = dot(row, &v);
                for kk in 0..dim {
                    cv[kk] += proj * row[kk];
                }
            }
            gram_schmidt(&mut cv, &pcs);
            l2_normalize(&mut cv);
            v = cv;
        }
        pcs.push(v);
    }
    pcs
}

fn gram_schmidt(v: &mut [f32], basis: &[Vec<f32>]) {
    for b in basis {
        let d = dot(v, b);
        for i in 0..v.len() {
            v[i] -= d * b[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

fn write_model(key: &str, bundle: &Bundle) {
    let assets = assets_dir();
    let _ = fs::create_dir_all(&assets);
    let json = serde_json::to_vec(&bundle.meta).expect("serialize meta");
    let bin = shared::vectors_to_bytes(&bundle.vectors);
    fs::write(assets.join(format!("person2vec-{key}.json")), &json).expect("write json");
    fs::write(assets.join(format!("person2vec-{key}.bin")), &bin).expect("write bin");
    println!(
        "  wrote person2vec-{key}.{{json,bin}}  ({:.1} + {:.1} MB)",
        json.len() as f64 / 1e6,
        bin.len() as f64 / 1e6
    );
}

fn write_manifest(models: &[ModelInfo]) {
    let manifest = shared::Manifest {
        default: models[0].key.clone(),
        models: models.to_vec(),
    };
    let json = serde_json::to_vec(&manifest).expect("serialize manifest");
    fs::write(assets_dir().join("person2vec-models.json"), json).expect("write manifest");
    println!("\nWrote person2vec-models.json ({} models).", models.len());
}

fn assets_dir() -> PathBuf {
    manifest_dir().join("..").join("web").join("assets")
}

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Distinct per-author color: golden-ratio hue spacing so even ~50 authors stay apart.
fn author_color(i: usize) -> String {
    let hue = (i as f32 * 137.508) % 360.0;
    hsl_to_hex(hue, 0.62, 0.52)
}

fn hsl_to_hex(h: f32, s: f32, l: f32) -> String {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let hp = h / 60.0;
    let x = c * (1.0 - ((hp % 2.0) - 1.0).abs());
    let (r, g, b) = match hp as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = l - c / 2.0;
    let to = |v: f32| ((v + m) * 255.0).round().clamp(0.0, 255.0) as u8;
    format!("#{:02x}{:02x}{:02x}", to(r), to(g), to(b))
}
