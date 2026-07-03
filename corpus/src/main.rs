//! person2vec offline corpus pipeline (prose authors).
//!
//! Run once (needs network the first time): downloads public-domain books from
//! Project Gutenberg, strips boilerplate, chunks them into ~200-word passages,
//! embeds every passage locally with all-MiniLM-L6-v2 via `fastembed`, projects
//! the vectors to 2D with a small power-iteration PCA, holds out a "mystery" set,
//! and writes `web/assets/person2vec-minilm.{json,bin}` for the web app to consume.
//! The reusable back half (embed → PCA → score → write) lives in `corpus`'s lib so
//! the code dataset (`bin/coders.rs`) shares it.
//!
//!   cargo run -p corpus --release

use std::fs;
use std::io::Read;
use std::path::Path;
use std::thread::sleep;
use std::time::Duration;

use fastembed::EmbeddingModel;
use regex::Regex;
use serde::Deserialize;
use shared::{AuthorMeta, LadderLabels, Passage};

use corpus::{
    author_color, embed_and_write, manifest_dir, mark_mystery, sample_even,
    write_datasets_manifest, write_manifest, ModelSpec,
};

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
    #[serde(default)]
    raised: String,
    #[serde(default)]
    educated: String,
    #[serde(default)]
    college: String,
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
            raised: a.raised.clone(),
            educated: a.educated.clone(),
            college: a.college.clone(),
            traits: Vec::new(),
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
        for (title, series, text) in sampled {
            texts.push(text.clone());
            passages.push(Passage {
                author_id,
                book_title: title,
                series,
                text,
                x: 0.0,
                y: 0.0,
                is_mystery: false,
                authored: 0,
            });
        }
        println!("  => {} total: {} passages", a.name, n);
    }

    assert!(!passages.is_empty(), "no passages collected; check network/IDs");
    // Hold out spread-out passages per author as the "new work" test set.
    mark_mystery(&mut passages, MYSTERY_PER_AUTHOR);
    println!("\nCollected {} passages. Embedding with each model...", passages.len());

    // ---- 5..8: embed the SAME corpus with each model, project, serialize --
    let specs = [ModelSpec {
        key: "minilm".into(),
        model: EmbeddingModel::AllMiniLML6V2,
        name: "all-MiniLM-L6-v2".into(),
        size: "tiny · 23M params".into(),
    }];

    let model_infos = embed_and_write(&specs, &authors, &passages, &texts, &LadderLabels::prose());
    assert!(!model_infos.is_empty(), "no models embedded successfully");
    write_manifest(&model_infos);
    write_datasets_manifest();
}

// ---------------------------------------------------------------------------
// Fetching + cleaning (prose-specific)
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
