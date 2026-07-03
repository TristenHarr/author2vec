//! Shared offline helpers for the person2vec corpus pipelines.
//!
//! The prose pipeline (`main.rs`) and the code pipeline (`bin/coders.rs`) both turn a
//! set of `AuthorMeta` + `Passage` + parallel `texts` into embedded, PCA-projected,
//! scored bundles written to `web/assets/`. Everything model/IO/PCA/output that they
//! share lives here so a second dataset is a thin front-end over the same back half.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use shared::{
    compute_results_with_labels, dot, l2_normalize, AuthorMeta, Bundle, DatasetInfo,
    DatasetManifest, LadderLabels, Meta, ModelInfo, Passage, Results,
};

pub mod coders;

// ---------------------------------------------------------------------------
// Embedding driver — embed the SAME corpus with each model, project, score, write
// ---------------------------------------------------------------------------

/// One embedding model to run over the corpus.
pub struct ModelSpec {
    /// Filename key: `person2vec-<key>.{json,bin}`.
    pub key: String,
    pub model: EmbeddingModel,
    pub name: String,
    /// Human size label, e.g. "tiny · 23M params".
    pub size: String,
}

/// Embed `texts` with one model, PCA to 2D, and compute results with `labels`. Returns
/// the assembled, scored `Bundle` (not written to disk) — or None if the model failed
/// to load/embed. The audit uses this to embed scrub-on / scrub-off variants.
pub fn embed_one(
    spec: &ModelSpec,
    authors: &[AuthorMeta],
    passages: &[Passage],
    texts: &[String],
    labels: &LadderLabels,
) -> Option<Bundle> {
    println!("\n=== {} ===", spec.name);
    let mut embedder = match TextEmbedding::try_new(
        TextInitOptions::new(spec.model.clone()).with_show_download_progress(true),
    ) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("  !! skipping {}: {e}", spec.name);
            return None;
        }
    };
    let embeddings = match embedder.embed(texts.to_vec(), None) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("  !! embed failed for {}: {e}", spec.name);
            return None;
        }
    };
    let dim = embeddings[0].len();
    let mut vectors = Vec::with_capacity(embeddings.len() * dim);
    for e in &embeddings {
        vectors.extend_from_slice(e);
    }

    // PCA to 2D for the map.
    let coords = pca_2d(&vectors, passages.len(), dim);
    let mut passages_m = passages.to_vec();
    for (p, (x, y)) in passages_m.iter_mut().zip(coords) {
        p.x = x;
        p.y = y;
    }

    let mut bundle = Bundle {
        meta: Meta {
            dim,
            authors: authors.to_vec(),
            passages: passages_m,
            results: Results::default(),
        },
        vectors,
    };
    // Precompute the whole familiarity ladder so the browser just displays it.
    let results = compute_results_with_labels(&bundle, labels);
    println!("  dim {}  ·  familiarity ladder:", dim);
    for rung in &results.rungs {
        println!("    {:<28} {:.1}%", rung.label, rung.accuracy * 100.0);
    }
    bundle.meta.results = results;
    Some(bundle)
}

/// Embed `texts` with each spec and write one bundle per model. Returns the
/// `ModelInfo` list of models that succeeded.
pub fn embed_and_write(
    specs: &[ModelSpec],
    authors: &[AuthorMeta],
    passages: &[Passage],
    texts: &[String],
    labels: &LadderLabels,
) -> Vec<ModelInfo> {
    let mut model_infos: Vec<ModelInfo> = Vec::new();
    for spec in specs {
        if let Some(bundle) = embed_one(spec, authors, passages, texts, labels) {
            write_model(&spec.key, &bundle);
            model_infos.push(ModelInfo {
                key: spec.key.clone(),
                name: spec.name.clone(),
                dim: bundle.meta.dim,
                size: spec.size.clone(),
            });
        }
    }
    model_infos
}

// ---------------------------------------------------------------------------
// Sampling + mystery set
// ---------------------------------------------------------------------------

/// Evenly sample `items` down to at most `max`, preserving order.
pub fn sample_even<T: Clone>(items: Vec<T>, max: usize) -> Vec<T> {
    if items.len() <= max {
        return items;
    }
    let step = items.len() as f64 / max as f64;
    (0..max)
        .map(|i| items[((i as f64) * step) as usize].clone())
        .collect()
}

/// How many held-out "mystery" passages an author already has.
pub fn count_mystery(passages: &[Passage], author_id: usize) -> usize {
    passages
        .iter()
        .filter(|p| p.author_id == author_id && p.is_mystery)
        .count()
}

/// Mark up to `per_author` spread-out passages per author as held-out "new work".
/// Picks positions n/5, 2n/5, 3n/5, 4n/5 among an author's passages (in order), only
/// for authors with more than 8 passages — identical selection to the original inline
/// prose logic, so both datasets share it.
pub fn mark_mystery(passages: &mut [Passage], per_author: usize) {
    let mut by_author: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, p) in passages.iter().enumerate() {
        by_author.entry(p.author_id).or_default().push(i);
    }
    for (_a, idxs) in by_author {
        let n = idxs.len();
        if n <= 8 {
            continue;
        }
        let positions = [n / 5, 2 * n / 5, 3 * n / 5, 4 * n / 5];
        let mut marked = 0usize;
        for pos in positions {
            if marked >= per_author {
                break;
            }
            passages[idxs[pos]].is_mystery = true;
            marked += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// PCA (power iteration with deflation) — offline only, no linalg dependency
// ---------------------------------------------------------------------------

pub fn pca_2d(vectors: &[f32], n: usize, dim: usize) -> Vec<(f32, f32)> {
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

pub fn write_model(key: &str, bundle: &Bundle) {
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

pub fn write_manifest(models: &[ModelInfo]) {
    let manifest = shared::Manifest {
        default: models[0].key.clone(),
        models: models.to_vec(),
    };
    let json = serde_json::to_vec(&manifest).expect("serialize manifest");
    fs::write(assets_dir().join("person2vec-models.json"), json).expect("write manifest");
    println!("\nWrote person2vec-models.json ({} models).", models.len());
}

/// The static list of dataset lenses the site offers. Both pipelines write the same
/// file, so running either keeps `person2vec-datasets.json` correct. The authors
/// dataset keeps the existing `minilm` file key (no rename / re-embed needed).
pub fn site_datasets() -> DatasetManifest {
    DatasetManifest {
        // Coders is the default view; Authors is the tab you switch to.
        default: "coders".to_string(),
        datasets: vec![
            DatasetInfo {
                key: "coders".into(),
                label: "Coders".into(),
                blurb: "code · open-source commits".into(),
            },
            DatasetInfo {
                key: "minilm".into(),
                label: "Authors".into(),
                blurb: "prose · public-domain books".into(),
            },
        ],
    }
}

pub fn write_datasets_manifest() {
    let json = serde_json::to_vec(&site_datasets()).expect("serialize datasets");
    fs::write(assets_dir().join("person2vec-datasets.json"), json).expect("write datasets manifest");
    println!("Wrote person2vec-datasets.json.");
}

pub fn assets_dir() -> PathBuf {
    manifest_dir().join("..").join("web").join("assets")
}

pub fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Distinct per-author color: golden-ratio hue spacing so even ~50 authors stay apart.
pub fn author_color(i: usize) -> String {
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
