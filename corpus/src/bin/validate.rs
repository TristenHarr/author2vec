//! Offline audit of the generated person2vec dataset — no network, no re-embedding.
//! Loads `web/assets/person2vec.{json,bin}` and runs honesty checks:
//!   cargo run -p corpus --bin validate --release
//!
//! Key question it answers: is the 74.7% real *authorship* signal, or is the model
//! secretly identifying each BOOK by its vocabulary? The leave-one-book-out number
//! is the honest "pure style" figure.

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::PathBuf;

use shared::{dot, l2_normalize, Bundle, Meta};

fn main() {
    let assets = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("web")
        .join("assets");
    let manifest: shared::Manifest = serde_json::from_slice(
        &fs::read(assets.join("person2vec-models.json")).expect("read manifest"),
    )
    .expect("parse manifest");

    // Audit every model, so we can compare them side by side.
    for m in &manifest.models {
        let bundle = load(&assets, &m.key);
        let n = bundle.n_authors();
        let baseline = 100.0 / n as f32;
        println!("\n######### {} · dim {} · {} #########", m.name, m.dim, m.size);
        println!(
            "{} passages · {} authors · random baseline {:.1}%",
            bundle.len(),
            n,
            baseline
        );

        let loo = shared::loocv_nearest_centroid(&bundle);
        let knn = shared::loocv_knn(&bundle, 5);
        println!(
            "[1] LOO nearest-centroid {:.1}%  ·  k-NN(k=5) {:.1}%",
            loo.accuracy * 100.0,
            knn.accuracy * 100.0
        );
        leave_one_book_out(&bundle, baseline);
        shuffle_control(&bundle, baseline);
        vector_sanity(&bundle);
    }
}

fn load(assets: &std::path::Path, key: &str) -> Bundle {
    let meta: Meta = serde_json::from_slice(
        &fs::read(assets.join(format!("person2vec-{key}.json"))).expect("read json"),
    )
    .expect("parse json");
    let vectors = shared::vectors_from_bytes(
        &fs::read(assets.join(format!("person2vec-{key}.bin"))).expect("read bin"),
    );
    Bundle { meta, vectors }
}

/// Build normalized per-author centroids over the passages where `include(i)` holds.
/// `None` for an author with no included passages.
fn centroids(bundle: &Bundle, include: impl Fn(usize) -> bool) -> Vec<Option<Vec<f32>>> {
    let dim = bundle.meta.dim;
    let n = bundle.n_authors();
    let mut sums = vec![vec![0f32; dim]; n];
    let mut counts = vec![0usize; n];
    for (i, p) in bundle.meta.passages.iter().enumerate() {
        if !include(i) {
            continue;
        }
        let v = bundle.vector(i);
        for k in 0..dim {
            sums[p.author_id][k] += v[k];
        }
        counts[p.author_id] += 1;
    }
    (0..n)
        .map(|a| {
            if counts[a] == 0 {
                None
            } else {
                let mut c: Vec<f32> = sums[a].iter().map(|x| x / counts[a] as f32).collect();
                l2_normalize(&mut c);
                Some(c)
            }
        })
        .collect()
}

fn argmax(cents: &[Option<Vec<f32>>], v: &[f32]) -> Option<usize> {
    let mut best = None;
    let mut best_sim = f32::NEG_INFINITY;
    for (a, c) in cents.iter().enumerate() {
        if let Some(c) = c {
            let s = dot(v, c);
            if s > best_sim {
                best_sim = s;
                best = Some(a);
            }
        }
    }
    best
}

/// Leave-one-BOOK-out: hold out an entire book, so the held author's centroid comes
/// only from their *other* book. High accuracy here ⇒ the model captures the author's
/// style across topics, not just per-book vocabulary. This is the honest number.
fn leave_one_book_out(bundle: &Bundle, baseline: f32) {
    let mut books: BTreeMap<(usize, &str), Vec<usize>> = BTreeMap::new();
    for (i, p) in bundle.meta.passages.iter().enumerate() {
        books
            .entry((p.author_id, p.book_title.as_str()))
            .or_default()
            .push(i);
    }

    let mut correct = 0u32;
    let mut total = 0u32;
    let mut skipped = 0u32;
    for ((author, _title), idxs) in &books {
        let held: HashSet<usize> = idxs.iter().copied().collect();
        let cents = centroids(bundle, |i| !held.contains(&i));
        if cents[*author].is_none() {
            // Author has no other book to train on — can't fairly test.
            skipped += idxs.len() as u32;
            continue;
        }
        for &i in idxs {
            match argmax(&cents, bundle.vector(i)) {
                Some(pred) => {
                    total += 1;
                    if pred == *author {
                        correct += 1;
                    }
                }
                None => skipped += 1,
            }
        }
    }
    println!(
        "[2] Leave-one-BOOK-out (train on the author's OTHER book only):"
    );
    println!(
        "    {:.1}%  ({}/{})  — pure style, book/topic removed  [{} skipped, {:.1}% baseline]\n",
        correct as f32 / total.max(1) as f32 * 100.0,
        correct,
        total,
        skipped,
        baseline
    );
}

/// Randomize the author labels. Accuracy MUST collapse to ~baseline; if it doesn't,
/// a bug is leaking information and inflating the real score.
fn shuffle_control(bundle: &Bundle, baseline: f32) {
    let mut labels: Vec<usize> = bundle.meta.passages.iter().map(|p| p.author_id).collect();
    let mut rng: u64 = 0x9E3779B97F4A7C15; // fixed seed → deterministic
    for i in (1..labels.len()).rev() {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let j = (rng % (i as u64 + 1)) as usize;
        labels.swap(i, j);
    }
    let mut meta = bundle.meta.clone();
    for (p, &l) in meta.passages.iter_mut().zip(&labels) {
        p.author_id = l;
        p.is_mystery = false;
    }
    let shuffled = Bundle {
        meta,
        vectors: bundle.vectors.clone(),
    };
    let loo = shared::loocv_nearest_centroid(&shuffled);
    println!("[3] Label-shuffle control (author labels randomized):");
    println!(
        "    {:.1}%  ({}/{})  — should be ~{:.1}%; confirms no leakage inflates the score\n",
        loo.accuracy * 100.0,
        loo.correct,
        loo.total,
        baseline
    );
}

fn vector_sanity(bundle: &Bundle) {
    let dim = bundle.meta.dim;
    let mut min_norm = f32::MAX;
    let mut max_norm = f32::MIN;
    let mut seen: HashSet<Vec<u32>> = HashSet::new();
    let mut dups = 0u32;
    for i in 0..bundle.len() {
        let v = bundle.vector(i);
        let norm = dot(v, v).sqrt();
        min_norm = min_norm.min(norm);
        max_norm = max_norm.max(norm);
        if !seen.insert(v.iter().map(|x| x.to_bits()).collect()) {
            dups += 1;
        }
    }
    let expected = bundle.len() * dim;
    println!("[4] Vector sanity:");
    println!(
        "    L2 norms in [{:.4}, {:.4}]  (≈1.0 ⇒ cosine == dot, as assumed)",
        min_norm, max_norm
    );
    println!("    exact-duplicate passages: {dups}");
    println!(
        "    buffer len {} == passages*dim {} : {}",
        bundle.vectors.len(),
        expected,
        bundle.vectors.len() == expected
    );
}
