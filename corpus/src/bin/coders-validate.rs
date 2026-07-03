//! Honesty audit of the `coders` (code-authorship) dataset — the credibility layer.
//!
//!   cargo run -p corpus --bin coders-validate --release
//!   cargo run -p corpus --bin coders-validate --release -- --ablation   # + re-embed controls
//!
//! Answers two questions a skeptic will ask:
//!   1. Is the accuracy real personal style, or just detecting the repo/language?
//!      → the leave-repo-out rung + the same-repo/different-devs control.
//!   2. Did we leak the author's NAME into the embedded text?
//!      → the residual-identity scan (fast) and the scrub ablation (--ablation).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use fastembed::EmbeddingModel;

use corpus::{coders, embed_one, manifest_dir, ModelSpec};
use shared::{dot, l2_normalize, Bundle, Meta};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let config = arg_value(&args, "--config")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir().join("coders.toml"));
    let assets = manifest_dir().join("..").join("web").join("assets");

    let bundle = load(&assets, "coders");
    let n = bundle.n_authors();
    let baseline = 100.0 / n as f32;
    println!("######### coders · dim {} #########", bundle.meta.dim);
    println!(
        "{} passages · {} coders · random baseline {:.1}%\n",
        bundle.len(),
        n,
        baseline
    );

    ladder(&bundle, baseline);
    shuffle_control(&bundle, baseline);
    residual_scan(&config, &bundle);
    same_repo_multi_dev(&bundle);
    vector_sanity(&bundle);

    if args.iter().any(|a| a == "--ablation") {
        ablation(&config);
    } else {
        println!("(Run with --ablation to embed the scrub-on/off + name-injection controls.)");
    }
}

// ---------------------------------------------------------------------------

fn load(assets: &Path, key: &str) -> Bundle {
    let meta: Meta = serde_json::from_slice(
        &fs::read(assets.join(format!("person2vec-{key}.json")))
            .unwrap_or_else(|e| panic!("read person2vec-{key}.json ({e}); generate it first")),
    )
    .expect("parse json");
    let vectors = shared::vectors_from_bytes(
        &fs::read(assets.join(format!("person2vec-{key}.bin"))).expect("read bin"),
    );
    Bundle { meta, vectors }
}

/// [1] The shipped familiarity ladder: coder → repo → commit → seen.
fn ladder(bundle: &Bundle, baseline: f32) {
    let seen = shared::loocv_nearest_centroid(bundle);
    let commit = shared::loocv_leave_book_out(bundle);
    let repo = shared::loocv_leave_series_out(bundle);
    let coder = shared::loocv_leave_author_out(bundle);
    println!("[1] Familiarity ladder (random baseline {:.1}%):", baseline);
    println!("    has read this commit      {:5.1}%", seen.accuracy * 100.0);
    println!("    never seen this commit    {:5.1}%", commit.accuracy * 100.0);
    println!(
        "    never seen this repo      {:5.1}%   <- confound headline (style, not project vocab?)",
        repo.accuracy * 100.0
    );
    println!("    never seen this coder     {:5.1}%\n", coder.accuracy * 100.0);
}

/// [2] Randomize author labels — accuracy MUST collapse to ~baseline or a bug leaks.
fn shuffle_control(bundle: &Bundle, baseline: f32) {
    let mut labels: Vec<usize> = bundle.meta.passages.iter().map(|p| p.author_id).collect();
    let mut rng: u64 = 0x9E3779B97F4A7C15;
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
    let shuffled = Bundle { meta, vectors: bundle.vectors.clone() };
    let loo = shared::loocv_nearest_centroid(&shuffled);
    println!("[2] Label-shuffle control (labels randomized):");
    println!(
        "    {:.1}%  — should be ~{:.1}%; confirms no bug inflates the score\n",
        loo.accuracy * 100.0,
        baseline
    );
}

/// [3] Scan the SHIPPED (already-scrubbed) passages for any roster name that survived.
fn residual_scan(config: &Path, bundle: &Bundle) {
    let texts: Vec<String> = bundle.meta.passages.iter().map(|p| p.text.clone()).collect();
    let (hits, toks) = coders::residual_identity_hits(config, &texts);
    println!("[3] Residual-identity scan (names surviving the scrub):");
    println!(
        "    {hits} / {} passages contain a roster identity substring {toks:?}",
        texts.len()
    );
    println!("    (target ≈ 0 — confirms we embedded style, not names)\n");
}

/// [4] Same repo, ≥2 roster devs: domain/vocabulary held constant, so accuracy above
/// chance is PURE personal style — the honest arbiter of the language/project confound.
fn same_repo_multi_dev(bundle: &Bundle) {
    let mut by_repo: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, p) in bundle.meta.passages.iter().enumerate() {
        if p.is_mystery {
            continue;
        }
        by_repo.entry(p.series.as_str()).or_default().push(i);
    }
    println!("[4] Same-repo, multiple-devs (pure personal style, domain held constant):");
    let mut any = false;
    for (repo, idxs) in &by_repo {
        let devs: BTreeSet<usize> =
            idxs.iter().map(|&i| bundle.meta.passages[i].author_id).collect();
        if devs.len() < 2 {
            continue;
        }
        any = true;
        let (correct, total) = loocv_within(bundle, idxs);
        println!(
            "    {repo:<14} {} devs · {:.1}%  (chance {:.1}%)",
            devs.len(),
            correct as f32 / total.max(1) as f32 * 100.0,
            100.0 / devs.len() as f32
        );
    }
    if !any {
        println!("    (no repo has ≥2 roster devs — add a shared-repo cluster to enable this)");
    }
    println!();
}

/// Leave-one-out nearest-centroid restricted to a set of passage indices.
fn loocv_within(bundle: &Bundle, idxs: &[usize]) -> (u32, u32) {
    let dim = bundle.meta.dim;
    let authors: Vec<usize> = idxs
        .iter()
        .map(|&i| bundle.meta.passages[i].author_id)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let mut sums: HashMap<usize, Vec<f32>> = HashMap::new();
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for &i in idxs {
        let a = bundle.meta.passages[i].author_id;
        let v = bundle.vector(i);
        let s = sums.entry(a).or_insert_with(|| vec![0.0; dim]);
        for k in 0..dim {
            s[k] += v[k];
        }
        *counts.entry(a).or_insert(0) += 1;
    }
    let (mut correct, mut total) = (0u32, 0u32);
    for &i in idxs {
        let a = bundle.meta.passages[i].author_id;
        let v = bundle.vector(i);
        let mut best = None;
        let mut best_sim = f32::NEG_INFINITY;
        for &b in &authors {
            let mut c = sums[&b].clone();
            let mut cn = counts[&b];
            if b == a {
                for k in 0..dim {
                    c[k] -= v[k];
                }
                cn -= 1;
            }
            if cn == 0 {
                continue;
            }
            for x in c.iter_mut() {
                *x /= cn as f32;
            }
            l2_normalize(&mut c);
            let sim = dot(v, &c);
            if sim > best_sim {
                best_sim = sim;
                best = Some(b);
            }
        }
        if let Some(pred) = best {
            total += 1;
            if pred == a {
                correct += 1;
            }
        }
    }
    (correct, total)
}

/// [5] L2 norms ≈ 1, exact + near duplicate rate, buffer length.
fn vector_sanity(bundle: &Bundle) {
    let dim = bundle.meta.dim;
    let (mut min_norm, mut max_norm) = (f32::MAX, f32::MIN);
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
    println!("[5] Vector sanity:");
    println!("    L2 norms in [{min_norm:.4}, {max_norm:.4}]  (≈1.0 ⇒ cosine == dot)");
    println!("    exact-duplicate passages: {dups}");
    println!(
        "    buffer {} == passages*dim {} : {}\n",
        bundle.vectors.len(),
        bundle.len() * dim,
        bundle.vectors.len() == bundle.len() * dim
    );
}

/// [6] The decisive leakage control: re-embed with scrubbing ON vs OFF and inject
/// names. Uses all-MiniLM-L6-v2 for speed — the DELTA is what matters, not the level.
fn ablation(config: &Path) {
    println!("[6] Scrub ablation + name-injection (re-embeds; cached clones)…");
    let scrubbed = coders::collect(config, true);
    let unscrubbed = coders::collect(config, false);
    let labels = coders::code_ladder_labels();
    let spec = ModelSpec {
        key: "ablate".into(),
        model: EmbeddingModel::AllMiniLML6V2,
        name: "all-MiniLM-L6-v2 (control)".into(),
        size: String::new(),
    };

    let bs = embed_one(&spec, &scrubbed.authors, &scrubbed.passages, &scrubbed.texts, &labels);
    let bu = embed_one(&spec, &unscrubbed.authors, &unscrubbed.passages, &unscrubbed.texts, &labels);
    let (Some(bs), Some(bu)) = (bs, bu) else {
        eprintln!("    ablation embedding unavailable");
        return;
    };
    let rs = shared::loocv_leave_series_out(&bs).accuracy * 100.0;
    let ru = shared::loocv_leave_series_out(&bu).accuracy * 100.0;
    println!(
        "    leave-repo-out: scrubbed {rs:.1}%  vs  unscrubbed {ru:.1}%  (Δ {:+.1})",
        ru - rs
    );
    println!("    small Δ ⇒ the signal is style, not literal names.");

    // Name-injection positive control: inject each coder's name into their passages.
    let inj: Vec<String> = scrubbed
        .passages
        .iter()
        .map(|p| {
            let nm = &scrubbed.authors[p.author_id].name;
            format!("{nm} {nm} {nm}\n{}", p.text)
        })
        .collect();
    if let Some(bi) = embed_one(&spec, &scrubbed.authors, &scrubbed.passages, &inj, &labels) {
        let ri = shared::loocv_leave_series_out(&bi).accuracy * 100.0;
        println!("    name-injection control: {ri:.1}%  (should jump well above scrubbed {rs:.1}%)");
        println!("    a big jump ⇒ the model WOULD exploit names, so the clean number is meaningful.");
    }
}

fn arg_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}
