//! Identity ignition (Phase 2, flagship) → `web/assets/person2vec-ignition-<dataset>.json`.
//!
//! Question: does the encoder's internal representation COMMIT to a single author at a
//! characteristic depth, rather than staying a graded blend? The identity analog of the
//! paper's ambiguous-input "ignition".
//!
//! Method (no averaged Jacobian needed):
//!  1. Per author, per depth ℓ, build a centroid c_{A,ℓ} from that author's TRAINING
//!     passages' pooled activations. c_{A,ℓ} − c_{B,ℓ} is a difference-of-means A↔B axis
//!     in each layer's own space.
//!  2. For a pair (A,B) take two HELD-OUT (mystery) passages and blend them at the input
//!     embedding: h₀(α) = (1−α)·h₀ᴮ + α·h₀ᴬ; sweep α∈[0,1]; forward from layer 0.
//!  3. Read bipolar commitment s_ℓ(α) = û_ℓ·(p̄_ℓ(α) − m_ℓ)/‖½(c_A−c_B)‖  (≈ +1 at A,
//!     −1 at B) at every depth.
//! A graded layer ramps ~linearly in α; an "ignited" layer snaps. Per depth we record the
//! endpoint SEPARATION (does the axis tell A from B at all?) and, where it separates, an
//! IGNITION INDEX (1 − 2·ambiguous-fraction; ~0 graded, →1 all-or-none). Control: a
//! shuffled-label null (split the pooled A+B passages into two fake groups) — expected ~0.
//!
//! Usage: JLENS_DEVICE=cpu cargo run -p jlens --bin ignition --release [minilm|coders]

use anyhow::Result;
use jlens::{dot, normalize, Harness};
use serde::Serialize;
use shared::Meta;

const CAP: usize = 128; // token context (fast + consistent across endpoints/centroids)
const K_CENT: usize = 12; // training passages per author for a centroid
const N_PAIRS: usize = 15; // author pairs
const N_ALPHA: usize = 11; // α grid 0.0 … 1.0
const SEP_MIN: f32 = 0.5; // endpoint separation required to call a depth "valid"

#[derive(Serialize)]
struct Curve {
    a: String,
    b: String,
    commitment: Vec<Vec<f32>>, // [depth][alpha]
}

#[derive(Serialize)]
struct IgnitionBundle {
    model: String,
    subject: String,
    depths: Vec<usize>,
    alphas: Vec<f32>,
    n_pairs: usize,
    separation_mean: Vec<f32>,
    separation_std: Vec<f32>,
    ignition_index_mean: Vec<f32>,
    ignition_n_valid: Vec<usize>,
    separation_null_shuffled_mean: Vec<f32>,
    separation_null_random_mean: Vec<f32>,
    ignition_depth: usize,
    example: Curve,
}

/// mean over token positions at each depth → [depth][dim].
fn pooled_per_depth(hs: &[candle_core::Tensor]) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(hs.len());
    for h in hs {
        out.push(h.mean(1)?.squeeze(0)?.to_vec1::<f32>()?);
    }
    Ok(out)
}

/// Per-depth centroid over `idxs` training passages (context-capped forwards).
fn centroids(h: &Harness, meta: &Meta, idxs: &[usize], n_depth: usize, dim: usize) -> Result<Vec<Vec<f32>>> {
    let mut acc = vec![vec![0f64; dim]; n_depth];
    for &pi in idxs {
        let fwd = h.forward_capped(&meta.passages[pi].text, CAP)?;
        for (d, a) in acc.iter_mut().enumerate() {
            let p = fwd.hidden[d].mean(1)?.squeeze(0)?.to_vec1::<f32>()?;
            for (k, ak) in a.iter_mut().enumerate() {
                *ak += p[k] as f64;
            }
        }
    }
    let n = idxs.len().max(1) as f64;
    Ok(acc.iter().map(|v| v.iter().map(|&x| (x / n) as f32).collect()).collect())
}

/// Commitment: signed projection of a pooled activation onto the UNIT A↔B axis,
/// centred at the class midpoint (raw units — positive toward A, negative toward B).
/// Deliberately NOT normalised by the axis length, so a near-degenerate (null) axis
/// does not inflate the score.
fn commitment(p: &[f32], ca: &[f32], cb: &[f32]) -> f32 {
    let u = normalize(ca.iter().zip(cb).map(|(x, y)| x - y).collect());
    let d: Vec<f32> = p.iter().zip(ca.iter().zip(cb)).map(|(pi, (x, y))| pi - 0.5 * (x + y)).collect();
    dot(&u, &d)
}

/// Deterministic, reproducible unit vector (splitmix64) — the random-direction null.
fn seeded_unit(seed: u64, dim: usize) -> Vec<f32> {
    let mut s = seed;
    let mut v = vec![0f32; dim];
    for x in v.iter_mut() {
        s = s.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        *x = (z as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0;
    }
    normalize(v)
}

/// Ignition index of a commitment curve: 1 − 2·(fraction of α in the ambiguous middle,
/// |normalized|<0.5). ~0 = graded/linear, →1 = all-or-none snap. `None` if the endpoints
/// do not separate (axis can't tell A from B at this depth).
fn ignition_index(c: &[f32]) -> Option<f32> {
    let (lo, hi) = (c[0], *c.last().unwrap());
    let half = 0.5 * (hi - lo);
    if half.abs() < 0.5 * SEP_MIN {
        return None;
    }
    let mid = 0.5 * (hi + lo);
    let amb = c.iter().filter(|&&v| ((v - mid) / half).abs() < 0.5).count() as f32 / c.len() as f32;
    Some(1.0 - 2.0 * amb)
}

fn mean(v: &[f32]) -> f32 {
    if v.is_empty() { 0.0 } else { v.iter().sum::<f32>() / v.len() as f32 }
}
fn std(v: &[f32]) -> f32 {
    if v.len() < 2 { return 0.0; }
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / (v.len() - 1) as f32).sqrt()
}

fn main() -> Result<()> {
    let dataset = std::env::args().nth(1).unwrap_or_else(|| "minilm".into());
    let h = Harness::load_dataset(&dataset)?;
    let (dim, n_depth) = (h.dim, h.num_layers + 1);
    let model_name = if dataset == "coders" { "jinaai/jina-embeddings-v2-base-code" } else { jlens::MODEL_ID };

    let assets = jlens::assets_dir();
    let meta: Meta = serde_json::from_slice(&std::fs::read(assets.join(format!("person2vec-{dataset}.json")))?)?;
    let n_auth = meta.authors.len();

    // split passages into training (non-mystery) and held-out (mystery) per author.
    let (mut train, mut held) = (vec![Vec::new(); n_auth], vec![Vec::new(); n_auth]);
    for (i, p) in meta.passages.iter().enumerate() {
        if p.is_mystery { held[p.author_id].push(i) } else { train[p.author_id].push(i) }
    }

    // deterministic distant pairs: author i with the one "across" the roster.
    let pairs: Vec<(usize, usize)> = (0..N_PAIRS)
        .map(|i| (i % n_auth, (i + n_auth / 2) % n_auth))
        .filter(|&(a, b)| a != b && !held[a].is_empty() && !held[b].is_empty() && train[a].len() >= 2 && train[b].len() >= 2)
        .collect();
    println!("== identity ignition (dataset={dataset}) — {} pairs, {} depths ==", pairs.len(), n_depth);

    let alphas: Vec<f32> = (0..N_ALPHA).map(|i| i as f32 / (N_ALPHA - 1) as f32).collect();

    // per-author real centroids (cached over the authors that appear in pairs).
    let mut cache: std::collections::HashMap<usize, Vec<Vec<f32>>> = std::collections::HashMap::new();
    let mut needed: Vec<usize> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
    needed.sort_unstable();
    needed.dedup();
    for &a in &needed {
        let idxs: Vec<usize> = train[a].iter().take(K_CENT).cloned().collect();
        cache.insert(a, centroids(&h, &meta, &idxs, n_depth, dim)?);
    }

    let mut sep: Vec<Vec<f32>> = vec![Vec::new(); n_depth]; // per depth → per-pair separation
    let mut ign: Vec<Vec<f32>> = vec![Vec::new(); n_depth]; // per depth → per valid-pair ignition
    let mut sep_null: Vec<Vec<f32>> = vec![Vec::new(); n_depth]; // shuffled-label null
    let mut sep_rand: Vec<Vec<f32>> = vec![Vec::new(); n_depth]; // random-direction null
    let mut example: Option<Curve> = None;
    // fixed random axis per depth (reused across pairs) for the random-direction null.
    let rand_axis: Vec<Vec<f32>> = (0..n_depth).map(|d| seeded_unit(1000 + d as u64, dim)).collect();

    for &(a, b) in &pairs {
        let (ca, cb) = (&cache[&a], &cache[&b]);
        // shuffled-label null: pool A+B training passages, split into two fake groups by
        // even/odd (each group ends up ~half-A/half-B, so its axis should carry no signal).
        let pool: Vec<usize> = train[a].iter().take(K_CENT).chain(train[b].iter().take(K_CENT)).cloned().collect();
        let gi0: Vec<usize> = pool.iter().step_by(2).cloned().collect();
        let gi1: Vec<usize> = pool.iter().skip(1).step_by(2).cloned().collect();
        let g0 = centroids(&h, &meta, &gi0, n_depth, dim)?;
        let g1 = centroids(&h, &meta, &gi1, n_depth, dim)?;

        // held-out endpoints, blended at the input embedding.
        let fa = h.forward_capped(&meta.passages[held[a][0]].text, CAP)?;
        let fb = h.forward_capped(&meta.passages[held[b][0]].text, CAP)?;
        let t = fa.t_len.min(fb.t_len);
        let h0a = fa.hidden[0].narrow(1, 0, t)?;
        let h0b = fb.hidden[0].narrow(1, 0, t)?;

        let mut commit = vec![vec![0f32; alphas.len()]; n_depth];
        let mut commit_null = vec![vec![0f32; alphas.len()]; n_depth];
        let (mut pooled0, mut pooled1) = (Vec::new(), Vec::new()); // α=0 and α=1 endpoints
        let last = alphas.len() - 1;
        for (ai, &al) in alphas.iter().enumerate() {
            let blended = h0a.affine(al as f64, 0.0)?.add(&h0b.affine((1.0 - al) as f64, 0.0)?)?;
            let pooled = pooled_per_depth(&h.hidden_states_from(0, &blended)?)?;
            for d in 0..n_depth {
                commit[d][ai] = commitment(&pooled[d], &ca[d], &cb[d]);
                commit_null[d][ai] = commitment(&pooled[d], &g0[d], &g1[d]);
            }
            if ai == 0 {
                pooled0 = pooled;
            } else if ai == last {
                pooled1 = pooled;
            }
        }
        for d in 0..n_depth {
            sep[d].push(commit[d][last] - commit[d][0]);
            sep_null[d].push((commit_null[d][last] - commit_null[d][0]).abs());
            // random-direction null: project the endpoint difference onto a random unit axis.
            let diff: Vec<f32> = pooled1[d].iter().zip(&pooled0[d]).map(|(x, y)| x - y).collect();
            sep_rand[d].push(dot(&rand_axis[d], &diff).abs());
            if let Some(ii) = ignition_index(&commit[d]) {
                ign[d].push(ii);
            }
        }
        if example.is_none() {
            example = Some(Curve { a: meta.authors[a].name.clone(), b: meta.authors[b].name.clone(), commitment: commit });
        }
    }

    let separation_mean: Vec<f32> = sep.iter().map(|v| mean(v)).collect();
    let separation_std: Vec<f32> = sep.iter().map(|v| std(v)).collect();
    let ignition_index_mean: Vec<f32> = ign.iter().map(|v| mean(v)).collect();
    let ignition_n_valid: Vec<usize> = ign.iter().map(|v| v.len()).collect();
    let separation_null_shuffled_mean: Vec<f32> = sep_null.iter().map(|v| mean(v)).collect();
    let separation_null_random_mean: Vec<f32> = sep_rand.iter().map(|v| mean(v)).collect();
    // headline: shallowest depth that both separates (sep>SEP_MIN) and is the sharpest so far.
    let ignition_depth = (0..n_depth)
        .filter(|&d| separation_mean[d] > SEP_MIN)
        .max_by(|&x, &y| ignition_index_mean[x].total_cmp(&ignition_index_mean[y]))
        .unwrap_or(n_depth - 1);

    println!("  separation  (real)       : {:?}", round2(&separation_mean));
    println!("  separation  (shuffled null): {:?}", round2(&separation_null_shuffled_mean));
    println!("  separation  (random null)  : {:?}", round2(&separation_null_random_mean));
    println!("  ignition idx(real)       : {:?}", round2(&ignition_index_mean));
    println!("  valid pairs / depth        : {ignition_n_valid:?}");
    println!("  → ignition depth = {ignition_depth}");

    let bundle = IgnitionBundle {
        model: model_name.to_string(),
        subject: if dataset == "coders" { "coder" } else { "author" }.to_string(),
        depths: (0..n_depth).collect(),
        alphas,
        n_pairs: pairs.len(),
        separation_mean,
        separation_std,
        ignition_index_mean,
        ignition_n_valid,
        separation_null_shuffled_mean,
        separation_null_random_mean,
        ignition_depth,
        example: example.expect("at least one pair"),
    };
    let out = assets.join(format!("person2vec-ignition-{dataset}.json"));
    std::fs::write(&out, serde_json::to_vec(&bundle)?)?;
    println!("  wrote {}", out.display());
    Ok(())
}

fn round2(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| (x * 100.0).round() / 100.0).collect()
}
