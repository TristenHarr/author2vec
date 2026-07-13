//! Persist the encoder identity-steering result + control → `person2vec-steer-<ds>.json` (T1.3).
//!
//! For each style axis $A$ we form the residual steering direction
//! $\hat\delta = \widehat{J_{\text{emb}}^{\top}A}$ at the mid layer, inject $\alpha\hat\delta$
//! (broadcast to every position), re-embed, and record the output's loading on $A$ across an
//! $\alpha$ sweep. Control: a **matched-norm random direction** (same injection, meaningless
//! direction) — its loading should stay flat. Kept separate from `bin/steer` so the shipped
//! identity bundle is never overwritten, and only the mid layer's Jacobian is averaged.
//!
//! Usage: JLENS_DEVICE=metal cargo run -p jlens --bin steer_bundle --release -- <minilm|coders> [N_JAC]

use anyhow::Result;
use candle_core::Tensor;
use jlens::{dataset_axes, dot, even_nonmystery, normalize, to_embedding_jacobian, Forward, Harness};
use serde::Serialize;
use shared::{vectors_from_bytes, Meta};

#[derive(Serialize)]
struct SteerAxis {
    name: String,
    loading: Vec<f32>,
    loading_random: Vec<f32>,
}

#[derive(Serialize)]
struct SteerBundle {
    model: String,
    subject: String,
    steer_layer: usize,
    alphas: Vec<f32>,
    n_probe: usize,
    axes: Vec<SteerAxis>,
}

fn main() -> Result<()> {
    let dataset = std::env::args().nth(1).unwrap_or_else(|| "minilm".into());
    let n_jac: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(48);
    let jl: usize = std::env::var("JLENS_JAC_LEN").ok().and_then(|s| s.parse().ok()).unwrap_or(64);

    let h = Harness::load_dataset(&dataset)?;
    let dim = h.dim;
    let steer_layer = h.num_layers / 2;
    let assets = jlens::assets_dir();
    let meta: Meta = serde_json::from_slice(&std::fs::read(assets.join(format!("person2vec-{dataset}.json")))?)?;
    let ref_vecs = vectors_from_bytes(&std::fs::read(assets.join(format!("person2vec-{dataset}.bin")))?);
    let axes = dataset_axes(&meta.passages, &meta.authors, &ref_vecs, dim);

    // averaged embedding-Jacobian at the steer layer only.
    let sample = even_nonmystery(&meta, n_jac);
    println!("== steering bundle (dataset={dataset}) — J_emb@L{steer_layer} over {} passages ==", sample.len());
    let mut jemb = vec![0f64; dim * dim];
    for &pi in &sample {
        let fwd = h.forward_capped(&meta.passages[pi].text, jl)?;
        let je = to_embedding_jacobian(&h.layer_jacobian(&fwd, steer_layer)?, &fwd.embedding, fwd.pooled_norm, dim);
        for (k, j) in jemb.iter_mut().enumerate() {
            *j += je[k] as f64;
        }
    }
    let scale = 1.0 / sample.len() as f64;
    let jemb: Vec<f32> = jemb.iter().map(|&x| (x * scale) as f32).collect();

    let alphas = vec![-6.0f32, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0];
    let probe: Vec<usize> = sample.iter().step_by((sample.len() / 12).max(1)).take(12).cloned().collect();
    let mut out_axes = Vec::new();
    for (ai, axis) in axes.iter().enumerate() {
        let delta = normalize(jt_mul(&jemb, &axis.vec, dim));
        let rand = seeded_unit(7000 + ai as u64, dim);
        let (mut load, mut load_r) = (vec![0f32; alphas.len()], vec![0f32; alphas.len()]);
        for &pi in &probe {
            let fwd = h.forward_capped(&meta.passages[pi].text, jl)?;
            for (k, &al) in alphas.iter().enumerate() {
                load[k] += dot(&normalize(steer_embed(&h, &fwd, steer_layer, &delta, al, dim)?), &axis.vec);
                load_r[k] += dot(&normalize(steer_embed(&h, &fwd, steer_layer, &rand, al, dim)?), &axis.vec);
            }
        }
        let n = probe.len() as f32;
        load.iter_mut().for_each(|x| *x /= n);
        load_r.iter_mut().for_each(|x| *x /= n);
        println!("  {:<30} α={:?}", axis.name, alphas);
        println!("    real   loading: {:?}", r3(&load));
        println!("    random loading: {:?}", r3(&load_r));
        out_axes.push(SteerAxis { name: axis.name.clone(), loading: load, loading_random: load_r });
    }

    let model = if dataset == "coders" { "jinaai/jina-embeddings-v2-base-code" } else { "sentence-transformers/all-MiniLM-L6-v2" };
    let subject = if dataset == "coders" { "coder" } else { "author" };
    let sb = SteerBundle {
        model: model.into(),
        subject: subject.into(),
        steer_layer,
        alphas,
        n_probe: probe.len(),
        axes: out_axes,
    };
    let out = assets.join(format!("person2vec-steer-{dataset}.json"));
    std::fs::write(&out, serde_json::to_vec(&sb)?)?;
    println!("  wrote {}", out.display());
    Ok(())
}

/// Re-embed after adding `alpha·delta` (broadcast to every position) at `layer`.
fn steer_embed(h: &Harness, fwd: &Forward, layer: usize, delta: &[f32], alpha: f32, dim: usize) -> Result<Vec<f32>> {
    let d: Vec<f32> = delta.iter().map(|x| x * alpha).collect();
    let h_op = fwd.hidden[layer].detach();
    let pert = Tensor::new(d.as_slice(), h_op.device())?.reshape((1, 1, dim))?;
    let tail = h.forward_from(layer, &h_op.broadcast_add(&pert)?)?;
    Ok(tail.mean(1)?.squeeze(0)?.to_vec1::<f32>()?)
}

/// `Jᵀ · v` for a row-major `dim×dim` J.
fn jt_mul(j: &[f32], v: &[f32], dim: usize) -> Vec<f32> {
    let mut out = vec![0f32; dim];
    for i in 0..dim {
        let vi = v[i];
        for (k, o) in out.iter_mut().enumerate() {
            *o += j[i * dim + k] * vi;
        }
    }
    out
}

/// Deterministic unit vector (splitmix64) — the matched-norm random control.
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


fn r3(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| (x * 1000.0).round() / 1000.0).collect()
}
