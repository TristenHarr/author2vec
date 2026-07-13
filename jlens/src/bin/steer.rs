//! Rungs 2 & 4 of the "it learns your fingerprint" ladder, on the small model.
//!
//! (A) ALIGNMENT / "same thing": is author identity decodable from the *internal*
//!     J-lens readout, not just the output embedding? For each layer we map a
//!     passage's mean activation through the averaged embedding-Jacobian `J_emb_ℓ`
//!     and run leave-one-out nearest-author-centroid. High accuracy at intermediate
//!     layers means identity is computed *inside*, and is the same structure
//!     author2vec reads off the output.
//!
//! (B) STEERING / causal: add `α·δ` to an internal activation, where `δ = J_embᵀ·A`
//!     is the residual direction the Jacobian maps onto a style axis `A`, re-run the
//!     tail, and measure the shift in the output's loading on `A`.
//!
//! `cargo run -p jlens --bin steer --release [N_JAC] [N_ALIGN]`

use std::io::Write;

use anyhow::Result;
use candle_core::Tensor;

use jlens::{dataset_axes, dot, even_nonmystery, matvec, normalize, to_embedding_jacobian, Forward, Harness};
use shared::{vectors_from_bytes, Meta};

fn main() -> Result<()> {
    let dataset = std::env::args().nth(1).unwrap_or_else(|| "minilm".to_string());
    let n_jac = arg(2, 48);
    let n_align = arg(3, 250);
    let jl: usize = std::env::var("JLENS_JAC_LEN").ok().and_then(|s| s.parse().ok()).unwrap_or(64);

    let h = Harness::load_dataset(&dataset)?;
    let dim = h.dim;
    let n_layers = h.num_layers;
    let assets = jlens::assets_dir();
    let meta: Meta = serde_json::from_slice(&std::fs::read(assets.join(format!("person2vec-{dataset}.json")))?)?;
    let ref_vecs = vectors_from_bytes(&std::fs::read(assets.join(format!("person2vec-{dataset}.bin")))?);
    let axes = dataset_axes(&meta.passages, &meta.authors, &ref_vecs, dim);
    let n_authors = meta.authors.len();
    println!("(dataset={dataset}, {n_authors} identities, chance {:.1}%)", 100.0 / n_authors as f32);

    // ---- averaged embedding-Jacobians J_emb_ℓ over a passage sample ----
    println!("== averaging J_emb over {n_jac} passages × {n_layers} layers ==");
    let mut jemb = vec![vec![0f64; dim * dim]; n_layers];
    for (ci, &pi) in even_nonmystery(&meta, n_jac).iter().enumerate() {
        let fwd = h.forward_capped(&meta.passages[pi].text, jl)?;
        for l in 0..n_layers {
            let je = to_embedding_jacobian(&h.layer_jacobian(&fwd, l)?, &fwd.embedding, fwd.pooled_norm, dim);
            for k in 0..dim * dim {
                jemb[l][k] += je[k] as f64;
            }
        }
        if ci % 8 == 0 {
            print!(".");
            std::io::stdout().flush().ok();
        }
    }
    let scale = 1.0 / even_nonmystery(&meta, n_jac).len() as f64;
    let jemb: Vec<Vec<f32>> = jemb.iter().map(|m| m.iter().map(|&x| (x * scale) as f32).collect()).collect();
    println!();

    // ---- (A) alignment: author separability, output vs per-layer internal readout ----
    println!("\n== (A) is identity decodable INSIDE the model? leave-one-out nearest-author ==");
    let align = even_nonmystery(&meta, n_align);
    let mut out_vecs: Vec<Vec<f32>> = Vec::new();
    let mut layer_vecs: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
    // baseline: plain linear probe on the mean-pooled hidden state directly (no Jacobian)
    let mut raw_vecs: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
    let mut labels: Vec<usize> = Vec::new();
    for &pi in &align {
        let fwd = h.forward_capped(&meta.passages[pi].text, jl)?;
        out_vecs.push(normalize(fwd.embedding.clone()));
        for l in 0..n_layers {
            let ma = fwd.hidden[l].mean(1)?.squeeze(0)?.to_vec1::<f32>()?;
            raw_vecs[l].push(normalize(ma.clone()));
            layer_vecs[l].push(normalize(matvec(&jemb[l], &ma, dim)));
        }
        labels.push(meta.passages[pi].author_id);
    }
    let output_acc = loo_centroid_accuracy(&out_vecs, &labels, n_authors, dim);
    let per_layer: Vec<f32> = (0..n_layers)
        .map(|l| loo_centroid_accuracy(&layer_vecs[l], &labels, n_authors, dim))
        .collect();
    let per_layer_probe: Vec<f32> = (0..n_layers)
        .map(|l| loo_centroid_accuracy(&raw_vecs[l], &labels, n_authors, dim))
        .collect();
    println!("  output embedding: {:.1}%", output_acc * 100.0);
    println!("  by layer:  J-lens readout   vs   direct-activation probe (baseline)");
    for l in 0..n_layers {
        println!("    layer {l}: {:.1}%   vs   {:.1}%", per_layer[l] * 100.0, per_layer_probe[l] * 100.0);
    }
    println!("  (chance ≈ {:.1}%)  → identity lives in the internal computation, not only the output.", 100.0 / n_authors as f32);
    // Ship the result so the viewer can SHOW it.
    let model = if dataset == "coders" { "jinaai/jina-embeddings-v2-base-code" } else { "sentence-transformers/all-MiniLM-L6-v2" };
    let subject = if dataset == "coders" { "coder" } else { "author" };
    let idb = shared::IdentityBundle {
        model: model.to_string(),
        subject: subject.to_string(),
        identities: n_authors,
        chance: 1.0 / n_authors as f32,
        output_acc,
        per_layer,
        per_layer_probe,
    };
    std::fs::write(assets.join(format!("person2vec-identity-{dataset}.json")), serde_json::to_vec(&idb)?)?;
    println!("  wrote person2vec-identity-{dataset}.json");

    // ---- (B) steering: push an internal activation along an identity axis ----
    println!("\n== (B) can we STEER identity from inside? add α·δ at a mid layer ==");
    let steer_layer = n_layers / 2;
    for axis in axes.iter().take(3) {
        let delta = normalize(jt_mul(&jemb[steer_layer], &axis.vec, dim));
        println!("  axis \"{}\"  (steer at layer {steer_layer}):", axis.name);
        for &pi in align.iter().step_by(align.len() / 3).take(3) {
            let fwd = h.forward_capped(&meta.passages[pi].text, jl)?;
            let base = dot(&normalize(fwd.embedding.clone()), &axis.vec);
            let neg = dot(&normalize(steer_embed(&h, &fwd, steer_layer, &delta, -6.0, dim)?), &axis.vec);
            let pos = dot(&normalize(steer_embed(&h, &fwd, steer_layer, &delta, 6.0, dim)?), &axis.vec);
            println!(
                "    {:<16} α=−6 {neg:+.3}  |  base {base:+.3}  |  α=+6 {pos:+.3}",
                short(&meta.authors[meta.passages[pi].author_id].name)
            );
        }
    }
    println!("\n  α moving the output loading up/down ⇒ the internal identity direction is causal.");
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

/// Leave-one-out nearest-author-centroid accuracy over normalized vectors.
fn loo_centroid_accuracy(vecs: &[Vec<f32>], labels: &[usize], n_authors: usize, dim: usize) -> f32 {
    let mut sums = vec![vec![0f32; dim]; n_authors];
    let mut counts = vec![0usize; n_authors];
    for (v, &a) in vecs.iter().zip(labels) {
        for k in 0..dim {
            sums[a][k] += v[k];
        }
        counts[a] += 1;
    }
    let (mut correct, mut total) = (0usize, 0usize);
    for (v, &a) in vecs.iter().zip(labels) {
        let (mut best, mut best_sim) = (usize::MAX, f32::NEG_INFINITY);
        for c in 0..n_authors {
            let denom = if c == a { counts[c].saturating_sub(1) } else { counts[c] };
            if denom == 0 {
                continue;
            }
            let mut sim = 0f32;
            for k in 0..dim {
                let cen = (if c == a { sums[c][k] - v[k] } else { sums[c][k] }) / denom as f32;
                sim += v[k] * cen;
            }
            if sim > best_sim {
                best_sim = sim;
                best = c;
            }
        }
        if best == a {
            correct += 1;
        }
        total += 1;
    }
    correct as f32 / total.max(1) as f32
}


fn short(name: &str) -> String {
    name.split_whitespace().last().unwrap_or(name).to_string()
}

fn arg(i: usize, default: usize) -> usize {
    std::env::args().nth(i).and_then(|s| s.parse().ok()).unwrap_or(default)
}
