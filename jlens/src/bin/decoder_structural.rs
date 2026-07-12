//! Decoder-track M1: averaged **causal** Jacobian structural signatures on GPT-2 — the paper's
//! Figure-28 depth series, on a real open generative decoder (→ `person2vec-decoder-structural-gpt2.json`).
//!
//! Per layer we form the δ-broadcast averaged Jacobian `J_ℓ = ∂h_final,last/∂δ` (perturb every
//! position, read the last position — the next-token driver), averaged over prompts, then compute
//! the same four signatures as the encoders (stable rank, effective dim, verbalizability =
//! kurtosis of the **real-unembedding** vocab lens, autocorrelation) plus a decoder-only fifth:
//! next-token logit-lens accuracy by depth. Reuses `jlens::{stable_rank, effective_dim,
//! excess_kurtosis, readout_autocorrelation}` unchanged — the method is the same, the model generates.
//!
//! Usage: JLENS_DEVICE=metal cargo run -p jlens --bin decoder_structural --release -- [N_PROMPTS]

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;
use serde::Serialize;
use tokenizers::Tokenizer;

use jlens::gpt2::{Config, Gpt2};
use jlens::{effective_dim, excess_kurtosis, matvec, readout_autocorrelation, stable_rank, standardize};
use shared::Meta;

#[derive(Serialize)]
struct DecoderStructural {
    model: String,
    layers: usize,
    d_model: usize,
    vocab: usize,
    prompts: usize,
    stable_rank: Vec<f32>,
    effective_dim: Vec<f32>,
    verbalizability: Vec<f32>,
    autocorrelation: Vec<f32>,
    next_token_acc: Vec<f32>, // logit-lens, length n_layer+1 (per residual depth)
}

fn main() -> Result<()> {
    let n_prompt: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(10);
    let jl: usize = std::env::var("JLENS_JAC_LEN").ok().and_then(|s| s.parse().ok()).unwrap_or(48);
    let device = match std::env::var("JLENS_DEVICE").as_deref() {
        Ok("cpu") => Device::Cpu,
        _ => Device::new_metal(0).unwrap_or(Device::Cpu),
    };
    eprintln!("  device: {device:?}");

    let api = Api::new()?;
    let repo = api.model("openai-community/gpt2".to_string());
    let cfg: Config = serde_json::from_slice(&std::fs::read(repo.get("config.json")?)?)?;
    let (n_layer, dim, vocab) = (cfg.n_layer, cfg.n_embd, cfg.vocab_size);
    let tok = Tokenizer::from_file(repo.get("tokenizer.json")?).map_err(|e| anyhow!("tokenizer: {e}"))?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[repo.get("model.safetensors")?], DType::F32, &device)? };
    let model = Gpt2::load(vb, cfg)?;

    let assets = jlens::assets_dir();
    let meta: Meta = serde_json::from_slice(&std::fs::read(assets.join("person2vec-minilm.json"))?)?;
    let all: Vec<&str> = meta.passages.iter().filter(|p| !p.is_mystery).map(|p| p.text.as_str()).collect();
    let step = (all.len() / n_prompt.max(1)).max(1);
    let prompts: Vec<&str> = all.iter().step_by(step).take(n_prompt).cloned().collect();
    println!("== decoder M1 structural (GPT-2, {n_layer} layers, d={dim}) over {} prompts ==", prompts.len());

    let mut jacc = vec![vec![0f64; dim * dim]; n_layer];
    let mut nt_correct = vec![0usize; n_layer + 1];
    let mut nt_total = 0usize;
    let mut hidden_cache: Vec<Vec<Tensor>> = Vec::new();

    for (pi, text) in prompts.iter().enumerate() {
        let enc = tok.encode(*text, false).map_err(|e| anyhow!("encode: {e}"))?;
        let mut ids: Vec<u32> = enc.get_ids().to_vec();
        ids.truncate(jl.max(2));
        let t = ids.len();
        let input = Tensor::new(ids.as_slice(), &device)?.reshape((1, t))?;
        let hs = model.hidden_states(&input)?;

        // next-token logit-lens accuracy per residual depth (argmax on-device).
        for l in 0..=n_layer {
            let pred = model.logits(&hs[l])?.argmax(D::Minus1)?.i(0)?.to_vec1::<u32>()?;
            for pos in 0..t - 1 {
                if pred[pos] == ids[pos + 1] {
                    nt_correct[l] += 1;
                }
            }
        }
        nt_total += t - 1;

        // δ-broadcast averaged Jacobian per layer.
        for l in 0..n_layer {
            let j = gpt2_jacobian(&model, &hs[l].detach(), l, dim, &device, t)?;
            for (k, jc) in jacc[l].iter_mut().enumerate() {
                *jc += j[k] as f64;
            }
        }
        hidden_cache.push(hs);
        println!("  [{}/{}] done", pi + 1, prompts.len());
    }

    let scale = 1.0 / prompts.len() as f64;
    let javg: Vec<Vec<f32>> = jacc.iter().map(|m| m.iter().map(|&x| (x * scale) as f32).collect()).collect();

    // structural signatures.
    let unembed = model.unembed();
    let (mut stable, mut effd, mut verb, mut autoc) =
        (vec![0f32; n_layer], vec![0f32; n_layer], vec![0f32; n_layer], vec![0f32; n_layer]);
    for l in 0..n_layer {
        stable[l] = stable_rank(&javg[l], dim);
        effd[l] = effective_dim(&javg[l], dim);
        let mut ks = Vec::new();
        let (mut acc, mut cnt) = (0f32, 0usize);
        for hs in &hidden_cache {
            // verbalizability: kurtosis of the real-unembedding vocab lens on the mean activation.
            let ma = hs[l].mean(1)?.squeeze(0)?.to_vec1::<f32>()?;
            let mut jh = matvec(&javg[l], &ma, dim);
            standardize(&mut jh);
            let jt = Tensor::new(jh.as_slice(), &device)?.reshape((dim, 1))?;
            let logits = unembed.matmul(&jt)?.squeeze(1)?.to_vec1::<f32>()?;
            ks.push(excess_kurtosis(&logits));
            // autocorrelation: per-position readouts of the averaged J.
            let t = hs[l].dim(1)?;
            let mut seq = Vec::with_capacity(t);
            for pos in 0..t {
                seq.push(matvec(&javg[l], &hs[l].i((0, pos))?.to_vec1::<f32>()?, dim));
            }
            let a = readout_autocorrelation(&seq);
            if a.is_finite() {
                acc += a;
                cnt += 1;
            }
        }
        verb[l] = ks.iter().sum::<f32>() / ks.len().max(1) as f32;
        autoc[l] = if cnt > 0 { acc / cnt as f32 } else { 0.0 };
    }
    let next_token_acc: Vec<f32> = nt_correct.iter().map(|&c| c as f32 / nt_total.max(1) as f32).collect();

    println!("  stable_rank    : {:?}", r2(&stable));
    println!("  effective_dim  : {:?}", r2(&effd));
    println!("  verbalizability: {:?}", r2(&verb));
    println!("  autocorrelation: {:?}", r2(&autoc));
    println!("  next_token_acc : {:?}", r2(&next_token_acc));

    let out = DecoderStructural {
        model: "openai-community/gpt2".into(),
        layers: n_layer,
        d_model: dim,
        vocab,
        prompts: prompts.len(),
        stable_rank: stable,
        effective_dim: effd,
        verbalizability: verb,
        autocorrelation: autoc,
        next_token_acc,
    };
    let path = assets.join("person2vec-decoder-structural-gpt2.json");
    std::fs::write(&path, serde_json::to_vec(&out)?)?;
    println!("  wrote {}", path.display());
    Ok(())
}

/// δ-broadcast averaged Jacobian at block `l`, read at the last position (next-token driver).
fn gpt2_jacobian(model: &Gpt2, h_op: &Tensor, l: usize, dim: usize, device: &Device, t: usize) -> Result<Vec<f32>> {
    let eps = 0.05f32;
    let inv = 1.0 / (2.0 * eps);
    let chunk = 128usize;
    let mut j = vec![0f32; dim * dim];
    for start in (0..dim).step_by(chunk) {
        let end = (start + chunk).min(dim);
        let b = end - start;
        let rows = 2 * b;
        let mut pert = vec![0f32; rows * dim];
        for c in 0..b {
            pert[(2 * c) * dim + (start + c)] = eps;
            pert[(2 * c + 1) * dim + (start + c)] = -eps;
        }
        let pert = Tensor::new(pert.as_slice(), device)?.reshape((rows, 1, dim))?;
        let batch = h_op.broadcast_as((rows, t, dim))?.broadcast_add(&pert)?;
        let last = model.forward_from(l, &batch)?.i((.., t - 1, ..))?.to_vec2::<f32>()?;
        for c in 0..b {
            let (pp, pm) = (&last[2 * c], &last[2 * c + 1]);
            for i in 0..dim {
                j[i * dim + (start + c)] = (pp[i] - pm[i]) * inv;
            }
        }
    }
    Ok(j)
}

fn r2(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| (x * 100.0).round() / 100.0).collect()
}
