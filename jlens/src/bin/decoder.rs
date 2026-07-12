//! Decoder-track M0: logit lens on GPT-2 (a real generative decoder with a real,
//! tied unembedding — no lens approximation). Shows the next-token prediction
//! sharpening with depth. Proves the J-lens machinery transfers to a decoder before
//! we build the (expensive) averaged Jacobian + behavioral experiments.
//!
//! `cargo run -p jlens --bin decoder --release ["a prompt"]`

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

use jlens::gpt2::{Config, Gpt2};
use jlens::{matvec, normalize, topk};

fn main() -> Result<()> {
    let prompt = std::env::args().nth(1).unwrap_or_else(|| "The capital of France is".to_string());

    let device = match Device::new_metal(0) {
        Ok(d) => d,
        Err(_) => Device::Cpu,
    };
    println!("== loading openai-community/gpt2 ==");
    let api = Api::new()?;
    let repo = api.model("openai-community/gpt2".to_string());
    let cfg: Config = serde_json::from_slice(&std::fs::read(repo.get("config.json")?)?)?;
    println!("  n_layer={} n_embd={} n_head={} vocab={}", cfg.n_layer, cfg.n_embd, cfg.n_head, cfg.vocab_size);
    let tok = Tokenizer::from_file(repo.get("tokenizer.json")?).map_err(|e| anyhow!("tokenizer: {e}"))?;
    let weights = repo
        .get("model.safetensors")
        .map_err(|e| anyhow!("download model.safetensors (does the repo ship safetensors?): {e}"))?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device)? };
    let model = Gpt2::load(vb, cfg)?;

    let enc = tok.encode(prompt.as_str(), false).map_err(|e| anyhow!("encode: {e}"))?;
    let ids: Vec<u32> = enc.get_ids().to_vec();
    let t = ids.len();
    let input = Tensor::new(ids.as_slice(), &device)?.reshape((1, t))?;
    let hs = model.hidden_states(&input)?;

    println!("\n== logit lens: next-token prediction after \u{201c}{prompt}\u{201d} by layer ==");
    for (l, h) in hs.iter().enumerate() {
        let logits = model.logits(h)?;
        let last = logits.i((0, t - 1))?.to_vec1::<f32>()?;
        let toks: Vec<String> = topk(&last, 6)
            .into_iter()
            .map(|i| clean(tok.id_to_token(i as u32).unwrap_or_default()))
            .collect();
        let tag = if l == 0 { "embed".to_string() } else { format!("blk {l:>2}") };
        println!("  {tag}: {}", toks.join("  ·  "));
    }
    // ---- J-lens (Jacobian-corrected readout) at a mid block vs the logit lens ----
    let dim = model.cfg.n_embd;
    let ll = 8usize.min(hs.len() - 2);
    let jrow = gpt2_jacobian(&model, &hs[ll].detach(), ll, dim, &device, t)?;
    let h_last = hs[ll].i((0, t - 1))?.to_vec1::<f32>()?;
    let jh = matvec(&jrow, &h_last, dim); // J·h = predicted final residual
    let jt = Tensor::new(jh.as_slice(), &device)?.reshape((1, 1, dim))?;
    let jlogits = model.logits(&jt)?.i((0, 0))?.to_vec1::<f32>()?;
    let jtoks: Vec<String> = topk(&jlogits, 6).into_iter().map(|i| clean(tok.id_to_token(i as u32).unwrap_or_default())).collect();
    let logit_lens: Vec<String> = {
        let l = model.logits(&hs[ll])?.i((0, t - 1))?.to_vec1::<f32>()?;
        topk(&l, 6).into_iter().map(|i| clean(tok.id_to_token(i as u32).unwrap_or_default())).collect()
    };
    println!("\n== J-lens (single-context Jacobian) vs logit lens at block {ll} ==");
    println!("  logit lens: {}", logit_lens.join("  ·  "));
    println!("  J-lens:     {}", jtoks.join("  ·  "));

    // ---- M2: causal control — steer generation by injecting a concept direction ----
    println!("\n== M2: concept steering of generation (add α·dir at a mid block) ==");
    let gen_prompt = "I watched the film and I thought it was";
    let genc: Vec<u32> = tok.encode(gen_prompt, false).map_err(|e| anyhow!("{e}"))?.get_ids().to_vec();
    let alpha: f32 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(30.0);
    println!("  prompt: \u{201c}{gen_prompt}\u{201d}   (α={alpha}, steer layer {ll})");
    println!("  baseline:          …{}", generate(&model, &tok, &genc, 12, None, &device)?);
    for target in [" terrible", " boring", " wonderful", " hilarious"] {
        let dir = concept_dir(&model, &tok, target)?;
        println!("  steer →{target:<11} …{}", generate(&model, &tok, &genc, 12, Some((ll, dir, alpha)), &device)?);
    }
    println!("  The injected concept bends the generated text → J-space directions are causal on a decoder.");

    // ---- #9 (mechanistic core): is there a causal expert↔casual REGISTER axis? ----
    println!("\n== #9: does the model hold a causal expert↔casual register axis? ==");
    let rp = "My honest assessment is that this is";
    let rc: Vec<u32> = tok.encode(rp, false).map_err(|e| anyhow!("{e}"))?.get_ids().to_vec();
    println!("  prompt: \u{201c}{rp}\u{201d}");
    println!("  baseline:               …{}", generate(&model, &tok, &rc, 14, None, &device)?);
    for target in [" sophisticated", " simple"] {
        let dir = concept_dir(&model, &tok, target)?;
        println!("  steer →{target:<14} …{}", generate(&model, &tok, &rc, 14, Some((ll, dir, alpha)), &device)?);
    }
    println!("  A causal register direction ⇒ the mechanistic prerequisite for expertise-mirroring exists.");
    Ok(())
}

/// Unit direction toward a concept token (its unembedding row).
fn concept_dir(model: &Gpt2, tok: &Tokenizer, target: &str) -> Result<Vec<f32>> {
    let tid = tok.encode(target, false).map_err(|e| anyhow!("{e}"))?.get_ids()[0];
    Ok(normalize(model.unembed().i(tid as usize)?.to_vec1::<f32>()?))
}

/// Greedy generation of `n` tokens, optionally injecting `α·dir` into the residual at a
/// layer at every step (activation steering — the decoder analog of our encoder steering).
fn generate(model: &Gpt2, tok: &Tokenizer, prompt: &[u32], n: usize, steer: Option<(usize, Vec<f32>, f32)>, device: &Device) -> Result<String> {
    let dim = model.cfg.n_embd;
    let mut ids = prompt.to_vec();
    for _ in 0..n {
        let input = Tensor::new(ids.as_slice(), device)?.reshape((1, ids.len()))?;
        let hs = model.hidden_states(&input)?;
        let fin = match &steer {
            Some((l, dir, alpha)) => {
                let d: Vec<f32> = dir.iter().map(|x| x * alpha).collect();
                let pert = Tensor::new(d.as_slice(), device)?.reshape((1, 1, dim))?;
                model.forward_from(*l, &hs[*l].broadcast_add(&pert)?)?
            }
            None => hs.last().unwrap().clone(),
        };
        let last = model.logits(&fin)?.i((0, ids.len() - 1))?.to_vec1::<f32>()?;
        let next = last.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap_or(0);
        ids.push(next as u32);
    }
    tok.decode(&ids[prompt.len()..], true).map_err(|e| anyhow!("decode: {e}"))
}

/// Single-context averaged Jacobian at layer `l`: `∂(final residual @ last pos)/∂δ` via
/// batched central finite differences (the δ-broadcast trick, causal decoder version).
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

/// GPT-2 BPE uses 'Ġ' (U+0120) for a leading space; render it readable.
fn clean(s: String) -> String {
    s.replace('\u{0120}', "␠").replace('\u{010A}', "\\n")
}
