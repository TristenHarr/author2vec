//! Decoder-track M2 (scoped, honest): **directed modulation** on GPT-2 →
//! `person2vec-decoder-steer-gpt2.json`.
//!
//! Hypothesis: if J-space directions are causal on a decoder, injecting a *concept-context*
//! direction into the residual stream should raise concept-related tokens in the output — more
//! than a matched-norm random direction. We build the direction **leakage-free**: the difference
//! of mid-layer mean residuals between concept-primed prompts and neutral prompts (a
//! difference-of-means axis, *not* the target's unembedding). We inject $\pm\alpha\hat\delta$ into
//! HELD-OUT neutral prompts and measure the mean next-token log-prob of held-out concept-target
//! tokens, vs baseline and vs a random-direction control. This tests the mechanistic
//! *prerequisite* for the paper's behavioral claims; full multi-hop "can't→can reason" needs a
//! capable model, which we do not claim here.
//!
//! Usage: JLENS_DEVICE=metal cargo run -p jlens --bin decoder_steer --release

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;
use serde::Serialize;
use tokenizers::Tokenizer;

use jlens::gpt2::{Config, Gpt2};
use jlens::normalize;

struct Concept {
    name: &'static str,
    prime: &'static [&'static str], // concept-priming contexts (build the direction)
    targets: &'static [&'static str], // held-out concept words to elicit (measured)
}

#[derive(Serialize)]
struct ConceptResult {
    name: String,
    base: f32,
    steer_pos: f32,
    steer_neg: f32,
    random: f32,
}

#[derive(Serialize)]
struct SteerBundle {
    model: String,
    layer: usize,
    alpha: f32,
    n_neutral: usize,
    concepts: Vec<ConceptResult>,
    // aggregate deltas (mean over concepts): the headline numbers
    mean_delta_real: f32,
    mean_delta_random: f32,
}

fn main() -> Result<()> {
    let alpha: f32 = std::env::var("STEER_ALPHA").ok().and_then(|s| s.parse().ok()).unwrap_or(8.0);
    let device = match std::env::var("JLENS_DEVICE").as_deref() {
        Ok("cpu") => Device::Cpu,
        _ => Device::new_metal(0).unwrap_or(Device::Cpu),
    };
    eprintln!("  device: {device:?}");
    let api = Api::new()?;
    let repo = api.model("openai-community/gpt2".to_string());
    let cfg: Config = serde_json::from_slice(&std::fs::read(repo.get("config.json")?)?)?;
    let (n_layer, dim) = (cfg.n_layer, cfg.n_embd);
    let tok = Tokenizer::from_file(repo.get("tokenizer.json")?).map_err(|e| anyhow!("tok: {e}"))?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[repo.get("model.safetensors")?], DType::F32, &device)? };
    let model = Gpt2::load(vb, cfg)?;
    let layer = n_layer / 2;

    let concepts = [
        Concept {
            name: "money/finance",
            prime: &["The bank raised interest rates and the stock market", "He counted the dollars and coins in his wallet", "The company reported record profits and revenue this quarter", "She paid the invoice and balanced the budget"],
            targets: &[" money", " cash", " price", " bank", " profit", " dollars"],
        },
        Concept {
            name: "music",
            prime: &["The orchestra tuned their instruments before the concert", "She played a beautiful melody on the piano", "The band recorded a new song in the studio", "He listened to the rhythm and the guitar solo"],
            targets: &[" music", " song", " guitar", " melody", " concert", " sound"],
        },
        Concept {
            name: "war/military",
            prime: &["The soldiers marched into battle at dawn", "The general ordered the army to advance", "Tanks and artillery bombarded the enemy position", "The war left the city in ruins"],
            targets: &[" war", " army", " battle", " soldiers", " enemy", " weapon"],
        },
    ];
    // held-out neutral prompts (never mention any concept).
    let neutral = [
        "The next thing that happened was",
        "I want to tell you about the",
        "Everyone in the room looked at the",
        "After a long pause, she said the",
        "It was clear to me that the",
        "The most surprising part was the",
    ];

    println!("== directed modulation on GPT-2 (inject ±{alpha}·δ at layer {layer}) ==");
    // neutral direction anchor: mean mid-layer residual over the neutral prompts.
    let neutral_mean = mean_resid(&model, &tok, &neutral, layer, dim, &device)?;
    let rand_dir = seeded_unit(4242, dim);

    let mut results = Vec::new();
    let (mut sum_real, mut sum_rand) = (0f32, 0f32);
    for c in &concepts {
        let prime_mean = mean_resid(&model, &tok, c.prime, layer, dim, &device)?;
        let delta = normalize(prime_mean.iter().zip(&neutral_mean).map(|(a, b)| a - b).collect());
        let tgt_ids = target_ids(&tok, c.targets);
        // measure mean target log-prob at the last position over held-out neutral prompts.
        let base = mean_target_logprob(&model, &tok, &neutral, layer, None, &tgt_ids, dim, &device)?;
        let pos = mean_target_logprob(&model, &tok, &neutral, layer, Some((&delta, alpha)), &tgt_ids, dim, &device)?;
        let neg = mean_target_logprob(&model, &tok, &neutral, layer, Some((&delta, -alpha)), &tgt_ids, dim, &device)?;
        let rnd = mean_target_logprob(&model, &tok, &neutral, layer, Some((&rand_dir, alpha)), &tgt_ids, dim, &device)?;
        println!("  {:<14} base {base:+.2}  +δ {pos:+.2} (Δ{:+.2})  −δ {neg:+.2}  rand {rnd:+.2} (Δ{:+.2})",
                 c.name, pos - base, rnd - base);
        sum_real += pos - base;
        sum_rand += rnd - base;
        results.push(ConceptResult { name: c.name.into(), base, steer_pos: pos, steer_neg: neg, random: rnd });
    }
    let (mean_delta_real, mean_delta_random) = (sum_real / concepts.len() as f32, sum_rand / concepts.len() as f32);
    println!("\n  mean Δlog-prob(target):  real steer {mean_delta_real:+.3}   |   random control {mean_delta_random:+.3}");
    println!("  {}", if mean_delta_real > mean_delta_random + 0.1 {
        "→ concept-context injection raises concept tokens above the random control: directed modulation is causal."
    } else {
        "→ real steer does NOT beat the random control here: report as a null/weak result (honest)."
    });

    let bundle = SteerBundle {
        model: "openai-community/gpt2".into(),
        layer,
        alpha,
        n_neutral: neutral.len(),
        concepts: results,
        mean_delta_real,
        mean_delta_random,
    };
    let out = jlens::assets_dir().join("person2vec-decoder-steer-gpt2.json");
    std::fs::write(&out, serde_json::to_vec(&bundle)?)?;
    println!("  wrote {}", out.display());
    Ok(())
}

/// Mean (over prompts) mid-layer residual, averaged over token positions.
fn mean_resid(model: &Gpt2, tok: &Tokenizer, prompts: &[&str], layer: usize, dim: usize, device: &Device) -> Result<Vec<f32>> {
    let mut acc = vec![0f64; dim];
    for p in prompts {
        let ids = encode(tok, p, device)?;
        let hs = model.hidden_states(&ids)?;
        let m = hs[layer].mean(1)?.squeeze(0)?.to_vec1::<f32>()?;
        for (a, &v) in acc.iter_mut().zip(&m) {
            *a += v as f64;
        }
    }
    let n = prompts.len() as f64;
    Ok(acc.iter().map(|&x| (x / n) as f32).collect())
}

/// Mean over prompts of the summed log-prob of `tgt_ids` as the next token, optionally
/// injecting `alpha·dir` (broadcast to every position) at `layer` before the tail.
fn mean_target_logprob(
    model: &Gpt2,
    tok: &Tokenizer,
    prompts: &[&str],
    layer: usize,
    steer: Option<(&[f32], f32)>,
    tgt_ids: &[u32],
    dim: usize,
    device: &Device,
) -> Result<f32> {
    let mut total = 0f32;
    for p in prompts {
        let ids = encode(tok, p, device)?;
        let t = ids.dim(1)?;
        let hs = model.hidden_states(&ids)?;
        let h = hs[layer].detach();
        let h = match steer {
            Some((dir, a)) => {
                let d: Vec<f32> = dir.iter().map(|x| x * a).collect();
                let pert = Tensor::new(d.as_slice(), device)?.reshape((1, 1, dim))?;
                h.broadcast_add(&pert)?
            }
            None => h,
        };
        let final_h = model.forward_from(layer, &h)?;
        let logits = model.logits(&final_h)?.i((0, t - 1))?.to_vec1::<f32>()?;
        let lse = logsumexp(&logits);
        // mean log-prob across the target tokens (a small held-out concept word set).
        let mut s = 0f32;
        for &ti in tgt_ids {
            s += logits[ti as usize] - lse;
        }
        total += s / tgt_ids.len() as f32;
    }
    Ok(total / prompts.len() as f32)
}

fn target_ids(tok: &Tokenizer, targets: &[&str]) -> Vec<u32> {
    targets
        .iter()
        .filter_map(|w| {
            let enc = tok.encode(*w, false).ok()?;
            enc.get_ids().first().copied() // first BPE token of the (space-prefixed) word
        })
        .collect()
}

fn encode(tok: &Tokenizer, text: &str, device: &Device) -> Result<Tensor> {
    let enc = tok.encode(text, false).map_err(|e| anyhow!("encode: {e}"))?;
    let ids: Vec<u32> = enc.get_ids().to_vec();
    let t = ids.len().max(1);
    Ok(Tensor::new(ids.as_slice(), device)?.reshape((1, t))?)
}

fn logsumexp(v: &[f32]) -> f32 {
    let m = v.iter().cloned().fold(f32::MIN, f32::max);
    m + v.iter().map(|x| (x - m).exp()).sum::<f32>().ln()
}

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
