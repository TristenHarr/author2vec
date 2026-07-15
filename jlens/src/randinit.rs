//! Random-initialization checkpoint control — the paper's §7 "untrained-model control".
//!
//! Produces a randomly-initialized `.safetensors` with the *same* tensor names and shapes as a
//! real checkpoint, so the identical model-loading, Jacobian, and structural-metric code (§4.7,
//! §5.3, §5.4) runs on an untrained network. Whatever depth structure survives on random weights
//! is architectural; whatever vanishes is training-induced.
//!
//! Initialization follows HuggingFace's `_init_weights` so the file *is* an untrained
//! `from_config` checkpoint:
//!   - linear / conv1d / embedding weights ~ Normal(0, `initializer_range`)  (0.02 for BERT & GPT-2)
//!   - LayerNorm weight = 1, LayerNorm bias = 0
//!   - all other biases = 0
//!   - GPT-2 refinement: residual-projection `c_proj.weight` std scaled by `1/sqrt(2·n_layer)`.
//! Non-float buffers (position_ids, token_type_ids, unused attention masks) are copied verbatim.
//!
//! Determinism: fully reproducible from `seed`. Each tensor draws from its own splitmix64 stream
//! keyed by a stable hash of the tensor name, so the result is independent of HashMap iteration
//! order (matches the paper's bit-for-bit reproducibility discipline, `PAPER.md:529`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

/// How to initialize a random checkpoint for one architecture.
#[derive(Clone, Copy)]
pub struct InitSpec {
    /// `initializer_range` from the model config (0.02 for BERT and GPT-2).
    pub init_range: f64,
    /// If `Some(n_layer)`, apply the GPT-2 residual-projection init: `c_proj.weight` std is
    /// scaled by `1/sqrt(2·n_layer)`. `None` for the encoders (no such rescaling).
    pub residual_nlayer: Option<usize>,
}

/// The control's seed, read from `JLENS_RANDOM_INIT` (unset ⇒ `None` ⇒ use the trained weights).
pub fn random_init_seed() -> Option<u64> {
    std::env::var("JLENS_RANDOM_INIT").ok().and_then(|s| s.trim().parse::<u64>().ok())
}

/// splitmix64 stream (the repo's RNG convention, cf. `seeded_unit` in `bin/steer_bundle.rs`),
/// keyed per tensor so draws are order-independent. Yields standard normals via Box–Muller.
struct SplitMix64 {
    s: u64,
    spare: Option<f64>,
}

impl SplitMix64 {
    fn new(seed: u64, name: &str) -> Self {
        // FNV-1a over the tensor name, mixed into the global seed.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in name.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        Self { s: seed ^ h.wrapping_mul(0x9E37_79B9_7F4A_7C15), spare: None }
    }

    fn next_u64(&mut self) -> u64 {
        self.s = self.s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1) — 53-bit mantissa, offset so it is strictly positive for `ln`.
    fn next_unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
    }

    /// Standard normal via Box–Muller (caches the paired draw).
    fn next_normal(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        let u1 = self.next_unit();
        let u2 = self.next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        self.spare = Some(r * theta.sin());
        r * theta.cos()
    }
}

/// A `<name>.weight` belonging to a LayerNorm (γ, initialized to ones), across BERT/JinaBERT
/// (`LayerNorm`) and GPT-2 (`ln_1`, `ln_2`, `ln_f`) naming.
fn is_layernorm_weight(name: &str) -> bool {
    if !name.ends_with(".weight") {
        return false;
    }
    let l = name.to_lowercase();
    l.contains("layernorm") || l.contains("layer_norm") || l.contains("ln_1") || l.contains("ln_2") || l.contains("ln_f")
}

/// Build (once, then cache) a randomly-initialized sibling of `real_weights`, returning its path.
/// Re-running with the same seed returns the cached file, so runs are reproducible and cheap.
pub fn random_init_checkpoint(real_weights: &Path, spec: InitSpec, seed: u64) -> Result<PathBuf> {
    let out = real_weights
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!("model.random-seed{seed}.safetensors"));
    if out.exists() {
        return Ok(out);
    }

    let src = candle_core::safetensors::load(real_weights, &Device::Cpu)
        .with_context(|| format!("load real checkpoint {}", real_weights.display()))?;

    let residual_scale = spec
        .residual_nlayer
        .map(|nl| 1.0 / ((2 * nl.max(1)) as f64).sqrt())
        .unwrap_or(1.0);

    let mut dst: HashMap<String, Tensor> = HashMap::with_capacity(src.len());
    for (name, t) in &src {
        let shape = t.dims().to_vec();
        let numel: usize = shape.iter().product();
        let is_float = matches!(t.dtype(), DType::F16 | DType::BF16 | DType::F32 | DType::F64);

        let new = if !is_float {
            // Integer buffers (position_ids, token_type_ids, causal-mask buffers) — copy verbatim.
            t.clone()
        } else if name.ends_with(".bias") {
            Tensor::zeros(shape.clone(), DType::F32, &Device::Cpu)?
        } else if is_layernorm_weight(name) {
            Tensor::ones(shape.clone(), DType::F32, &Device::Cpu)?
        } else {
            let mut std = spec.init_range;
            if name.ends_with("c_proj.weight") {
                std *= residual_scale;
            }
            let mut rng = SplitMix64::new(seed, name);
            let data: Vec<f32> = (0..numel).map(|_| (rng.next_normal() * std) as f32).collect();
            Tensor::from_vec(data, shape.clone(), &Device::Cpu)?
        };
        dst.insert(name.clone(), new);
    }

    candle_core::safetensors::save(&dst, &out)
        .with_context(|| format!("save random checkpoint {}", out.display()))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_is_deterministic_per_name() {
        let draws = |seed, name| {
            let mut r = SplitMix64::new(seed, name);
            (0..8).map(|_| r.next_normal()).collect::<Vec<_>>()
        };
        // Same (seed, name) ⇒ identical stream; different name ⇒ different stream.
        assert_eq!(draws(7, "encoder.layer.0.attention.self.query.weight"), draws(7, "encoder.layer.0.attention.self.query.weight"));
        assert_ne!(draws(7, "a.weight"), draws(7, "b.weight"));
        assert_ne!(draws(7, "a.weight"), draws(8, "a.weight"));
    }

    #[test]
    fn normals_are_roughly_standard() {
        let mut r = SplitMix64::new(1, "wte.weight");
        let n = 100_000;
        let xs: Vec<f64> = (0..n).map(|_| r.next_normal()).collect();
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.02, "mean {mean} not ~0");
        assert!((var - 1.0).abs() < 0.05, "var {var} not ~1");
    }

    #[test]
    fn layernorm_weight_detection() {
        assert!(is_layernorm_weight("embeddings.LayerNorm.weight"));
        assert!(is_layernorm_weight("h.0.ln_1.weight"));
        assert!(is_layernorm_weight("encoder.layer.3.output.LayerNorm.weight"));
        assert!(!is_layernorm_weight("embeddings.LayerNorm.bias"));
        assert!(!is_layernorm_weight("h.0.attn.c_attn.weight"));
    }
}
