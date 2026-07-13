//! Minimal JinaBERT-v2-qk-post-norm forward in candle — the coders model
//! (`jinaai/jina-embeddings-v2-base-code`, 768-d, 12 layers).
//!
//! Reimplemented faithfully from jina's `modeling_bert.py` (the `jina-bert-v2-qk-post-norm`
//! base): ALiBi positional bias, **QK-LayerNorm** on Q and K, a SelfOutput LayerNorm, two
//! block-level post-norms (`layer_norm_1/2` with a double residual), and a **GeGLU** MLP
//! (`up = states[:inter]`, `gated = states[inter:]`, `up * gelu(gated)`). Validated against
//! the shipped coders embeddings by cosine. Finite-difference Jacobian ⇒ forward-only, so
//! candle_nn's fused LayerNorm is fine here.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{embedding, layer_norm, linear, linear_no_bias, Embedding, LayerNorm, Linear, Module, VarBuilder};
use serde::Deserialize;
use std::cell::RefCell;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub type_vocab_size: usize,
    #[serde(default = "default_eps")]
    pub layer_norm_eps: f64,
}

fn default_eps() -> f64 {
    1e-12
}

impl Config {
    fn head_size(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

struct Embeddings {
    word_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
}
impl Embeddings {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        Ok(Self {
            word_embeddings: embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("word_embeddings"))?,
            token_type_embeddings: embedding(cfg.type_vocab_size, cfg.hidden_size, vb.pp("token_type_embeddings"))?,
            layer_norm: layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("LayerNorm"))?,
        })
    }
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (b, t) = input_ids.dims2()?;
        let we = self.word_embeddings.forward(input_ids)?;
        let tt = Tensor::zeros(t, DType::U32, input_ids.device())?
            .broadcast_left(b)?
            .apply(&self.token_type_embeddings)?;
        self.layer_norm.forward(&(we + tt)?)
    }
}

/// Self-attention with QK-LayerNorm on Q and K, plus an additive ALiBi bias.
struct SelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    ln_q: LayerNorm,
    ln_k: LayerNorm,
    n_heads: usize,
    head_size: usize,
}
impl SelfAttention {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let hs = cfg.head_size();
        let all = cfg.num_attention_heads * hs;
        Ok(Self {
            query: linear(cfg.hidden_size, all, vb.pp("query"))?,
            key: linear(cfg.hidden_size, all, vb.pp("key"))?,
            value: linear(cfg.hidden_size, all, vb.pp("value"))?,
            ln_q: layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layer_norm_q"))?,
            ln_k: layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layer_norm_k"))?,
            n_heads: cfg.num_attention_heads,
            head_size: hs,
        })
    }
    fn shape(&self, xs: &Tensor) -> Result<Tensor> {
        let mut s = xs.dims().to_vec();
        s.pop();
        s.push(self.n_heads);
        s.push(self.head_size);
        xs.reshape(s)?.transpose(1, 2)?.contiguous()
    }
    fn forward(&self, xs: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let q = self.shape(&self.ln_q.forward(&self.query.forward(xs)?)?)?;
        let k = self.shape(&self.ln_k.forward(&self.key.forward(xs)?)?)?;
        let v = self.shape(&self.value.forward(xs)?)?;
        let scores = (q.matmul(&k.t()?)? / (self.head_size as f64).sqrt())?;
        let scores = scores.broadcast_add(bias)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        probs.matmul(&v)?.transpose(1, 2)?.contiguous()?.flatten_from(D::Minus2)
    }
}

struct SelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
}
impl SelfOutput {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        Ok(Self {
            dense: linear(cfg.hidden_size, cfg.hidden_size, vb.pp("dense"))?,
            layer_norm: layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("LayerNorm"))?,
        })
    }
    fn forward(&self, xs: &Tensor, input: &Tensor) -> Result<Tensor> {
        self.layer_norm.forward(&(self.dense.forward(xs)? + input)?)
    }
}

struct GeGluMlp {
    up_gated: Linear, // hidden → 2·inter, no bias
    down: Linear,     // inter → hidden
    inter: usize,
}
impl GeGluMlp {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        Ok(Self {
            up_gated: linear_no_bias(cfg.hidden_size, cfg.intermediate_size * 2, vb.pp("up_gated_layer"))?,
            down: linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_layer"))?,
            inter: cfg.intermediate_size,
        })
    }
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.up_gated.forward(xs)?;
        let up = h.narrow(D::Minus1, 0, self.inter)?;
        let gated = h.narrow(D::Minus1, self.inter, self.inter)?;
        self.down.forward(&(up * gated.gelu_erf()?)?)
    }
}

struct Layer {
    self_attn: SelfAttention,
    self_out: SelfOutput,
    mlp: GeGluMlp,
    ln1: LayerNorm,
    ln2: LayerNorm,
}
impl Layer {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        Ok(Self {
            self_attn: SelfAttention::load(vb.pp("attention").pp("self"), cfg)?,
            self_out: SelfOutput::load(vb.pp("attention").pp("output"), cfg)?,
            mlp: GeGluMlp::load(vb.pp("mlp"), cfg)?,
            ln1: layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layer_norm_1"))?,
            ln2: layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layer_norm_2"))?,
        })
    }
    fn forward(&self, x: &Tensor, bias: &Tensor) -> Result<Tensor> {
        // attention_output = SelfOutput(SelfAttn(x), x)  [post-norm inside]
        let attn = self.self_out.forward(&self.self_attn.forward(x, bias)?, x)?;
        let residual = self.ln1.forward(&(x + attn)?)?; // LN1(x + attention_output)
        let mlp = self.mlp.forward(&residual)?;
        self.ln2.forward(&(residual + mlp)?) // LN2(residual + mlp)
    }
}

pub struct JinaModel {
    embeddings: Embeddings,
    layers: Vec<Layer>,
    n_heads: usize,
    // ALiBi bias is a pure function of seq length, which is constant across a whole Jacobian
    // sweep; cache it so forward_from doesn't rebuild (1,H,T,T) on every finite-difference call.
    bias_cache: RefCell<Option<(usize, Tensor)>>,
}
impl JinaModel {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let embeddings = Embeddings::load(vb.pp("embeddings"), cfg)?;
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| Layer::load(vb.pp("encoder").pp("layer").pp(i.to_string()), cfg))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { embeddings, layers, n_heads: cfg.num_attention_heads, bias_cache: RefCell::new(None) })
    }

    pub fn word_embeddings(&self) -> &Tensor {
        self.embeddings.word_embeddings.embeddings()
    }

    /// Per-head symmetric ALiBi bias `(1, n_heads, seq, seq)` for the actual length.
    /// Memoized on `seq`: identical output, but computed once per length instead of once per
    /// forward (the dominant redundant work in a Jacobian sweep, where `seq` is fixed).
    fn alibi(&self, seq: usize, device: &Device) -> Result<Tensor> {
        if let Some((s, t)) = self.bias_cache.borrow().as_ref() {
            if *s == seq {
                return Ok(t.clone());
            }
        }
        let a = Tensor::arange(0, seq as i64, device)?.to_dtype(DType::F32)?;
        let dist = a.reshape((1, seq))?.broadcast_sub(&a.reshape((seq, 1))?)?.abs()?;
        let n = self.n_heads;
        let mut n2 = 1;
        while n2 < n {
            n2 *= 2;
        }
        let base: Vec<f32> = (1..=n2).map(|v| -1f32 / 2f32.powf((v * 8) as f32 / n2 as f32)).collect();
        let slopes: Vec<f32> = if n2 == n {
            base
        } else {
            base.iter().skip(1).step_by(2).chain(base.iter().step_by(2)).take(n).copied().collect()
        };
        let slopes = Tensor::new(slopes, device)?.reshape((1, n, 1, 1))?;
        let bias = dist.reshape((1, 1, seq, seq))?.broadcast_mul(&slopes)?;
        *self.bias_cache.borrow_mut() = Some((seq, bias.clone()));
        Ok(bias)
    }

    pub fn hidden_states(&self, input_ids: &Tensor) -> Result<Vec<Tensor>> {
        let seq = input_ids.dim(1)?;
        let bias = self.alibi(seq, input_ids.device())?;
        let mut h = self.embeddings.forward(input_ids)?;
        let mut out = vec![h.clone()];
        for layer in &self.layers {
            h = layer.forward(&h, &bias)?;
            out.push(h.clone());
        }
        Ok(out)
    }

    pub fn forward_from(&self, start: usize, hidden: &Tensor) -> Result<Tensor> {
        let seq = hidden.dim(1)?;
        let bias = self.alibi(seq, hidden.device())?;
        let mut h = hidden.clone();
        for layer in &self.layers[start..] {
            h = layer.forward(&h, &bias)?;
        }
        Ok(h)
    }

    /// Like `hidden_states`, but starting from an injected hidden at layer `start`.
    /// Returns `[h[start], …, h[last]]`.
    pub fn hidden_states_from(&self, start: usize, hidden: &Tensor) -> Result<Vec<Tensor>> {
        let seq = hidden.dim(1)?;
        let bias = self.alibi(seq, hidden.device())?;
        let mut h = hidden.clone();
        let mut out = Vec::with_capacity(self.layers.len() - start + 1);
        out.push(h.clone());
        for layer in &self.layers[start..] {
            h = layer.forward(&h, &bias)?;
            out.push(h.clone());
        }
        Ok(out)
    }
}
