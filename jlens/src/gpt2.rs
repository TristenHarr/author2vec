//! Minimal GPT-2 forward in candle — vendored for the decoder J-lens track.
//!
//! GPT-2 is a *generative decoder* with a *real* (tied) unembedding, so unlike the
//! MiniLM encoder it needs no logit-lens approximation. Extended (like `bert.rs`) to
//! expose per-layer hidden states, a forward-from-layer path, and the unembedding —
//! the pieces the averaged-Jacobian J-lens needs. Autograd-friendly LayerNorm.

use candle_core::{Device, Result, Tensor, D};
use candle_nn::{embedding, Embedding, Module, VarBuilder};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub n_positions: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    #[serde(default = "default_eps")]
    pub layer_norm_epsilon: f64,
}

fn default_eps() -> f64 {
    1e-5
}

/// LayerNorm from primitive ops (candle_nn's fused path has a no-op backward).
struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    fn load(vb: VarBuilder, size: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: vb.get(size, "weight")?,
            bias: vb.get(size, "bias")?,
            eps,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = x.dim(D::Minus1)? as f64;
        let mean = (x.sum_keepdim(D::Minus1)? / hidden)?;
        let xc = x.broadcast_sub(&mean)?;
        let var = (xc.sqr()?.sum_keepdim(D::Minus1)? / hidden)?;
        let xn = xc.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        xn.broadcast_mul(&self.weight)?.broadcast_add(&self.bias)
    }
}

/// GPT-2's Conv1D: `y = x @ W + b`, with W stored as `(in, out)` (not `(out, in)`).
struct Conv1D {
    weight: Tensor, // (in, out)
    bias: Tensor,   // (out,)
}

impl Conv1D {
    fn load(vb: VarBuilder, nin: usize, nout: usize) -> Result<Self> {
        Ok(Self {
            weight: vb.get((nin, nout), "weight")?,
            bias: vb.get(nout, "bias")?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, nin) = x.dims3()?;
        let nout = self.weight.dim(1)?;
        let y = x.reshape((b * t, nin))?.matmul(&self.weight)?.reshape((b, t, nout))?;
        y.broadcast_add(&self.bias)
    }
}

struct Block {
    ln_1: LayerNorm,
    c_attn: Conv1D,
    c_proj: Conv1D,
    ln_2: LayerNorm,
    mlp_fc: Conv1D,
    mlp_proj: Conv1D,
    n_head: usize,
}

impl Block {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let d = cfg.n_embd;
        Ok(Self {
            ln_1: LayerNorm::load(vb.pp("ln_1"), d, cfg.layer_norm_epsilon)?,
            c_attn: Conv1D::load(vb.pp("attn").pp("c_attn"), d, 3 * d)?,
            c_proj: Conv1D::load(vb.pp("attn").pp("c_proj"), d, d)?,
            ln_2: LayerNorm::load(vb.pp("ln_2"), d, cfg.layer_norm_epsilon)?,
            mlp_fc: Conv1D::load(vb.pp("mlp").pp("c_fc"), d, 4 * d)?,
            mlp_proj: Conv1D::load(vb.pp("mlp").pp("c_proj"), 4 * d, d)?,
            n_head: cfg.n_head,
        })
    }

    fn attn(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        let hd = d / self.n_head;
        let qkv = self.c_attn.forward(x)?; // (b,t,3d)
        let q = qkv.narrow(2, 0, d)?;
        let k = qkv.narrow(2, d, d)?;
        let v = qkv.narrow(2, 2 * d, d)?;
        let split = |t_: &Tensor| -> Result<Tensor> {
            t_.reshape((b, t, self.n_head, hd))?.transpose(1, 2)?.contiguous()
        };
        let (q, k, v) = (split(&q)?, split(&k)?, split(&v)?);
        let scores = (q.matmul(&k.t()?)? / (hd as f64).sqrt())?;
        let scores = scores.broadcast_add(mask)?; // causal
        let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let ctx = probs.matmul(&v)?.transpose(1, 2)?.contiguous()?.reshape((b, t, d))?;
        self.c_proj.forward(&ctx)
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let x = (x + self.attn(&self.ln_1.forward(x)?, mask)?)?;
        let h = self.mlp_proj.forward(&self.mlp_fc.forward(&self.ln_2.forward(&x)?)?.gelu()?)?;
        x + h
    }
}

pub struct Gpt2 {
    wte: Embedding,
    wpe: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    pub cfg: Config,
    device: Device,
}

impl Gpt2 {
    pub fn load(vb: VarBuilder, cfg: Config) -> Result<Self> {
        // openai-community/gpt2 safetensors use bare keys (wte, wpe, h.N, ln_f) — no prefix.
        let device = vb.device().clone();
        let wte = embedding(cfg.vocab_size, cfg.n_embd, vb.pp("wte"))?;
        let wpe = embedding(cfg.n_positions, cfg.n_embd, vb.pp("wpe"))?;
        let blocks = (0..cfg.n_layer)
            .map(|i| Block::load(vb.pp("h").pp(i.to_string()), &cfg))
            .collect::<Result<Vec<_>>>()?;
        let ln_f = LayerNorm::load(vb.pp("ln_f"), cfg.n_embd, cfg.layer_norm_epsilon)?;
        Ok(Self { wte, wpe, blocks, ln_f, cfg, device })
    }

    /// Tied unembedding `(vocab, n_embd)`.
    pub fn unembed(&self) -> &Tensor {
        self.wte.embeddings()
    }

    fn causal_mask(&self, t: usize) -> Result<Tensor> {
        let mut m = vec![0f32; t * t];
        for i in 0..t {
            for j in (i + 1)..t {
                m[i * t + j] = f32::MIN;
            }
        }
        Tensor::from_vec(m, (1, 1, t, t), &self.device)
    }

    /// Residual stream at every depth: `h[0]` = token+pos embeddings, `h[k]` = output
    /// of block `k-1`. Length `n_layer + 1`.
    pub fn hidden_states(&self, input_ids: &Tensor) -> Result<Vec<Tensor>> {
        let (_b, t) = input_ids.dims2()?;
        let pos: Vec<u32> = (0..t as u32).collect();
        let pos = Tensor::new(pos.as_slice(), &self.device)?.reshape((1, t))?;
        let mut h = (self.wte.forward(input_ids)? + self.wpe.forward(&pos)?)?;
        let mask = self.causal_mask(t)?;
        let mut out = vec![h.clone()];
        for blk in &self.blocks {
            h = blk.forward(&h, &mask)?;
            out.push(h.clone());
        }
        Ok(out)
    }

    /// Apply final LayerNorm + unembedding to a residual state → vocab logits.
    pub fn logits(&self, h: &Tensor) -> Result<Tensor> {
        let normed = self.ln_f.forward(h)?;
        normed.broadcast_matmul(&self.unembed().t()?)
    }

    /// Run blocks `start..` from an (already-perturbed) residual state — the tail the
    /// decoder J-lens Jacobian differentiates. Rebuilds the causal mask from the length.
    pub fn forward_from(&self, start: usize, hidden: &Tensor) -> Result<Tensor> {
        let t = hidden.dim(1)?;
        let mask = self.causal_mask(t)?;
        let mut h = hidden.clone();
        for blk in &self.blocks[start..] {
            h = blk.forward(&h, &mask)?;
        }
        Ok(h)
    }
}
