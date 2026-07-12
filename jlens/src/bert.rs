//! Vendored + extended BERT forward for all-MiniLM-L6-v2.
//!
//! Structure follows candle-transformers' `models::bert` (Apache-2.0), trimmed to
//! what the J-lens harness needs and **extended** to expose:
//!   * per-layer hidden states `h_0..h_6` (`hidden_states`),
//!   * a forward-from-layer-ℓ path (`forward_from`) — the core of the δ-broadcast
//!     Jacobian, which perturbs the residual stream at layer ℓ and re-runs ℓ..end,
//!   * the tied word-embedding matrix (`word_embeddings`) used as the pseudo-
//!     unembedding `W_U` for the vocab readout.
//!
//! `Config`/`HiddenAct` are reused from candle-transformers (they derive the
//! `config.json` deserialization we need); everything else is reimplemented on
//! candle_nn primitives so the layers are callable and differentiable.

use candle_core::{Result, Tensor, D};
use candle_nn::{embedding, linear, Embedding, Linear, Module, VarBuilder};

pub use candle_transformers::models::bert::{Config, HiddenAct};

fn activation(x: &Tensor, act: HiddenAct) -> Result<Tensor> {
    match act {
        HiddenAct::Gelu => x.gelu_erf(),
        HiddenAct::GeluApproximate => x.gelu(),
        HiddenAct::Relu => x.relu(),
    }
}

/// LayerNorm from primitive ops. We deliberately do NOT use `candle_nn::LayerNorm`:
/// its contiguous fast-path dispatches to a fused custom op whose backward is a
/// no-op, which silently severs the autograd graph (grads never reach the input).
/// This mirrors candle_nn's own differentiable *fallback* math instead.
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

/// Additive attention mask: `(1 - mask) * f32::MIN`, shape `(b,1,1,T)`. For an
/// all-ones mask (single passage, no padding) this is all zeros.
pub fn extended_attention_mask(mask: &Tensor) -> Result<Tensor> {
    let m = mask.unsqueeze(1)?.unsqueeze(1)?; // (b,1,1,T)
    let on = m.ones_like()?;
    (on - &m)?.affine(f32::MIN as f64, 0.0)
}

struct BertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
}

impl BertEmbeddings {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        Ok(Self {
            word_embeddings: embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("word_embeddings"))?,
            position_embeddings: embedding(
                cfg.max_position_embeddings,
                cfg.hidden_size,
                vb.pp("position_embeddings"),
            )?,
            token_type_embeddings: embedding(
                cfg.type_vocab_size,
                cfg.hidden_size,
                vb.pp("token_type_embeddings"),
            )?,
            layer_norm: LayerNorm::load(vb.pp("LayerNorm"), cfg.hidden_size, cfg.layer_norm_eps)?,
        })
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let (_b, seq_len) = input_ids.dims2()?;
        let we = self.word_embeddings.forward(input_ids)?;
        let te = self.token_type_embeddings.forward(token_type_ids)?;
        let mut emb = (&we + te)?;
        let position_ids: Vec<u32> = (0..seq_len as u32).collect();
        let position_ids = Tensor::new(&position_ids[..], input_ids.device())?;
        emb = emb.broadcast_add(&self.position_embeddings.forward(&position_ids)?)?;
        self.layer_norm.forward(&emb)
    }
}

struct SelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    num_heads: usize,
    head_size: usize,
}

impl SelfAttention {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let head_size = cfg.hidden_size / cfg.num_attention_heads;
        let all = cfg.num_attention_heads * head_size;
        Ok(Self {
            query: linear(cfg.hidden_size, all, vb.pp("query"))?,
            key: linear(cfg.hidden_size, all, vb.pp("key"))?,
            value: linear(cfg.hidden_size, all, vb.pp("value"))?,
            num_heads: cfg.num_attention_heads,
            head_size,
        })
    }

    fn shape_heads(&self, xs: &Tensor) -> Result<Tensor> {
        let mut dims = xs.dims().to_vec();
        dims.pop();
        dims.push(self.num_heads);
        dims.push(self.head_size);
        xs.reshape(dims.as_slice())?.transpose(1, 2)?.contiguous()
    }

    fn forward(&self, hidden: &Tensor, ext_mask: &Tensor) -> Result<Tensor> {
        let q = self.shape_heads(&self.query.forward(hidden)?)?;
        let k = self.shape_heads(&self.key.forward(hidden)?)?;
        let v = self.shape_heads(&self.value.forward(hidden)?)?;
        let scores = (q.matmul(&k.t()?)? / (self.head_size as f64).sqrt())?;
        let scores = scores.broadcast_add(ext_mask)?;
        let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let ctx = probs.matmul(&v)?.transpose(1, 2)?.contiguous()?;
        ctx.flatten_from(D::Minus2)
    }
}

struct DenseLn {
    dense: Linear,
    layer_norm: LayerNorm,
}

impl DenseLn {
    fn load(vb: VarBuilder, in_dim: usize, out_dim: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            dense: linear(in_dim, out_dim, vb.pp("dense"))?,
            layer_norm: LayerNorm::load(vb.pp("LayerNorm"), out_dim, eps)?,
        })
    }

    /// Residual add of `input` then LayerNorm (BERT self-output / output block).
    fn forward(&self, hidden: &Tensor, input: &Tensor) -> Result<Tensor> {
        let h = self.dense.forward(hidden)?;
        self.layer_norm.forward(&(h + input)?)
    }
}

/// One transformer block. `forward` is `pub(crate)` so the Jacobian code can drive
/// individual layers when re-running the tail of the network.
pub struct BertLayer {
    attn: SelfAttention,
    attn_out: DenseLn,
    inter_dense: Linear,
    inter_act: HiddenAct,
    out: DenseLn,
}

impl BertLayer {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let attn = SelfAttention::load(vb.pp("attention").pp("self"), cfg)?;
        let attn_out = DenseLn::load(
            vb.pp("attention").pp("output"),
            cfg.hidden_size,
            cfg.hidden_size,
            cfg.layer_norm_eps,
        )?;
        let inter_dense = linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("intermediate").pp("dense"))?;
        let out = DenseLn::load(vb.pp("output"), cfg.intermediate_size, cfg.hidden_size, cfg.layer_norm_eps)?;
        Ok(Self {
            attn,
            attn_out,
            inter_dense,
            inter_act: cfg.hidden_act,
            out,
        })
    }

    pub fn forward(&self, hidden: &Tensor, ext_mask: &Tensor) -> Result<Tensor> {
        let attn = self.attn.forward(hidden, ext_mask)?;
        let attn = self.attn_out.forward(&attn, hidden)?;
        let inter = activation(&self.inter_dense.forward(&attn)?, self.inter_act)?;
        self.out.forward(&inter, &attn)
    }
}

/// all-MiniLM-L6-v2 encoder, extended for interpretability.
pub struct BertModel {
    embeddings: BertEmbeddings,
    pub layers: Vec<BertLayer>,
}

impl BertModel {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        // sentence-transformers checkpoints save a bare BertModel (no prefix).
        let embeddings = BertEmbeddings::load(vb.pp("embeddings"), cfg)?;
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| BertLayer::load(vb.pp("encoder").pp("layer").pp(i.to_string()), cfg))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { embeddings, layers })
    }

    /// The tied WordPiece embedding matrix `(vocab, hidden)` — used as the pseudo-
    /// unembedding `W_U` for the vocab J-lens readout.
    pub fn word_embeddings(&self) -> &Tensor {
        self.embeddings.word_embeddings.embeddings()
    }

    /// Residual stream at every depth: `h[0]` = post-embedding-LayerNorm (input to
    /// layer 0), `h[k]` = output of layer `k-1`. Length `num_layers + 1`; `h[last]`
    /// is the final `last_hidden_state` (pre-pool).
    pub fn hidden_states(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        ext_mask: &Tensor,
    ) -> Result<Vec<Tensor>> {
        let mut h = self.embeddings.forward(input_ids, token_type_ids)?;
        let mut out = Vec::with_capacity(self.layers.len() + 1);
        out.push(h.clone());
        for layer in &self.layers {
            h = layer.forward(&h, ext_mask)?;
            out.push(h.clone());
        }
        Ok(out)
    }

    /// Run layers `start..` from an (already-perturbed) hidden state — the tail the
    /// δ-broadcast Jacobian differentiates. `start` indexes into `hidden_states`, so
    /// `hidden` must be `h[start]` (the input to layer `start`).
    pub fn forward_from(&self, start: usize, hidden: &Tensor, ext_mask: &Tensor) -> Result<Tensor> {
        let mut h = hidden.clone();
        for layer in &self.layers[start..] {
            h = layer.forward(&h, ext_mask)?;
        }
        Ok(h)
    }

    /// Like `hidden_states`, but starting from an (injected) hidden state at layer
    /// `start` — so `hidden` must be `h[start]`. Returns `[h[start], …, h[last]]`.
    /// Used by the ignition experiment to read commitment at every depth under a
    /// blended input.
    pub fn hidden_states_from(
        &self,
        start: usize,
        hidden: &Tensor,
        ext_mask: &Tensor,
    ) -> Result<Vec<Tensor>> {
        let mut h = hidden.clone();
        let mut out = Vec::with_capacity(self.layers.len() - start + 1);
        out.push(h.clone());
        for layer in &self.layers[start..] {
            h = layer.forward(&h, ext_mask)?;
            out.push(h.clone());
        }
        Ok(out)
    }
}
