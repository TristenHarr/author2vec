//! J-lens computational core for all-MiniLM-L6-v2.
//!
//! Everything the offline harness needs: a loaded model (`Harness`), the
//! δ-broadcast averaged Jacobians per layer, the two readout heads (tied-embedding
//! vocab lens + author2vec-native style axes), and the depth-wise structural
//! metrics. The `bin/` targets (`spike`, `jlens`) are thin orchestrators over this.

pub mod bert;
pub mod gpt2;
pub mod jina;

/// A loaded interpretability model — the authors encoder (BERT/MiniLM) or the coders
/// encoder (JinaBERT). Both expose per-layer hidden states, a forward-from-layer path,
/// and a tied word-embedding unembedding; attention masks are built internally.
pub enum LensModel {
    Bert(bert::BertModel),
    Jina(jina::JinaModel),
}

impl LensModel {
    pub fn word_embeddings(&self) -> &Tensor {
        match self {
            LensModel::Bert(m) => m.word_embeddings(),
            LensModel::Jina(m) => m.word_embeddings(),
        }
    }
    pub fn hidden_states(&self, input_ids: &Tensor) -> candle_core::Result<Vec<Tensor>> {
        match self {
            LensModel::Bert(m) => {
                let (b, t) = input_ids.dims2()?;
                let tt = Tensor::zeros((b, t), DType::U32, input_ids.device())?;
                let ones = Tensor::ones((b, t), DType::F32, input_ids.device())?;
                m.hidden_states(input_ids, &tt, &bert::extended_attention_mask(&ones)?)
            }
            LensModel::Jina(m) => m.hidden_states(input_ids),
        }
    }
    pub fn forward_from(&self, layer: usize, hidden: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            LensModel::Bert(m) => {
                let (b, t, _) = hidden.dims3()?;
                let ones = Tensor::ones((b, t), DType::F32, hidden.device())?;
                m.forward_from(layer, hidden, &bert::extended_attention_mask(&ones)?)
            }
            LensModel::Jina(m) => m.forward_from(layer, hidden),
        }
    }
}

use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;
use tokenizers::{Tokenizer, TruncationParams};

use shared::AuthorMeta;

pub const MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";
/// fastembed uses the model's full 512-token window (verified: cosine 1.0 vs the
/// shipped bundle only at 512; 256 diverges).
pub const MAX_LEN: usize = 512;

/// A loaded model + tokenizer, ready to embed and differentiate.
pub struct Harness {
    model: LensModel,
    tok: Tokenizer,
    device: Device,
    pub dim: usize,
    pub num_layers: usize,
}

/// Per-passage forward: residual stream at each depth, sequence length, wordpiece
/// tokens, the L2-normalized embedding `e`, and the pre-norm pooled magnitude `||p||`.
pub struct Forward {
    pub hidden: Vec<Tensor>, // len num_layers+1; hidden[ℓ] is input to layer ℓ, hidden[last]=final
    pub t_len: usize,
    pub tokens: Vec<String>,
    pub embedding: Vec<f32>,
    pub pooled_norm: f32,
}

impl Harness {
    /// The authors model (all-MiniLM-L6-v2). Back-compat entry for the author-only bins.
    pub fn load() -> Result<Self> {
        Self::load_dataset("minilm")
    }

    /// Load the interpretability model for a dataset: `minilm`/`authors` → all-MiniLM
    /// (BERT); `coders` → jina-embeddings-v2-base-code (JinaBERT).
    pub fn load_dataset(dataset: &str) -> Result<Self> {
        // Device is user-selectable (JLENS_DEVICE=cpu|metal); defaults to Metal with a
        // CPU fallback so the harness runs anywhere.
        let device = match std::env::var("JLENS_DEVICE").as_deref() {
            Ok("cpu") => {
                eprintln!("  device: CPU (forced via JLENS_DEVICE)");
                Device::Cpu
            }
            _ => match Device::new_metal(0) {
                Ok(d) => {
                    eprintln!("  device: Metal (GPU)");
                    d
                }
                Err(_) => {
                    eprintln!("  device: CPU (Metal unavailable)");
                    Device::Cpu
                }
            },
        };
        let is_jina = dataset == "coders";
        let model_id = if is_jina { "jinaai/jina-embeddings-v2-base-code" } else { MODEL_ID };
        eprintln!("  model: {model_id}");
        let api = Api::new()?;
        let repo = api.model(model_id.to_string());
        let cfg_bytes = std::fs::read(repo.get("config.json").context("download config.json")?)?;
        let tok_path = repo.get("tokenizer.json").context("download tokenizer.json")?;
        let weights = repo.get("model.safetensors").context("download model.safetensors")?;
        let mut tok = Tokenizer::from_file(tok_path).map_err(|e| anyhow!("tokenizer: {e}"))?;
        tok.with_truncation(Some(TruncationParams {
            max_length: MAX_LEN,
            ..Default::default()
        }))
        .map_err(|e| anyhow!("truncation: {e}"))?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device)? };
        let (model, dim, num_layers) = if is_jina {
            let cfg: jina::Config = serde_json::from_slice(&cfg_bytes)?;
            let (d, nl) = (cfg.hidden_size, cfg.num_hidden_layers);
            // This checkpoint stores tensors at the root (embeddings.*, encoder.layer.N.*).
            (LensModel::Jina(jina::JinaModel::load(vb, &cfg)?), d, nl)
        } else {
            let cfg: bert::Config = serde_json::from_slice(&cfg_bytes)?;
            let (d, nl) = (cfg.hidden_size, cfg.num_hidden_layers);
            (LensModel::Bert(bert::BertModel::load(vb, &cfg)?), d, nl)
        };
        Ok(Self { model, tok, device, dim, num_layers })
    }

    /// The tied WordPiece embedding matrix `(vocab, dim)` — the pseudo-unembedding.
    pub fn word_embeddings(&self) -> &Tensor {
        self.model.word_embeddings()
    }

    /// Run encoder layers `layer..` from an (already-perturbed) hidden state — used by
    /// the steering experiment to re-embed after injecting a direction.
    pub fn forward_from(&self, layer: usize, hidden: &Tensor) -> Result<Tensor> {
        Ok(self.model.forward_from(layer, hidden)?)
    }

    pub fn token_str(&self, id: u32) -> String {
        self.tok.id_to_token(id).unwrap_or_else(|| format!("<{id}>"))
    }

    /// Full vocab logits `W_U · jh` for a (standardized) readout vector, on the model's device.
    pub fn vocab_logits(&self, jh: &[f32]) -> Result<Vec<f32>> {
        let jt = Tensor::new(jh, &self.device)?.reshape((self.dim, 1))?;
        Ok(self.word_embeddings().matmul(&jt)?.squeeze(1)?.to_vec1::<f32>()?)
    }

    /// The J-space dictionary at a layer: unit-normalized rows of `W_U · J_ℓ` — each row
    /// is the residual-space **J-lens vector** `v_t` for token `t` (`logit_t = v_t · h`).
    pub fn jlens_dictionary(&self, j: &[f32]) -> Result<Tensor> {
        let jt = Tensor::new(j, &self.device)?.reshape((self.dim, self.dim))?;
        let d = self.word_embeddings().matmul(&jt)?; // (V, dim)
        let norm = (d.sqr()?.sum_keepdim(1)?.sqrt()? + 1e-6)?; // (V,1)
        Ok(d.broadcast_div(&norm)?)
    }

    /// Non-negative matching pursuit: reconstruct activation `target` from up to `k`
    /// J-lens token directions in `d_unit` (V×dim, unit rows). Returns `(token_id, coeff)`
    /// pairs and the fraction of variance captured — our sparse **J-space** decomposition.
    pub fn jspace_nmp(&self, d_unit: &Tensor, target: &[f32], k: usize) -> Result<(Vec<(u32, f32)>, f32)> {
        let dim = target.len();
        let target_norm2 = dot(target, target).max(1e-9);
        let mut residual = target.to_vec();
        let mut used = std::collections::HashSet::new();
        let mut picked = Vec::new();
        for _ in 0..k {
            let r = Tensor::new(residual.as_slice(), &self.device)?.reshape((dim, 1))?;
            let corr = d_unit.matmul(&r)?.squeeze(1)?.to_vec1::<f32>()?;
            let (mut best, mut best_c) = (usize::MAX, 0f32);
            for (t, &c) in corr.iter().enumerate() {
                if c > best_c && !used.contains(&t) {
                    best_c = c;
                    best = t;
                }
            }
            if best == usize::MAX {
                break;
            }
            used.insert(best);
            let row = d_unit.i(best)?.to_vec1::<f32>()?;
            for (ri, rv) in residual.iter_mut().zip(&row) {
                *ri -= best_c * rv;
            }
            picked.push((best as u32, best_c));
        }
        let cvar = 1.0 - dot(&residual, &residual) / target_norm2;
        Ok((picked, cvar))
    }

    fn encode(&self, text: &str, cap: usize) -> Result<(Tensor, usize, Vec<String>)> {
        let enc = self.tok.encode(text, true).map_err(|e| anyhow!("encode: {e}"))?;
        let mut ids = enc.get_ids().to_vec();
        let mut tokens = enc.get_tokens().to_vec();
        let t = ids.len().min(cap).max(1);
        ids.truncate(t);
        tokens.truncate(t);
        let input_ids = Tensor::new(ids.as_slice(), &self.device)?.reshape((1, t))?;
        Ok((input_ids, t, tokens))
    }

    /// Full forward (up to the model's 512-token window) — matches the shipped embeddings.
    pub fn forward(&self, text: &str) -> Result<Forward> {
        self.forward_capped(text, MAX_LEN)
    }

    /// Forward with the sequence truncated to `cap` tokens. Used for the averaged
    /// Jacobian, whose per-backward cost is dominated by the tail sequence length;
    /// the Jacobian averages over positions, so a short context still captures the
    /// layer's general disposition.
    pub fn forward_capped(&self, text: &str, cap: usize) -> Result<Forward> {
        let (ids, t_len, tokens) = self.encode(text, cap)?;
        let hidden = self.model.hidden_states(&ids)?;
        let last = hidden.last().context("no hidden states")?;
        let p = last.mean(1)?.squeeze(0)?;
        let pooled_norm = p.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        let embedding = p.affine((1.0 / pooled_norm as f64).max(0.0), 0.0)?.to_vec1::<f32>()?;
        Ok(Forward { hidden, t_len, tokens, embedding, pooled_norm })
    }

    /// The δ-broadcast averaged Jacobian at `layer` for one passage: perturb every
    /// source position by a shared δ, measure the masked-mean pooled output.
    /// Returns `J = (1/T)·∂p/∂δ` as a row-major `dim×dim` matrix (`j[i*dim+c] = ∂pᵢ/∂δ_c`).
    ///
    /// Computed by **batched central finite differences**: for a chunk of basis
    /// directions we build a batch `{ h_ℓ ± ε·e_c }` (δ broadcast to every position)
    /// and run one forward through layers `ℓ..`. This is pure batched gemm — far
    /// faster on CPU than one autograd backward per output dim — and every downstream
    /// quantity normalizes, so the absolute scale is immaterial.
    pub fn layer_jacobian(&self, fwd: &Forward, layer: usize) -> Result<Vec<f32>> {
        let dim = self.dim;
        let t = fwd.t_len;
        let eps = 0.05f32;
        let h_op = fwd.hidden[layer].detach(); // (1,T,dim)
        let inv = (1.0 / (2.0 * eps)) * (1.0 / t as f32);
        // Large chunk → few GPU→CPU syncs. Batch = 2·chunk rows; attention is the memory
        // driver at (batch·heads·T²), comfortably bounded here. Tunable via JLENS_CHUNK.
        let chunk = std::env::var("JLENS_CHUNK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(192usize)
            .min(dim);
        let mut j = vec![0f32; dim * dim];
        for start in (0..dim).step_by(chunk) {
            let end = (start + chunk).min(dim);
            let b = end - start;
            let rows = 2 * b;
            // perturbation (rows, dim): row 2c = +ε·e_{start+c}, row 2c+1 = −ε·e_{start+c}
            let mut pert = vec![0f32; rows * dim];
            for c in 0..b {
                pert[(2 * c) * dim + (start + c)] = eps;
                pert[(2 * c + 1) * dim + (start + c)] = -eps;
            }
            let pert = Tensor::new(pert.as_slice(), &self.device)?.reshape((rows, 1, dim))?;
            let batch = h_op.broadcast_as((rows, t, dim))?.broadcast_add(&pert)?;
            let pooled = self
                .model
                .forward_from(layer, &batch)?
                .mean(1)?
                .to_vec2::<f32>()?; // (rows, dim)
            for c in 0..b {
                let (pp, pm) = (&pooled[2 * c], &pooled[2 * c + 1]);
                for i in 0..dim {
                    j[i * dim + (start + c)] = (pp[i] - pm[i]) * inv;
                }
            }
        }
        Ok(j)
    }
}

// ---------------------------------------------------------------------------
// Jacobian post-processing: raw-pooled → normalized-embedding via the local
// L2-normalize Jacobian, so both readouts come from ONE set of backward passes.
// ---------------------------------------------------------------------------

/// `J_emb = (1/‖p‖)·(I − e·eᵀ)·J_raw` — the pooled-embedding Jacobian, evaluated at
/// this passage's operating point (`e`, `‖p‖`).
pub fn to_embedding_jacobian(j_raw: &[f32], e: &[f32], pooled_norm: f32, dim: usize) -> Vec<f32> {
    // colproj[j] = Σ_k e[k]·J_raw[k][j]
    let mut colproj = vec![0f32; dim];
    for k in 0..dim {
        let ek = e[k];
        let row = &j_raw[k * dim..(k + 1) * dim];
        for j in 0..dim {
            colproj[j] += ek * row[j];
        }
    }
    let inv = 1.0 / pooled_norm.max(1e-9);
    let mut out = vec![0f32; dim * dim];
    for i in 0..dim {
        let ei = e[i];
        for j in 0..dim {
            out[i * dim + j] = (j_raw[i * dim + j] - ei * colproj[j]) * inv;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Readouts
// ---------------------------------------------------------------------------

/// Top-`k` vocab tokens of `softmax(W_U · norm(J·h))` — the tied-embedding logit lens.
pub fn vocab_topk(
    h: &Harness,
    j: &[f32],
    activation: &[f32],
    k: usize,
) -> Result<Vec<(u32, f32)>> {
    let dim = h.dim;
    let mut jh = matvec(j, activation, dim);
    standardize(&mut jh);
    let jt = Tensor::new(jh.as_slice(), &h.device)?.reshape((dim, 1))?;
    let logits = h.word_embeddings().matmul(&jt)?.squeeze(1)?.to_vec1::<f32>()?;
    let top = topk(&logits, k);
    Ok(top.into_iter().map(|i| (i as u32, logits[i])).collect())
}

/// Per-axis style score: `A · normalize(J_emb·h)`.
pub fn style_scores(j_emb: &[f32], activation: &[f32], axes: &[Axis], dim: usize) -> Vec<f32> {
    let jh = normalize(matvec(j_emb, activation, dim));
    axes.iter().map(|a| dot(&jh, &a.vec)).collect()
}

// ---------------------------------------------------------------------------
// Style axes (built from the shipped, already-normalized reference embeddings)
// ---------------------------------------------------------------------------

pub struct Axis {
    pub name: String,
    pub vec: Vec<f32>,
}

/// A normalized `centroid(pos) − centroid(neg)` axis over reference embeddings.
pub fn axis(
    name: impl Into<String>,
    passages: &[shared::Passage],
    authors: &[AuthorMeta],
    ref_vecs: &[f32],
    dim: usize,
    pos: impl Fn(&AuthorMeta) -> bool,
    neg: impl Fn(&AuthorMeta) -> bool,
) -> Axis {
    let (mut cp, mut cn) = (vec![0f32; dim], vec![0f32; dim]);
    let (mut np, mut nn) = (0usize, 0usize);
    for (pi, p) in passages.iter().enumerate() {
        let a = &authors[p.author_id];
        let row = &ref_vecs[pi * dim..(pi + 1) * dim];
        if pos(a) {
            for k in 0..dim {
                cp[k] += row[k];
            }
            np += 1;
        } else if neg(a) {
            for k in 0..dim {
                cn[k] += row[k];
            }
            nn += 1;
        }
    }
    let diff: Vec<f32> = (0..dim)
        .map(|k| cp[k] / np.max(1) as f32 - cn[k] / nn.max(1) as f32)
        .collect();
    Axis {
        name: name.into(),
        vec: normalize(diff),
    }
}

/// Build the author style-axis panel: gender + one-vs-rest for the top education
/// and upbringing classes present in the roster.
pub fn author_axes(
    passages: &[shared::Passage],
    authors: &[AuthorMeta],
    ref_vecs: &[f32],
    dim: usize,
) -> Vec<Axis> {
    let mut axes = vec![axis(
        "male ↔ female",
        passages,
        authors,
        ref_vecs,
        dim,
        |a| a.gender == "Male",
        |a| a.gender == "Female",
    )];
    axes.extend(field_axes("educated", passages, authors, ref_vecs, dim, |a: &AuthorMeta| a.educated.clone(), 2));
    axes.extend(field_axes("raised", passages, authors, ref_vecs, dim, |a: &AuthorMeta| a.raised.clone(), 2));
    axes
}

/// One-vs-rest axes for the top `k` classes of a categorical author field.
fn field_axes(
    label: &str,
    passages: &[shared::Passage],
    authors: &[AuthorMeta],
    ref_vecs: &[f32],
    dim: usize,
    field: impl Fn(&AuthorMeta) -> String + Clone,
    k: usize,
) -> Vec<Axis> {
    top_classes(authors, &field, k)
        .into_iter()
        .map(|class| {
            let (c, c2) = (class.clone(), class.clone());
            let (f1, f2) = (field.clone(), field.clone());
            axis(
                format!("{label}={class} ↔ rest"),
                passages,
                authors,
                ref_vecs,
                dim,
                move |a| f1(a) == c,
                move |a| {
                    let v = f2(a);
                    !v.is_empty() && v != c2
                },
            )
        })
        .collect()
}

/// Style axes appropriate to the dataset: coder `traits` (paradigm / language / era)
/// one-vs-rest when present, else the fixed author fields (gender / education / …).
pub fn dataset_axes(passages: &[shared::Passage], authors: &[AuthorMeta], ref_vecs: &[f32], dim: usize) -> Vec<Axis> {
    if !authors.iter().any(|a| !a.traits.is_empty()) {
        return author_axes(passages, authors, ref_vecs, dim);
    }
    let mut names: Vec<String> = Vec::new();
    for a in authors {
        for t in &a.traits {
            if !names.iter().any(|n| n == &t.name) {
                names.push(t.name.clone());
            }
        }
    }
    let mut axes = Vec::new();
    for name in names {
        let nm = name.clone();
        let field = move |a: &AuthorMeta| {
            a.traits.iter().find(|t| t.name == nm).map(|t| t.value.clone()).unwrap_or_default()
        };
        axes.extend(field_axes(&name, passages, authors, ref_vecs, dim, field, 2));
    }
    axes
}

fn top_classes(authors: &[AuthorMeta], field: &dyn Fn(&AuthorMeta) -> String, k: usize) -> Vec<String> {
    use std::collections::HashMap;
    let mut counts: HashMap<String, usize> = HashMap::new();
    for a in authors {
        let v = field(a);
        if !v.is_empty() {
            *counts.entry(v).or_default() += 1;
        }
    }
    let mut v: Vec<(String, usize)> = counts.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v.into_iter().take(k).map(|(s, _)| s).collect()
}

// ---------------------------------------------------------------------------
// Structural metrics across depth
// ---------------------------------------------------------------------------

/// Stable rank `‖J‖_F² / σ₁²`.
pub fn stable_rank(j: &[f32], dim: usize) -> f32 {
    let frob2 = j.iter().map(|x| x * x).sum::<f32>();
    let s1 = top_singular_value(j, dim);
    frob2 / (s1 * s1 + 1e-12)
}

/// Effective dimensionality = participation ratio of the eigenvalues of JᵀJ,
/// `‖J‖_F⁴ / ‖JᵀJ‖_F²` (no eigensolver needed).
pub fn effective_dim(j: &[f32], dim: usize) -> f32 {
    let frob2 = j.iter().map(|x| x * x).sum::<f32>();
    // ‖JᵀJ‖_F² = Σ_ij (Σ_k J_ki J_kj)²
    let mut gram_fro2 = 0f64;
    for i in 0..dim {
        for jj in i..dim {
            let mut s = 0f32;
            for k in 0..dim {
                s += j[k * dim + i] * j[k * dim + jj];
            }
            let s2 = (s as f64) * (s as f64);
            gram_fro2 += if i == jj { s2 } else { 2.0 * s2 };
        }
    }
    ((frob2 as f64 * frob2 as f64) / (gram_fro2 + 1e-12)) as f32
}

/// Excess kurtosis of a readout distribution (peakiness ⇒ verbalizability).
pub fn excess_kurtosis(v: &[f32]) -> f32 {
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    if var < 1e-12 {
        return 0.0;
    }
    let m4 = v.iter().map(|x| (x - mean).powi(4)).sum::<f32>() / n;
    m4 / (var * var) - 3.0
}

/// Linear CKA between two probe-readout matrices (each `n × dim`, row-major).
pub fn linear_cka(x: &[f32], y: &[f32], n: usize, dim: usize) -> f32 {
    let xc = center_rows(x, n, dim);
    let yc = center_rows(y, n, dim);
    let hsic = |a: &[f32], b: &[f32]| -> f64 {
        // ‖aᵀb‖_F² = Σ_pq (Σ_r a_rp b_rq)²  → but cheaper via Gram: Σ_ij (aᵀa)... use direct.
        let mut s = 0f64;
        for p in 0..dim {
            for q in 0..dim {
                let mut acc = 0f32;
                for r in 0..n {
                    acc += a[r * dim + p] * b[r * dim + q];
                }
                s += (acc as f64) * (acc as f64);
            }
        }
        s
    };
    let num = hsic(&xc, &yc);
    let den = (hsic(&xc, &xc) * hsic(&yc, &yc)).sqrt();
    if den < 1e-12 {
        0.0
    } else {
        (num / den) as f32
    }
}

fn center_rows(x: &[f32], n: usize, dim: usize) -> Vec<f32> {
    let mut mean = vec![0f32; dim];
    for r in 0..n {
        for c in 0..dim {
            mean[c] += x[r * dim + c];
        }
    }
    for m in mean.iter_mut() {
        *m /= n as f32;
    }
    let mut out = vec![0f32; n * dim];
    for r in 0..n {
        for c in 0..dim {
            out[r * dim + c] = x[r * dim + c] - mean[c];
        }
    }
    out
}

/// Lag-1 autocorrelation of a per-position readout sequence, relative to a
/// position-shuffled null — the paper's 4th Figure-28 signature, adapted to the
/// encoder. Returns the mean cosine of adjacent-position readouts minus the mean
/// cosine over all distinct position pairs. Positive ⇒ the layer's readout
/// persists across the token sequence (abstract, workspace-like); ≈0 ⇒
/// token-local / noise. Inputs need not be normalized (done internally).
pub fn readout_autocorrelation(seq: &[Vec<f32>]) -> f32 {
    let n = seq.len();
    if n < 3 {
        return 0.0;
    }
    let unit: Vec<Vec<f32>> = seq.iter().map(|r| normalize(r.clone())).collect();
    let mut adj = 0f64;
    for t in 0..n - 1 {
        adj += dot(&unit[t], &unit[t + 1]) as f64;
    }
    adj /= (n - 1) as f64;
    let mut all = 0f64;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                all += dot(&unit[i], &unit[j]) as f64;
            }
        }
    }
    all /= (n * (n - 1)) as f64;
    (adj - all) as f32
}

// ---------------------------------------------------------------------------
// small math
// ---------------------------------------------------------------------------

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn matvec(j: &[f32], v: &[f32], dim: usize) -> Vec<f32> {
    (0..dim).map(|i| dot(&j[i * dim..(i + 1) * dim], v)).collect()
}

pub fn standardize(v: &mut [f32]) {
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
    let std = var.sqrt().max(1e-6);
    for x in v.iter_mut() {
        *x = (*x - mean) / std;
    }
}

pub fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = dot(&v, &v).sqrt().max(1e-9);
    for x in v.iter_mut() {
        *x /= norm;
    }
    v
}

pub fn topk(scores: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    idx.truncate(k);
    idx
}

pub fn top_singular_value(j: &[f32], dim: usize) -> f32 {
    let mut v = normalize((0..dim).map(|i| ((i as f32 * 0.7).sin()).abs() + 0.1).collect());
    for _ in 0..80 {
        let u = matvec(j, &v, dim);
        let mut w = vec![0f32; dim];
        for i in 0..dim {
            let ui = u[i];
            for jj in 0..dim {
                w[jj] += j[i * dim + jj] * ui;
            }
        }
        v = normalize(w);
    }
    let u = matvec(j, &v, dim);
    dot(&u, &u).sqrt()
}

/// Resolve `web/assets/` relative to the crate, matching the corpus convention.
pub fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("web")
        .join("assets")
}

// ===========================================================================
// Tests — lock the correctness of the Jacobian/readout math (TDD).
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn dot_and_matvec() {
        assert!(close(dot(&[1., 2., 3.], &[4., 5., 6.]), 32.0, 1e-5));
        // identity matvec returns the vector unchanged.
        let id = vec![1., 0., 0., 0., 1., 0., 0., 0., 1.];
        assert_eq!(matvec(&id, &[7., 8., 9.], 3), vec![7., 8., 9.]);
        // known 2×2: [[1,2],[3,4]] · [1,1] = [3,7]
        assert_eq!(matvec(&[1., 2., 3., 4.], &[1., 1.], 2), vec![3., 7.]);
    }

    #[test]
    fn normalize_is_unit_and_directional() {
        let v = normalize(vec![3.0, 4.0]);
        assert!(close(dot(&v, &v).sqrt(), 1.0, 1e-6));
        assert!(close(v[0] / v[1], 0.75, 1e-6)); // direction preserved
    }

    #[test]
    fn standardize_zero_mean_unit_std() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        standardize(&mut v);
        let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
        let var: f32 = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / v.len() as f32;
        assert!(close(mean, 0.0, 1e-5));
        assert!(close(var, 1.0, 1e-4));
    }

    #[test]
    fn topk_picks_largest_in_order() {
        assert_eq!(topk(&[0.1, 0.9, 0.3, 0.7], 2), vec![1, 3]);
    }

    #[test]
    fn top_singular_value_of_diagonal() {
        // diag(3,1): σ₁ = 3
        assert!(close(top_singular_value(&[3., 0., 0., 1.], 2), 3.0, 1e-3));
    }

    #[test]
    fn stable_rank_identity_is_n_rank1_is_one() {
        let id3 = vec![1., 0., 0., 0., 1., 0., 0., 0., 1.];
        assert!(close(stable_rank(&id3, 3), 3.0, 1e-2));
        // rank-1: outer([1,1],[1,0]) = [[1,0],[1,0]]
        assert!(close(stable_rank(&[1., 0., 1., 0.], 2), 1.0, 1e-3));
    }

    #[test]
    fn effective_dim_identity_is_n_rank1_is_one() {
        let id3 = vec![1., 0., 0., 0., 1., 0., 0., 0., 1.];
        assert!(close(effective_dim(&id3, 3), 3.0, 1e-3));
        assert!(close(effective_dim(&[1., 0., 1., 0.], 2), 1.0, 1e-3));
    }

    #[test]
    fn linear_cka_self_is_one() {
        // 4 samples × 2 dims
        let x = vec![1., 0., 0., 1., 1., 1., 2., -1.];
        assert!(close(linear_cka(&x, &x, 4, 2), 1.0, 1e-4));
    }

    #[test]
    fn excess_kurtosis_peaky_beats_flat() {
        let flat = vec![-1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0];
        let peaky = vec![0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0];
        assert!(excess_kurtosis(&peaky) > excess_kurtosis(&flat));
    }

    #[test]
    fn autocorrelation_persistent_beats_shuffled() {
        // Smoothly drifting rows: adjacent nearly identical ⇒ high autocorrelation.
        let smooth: Vec<Vec<f32>> = (0..12)
            .map(|t| {
                let a = t as f32 * 0.05;
                vec![a.cos(), a.sin(), 0.2]
            })
            .collect();
        // Sign-alternating rows: adjacent anti-correlated ⇒ low/negative.
        let jumpy: Vec<Vec<f32>> = (0..12)
            .map(|t| {
                let s = if t % 2 == 0 { 1.0 } else { -1.0 };
                vec![s, 0.1, 0.0]
            })
            .collect();
        assert!(readout_autocorrelation(&smooth) > readout_autocorrelation(&jumpy));
        assert!(readout_autocorrelation(&smooth) > 0.0);
        assert!(readout_autocorrelation(&[vec![1.0, 0.0]]) == 0.0); // too short
    }

    #[test]
    fn embedding_jacobian_is_orthogonal_to_e() {
        // J_emb removes the e-component of the output, so eᵀ·(J_emb·v) ≈ 0 for any v.
        let dim = 4;
        let e = normalize(vec![0.3, -0.5, 0.8, 0.1]);
        let j_raw: Vec<f32> = (0..dim * dim).map(|i| ((i as f32 * 1.3).sin())).collect();
        let j_emb = to_embedding_jacobian(&j_raw, &e, 2.0, dim);
        let v = vec![0.7, -0.2, 0.4, 0.9];
        assert!(close(dot(&e, &matvec(&j_emb, &v, dim)), 0.0, 1e-5));
    }
}
