# jlens — averaged-Jacobian ("J-lens") interpretability for the small model

Replicates the method from Anthropic's *"Verbalizable Representations Form a Global
Workspace in Language Models"* (transformer-circuits.pub/2026/workspace) on the model
author2vec uses for prose: **`sentence-transformers/all-MiniLM-L6-v2`** (a 6-layer BERT
encoder, d=384). Because the model is used for *embeddings* (not next-token prediction),
the method is adapted with two readout heads off the same layer Jacobians:

- **Vocab lens** — read `J_ℓ` through the tied WordPiece embeddings (a "logit lens"; noisy,
  since this checkpoint ships no MLM head).
- **Style lens** — read `J_ℓ` onto author2vec's own recovered style axes (gender, education,
  upbringing). No unembedding needed; this is the trustworthy signal.

The heavy compute (autodiff through the layers) is done natively in Rust via **candle** —
not fastembed/ort, which is forward-only. Everything is precomputed offline and shipped as
`web/assets/person2vec-jlens-minilm.json` for the Dioxus `/jlens` viewer.

## Run

```bash
# quick re-validation: pooling gate + one-layer Jacobian
cargo run -p jlens --bin spike --release

# full pipeline → web/assets/person2vec-jlens-minilm.json  (arg = #passages to average)
cargo run -p jlens --bin jlens --release 96
```

## Options / knobs (env vars)

| var | default | meaning |
|-----|---------|---------|
| `JLENS_DEVICE` | `metal` | `cpu` or `metal`; falls back to CPU if Metal is unavailable |
| `JLENS_JAC_LEN` | `64` | token context for the Jacobian pass — `32` fast · `64` proper · `96`+ thorough (attention is O(T²)) |
| `JLENS_CHUNK` | `192` | finite-difference batch = 2·chunk columns per forward (GPU sync granularity) |

At the default settings the averaged Jacobians run ~11–12 s/passage on an M-series GPU
(compute-bound). Example: `JLENS_DEVICE=cpu cargo run -p jlens --bin jlens --release 32`.

## Method notes

- **δ-broadcast reduction.** Perturb every source position by a shared δ and differentiate
  the masked-mean pooled output — this collapses the intractable full Jacobian to one
  `384×384` matrix per layer (`J_ℓ = (1/T)·∂p/∂δ`).
- **Batched central finite differences.** The Jacobian is built by perturbing basis
  directions in a batch and running one forward through layers `ℓ..` — pure gemm, far faster
  on GPU than one autograd backward per output dim. (candle's fused LayerNorm has a no-op
  backward, so `bert.rs` reimplements LayerNorm from primitive ops for the autograd path.)
- **Faithfulness gate.** The spike asserts candle's masked-mean+L2 embedding matches the
  shipped fastembed vectors (`person2vec-minilm.bin`) at cosine 1.0.

## Honest caveats

- **6 layers is shallow** vs. the paper's 40+, so we don't expect the clean
  sensory→workspace→motor tripartition; we report the raw depth series. (What we *do* see:
  a low-rank early bottleneck that widens with depth.)
- The **vocab lens is approximate** (tied embeddings, no trained MLM head) — real but noisy.
- The Jacobian context is capped (`JLENS_JAC_LEN`) for tractability; it averages over
  positions, so a short context still captures the layer's general disposition.
