# Method → code map (audit T0.2)

Every method claim in PAPER.md §4 resolves to code below. All paths are in the
`jlens` crate unless noted. Line numbers verified against the working tree.

| Method claim (§4) | Function | Location |
|---|---|---|
| δ-broadcast averaged Jacobian `J_ℓ = (1/T)·∂p/∂δ`, batched central finite differences | `Harness::layer_jacobian` | `jlens/src/lib.rs:241` |
| Embedding-space projection `J_emb = (1/‖p‖)(I − eeᵀ)·J_raw` (unit-tested orthogonal to `e`) | `to_embedding_jacobian` | `jlens/src/lib.rs:290` |
| Vocab (logit) lens: standardized `J·h` × tied WordPiece `W_U` → top-k | `vocab_topk` / `vocab_logits` | `jlens/src/lib.rs:316` / `:155` |
| Style lens: `A · normalize(J_emb·h)` | `style_scores` | `jlens/src/lib.rs:332` |
| Style axis = `normalize(centroid(pos) − centroid(neg))` (unit Fisher direction) | `axis` | `jlens/src/lib.rs:347` |
| Axis panel (gender / top-2 educated / top-2 raised; coder traits one-vs-rest) | `author_axes` / `dataset_axes` | `jlens/src/lib.rs:384` / `:437` |
| J-space dictionary (unit rows of `W_U·J_ℓ`) | `jlens_dictionary` | `jlens/src/lib.rs:162` |
| J-space decomposition (non-negative matching pursuit, captured variance) | `jspace_nmp` | `jlens/src/lib.rs:172` |
| Stable rank `‖J‖_F²/σ₁²` (σ₁ via power iteration) | `stable_rank` / `top_singular_value` | `jlens/src/lib.rs:479` / `:599` |
| Effective dim (participation ratio `‖J‖_F⁴/‖JᵀJ‖_F²`) | `effective_dim` | `jlens/src/lib.rs:487` |
| Verbalizability (excess kurtosis of the vocab-lens readout) | `excess_kurtosis` | `jlens/src/lib.rs:505` |
| Layer×layer linear CKA | `linear_cka` | `jlens/src/lib.rs:517` |
| **Autocorrelation** signature (paper's 4th Fig-28 metric) | — | **NOT YET IMPLEMENTED** (T1.1) |

## Binary → shipped asset

| Binary | Produces | Consumed by |
|---|---|---|
| `jlens/src/bin/jlens.rs` | `web/assets/person2vec-jlens-{minilm,coders}.json` | `/jlens` viewer (`web/src/views/jlens.rs`) |
| `jlens/src/bin/steer.rs` | `web/assets/person2vec-identity-{minilm,coders}.json` (part A) | `/jlens` IdentityPanel |
| `jlens/src/bin/fingerprint.rs` | `web/assets/person2vec-fingerprint-{minilm,coders}.json` | `/jlens` FingerprintPanel |
| `jlens/src/bin/spike.rs` | (stdout) faithfulness gate: candle forward vs shipped fastembed, cosine > 0.99 | — |
| `jlens/src/bin/steer.rs` (part B) | (stdout only — to be shipped in T1.3) | — |
| `jlens/src/bin/decoder.rs` | (stdout only — Phase 3 scaffold) | — |

## Honesty flags (things the code does NOT do — must not be claimed)
- **Autocorrelation** structural metric: not implemented (only named in `DECODER_PLAN.md`). → T1.1.
- **Behavioral experiments** (verbal report, reasoning swaps, ablation-kills-reasoning, ignition
  on a decoder): not implemented; `decoder.rs` is a stdout scaffold. → Phases 3–5.
- Encoder **steering** (`steer.rs` part B) runs but is not yet persisted to an asset. → T1.3.
