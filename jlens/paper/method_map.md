# Method → code map (audit T0.2)

Every method claim in PAPER.md §4 resolves to code below. All paths are in the
`jlens` crate unless noted. Line numbers verified against the working tree
(regenerated after the autocorrelation signature landed).

| Method claim (§4) | Function | Location |
|---|---|---|
| δ-broadcast averaged Jacobian `J_ℓ = (1/T)·∂p/∂δ`, batched central finite differences | `Harness::layer_jacobian` | `jlens/src/lib.rs:258` |
| Embedding-space projection `J_emb = (1/‖p‖)(I − eeᵀ)·J_raw` (unit-tested orthogonal to `e`) | `to_embedding_jacobian` | `jlens/src/lib.rs:307` |
| Vocab (logit) lens: standardized `J·h` × tied WordPiece `W_U` → top-k | `vocab_topk` / `vocab_logits` | `jlens/src/lib.rs:333` / `:172` |
| Style lens: `A · normalize(J_emb·h)` | `style_scores` | `jlens/src/lib.rs:349` |
| Style axis = `normalize(centroid(pos) − centroid(neg))` (unit Fisher direction) | `axis` | `jlens/src/lib.rs:364` |
| Axis panel (gender / top-2 educated / top-2 raised; coder traits one-vs-rest) | `author_axes` / `dataset_axes` | `jlens/src/lib.rs:401` / `:454` |
| J-space dictionary (unit rows of `W_U·J_ℓ`) | `jlens_dictionary` | `jlens/src/lib.rs:179` |
| J-space decomposition (non-negative matching pursuit, captured variance) | `jspace_nmp` | `jlens/src/lib.rs:189` |
| Stable rank `‖J‖_F²/σ₁²` (σ₁ via power iteration) | `stable_rank` / `top_singular_value` | `jlens/src/lib.rs:496` / `:645` |
| Effective dim (participation ratio `‖J‖_F⁴/‖JᵀJ‖_F²`) | `effective_dim` | `jlens/src/lib.rs:504` |
| Verbalizability (excess kurtosis of the vocab-lens readout) | `excess_kurtosis` | `jlens/src/lib.rs:522` |
| Layer×layer linear CKA | `linear_cka` | `jlens/src/lib.rs:534` |
| Autocorrelation (lag-1 readout persistence vs. position-shuffled null — the 4th Fig-28 metric) | `readout_autocorrelation` | `jlens/src/lib.rs:585` |

All four structural signatures the paper reports (stable rank, effective dim,
verbalizability, autocorrelation) plus CKA are implemented; the decoder track
reuses `stable_rank`, `effective_dim`, `excess_kurtosis`, and
`readout_autocorrelation` on GPT-2 (`src/bin/decoder_structural.rs`).
