# Method → code map (audit T0.2)

Every method claim in PAPER.md §4 resolves to code below. All paths are in the
`jlens` crate unless noted. Line numbers verified against the working tree
(regenerated after the autocorrelation signature landed).

| Method claim (§4) | Function | Location |
|---|---|---|
| δ-broadcast averaged Jacobian `J_ℓ = (1/T)·∂p/∂δ`, batched central finite differences | `Harness::layer_jacobian` | `jlens/src/lib.rs:275` |
| Embedding-space projection `J_emb = (1/‖p‖)(I − eeᵀ)·J_raw` (unit-tested orthogonal to `e`) | `to_embedding_jacobian` | `jlens/src/lib.rs:324` |
| Vocab (logit) lens: standardized `J·h` × tied WordPiece `W_U` → top-k | `vocab_topk` / `vocab_logits` | `jlens/src/lib.rs:350` / `:189` |
| Style lens: `A · normalize(J_emb·h)` | `style_scores` | `jlens/src/lib.rs:366` |
| Style axis = `normalize(centroid(pos) − centroid(neg))` (unit Fisher direction) | `axis` | `jlens/src/lib.rs:381` |
| Axis panel (gender / top-2 educated / top-2 raised; coder traits one-vs-rest) | `author_axes` / `dataset_axes` | `jlens/src/lib.rs:418` / `:471` |
| J-space dictionary (unit rows of `W_U·J_ℓ`) | `jlens_dictionary` | `jlens/src/lib.rs:196` |
| J-space decomposition (non-negative matching pursuit, captured variance) | `jspace_nmp` | `jlens/src/lib.rs:206` |
| Stable rank `‖J‖_F²/σ₁²` (σ₁ via power iteration) | `stable_rank` / `top_singular_value` | `jlens/src/lib.rs:525` / `:695` |
| Effective dim (participation ratio `‖J‖_F⁴/‖JᵀJ‖_F²`) | `effective_dim` | `jlens/src/lib.rs:533` |
| Jackknife SE of the structural metrics (delete-a-group) | `jackknife_se` / `loo_group_mean` | `jlens/src/lib.rs:554` / `:566` |
| Verbalizability (excess kurtosis of the vocab-lens readout) | `excess_kurtosis` | `jlens/src/lib.rs:572` |
| Layer×layer linear CKA | `linear_cka` | `jlens/src/lib.rs:584` |
| Autocorrelation (lag-1 readout persistence vs. position-shuffled null — the 4th Fig-28 metric) | `readout_autocorrelation` | `jlens/src/lib.rs:635` |
| Untrained-model control: random-init `.safetensors` twin (HF `_init_weights`; seeded, cached) | `random_init_checkpoint` | `jlens/src/randinit.rs:98` |
| Untrained-control toggle `JLENS_RANDOM_INIT=<seed>` (swaps weights in the encoder/decoder loaders) | `random_init_seed` | `jlens/src/randinit.rs:37` |

All four structural signatures the paper reports (stable rank, effective dim,
verbalizability, autocorrelation) plus CKA are implemented; the decoder track
reuses `stable_rank`, `effective_dim`, `excess_kurtosis`, and
`readout_autocorrelation` on GPT-2 (`src/bin/decoder_structural.rs`).
