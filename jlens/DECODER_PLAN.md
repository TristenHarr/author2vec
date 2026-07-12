# Decoder-clone plan — the behavioral half of the paper

## Why this is a separate track (honest framing)

The MiniLM work replicates the paper's **mechanistic apparatus** (averaged Jacobian, J-lens,
J-space decomposition, structural depth signatures) and adds a novel authorship/identity
angle. But the paper's **behavioral claims** — verbal report, directed modulation, reasoning
swaps ("spider"→"ant" flips "8"→"6"), selectivity, ablation-kills-reasoning, and the
sensory→workspace→motor regimes across ~100 layers — **cannot be shown on MiniLM**, because
MiniLM is a 6-layer *encoder* that does not generate, reason, or follow instructions.

To replicate those, we need a model that (a) **generates**, and (b) exposes **internal
activations + gradients** (white-box). The Claude API generates but exposes no internals, so
it can't be the subject. → We need **open weights + a decoder**.

## Model choice

- **candle 0.11 has no `gpt2.rs`**, but ships `qwen2.rs`, `phi.rs`, `llama2_c.rs`, gemma, etc.
- **Recommended: `Qwen2.5-0.5B` (or 1.5B)** via candle's `qwen2` — small, open, and *actually
  capable of light reasoning* (GPT-2-small cannot do multi-hop swaps, so it would prove the
  *method* but not the *behavior*). Alternative: vendor a minimal GPT-2 forward ourselves
  (GPT-2 is simple: learned pos-emb, LayerNorm, causal attn, GELU MLP, tied unembedding) for
  a cheap method-only M0.

## Adaptations vs. the MiniLM harness

Same shape as `bert.rs`, but:
1. **Vendor + extend the decoder forward** for per-layer hidden states, a forward-from-layer-ℓ
   path, and the real `lm_head` unembedding (decoders have a *true* unembedding — no
   tied-embedding lens approximation needed).
2. **Causal J-lens**: the averaged Jacobian is `E[∂h_final,t'/∂h_ℓ,t]` over `t' ≥ t` (causal),
   not all positions. The δ-broadcast trick still applies with a causal mask.
3. **RoPE / RMSNorm / SwiGLU / GQA / KV-cache** — reuse candle's impls; ensure the differentiable
   path avoids fused no-op-backward ops (the LayerNorm lesson from MiniLM).
4. **Compute**: d_model ~896 (Qwen0.5B) × 24 layers × causal → the finite-difference Jacobian is
   substantially heavier than MiniLM's. Cap context length; sample layers; Metal required.

## Milestones

- **M0 (method proof)**: load the decoder, compute one layer's averaged Jacobian, apply the
  J-lens with the **real unembedding**, print top tokens for a prompt. Confirms the pipeline on
  a generative decoder.
- **M1 (structural)**: all-layer Jacobians → sensory/workspace/motor curves (top-k next-token
  accuracy, kurtosis, autocorrelation, effective dim) — the paper's Figure 28, on an open model.
- **M2 (behavioral)**: one verbal-report swap experiment (inject/swap a J-lens vector, generate,
  show the output token changes) + one two-hop reasoning swap. These need a capable model (Qwen
  1.5B+), generation loops, and careful prompts.

## Honest scope

A faithful clone of a frontier interpretability paper is a **research program**, not a session.
M0 is very achievable; M2 (behavior) depends on model capability and is where the real cost is.
This doc is the scaffold; the MiniLM authorship study stands on its own as the shipped companion.

## Where our experiments live

- **Persona injection (Einstein vectors)**, **IQ/expertise mirroring**, **suggestibility/
  elicitation** — all require this decoder track (generation). Steering vectors built from a
  person's writing, injected into the residual stream, measured on generation quality/register.
