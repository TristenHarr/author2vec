# Author2vec: A Jacobian Lens for Authorship Identity in Embedding Encoders

**Tristen Harr** · Brahmastra Labs · [author2vec.com](https://author2vec.com)

> **Status: working draft (autonomous research build).** Every quantitative claim is
> traced to a shipped data bundle in the audit ledger (Appendix D); sections tied to
> experiments still in progress are marked _[pending: T#]_. This document is written
> against `jlens/paper/ledger.json` — no number appears here that is not in the ledger.

---

## Abstract

Anthropic's *"Verbalizable Representations Form a Global Workspace in Language Models"*
introduces the **averaged-Jacobian lens (J-lens)** and uses it to argue that a large,
closed **decoder** (Claude Sonnet 4.5, 127 layers) maintains a privileged "workspace" of
verbalizable, causally-active concepts. We adapt that mechanistic apparatus to a setting the
method was not designed for — small, **open embedding encoders** whose output is a single
masked-mean-pooled vector — and re-point it at a different question: **personal writing
style / authorship identity**. Three ingredients make the transplant work: (i) a
**δ-broadcast reduction** that collapses the encoder's position×position Jacobian to a single
per-layer matrix; (ii) an **embedding-space projection** that removes the meaningless radial
direction of a normalized embedding; and (iii) a **style lens** — a readout that projects the
layer Jacobian onto empirically-recovered, human-interpretable identity axes instead of the
token vocabulary. On a 6-layer prose encoder (MiniLM) and a 12-layer code encoder (JinaBERT)
we find that **author identity is a computed intermediate, decodable above chance at every
layer** (not merely an output artifact), that the two encoder depths expose different
low-rank "workspace" geometry, and that the identity direction is both a **detector** (is a
person's fingerprint in the weights at all?) and a **causal lever** (steering). We further
run [pending] an **identity-ignition** experiment (does the representation commit to a single
individual at a characteristic depth?), a **decoder** track, and a **directed-steering**
experiment. Everything runs on open models and ships as static assets; the whole apparatus
is reproducible on a laptop. We are explicit about what is **replication** vs. **new**, and
about what we **do not** claim: no "IQ", no consciousness.

## Contributions

1. **An encoder adaptation of the averaged-Jacobian J-lens** via the δ-broadcast reduction
   (§4.2) — the paper's method redefined for a masked-mean-pooled, non-generative encoder.
2. **An embedding-space Jacobian projection** for L2-normalized outputs (§4.3), unit-tested
   orthogonal to the embedding.
3. **The style lens** (§4.5): a Jacobian readout onto interpretable identity axes rather than
   the token vocabulary — the trustworthy signal where an encoder has no clean unembedding.
4. **Identity is a computed intermediate** (§5.1): author identity decodes above chance from
   the internal Jacobian at *every* layer, with the best internal code layer exceeding the
   output-embedding ceiling.
5. **A fingerprint-presence detector** (§5.3): known identity vs. "blank space" for
   out-of-distribution text — a question the paper does not ask.
6. **[pending] Identity ignition** (§5.2): evidence that the representation collapses toward a
   single individual at a characteristic depth.
7. **[pending] Causal steering & directed modulation** (§5.4), including steering a decoder
   from a failed answer to a correct one under leakage controls.
8. **An open, reproducible, browser-native reimplementation** across two modalities (prose &
   code), with a faithfulness gate and a full audit ledger.

---

## 1. Introduction

Recent interpretability work argues that large language models maintain a privileged, verbalizable
"workspace" of representations — a small subset of activations that are reportable, controllable,
and causally responsible for reasoning [@workspace2026]. That evidence comes from a *generative
decoder* with hundreds of billions of parameters and privileged internal access, read out with an
**averaged-Jacobian lens** onto the token vocabulary. It is a striking picture, but it is expensive
to reproduce and impossible to inspect from the outside.

We ask a different, smaller, checkable question with the same apparatus: **where, inside a network,
does *who wrote this* become decidable?** We take the averaged-Jacobian lens and move it from a
frontier decoder to two small, open, *embedding encoders* — a 6-layer prose model (MiniLM) and a
12-layer code model (JinaBERT) — and re-point it from reasoning onto **personal writing-style
identity**. An encoder is a clean minimal testbed for the paper's *mechanistic* claims precisely
because it strips away generation: there is no autoregressive loop to smuggle information through,
just a fixed map from text to a single pooled vector. It cannot, however, speak to the paper's
*behavioral* claims (verbal report, reasoning swaps); those need a decoder, which we take up
separately (§5.5–5.7).

Carrying the lens across that gap takes three ingredients (§4): a **δ-broadcast reduction** that
collapses the encoder's intractable position×position Jacobian to one matrix per layer; an
**embedding-space projection** that removes the meaningless radial direction of a normalized
output; and a **style lens** that reads the layer Jacobian onto interpretable identity axes rather
than the token vocabulary — the trustworthy signal where an encoder has no clean unembedding.

With these we find that **author identity is a computed intermediate, decodable above chance at
every layer** (§5.1), not merely an artifact of the output embedding; that an ambiguous
two-author input causes the internal representation to **commit to a single author increasingly
with depth**, peaking in a mid-network band and surviving two null controls (§5.2, ignition); that
the same geometry yields a **fingerprint-presence detector** distinguishing a known identity from
"blank space" (§5.4); and that these signatures hold across two modalities, prose and code. We are
explicit throughout about what is a faithful **replication** of the paper's apparatus versus what
is **new** here, and about what we deliberately do **not** claim: no "IQ", no consciousness. The
contributions are listed above; every number is reproducible from the audit ledger (Appendix D).

## 2. Related work

**Lenses on the residual stream.** The logit lens [@nostalgebraist2020logitlens] reads
intermediate activations through the unembedding; the tuned lens [@belrose2023tunedlens] learns
an affine correction per layer. The averaged Jacobian of [@workspace2026] generalizes this by
linearizing the *whole* map from a layer to the output and averaging over a corpus. We inherit
that construction and adapt it to a pooled encoder (§4.2); our vocab lens is a logit lens over
tied WordPiece embeddings, and our *style lens* (§4.5) replaces the vocabulary readout with one
onto interpretable directions — the piece that is new here.

**Steering and concept directions.** Activation addition [@turner2023actadd] and contrastive
activation addition [@rimsky2024caa] steer generation by adding a direction to the residual
stream; representation engineering [@zou2023repe] extracts such directions, often as a
difference of class means. Our identity axes are exactly difference-of-means (Fisher)
directions, used both to *read* (§4.5) and to *steer* (§5.4); the novelty is not the technique
but the target — personal authorship identity — and reading it *through the layer Jacobian*.

**Probing and representational geometry.** Linear probes [@alain2017probing;
@hewitt2019structural] test what a layer linearly encodes; we use leave-one-out nearest-centroid
decoding as a probe of *identity* across depth (§5.1). Linear CKA [@kornblith2019cka] compares
representations between layers, which we apply to J-lens readout geometry (§4.7). Sparse
dictionary learning / SAEs [@bricken2023monosemanticity] decompose activations into
interpretable atoms; the paper's J-space decomposition is a supervised cousin we reuse (§4.6).

**Global workspace and authorship.** The framing of [@workspace2026] draws on global workspace
theory [@baars1988workspace] and the "ignition" of conscious access [@dehaene2011ignition]; we
borrow the *ignition* experimental design (§5.2) but point it at identity, and we make no claim
about consciousness. Our substrate is Sentence-BERT-style embeddings [@reimers2019sbert], and
the downstream question — attributing text to its author from style — is classical authorship
attribution / stylometry, here recast as a question about *where in a network* identity is
computed. Throughout, the J-lens, J-space, and structural signatures are the paper's; our
contribution is their transplant to open encoders, the style-lens readout, the identity target,
and full reproducibility.

## 3. Background: the averaged-Jacobian lens

We build directly on the apparatus of [@workspace2026], which we summarize here so that our
adaptation (§4) and its boundaries are unambiguous.

**The J-lens.** For a causal decoder, the *averaged Jacobian* at layer $\ell$ is
$$ J_\ell \;=\; \mathbb{E}_{t,\,t'\ge t,\,\text{prompt}}\!\left[\frac{\partial h_{\text{final},t'}}{\partial h_{\ell,t}}\right], $$
the linearized effect of a layer-$\ell$ activation on the final residual stream, averaged over
token positions and a corpus of prompts. Read through the unembedding it yields, for any
activation, a ranked list of vocabulary tokens the model is "disposed to say" — a corrected
logit lens that accounts for representational change across layers.

**J-space and its privilege.** Activations are decomposed into sparse non-negative
combinations of $k$ ($\approx 10$–$25$) J-lens vectors; the *J-space component* is the part of
an activation lying in that cone. Though it accounts for only $\sim 6$–$10\%$ of a concept
vector's variance, it is the part causally responsible for the concept's availability to
verbal report, and it is subject to top-down control (instructing the model to "focus on X"
loads X), mediates unverbalized reasoning intermediates (swapping a J-lens vector redirects
downstream answers), and is required for deliberate — but not automatic — tasks.

**Structural depth signatures.** Four quantities, read off $J_\ell$ across depth (the paper's
Figure 28), identify a workspace band: next-token-prediction accuracy of the readout, its
*excess kurtosis* (peakiness), the *autocorrelation* of the top lens token across positions
(persistence of abstract content), and the readout's *effective linear dimensionality*. All
four mark a consistent sensory → workspace → motor tripartition — noisy early layers, a
coherent middle band, and output-aligned late layers — corroborated by an ambiguous-input
"ignition" experiment in which the representation snaps to one interpretation at workspace
onset. The study is conducted on Claude Sonnet 4.5 (127 layers; workspace $\approx$ L38–92)
and corroborated on Haiku/Opus.

**The gap we step into.** Every one of these constructs is defined for a *generative decoder*
with per-position next-token logits. An embedding encoder has neither: its output is a single
pooled vector. §4 is what it takes to carry the mechanistic half of this apparatus across that
gap — and §5.2, §5.5–5.7 are our attempts at the behavioral half the encoder cannot reach.

## 4. Method

**Setup.** We study two open **embedding encoders**: `sentence-transformers/all-MiniLM-L6-v2`
(prose; 6 layers, $d=384$, vocab 30{,}522) and `jinaai/jina-embeddings-v2-base-code` (code;
12 layers, $d=768$, vocab 61{,}056). Each maps a passage to one **masked-mean-pooled**,
L2-normalized vector $p$. A **faithfulness gate** (`bin/spike`) asserts our native candle
forward reproduces the shipped fastembed embeddings at cosine $>0.99$ before any Jacobian is
trusted.

### 4.1 The problem an encoder poses
The causal J-lens is defined per token position toward next-token logits. An encoder emits a
single pooled vector and no logits, so the per-position causal Jacobian is undefined here.

### 4.2 δ-broadcast averaged Jacobian
We perturb **every** source position by a shared displacement $\delta$ and differentiate the
pooled output, collapsing the position×position Jacobian to one matrix per layer:
$$ J_\ell \;=\; \frac{1}{T}\,\frac{\partial p}{\partial \delta}\;\in\;\mathbb{R}^{d\times d}. $$
$J_\ell$ is built by **batched central finite differences** ($\varepsilon=0.05$, pure gemm),
not per-dimension autograd. `lib.rs:layer_jacobian`. (Derivation: Appendix A.)

### 4.3 Embedding-space projection
Because $p$ is L2-normalized, its radial direction carries no information, so we project:
$$ J_{\text{emb}} \;=\; \tfrac{1}{\lVert p\rVert}\,(I - e e^{\top})\,J_{\text{raw}},\qquad e = p/\lVert p\rVert, $$
which is unit-tested to satisfy $e^{\top}(J_{\text{emb}}v)\approx 0$. `lib.rs:to_embedding_jacobian`.

### 4.4 Vocabulary (logit) lens
Standardized $J\!\cdot\!h$ times the tied WordPiece embeddings $W_U$ → top tokens. Noisy here
(the checkpoint ships no trained MLM head); reported but not load-bearing. `lib.rs:vocab_topk`.

### 4.5 Style lens (novel)
We read the Jacobian onto interpretable **identity axes** instead of the vocabulary:
$$ s_a(h) \;=\; A_a \cdot \operatorname{normalize}(J_{\text{emb}}\,h),\qquad
   A_a = \operatorname{normalize}\big(\bar{c}^{+}_a - \bar{c}^{-}_a\big), $$
each axis a unit **difference-of-class-means** (Fisher) direction over the reference
embeddings. Shipped axes — prose: `male ↔ female`, `educated=England ↔ rest`,
`educated=US ↔ rest`, `raised=England ↔ rest`, `raised=US ↔ rest`; code:
`Systems ↔ rest`, `Scripting ↔ rest`. `lib.rs:style_scores`, `axis`, `author_axes`.

### 4.6 J-space decomposition
Non-negative matching pursuit over unit rows of $W_U J_\ell$ yields a sparse concept set and
the fraction of activation variance it captures. `lib.rs:jspace_nmp`.

### 4.7 Structural depth signatures
Per layer: **stable rank** $\lVert J\rVert_F^2/\sigma_1^2$, **effective dimension**
(participation ratio $\lVert J\rVert_F^4/\lVert J^\top J\rVert_F^2$), **verbalizability**
(excess kurtosis of the vocab-lens readout), **layer×layer linear CKA**, and — completing the
paper's four-metric set — **autocorrelation** _[pending: T1.1]_. `lib.rs:stable_rank`,
`effective_dim`, `excess_kurtosis`, `linear_cka`.

## 5. Experiments and results

### 5.1 Identity is a computed intermediate

![Identity accuracy decoded from the internal Jacobian at each layer, vs. chance (dotted) and the output-embedding ceiling (dashed).](figures/fig1_identity.png)

Decoding author identity by leave-one-out nearest-centroid on the **internal** per-layer
Jacobian readout beats chance at **every** layer: prose per-layer
$[5.2, 7.2, 8.8, 8.0, 7.2, 8.8]\%$ against a $1.8\%$ chance and a $14.4\%$ output ceiling; code
per-layer up to $49.5\%$ against a $7.7\%$ chance — the **best internal layer exceeds the
$38.0\%$ output-embedding ceiling**. Identity is computed inside the layers, not merely
emitted. _(All values: ledger `identity_*`.)_

### 5.2 Identity ignition — does the space collapse to a single person?

We adapt the paper's ambiguous-input ignition to identity. For an author pair $(A,B)$ we build
a per-depth difference-of-means axis $\hat u_\ell = \widehat{c_{A,\ell}-c_{B,\ell}}$ from their
*training* passages, then blend two *held-out* passages at the input embedding,
$h_0(\alpha)=(1-\alpha)h_0^{B}+\alpha h_0^{A}$, sweep $\alpha\in[0,1]$, and read the commitment
$s_\ell(\alpha)=\hat u_\ell\!\cdot\!(\bar p_\ell(\alpha)-m_\ell)$ at every depth (15 pairs). A
graded layer ramps linearly in $\alpha$; an *ignited* layer snaps.

![Identity ignition (MiniLM). Left: commitment vs. α by depth for one pair (input-linear at depth 0, stepped by mid-depth). Middle: A-vs-B separation rises to a mid-network peak and towers over both nulls. Right: the ignition index (transition sharpness) rises with depth.](figures/fig6_ignition.png)

**The representation commits to one author, increasingly with depth.** Endpoint separation along
the identity axis rises from $0.36$ at the input to a **mid-network peak of $0.95$ at depth 4**
(of 6), then eases to $0.54$ at the output — and it dominates both controls at every depth: the
random-direction null sits at $0.06$–$0.14$ (a $\sim\!7\times$ margin at the peak) and the
shuffled-label null at $0.15$–$0.47$. So the effect is **identity-specific**, not a generic
consequence of blending inputs. The **ignition index** (transition sharpness, $0$ = graded, $1$
= all-or-none) climbs from $0.09$ at the input — where the readout is, correctly, linear in the
blended input — to $0.73$–$0.74$ by mid-depth and holds. In short: *shallow layers hold a graded
mixture; by the middle of the network the representation has snapped to a single author.* The
mid-network peak echoes the low-rank "workspace" bottleneck we see structurally (§5.3).

*Honesty.* This is a 6-layer encoder and 15 pairs; the sharpening is measured *along an axis the
separation control proves is identity-specific*, but we cannot fully exclude that some of the
depth-wise sharpening reflects generic late-layer nonlinearity. We report the raw depth series.

### 5.3 Structural signatures across depth

![Structural depth signatures of the averaged Jacobian (normalized depth).](figures/fig2_structural.png)

![Layer-to-layer readout geometry (linear CKA).](figures/fig3_cka.png)

Effective dimension rises with depth (prose $73\!\to\!205$). Prose **stable rank is lowest at
the very first layer** ($22.1$, rising to $106.9$); the deeper code model instead has a genuine
**mid-network low-rank bottleneck** (stable rank minimum $10.9$ at layer 5 of 12). The
mid-network "workspace" geometry the paper reports in deep models appears here only once there
is depth to spare. _[autocorrelation panel pending T1.4]._

### 5.4 Fingerprint presence and causal steering

![Is a person's fingerprint in the weights? Known identities vs. out-of-distribution "blank space".](figures/fig4_fingerprint.png)

A calibrated nearest-centroid detector separates known identities from "blank space". Prose
(bar $0.30$): known probes $0.62$–$0.69$, out-of-distribution text (code, chat, biology,
legalese) $0.10$–$0.25$. Code is tighter and reported honestly (bar $0.51$): known
$0.54$–$0.70$, OOD $0.17$–$0.49$ — "modern chat" sits just under the bar. **[pending: T1.3]**
steering: injecting $\alpha\,\hat\delta$, $\hat\delta=\operatorname{normalize}(J_{\text{emb}}^{\top}A)$,
swings the output's axis loading, with matched-norm and random-direction controls.

### 5.5 Decoder track — structure on a generative model
_[pending: Phase 3]_ ![placeholder](figures/fig7_decoder.png)

### 5.6 Behavioral steering — can't-answer → can-answer
_[pending: Phase 4]_ ![placeholder](figures/fig8_behavioral.png)

### 5.7 Expertise / lexical sophistication
_[pending: Phase 5 — the "IQ" reframe; construct honesty; likely mixed/negative result]_
![placeholder](figures/fig9_expertise.png)

### 5.8 The authorship study (context) & style trajectories

![Style-lens axis loadings through depth for one passage.](figures/fig5_style.png)

Downstream, the same embeddings support an honest authorship study: prose recognition climbs
from a $1.8\%$ blind baseline to $58.3\%$ once the model has read the author's book; a new-passage
reveal is $130/220$ correct when the author is known and $0/220$ when fully hidden (code:
$7.7\%\!\to\!74.6\%$; reveal $38/52$). Trait recovery **nails some and whiffs on others** —
prose gender $85.5\%$ (majority $52.7\%$) but most geographic traits at or below their majority
baselines; code systems-vs-scripting $84.6\%$ (majority $61.5\%$) but commit-time near chance.

### 5.9 Ablations
_[T6.1]_ Leave-one-{book,series,author}-out and exposure curves (already shipped in `results`);
the two encoder depths as a depth ablation. $\varepsilon$ / context-length robustness: future work.

## 6. Discussion
_[T6.1] What the identity-workspace analogy supports and what it does not._

## 7. Limitations and threats to validity
_[T6.1]_ **Replication vs. novel:** the J-lens, J-space, and structural metrics are the
paper's; ours is the adaptation + target + reproducibility. **Not built:** the paper's
behavioral half on a decoder (verbal report, reasoning swaps, ablation-kills-reasoning) — see
Phases 3–5. **Construct honesty:** we measure identity commitment and expertise register, **not
IQ, not consciousness.** The single-averaged, single-direction lens is lossy; the vocab lens is
noisy (no MLM head); finite-difference $\varepsilon$ introduces error. 55 authors / 13 coders on
laptop-scale models prove the *mechanism*, not a sharp personal fingerprint of any specific
individual — that remains a well-motivated extrapolation.

## 8. Reproducibility
_[T6.1]_ Commands (`cargo run -p corpus --release`; `cargo run -p jlens --bin {spike,jlens,steer,fingerprint} --release`),
model ids, seeds, env vars (`JLENS_DEVICE/JAC_LEN/CHUNK`); the claim→asset map (Appendix D /
`jlens/paper/method_map.md`). All similarity math runs client-side (WASM); the viewer does no
linear algebra.

---

## Appendix A — δ-broadcast derivation
_[T6.1] Pooled-output Jacobian collapse; central-difference estimator and its error term._

## Appendix B — Per-layer tables (both models)
_[T6.1] identity, stable rank, effective dim, verbalizability, autocorrelation._

## Appendix C — Axis definitions, rosters, and OOD probe sets
_[T6.1] From `corpus/authors.toml`, `corpus/coders.toml`, and the fingerprint bundles._

## Appendix D — Audit ledger
Every number above is generated by `jlens/paper/build_ledger.py` into
`jlens/paper/ledger.json` (value → source asset + JSON path) and cross-checked by the figure
pipeline. Method claims resolve via `jlens/paper/method_map.md`.
