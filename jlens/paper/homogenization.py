#!/usr/bin/env python3
"""Homogenization-over-time analysis (C3): are coders' styles converging in the AI era?

Hypothesis: if a shared external influence (e.g. AI assistants) were homogenizing code style,
cross-coder similarity would RISE in the AI era vs. the pre-AI baseline. Hold-out = time:
pre-2021 commits are provably AI-free (Copilot mid-2021, ChatGPT late-2022).

We report the population-level trend ONLY, with a permutation null. We do NOT attribute AI usage
to any individual — the causal direction is confounded (AI was trained on these devs) and style
drift has many causes (language/project/tooling). Convergence != proof of AI.

Writes web/assets/person2vec-homogenization-coders.json + prints.
"""
import json
import os
import datetime
import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ASSETS = os.path.join(ROOT, "web", "assets")

d = json.load(open(os.path.join(ASSETS, "person2vec-coders.json")))
dim = d["dim"]
ps = d["passages"]
authors = d["authors"]
vecs = np.fromfile(os.path.join(ASSETS, "person2vec-coders.bin"), dtype="<f4").reshape(-1, dim)
assert vecs.shape[0] == len(ps), (vecs.shape, len(ps))
year = np.array([datetime.datetime.utcfromtimestamp(p["authored"]).year if p.get("authored", 0) > 0 else 0 for p in ps])
aid = np.array([p["author_id"] for p in ps])

PRE = (year >= 1990) & (year <= 2020)   # provably AI-free
POST = year >= 2023                     # AI-era
MIN = 8                                 # min passages per coder per era


def centroids(mask):
    out = {}
    for a in range(len(authors)):
        m = mask & (aid == a)
        if m.sum() >= MIN:
            c = vecs[m].mean(0)
            out[a] = c / (np.linalg.norm(c) + 1e-9)
    return out


def mean_cross(cent):
    ks = list(cent)
    s = [float(cent[ks[i]] @ cent[ks[j]]) for i in range(len(ks)) for j in range(i + 1, len(ks))]
    return float(np.mean(s)) if s else float("nan")


pc, qc = centroids(PRE), centroids(POST)
common = sorted(set(pc) & set(qc))
pc = {a: pc[a] for a in common}
qc = {a: qc[a] for a in common}
pre_sim, post_sim = mean_cross(pc), mean_cross(qc)
delta = post_sim - pre_sim

# permutation null: within each common coder, randomly relabel its pre/post passages, recompute Δ.
# (deterministic RNG for reproducibility)
rng = np.random.default_rng(0)
idx_pre = {a: np.where(PRE & (aid == a))[0] for a in common}
idx_post = {a: np.where(POST & (aid == a))[0] for a in common}
null_deltas = []
for _ in range(500):
    pc_n, qc_n = {}, {}
    for a in common:
        pool = np.concatenate([idx_pre[a], idx_post[a]])
        perm = rng.permutation(pool)
        npre = len(idx_pre[a])
        gp, gq = perm[:npre], perm[npre:]
        cp = vecs[gp].mean(0); cq = vecs[gq].mean(0)
        pc_n[a] = cp / (np.linalg.norm(cp) + 1e-9)
        qc_n[a] = cq / (np.linalg.norm(cq) + 1e-9)
    null_deltas.append(mean_cross(qc_n) - mean_cross(pc_n))
null_deltas = np.array(null_deltas)
# two-sided p: fraction of |null| >= |observed|
p_two = float(np.mean(np.abs(null_deltas) >= abs(delta)))

within = float(np.mean([pc[a] @ qc[a] for a in common]))

print("=== HOMOGENIZATION OVER TIME (coders) ===")
print(f"coders spanning both eras (>= {MIN} passages each): {len(common)} / {len(authors)}")
print(f"mean cross-coder similarity  pre-2021 (AI-free): {pre_sim:.4f}")
print(f"mean cross-coder similarity  2023+   (AI-era)  : {post_sim:.4f}")
print(f"observed convergence delta:  {delta:+.4f}   ({'CONVERGED' if delta > 0 else 'diverged / flat'})")
print(f"permutation null delta:  mean {null_deltas.mean():+.4f}, sd {null_deltas.std():.4f}   |   two-sided p = {p_two:.3f}")
print(f"within-coder self-consistency (pre->post cosine): {within:.4f}")
verdict = ("no evidence of homogenization" if not (delta > 0 and p_two < 0.05)
           else "significant convergence (population level; confounds apply)")
print(f"VERDICT: {verdict}")

out = {
    "n_coders_both_eras": len(common),
    "min_passages_per_era": MIN,
    "cross_sim_pre2021": round(pre_sim, 4),
    "cross_sim_post2022": round(post_sim, 4),
    "convergence_delta": round(delta, 4),
    "null_mean": round(float(null_deltas.mean()), 4),
    "null_sd": round(float(null_deltas.std()), 4),
    "p_two_sided": round(p_two, 3),
    "within_coder_consistency": round(within, 4),
    "verdict": verdict,
}
with open(os.path.join(ASSETS, "person2vec-homogenization-coders.json"), "w") as f:
    json.dump(out, f, indent=2)
print("wrote person2vec-homogenization-coders.json")
