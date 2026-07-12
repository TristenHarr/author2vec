#!/usr/bin/env python3
"""AI-model fingerprint analysis: do frontier models have distinct code styles, and how do they
relate to human coders? Clean ground truth (we know which model wrote each sample).

Inputs (scratchpad): ai_embed_out.json ([{id:'model##idx', vec:[768]}]).
Human coders: web/assets/person2vec-coders.{json,bin}. Writes person2vec-aifp.json + prints.
"""
import json
import os
import numpy as np

SCRATCH = "/private/tmp/claude-501/-Users-tristenharr-duhclaudeknowsyou/4d4c9584-9d59-4b78-a64a-36a229af2782/scratchpad"
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ASSETS = os.path.join(ROOT, "web", "assets")


def unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-9)


emb = json.load(open(os.path.join(SCRATCH, "ai_embed_out.json")))
X = np.array([e["vec"] for e in emb])
models = [e["id"].split("##")[0] for e in emb]
uniq = sorted(set(models))
short = {m: m.split("/")[-1] for m in uniq}
midx = np.array([uniq.index(m) for m in models])

# ---- 1. model separability: leave-one-out nearest-model-centroid (ground truth = model) ----
correct = 0
for i in range(len(X)):
    best, bi = -2, -1
    for k, m in enumerate(uniq):
        sel = (midx == k) & (np.arange(len(X)) != i)
        if sel.sum() == 0:
            continue
        c = unit(X[sel].mean(0))
        s = float(unit(X[i]) @ c)
        if s > best:
            best, bi = s, k
    if bi == midx[i]:
        correct += 1
acc = correct / len(X)
print("=== AI MODEL FINGERPRINTS ===")
print(f"samples: {len(X)} across {len(uniq)} models")
print(f"leave-one-out model ID accuracy: {acc*100:.1f}%   (chance {100/len(uniq):.1f}%)  <- can we tell which AI wrote it?")

# ---- 2. cross-model centroid similarity (do the AIs cluster / how distinct) ----
cents = {m: unit(X[midx == k].mean(0)) for k, m in enumerate(uniq)}
print("\ncross-model centroid cosine (1.0 = identical style):")
print("        " + "  ".join(f"{short[m][:10]:>10}" for m in uniq))
for a in uniq:
    row = "  ".join(f"{float(cents[a]@cents[b]):>10.2f}" for b in uniq)
    print(f"{short[a][:8]:>8} {row}")

# ---- 3. AI vs human: nearest human coder to each model ----
cj = json.load(open(os.path.join(ASSETS, "person2vec-coders.json")))
dim = cj["dim"]
cvec = np.fromfile(os.path.join(ASSETS, "person2vec-coders.bin"), dtype="<f4").reshape(-1, dim)
aid = np.array([p["author_id"] for p in cj["passages"]])
coders = {cj["authors"][a]["name"]: unit(cvec[aid == a].mean(0)) for a in range(len(cj["authors"]))}
hmean = unit(np.mean(list(coders.values()), 0))
print("\nnearest HUMAN coder to each model (and AI-vs-human-centroid gap):")
for m in uniq:
    sims = sorted(((float(cents[m] @ hv), n) for n, hv in coders.items()), reverse=True)
    near = sims[0]
    to_human = float(cents[m] @ hmean)
    print(f"  {short[m]:<22} nearest coder: {near[1]:<20} ({near[0]:.2f})   sim-to-human-avg {to_human:.2f}")

# ---- 4. are AI samples separable from human samples? ----
ai_centroid = unit(X.mean(0))
ai_to_ai = float(np.mean([unit(x) @ ai_centroid for x in X]))
# human passages' cosine to the AI centroid vs to their own human mean
h_to_ai = float(np.mean([unit(cvec[i]) @ ai_centroid for i in range(0, len(cvec), 5)]))
print(f"\nAI samples' mean cosine to AI centroid: {ai_to_ai:.3f}")
print(f"human samples' mean cosine to AI centroid: {h_to_ai:.3f}   (gap => AI code occupies its own region)")

out = {
    "n_samples": len(X),
    "models": [short[m] for m in uniq],
    "model_id_accuracy": round(acc, 3),
    "chance": round(1 / len(uniq), 3),
    "cross_model_cosine": {short[a]: {short[b]: round(float(cents[a] @ cents[b]), 3) for b in uniq} for a in uniq},
    "nearest_coder": {short[m]: max(coders.items(), key=lambda kv: float(cents[m] @ kv[1]))[0] for m in uniq},
    "ai_vs_human": {"ai_to_ai": round(ai_to_ai, 3), "human_to_ai": round(h_to_ai, 3)},
}
json.dump(out, open(os.path.join(ASSETS, "person2vec-aifp.json"), "w"), indent=1)
print("\nwrote person2vec-aifp.json")
