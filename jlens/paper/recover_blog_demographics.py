#!/usr/bin/env python3
"""Positive controls + a NEGATIVE control on one corpus, one pipeline.

Blog Authorship Corpus (Schler et al. 2006): every blogger self-reports gender, age, and
astrological sign. Gender and age have real linguistic correlates; astrological sign does not.
If author2vec is finding *real* signal and not spurious structure, gender/age should recover
above chance while zodiac should NOT — even though all three are self-reported categorical labels
in the same corpus, embedded and classified identically. Zodiac is the negative control.

Author-level (no per-post leakage): each author = mean of their post embeddings; leave-one-
author-out nearest-centroid vs. majority baseline, with a shuffled-label null. The honest,
apples-to-apples statistic is real-label accuracy vs. shuffled-label accuracy.

Inputs (scratchpad): bac_authors_out.json + bac_authors_labels.json. Writes person2vec-blog-demographics.json.
"""
import json
import os
import numpy as np

SCRATCH = "/private/tmp/claude-501/-Users-tristenharr-duhclaudeknowsyou/4d4c9584-9d59-4b78-a64a-36a229af2782/scratchpad"
ASSETS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "web", "assets"))

emb = {e["id"]: np.asarray(e["vec"], float) for e in json.load(open(os.path.join(SCRATCH, "bac_authors_out.json")))}
labels = {l["id"]: l for l in json.load(open(os.path.join(SCRATCH, "bac_authors_labels.json")))}
ids = [i for i in emb if i in labels]
X = np.array([emb[i] for i in ids])            # already L2-normalized


def loo_nc(Xk, y):
    """Vectorized multiclass leave-one-out nearest-centroid accuracy (cosine==dot; unit vecs).

    Own-class centroid with i removed reduces to (D[i,c]-1)/(n_c-1), D = Xk @ S^T. Fully
    vectorized so a 1000-permutation null is fast even for a 12-way label.
    """
    classes, yi = np.unique(y, return_inverse=True)
    k = len(classes)
    S = np.zeros((k, Xk.shape[1]))
    for c in range(k):
        S[c] = Xk[yi == c].sum(0)
    cnt = np.bincount(yi, minlength=k).astype(float)
    D = Xk @ S.T
    with np.errstate(divide="ignore", invalid="ignore"):
        score = D / cnt[None, :]
    n = len(y)
    rows = np.arange(n)
    score[rows, yi] = (D[rows, yi] - 1.0) / np.maximum(cnt[yi] - 1.0, 1.0)
    score[:, cnt == 0] = -9.0
    return (score.argmax(1) == yi).mean()


N_PERM = 1000


def evaluate(name, y):
    y = np.array(y)
    keep = y != None  # noqa: E711
    Xk, yk = X[keep], y[keep]
    acc = loo_nc(Xk, yk)
    _, counts = np.unique(yk, return_counts=True)
    maj = counts.max() / counts.sum()
    rng = np.random.default_rng(0)
    perm = np.array([loo_nc(Xk, rng.permutation(yk)) for _ in range(N_PERM)])
    null = float(perm.mean())
    p = float((np.sum(perm >= acc) + 1) / (N_PERM + 1))
    return {"n": int(keep.sum()), "classes": int(len(set(yk))), "loo_acc": round(acc, 3),
            "majority": round(float(maj), 3), "shuffled_null": round(null, 3),
            "lift_over_majority": round(acc - maj, 3), "lift_over_null": round(acc - null, 3),
            "p_value": round(p, 4)}


def age_bracket(a):
    if not isinstance(a, int):
        return None
    return "teens(<20)" if a < 20 else "20s" if a < 30 else "30+"


gender = [labels[i].get("gender") for i in ids]
sign = [labels[i].get("sign") for i in ids]
age = [age_bracket(labels[i].get("age")) for i in ids]

out = {"n_authors": len(ids), "model": "sentence-transformers/all-MiniLM-L6-v2",
       "note": "author-level leave-one-out nearest-centroid; zodiac is the negative control",
       "attributes": {}}
print(f"=== Blog corpus demographics: {len(ids)} authors, author-level LOO ({N_PERM} perms) ===")
print(f"{'attribute':<14} {'k':>2} {'LOO':>7} {'major':>7} {'null':>7} {'lift/null':>10} {'p':>8}")
for nm, y in [("gender", gender), ("age", age), ("zodiac", sign)]:
    r = evaluate(nm, y)
    out["attributes"][nm] = r
    tag = "  <- NEGATIVE CONTROL" if nm == "zodiac" else ""
    print(f"{nm:<14} {r['classes']:>2} {r['loo_acc']*100:>6.1f}% {r['majority']*100:>6.1f}% "
          f"{r['shuffled_null']*100:>6.1f}% {r['lift_over_null']*100:>+9.1f} {r['p_value']:>8.4f}{tag}")

json.dump(out, open(os.path.join(ASSETS, "person2vec-blog-demographics.json"), "w"), indent=1)
print("\nwrote person2vec-blog-demographics.json")
