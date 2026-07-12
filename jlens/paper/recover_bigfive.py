#!/usr/bin/env python3
"""Can author2vec recover a VALIDATED measured attribute — Big Five personality — from prose?

Pennebaker & King essays (2,467), each labelled with the writer's Big Five traits from a
self-assessment questionnaire (ground truth, not a guess). We embed with the same MiniLM used
throughout, and for each trait run leave-one-out nearest-centroid (high vs. low). Honest
expectation from the literature: weak-but-real. A shuffled-label null confirms any signal.

Inputs (scratchpad): essays_out.json + essays_labels.json. Writes person2vec-bigfive.json + prints.
"""
import json
import os
import numpy as np

SCRATCH = "/private/tmp/claude-501/-Users-tristenharr-duhclaudeknowsyou/4d4c9584-9d59-4b78-a64a-36a229af2782/scratchpad"
ASSETS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "web", "assets"))

emb = json.load(open(os.path.join(SCRATCH, "essays_out.json")))
labels = json.load(open(os.path.join(SCRATCH, "essays_labels.json")))
X = np.array([e["vec"] for e in emb], dtype=float)          # already L2-normalized
ids = [int(e["id"]) for e in emb]
traits = ["cEXT", "cNEU", "cAGR", "cCON", "cOPN"]
trait_name = {"cEXT": "Extraversion", "cNEU": "Neuroticism", "cAGR": "Agreeableness",
              "cCON": "Conscientiousness", "cOPN": "Openness"}


def loo_nc(X, y):
    """Vectorized leave-one-out nearest-centroid accuracy (cosine == dot; X unit-norm).

    For unit vectors, x_i . x_i = 1, so the own-class centroid with i removed reduces to
    (D[i,c]-1)/(n_c-1) where D = X @ S^T and S_c is the class sum. Fully vectorized so a
    1000-permutation null runs in seconds.
    """
    classes, yi = np.unique(y, return_inverse=True)
    k = len(classes)
    S = np.zeros((k, X.shape[1]))
    for c in range(k):
        S[c] = X[yi == c].sum(0)
    cnt = np.bincount(yi, minlength=k).astype(float)
    D = X @ S.T                                   # n x k
    score = D / cnt[None, :]                       # other-class centroids
    n = len(y)
    rows = np.arange(n)
    score[rows, yi] = (D[rows, yi] - 1.0) / np.maximum(cnt[yi] - 1.0, 1.0)  # own-class, self removed
    return (score.argmax(1) == yi).mean()


N_PERM = 1000
rng = np.random.default_rng(0)
out = {"n_essays": len(X), "model": "sentence-transformers/all-MiniLM-L6-v2",
       "n_perm": N_PERM, "traits": {}}
print(f"=== Big Five recovery from prose ({len(X)} Pennebaker essays, MiniLM; {N_PERM} perms) ===")
print(f"{'trait':<18} {'LOO acc':>8} {'majority':>9} {'lift':>6} {'null':>7} {'p':>8}")
for t in traits:
    y = np.array([1 if labels[i][t] == "y" else 0 for i in ids])
    acc = loo_nc(X, y)
    maj = max(y.mean(), 1 - y.mean())
    perm = np.array([loo_nc(X, rng.permutation(y)) for _ in range(N_PERM)])
    null = float(perm.mean())
    p = float((np.sum(perm >= acc) + 1) / (N_PERM + 1))
    lift = acc - maj
    print(f"{trait_name[t]:<18} {acc*100:>7.1f}% {maj*100:>8.1f}% {lift*100:>+5.1f} {null*100:>6.1f}% {p:>8.4f}")
    out["traits"][trait_name[t]] = {"loo_acc": round(acc, 3), "majority": round(maj, 3),
                                    "lift": round(lift, 3), "shuffled_null": round(null, 3),
                                    "p_value": round(p, 4)}
best = max(out["traits"].items(), key=lambda kv: kv[1]["lift"])
out["best_trait"] = best[0]
out["mean_lift"] = round(float(np.mean([v["lift"] for v in out["traits"].values()])), 3)
print(f"\nbest-recovered trait: {best[0]} (+{best[1]['lift']*100:.1f} pts over majority); mean lift {out['mean_lift']*100:+.1f} pts")
json.dump(out, open(os.path.join(ASSETS, "person2vec-bigfive.json"), "w"), indent=1)
print("wrote person2vec-bigfive.json")
