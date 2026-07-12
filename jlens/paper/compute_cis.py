#!/usr/bin/env python3
"""Confidence intervals for every headline accuracy — the uncertainty a 10/10 paper reports.

All inputs are already-shipped bundle numbers (counts, per-layer accuracies, ignition stds), so
nothing here needs a model re-run. Wilson score intervals for proportions k/n; SEM bands for the
ignition separation (std shipped, n_pairs known). Writes person2vec-cis.json; build_ledger.py then
makes every bound traceable so the paper can cite "89.7% [83.1, 93.9]" without inventing a number.
"""
import json
import os
from math import sqrt

ASSETS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "web", "assets"))
N_IDENTITY = 250  # steer.rs n_align default; the identity decode leave-one-out sample (documented)
Z = 1.96


def load(fn):
    p = os.path.join(ASSETS, fn)
    return json.load(open(p)) if os.path.exists(p) else None


def wilson(k, n, z=Z):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 3), round(min(1.0, c + h), 3))


out = {"z": Z, "note": "Wilson 95% CIs on proportions; SEM bands (std/sqrt n) for ignition."}

# ---- §5.9 per-coder recognizability (n=126 each; correct/total shipped) ----
cod = load("person2vec-coders.json")
if cod and cod.get("results", {}).get("per_author"):
    names = [a["name"] for a in cod["authors"]]
    rec = []
    for pa in cod["results"]["per_author"]:
        lo, hi = wilson(pa["correct"], pa["total"])
        rec.append({"name": names[pa["author_id"]], "acc": round(pa["accuracy"], 3),
                    "correct": pa["correct"], "total": pa["total"], "lo": lo, "hi": hi})
    rec.sort(key=lambda r: r["acc"])
    out["coders_recognizability"] = rec

# ---- §5.1 identity per-layer + output (n=250) ----
out["identity"] = {}
for ds in ("minilm", "coders"):
    idb = load(f"person2vec-identity-{ds}.json")
    if not idb:
        continue
    pl = [{"acc": round(a, 3), "lo": wilson(round(a * N_IDENTITY), N_IDENTITY)[0],
           "hi": wilson(round(a * N_IDENTITY), N_IDENTITY)[1]} for a in idb["per_layer"]]
    o = round(idb["output_acc"], 3)
    out["identity"][ds] = {"n": N_IDENTITY, "per_layer": pl,
                           "output": {"acc": o, "lo": wilson(round(idb["output_acc"] * N_IDENTITY), N_IDENTITY)[0],
                                      "hi": wilson(round(idb["output_acc"] * N_IDENTITY), N_IDENTITY)[1]}}

# ---- §5.7 Big Five (n=2467) Wilson vs majority ----
bf = load("person2vec-bigfive.json")
if bf:
    n = bf["n_essays"]
    out["bigfive"] = {}
    for t, v in bf["traits"].items():
        lo, hi = wilson(round(v["loo_acc"] * n), n)
        out["bigfive"][t] = {"acc": v["loo_acc"], "majority": v["majority"], "lo": lo, "hi": hi,
                             "clears_majority": lo > v["majority"]}

# ---- §5.2 ignition SEM bands (std/sqrt n_pairs) ----
out["ignition"] = {}
for ds in ("minilm", "coders"):
    ig = load(f"person2vec-ignition-{ds}.json")
    if not ig or "separation_std" not in ig:
        continue
    n = ig["n_pairs"]
    sem = [round(s / sqrt(n), 3) for s in ig["separation_std"]]
    out["ignition"][ds] = {"n_pairs": n, "separation_mean": [round(x, 3) for x in ig["separation_mean"]],
                           "separation_sem": sem}

# ---- §5.10 AI fingerprint Wilson + null SD ----
aifp = load("person2vec-aifp.json")
if aifp:
    lo, hi = wilson(round(aifp["task_controlled_model_id"] * aifp["n_samples"]), aifp["n_samples"])
    out["aifp"] = {"id_acc": aifp["task_controlled_model_id"], "n": aifp["n_samples"],
                   "lo": lo, "hi": hi, "null_mean": aifp.get("task_controlled_null_mean")}

# ---- §5.7 blog demographics (n=138) Wilson ----
bd = load("person2vec-blog-demographics.json")
if bd:
    n = bd["n_authors"]
    out["blog"] = {}
    for a, v in bd["attributes"].items():
        lo, hi = wilson(round(v["loo_acc"] * v["n"]), v["n"])
        out["blog"][a] = {"acc": v["loo_acc"], "n": v["n"], "lo": lo, "hi": hi}

json.dump(out, open(os.path.join(ASSETS, "person2vec-cis.json"), "w"), indent=1)

# human-readable summary
print("=== confidence intervals (Wilson 95% unless noted) ===")
if "coders_recognizability" in out:
    r = out["coders_recognizability"]
    print(f"per-coder recognizability (n=126): top {r[-1]['name']} {r[-1]['acc']*100:.1f}% "
          f"[{r[-1]['lo']*100:.1f},{r[-1]['hi']*100:.1f}]  bottom {r[0]['name']} {r[0]['acc']*100:.1f}% "
          f"[{r[0]['lo']*100:.1f},{r[0]['hi']*100:.1f}]  disjoint={r[-1]['lo']>r[0]['hi']}")
for ds in out.get("identity", {}):
    o = out["identity"][ds]["output"]; best = max(out["identity"][ds]["per_layer"], key=lambda x: x["acc"])
    print(f"identity {ds}: best-internal {best['acc']*100:.1f}% [{best['lo']*100:.1f},{best['hi']*100:.1f}] "
          f"vs output {o['acc']*100:.1f}% [{o['lo']*100:.1f},{o['hi']*100:.1f}]")
for t, v in out.get("bigfive", {}).items():
    print(f"bigfive {t}: {v['acc']*100:.1f}% [{v['lo']*100:.1f},{v['hi']*100:.1f}] vs maj "
          f"{v['majority']*100:.1f} clears={v['clears_majority']}")
print("wrote person2vec-cis.json")
