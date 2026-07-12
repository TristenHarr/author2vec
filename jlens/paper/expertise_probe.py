#!/usr/bin/env python3
"""Phase 5 (encoder-side, honest): does a recovered *education* style axis track measurable
lexical sophistication (mean word length, type-token ratio)? A defensible construct — NOT "IQ".

Uses only shipped bundles (no model run). Writes person2vec-expertise-minilm.json + prints.
"""
import json
import os
import re
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.normpath(os.path.join(HERE, "..", "..", "web", "assets"))


def load(fn):
    with open(os.path.join(ASSETS, fn)) as f:
        return json.load(f)


def norm(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-9)


def pearson(x, y):
    return float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])


core = load("person2vec-minilm.json")
fp = load("person2vec-fingerprint-minilm.json")
authors = core["authors"]
centroids = {a["name"]: a["centroid"] for a in fp["authors"]}

# per-author text → lexical metrics.
texts = defaultdict(list)
for p in core["passages"]:
    if not p["is_mystery"]:
        texts[p["author_id"]].append(p["text"])


def lexical(txts):
    words = []
    for t in txts:
        words += re.findall(r"[A-Za-z']+", t.lower())
    if len(words) < 200:
        return None
    mwl = sum(len(w) for w in words) / len(words)          # mean word length
    cap = words[:2000]                                       # cap for a fair type-token ratio
    ttr = len(set(cap)) / len(cap)
    long_frac = sum(1 for w in words if len(w) >= 8) / len(words)  # fraction of long words
    return mwl, ttr, long_frac


# education axis: England-educated vs rest, from author centroids (difference of means).
def axis_for(field, value):
    pos = [centroids[a["name"]] for a in authors if a.get(field) == value and a["name"] in centroids]
    neg = [centroids[a["name"]] for a in authors if a.get(field) != value and a["name"] in centroids]
    if len(pos) < 2 or len(neg) < 2:
        return None
    return norm(np.mean(pos, 0) - np.mean(neg, 0))

ax_edu = axis_for("educated", "England")

proj, mwl, ttr, longf = [], [], [], []
for a in authors:
    if a["name"] not in centroids or a["id"] not in texts:
        continue
    lx = lexical(texts[a["id"]])
    if lx is None or ax_edu is None:
        continue
    proj.append(float(np.dot(norm(centroids[a["name"]]), ax_edu)))
    mwl.append(lx[0]); ttr.append(lx[1]); longf.append(lx[2])

r_mwl = pearson(proj, mwl)
r_ttr = pearson(proj, ttr)
r_long = pearson(proj, longf)
print(f"n authors = {len(proj)}")
print(f"education-axis projection  vs  mean word length : r = {r_mwl:+.3f}")
print(f"education-axis projection  vs  type-token ratio : r = {r_ttr:+.3f}")
print(f"education-axis projection  vs  long-word frac   : r = {r_long:+.3f}")

# also: is lexical sophistication itself linearly encoded? correlate the two lexical metrics with
# the FIRST PC-free direction that best separates high/low mean-word-length (sanity: are longer-word
# authors distinguishable?). Report the plain metric spread.
out = {
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "n_authors": len(proj),
    "r_education_axis_vs_mean_word_length": round(r_mwl, 3),
    "r_education_axis_vs_type_token_ratio": round(r_ttr, 3),
    "r_education_axis_vs_long_word_frac": round(r_long, 3),
    "mean_word_length_range": [round(min(mwl), 2), round(max(mwl), 2)],
}
with open(os.path.join(ASSETS, "person2vec-expertise-minilm.json"), "w") as f:
    json.dump(out, f, indent=2)
print("wrote person2vec-expertise-minilm.json")
