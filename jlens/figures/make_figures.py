#!/usr/bin/env python3
"""Publication figures for jlens/PAPER.md, rendered from the shipped bundles.

Aggregate figures (1-4) read jlens/paper/ledger.json so prose and figures cannot
drift; per-example detail (Fig 5) reads the shipped jlens bundle directly. Palette
is the site's, validated CVD-safe (Authors #5b4be0 / Coders #0d9488, ΔE 70.7).

Run:  python3 jlens/figures/make_figures.py
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
ASSETS = os.path.join(ROOT, "..", "web", "assets")
LEDGER = os.path.join(ROOT, "paper", "ledger.json")
OUT = HERE

# ---- palette (site brand, validated) ----
AUTHORS, CODERS = "#5b4be0", "#0d9488"
INK, MUTED, GRID = "#2c2c33", "#6b6b7b", "#e7e7ef"
POS, NEG = "#1a7a3c", "#b0344b"
PURPLE_SEQ = LinearSegmentedColormap.from_list(
    "brandpurp", ["#f4f2fd", "#b9aef4", "#5b4be0", "#2a1f6b"])

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#33333f", "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "text.color": INK, "xtick.color": "#545454", "ytick.color": "#545454",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 9,
})

with open(LEDGER) as f:
    LED = {(e["dataset"], e["metric"]): e["value"] for e in json.load(f)}


def led(ds, metric):
    return LED[(ds, metric)]


def load_bundle(name):
    with open(os.path.join(ASSETS, name)) as f:
        return json.load(f)


DS = [("minilm", "Authors (prose · MiniLM 6L)", AUTHORS),
      ("coders", "Coders (code · JinaBERT 12L)", CODERS)]


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(p, ROOT)}")


# ---- Fig 1: identity accuracy vs layer ----
def fig1_identity():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, (ds, title, color) in zip(axes, DS):
        pl = [v * 100 for v in led(ds, "identity_per_layer")]
        chance = led(ds, "identity_chance") * 100
        ceil = led(ds, "identity_output_acc") * 100
        x = list(range(len(pl)))
        ax.axhline(ceil, ls="--", lw=1.3, color=MUTED)
        ax.axhline(chance, ls=":", lw=1.3, color=MUTED)
        ax.plot(x, pl, "-o", color=color, lw=2, ms=5)
        ax.text(x[-1], ceil, f" output ceiling {ceil:.1f}%", va="bottom",
                ha="right", color=MUTED, fontsize=8)
        ax.text(0, chance, f" chance {chance:.1f}%", va="bottom", ha="left",
                color=MUTED, fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("layer")
        ax.set_xticks(x)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("nearest-centroid identity accuracy (%)")
    fig.suptitle("Identity is decodable from the internal Jacobian at every layer",
                 fontsize=12, fontweight="bold", y=1.02)
    save(fig, "fig1_identity.png")


# ---- Fig 2: structural signatures vs normalized depth ----
def fig2_structural():
    metrics = [("stable_rank", "stable rank"),
               ("effective_dim", "effective dim"),
               ("verbalizability", "verbalizability (kurtosis)")]
    # autocorrelation (paper's 4th signature) — included once any dataset has it;
    # datasets still lacking it (mid-regen) are simply skipped on that panel.
    if any((ds, "structural::autocorrelation") in LED for ds, _, _ in DS):
        metrics.append(("autocorrelation", "autocorrelation"))
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics), 3.2))
    for ax, (mk, mlabel) in zip(axes, metrics):
        for ds, title, color in DS:
            key = (ds, f"structural::{mk}")
            if key not in LED:
                continue
            vals = LED[key]
            n = len(vals)
            xs = [i / (n - 1) for i in range(n)]
            ax.plot(xs, vals, "-o", color=color, lw=2, ms=4,
                    label=title.split(" (")[0])
        ax.set_title(mlabel)
        ax.set_xlabel("normalized depth")
    axes[0].set_ylabel("value")
    axes[-1].legend(loc="upper left")
    fig.suptitle("Structural depth signatures of the averaged Jacobian",
                 fontsize=12, fontweight="bold", y=1.02)
    save(fig, "fig2_structural.png")


# ---- Fig 3: layer x layer CKA ----
def fig3_cka():
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7))
    for ax, (ds, title, _) in zip(axes, DS):
        b = load_bundle(f"person2vec-jlens-{ds}.json")
        cka = b["structural"]["cka"]
        im = ax.imshow(cka, cmap=PURPLE_SEQ, vmin=0.5, vmax=1.0, aspect="equal")
        ax.set_title(title.split(" (")[0] + f" · {len(cka)}L")
        ax.set_xlabel("layer")
        ax.set_ylabel("layer")
        ax.set_xticks(range(len(cka)))
        ax.set_yticks(range(len(cka)))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="linear CKA")
    fig.suptitle("Layer-to-layer readout geometry (CKA)",
                 fontsize=12, fontweight="bold", y=1.03)
    save(fig, "fig3_cka.png")


# ---- Fig 4: fingerprint KNOWN vs blank-space ----
def fig4_fingerprint():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for ax, (ds, title, _) in zip(axes, DS):
        thr = led(ds, "fingerprint_threshold")
        b = load_bundle(f"person2vec-fingerprint-{ds}.json")
        cents = [a["centroid"] for a in b["authors"]]
        known_x, known_y, ood_x, ood_y = [], [], [], []
        import random
        random.seed(0)
        for p in b["probes"]:
            cos = max(sum(a * c for a, c in zip(p["vec"], cen)) for cen in cents)
            if cos >= thr:
                known_x.append(0 + random.uniform(-0.06, 0.06)); known_y.append(cos)
            else:
                ood_x.append(1 + random.uniform(-0.06, 0.06)); ood_y.append(cos)
        ax.axhline(thr, ls="--", lw=1.3, color=INK)
        ax.text(-0.42, thr + 0.015, f"bar {thr:.2f}", va="bottom", ha="left",
                color=INK, fontsize=8)
        ax.scatter(known_x, known_y, s=70, color=AUTHORS, zorder=3,
                   label="KNOWN fingerprint", edgecolor="white", linewidth=0.8)
        ax.scatter(ood_x, ood_y, s=70, color=NEG, zorder=3, marker="D",
                   label="blank space (OOD)", edgecolor="white", linewidth=0.8)
        ax.set_title(title.split(" (")[0])
        ax.set_xticks([0, 1]); ax.set_xticklabels(["known", "stranger"])
        ax.set_xlim(-0.5, 1.5); ax.set_ylim(0, 1)
        ax.set_ylabel("nearest-centroid cosine")
    axes[0].legend(loc="upper right")
    fig.suptitle("Is the fingerprint in the weights? Known identities vs. blank space",
                 fontsize=12, fontweight="bold", y=1.02)
    save(fig, "fig4_fingerprint.png")


# ---- Fig 5: style-axis trajectories through depth (one example) ----
def fig5_style():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, (ds, title, _) in zip(axes, DS):
        b = load_bundle(f"person2vec-jlens-{ds}.json")
        axes_names = b["axes"]
        ex = b["examples"][0]
        traj = ex["style"]  # [{axis, per_layer}]
        # pick up to 3 axes with largest final-layer |score|
        ranked = sorted(traj, key=lambda t: -abs(t["per_layer"][-1]))[:3]
        pair = [AUTHORS, CODERS, MUTED]
        ax.axhline(0, lw=1, color=MUTED, alpha=0.6)
        for t, col in zip(ranked, pair):
            nm = axes_names[t["axis"]]
            x = list(range(len(t["per_layer"])))
            ax.plot(x, t["per_layer"], "-o", color=col, lw=2, ms=4,
                    label=nm[:26])
        ax.set_title(f"{title.split(' (')[0]} · '{ex['author']}'")
        ax.set_xlabel("layer")
        ax.legend(loc="best")
    axes[0].set_ylabel("style-axis loading  A·norm(J·h)")
    fig.suptitle("Style-lens: axis loadings through depth (one passage)",
                 fontsize=12, fontweight="bold", y=1.02)
    save(fig, "fig5_style.png")


# ---- Fig 6: identity ignition ----
def fig6_ignition(ds="minilm", dslabel="Authors (prose)"):
    path = os.path.join(ASSETS, f"person2vec-ignition-{ds}.json")
    if not os.path.exists(path):
        print(f"  (skip fig6: {os.path.basename(path)} not present yet)")
        return
    b = json.load(open(path))
    alphas, depths = b["alphas"], b["depths"]
    commit = b["example"]["commitment"]  # [depth][alpha]
    nd = len(depths)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5))

    # A: commitment curves colored by depth (graded shallow → snap deep)
    ax = axes[0]
    for d in range(nd):
        col = PURPLE_SEQ(0.12 + 0.82 * d / (nd - 1))
        ax.plot(alphas, commit[d], "-", color=col, lw=1.9)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title(f"Commitment vs α by depth\n{b['example']['a']} ↔ {b['example']['b']}")
    ax.set_xlabel("α   (input blend B → A)")
    ax.set_ylabel("commitment  (− = B, + = A)")
    sm = plt.cm.ScalarMappable(cmap=PURPLE_SEQ, norm=plt.Normalize(0, nd - 1))
    fig.colorbar(sm, ax=ax, label="depth", fraction=0.046, pad=0.04)

    # B: separation vs depth, real vs two nulls
    ax = axes[1]
    m = np.array(b["separation_mean"]); sd = np.array(b["separation_std"])
    ax.fill_between(depths, m - sd, m + sd, color=AUTHORS, alpha=0.15)
    ax.plot(depths, m, "-o", color=AUTHORS, lw=2, label="identity axis")
    ax.plot(depths, b["separation_null_shuffled_mean"], "--s", color=MUTED, lw=1.5, label="shuffled-label null")
    ax.plot(depths, b["separation_null_random_mean"], ":^", color=NEG, lw=1.5, label="random-direction null")
    ax.set_title("Endpoint separation vs depth")
    ax.set_xlabel("depth"); ax.set_ylabel("A–vs–B separation")
    ax.legend(fontsize=8)

    # C: ignition index vs depth (graded → all-or-none)
    ax = axes[2]
    ax.plot(depths, b["ignition_index_mean"], "-o", color=CODERS, lw=2)
    ax.set_ylim(0, 1)
    ax.set_title("Ignition index vs depth")
    ax.set_xlabel("depth"); ax.set_ylabel("sharpness  (0 graded → 1 snap)")

    fig.suptitle("Identity ignition: commitment to one author sharpens with depth",
                 fontsize=12, fontweight="bold", y=1.03)
    save(fig, "fig6_ignition.png")


# ---- Fig 7: decoder structural signatures (GPT-2) ----
def fig7_decoder():
    path = os.path.join(ASSETS, "person2vec-decoder-structural-gpt2.json")
    if not os.path.exists(path):
        print("  (skip fig7: decoder bundle not present yet)")
        return
    b = json.load(open(path))
    DEC = "#c2410c"  # decoder = deep orange, distinct from encoder purple/teal
    nl = b["layers"]
    xJ, xN = list(range(nl)), list(range(nl + 1))
    panels = [
        ("next-token acc\n(logit lens)", xN, [v * 100 for v in b["next_token_acc"]], "%"),
        ("stable rank", xJ, b["stable_rank"], ""),
        ("effective dim", xJ, b["effective_dim"], ""),
        ("verbalizability", xJ, b["verbalizability"], ""),
        ("autocorrelation", xJ, b["autocorrelation"], ""),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(15.5, 3.1))
    for ax, (label, xs, ys, unit) in zip(axes, panels):
        ax.plot(xs, ys, "-o", color=DEC, lw=2, ms=4)
        ax.set_title(label)
        ax.set_xlabel("depth")
        if unit == "%":
            ax.set_ylabel("%")
    fig.suptitle(f"Decoder structural signatures — GPT-2 ({nl} layers, {b['prompts']} prompts)",
                 fontsize=12, fontweight="bold", y=1.05)
    save(fig, "fig7_decoder.png")


if __name__ == "__main__":
    print("rendering figures ->", os.path.relpath(OUT, ROOT))
    fig1_identity()
    fig2_structural()
    fig3_cka()
    fig4_fingerprint()
    fig5_style()
    fig6_ignition()
    fig7_decoder()
    print("done.")
