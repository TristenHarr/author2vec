#!/usr/bin/env python3
"""Publication figures for jlens/PAPER.md, rendered from the shipped bundles.

Aggregate figures (1-4) read jlens/paper/ledger.json so prose and figures cannot
drift; per-example detail (Fig 5) reads the shipped jlens bundle directly. Palette
is the site's, validated CVD-safe (Authors #5b4be0 / Coders #0d9488, ΔE 70.7).

Run:  python3 jlens/figures/make_figures.py
"""
import json
import os
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
    # autocorrelation added by T1.4 once shipped
    if ("minilm", "structural::autocorrelation") in LED:
        metrics.append(("autocorrelation", "autocorrelation"))
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics), 3.2))
    for ax, (mk, mlabel) in zip(axes, metrics):
        for ds, title, color in DS:
            vals = led(ds, f"structural::{mk}")
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


if __name__ == "__main__":
    print("rendering figures ->", os.path.relpath(OUT, ROOT))
    fig1_identity()
    fig2_structural()
    fig3_cka()
    fig4_fingerprint()
    fig5_style()
    print("done.")
