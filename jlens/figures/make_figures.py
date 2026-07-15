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
OUT = os.environ.get("FIG_OUT", HERE)  # override to render into a temp dir (freshness gate)

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
        lo = [v * 100 for v in led(ds, "ci_identity_per_layer_lo")]
        hi = [v * 100 for v in led(ds, "ci_identity_per_layer_hi")]
        chance = led(ds, "identity_chance") * 100
        ceil = led(ds, "identity_output_acc") * 100
        x = list(range(len(pl)))
        ax.axhline(ceil, ls="--", lw=1.3, color=MUTED)
        ax.axhline(chance, ls=":", lw=1.3, color=MUTED)
        ax.fill_between(x, lo, hi, color=color, alpha=0.16, lw=0)  # Wilson 95% CI (n=250)
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
    fig.suptitle("Identity is decodable from the internal Jacobian at every layer\n"
                 "(shaded = Wilson 95% CI, n=250 decode passages)",
                 fontsize=12, fontweight="bold", y=1.06)
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
            se_key = (ds, f"structural::{mk}_se")
            if se_key in LED:  # jackknife 95% CI band on the two spectral signatures
                se = LED[se_key]
                lo = [v - 1.96 * s for v, s in zip(vals, se)]
                hi = [v + 1.96 * s for v, s in zip(vals, se)]
                ax.fill_between(xs, lo, hi, color=color, alpha=0.16, lw=0)
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
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7))
    for i, (ax, (ds, title, _)) in enumerate(zip(axes, DS)):
        b = load_bundle(f"person2vec-jlens-{ds}.json")
        cka = b["structural"]["cka"]
        im = ax.imshow(cka, cmap=PURPLE_SEQ, vmin=0.5, vmax=1.0, aspect="equal")
        ax.set_title(title.split(" (")[0] + f" · {len(cka)}L")
        ax.set_xlabel("layer")
        ax.set_ylabel("layer")
        ax.set_xticks(range(len(cka)))
        ax.set_yticks(range(len(cka)))
        # only the rightmost colorbar carries the label, so it can't collide with the
        # neighbouring panel's y-axis "layer" label
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="linear CKA" if i == len(DS) - 1 else "")
    fig.subplots_adjust(wspace=0.35)
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
        import random
        random.seed(0)
        # position/colour by GROUND TRUTH (probe provenance), not by the threshold decision,
        # so any misclassification (a known below the bar, or an OOD above it) is visible.
        known_y, ood_y = [], []
        for p in b["probes"]:
            cos = max(sum(a * c for a, c in zip(p["vec"], cen)) for cen in cents)
            (known_y if "held-out" in p["label"] else ood_y).append(cos)
        ax.axhline(thr, ls="--", lw=1.3, color=INK)
        ax.text(-0.42, thr + 0.015, f"decision bar {thr:.2f}", va="bottom", ha="left",
                color=INK, fontsize=8)
        ax.scatter([random.uniform(-0.06, 0.06) for _ in known_y], known_y, s=70, color=AUTHORS,
                   zorder=3, label=f"known author (held-out, n={len(known_y)})", edgecolor="white", linewidth=0.8)
        ax.scatter([1 + random.uniform(-0.06, 0.06) for _ in ood_y], ood_y, s=70, color=NEG,
                   zorder=3, marker="D", label=f"stranger (OOD text, n={len(ood_y)})", edgecolor="white", linewidth=0.8)
        ax.set_title(title.split(" (")[0])
        ax.set_xticks([0, 1]); ax.set_xticklabels(["known\nauthor", "stranger\n(OOD)"])
        ax.set_xlim(-0.5, 1.5); ax.set_ylim(0, 1)
        ax.set_ylabel("nearest-centroid cosine")
    axes[0].legend(loc="center right", fontsize=8)
    fig.suptitle("Is the fingerprint in the weights? Known identities vs. blank space\n"
                 "(x = ground-truth provenance; a point on the wrong side of the bar is an error)",
                 fontsize=11.5, fontweight="bold", y=1.06)
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
            # clean, full (non-truncated) legend labels
            # keep the axis values verbatim (England != UK: the axis excludes Scotland/Wales);
            # only trim the "↔ rest" suffix and tidy the separator for the legend.
            nm = (nm.replace("United States", "US")
                    .replace(" ↔ rest", "").replace("?", "").replace("=", ": "))
            x = list(range(len(t["per_layer"])))
            ax.plot(x, t["per_layer"], "-o", color=col, lw=2, ms=4, label=nm)
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
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.5))
    fig.subplots_adjust(wspace=0.42)

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

    # B: separation vs depth, real vs two nulls. Band = ±SEM (std/√n_pairs) — the right
    # quantity for "the MEAN separation exceeds the null"; the raw pair-to-pair SD is far larger.
    ax = axes[1]
    m = np.array(b["separation_mean"]); sd = np.array(b["separation_std"])
    sem = sd / np.sqrt(b["n_pairs"])
    ax.fill_between(depths, m - sem, m + sem, color=AUTHORS, alpha=0.18)
    ax.plot(depths, m, "-o", color=AUTHORS, lw=2, label="identity axis (±SEM)")
    ax.plot(depths, b["separation_null_shuffled_mean"], "--s", color=MUTED, lw=1.5, label="shuffled-label null")
    ax.plot(depths, b["separation_null_random_mean"], ":^", color=NEG, lw=1.5, label="random-direction null")
    ax.set_title(f"Endpoint separation vs depth (n={b['n_pairs']} pairs)")
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
        ("next-token acc\n(logit lens)", xN, [v * 100 for v in b["next_token_acc"]], "%", None),
        ("stable rank", xJ, b["stable_rank"], "", b.get("stable_rank_se")),
        ("effective dim", xJ, b["effective_dim"], "", b.get("effective_dim_se")),
        ("verbalizability", xJ, b["verbalizability"], "", None),
        ("autocorrelation", xJ, b["autocorrelation"], "", None),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(15.5, 3.1))
    for ax, (label, xs, ys, unit, se) in zip(axes, panels):
        if se:  # jackknife 95% CI band
            lo = [v - 1.96 * s for v, s in zip(ys, se)]
            hi = [v + 1.96 * s for v, s in zip(ys, se)]
            ax.fill_between(xs, lo, hi, color=DEC, alpha=0.16, lw=0)
        ax.plot(xs, ys, "-o", color=DEC, lw=2, ms=4)
        ax.set_title(label)
        ax.set_xlabel("depth")
        if unit == "%":
            ax.set_ylabel("%")
    fig.suptitle(f"Decoder structural signatures — GPT-2 ({nl} layers, {b['prompts']} prompts)",
                 fontsize=12, fontweight="bold", y=1.05)
    save(fig, "fig7_decoder.png")


# ---- Fig 8: per-coder recognizability (15 coders) ----
def fig8_coders_recognizability():
    b = load_bundle("person2vec-coders.json")
    res = b.get("results", {})
    if "per_author" not in res:
        print("  (skip fig8: coders results not present)")
        return
    names = [b["authors"][pa["author_id"]]["name"] for pa in res["per_author"]]
    accs = [pa["accuracy"] * 100 for pa in res["per_author"]]
    order = sorted(range(len(accs)), key=lambda i: accs[i])
    names = [names[i] for i in order]
    accs = [accs[i] for i in order]
    chance = 100.0 / len(b["authors"])
    cis = {r["name"]: (r["lo"] * 100, r["hi"] * 100)
           for r in load_bundle("person2vec-cis.json").get("coders_recognizability", [])}
    colors = [CODERS if ("Kelley" in n or "Sumner" in n) else "#b9b9d0" for n in names]
    xerr = [[a - cis[n][0] for n, a in zip(names, accs)],
            [cis[n][1] - a for n, a in zip(names, accs)]]
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    y = list(range(len(names)))
    ax.barh(y, accs, color=colors, height=0.72, zorder=3,
            xerr=xerr, error_kw=dict(ecolor="#5c5c6b", elinewidth=1.0, capsize=2.5, zorder=4))
    ax.axvline(chance, ls=":", color=NEG, lw=1.4, label=f"chance {chance:.1f}%")
    for yi, (n, a) in enumerate(zip(names, accs)):
        ax.text(cis[n][1] + 1.2, yi, f"{a:.0f}%", va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("same-file recognition accuracy (%)")
    ax.set_xlim(0, 104)
    ax.set_title("Some developers are far more recognizable than others\n"
                 "(bars = Wilson 95% CI, n=126 files; adjacent ranks overlap — only top vs. bottom is separable)",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "fig8_coders_recognizability.png")


def fig10_bigfive():
    """Measured-population validation: Big Five recovery from prose.

    Dumbbell (not zero-baseline bars) so the small-but-real lift is shown honestly:
    muted dot = majority baseline, purple dot = leave-one-out accuracy, x = shuffled null.
    """
    traits = ["Openness", "Neuroticism", "Conscientiousness", "Extraversion", "Agreeableness"]
    rows = []
    for t in traits:
        acc, maj, lift, null = led("minilm", f"bigfive::{t}")[:4]
        rows.append((t, acc * 100, maj * 100, null * 100, lift * 100))
    rows.sort(key=lambda r: r[4])  # ascending lift -> best on top
    names = [r[0] for r in rows]
    y = list(range(len(names)))
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    ax.axvline(50, ls=":", color=MUTED, lw=1.3, zorder=1, label="chance (50%)")
    for yi, (t, acc, maj, null, lift) in zip(y, rows):
        ax.plot([maj, acc], [yi, yi], color="#c9c4ef", lw=3.0, zorder=2, solid_capstyle="round")
        ax.scatter([null], [yi], marker="x", s=34, color=MUTED, lw=1.6, zorder=3)
        ax.scatter([maj], [yi], s=58, color="#b9b9d0", zorder=4, edgecolor="white", linewidth=1.0)
        ax.scatter([acc], [yi], s=78, color=AUTHORS, zorder=5, edgecolor="white", linewidth=1.0)
        ax.text(acc + 0.35, yi, f"+{lift:.1f}", va="center", ha="left", fontsize=8.5,
                color=AUTHORS, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9.5)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlim(47.5, 60.5)
    ax.set_xlabel("leave-one-out recovery accuracy (%)")
    ax.set_title("Personality is weakly-but-really recoverable from prose\n"
                 "Pennebaker essays (n=2,467), MiniLM — validated psychometric labels",
                 fontsize=11.5, fontweight="bold")
    # legend proxies
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=AUTHORS, markersize=9, label="recovered (LOO)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#b9b9d0", markersize=9, label="majority baseline"),
        Line2D([0], [0], marker="x", color=MUTED, markersize=8, lw=0, label="shuffled-label null"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5)
    save(fig, "fig10_bigfive.png")


def fig11_negative_control():
    """Construct validity: real demographic constructs recover; the negative control does not.

    One corpus (Blog Authorship), one pipeline. Metric = lift over the shuffled-label null
    (accuracy - null), comparable across attributes with different class counts. Gender/age are
    positive controls; zodiac is the negative control and should sit at ~0.
    """
    order = ["gender", "age", "zodiac"]
    labelmap = {"gender": "Gender (2-way)", "age": "Age band (3-way)", "zodiac": "Zodiac (12-way)"}
    rows = []
    for a in order:
        acc, maj, null, lift, p = led("minilm", f"blog::{a}")[:5]
        rows.append((labelmap[a], lift * 100, acc * 100, maj * 100, null * 100, p, a == "zodiac"))
    rows = rows[::-1]  # zodiac at bottom
    names = [r[0] for r in rows]
    lifts = [r[1] for r in rows]
    y = list(range(len(names)))
    # color: negative control grey; significant positive control purple; n.s. -> faded + hatched
    colors, hatches = [], []
    for r in rows:
        if r[6]:
            colors.append(MUTED); hatches.append("")
        elif r[5] <= 0.05:
            colors.append(AUTHORS); hatches.append("")
        else:
            colors.append("#c9c4ef"); hatches.append("////")
    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    ax.axvline(0, color="#33333f", lw=1.2, zorder=2)
    bars = ax.barh(y, lifts, color=colors, height=0.62, zorder=3)
    for bar, h in zip(bars, hatches):
        if h:
            bar.set_hatch(h); bar.set_edgecolor("white")
    for yi, (nm, lift, acc, maj, null, p, isneg) in zip(y, rows):
        psig = "p<.001" if p <= 0.001 else f"p={p:.2f}"
        xtext = (lift + 0.6) if lift >= 0 else 0.6
        ax.text(xtext, yi, f"{acc:.0f}%  (maj {maj:.0f}, null {null:.0f})  {psig}",
                va="center", ha="left", fontsize=8.3, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9.5)
    ax.set_ylim(-0.6, len(names) - 0.4)
    lo = min(lifts + [0]) - 2
    hi = max(lifts + [0]) + 24
    ax.set_xlim(lo, hi)
    ax.set_xlabel("recovery lift over shuffled-label null (percentage points)")
    ax.set_title("The method recovers real constructs, not noise\n"
                 "Age recovers (p<.001); the astrological negative control does not (p=.96)",
                 fontsize=11.5, fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=AUTHORS, label="real construct (significant)"),
                       Patch(facecolor="#c9c4ef", hatch="////", edgecolor="white", label="positive but n.s. (gender, p=.08)"),
                       Patch(color=MUTED, label="negative control")],
              loc="lower right", fontsize=8)
    save(fig, "fig11_negative_control.png")


def fig9_ai_fingerprint():
    """AI-model code fingerprint — house-styled, reproducible, with the permutation null + CI.

    Left: task effect (same task, different model) vs. model effect (same model, different task) —
    the task dominates. Right: task-controlled model ID with its Wilson 95% CI, against the
    within-task shuffle null and chance — a faint but real (p<0.001) fingerprint.
    """
    b = load_bundle("person2vec-aifp.json")
    te, me = led("ai", "aifp_task_effect") * 100, led("ai", "aifp_model_effect") * 100
    idacc = led("ai", "aifp_task_controlled_model_id") * 100
    null = led("ai", "aifp_task_controlled_null_mean") * 100
    chance = b["chance"] * 100
    lo, hi = [v * 100 for v in led("ai", "ci_aifp_model_id")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    ax = axes[0]
    ax.bar([0, 1], [te, me], color=[CODERS, "#b9b9d0"], width=0.62, zorder=3)
    for x, v in zip([0, 1], [te, me]):
        ax.text(x, v + 2, f"{v:.0f}", ha="center", fontsize=10, color=INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["same task,\ndiff. model", "same model,\ndiff. task"])
    ax.set_ylabel("mean cosine (×100)"); ax.set_ylim(0, 100)
    ax.set_title("The task, not the model,\ndrives the embedding")
    ax = axes[1]
    ax.bar([0], [idacc], color=CODERS, width=0.5, zorder=3,
           yerr=[[idacc - lo], [hi - idacc]], error_kw=dict(ecolor=INK, elinewidth=1.2, capsize=4))
    ax.axhline(chance, ls=":", color=MUTED, lw=1.4, label=f"chance {chance:.0f}%")
    ax.axhline(null, ls="--", color=NEG, lw=1.4, label=f"shuffle null {null:.0f}%")
    ax.text(0, hi + 2, f"{idacc:.0f}%", ha="center", fontsize=10, color=INK)
    ax.set_xticks([0]); ax.set_xticklabels(["task-controlled\nmodel ID (n=149)"])
    ax.set_xlim(-0.7, 1.0); ax.set_ylim(0, 62)
    ax.set_ylabel("accuracy (%)"); ax.legend(fontsize=8, loc="upper right")
    ax.set_title("A faint but real model fingerprint\n(Wilson 95% CI; p<0.001, 1000-perm)")
    fig.suptitle("Do the AI models have code fingerprints? Task dominates; a faint model signal survives",
                 fontsize=11.5, fontweight="bold", y=1.04)
    save(fig, "fig9_ai_fingerprint.png")


# ---- Fig 12: untrained-model control (random init vs trained) ----
def fig12_untrained():
    def g(ds, m):
        return LED.get((ds, m))
    if g("minilm", "structural::stable_rank_untrained") is None and \
       g("gpt2", "decoder::stable_rank_untrained") is None:
        print("  (skip fig12: untrained control not in ledger yet)")
        return
    UNTR = "#9a9aa8"          # random-init = neutral grey, behind the trained curve
    DEC = "#c2410c"           # GPT-2 decoder orange (matches Fig 7)
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.2))

    def overlay(ax, ds, trained_key, untr_key, color, ci=True, pct=False):
        tr = g(ds, trained_key)
        if tr is None:
            return
        scale = 100.0 if pct else 1.0
        x = list(range(len(tr)))
        if ci:
            lo, hi = g(ds, trained_key + "_ci_lo"), g(ds, trained_key + "_ci_hi")
            if lo and hi:
                ax.fill_between(x, [v * scale for v in lo], [v * scale for v in hi],
                                color=color, alpha=0.16, lw=0)
        un, sd = g(ds, untr_key), g(ds, untr_key + "_sd")
        if un is not None:
            xu = list(range(len(un)))
            if sd:
                ax.fill_between(xu, [(v - s) * scale for v, s in zip(un, sd)],
                                [(v + s) * scale for v, s in zip(un, sd)], color=UNTR, alpha=0.25, lw=0)
            ax.plot(xu, [v * scale for v in un], "--s", color=UNTR, lw=2, ms=4, label="random init")
        ax.plot(x, [v * scale for v in tr], "-o", color=color, lw=2, ms=4, label="trained")
        ax.set_xlabel("layer")

    overlay(axes[0], "minilm", "structural::stable_rank", "structural::stable_rank_untrained", AUTHORS)
    axes[0].set_title("MiniLM · stable rank")
    axes[0].set_ylabel("value")
    overlay(axes[1], "minilm", "structural::autocorrelation", "structural::autocorrelation_untrained", AUTHORS, ci=False)
    axes[1].set_title("MiniLM · autocorrelation")
    overlay(axes[2], "gpt2", "decoder::stable_rank", "decoder::stable_rank_untrained", DEC)
    axes[2].set_title("GPT-2 · stable rank")
    overlay(axes[3], "gpt2", "decoder::next_token_acc", "decoder::next_token_acc_untrained", DEC, ci=False, pct=True)
    axes[3].set_title("GPT-2 · next-token acc")
    axes[3].set_ylabel("%")
    axes[0].legend(loc="upper left")
    fig.suptitle("Untrained-model control — which depth structure is training-induced vs. architectural\n"
                 "(dashed = random init; shaded = trained jackknife 95% CI / random-init across-seed ±1 SD)",
                 fontsize=12, fontweight="bold", y=1.10)
    save(fig, "fig12_untrained.png")


if __name__ == "__main__":
    print("rendering figures ->", os.path.relpath(OUT, ROOT))
    fig12_untrained()
    fig9_ai_fingerprint()
    fig11_negative_control()
    fig10_bigfive()
    fig8_coders_recognizability()
    fig1_identity()
    fig2_structural()
    fig3_cka()
    fig4_fingerprint()
    fig5_style()
    fig6_ignition()
    fig7_decoder()
    print("done.")
