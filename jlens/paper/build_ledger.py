#!/usr/bin/env python3
"""Audit ledger builder.

Machine-extracts every citable number in the paper straight from the shipped
`web/assets/person2vec-*.json` bundles into `jlens/paper/ledger.json`. The paper
may only cite values that appear here; figures plot the same values. This is the
single source of truth for the "no fabricated numbers" guarantee.

Run:  python3 jlens/paper/build_ledger.py
"""
import json
import os
import math

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.normpath(os.path.join(HERE, "..", "..", "web", "assets"))

# (dataset key, human label, core-bundle filename)
DATASETS = [
    ("minilm", "Authors (prose)", "person2vec-minilm.json"),
    ("coders", "Coders (code)", "person2vec-coders.json"),
]

ledger = []


def add(dataset, metric, value, asset, path, note=""):
    ledger.append({
        "dataset": dataset,
        "metric": metric,
        "value": value,
        "asset": asset,
        "path": path,
        "note": note,
    })


def load(fn):
    p = os.path.join(ASSETS, fn)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def r3(x):
    return round(x, 3) if isinstance(x, float) else x


def cosine_nn(vec, centroids):
    """Vectors are L2-normalized => cosine == dot. Return (max_cos, argmax)."""
    best, bi = -2.0, -1
    for i, c in enumerate(centroids):
        s = sum(a * b for a, b in zip(vec, c))
        if s > best:
            best, bi = s, i
    return best, bi


for key, label, core_fn in DATASETS:
    # ---- core bundle: authorship study numbers ----
    core = load(core_fn)
    if core:
        n_auth = len(core["authors"])
        n_pass = len(core["passages"])
        res = core["results"]
        add(key, "n_identities", n_auth, core_fn, "authors[]", label)
        add(key, "n_passages", n_pass, core_fn, "passages[]", label)
        add(key, "chance", r3(1.0 / n_auth), core_fn, "1/len(authors)",
            "blind-guess baseline")
        add(key, "dim", core["dim"], core_fn, "dim", "embedding dimension")
        for rung in res["rungs"]:
            add(key, f"rung::{rung['label']}", r3(rung["accuracy"]), core_fn,
                "results.rungs[].accuracy", rung.get("sublabel", ""))
        add(key, "headline_correct", res["headline_correct"], core_fn,
            "results.headline_correct", "same-unit reveal")
        add(key, "headline_total", res["headline_total"], core_fn,
            "results.headline_total", "")
        add(key, "reel_hits_seen", res["reel_hits_seen"], core_fn,
            "results.reel_hits_seen", "predict new passage, author read")
        add(key, "reel_hits_unseen", res["reel_hits_unseen"], core_fn,
            "results.reel_hits_unseen", "predict new passage, author hidden")
        add(key, "reel_total", res["reel_total"], core_fn, "results.reel_total", "")
        add(key, "trained_sim", r3(res["trained_sim"]), core_fn,
            "results.trained_sim", "cosine to own centroid")
        add(key, "untrained_sim", r3(res["untrained_sim"]), core_fn,
            "results.untrained_sim", "cosine to global centroid")
        accs = [a["accuracy"] for a in res["per_author"]]
        add(key, "per_author_acc_min", r3(min(accs)), core_fn,
            "min results.per_author[].accuracy", "")
        add(key, "per_author_acc_max", r3(max(accs)), core_fn,
            "max results.per_author[].accuracy", "")
        add(key, "per_author_acc_mean", r3(sum(accs) / len(accs)), core_fn,
            "mean results.per_author[].accuracy", "")
        for attr in res["attributes"]:
            nm = attr["name"]
            add(key, f"attr::{nm}::fair", r3(attr["fair_accuracy"]), core_fn,
                "results.attributes[].fair_accuracy", "leave-one-author-out")
            add(key, f"attr::{nm}::leaky", r3(attr["leaky_accuracy"]), core_fn,
                "results.attributes[].leaky_accuracy", "identity-leaking")
            add(key, f"attr::{nm}::majority", r3(attr["baseline"]), core_fn,
                "results.attributes[].baseline", "majority-class")
            add(key, f"attr::{nm}::random", r3(attr["random_baseline"]), core_fn,
                "results.attributes[].random_baseline", "")

    # ---- jlens bundle: method + structural signatures ----
    jl = load(f"person2vec-jlens-{key}.json")
    if jl:
        fn = f"person2vec-jlens-{key}.json"
        m = jl["model"]
        for mk in ("name", "layers", "d_model", "vocab", "passages", "w_u_source"):
            if mk in m:
                add(key, f"jlens_model::{mk}", m[mk], fn, f"model.{mk}", "")
        add(key, "jlens_axes", jl["axes"], fn, "axes", "style-lens axes")
        st = jl["structural"]
        for sk in ("stable_rank", "effective_dim", "verbalizability", "autocorrelation"):
            if sk in st:
                add(key, f"structural::{sk}", [r3(v) for v in st[sk]], fn,
                    f"structural.{sk}", "per layer")
        if "cka" in st:
            cka = st["cka"]
            off = [cka[i][j] for i in range(len(cka)) for j in range(len(cka)) if i != j]
            add(key, "cka_offdiag_mean", r3(sum(off) / len(off)), fn,
                "mean off-diag structural.cka", "")
            add(key, "cka_offdiag_min", r3(min(off)), fn, "min off-diag structural.cka", "")

    # ---- identity bundle: identity-across-depth ----
    idb = load(f"person2vec-identity-{key}.json")
    if idb:
        fn = f"person2vec-identity-{key}.json"
        add(key, "identity_chance", r3(idb["chance"]), fn, "chance", "")
        add(key, "identity_output_acc", r3(idb["output_acc"]), fn, "output_acc",
            "output-embedding ceiling")
        pl = idb["per_layer"]
        add(key, "identity_per_layer", [r3(v) for v in pl], fn, "per_layer", "")
        add(key, "identity_best_layer_acc", r3(max(pl)), fn, "max per_layer",
            f"argmax layer {pl.index(max(pl))}")
        add(key, "identity_best_layer_idx", pl.index(max(pl)), fn, "argmax per_layer", "")
        probe = idb.get("per_layer_probe") or []
        if probe:
            add(key, "identity_per_layer_probe", [r3(v) for v in probe], fn, "per_layer_probe",
                "baseline: plain linear probe on mean-pooled activation (no Jacobian)")
            add(key, "identity_probe_best_acc", r3(max(probe)), fn, "max per_layer_probe", "")
            add(key, "identity_jlens_mean", r3(sum(pl) / len(pl)), fn, "mean per_layer", "")
            add(key, "identity_probe_mean", r3(sum(probe) / len(probe)), fn, "mean per_layer_probe", "")

    # ---- fingerprint bundle: recompute nearest-centroid cosines ----
    fp = load(f"person2vec-fingerprint-{key}.json")
    if fp:
        fn = f"person2vec-fingerprint-{key}.json"
        thr = fp["threshold"]
        centroids = [a["centroid"] for a in fp["authors"]]
        names = [a["name"] for a in fp["authors"]]
        add(key, "fingerprint_threshold", r3(thr), fn, "threshold", "")
        add(key, "fingerprint_n_identities", len(fp["authors"]), fn, "len(authors)", "")
        known_cos, ood_cos = [], []
        for p in fp["probes"]:
            cos, bi = cosine_nn(p["vec"], centroids)
            is_known = cos >= thr
            (known_cos if is_known else ood_cos).append(cos)
            add(key, f"probe::{p['label']}", r3(cos), fn,
                "recomputed max cos(probe.vec, authors[].centroid)",
                f"nearest={names[bi]}; {'KNOWN' if is_known else 'BLANK-SPACE'} (thr={r3(thr)})")
        if known_cos:
            add(key, "fingerprint_known_cos_range",
                [r3(min(known_cos)), r3(max(known_cos))], fn,
                "recomputed", "KNOWN probes cosine range")
        if ood_cos:
            add(key, "fingerprint_ood_cos_range",
                [r3(min(ood_cos)), r3(max(ood_cos))], fn,
                "recomputed", "blank-space probes cosine range")

    # ---- ignition bundle: identity commitment across depth ----
    ig = load(f"person2vec-ignition-{key}.json")
    if ig:
        fn = f"person2vec-ignition-{key}.json"
        add(key, "ignition_n_pairs", ig["n_pairs"], fn, "n_pairs", "")
        add(key, "ignition_separation_mean", [r3(v) for v in ig["separation_mean"]], fn,
            "separation_mean", "A-vs-B separation, identity axis, per depth")
        add(key, "ignition_separation_null_random", [r3(v) for v in ig["separation_null_random_mean"]],
            fn, "separation_null_random_mean", "random-direction null")
        add(key, "ignition_separation_null_shuffled", [r3(v) for v in ig["separation_null_shuffled_mean"]],
            fn, "separation_null_shuffled_mean", "shuffled-label null")
        add(key, "ignition_index_mean", [r3(v) for v in ig["ignition_index_mean"]], fn,
            "ignition_index_mean", "transition sharpness per depth (0 graded → 1 snap)")
        add(key, "ignition_depth", ig["ignition_depth"], fn, "ignition_depth", "")
        sm = ig["separation_mean"]
        add(key, "ignition_separation_peak", r3(max(sm)), fn, "max separation_mean",
            f"peak at depth {sm.index(max(sm))}")

    # ---- steering bundle: causal identity lever ----
    st = load(f"person2vec-steer-{key}.json")
    if st:
        fn = f"person2vec-steer-{key}.json"
        add(key, "steer_layer", st["steer_layer"], fn, "steer_layer", "")
        add(key, "steer_alphas", st["alphas"], fn, "alphas", "")
        real_swings = [ax["loading"][-1] - ax["loading"][0] for ax in st["axes"]]
        rand_drifts = [abs(ax["loading_random"][-1] - ax["loading_random"][0]) for ax in st["axes"]]
        add(key, "steer_n_axes", len(st["axes"]), fn, "len(axes)", "")
        add(key, "steer_real_swing_range", [r3(min(real_swings)), r3(max(real_swings))], fn,
            "loading[-1]-loading[0] per axis", "real α-sweep swing (α:−6→+6)")
        add(key, "steer_random_drift_max", r3(max(rand_drifts)), fn,
            "max |loading_random[-1]-loading_random[0]|", "matched-norm control drift")

# ---- decoder structural (GPT-2) — single bundle, not per-dataset ----
dec = load("person2vec-decoder-structural-gpt2.json")
if dec:
    fn = "person2vec-decoder-structural-gpt2.json"
    add("gpt2", "decoder_prompts", dec["prompts"], fn, "prompts", "")
    add("gpt2", "decoder_layers", dec["layers"], fn, "layers", "")
    for k in ("stable_rank", "effective_dim", "verbalizability", "autocorrelation", "next_token_acc"):
        add("gpt2", f"decoder::{k}", [r3(v) for v in dec[k]], fn, k, "per depth")
    sr, ed = dec["stable_rank"], dec["effective_dim"]
    add("gpt2", "decoder_stable_rank_peak", r3(max(sr)), fn, "max stable_rank",
        f"workspace peak at layer {sr.index(max(sr))}; ends {r3(sr[-1])}")
    add("gpt2", "decoder_effdim_peak", r3(max(ed)), fn, "max effective_dim",
        f"peak at layer {ed.index(max(ed))}; ends {r3(ed[-1])}")
    add("gpt2", "decoder_nexttok_final", r3(dec["next_token_acc"][-1]), fn,
        "next_token_acc[-1]", "final-depth logit-lens next-token accuracy")

# ---- decoder directed-modulation (GPT-2) ----
sti = load("person2vec-decoder-steer-gpt2.json")
if sti:
    fn = "person2vec-decoder-steer-gpt2.json"
    add("gpt2", "decoder_steer_alpha", sti["alpha"], fn, "alpha", "")
    add("gpt2", "decoder_steer_mean_real", r3(sti["mean_delta_real"]), fn,
        "mean_delta_real", "mean Δlog-prob(target), real concept direction")
    add("gpt2", "decoder_steer_mean_random", r3(sti["mean_delta_random"]), fn,
        "mean_delta_random", "matched-norm random control")
    for c in sti["concepts"]:
        add("gpt2", f"decoder_steer::{c['name']}",
            [r3(c["steer_pos"] - c["base"]), r3(c["steer_neg"] - c["base"]), r3(c["random"] - c["base"])],
            fn, "[+δ Δ, −δ Δ, random Δ]", c["name"])

# ---- expertise / lexical-sophistication probe (encoder-side) ----
exp = load("person2vec-expertise-minilm.json")
if exp:
    fn = "person2vec-expertise-minilm.json"
    add("minilm", "expertise_n_authors", exp["n_authors"], fn, "n_authors", "")
    add("minilm", "expertise_r_word_length", exp["r_education_axis_vs_mean_word_length"], fn,
        "r_education_axis_vs_mean_word_length", "education axis vs lexical sophistication (weak)")
    add("minilm", "expertise_r_ttr", exp["r_education_axis_vs_type_token_ratio"], fn,
        "r_education_axis_vs_type_token_ratio", "")
    add("minilm", "expertise_r_long_word", exp["r_education_axis_vs_long_word_frac"], fn,
        "r_education_axis_vs_long_word_frac", "")

# ---- Big Five recovery from prose (measured-population validation) ----
bf = load("person2vec-bigfive.json")
if bf:
    fn = "person2vec-bigfive.json"
    add("minilm", "bigfive_n_essays", bf["n_essays"], fn, "n_essays",
        "Pennebaker & King labelled essays")
    for tname, t in bf["traits"].items():
        add("minilm", f"bigfive::{tname}",
            [t["loo_acc"], t["majority"], t["lift"], t["shuffled_null"], t.get("p_value")], fn,
            "traits[].[loo_acc, majority, lift, shuffled_null, p_value]",
            f"leave-one-out nearest-centroid vs majority; {bf.get('n_perm', 0)}-perm null")
    add("minilm", "bigfive_best_trait", bf["best_trait"], fn, "best_trait", "")
    add("minilm", "bigfive_mean_lift", bf["mean_lift"], fn, "mean_lift",
        "mean lift over majority across 5 traits")

# ---- Blog corpus demographics: gender/age positive controls + zodiac NEGATIVE control ----
bd = load("person2vec-blog-demographics.json")
if bd:
    fn = "person2vec-blog-demographics.json"
    add("minilm", "blog_n_authors", bd["n_authors"], fn, "n_authors",
        "Blog Authorship Corpus, author-level")
    for aname, a in bd["attributes"].items():
        note = "NEGATIVE CONTROL (should be ~chance)" if aname == "zodiac" else "positive control"
        add("minilm", f"blog::{aname}",
            [a["loo_acc"], a["majority"], a["shuffled_null"], a["lift_over_null"], a.get("p_value")], fn,
            "attributes[].[loo_acc, majority, shuffled_null, lift_over_null, p_value]",
            f"{a['classes']}-way author-level LOO, 1000-perm p; {note}")

# ---- homogenization over time (coders) ----
hg = load("person2vec-homogenization-coders.json")
if hg:
    fn = "person2vec-homogenization-coders.json"
    add("coders", "homogenization_pre2021", hg["cross_sim_pre2021"], fn, "cross_sim_pre2021", "mean cross-coder sim, AI-free era")
    add("coders", "homogenization_post2022", hg["cross_sim_post2022"], fn, "cross_sim_post2022", "mean cross-coder sim, AI era")
    add("coders", "homogenization_delta", hg["convergence_delta"], fn, "convergence_delta", "+ = converged")
    add("coders", "homogenization_p_two_sided", hg["p_two_sided"], fn, "p_two_sided", "500-sample permutation null")
    add("coders", "homogenization_null_mean", r3(hg["null_mean"]), fn, "null_mean", "within-coder permutation null mean")
    add("coders", "homogenization_null_sd", r3(hg["null_sd"]), fn, "null_sd", "")
    add("coders", "homogenization_within_consistency", hg["within_coder_consistency"], fn, "within_coder_consistency", "")
    add("coders", "homogenization_n_coders", hg["n_coders_both_eras"], fn, "n_coders_both_eras", "")

# ---- per-coder recognizability specifics (the two added devs) ----
core_c = load("person2vec-coders.json")
if core_c and "results" in core_c and core_c["results"].get("per_author"):
    fn = "person2vec-coders.json"
    nm = [a["name"] for a in core_c["authors"]]
    accs = {nm[pa["author_id"]]: pa["accuracy"] for pa in core_c["results"]["per_author"]}
    add("coders", "recognizability_spread", [r3(min(accs.values())), r3(max(accs.values()))], fn,
        "min/max results.per_author[].accuracy", "some coders known better than others")
    for who in ("Andrew Kelley", "Jarred Sumner"):
        if who in accs:
            add("coders", f"recognizability::{who}", r3(accs[who]), fn, "results.per_author[].accuracy", "same-file")

# ---- AI-model fingerprint (do frontier models have distinct code styles?) ----
aifp = load("person2vec-aifp.json")
if aifp:
    fn = "person2vec-aifp.json"
    add("ai", "aifp_n_samples", aifp["n_samples"], fn, "n_samples", "valid embedded cells")
    add("ai", "aifp_n_grid", aifp.get("n_grid"), fn, "n_grid", "models x tasks grid")
    add("ai", "aifp_n_tasks", aifp["n_tasks"], fn, "n_tasks", "")
    add("ai", "aifp_n_tasks_canonical", aifp.get("n_tasks_canonical"), fn, "n_tasks_canonical", "")
    add("ai", "aifp_n_tasks_open_ended", aifp.get("n_tasks_open_ended"), fn, "n_tasks_open_ended", "")
    add("ai", "aifp_models", aifp["models"], fn, "models", "")
    add("ai", "aifp_task_effect", aifp["task_effect"], fn, "task_effect", "same-task, different-model cosine")
    add("ai", "aifp_model_effect", aifp["model_effect"], fn, "model_effect", "same-model, different-task cosine")
    add("ai", "aifp_task_controlled_model_id", aifp["task_controlled_model_id"], fn,
        "task_controlled_model_id", f"leave-one-task-out; chance {aifp['chance']}")
    add("ai", "aifp_task_controlled_null_mean", aifp.get("task_controlled_null_mean"), fn,
        "task_controlled_null_mean", "within-task model-label permutation null")
    add("ai", "aifp_task_controlled_p", aifp.get("task_controlled_p"), fn,
        "task_controlled_p", "1000-permutation p-value")
    add("ai", "aifp_verdict", aifp["verdict"], fn, "verdict", "")

# ---- confidence intervals (Wilson / SEM) so every cited bound is traceable ----
cis = load("person2vec-cis.json")
if cis:
    fn = "person2vec-cis.json"
    for r in cis.get("coders_recognizability", []):
        add("coders", f"ci_recognizability::{r['name']}", [r["lo"], r["hi"]], fn,
            "coders_recognizability[].[lo,hi]", f"Wilson 95% on {r['correct']}/{r['total']}")
    for ds, d in cis.get("identity", {}).items():
        best = max(d["per_layer"], key=lambda x: x["acc"])
        add(ds, "ci_identity_best_internal", [best["lo"], best["hi"]], fn,
            "identity best per_layer Wilson", f"n={d['n']}")
        add(ds, "ci_identity_output", [d["output"]["lo"], d["output"]["hi"]], fn,
            "identity output Wilson", f"n={d['n']}")
        add(ds, "ci_identity_per_layer_lo", [p["lo"] for p in d["per_layer"]], fn,
            "identity per_layer Wilson lo", "")
        add(ds, "ci_identity_per_layer_hi", [p["hi"] for p in d["per_layer"]], fn,
            "identity per_layer Wilson hi", "")
        if "best_probe" in d:
            add(ds, "ci_identity_best_probe", [d["best_probe"], d["best_jlens"]], fn,
                "[best probe acc, best J-lens acc]", "Jacobian-vs-probe ablation")
    for t, v in cis.get("bigfive", {}).items():
        add("minilm", f"ci_bigfive::{t}", [v["lo"], v["hi"]], fn,
            "bigfive trait Wilson", f"clears majority={v['clears_majority']}")
    for ds, d in cis.get("ignition", {}).items():
        add(ds, "ci_ignition_sem", d["separation_sem"], fn, "ignition separation SEM",
            f"std/sqrt(n_pairs={d['n_pairs']})")
    if "aifp" in cis:
        add("ai", "ci_aifp_model_id", [cis["aifp"]["lo"], cis["aifp"]["hi"]], fn,
            "aifp task_controlled_model_id Wilson", f"n={cis['aifp']['n']}")
    for a, v in cis.get("blog", {}).items():
        add("minilm", f"ci_blog::{a}", [v["lo"], v["hi"]], fn, "blog attr Wilson", f"n={v['n']}")

out = os.path.join(HERE, "ledger.json")
with open(out, "w") as f:
    json.dump(ledger, f, indent=2)

# human-readable summary to stdout
print(f"wrote {len(ledger)} ledger entries -> {os.path.relpath(out)}")
for key, label, _ in DATASETS:
    print(f"\n=== {label} [{key}] ===")
    for e in ledger:
        if e["dataset"] == key:
            print(f"  {e['metric']:38s} = {e['value']}")
