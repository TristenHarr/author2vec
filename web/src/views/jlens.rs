//! Tab: the J-lens viewer. Displays the precomputed averaged-Jacobian interpretability
//! bundle for the small model (all-MiniLM-L6-v2) — the depth-wise structural signatures,
//! the (layer × position) vocab-lens token grid, and per-axis style trajectories.
//! All numbers are computed offline by the `jlens` crate; the browser only displays them.

use dioxus::prelude::*;

use super::Swatch;
use shared::{FingerprintBundle, IdentityBundle, JlensBundle, JlensExample, JlensJspace};

#[component]
pub fn Jlens() -> Element {
    // Reload the J-lens bundle for whichever dataset (authors / coders) is selected.
    let sel = crate::data::use_selector();
    let mut state = use_signal(|| None::<Result<JlensBundle, String>>);
    use_effect(move || {
        let key = sel.selected.read().clone();
        let mut state = state;
        spawn(async move {
            state.set(None);
            state.set(Some(crate::data::load_jlens(&key).await));
        });
    });
    // "Does it learn who wrote it?" — identity decoded from the Jacobian, by layer.
    let mut idstate = use_signal(|| None::<Option<IdentityBundle>>);
    use_effect(move || {
        let key = sel.selected.read().clone();
        let mut idstate = idstate;
        spawn(async move {
            idstate.set(Some(crate::data::load_identity(&key).await.ok()));
        });
    });
    let mut fpstate = use_signal(|| None::<Option<FingerprintBundle>>);
    use_effect(move || {
        let key = sel.selected.read().clone();
        let mut fpstate = fpstate;
        spawn(async move {
            fpstate.set(Some(crate::data::load_fingerprint(&key).await.ok()));
        });
    });
    let selected = use_signal(|| 0usize);

    let b = match state() {
        None => {
            return rsx! {
                div { class: "status",
                    div { class: "spinner" }
                    p { "Loading the J-lens…" }
                }
            }
        }
        Some(Err(e)) => {
            return rsx! {
                div { class: "status error",
                    p { "Couldn't load the J-lens bundle." }
                    pre { "{e}" }
                    p { class: "hint",
                        "Generate it with "
                        code { "cargo run -p jlens --bin jlens --release" }
                        "."
                    }
                }
            }
        }
        Some(Ok(b)) => b,
    };

    let m = &b.model;
    let sel = selected().min(b.examples.len().saturating_sub(1));

    rsx! {
        div { class: "jlens-view",
            p { class: "explainer",
                "A "
                strong { "J-lens" }
                " replaces everything after a layer with a single averaged Jacobian "
                span { class: "mono", "J_ℓ = E[∂e/∂h_ℓ]" }
                " — the linear map from an internal activation to the model's output, averaged over "
                "{m.passages} passages. Reading it out two ways — through the tied word embeddings "
                "(a token \u{201c}logit lens\u{201d}) and onto author2vec's own style axes — shows what each "
                "internal direction \u{201c}verbalizes\u{201d}. This is the paper's method, adapted to our "
                "{m.layers}-layer encoder."
            }
            p { class: "jlens-model",
                span { class: "mono", "{m.name}" }
                " · {m.layers} layers · d={m.d_model} · vocab {m.vocab} · readout: {m.w_u_source}"
            }

            if let Some(Some(fp)) = fpstate() {
                if !fp.probes.is_empty() {
                    FingerprintPanel { fp: fp.clone() }
                }
            }

            if let Some(Some(idb)) = idstate() {
                if !idb.per_layer.is_empty() {
                    IdentityPanel { idb: idb.clone() }
                }
            }

            DepthPanel { structural: b.structural.clone(), layers: b.layers.clone() }

            div { class: "jlens-picker",
                span { class: "mp-label", "Passage:" }
                for (i, ex) in b.examples.iter().enumerate() {
                    {
                        let mut selected = selected;
                        let active = i == sel;
                        rsx! {
                            button {
                                key: "{ex.passage_idx}",
                                class: if active { "chip active" } else { "chip" },
                                onclick: move |_| selected.set(i),
                                style: "border-left:5px solid {ex.color}",
                                "{ex.author}"
                            }
                        }
                    }
                }
            }

            if let Some(ex) = b.examples.get(sel) {
                div { class: "jlens-example",
                    p { class: "matrix-cap", "\u{201c}{ex.snippet}\u{201d}" }
                    h3 { "What each layer × position verbalizes (top tokens)" }
                    p { class: "matrix-cap",
                        "Rows are layers (0 = first, {m.layers}−1 = last); columns are token positions. "
                        "Each cell lists the top vocab-lens tokens for that activation. Hover for more."
                    }
                    TokenGrid { example: ex.clone(), layers: b.layers.clone() }

                    h3 { "Style trajectories through depth" }
                    p { class: "matrix-cap",
                        "How strongly each internal layer's Jacobian projects this passage onto a "
                        "recovered style axis. Positive = toward the first named class."
                    }
                    StylePanel { example: ex.clone(), axes: b.axes.clone(), layers: b.layers.clone() }

                    if !ex.concepts.is_empty() {
                        h3 { "Concept rank through depth" }
                        p { class: "matrix-cap",
                            "The strongest final-layer concepts, traced back through every layer — where each "
                            "\u{201c}surfaces.\u{201d} Lower on the axis = ranked higher (more strongly verbalized)."
                        }
                        ConceptChart { example: ex.clone(), layers: b.layers.clone() }
                    }

                    if !ex.jspace.items.is_empty() {
                        h3 { "J-space — the verbalizable decomposition" }
                        p { class: "matrix-cap",
                            "The layer-{ex.jspace.layer} activation decomposed into a sparse set of J-lens token "
                            "directions (non-negative matching pursuit). The bar is the share of the activation's "
                            "variance these verbalizable concepts capture."
                        }
                        JspacePanel { jspace: ex.jspace.clone() }
                    }
                }
            }

            h3 { "Layer-to-layer geometry (CKA)" }
            p { class: "matrix-cap",
                "Linear CKA between the J-lens readout geometry of each pair of layers. Bright = the two "
                "layers verbalize activations the same way."
            }
            CkaHeatmap { cka: b.structural.cka.clone() }
        }
    }
}

/// Interactive fingerprint-presence detector: pick a sample, the browser finds the
/// nearest identity live and calls "known fingerprint" vs "blank space".
#[component]
fn FingerprintPanel(fp: FingerprintBundle) -> Element {
    let mut pick = use_signal(|| 0usize);
    let i = pick().min(fp.probes.len().saturating_sub(1));
    let probe = &fp.probes[i];
    let (aidx, cos) = shared::nearest_centroid(&probe.vec, &fp.authors);
    let known = cos >= fp.threshold;
    let author = &fp.authors[aidx];
    let thr = fp.threshold;
    let subject = fp.subject.clone();
    rsx! {
        div { class: "fp-panel",
            h3 { "Is the fingerprint in the weights?" }
            p { class: "matrix-cap",
                "Pick a writing sample. Each sample's vector is precomputed, but the browser finds its "
                "nearest {subject} fingerprint on the spot. Clear the bar ⇒ the model KNOWS this {subject}; "
                "nothing close ⇒ blank space, no fingerprint."
            }
            div { class: "fp-chips",
                for (k, p) in fp.probes.iter().enumerate() {
                    {
                        let active = k == i;
                        rsx! {
                            button { key: "{k}", class: if active { "chip active" } else { "chip" },
                                onclick: move |_| pick.set(k), "{p.label}" }
                        }
                    }
                }
            }
            div { class: "fp-result",
                p { class: "fp-snippet", "\u{201c}{probe.snippet}\u{201d}" }
                div { class: if known { "fp-verdict known" } else { "fp-verdict blank" },
                    span { class: "fp-badge", { if known { "✓ KNOWN FINGERPRINT" } else { "· BLANK SPACE" } } }
                    span { class: "fp-cos", "cosine {cos:.2}  ·  bar {thr:.2}" }
                }
                if known {
                    div { class: "fp-nearest",
                        Swatch { color: author.color.clone() }
                        span { "nearest: " strong { "{author.name}" } }
                    }
                    if !author.tokens.is_empty() {
                        p { class: "matrix-cap", "Tokens this {subject}'s style leans toward (embedding lens):" }
                        div { class: "jspace-chips",
                            for (t, tok) in author.tokens.iter().enumerate() {
                                span { key: "{t}", class: "jspace-chip mono", "{tok}" }
                            }
                        }
                    }
                } else {
                    p { class: "fp-nearest",
                        "Closest is {author.name} at cosine {cos:.2} — below the bar, so the model holds no "
                        "fingerprint for this. It is, to the model, a stranger."
                    }
                }
            }
        }
    }
}

/// "Does it learn who wrote it?" — nearest-identity accuracy decoded from the internal
/// Jacobian readout at each layer, vs the output embedding (ceiling) and chance (floor).
#[component]
fn IdentityPanel(idb: IdentityBundle) -> Element {
    let (w, h) = (560.0_f64, 220.0_f64);
    let (ml, mr, mt, mb) = (44.0_f64, 160.0_f64, 16.0_f64, 28.0_f64);
    let (pw, ph) = (w - ml - mr, h - mt - mb);
    let n = idb.per_layer.len().max(1);
    let hi = idb.per_layer.iter().cloned().fold(idb.output_acc, f32::max).max(0.02) * 1.15;
    let px = |i: usize| ml + if n > 1 { i as f64 / (n - 1) as f64 } else { 0.5 } * pw;
    let py = |v: f32| mt + (1.0 - (v / hi) as f64) * ph;
    let pts: String = idb.per_layer.iter().enumerate().map(|(i, &v)| format!("{:.1},{:.1}", px(i), py(v))).collect::<Vec<_>>().join(" ");
    let x_right = ml + pw;
    let (chance_y, out_y) = (py(idb.chance), py(idb.output_acc));
    let (chance_pct, out_pct) = (idb.chance * 100.0, idb.output_acc * 100.0);
    let best = idb.per_layer.iter().cloned().fold(0f32, f32::max) * 100.0;
    rsx! {
        div { class: "identity-panel",
            h3 { "Does it learn who wrote it?" }
            p { class: "matrix-cap",
                "Nearest-{idb.subject} accuracy decoded purely from the model's INTERNAL Jacobian readout at each "
                "layer, across {idb.identities} {idb.subject}s held out one at a time. Above chance ⇒ the model "
                "computes {idb.subject} identity inside its layers — the fingerprint is in the Jacobians, not only "
                "the output. (best layer {best:.0}%, output ceiling {out_pct:.0}%, chance {chance_pct:.0}%)"
            }
            svg { width: "{w}", height: "{h}", view_box: "0 0 {w} {h}", role: "img",
                "aria-label": "Identity accuracy decoded from the Jacobian by layer",
                line { x1: "{ml}", y1: "{chance_y}", x2: "{x_right}", y2: "{chance_y}", stroke: "#8a8a99", stroke_width: "1.5", stroke_dasharray: "5 4" }
                text { x: "{x_right + 8.0}", y: "{chance_y + 3.0}", class: "gref", fill: "#595454", "chance {chance_pct:.0}%" }
                line { x1: "{ml}", y1: "{out_y}", x2: "{x_right}", y2: "{out_y}", stroke: "#0d5b52", stroke_width: "1.5", stroke_dasharray: "5 4" }
                text { x: "{x_right + 8.0}", y: "{out_y + 3.0}", class: "gref", fill: "#0d5b52", "output {out_pct:.0}%" }
                polyline { points: "{pts}", fill: "none", stroke: "#14b8a6", stroke_width: "2.5" }
                for (i, v) in idb.per_layer.iter().enumerate() {
                    circle { key: "d{i}", cx: "{px(i)}", cy: "{py(*v)}", r: "3.5", fill: "#14b8a6" }
                    text { key: "x{i}", x: "{px(i)}", y: "{h - 8.0}", class: "matrix-label", text_anchor: "middle", "{i}" }
                }
            }
        }
    }
}

/// Small multiples: the three depth-wise structural signatures, each normalized to its
/// own range so the *shape* across layers is comparable.
#[component]
fn DepthPanel(structural: shared::JlensStructural, layers: Vec<usize>) -> Element {
    let series = [
        ("stable rank", &structural.stable_rank, "#14b8a6"),
        ("effective dim", &structural.effective_dim, "#0e7490"),
        ("verbalizability", &structural.verbalizability, "#ca8a04"),
    ];
    rsx! {
        div { class: "depth-panel",
            h3 { "Structural signatures across depth" }
            p { class: "matrix-cap",
                "Read off the averaged Jacobians. Where stable rank dips, that layer routes the output "
                "through a low-rank \u{201c}workspace\u{201d} bottleneck. The 6-layer prose model compresses "
                "at the very start; the deeper 12-layer code model compresses in the middle — the "
                "mid-network workspace the paper finds in deep models, which only appears once there is "
                "depth to spare."
            }
            div { class: "depth-grid",
                for (name, vals, color) in series {
                    DepthChart { name: name.to_string(), vals: vals.clone(), color: color.to_string(), layers: layers.clone() }
                }
            }
        }
    }
}

#[component]
fn DepthChart(name: String, vals: Vec<f32>, color: String, layers: Vec<usize>) -> Element {
    let (w, h) = (240.0_f64, 140.0_f64);
    let (ml, mr, mt, mb) = (30.0_f64, 12.0_f64, 24.0_f64, 24.0_f64);
    let (pw, ph) = (w - ml - mr, h - mt - mb);
    let n = vals.len().max(1);
    let lo = vals.iter().cloned().fold(f32::MAX, f32::min);
    let hi = vals.iter().cloned().fold(f32::MIN, f32::max);
    let span = (hi - lo).max(1e-6);
    let px = |i: usize| ml + if n > 1 { i as f64 / (n - 1) as f64 } else { 0.5 } * pw;
    let py = |v: f32| mt + (1.0 - ((v - lo) / span) as f64) * ph;
    let pts: String = vals
        .iter()
        .enumerate()
        .map(|(i, &v)| format!("{:.1},{:.1}", px(i), py(v)))
        .collect::<Vec<_>>()
        .join(" ");
    rsx! {
        figure { class: "depth-chart",
            figcaption { "{name}" }
            svg {
                width: "{w}", height: "{h}", view_box: "0 0 {w} {h}",
                role: "img",
                "aria-label": "{name} by layer: {vals:?}",
                polyline { points: "{pts}", fill: "none", stroke: "{color}", stroke_width: "2.5" }
                for (i, v) in vals.iter().enumerate() {
                    circle { key: "{i}", cx: "{px(i)}", cy: "{py(*v)}", r: "3", fill: "{color}" }
                }
                for l in layers.iter() {
                    text { key: "x{l}", x: "{px(*l)}", y: "{h - 8.0}", class: "matrix-label", text_anchor: "middle", "{l}" }
                }
                text { x: "{ml}", y: "{mt - 8.0}", class: "matrix-label", "{hi:.0}" }
                text { x: "{ml}", y: "{h - mb + 2.0}", class: "matrix-label", "{lo:.0}" }
            }
        }
    }
}

#[component]
fn TokenGrid(example: JlensExample, layers: Vec<usize>) -> Element {
    let mut positions: Vec<usize> = example.cells.iter().map(|c| c.pos).collect();
    positions.sort_unstable();
    positions.dedup();
    // input token label per column (from the first matching cell)
    let col_token = |p: usize| {
        example
            .cells
            .iter()
            .find(|c| c.pos == p)
            .map(|c| c.token.clone())
            .unwrap_or_default()
    };
    rsx! {
        div { class: "token-grid-wrap",
            table { class: "token-grid",
                thead {
                    tr {
                        th { scope: "col", "layer \\ pos" }
                        for p in positions.iter() {
                            th { key: "h{p}", scope: "col",
                                div { class: "tg-pos", "{p}" }
                                div { class: "tg-tok mono", "{col_token(*p)}" }
                            }
                        }
                    }
                }
                tbody {
                    for l in layers.iter().rev() {
                        tr { key: "r{l}",
                            th { scope: "row", "{l}" }
                            for p in positions.iter() {
                                {
                                    let cell = example.cells.iter().find(|c| c.layer == *l && c.pos == *p);
                                    match cell {
                                        Some(c) => {
                                            let full: String = c.top.iter().map(|t| t.tok.clone()).collect::<Vec<_>>().join(" · ");
                                            let head: Vec<String> = c.top.iter().take(3).map(|t| t.tok.clone()).collect();
                                            rsx! {
                                                td { key: "c{l}-{p}", title: "{full}",
                                                    for (k, t) in head.iter().enumerate() {
                                                        span { key: "{k}", class: if k == 0 { "tg-top mono" } else { "tg-alt mono" }, "{t}" }
                                                    }
                                                }
                                            }
                                        }
                                        None => rsx! { td { key: "c{l}-{p}", "" } },
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[component]
fn StylePanel(example: JlensExample, axes: Vec<String>, layers: Vec<usize>) -> Element {
    let palette = ["#0d9488", "#0891b2", "#ca8a04", "#7c3aed", "#db2777"];
    let (w, h) = (560.0_f64, 240.0_f64);
    let (ml, mr, mt, mb) = (40.0_f64, 190.0_f64, 16.0_f64, 30.0_f64);
    let (pw, ph) = (w - ml - mr, h - mt - mb);
    let n = layers.len().max(1);
    // symmetric y-range around 0
    let maxabs = example
        .style
        .iter()
        .flat_map(|s| s.per_layer.iter())
        .fold(0f32, |a, &v| a.max(v.abs()))
        .max(1e-3);
    let px = |i: usize| ml + if n > 1 { i as f64 / (n - 1) as f64 } else { 0.5 } * pw;
    let py = |v: f32| mt + (0.5 - (v / (2.0 * maxabs)) as f64) * ph;
    let zero = py(0.0);
    let x_right = ml + pw;
    rsx! {
        div { class: "style-panel",
            svg {
                width: "{w}", height: "{h}", view_box: "0 0 {w} {h}",
                role: "img",
                "aria-label": "Style-axis loadings by layer for {example.author}",
                line { x1: "{ml}", y1: "{zero}", x2: "{x_right}", y2: "{zero}", stroke: "#b0b0bd", stroke_width: "1", stroke_dasharray: "4 4" }
                for (ai, tr) in example.style.iter().enumerate() {
                    {
                        let color = palette[ai % palette.len()];
                        let pts: String = tr.per_layer.iter().enumerate()
                            .map(|(i, &v)| format!("{:.1},{:.1}", px(i), py(v)))
                            .collect::<Vec<_>>().join(" ");
                        let ly = mt + 6.0 + ai as f64 * 18.0;
                        let name = axes.get(tr.axis).cloned().unwrap_or_default();
                        rsx! {
                            polyline { key: "p{ai}", points: "{pts}", fill: "none", stroke: "{color}", stroke_width: "2.5" }
                            rect { key: "sw{ai}", x: "{x_right + 12.0}", y: "{ly - 9.0}", width: "10", height: "10", fill: "{color}" }
                            text { key: "lg{ai}", x: "{x_right + 28.0}", y: "{ly}", class: "gaxis", "{name}" }
                        }
                    }
                }
                for l in layers.iter() {
                    text { key: "sx{l}", x: "{px(*l)}", y: "{h - 10.0}", class: "matrix-label", text_anchor: "middle", "{l}" }
                }
                text { x: "{ml - 6.0}", y: "{mt + 6.0}", class: "matrix-label", text_anchor: "end", "+{maxabs:.2}" }
                text { x: "{ml - 6.0}", y: "{h - mb}", class: "matrix-label", text_anchor: "end", "−{maxabs:.2}" }
            }
        }
    }
}

/// Rank-vs-depth trajectory of each top concept (the paper's signature figure).
/// Log y-scale, rank 1 at the top (most strongly verbalized).
#[component]
fn ConceptChart(example: JlensExample, layers: Vec<usize>) -> Element {
    let palette = ["#0d9488", "#0891b2", "#ca8a04", "#7c3aed", "#db2777"];
    let (w, h) = (560.0_f64, 240.0_f64);
    let (ml, mr, mt, mb) = (40.0_f64, 170.0_f64, 16.0_f64, 30.0_f64);
    let (pw, ph) = (w - ml - mr, h - mt - mb);
    let n = layers.len().max(1);
    let cap = example
        .concepts
        .iter()
        .flat_map(|c| c.per_layer_rank.iter())
        .copied()
        .max()
        .unwrap_or(1)
        .max(2);
    let denom = (cap as f64).ln().max(1e-6);
    let px = |i: usize| ml + if n > 1 { i as f64 / (n - 1) as f64 } else { 0.5 } * pw;
    let py = |r: u32| mt + ((r.max(1) as f64).ln() / denom) * ph;
    rsx! {
        div { class: "style-panel",
            svg {
                width: "{w}", height: "{h}", view_box: "0 0 {w} {h}",
                role: "img",
                "aria-label": "Rank of top concepts by layer for {example.author}",
                for (ci, c) in example.concepts.iter().enumerate() {
                    {
                        let color = palette[ci % palette.len()];
                        let pts: String = c.per_layer_rank.iter().enumerate()
                            .map(|(i, &r)| format!("{:.1},{:.1}", px(i), py(r)))
                            .collect::<Vec<_>>().join(" ");
                        let ly = mt + 6.0 + ci as f64 * 18.0;
                        rsx! {
                            polyline { key: "cp{ci}", points: "{pts}", fill: "none", stroke: "{color}", stroke_width: "2.5" }
                            rect { key: "cs{ci}", x: "{ml + pw + 12.0}", y: "{ly - 9.0}", width: "10", height: "10", fill: "{color}" }
                            text { key: "cl{ci}", x: "{ml + pw + 28.0}", y: "{ly}", class: "gaxis mono", "{c.tok}" }
                        }
                    }
                }
                text { x: "{ml - 6.0}", y: "{mt + 6.0}", class: "matrix-label", text_anchor: "end", "rank 1" }
                text { x: "{ml - 6.0}", y: "{h - mb}", class: "matrix-label", text_anchor: "end", "{cap}" }
                for l in layers.iter() {
                    text { key: "cx{l}", x: "{px(*l)}", y: "{h - 10.0}", class: "matrix-label", text_anchor: "middle", "{l}" }
                }
            }
        }
    }
}

/// J-space panel: the captured-variance meter + the verbalizable concept chips.
#[component]
fn JspacePanel(jspace: JlensJspace) -> Element {
    let pct = (jspace.captured_variance * 100.0).clamp(0.0, 100.0);
    rsx! {
        div { class: "jspace-panel",
            div { class: "jspace-meter",
                div { class: "jspace-bar", style: "width:{pct}%" }
            }
            div { class: "jspace-pct", "{pct:.1}% of the activation's variance captured by {jspace.items.len()} concepts" }
            div { class: "jspace-chips",
                for (i, it) in jspace.items.iter().enumerate() {
                    span { key: "{i}", class: "jspace-chip mono", title: "coefficient {it.score:.3}", "{it.tok}" }
                }
            }
        }
    }
}

#[component]
fn CkaHeatmap(cka: Vec<Vec<f32>>) -> Element {
    let n = cka.len();
    let cell = 40.0_f64;
    let pad = 30.0_f64;
    let size = pad + n as f64 * cell + 6.0;
    rsx! {
        div { class: "matrix-wrap",
            svg { class: "matrix", width: "{size}", height: "{size}", view_box: "0 0 {size} {size}",
                for i in 0..n {
                    text { key: "cl{i}", x: "{pad + i as f64 * cell + cell/2.0}", y: "{pad - 8.0}", class: "matrix-label", text_anchor: "middle", "{i}" }
                    text { key: "rl{i}", x: "{pad - 8.0}", y: "{pad + i as f64 * cell + cell/2.0 + 3.0}", class: "matrix-label", text_anchor: "end", "{i}" }
                }
                for i in 0..n {
                    for j in 0..n {
                        {
                            let v = cka[i][j].clamp(0.0, 1.0);
                            let x = pad + j as f64 * cell;
                            let y = pad + i as f64 * cell;
                            rsx! {
                                g { key: "{i}-{j}",
                                    title { "CKA(layer {i}, layer {j}) = {v:.2}" }
                                    rect { x: "{x}", y: "{y}", width: "{cell}", height: "{cell}", fill: "#2dd4bf", fill_opacity: "{v}", stroke: "#e7e7ef", stroke_width: "1" }
                                    text { x: "{x + cell/2.0}", y: "{y + cell/2.0 + 3.0}", class: "matrix-count", text_anchor: "middle", fill: "#0d5b52", "{v:.2}" }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
