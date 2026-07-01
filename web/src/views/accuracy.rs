//! Tab: precomputed authorship results. All numbers are computed offline by the
//! corpus (the familiarity ladder, learning curve, per-author accuracy, confusion
//! matrix) and shipped in the dataset; this view only displays them.

use dioxus::prelude::*;

use super::{gate, Swatch};
use crate::data::{use_dataset, LoadState};

#[component]
pub fn Accuracy() -> Element {
    let state = use_dataset();
    if let Some(node) = gate(&state()) {
        return node;
    }
    let LoadState::Loaded(bundle) = state() else {
        unreachable!()
    };
    let r = &bundle.meta.results;
    let n = bundle.n_authors();
    let baseline_pct = r.baseline * 100.0;
    let book_out = r
        .rungs
        .iter()
        .find(|x| x.label == "Never read this book")
        .map(|x| x.accuracy)
        .unwrap_or(0.0);

    let mut per = r.per_author.clone();
    per.sort_by(|a, b| b.accuracy.total_cmp(&a.accuracy));

    rsx! {
        div { class: "accuracy-view",
            p { class: "explainer",
                "Can a computer name the author of a passage it was never shown? It depends "
                "entirely on how much of that author it has already read. With {n} authors, blind "
                "guessing scores about {baseline_pct:.1}%. Every number below is precomputed in "
                "Rust and just displayed here."
            }

            div { class: "ladder-panel",
                h3 { "The familiarity ladder" }
                p { class: "matrix-cap",
                    "The same held-out writing, judged at each level of exposure. The more of "
                    "someone the model has read, the better it knows them."
                }
                for rung in r.rungs.iter() {
                    {
                        let acc = rung.accuracy;
                        let hue = (acc * 150.0).clamp(0.0, 150.0);
                        let pct = acc * 100.0;
                        rsx! {
                            div { class: "ladder-row", key: "{rung.label}",
                                div { class: "ladder-label",
                                    div { class: "ll-main", "{rung.label}" }
                                    div { class: "ll-sub", "{rung.sublabel}" }
                                }
                                div { class: "bar-track ladder-track",
                                    div { class: "bar-fill", style: "width:{pct}%; background:hsl({hue}, 62%, 46%)" }
                                }
                                div { class: "ladder-val", "{pct:.1}%" }
                            }
                        }
                    }
                }
            }

            div { class: "gradient-panel",
                h3 { "It is a smooth gradient" }
                p { class: "matrix-cap",
                    "Give the model a book it has never seen, then vary how many passages of the "
                    "author's OTHER writing it has read. Recognition climbs with exposure."
                }
                FamiliarityGradient { curve: r.curve.clone(), unseen: book_out, baseline: r.baseline }
            }

            div { class: "acc-grid",
                div { class: "per-author",
                    h3 { "Which authors are easiest to identify?" }
                    p { class: "matrix-cap", "When the model has read the same book." }
                    for a in per.iter() {
                        {
                            let author = &bundle.meta.authors[a.author_id];
                            let acc = a.accuracy;
                            rsx! {
                                div { class: "bar-row", key: "{a.author_id}",
                                    Swatch { color: author.color.clone() }
                                    span { class: "bar-name", "{author.name}" }
                                    div { class: "bar-track",
                                        div { class: "bar-fill", style: "width:{acc * 100.0}%; background:{author.color}" }
                                    }
                                    span { class: "bar-val", "{acc * 100.0:.0}% ({a.correct}/{a.total})" }
                                }
                            }
                        }
                    }
                }
                div { class: "matrix-wrap",
                    h3 { "Confusion matrix" }
                    p { class: "matrix-cap", "Row = true author, column = the model's guess (same-book case)." }
                    ConfusionMatrix { confusion: r.confusion.clone() }
                }
            }

            div { class: "reel-panel",
                h3 { "The reveal: name the author of a brand-new passage" }
                p { class: "matrix-cap",
                    "Each row is a passage held out of training entirely. We guess its author two "
                    "ways: when the model HAS read that author's other work, and when their whole "
                    "body of work is hidden. When it is hidden the true author is not even an "
                    "option, so it guesses the nearest author it DOES know: still just vectors, "
                    "confidently landing next to your closest stylistic neighbor."
                }
                {
                    let sp = r.reel_hits_seen as f32 / r.reel_total.max(1) as f32 * 100.0;
                    let up = r.reel_hits_unseen as f32 / r.reel_total.max(1) as f32 * 100.0;
                    rsx! {
                        div { class: "reel-stats",
                            div { class: "reel-stat hit",
                                div { class: "rs-num", "{r.reel_hits_seen}/{r.reel_total}" }
                                div { class: "rs-lab", "correct when it HAS read the author ({sp:.0}%)" }
                            }
                            div { class: "reel-stat miss",
                                div { class: "rs-num", "{r.reel_hits_unseen}/{r.reel_total}" }
                                div { class: "rs-lab", "correct when it has NEVER read the author ({up:.0}%)" }
                            }
                        }
                    }
                }
                p { class: "reel-super",
                    "So what is “you” to a model that has never read you? Nothing in particular: it "
                    "falls back on the average of all {n} authors. That average already sits fairly "
                    "close to you (cosine {r.untrained_sim:.2}) because you are human and write in "
                    "English. But it is the last sliver of distinctiveness, your own coordinate at "
                    "{r.trained_sim:.2}, that lets it pick YOU out of the crowd. Let it read you, and "
                    "you go from the average of everyone to a point of your own. That is the superpower."
                }
                div { class: "reel-list",
                    for pred in r.reel.iter() {
                        {
                            let p = &bundle.meta.passages[pred.passage_idx];
                            let truth = &bundle.meta.authors[pred.true_author];
                            let seen = &bundle.meta.authors[pred.pred_seen];
                            let unseen = &bundle.meta.authors[pred.pred_unseen];
                            let seen_hit = pred.pred_seen == pred.true_author;
                            let unseen_hit = pred.pred_unseen == pred.true_author;
                            let seen_mark = if seen_hit { "✓" } else { "✗" };
                            let unseen_mark = if unseen_hit { "✓" } else { "✗" };
                            let seen_cls = if seen_hit { "reel-guess hit" } else { "reel-guess miss" };
                            let unseen_cls = if unseen_hit { "reel-guess hit" } else { "reel-guess miss" };
                            let text = snippet(&p.text, 130);
                            rsx! {
                                div { class: "reel-row", key: "{pred.passage_idx}",
                                    div { class: "reel-truth",
                                        Swatch { color: truth.color.clone() }
                                        strong { "{truth.name}" }
                                    }
                                    p { class: "reel-snippet", "“{text}”" }
                                    div { class: "{seen_cls}",
                                        span { class: "rg-cond", "has read" }
                                        span { "{seen_mark} {seen.name}" }
                                    }
                                    div { class: "{unseen_cls}",
                                        span { class: "rg-cond", "never read" }
                                        span { "{unseen_mark} {unseen.name}" }
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

/// Small SVG line chart: accuracy vs. how many passages/author the model has read,
/// with dashed reference lines for the "never read this book" and chance levels.
#[component]
fn FamiliarityGradient(curve: Vec<(usize, f32)>, unseen: f32, baseline: f32) -> Element {
    let w = 580.0_f64;
    let h = 232.0_f64;
    let ml = 44.0_f64;
    let mr = 152.0_f64;
    let mt = 16.0_f64;
    let mb = 42.0_f64;
    let pw = w - ml - mr;
    let ph = h - mt - mb;
    let m = curve.len().max(1);
    let px = |i: usize| ml + if m > 1 { i as f64 / (m - 1) as f64 } else { 0.5 } * pw;
    let py = |v: f64| mt + (1.0 - v) * ph;

    let pts: String = curve
        .iter()
        .enumerate()
        .map(|(i, &(_, acc))| format!("{:.1},{:.1}", px(i), py(acc as f64)))
        .collect::<Vec<_>>()
        .join(" ");
    let x_right = ml + pw;
    let by = py(baseline as f64);
    let uy = py(unseen as f64);

    rsx! {
        svg {
            class: "gradient-chart",
            width: "{w}", height: "{h}", view_box: "0 0 {w} {h}",
            for pct in [0, 25, 50, 75, 100] {
                {
                    let gy = py(pct as f64 / 100.0);
                    let lx = ml - 8.0;
                    let ty = gy + 3.0;
                    rsx! {
                        line { x1: "{ml}", y1: "{gy}", x2: "{x_right}", y2: "{gy}", stroke: "#eee", stroke_width: "1" }
                        text { x: "{lx}", y: "{ty}", class: "matrix-label", text_anchor: "end", "{pct}%" }
                    }
                }
            }
            line { x1: "{ml}", y1: "{by}", x2: "{x_right}", y2: "{by}", stroke: "#b0b0bd", stroke_width: "1.5", stroke_dasharray: "5 4" }
            text { x: "{x_right + 8.0}", y: "{by + 3.0}", class: "gref", fill: "#8a8a99", "chance {baseline * 100.0:.0}%" }
            line { x1: "{ml}", y1: "{uy}", x2: "{x_right}", y2: "{uy}", stroke: "#e0a0a0", stroke_width: "1.5", stroke_dasharray: "5 4" }
            text { x: "{x_right + 8.0}", y: "{uy + 3.0}", class: "gref", fill: "#c0392b", "unseen book {unseen * 100.0:.0}%" }
            polyline { points: "{pts}", fill: "none", stroke: "#5b4be0", stroke_width: "2.5" }
            for (i, (cap, acc)) in curve.iter().copied().enumerate() {
                {
                    let cx = px(i);
                    let cy = py(acc as f64);
                    let ly = h - mb + 16.0;
                    rsx! {
                        circle { cx: "{cx}", cy: "{cy}", r: "3.5", fill: "#5b4be0" }
                        text { x: "{cx}", y: "{ly}", class: "matrix-label", text_anchor: "middle", "{cap}" }
                    }
                }
            }
            {
                let axx = ml + pw / 2.0;
                let axy = h - 6.0;
                rsx! {
                    text { x: "{axx}", y: "{axy}", class: "gaxis", text_anchor: "middle", "passages per author the model has read" }
                }
            }
        }
    }
}

#[component]
fn ConfusionMatrix(confusion: Vec<Vec<u32>>) -> Element {
    let state = use_dataset();
    let LoadState::Loaded(bundle) = state() else {
        return rsx! {};
    };
    let n = bundle.n_authors();
    let cell = if n > 18 { 26.0_f64 } else { 30.0_f64 };
    let left = 82.0_f64;
    let top = 88.0_f64;
    let w = left + n as f64 * cell + 12.0;
    let h = top + n as f64 * cell + 12.0;

    rsx! {
        svg {
            class: "matrix",
            width: "{w}", height: "{h}", view_box: "0 0 {w} {h}",
            for c in 0..n {
                {
                    let name = short(&bundle.meta.authors[c].name);
                    let cx = left + c as f64 * cell + cell * 0.55;
                    let cy = top - 6.0;
                    rsx! {
                        text {
                            key: "col{c}",
                            x: "{cx}", y: "{cy}",
                            class: "matrix-label",
                            text_anchor: "start",
                            transform: "rotate(-45 {cx} {cy})",
                            "{name}"
                        }
                    }
                }
            }
            for row in 0..n {
                {
                    let row_total: u32 = confusion[row].iter().sum();
                    let author = &bundle.meta.authors[row];
                    let ly = top + row as f64 * cell + cell * 0.68;
                    rsx! {
                        text {
                            key: "row{row}",
                            x: "{left - 6.0}", y: "{ly}",
                            class: "matrix-label",
                            text_anchor: "end",
                            "{short(&author.name)}"
                        }
                        for c in 0..n {
                            {
                                let count = confusion[row][c];
                                let frac = if row_total > 0 { count as f64 / row_total as f64 } else { 0.0 };
                                let opacity = frac.powf(0.6);
                                let x = left + c as f64 * cell;
                                let y = top + row as f64 * cell;
                                rsx! {
                                    g { key: "{row}-{c}",
                                        title { "True {short(&bundle.meta.authors[row].name)} → Pred {short(&bundle.meta.authors[c].name)}: {count}" }
                                        rect {
                                            x: "{x}", y: "{y}", width: "{cell}", height: "{cell}",
                                            fill: "{author.color}", fill_opacity: "{opacity}",
                                            stroke: "#e7e7ef", stroke_width: "1",
                                        }
                                        if count > 0 {
                                            text {
                                                x: "{x + cell / 2.0}", y: "{y + cell / 2.0 + 3.0}",
                                                class: "matrix-count",
                                                text_anchor: "middle",
                                                fill: if row == c { "#fff" } else { "#444" },
                                                "{count}"
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
    }
}

fn short(name: &str) -> String {
    name.split_whitespace().last().unwrap_or(name).to_string()
}

fn snippet(s: &str, n: usize) -> String {
    let t: String = s.chars().take(n).collect();
    if s.chars().count() > n {
        format!("{t}…")
    } else {
        t
    }
}
