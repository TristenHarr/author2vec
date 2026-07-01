//! Tab: recovering author traits (gender, birth country) from writing style alone,
//! using leave-one-author-out so the model can't cheat by recognizing the author.

use dioxus::prelude::*;

use super::{gate, Swatch};
use crate::data::{use_dataset, LoadState};
use shared::AttrResult;

#[component]
pub fn Dimensions() -> Element {
    let state = use_dataset();
    if let Some(node) = gate(&state()) {
        return node;
    }
    let LoadState::Loaded(bundle) = state() else {
        unreachable!()
    };
    let attrs = bundle.meta.results.attributes.clone();

    rsx! {
        div { class: "dims-view",
            p { class: "explainer",
                "Trippy question: can writing style alone reveal an author's gender or birth "
                "country, for an author the model has NEVER read? We build each trait's profile "
                "from the OTHER authors and guess (leave-one-author-out), so it can't cheat by "
                "recognizing who wrote it. Compare it against blind guessing and against always "
                "guessing the biggest group."
            }
            if attrs.is_empty() {
                p { class: "matrix-cap", "No trait data in this dataset yet." }
            }
            for attr in attrs.iter() {
                AttrPanel { attr: attr.clone() }
            }
        }
    }
}

#[component]
fn AttrPanel(attr: AttrResult) -> Element {
    let beats_random = attr.fair_accuracy > attr.random_baseline;
    let beats_majority = attr.fair_accuracy > attr.baseline;
    let (verdict, vcolor) = if beats_majority {
        ("Style beats the majority guess: a real signal.", "#1a7a3c")
    } else if beats_random {
        ("Style beats random, but not the majority guess: barely a signal.", "#b7791f")
    } else {
        ("Style does worse than random guessing: no usable signal here.", "#b0344b")
    };
    let rows = [
        ("Blind guessing (random)", attr.random_baseline, "#9a9aa8"),
        ("Always guess the majority", attr.baseline, "#c0392b"),
        ("Never read the author", attr.fair_accuracy, "#5b4be0"),
        ("Has read the author (leaks identity)", attr.leaky_accuracy, "#2a9d5c"),
    ];

    rsx! {
        div { class: "attr-panel",
            h3 { "{attr.name}" }
            p { class: "matrix-cap",
                "Guessed from writing style, for an author never seen. Fair test covers "
                "{attr.tested_authors}/{attr.total_authors} authors."
            }
            div { class: "attr-bars",
                for (label, val, color) in rows {
                    {
                        let pct = val * 100.0;
                        rsx! {
                            div { class: "ladder-row", key: "{label}",
                                div { class: "ladder-label", div { class: "ll-main", "{label}" } }
                                div { class: "bar-track ladder-track",
                                    div { class: "bar-fill", style: "width:{pct}%; background:{color}" }
                                }
                                div { class: "ladder-val", "{pct:.0}%" }
                            }
                        }
                    }
                }
            }
            p { class: "attr-verdict", style: "color:{vcolor}", "{verdict}" }
            h4 { class: "attr-sub", "Per class, never-seen authors" }
            div { class: "attr-classes",
                for (i, cls) in attr.classes.iter().enumerate() {
                    {
                        let (c, t) = attr.per_class[i];
                        let color = attr.colors[i].clone();
                        if t == 0 {
                            // Only one author in this class: holding them out leaves no one to
                            // learn the class from, so it cannot be tested fairly.
                            rsx! {
                                div { class: "bar-row muted", key: "{cls}",
                                    Swatch { color: color.clone() }
                                    span { class: "bar-name", "{cls}" }
                                    span { class: "bar-note", "only 1 author, not testable" }
                                }
                            }
                        } else {
                            let acc = c as f32 / t as f32 * 100.0;
                            rsx! {
                                div { class: "bar-row", key: "{cls}",
                                    Swatch { color: color.clone() }
                                    span { class: "bar-name", "{cls}" }
                                    div { class: "bar-track",
                                        div { class: "bar-fill", style: "width:{acc}%; background:{color}" }
                                    }
                                    span { class: "bar-val", "{acc:.0}% ({c}/{t})" }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
