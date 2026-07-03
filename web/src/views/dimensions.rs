//! Tab: recovering author traits (gender, birth country) from writing style alone,
//! using leave-one-author-out so the model can't cheat by recognizing the author.

use dioxus::prelude::*;

use super::{gate, Swatch};
use crate::copy::copy_for;
use crate::data::{use_dataset, use_selector, LoadState};
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
    let c = copy_for(&use_selector().selected.read().clone());

    rsx! {
        div { class: "dims-view",
            p { class: "explainer", "{c.dims_explainer}" }
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
    let c = copy_for(&use_selector().selected.read().clone());
    let sub = c.subject;
    let ents = c.entities;
    let rows = [
        ("Blind guessing (random)".to_string(), attr.random_baseline, "#9a9aa8"),
        ("Always guess the majority".to_string(), attr.baseline, "#c0392b"),
        (format!("Never read the {sub}"), attr.fair_accuracy, "#5b4be0"),
        (format!("Has read the {sub} (leaks identity)"), attr.leaky_accuracy, "#2a9d5c"),
    ];
    let fair_caption = format!(
        "Guessed from style, for a {sub} never seen. Fair test covers {}/{} {ents}.",
        attr.tested_authors, attr.total_authors
    );

    rsx! {
        div { class: "attr-panel",
            h3 { "{attr.name}" }
            p { class: "matrix-cap", "{fair_caption}" }
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
            h4 { class: "attr-sub", "Per class, never-seen {ents}" }
            div { class: "attr-classes",
                for (i, cls) in attr.classes.iter().enumerate() {
                    {
                        let (c, t) = attr.per_class[i];
                        let color = attr.colors[i].clone();
                        let label = if cls == "None" { "No college (self-taught)" } else { cls.as_str() };
                        if t == 0 {
                            // Only one author in this class: holding them out leaves no one to
                            // learn the class from, so it cannot be tested fairly.
                            rsx! {
                                div { class: "bar-row muted", key: "{cls}",
                                    Swatch { color: color.clone() }
                                    span { class: "bar-name", "{label}" }
                                    span { class: "bar-note", "only 1 {sub}, not testable" }
                                }
                            }
                        } else {
                            let acc = c as f32 / t as f32 * 100.0;
                            rsx! {
                                div { class: "bar-row", key: "{cls}",
                                    Swatch { color: color.clone() }
                                    span { class: "bar-name", "{label}" }
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
