//! author2vec — a static Dioxus/WASM site exploring authorship in embedding space.

use dioxus::prelude::*;

mod data;
mod views;

use data::use_selector;
use views::{Accuracy, Dimensions, Map, Story};

static CSS: Asset = asset!("/assets/style.css");

fn main() {
    dioxus::launch(App);
}

#[derive(Routable, Clone, PartialEq)]
enum Route {
    #[layout(Shell)]
    #[route("/")]
    Map {},
    #[route("/accuracy")]
    Accuracy {},
    #[route("/dimensions")]
    Dimensions {},
    #[route("/why")]
    Story {},
}

#[component]
fn App() -> Element {
    // Provide the dataset to every view and start loading the manifest + default model.
    data::use_provide_dataset();
    rsx! {
        document::Stylesheet { href: CSS }
        Router::<Route> {}
    }
}

/// Shared shell: title, explainer, tab nav, model picker, and the routed content.
#[component]
fn Shell() -> Element {
    rsx! {
        div { class: "app",
            header { class: "hero",
                h1 { "author2vec" }
                p { class: "subtitle", "The AI's already know you, or they probably never will" }
                p { class: "tagline",
                    "Every author writes in a slightly different corner of vector space. "
                    "We embedded thousands of passages from famous public-domain books with a "
                    "local AI model. Here is what their fingerprints look like, and how well a "
                    "computer can tell them apart."
                }
                nav { class: "tabs",
                    Link { to: Route::Map {}, class: "tab", active_class: "active", "🗺 Vector-space map" }
                    Link { to: Route::Accuracy {}, class: "tab", active_class: "active", "🎯 Can we predict the author?" }
                    Link { to: Route::Dimensions {}, class: "tab", active_class: "active", "🔮 Hidden dimensions" }
                    Link { to: Route::Story {}, class: "tab", active_class: "active", "💭 Why I made this" }
                }
                ModelPicker {}
            }
            main { class: "content", Outlet::<Route> {} }
            footer { class: "footer",
                "Texts: "
                a { href: "https://www.gutenberg.org", "Project Gutenberg" }
                " · Embeddings via "
                a { href: "https://github.com/Anush008/fastembed-rs", "fastembed" }
                " (local ONNX) · Built with Rust + "
                a { href: "https://dioxuslabs.com", "Dioxus" }
                " · "
                a { href: "https://github.com/TristenHarr/author2vec", "Source on GitHub" }
                ". Embeddings and results precomputed once in Rust; the browser just displays them."
            }
        }
    }
}

/// Row of chips to switch which embedding model's vectors are loaded.
#[component]
fn ModelPicker() -> Element {
    let sel = use_selector();
    let manifest = sel.manifest.read();
    let Some(m) = manifest.as_ref() else {
        return rsx! {};
    };
    // With a single model there is nothing to pick.
    if m.models.len() <= 1 {
        return rsx! {};
    }
    let current = sel.selected.read().clone();
    rsx! {
        div { class: "model-picker",
            span { class: "mp-label", "Embedding model:" }
            for mi in m.models.iter() {
                {
                    let key = mi.key.clone();
                    let mut selected = sel.selected;
                    let active = current == mi.key;
                    rsx! {
                        button {
                            key: "{mi.key}",
                            class: if active { "chip active" } else { "chip" },
                            onclick: move |_| selected.set(key.clone()),
                            "{mi.name} "
                            span { class: "mp-dim", "{mi.dim}d · {mi.size}" }
                        }
                    }
                }
            }
        }
    }
}
