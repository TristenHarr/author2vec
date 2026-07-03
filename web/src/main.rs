//! author2vec — a static Dioxus/WASM site exploring authorship in embedding space.

use dioxus::prelude::*;

mod copy;
mod data;
mod views;

use copy::copy_for;
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
    Accuracy {},
    #[route("/map")]
    Map {},
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
    let sel = use_selector();
    let key = sel.selected.read().clone();
    let c = copy_for(&key);
    rsx! {
        div { class: "app",
            header { class: "hero",
                h1 { "author2vec" }
                p { class: "subtitle", "The AI's already know you, or they probably never will" }
                p { class: "tagline", "{c.tagline}" }
                DatasetToggle {}
                nav { class: "tabs",
                    Link { to: Route::Accuracy {}, class: "tab", active_class: "active", "{c.tab_predict}" }
                    Link { to: Route::Map {}, class: "tab", active_class: "active", "🗺 Vector-space map" }
                    Link { to: Route::Dimensions {}, class: "tab", active_class: "active", "🔮 Hidden dimensions" }
                    Link { to: Route::Story {}, class: "tab", active_class: "active", "💭 Why I made this" }
                }
            }
            main { class: "content", Outlet::<Route> {} }
            footer { class: "footer",
                "{c.source_prefix}"
                a { href: "{c.source_href}", "{c.source_name}" }
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

/// Segmented control to switch which dataset (Authors / Coders) is loaded.
#[component]
fn DatasetToggle() -> Element {
    let sel = use_selector();
    let manifest = sel.manifest.read();
    let Some(m) = manifest.as_ref() else {
        return rsx! {};
    };
    // With a single dataset there is nothing to switch.
    if m.datasets.len() <= 1 {
        return rsx! {};
    }
    let current = sel.selected.read().clone();
    rsx! {
        div { class: "model-picker",
            span { class: "mp-label", "Dataset:" }
            for d in m.datasets.iter() {
                {
                    let key = d.key.clone();
                    let mut selected = sel.selected;
                    let active = current == d.key;
                    rsx! {
                        button {
                            key: "{d.key}",
                            class: if active { "chip active" } else { "chip" },
                            onclick: move |_| selected.set(key.clone()),
                            "{d.label} "
                            span { class: "mp-dim", "{d.blurb}" }
                        }
                    }
                }
            }
        }
    }
}
