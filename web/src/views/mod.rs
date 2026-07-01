//! The three "hero" views plus small shared UI helpers.

mod accuracy;
mod dimensions;
mod map;
mod story;

pub use accuracy::Accuracy;
pub use dimensions::Dimensions;
pub use map::Map;
pub use story::Story;

use dioxus::prelude::*;

use crate::data::LoadState;

/// Render a loading spinner / error banner for the non-loaded states. Returns `None`
/// when data is ready so the caller can render its real content.
pub fn gate(state: &LoadState) -> Option<Element> {
    match state {
        LoadState::Loaded(_) => None,
        LoadState::Loading => Some(rsx! {
            div { class: "status",
                div { class: "spinner" }
                p { "Loading embeddings…" }
            }
        }),
        LoadState::Failed(e) => Some(rsx! {
            div { class: "status error",
                p { "Couldn't load the dataset." }
                pre { "{e}" }
                p { class: "hint",
                    "Did the corpus run? Generate the data with "
                    code { "cargo run -p corpus --release" }
                    "."
                }
            }
        }),
    }
}

/// A small colored dot used in legends and lists.
#[component]
pub fn Swatch(color: String) -> Element {
    rsx! { span { class: "swatch", style: "background:{color}" } }
}
