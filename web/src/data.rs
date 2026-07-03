//! Multi-model dataset loading. The corpus generates one dataset per embedding model
//! (`person2vec-<key>.{json,bin}`) plus a manifest. The browser loads the manifest,
//! then the selected model's precomputed vectors, and does all the math locally.

use std::rc::Rc;

use dioxus::prelude::*;
use gloo_net::http::Request;
use shared::{Bundle, DatasetManifest, Meta};

// Compile-time asset handles. Dataset keys MUST match the file suffixes below (the
// authors dataset keeps the existing `minilm` files — no rename/re-embed needed).
static MANIFEST: Asset = asset!("/assets/person2vec-datasets.json");
static MINILM_JSON: Asset = asset!("/assets/person2vec-minilm.json");
static MINILM_BIN: Asset = asset!("/assets/person2vec-minilm.bin");
static CODERS_JSON: Asset = asset!("/assets/person2vec-coders.json");
static CODERS_BIN: Asset = asset!("/assets/person2vec-coders.bin");

fn dataset_urls(key: &str) -> Option<(String, String)> {
    Some(match key {
        "minilm" => (MINILM_JSON.to_string(), MINILM_BIN.to_string()),
        "coders" => (CODERS_JSON.to_string(), CODERS_BIN.to_string()),
        _ => return None,
    })
}

/// Load state of the *currently selected* model's bundle.
#[derive(Clone)]
pub enum LoadState {
    Loading,
    Loaded(Rc<Bundle>),
    Failed(String),
}

/// Dataset manifest + current selection, used by the dataset toggle.
#[derive(Clone, Copy)]
pub struct Selector {
    pub manifest: Signal<Option<DatasetManifest>>,
    pub selected: Signal<String>,
}

pub fn use_selector() -> Selector {
    use_context()
}

pub fn use_dataset() -> Signal<LoadState> {
    use_context()
}

pub fn use_provide_dataset() {
    let manifest = use_signal(|| None::<DatasetManifest>);
    let selected = use_signal(String::new);
    let state = use_signal(|| LoadState::Loading);

    use_context_provider(|| state);
    use_context_provider(|| Selector { manifest, selected });

    // Load the manifest once and pick the default dataset.
    use_future(move || async move {
        let mut manifest = manifest;
        let mut selected = selected;
        match load_manifest().await {
            Ok(m) => {
                if selected.peek().is_empty() {
                    selected.set(m.default.clone());
                }
                manifest.set(Some(m));
            }
            Err(_) => {
                // Fall back to the default dataset so the app still works without a manifest.
                if selected.peek().is_empty() {
                    selected.set("coders".to_string());
                }
            }
        }
    });

    // (Re)load the selected dataset's bundle whenever the selection changes.
    use_effect(move || {
        let key = selected();
        if key.is_empty() {
            return;
        }
        let mut state = state;
        spawn(async move {
            state.set(LoadState::Loading);
            match load_bundle(&key).await {
                Ok(b) => state.set(LoadState::Loaded(Rc::new(b))),
                Err(e) => state.set(LoadState::Failed(e)),
            }
        });
    });
}

async fn load_manifest() -> Result<DatasetManifest, String> {
    let bytes = fetch(&MANIFEST.to_string()).await?;
    serde_json::from_slice(&bytes).map_err(|e| e.to_string())
}

async fn load_bundle(key: &str) -> Result<Bundle, String> {
    let (json_url, bin_url) = dataset_urls(key).ok_or_else(|| format!("unknown dataset {key}"))?;
    let meta: Meta = serde_json::from_slice(&fetch(&json_url).await?).map_err(|e| e.to_string())?;
    let vectors = shared::vectors_from_bytes(&fetch(&bin_url).await?);
    let expected = meta.passages.len() * meta.dim;
    if vectors.len() != expected {
        return Err(format!("vector/passage mismatch for {key}"));
    }
    Ok(Bundle { meta, vectors })
}

async fn fetch(url: &str) -> Result<Vec<u8>, String> {
    Request::get(url)
        .send()
        .await
        .map_err(|e| format!("request {url}: {e}"))?
        .binary()
        .await
        .map_err(|e| format!("read {url}: {e}"))
}
