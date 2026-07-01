//! Recompute the precomputed `results` for every model from the vectors already on
//! disk — no network, no re-embedding. Use this after changing the math in
//! `shared::compute_results` (e.g. the LOOCV metrics) without regenerating embeddings.
//!
//!   cargo run -p corpus --bin recompute --release

use std::fs;
use std::path::PathBuf;

use shared::{Bundle, Manifest, Meta};

fn main() {
    let assets = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("web")
        .join("assets");
    let manifest: Manifest = serde_json::from_slice(
        &fs::read(assets.join("person2vec-models.json")).expect("read manifest"),
    )
    .expect("parse manifest");

    for m in &manifest.models {
        let key = &m.key;
        let mut meta: Meta = serde_json::from_slice(
            &fs::read(assets.join(format!("person2vec-{key}.json"))).expect("read json"),
        )
        .expect("parse json");
        let vectors = shared::vectors_from_bytes(
            &fs::read(assets.join(format!("person2vec-{key}.bin"))).expect("read bin"),
        );
        let bundle = Bundle { meta: meta.clone(), vectors };
        meta.results = shared::compute_results(&bundle);
        let json = serde_json::to_vec(&meta).expect("serialize");
        fs::write(assets.join(format!("person2vec-{key}.json")), json).expect("write json");
        println!("recomputed {} ({} authors)", m.name, meta.authors.len());
    }
}
