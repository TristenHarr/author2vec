//! Embed arbitrary code/text with a dataset's model (JinaBERT for `coders`, MiniLM for `minilm`),
//! into the SAME space as the shipped corpus — so generated AI code can be compared to the human
//! coder centroids. Used for the AI-model-fingerprint experiment.
//!
//! Usage: JLENS_DEVICE=cpu cargo run -p jlens --bin embed_texts --release -- <dataset> <in.json> <out.json>
//!   in.json:  [{"id": "...", "text": "..."}]
//!   out.json: [{"id": "...", "vec": [f32; dim]}]   (L2-normalized, masked-mean pooled)

use anyhow::Result;
use jlens::{normalize, Harness};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct InItem {
    id: String,
    text: String,
}

#[derive(Serialize)]
struct OutItem {
    id: String,
    vec: Vec<f32>,
}

fn main() -> Result<()> {
    let dataset = std::env::args().nth(1).unwrap_or_else(|| "coders".into());
    let inp = std::env::args().nth(2).expect("input json path");
    let outp = std::env::args().nth(3).expect("output json path");
    let h = Harness::load_dataset(&dataset)?;
    let items: Vec<InItem> = serde_json::from_slice(&std::fs::read(&inp)?)?;
    eprintln!("embedding {} texts with the {dataset} model …", items.len());
    let mut out = Vec::with_capacity(items.len());
    for (i, it) in items.iter().enumerate() {
        let txt = if it.text.trim().is_empty() { " " } else { it.text.as_str() };
        let fwd = h.forward(txt)?;
        out.push(OutItem { id: it.id.clone(), vec: normalize(fwd.embedding.clone()) });
        if i % 16 == 0 || i + 1 == items.len() {
            eprintln!("  {}/{}", i + 1, items.len());
        }
    }
    std::fs::write(&outp, serde_json::to_vec(&out)?)?;
    eprintln!("wrote {} embeddings -> {}", out.len(), outp);
    Ok(())
}
