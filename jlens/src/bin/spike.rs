//! Fast M0 re-validation: pooling-faithfulness gate + one-layer Jacobian sanity,
//! using the shared `jlens` core. `cargo run -p jlens --bin spike --release`.

use anyhow::Result;
use jlens::Harness;
use shared::{vectors_from_bytes, Meta};

fn main() -> Result<()> {
    let dataset = std::env::args().nth(1).unwrap_or_else(|| "minilm".to_string());
    let h = Harness::load_dataset(&dataset)?;
    let dim = h.dim;
    let assets = jlens::assets_dir();
    let meta: Meta = serde_json::from_slice(&std::fs::read(assets.join(format!("person2vec-{dataset}.json")))?)?;
    let refv = vectors_from_bytes(&std::fs::read(assets.join(format!("person2vec-{dataset}.bin")))?);
    let n = meta.passages.len();
    let sample: Vec<usize> = (0..8).map(|i| i * n / 8).collect();

    // Pooling-faithfulness gate: does our candle forward reproduce the shipped
    // (fastembed) embedding? This is the ground-truth test that the model is correct.
    let mut min_cos = 1f32;
    for &i in &sample {
        let e = h.forward(&meta.passages[i].text)?.embedding;
        min_cos = min_cos.min(jlens::dot(&e, &refv[i * dim..(i + 1) * dim]));
    }
    println!(
        "[{dataset}] pooling gate: min cosine {min_cos:.5} → {}",
        if min_cos > 0.99 { "PASS" } else { "FAIL" }
    );

    let fwd = h.forward(&meta.passages[sample[1]].text)?;
    let j = h.layer_jacobian(&fwd, h.num_layers / 2)?;
    println!(
        "J_3: stable_rank {:.2}   effective_dim {:.2}",
        jlens::stable_rank(&j, dim),
        jlens::effective_dim(&j, dim)
    );
    Ok(())
}
