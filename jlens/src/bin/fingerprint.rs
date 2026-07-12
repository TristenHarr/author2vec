//! Rung 3, shown: the "is your fingerprint in the weights?" detector data.
//!
//! Ships identity centroids + per-identity signal tokens + precomputed sample
//! embeddings, so the browser can compute "known fingerprint vs blank space" live for a
//! chosen sample. Also prints the verdicts for verification.
//!
//! `cargo run -p jlens --bin fingerprint --release [minilm|coders]`

use anyhow::Result;

use jlens::{normalize, topk, Harness};
use shared::{vectors_from_bytes, FingerprintBundle, FpAuthor, FpProbe, Meta};

fn main() -> Result<()> {
    let dataset = std::env::args().nth(1).unwrap_or_else(|| "minilm".to_string());
    let h = Harness::load_dataset(&dataset)?;
    let dim = h.dim;
    let assets = jlens::assets_dir();
    let meta: Meta = serde_json::from_slice(&std::fs::read(assets.join(format!("person2vec-{dataset}.json")))?)?;
    let ref_vecs = vectors_from_bytes(&std::fs::read(assets.join(format!("person2vec-{dataset}.bin")))?);
    let n = meta.authors.len();
    let subject = if dataset == "coders" { "coder" } else { "author" };
    // Code embeddings cluster tighter than prose, so "known" needs a higher bar.
    let threshold = if dataset == "coders" { 0.51 } else { 0.30 };

    // Identity centroids + global mean (non-mystery passages).
    let (mut sums, mut counts) = (vec![vec![0f32; dim]; n], vec![0usize; n]);
    let (mut gsum, mut gcnt) = (vec![0f32; dim], 0usize);
    for (i, p) in meta.passages.iter().enumerate() {
        if p.is_mystery {
            continue;
        }
        let v = &ref_vecs[i * dim..(i + 1) * dim];
        for k in 0..dim {
            sums[p.author_id][k] += v[k];
            gsum[k] += v[k];
        }
        counts[p.author_id] += 1;
        gcnt += 1;
    }
    let global = normalize((0..dim).map(|k| gsum[k] / gcnt.max(1) as f32).collect());

    // Per-identity: centroid + the tokens its style direction (centroid − global) aligns
    // with, read through the word embeddings (embedding logit lens).
    let mut authors = Vec::new();
    for a in 0..n {
        let centroid = normalize((0..dim).map(|k| sums[a][k] / counts[a].max(1) as f32).collect());
        let dir = normalize((0..dim).map(|k| centroid[k] - global[k]).collect());
        let tokens: Vec<String> = topk(&h.vocab_logits(&dir)?, 8).into_iter().map(|i| h.token_str(i as u32)).collect();
        authors.push(FpAuthor {
            name: meta.authors[a].name.clone(),
            color: meta.authors[a].color.clone(),
            centroid,
            tokens,
        });
    }

    // Probe samples: real held-out (should be KNOWN) + out-of-distribution (BLANK SPACE).
    let mut probes = Vec::new();
    for p in meta.passages.iter().filter(|p| p.is_mystery).take(4) {
        let text: String = p.text.chars().take(800).collect();
        probes.push(FpProbe {
            label: format!("held-out {subject} sample"),
            snippet: snippet(&p.text, 90),
            vec: normalize(h.forward(&text)?.embedding),
        });
    }
    for (label, text) in ood_samples(&dataset) {
        probes.push(FpProbe {
            label: label.to_string(),
            snippet: snippet(text, 90),
            vec: normalize(h.forward(text)?.embedding),
        });
    }

    let bundle = FingerprintBundle { subject: subject.to_string(), threshold, authors, probes };
    std::fs::write(assets.join(format!("person2vec-fingerprint-{dataset}.json")), serde_json::to_vec(&bundle)?)?;

    println!("== fingerprint detector ({dataset}, threshold {threshold:.2}) ==");
    for p in &bundle.probes {
        let (idx, cos) = shared::nearest_centroid(&p.vec, &bundle.authors);
        let verdict = if cos >= threshold { "KNOWN" } else { "BLANK SPACE" };
        println!("  {:<26} cos {cos:.3} → {verdict:<11} (nearest {})", p.label, bundle.authors[idx].name);
    }
    println!("  wrote person2vec-fingerprint-{dataset}.json ({} {subject}s, {} probes)", bundle.authors.len(), bundle.probes.len());
    Ok(())
}

fn ood_samples(dataset: &str) -> Vec<(&'static str, &'static str)> {
    if dataset == "coders" {
        // Prose / non-code → should land in blank space for a code fingerprint.
        vec![
            ("Victorian prose", "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."),
            ("modern chat", "lol yeah just grabbing coffee brb, this standup is dragging so hard rn wyd after"),
            ("news headline", "Central bank holds interest rates steady amid signs of cooling inflation across the economy."),
            ("recipe", "Preheat the oven to 220C. Toss the potatoes in olive oil and salt, then roast for forty minutes until golden."),
        ]
    } else {
        vec![
            ("source code", "for (int i = 0; i < n; i++) { total += weights[i] * inputs[i]; } return sigmoid(total);"),
            ("modern chat", "lol yeah just grabbing coffee brb, this standup is dragging so hard rn wyd after"),
            ("biology abstract", "The mitochondrion generates ATP through oxidative phosphorylation across the inner membrane."),
            ("legalese", "The party of the first part hereby agrees to indemnify and hold harmless the party of the second part."),
        ]
    }
}

fn snippet(s: &str, n: usize) -> String {
    let t: String = s.chars().take(n).collect::<String>().replace('\n', " ");
    if s.chars().count() > n {
        format!("{t}…")
    } else {
        t
    }
}
