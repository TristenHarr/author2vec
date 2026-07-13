//! Full offline J-lens pipeline → `web/assets/person2vec-jlens-minilm.json`.
//!
//! Computes the averaged per-layer Jacobians (δ-broadcast) over a passage sample,
//! then precomputes everything the `/jlens` viewer displays: depth-wise structural
//! metrics (stable rank, effective dim, verbalizability, layer CKA), and, for a
//! handful of curated passages, the (layer × position) vocab-lens token grid plus
//! per-axis style trajectories through depth.
//!
//! Usage: `cargo run -p jlens --bin jlens --release [N_SAMPLE]` (default 160).

use std::io::Write;

use anyhow::Result;
use candle_core::IndexOp;

use jlens::{
    dataset_axes, effective_dim, even_nonmystery, excess_kurtosis, jackknife_se, linear_cka,
    loo_group_mean, matvec, normalize, readout_autocorrelation, stable_rank, style_scores,
    to_embedding_jacobian, vocab_topk, Harness,
};
use shared::{
    JlensBundle, JlensCell, JlensConcept, JlensExample, JlensJspace, JlensModel, JlensStructural,
    JlensStyleTraj, JlensTok, Meta,
};

const TOP_TOKENS: usize = 8;
// Sequence cap for the averaged-Jacobian pass (attention is O(T²)). Tunable via
// JLENS_JAC_LEN: 32 = fast, 64 = proper default, 96+ = thorough.
fn jac_len() -> usize {
    std::env::var("JLENS_JAC_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64)
}
const PROBE_PASSAGES: usize = 16; // passages contributing probe activations for metrics
const PROBE_POSITIONS: usize = 4;
const EXAMPLES: usize = 8; // curated passages shown in the viewer
const N_CONCEPTS: usize = 5; // top concepts tracked rank-vs-depth per example
const K_JSPACE: usize = 20; // sparse J-space atoms per example
const DISPLAY_POSITIONS: usize = 6; // token-grid columns per example
const STYLE_POSITIONS: usize = 16; // positions averaged for a style trajectory

fn main() -> Result<()> {
    let dataset = std::env::args().nth(1).unwrap_or_else(|| "minilm".to_string());
    let n_sample: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(160);
    // SE-only mode: recompute *only* the structural jackknife SE and merge it into the existing
    // bundle, preserving the committed point estimates and skipping all post-Jacobian work
    // (jspace, style, examples). Used for the slow 12-layer code encoder.
    let se_only = std::env::var("JLENS_SE_ONLY").is_ok();
    let model_name = if dataset == "coders" {
        "jinaai/jina-embeddings-v2-base-code"
    } else {
        jlens::MODEL_ID
    };

    println!("== J-lens pipeline (dataset={dataset}, N_SAMPLE={n_sample}) ==");
    let h = Harness::load_dataset(&dataset)?;
    let dim = h.dim;
    let n_layers = h.num_layers;

    let assets = jlens::assets_dir();
    let meta: Meta = serde_json::from_slice(&std::fs::read(assets.join(format!("person2vec-{dataset}.json")))?)?;
    let ref_vecs = shared::vectors_from_bytes(&std::fs::read(assets.join(format!("person2vec-{dataset}.bin")))?);

    // Style axes from the shipped reference embeddings (no model needed).
    let axes = dataset_axes(&meta.passages, &meta.authors, &ref_vecs, dim);
    println!("  {} style axes: {:?}", axes.len(), axes.iter().map(|a| &a.name).collect::<Vec<_>>());

    // Passage sample for the averaged Jacobians (non-mystery, evenly spread).
    let sample = even_nonmystery(&meta, n_sample);
    println!("  averaging Jacobians over {} passages × {} layers …", sample.len(), n_layers);

    let mut jraw_sum = vec![vec![0f64; dim * dim]; n_layers]; // f64 total over passages
    let mut jemb = vec![vec![0f64; dim * dim]; n_layers];
    // Per-fold Jacobian totals for a delete-a-group jackknife SE on the structural metrics.
    let n_folds = sample.len().clamp(1, 8);
    let mut fold_sum = vec![vec![vec![0f64; dim * dim]; n_layers]; n_folds];
    let mut fold_n = vec![0usize; n_folds];
    // Probe activations for structural metrics: [layer][sample][dim].
    let mut probe: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];

    let jl = jac_len();
    println!("  Jacobian context length = {jl} tokens");
    let t0 = std::time::Instant::now();
    for (ci, &pi) in sample.iter().enumerate() {
        let ts = std::time::Instant::now();
        let fwd = h.forward_capped(&meta.passages[pi].text, jl)?;
        let collect_probe = ci < PROBE_PASSAGES && !se_only;
        let probe_pos = spread_positions(fwd.t_len, PROBE_POSITIONS);
        let fold = ci % n_folds;
        fold_n[fold] += 1;
        for l in 0..n_layers {
            let jr = h.layer_jacobian(&fwd, l)?;
            for k in 0..dim * dim {
                let v = jr[k] as f64;
                jraw_sum[l][k] += v;
                fold_sum[fold][l][k] += v;
            }
            if !se_only {
                let je = to_embedding_jacobian(&jr, &fwd.embedding, fwd.pooled_norm, dim);
                for k in 0..dim * dim {
                    jemb[l][k] += je[k] as f64;
                }
            }
            if collect_probe {
                for &t in &probe_pos {
                    probe[l].push(fwd.hidden[l].i((0, t))?.to_vec1::<f32>()?);
                }
            }
        }
        // Per-passage timing exposes Metal JIT warmup (first few) vs steady state.
        if ci < 3 || ci % 16 == 0 || ci + 1 == sample.len() {
            println!(
                "    [{:>3}/{}] {:.2}s/passage  ·  elapsed {:.1}s",
                ci + 1,
                sample.len(),
                ts.elapsed().as_secs_f32(),
                t0.elapsed().as_secs_f32()
            );
            std::io::stdout().flush().ok();
        }
    }
    println!("  Jacobians done in {:.1}s", t0.elapsed().as_secs_f32());

    let scale = 1.0 / sample.len() as f64;
    let jraw: Vec<Vec<f32>> = jraw_sum
        .iter()
        .map(|m| m.iter().map(|&x| (x * scale) as f32).collect())
        .collect();
    let jemb: Vec<Vec<f32>> = jemb
        .iter()
        .map(|m| m.iter().map(|&x| (x * scale) as f32).collect())
        .collect();

    // ---- structural metrics across depth ----
    println!("  structural metrics …");
    let mut stable = vec![0f32; n_layers];
    let mut effdim = vec![0f32; n_layers];
    let mut verb = vec![0f32; n_layers];
    let mut readouts: Vec<Vec<f32>> = Vec::with_capacity(n_layers); // R_ℓ flattened (n_probes×dim)
    let n_probes = probe.first().map(|v| v.len()).unwrap_or(0);
    for l in 0..n_layers {
        stable[l] = stable_rank(&jraw[l], dim);
        effdim[l] = effective_dim(&jraw[l], dim);
        // verbalizability = mean excess kurtosis of the vocab lens over probe activations;
        // r collects normalized readout rows (n_probes × dim) for cross-layer CKA.
        let mut ks = Vec::new();
        let mut r = Vec::with_capacity(n_probes * dim);
        for act in &probe[l] {
            let raw = matvec(&jraw[l], act, dim);
            r.extend(normalize(raw.clone()));
            let mut jh = raw;
            jlens::standardize(&mut jh);
            ks.push(excess_kurtosis(&h.vocab_logits(&jh)?));
        }
        verb[l] = if ks.is_empty() { 0.0 } else { ks.iter().sum::<f32>() / ks.len() as f32 };
        readouts.push(r);
    }
    let mut cka = vec![vec![0f32; n_layers]; n_layers];
    for a in 0..n_layers {
        for b in 0..n_layers {
            cka[a][b] = if n_probes >= 2 {
                linear_cka(&readouts[a], &readouts[b], n_probes, dim)
            } else {
                if a == b { 1.0 } else { 0.0 }
            };
        }
    }
    // Delete-a-group jackknife SE on the two load-bearing structural metrics: recompute each
    // metric on every leave-one-fold-out passage average, then take Tukey's grouped-jackknife SE.
    let mut stable_se = vec![0f32; n_layers];
    let mut effdim_se = vec![0f32; n_layers];
    for l in 0..n_layers {
        let mut sr = Vec::with_capacity(n_folds);
        let mut ed = Vec::with_capacity(n_folds);
        for f in 0..n_folds {
            let loo = loo_group_mean(&jraw_sum[l], &fold_sum[f][l], sample.len(), fold_n[f]);
            sr.push(stable_rank(&loo, dim));
            ed.push(effective_dim(&loo, dim));
        }
        stable_se[l] = jackknife_se(&sr);
        effdim_se[l] = jackknife_se(&ed);
    }
    println!("    stable_rank: {:?}", round2(&stable));
    println!("    stable_rank_se(±): {:?}", round2(&stable_se));
    println!("    effective_dim: {:?}", round2(&effdim));
    println!("    effective_dim_se(±): {:?}", round2(&effdim_se));

    // SE-only: merge the two SE vectors into the committed bundle (point estimates preserved) and stop.
    if se_only {
        let path = assets.join(format!("person2vec-jlens-{dataset}.json"));
        let mut bundle: JlensBundle = serde_json::from_slice(&std::fs::read(&path)?)?;
        let drift: f32 = bundle.structural.stable_rank.iter().zip(&stable).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        println!("  (sanity: max |committed − recomputed| stable_rank = {drift:.4})");
        bundle.structural.stable_rank_se = stable_se;
        bundle.structural.effective_dim_se = effdim_se;
        std::fs::write(&path, serde_json::to_vec(&bundle)?)?;
        println!("  merged structural SE into {} (point estimates preserved)", path.display());
        return Ok(());
    }
    println!("    verbalizability(kurtosis): {:?}", round2(&verb));

    // ---- autocorrelation across depth (paper's 4th Fig-28 signature) ----
    // Per layer: lag-1 cosine autocorrelation of the per-position readout J·h_t
    // minus the position-shuffled null, averaged over the probe passages. High ⇒
    // the layer's readout persists across the token sequence (workspace-like).
    let mut autocorr = vec![0f32; n_layers];
    let mut ac_counts = vec![0u32; n_layers];
    for &pi in sample.iter().take(PROBE_PASSAGES) {
        let fwd = h.forward_capped(&meta.passages[pi].text, jl)?;
        let hi = fwd.t_len.saturating_sub(1); // drop trailing [SEP]
        if hi <= 2 {
            continue;
        }
        for l in 0..n_layers {
            let mut seq = Vec::with_capacity(hi - 1);
            for t in 1..hi {
                let act = fwd.hidden[l].i((0, t))?.to_vec1::<f32>()?;
                seq.push(matvec(&jraw[l], &act, dim));
            }
            let ac = readout_autocorrelation(&seq);
            if ac.is_finite() {
                autocorr[l] += ac;
                ac_counts[l] += 1;
            }
        }
    }
    for l in 0..n_layers {
        if ac_counts[l] > 0 {
            autocorr[l] /= ac_counts[l] as f32;
        }
    }
    println!("    autocorrelation: {:?}", round2(&autocorr));

    // ---- J-space dictionary at the most "verbalizable" (peak-kurtosis) layer ----
    let jspace_layer = verb
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(n_layers - 1);
    let d_unit = h.jlens_dictionary(&jraw[jspace_layer])?;
    println!("  J-space dictionary at layer {jspace_layer}");

    // ---- curated examples: token grid + style trajectories + concepts + J-space ----
    println!("  curated examples …");
    let example_idx = pick_examples(&meta, EXAMPLES);
    let mut examples = Vec::new();
    for &pi in &example_idx {
        let p = &meta.passages[pi];
        let author = &meta.authors[p.author_id];
        let fwd = h.forward(&p.text)?;
        let disp = spread_positions(fwd.t_len, DISPLAY_POSITIONS);

        let mut cells = Vec::new();
        for l in 0..n_layers {
            for &t in &disp {
                let act = fwd.hidden[l].i((0, t))?.to_vec1::<f32>()?;
                let top = vocab_topk(&h, &jraw[l], &act, TOP_TOKENS)?;
                cells.push(JlensCell {
                    layer: l,
                    pos: t,
                    token: fwd.tokens.get(t).cloned().unwrap_or_default(),
                    top: top
                        .into_iter()
                        .map(|(id, s)| JlensTok { tok: h.token_str(id), score: s })
                        .collect(),
                });
            }
        }

        // Style trajectory: per axis, per layer, averaged over spread positions.
        let spos = spread_positions(fwd.t_len, STYLE_POSITIONS);
        let mut style: Vec<JlensStyleTraj> = (0..axes.len())
            .map(|axis| JlensStyleTraj { axis, per_layer: vec![0f32; n_layers] })
            .collect();
        for l in 0..n_layers {
            let mut acc = vec![0f32; axes.len()];
            for &t in &spos {
                let act = fwd.hidden[l].i((0, t))?.to_vec1::<f32>()?;
                for (k, s) in style_scores(&jemb[l], &act, &axes, dim).into_iter().enumerate() {
                    acc[k] += s;
                }
            }
            for (k, tr) in style.iter_mut().enumerate() {
                tr.per_layer[l] = acc[k] / spos.len() as f32;
            }
        }

        // Concept rank-vs-depth trajectories + J-space decomposition (mean activation).
        let mut logits = Vec::with_capacity(n_layers);
        let mut mean_acts = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let ma = fwd.hidden[l].mean(1)?.squeeze(0)?.to_vec1::<f32>()?;
            let mut jh = matvec(&jraw[l], &ma, dim);
            jlens::standardize(&mut jh);
            logits.push(h.vocab_logits(&jh)?);
            mean_acts.push(ma);
        }
        let concepts = jlens::topk(&logits[n_layers - 1], N_CONCEPTS)
            .into_iter()
            .map(|cid| JlensConcept {
                tok: h.token_str(cid as u32),
                per_layer_rank: (0..n_layers)
                    .map(|l| 1 + logits[l].iter().filter(|&&x| x > logits[l][cid]).count() as u32)
                    .collect(),
            })
            .collect();
        let (items, cvar) = h.jspace_nmp(&d_unit, &mean_acts[jspace_layer], K_JSPACE)?;
        let jspace = JlensJspace {
            layer: jspace_layer,
            captured_variance: cvar,
            items: items
                .into_iter()
                .map(|(id, c)| JlensTok { tok: h.token_str(id), score: c })
                .collect(),
        };

        examples.push(JlensExample {
            passage_idx: pi,
            author: author.name.clone(),
            color: author.color.clone(),
            snippet: snippet(&p.text, 160),
            tokens: fwd.tokens.clone(),
            cells,
            style,
            concepts,
            jspace,
        });
    }

    // ---- write bundle ----
    let bundle = JlensBundle {
        model: JlensModel {
            name: model_name.to_string(),
            layers: n_layers,
            d_model: dim,
            vocab: h.word_embeddings().dim(0)?,
            max_len: jlens::MAX_LEN,
            w_u_source: "tied word embeddings (logit lens)".to_string(),
            passages: sample.len(),
        },
        axes: axes.iter().map(|a| a.name.clone()).collect(),
        layers: (0..n_layers).collect(),
        structural: JlensStructural {
            stable_rank: stable,
            effective_dim: effdim,
            verbalizability: verb,
            cka,
            autocorrelation: autocorr,
            stable_rank_se: stable_se,
            effective_dim_se: effdim_se,
        },
        examples,
    };
    let out = assets.join(format!("person2vec-jlens-{dataset}.json"));
    std::fs::write(&out, serde_json::to_vec(&bundle)?)?;
    println!(
        "  wrote {} ({:.1} KB, {} examples)",
        out.display(),
        std::fs::metadata(&out)?.len() as f64 / 1024.0,
        bundle.examples.len()
    );
    Ok(())
}


/// One representative passage for each of `k` authors spread across the roster.
fn pick_examples(meta: &Meta, k: usize) -> Vec<usize> {
    let n_authors = meta.authors.len();
    let mut out = Vec::new();
    for a in 0..k {
        let author = a * n_authors / k;
        if let Some((i, _)) = meta
            .passages
            .iter()
            .enumerate()
            .find(|(_, p)| p.author_id == author && !p.is_mystery)
        {
            out.push(i);
        }
    }
    out
}

/// `k` evenly-spread token positions in `[1, t_len-2]` (skipping [CLS]/[SEP]).
fn spread_positions(t_len: usize, k: usize) -> Vec<usize> {
    if t_len <= 2 {
        return vec![0];
    }
    let lo = 1usize;
    let hi = t_len - 2;
    if hi <= lo {
        return vec![lo];
    }
    let span = hi - lo;
    let mut v: Vec<usize> = (0..k)
        .map(|i| lo + (i * span) / (k.max(2) - 1).max(1))
        .filter(|&p| p <= hi)
        .collect();
    v.dedup();
    v
}

fn round2(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| (x * 100.0).round() / 100.0).collect()
}

fn snippet(s: &str, n: usize) -> String {
    let t: String = s.chars().take(n).collect::<String>().replace('\n', " ");
    if s.chars().count() > n {
        format!("{t}…")
    } else {
        t
    }
}
