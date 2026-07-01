//! Shared data model + vector math for **person2vec**.
//!
//! This crate is deliberately dependency-light (only `serde`) so it compiles for
//! *both* the native offline generator (`corpus`) and the wasm web app (`web`).
//! Embeddings come from all-MiniLM-L6-v2 and are already L2-normalized, so cosine
//! similarity is just a dot product throughout.

use std::collections::{BTreeMap, HashMap, HashSet};

use serde::{Deserialize, Serialize};

/// Embedding dimensionality of all-MiniLM-L6-v2.
pub const EMBED_DIM: usize = 384;

/// One author. `id` is a dense index into [`Meta::authors`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthorMeta {
    pub id: usize,
    pub name: String,
    /// CSS hex color, used consistently across every view.
    pub color: String,
    /// "Male" / "Female" — a trait we try to recover from writing style alone.
    #[serde(default)]
    pub gender: String,
    /// Country of birth — another trait we try to recover from style.
    #[serde(default)]
    pub birth_country: String,
}

/// One ~200-word passage of an author's prose.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Passage {
    pub author_id: usize,
    pub book_title: String,
    /// Series the book belongs to (defaults to the book title for standalone works).
    /// Used to hold out an entire series, not just one book.
    #[serde(default)]
    pub series: String,
    pub text: String,
    /// 2D PCA coordinates for the map view.
    pub x: f32,
    pub y: f32,
    /// Held-out passages: excluded from centroids/LOOCV, used as classifier queries.
    pub is_mystery: bool,
}

/// One embedding model available in the demo (different model / different size).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelInfo {
    /// Filename key: `person2vec-<key>.{json,bin}`.
    pub key: String,
    pub name: String,
    pub dim: usize,
    /// Human size label, e.g. "tiny · 23M params".
    pub size: String,
}

/// Lists the models the corpus generated; loaded first by the web app.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Manifest {
    pub default: String,
    pub models: Vec<ModelInfo>,
}

/// One rung of the familiarity ladder (how much the model has read of the author).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Rung {
    pub label: String,
    pub sublabel: String,
    pub accuracy: f32,
}

/// Per-author accuracy, precomputed.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AuthorAcc {
    pub author_id: usize,
    pub accuracy: f32,
    pub correct: u32,
    pub total: u32,
}

/// One "next-work" prediction: the author guessed for a held-out passage, both when
/// the model HAS read that author (their other work is in the index) and when it has
/// NOT (their whole body of work is held out).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Prediction {
    pub passage_idx: usize,
    pub true_author: usize,
    pub pred_seen: usize,
    pub pred_unseen: usize,
}

/// Recovering an author trait (gender, birth country) from writing style alone.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)] // tolerate older JSON missing newer fields
pub struct AttrResult {
    pub name: String,
    pub classes: Vec<String>,
    pub colors: Vec<String>,
    /// Always-guess-majority baseline (fraction).
    pub baseline: f32,
    /// Blind/random guessing baseline = 1 / number-of-classes.
    pub random_baseline: f32,
    /// Leave-one-author-out: predict the trait of an author never seen. The honest one.
    pub fair_accuracy: f32,
    /// Leave-one-passage-out: the model has read the author (leaks their identity).
    pub leaky_accuracy: f32,
    /// (correct, total) per class, fair.
    pub per_class: Vec<(u32, u32)>,
    pub confusion: Vec<Vec<u32>>,
    pub tested_authors: usize,
    pub total_authors: usize,
}

/// Everything the accuracy view displays, computed offline by the corpus.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)] // tolerate older JSON missing newer fields
pub struct Results {
    pub baseline: f32,
    pub rungs: Vec<Rung>,
    pub per_author: Vec<AuthorAcc>,
    pub confusion: Vec<Vec<u32>>,
    pub headline_correct: u32,
    pub headline_total: u32,
    pub curve: Vec<(usize, f32)>,
    /// Trait-recovery experiments (gender, birth country).
    pub attributes: Vec<AttrResult>,
    /// The "next-work prediction" reel over held-out passages.
    pub reel: Vec<Prediction>,
    pub reel_hits_seen: u32,
    pub reel_hits_unseen: u32,
    pub reel_total: u32,
    /// Cosine of held-out writing to the AI's model of "you" when it HAS read you
    /// (your own centroid) versus when it hasn't (the average of every author).
    pub trained_sim: f32,
    pub untrained_sim: f32,
}

/// Everything except the raw vectors — serialized as `person2vec-<key>.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Meta {
    pub dim: usize,
    pub authors: Vec<AuthorMeta>,
    pub passages: Vec<Passage>,
    /// Precomputed accuracy results (the browser just displays these).
    #[serde(default)]
    pub results: Results,
}

impl Meta {
    pub fn author(&self, id: usize) -> &AuthorMeta {
        &self.authors[id]
    }
}

/// [`Meta`] plus the packed vectors (`person2vec.bin`), assembled at runtime.
#[derive(Clone, Debug)]
pub struct Bundle {
    pub meta: Meta,
    /// Flat row-major buffer: passage `i` occupies `vectors[i*dim .. (i+1)*dim]`.
    pub vectors: Vec<f32>,
}

impl Bundle {
    pub fn len(&self) -> usize {
        self.meta.passages.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn n_authors(&self) -> usize {
        self.meta.authors.len()
    }
    /// The embedding vector for passage `i`.
    pub fn vector(&self, i: usize) -> &[f32] {
        let d = self.meta.dim;
        &self.vectors[i * d..(i + 1) * d]
    }
    pub fn author_of(&self, i: usize) -> &AuthorMeta {
        self.meta.author(self.meta.passages[i].author_id)
    }
}

// ---------------------------------------------------------------------------
// Vector math
// ---------------------------------------------------------------------------

/// Dot product. For L2-normalized vectors this equals cosine similarity.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Normalize `v` to unit length in place (no-op for a zero vector).
pub fn l2_normalize(v: &mut [f32]) {
    let norm = dot(v, v).sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Per-author summed training vectors + counts, over the *training* passages
/// (everything with `is_mystery == false`). Lets us recompute leave-one-out
/// centroids cheaply as `(sum - v) / (count - 1)`.
pub struct Centroids {
    pub dim: usize,
    pub sums: Vec<Vec<f32>>,
    pub counts: Vec<usize>,
    /// Cached full normalized centroids (one per author).
    full: Vec<Vec<f32>>,
}

impl Centroids {
    pub fn build(bundle: &Bundle) -> Self {
        let dim = bundle.meta.dim;
        let n = bundle.n_authors();
        let mut sums = vec![vec![0f32; dim]; n];
        let mut counts = vec![0usize; n];
        for (i, p) in bundle.meta.passages.iter().enumerate() {
            if p.is_mystery {
                continue;
            }
            let v = bundle.vector(i);
            let s = &mut sums[p.author_id];
            for k in 0..dim {
                s[k] += v[k];
            }
            counts[p.author_id] += 1;
        }
        let full = (0..n)
            .map(|a| Self::normalized(&sums[a], counts[a], dim))
            .collect();
        Self {
            dim,
            sums,
            counts,
            full,
        }
    }

    fn normalized(sum: &[f32], count: usize, dim: usize) -> Vec<f32> {
        let mut c = vec![0f32; dim];
        if count > 0 {
            for k in 0..dim {
                c[k] = sum[k] / count as f32;
            }
        }
        l2_normalize(&mut c);
        c
    }

    /// Full (all training passages) normalized centroid for author `a`.
    pub fn centroid(&self, a: usize) -> &[f32] {
        &self.full[a]
    }

    /// Leave-one-out normalized centroid for author `a`, removing `v`.
    fn centroid_without(&self, a: usize, v: &[f32]) -> Vec<f32> {
        let mut c = self.sums[a].clone();
        for k in 0..self.dim {
            c[k] -= v[k];
        }
        Self::normalized(&c, self.counts[a].saturating_sub(1), self.dim)
    }

    /// Rank authors by cosine of `query` to each author's full centroid, best first.
    pub fn rank(&self, query: &[f32]) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = (0..self.full.len())
            .filter(|&a| self.counts[a] > 0)
            .map(|a| (a, dot(query, &self.full[a])))
            .collect();
        scored.sort_by(|x, y| y.1.total_cmp(&x.1));
        scored
    }
}

/// Outcome of a leave-one-out cross-validation run.
#[derive(Clone, Debug, PartialEq)]
pub struct LoocvResult {
    pub accuracy: f32,
    /// `confusion[true_author][predicted_author]`.
    pub confusion: Vec<Vec<u32>>,
    /// Per-author `(correct, total)`.
    pub per_author: Vec<(u32, u32)>,
    pub correct: u32,
    pub total: u32,
}

impl LoocvResult {
    fn empty(n: usize) -> Self {
        Self {
            accuracy: 0.0,
            confusion: vec![vec![0u32; n]; n],
            per_author: vec![(0, 0); n],
            correct: 0,
            total: 0,
        }
    }
    fn record(&mut self, truth: usize, pred: usize) {
        self.confusion[truth][pred] += 1;
        self.per_author[truth].1 += 1;
        self.total += 1;
        if pred == truth {
            self.correct += 1;
            self.per_author[truth].0 += 1;
        }
    }
    fn finish(mut self) -> Self {
        self.accuracy = self.correct as f32 / self.total.max(1) as f32;
        self
    }
}

/// Exact leave-one-out nearest-centroid classification over all training passages.
/// Fast: O(N · A · d). This is the default, instant accuracy view.
pub fn loocv_nearest_centroid(bundle: &Bundle) -> LoocvResult {
    let n = bundle.n_authors();
    let cents = Centroids::build(bundle);
    let mut out = LoocvResult::empty(n);
    for (i, p) in bundle.meta.passages.iter().enumerate() {
        if p.is_mystery {
            continue;
        }
        let v = bundle.vector(i);
        let a = p.author_id;
        let loo = cents.centroid_without(a, v);
        let mut best = a;
        let mut best_sim = f32::NEG_INFINITY;
        for b in 0..n {
            if cents.counts[b] == 0 {
                continue;
            }
            let sim = if b == a { dot(v, &loo) } else { dot(v, cents.centroid(b)) };
            if sim > best_sim {
                best_sim = sim;
                best = b;
            }
        }
        out.record(a, best);
    }
    out.finish()
}

/// Leave-one-out k-nearest-neighbour classification. Heavier: O(N² · d). Offered
/// on demand in the UI. `k` is the neighbour count (majority vote, sim-weighted
/// tie-break).
pub fn loocv_knn(bundle: &Bundle, k: usize) -> LoocvResult {
    let n = bundle.n_authors();
    let train: Vec<usize> = bundle
        .meta
        .passages
        .iter()
        .enumerate()
        .filter(|(_, p)| !p.is_mystery)
        .map(|(i, _)| i)
        .collect();
    let mut out = LoocvResult::empty(n);
    for &i in &train {
        let vi = bundle.vector(i);
        let a = bundle.meta.passages[i].author_id;
        // Keep the top-k (sim, author) with a tiny insertion-sorted buffer.
        let mut best: Vec<(f32, usize)> = Vec::with_capacity(k + 1);
        for &j in &train {
            if j == i {
                continue;
            }
            let sim = dot(vi, bundle.vector(j));
            if best.len() < k {
                best.push((sim, bundle.meta.passages[j].author_id));
                best.sort_by(|x, y| y.0.total_cmp(&x.0));
            } else if sim > best[k - 1].0 {
                best[k - 1] = (sim, bundle.meta.passages[j].author_id);
                best.sort_by(|x, y| y.0.total_cmp(&x.0));
            }
        }
        out.record(a, majority_vote(&best, n));
    }
    out.finish()
}

fn majority_vote(neighbors: &[(f32, usize)], n_authors: usize) -> usize {
    let mut votes = vec![0u32; n_authors];
    let mut simsum = vec![0f32; n_authors];
    for &(s, a) in neighbors {
        votes[a] += 1;
        simsum[a] += s;
    }
    (0..n_authors)
        .max_by(|&x, &y| votes[x].cmp(&votes[y]).then(simsum[x].total_cmp(&simsum[y])))
        .unwrap_or(0)
}

/// Nearest training passages to `query`, best first (excludes mystery passages and
/// optionally one index).
pub fn nearest_passages(
    bundle: &Bundle,
    query: &[f32],
    exclude: Option<usize>,
    top: usize,
) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = (0..bundle.len())
        .filter(|&i| Some(i) != exclude && !bundle.meta.passages[i].is_mystery)
        .map(|i| (i, dot(query, bundle.vector(i))))
        .collect();
    scored.sort_by(|x, y| y.1.total_cmp(&x.1));
    scored.truncate(top);
    scored
}

// ---------------------------------------------------------------------------
// Binary (de)serialization of the packed f32 vector blob (person2vec.bin)
// ---------------------------------------------------------------------------

/// Pack a flat f32 buffer into little-endian bytes.
pub fn vectors_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Parse a little-endian f32 blob back into a flat buffer.
pub fn vectors_from_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ---------------------------------------------------------------------------
// Fairness: "has it seen this exact book?" and the familiarity gradient
// ---------------------------------------------------------------------------

fn centroids_excluding(bundle: &Bundle, held: &HashSet<usize>) -> Vec<Option<Vec<f32>>> {
    let dim = bundle.meta.dim;
    let n = bundle.n_authors();
    let mut sums = vec![vec![0f32; dim]; n];
    let mut counts = vec![0usize; n];
    for (i, p) in bundle.meta.passages.iter().enumerate() {
        if p.is_mystery || held.contains(&i) {
            continue;
        }
        let v = bundle.vector(i);
        for k in 0..dim {
            sums[p.author_id][k] += v[k];
        }
        counts[p.author_id] += 1;
    }
    (0..n).map(|a| centroid_of_sum(&sums[a], counts[a], dim)).collect()
}

fn centroid_of_sum(sum: &[f32], count: usize, dim: usize) -> Option<Vec<f32>> {
    if count == 0 {
        return None;
    }
    let mut c: Vec<f32> = (0..dim).map(|k| sum[k] / count as f32).collect();
    l2_normalize(&mut c);
    Some(c)
}

fn argmax_centroid(cents: &[Option<Vec<f32>>], v: &[f32]) -> Option<usize> {
    let mut best = None;
    let mut best_sim = f32::NEG_INFINITY;
    for (a, c) in cents.iter().enumerate() {
        if let Some(c) = c {
            let s = dot(v, c);
            if s > best_sim {
                best_sim = s;
                best = Some(a);
            }
        }
    }
    best
}

/// Leave-one-group-out nearest-centroid, where a "group" is picked by `group_of`
/// (book title, or series). Holding out a whole group forces the author's centroid to
/// come only from their *other* work.
fn loocv_leave_group_out(bundle: &Bundle, group_of: impl Fn(&Passage) -> &str) -> LoocvResult {
    let n = bundle.n_authors();
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    let mut slot_of: HashMap<(usize, &str), usize> = HashMap::new();
    for (i, p) in bundle.meta.passages.iter().enumerate() {
        if p.is_mystery {
            continue;
        }
        let key = (p.author_id, group_of(p));
        let slot = *slot_of.entry(key).or_insert_with(|| {
            groups.push((p.author_id, Vec::new()));
            groups.len() - 1
        });
        groups[slot].1.push(i);
    }
    let mut out = LoocvResult::empty(n);
    for (author, idxs) in &groups {
        let held: HashSet<usize> = idxs.iter().copied().collect();
        let cents = centroids_excluding(bundle, &held);
        if cents[*author].is_none() {
            continue; // no other work to train on
        }
        for &i in idxs {
            if let Some(pred) = argmax_centroid(&cents, bundle.vector(i)) {
                out.record(*author, pred);
            }
        }
    }
    out.finish()
}

/// Hold out the whole book; train on the author's other books (including same series).
pub fn loocv_leave_book_out(bundle: &Bundle) -> LoocvResult {
    loocv_leave_group_out(bundle, |p| p.book_title.as_str())
}

/// Hold out the whole SERIES; train on the author's other series / standalone works.
pub fn loocv_leave_series_out(bundle: &Bundle) -> LoocvResult {
    loocv_leave_group_out(bundle, |p| p.series.as_str())
}

/// Hold out EVERY work by the author. With no centroid for them, the model can never
/// answer correctly — the honest floor of "it has never read you".
pub fn loocv_leave_author_out(bundle: &Bundle) -> LoocvResult {
    let n = bundle.n_authors();
    let mut out = LoocvResult::empty(n);
    for author in 0..n {
        let held: HashSet<usize> = bundle
            .meta
            .passages
            .iter()
            .enumerate()
            .filter(|(_, p)| p.author_id == author && !p.is_mystery)
            .map(|(i, _)| i)
            .collect();
        if held.is_empty() {
            continue;
        }
        let cents = centroids_excluding(bundle, &held);
        for &i in &held {
            if let Some(pred) = argmax_centroid(&cents, bundle.vector(i)) {
                out.record(author, pred);
            }
        }
    }
    out.finish()
}

/// Precompute the whole familiarity ladder + supporting panels for the accuracy view.
pub fn compute_results(bundle: &Bundle) -> Results {
    let n = bundle.n_authors();
    let baseline = 1.0 / n as f32;
    let seen = loocv_nearest_centroid(bundle);
    let book_out = loocv_leave_book_out(bundle);
    let series_out = loocv_leave_series_out(bundle);
    let author_out = loocv_leave_author_out(bundle);
    let curve = familiarity_curve(bundle, &[2, 4, 8, 16, 32, 64, 128]);

    let rungs = vec![
        Rung { label: "Blind guessing".into(), sublabel: "no information".into(), accuracy: baseline },
        Rung { label: "Never read the author".into(), sublabel: "held out every work by them".into(), accuracy: author_out.accuracy },
        Rung { label: "Never read this series".into(), sublabel: "has read the author's other work".into(), accuracy: series_out.accuracy },
        Rung { label: "Never read this book".into(), sublabel: "has read the rest of the series".into(), accuracy: book_out.accuracy },
        Rung { label: "Has read the book".into(), sublabel: "has seen other passages from it".into(), accuracy: seen.accuracy },
    ];
    let per_author = (0..n)
        .map(|a| {
            let (c, t) = seen.per_author[a];
            AuthorAcc {
                author_id: a,
                accuracy: if t > 0 { c as f32 / t as f32 } else { 0.0 },
                correct: c,
                total: t,
            }
        })
        .collect();

    let (reel, reel_hits_seen, reel_hits_unseen, reel_total) = compute_reel(bundle);
    let attributes = vec![
        compute_attribute(bundle, "Gender", |a| a.gender.as_str()),
        compute_attribute(bundle, "Birth country", |a| a.birth_country.as_str()),
    ];

    // Superpower illustration: how well the AI's model of "you" fits your held-out
    // writing when trained (your own centroid) vs untrained (the average of ALL authors,
    // the generic point it defaults to for anyone it has never read).
    let dim = bundle.meta.dim;
    let mut gsum = vec![0f32; dim];
    let mut gcnt = 0usize;
    for (i, p) in bundle.meta.passages.iter().enumerate() {
        if p.is_mystery {
            continue;
        }
        let v = bundle.vector(i);
        for d in 0..dim {
            gsum[d] += v[d];
        }
        gcnt += 1;
    }
    let global = centroid_of_sum(&gsum, gcnt, dim);
    let cents = Centroids::build(bundle);
    let (mut trained, mut untrained, mut sc) = (0f32, 0f32, 0u32);
    for (i, p) in bundle.meta.passages.iter().enumerate() {
        if !p.is_mystery {
            continue;
        }
        let v = bundle.vector(i);
        trained += dot(v, cents.centroid(p.author_id));
        if let Some(g) = &global {
            untrained += dot(v, g);
        }
        sc += 1;
    }
    let denom = sc.max(1) as f32;

    Results {
        baseline,
        rungs,
        per_author,
        confusion: seen.confusion.clone(),
        headline_correct: seen.correct,
        headline_total: seen.total,
        curve,
        attributes,
        reel,
        reel_hits_seen,
        reel_hits_unseen,
        reel_total,
        trained_sim: trained / denom,
        untrained_sim: untrained / denom,
    }
}

/// The "next-work" prediction reel: for each held-out passage, guess its author both
/// when the model HAS read that author and when their whole body of work is held out.
fn compute_reel(bundle: &Bundle) -> (Vec<Prediction>, u32, u32, u32) {
    let full = Centroids::build(bundle);
    let test: Vec<usize> = bundle
        .meta
        .passages
        .iter()
        .enumerate()
        .filter(|(_, p)| p.is_mystery)
        .map(|(i, _)| i)
        .collect();

    let mut by_author: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &i in &test {
        by_author
            .entry(bundle.meta.passages[i].author_id)
            .or_default()
            .push(i);
    }

    let mut preds = Vec::new();
    let (mut hits_seen, mut hits_unseen) = (0u32, 0u32);
    for (author, idxs) in by_author {
        // Centroids with this author's ENTIRE body of work removed.
        let held: HashSet<usize> = (0..bundle.len())
            .filter(|&i| bundle.meta.passages[i].author_id == author)
            .collect();
        let unseen_cents = centroids_excluding(bundle, &held);
        for i in idxs {
            let v = bundle.vector(i);
            let pred_seen = full.rank(v).first().map(|&(a, _)| a).unwrap_or(author);
            let pred_unseen = argmax_centroid(&unseen_cents, v).unwrap_or(author);
            if pred_seen == author {
                hits_seen += 1;
            }
            if pred_unseen == author {
                hits_unseen += 1;
            }
            preds.push(Prediction {
                passage_idx: i,
                true_author: author,
                pred_seen,
                pred_unseen,
            });
        }
    }
    (preds, hits_seen, hits_unseen, test.len() as u32)
}

/// Summed per-class training vectors (non-mystery), optionally excluding one author.
fn class_sums(
    bundle: &Bundle,
    pclass: &[usize],
    k: usize,
    exclude_author: Option<usize>,
) -> (Vec<Vec<f32>>, Vec<usize>) {
    let dim = bundle.meta.dim;
    let mut sums = vec![vec![0f32; dim]; k];
    let mut cnts = vec![0usize; k];
    for (i, p) in bundle.meta.passages.iter().enumerate() {
        if p.is_mystery || Some(p.author_id) == exclude_author {
            continue;
        }
        let c = pclass[i];
        let v = bundle.vector(i);
        for d in 0..dim {
            sums[c][d] += v[d];
        }
        cnts[c] += 1;
    }
    (sums, cnts)
}

/// Recover an author trait from writing style. `get` maps an author to a class string.
fn compute_attribute(bundle: &Bundle, name: &str, get: impl Fn(&AuthorMeta) -> &str) -> AttrResult {
    let authors = &bundle.meta.authors;
    let n = authors.len();
    let dim = bundle.meta.dim;

    // Assign a dense class index per author (first-seen order).
    let mut classes: Vec<String> = Vec::new();
    let author_class: Vec<usize> = authors
        .iter()
        .map(|a| {
            let v = get(a);
            match classes.iter().position(|c| c == v) {
                Some(i) => i,
                None => {
                    classes.push(v.to_string());
                    classes.len() - 1
                }
            }
        })
        .collect();
    let k = classes.len().max(1);
    let palette = ["#e6194b", "#4363d8", "#3cb44b", "#f58231", "#911eb4", "#008080"];
    let colors: Vec<String> = (0..k).map(|i| palette[i % palette.len()].to_string()).collect();
    let pclass: Vec<usize> = bundle
        .meta
        .passages
        .iter()
        .map(|p| author_class[p.author_id])
        .collect();

    // Majority-class baseline (by passage count).
    let mut counts = vec![0u32; k];
    for (i, p) in bundle.meta.passages.iter().enumerate() {
        if !p.is_mystery {
            counts[pclass[i]] += 1;
        }
    }
    let total: u32 = counts.iter().sum();
    let baseline = *counts.iter().max().unwrap_or(&0) as f32 / total.max(1) as f32;

    // Leaky (has read the author): leave-one-passage-out over class centroids.
    let (sums, cnts) = class_sums(bundle, &pclass, k, None);
    let (mut leaky_c, mut leaky_t) = (0u32, 0u32);
    for (i, p) in bundle.meta.passages.iter().enumerate() {
        if p.is_mystery {
            continue;
        }
        let c = pclass[i];
        let v = bundle.vector(i);
        let cents: Vec<Option<Vec<f32>>> = (0..k)
            .map(|cc| {
                if cc == c {
                    let adj: Vec<f32> = (0..dim).map(|d| sums[c][d] - v[d]).collect();
                    centroid_of_sum(&adj, cnts[c].saturating_sub(1), dim)
                } else {
                    centroid_of_sum(&sums[cc], cnts[cc], dim)
                }
            })
            .collect();
        if let Some(pred) = argmax_centroid(&cents, v) {
            leaky_t += 1;
            if pred == c {
                leaky_c += 1;
            }
        }
    }
    let leaky_accuracy = leaky_c as f32 / leaky_t.max(1) as f32;

    // Fair (never read the author): leave-one-author-out over class centroids.
    let mut confusion = vec![vec![0u32; k]; k];
    let mut per_class = vec![(0u32, 0u32); k];
    let (mut fair_c, mut fair_t) = (0u32, 0u32);
    let mut tested = 0usize;
    for a in 0..n {
        let true_c = author_class[a];
        let (s2, c2) = class_sums(bundle, &pclass, k, Some(a));
        let cents: Vec<Option<Vec<f32>>> =
            (0..k).map(|cc| centroid_of_sum(&s2[cc], c2[cc], dim)).collect();
        if cents[true_c].is_none() {
            continue; // this class has no other author to learn from
        }
        tested += 1;
        for (i, p) in bundle.meta.passages.iter().enumerate() {
            if p.is_mystery || p.author_id != a {
                continue;
            }
            if let Some(pred) = argmax_centroid(&cents, bundle.vector(i)) {
                confusion[true_c][pred] += 1;
                per_class[true_c].1 += 1;
                fair_t += 1;
                if pred == true_c {
                    fair_c += 1;
                    per_class[true_c].0 += 1;
                }
            }
        }
    }
    let fair_accuracy = fair_c as f32 / fair_t.max(1) as f32;

    AttrResult {
        name: name.to_string(),
        classes,
        colors,
        baseline,
        random_baseline: 1.0 / k as f32,
        fair_accuracy,
        leaky_accuracy,
        per_class,
        confusion,
        tested_authors: tested,
        total_authors: n,
    }
}

/// The familiarity gradient: accuracy as a function of how many passages **per author**
/// the model is allowed to learn from. Uses a fixed held-out test split (every 5th
/// passage per author) and grows the training pool. Returns `(passages_per_author,
/// accuracy)` — a learning curve that climbs the more the model has read.
pub fn familiarity_curve(bundle: &Bundle, caps: &[usize]) -> Vec<(usize, f32)> {
    let n = bundle.n_authors();
    let mut test: Vec<usize> = Vec::new();
    let mut pool: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut local = vec![0usize; n];
    for (i, p) in bundle.meta.passages.iter().enumerate() {
        let a = p.author_id;
        if local[a] % 5 == 0 {
            test.push(i);
        } else {
            pool[a].push(i);
        }
        local[a] += 1;
    }

    caps.iter()
        .map(|&cap| {
            let cents: Vec<Option<Vec<f32>>> = (0..n)
                .map(|a| {
                    let sample = even_sample(&pool[a], cap);
                    let dim = bundle.meta.dim;
                    let mut sum = vec![0f32; dim];
                    for &i in &sample {
                        let v = bundle.vector(i);
                        for k in 0..dim {
                            sum[k] += v[k];
                        }
                    }
                    centroid_of_sum(&sum, sample.len(), dim)
                })
                .collect();
            let mut correct = 0u32;
            let mut total = 0u32;
            for &i in &test {
                let a = bundle.meta.passages[i].author_id;
                if let Some(pred) = argmax_centroid(&cents, bundle.vector(i)) {
                    total += 1;
                    if pred == a {
                        correct += 1;
                    }
                }
            }
            (cap, correct as f32 / total.max(1) as f32)
        })
        .collect()
}

fn even_sample(items: &[usize], k: usize) -> Vec<usize> {
    if items.len() <= k {
        return items.to_vec();
    }
    let step = items.len() as f32 / k as f32;
    (0..k).map(|i| items[(i as f32 * step) as usize]).collect()
}
