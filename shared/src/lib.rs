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
    /// Country where they grew up / learned to read and write.
    #[serde(default)]
    pub raised: String,
    /// Country of their main education.
    #[serde(default)]
    pub educated: String,
    /// Specific college/university, or "None" if self-taught.
    #[serde(default)]
    pub college: String,
    /// Generic recoverable traits (name → value). Used by the code dataset for
    /// language / era / ecosystem / role; when present they drive the Dimensions
    /// view, otherwise the fixed fields above are used. Skipped when empty so the
    /// prose bundle serializes unchanged.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub traits: Vec<Trait>,
}

/// A generic recoverable attribute of an author/coder (e.g. "Primary language" → "C").
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Trait {
    pub name: String,
    pub value: String,
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
    /// Unix timestamp (seconds) the code was authored — code dataset only; 0 = unknown
    /// (prose). Drives the "when was this written?" / "weekend?" style dimensions.
    #[serde(default)]
    pub authored: i64,
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

/// One dataset lens on the site: prose authors, or code "coders".
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DatasetInfo {
    /// Filename key: `person2vec-<key>.{json,bin}`.
    pub key: String,
    pub label: String,
    /// Short subtitle shown under the toggle, e.g. "prose" / "code".
    pub blurb: String,
}

/// Lists the datasets the site offers; loaded first by the web app.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DatasetManifest {
    pub default: String,
    pub datasets: Vec<DatasetInfo>,
}

/// One rung of the familiarity ladder (how much the model has read of the author).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Rung {
    pub label: String,
    pub sublabel: String,
    pub accuracy: f32,
}

/// Display strings for the five familiarity rungs, supplied per dataset so the same
/// `compute_results` machinery can narrate prose (book/series) or code (commit/repo).
#[derive(Clone, Debug)]
pub struct LadderLabels {
    /// Each is (label, sublabel). Bottom → top of the ladder.
    pub blind: (&'static str, &'static str),
    pub author_out: (&'static str, &'static str),
    pub series_out: (&'static str, &'static str),
    pub book_out: (&'static str, &'static str),
    pub seen: (&'static str, &'static str),
}

impl LadderLabels {
    /// Original prose labels: hold out books / series / the author.
    pub fn prose() -> Self {
        Self {
            blind: ("Blind guessing", "no information"),
            author_out: ("Never read the author", "held out every work by them"),
            series_out: ("Never read this series", "has read the author's other work"),
            book_out: ("Never read this book", "has read the rest of the series"),
            seen: ("Has read the book", "has seen other passages from it"),
        }
    }
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

/// Precompute the whole familiarity ladder + supporting panels for the accuracy view,
/// using the default prose rung labels.
pub fn compute_results(bundle: &Bundle) -> Results {
    compute_results_with_labels(bundle, &LadderLabels::prose())
}

/// Like [`compute_results`] but with dataset-specific rung labels (prose vs code).
pub fn compute_results_with_labels(bundle: &Bundle, labels: &LadderLabels) -> Results {
    let n = bundle.n_authors();
    let baseline = 1.0 / n as f32;
    let seen = loocv_nearest_centroid(bundle);
    let book_out = loocv_leave_book_out(bundle);
    let series_out = loocv_leave_series_out(bundle);
    let author_out = loocv_leave_author_out(bundle);
    let curve = familiarity_curve(bundle, &[2, 4, 8, 16, 32, 64, 128]);

    let rungs = vec![
        Rung { label: labels.blind.0.into(), sublabel: labels.blind.1.into(), accuracy: baseline },
        Rung { label: labels.author_out.0.into(), sublabel: labels.author_out.1.into(), accuracy: author_out.accuracy },
        Rung { label: labels.series_out.0.into(), sublabel: labels.series_out.1.into(), accuracy: series_out.accuracy },
        Rung { label: labels.book_out.0.into(), sublabel: labels.book_out.1.into(), accuracy: book_out.accuracy },
        Rung { label: labels.seen.0.into(), sublabel: labels.seen.1.into(), accuracy: seen.accuracy },
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
    let attributes = compute_attributes(bundle);

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

    // Represent each author by the mean of their (non-mystery) passage vectors: one
    // style point per author. All counts below are AUTHORS, not passages.
    let mut asum = vec![vec![0f32; dim]; n];
    let mut acnt = vec![0usize; n];
    for (i, p) in bundle.meta.passages.iter().enumerate() {
        if p.is_mystery {
            continue;
        }
        let v = bundle.vector(i);
        for d in 0..dim {
            asum[p.author_id][d] += v[d];
        }
        acnt[p.author_id] += 1;
    }
    let amean: Vec<Option<Vec<f32>>> =
        (0..n).map(|a| centroid_of_sum(&asum[a], acnt[a], dim)).collect();

    // Baselines over the *testable* authors (classes with >=2 authors), matching the
    // fair-accuracy population: majority-class share, and blind 1/(# testable classes).
    let mut class_authors = vec![0u32; k];
    for a in 0..n {
        class_authors[author_class[a]] += 1;
    }
    let n_testable_classes = class_authors.iter().filter(|&&c| c >= 2).count().max(1);
    let mut tested_counts = vec![0u32; k];
    for a in 0..n {
        if class_authors[author_class[a]] >= 2 {
            tested_counts[author_class[a]] += 1;
        }
    }
    let n_tested: u32 = tested_counts.iter().sum();
    let baseline = *tested_counts.iter().max().unwrap_or(&0) as f32 / n_tested.max(1) as f32;
    let random_baseline = 1.0 / n_testable_classes as f32;

    // Leaky (has read the author): the class profile includes this author.
    let (fsum, fcnt) = class_sums(bundle, &pclass, k, None);
    let full_cents: Vec<Option<Vec<f32>>> =
        (0..k).map(|c| centroid_of_sum(&fsum[c], fcnt[c], dim)).collect();
    let (mut leaky_c, mut leaky_t) = (0u32, 0u32);
    for a in 0..n {
        if let Some(m) = &amean[a] {
            if let Some(pred) = argmax_centroid(&full_cents, m) {
                leaky_t += 1;
                if pred == author_class[a] {
                    leaky_c += 1;
                }
            }
        }
    }
    let leaky_accuracy = leaky_c as f32 / leaky_t.max(1) as f32;

    // Fair (never read the author): class profile built only from the OTHER authors,
    // then one prediction per author. Classes with a single author aren't testable.
    let mut confusion = vec![vec![0u32; k]; k];
    let mut per_class = vec![(0u32, 0u32); k];
    let (mut fair_c, mut fair_t) = (0u32, 0u32);
    for a in 0..n {
        let true_c = author_class[a];
        let Some(m) = &amean[a] else { continue };
        if class_authors[true_c] < 2 {
            continue;
        }
        let (s2, c2) = class_sums(bundle, &pclass, k, Some(a));
        let cents: Vec<Option<Vec<f32>>> =
            (0..k).map(|cc| centroid_of_sum(&s2[cc], c2[cc], dim)).collect();
        if let Some(pred) = argmax_centroid(&cents, m) {
            confusion[true_c][pred] += 1;
            per_class[true_c].1 += 1;
            fair_t += 1;
            if pred == true_c {
                fair_c += 1;
                per_class[true_c].0 += 1;
            }
        }
    }
    let fair_accuracy = fair_c as f32 / fair_t.max(1) as f32;
    let tested = fair_t as usize;

    AttrResult {
        name: name.to_string(),
        classes,
        colors,
        baseline,
        random_baseline,
        fair_accuracy,
        leaky_accuracy,
        per_class,
        confusion,
        tested_authors: tested,
        total_authors: n,
    }
}

const ERA_LABELS: [&str; 4] = ["≤ 2009", "2010–2015", "2016–2020", "2021+"];

/// Year (civil calendar) from a Unix timestamp — Howard Hinnant's days→civil algorithm.
fn year_from_unix(ts: i64) -> i64 {
    let z = ts.div_euclid(86_400) + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    yoe + era * 400 + if m <= 2 { 1 } else { 0 }
}

fn era_bucket(ts: i64) -> Option<usize> {
    if ts <= 0 {
        return None;
    }
    Some(match year_from_unix(ts) {
        y if y <= 2009 => 0,
        y if y <= 2015 => 1,
        y if y <= 2020 => 2,
        _ => 3,
    })
}

/// Weekend? 1970-01-01 was a Thursday; 0=Sun … 6=Sat.
fn weekend_bucket(ts: i64) -> Option<usize> {
    if ts <= 0 {
        return None;
    }
    let dow = (ts.div_euclid(86_400) + 4).rem_euclid(7);
    Some(if dow == 0 || dow == 6 { 1 } else { 0 })
}

/// Build the trait-recovery experiments. For code (passages carry timestamps) we add
/// per-passage style dimensions (when was it written? weekend?) recovered with
/// leave-one-author-out. Per-author traits come from `AuthorMeta.traits` (code) or the
/// fixed prose fields (gender, country, …).
fn compute_attributes(bundle: &Bundle) -> Vec<AttrResult> {
    let mut out: Vec<AttrResult> = Vec::new();

    // Per-passage temporal dimensions (code dataset only).
    if bundle.meta.passages.iter().any(|p| p.authored > 0) {
        out.push(compute_passage_attribute(bundle, "When was it written?", &ERA_LABELS, era_bucket));
        out.push(compute_passage_attribute(
            bundle,
            "Weekend or weekday?",
            &["Weekday", "Weekend"],
            weekend_bucket,
        ));
    }

    // Per-author traits.
    let has_traits = bundle.meta.authors.iter().any(|a| !a.traits.is_empty());
    if has_traits {
        let mut names: Vec<String> = Vec::new();
        for a in &bundle.meta.authors {
            for t in &a.traits {
                if !names.iter().any(|nm| nm == &t.name) {
                    names.push(t.name.clone());
                }
            }
        }
        for name in &names {
            out.push(compute_attribute(bundle, name, move |a| {
                a.traits
                    .iter()
                    .find(|t| &t.name == name)
                    .map(|t| t.value.as_str())
                    .unwrap_or("")
            }));
        }
    } else {
        out.push(compute_attribute(bundle, "Gender", |a| a.gender.as_str()));
        out.push(compute_attribute(bundle, "Birth country", |a| a.birth_country.as_str()));
        out.push(compute_attribute(bundle, "Where they were raised", |a| a.raised.as_str()));
        out.push(compute_attribute(bundle, "Where they were educated", |a| a.educated.as_str()));
        out.push(compute_attribute(bundle, "College", |a| a.college.as_str()));
    }
    out
}

/// Recover a PER-PASSAGE attribute (era, weekend) from code style. Uses leave-one-author-out:
/// class centroids are built from OTHER authors' passages, so a correct guess means the
/// STYLE encodes the attribute across people — not that it recognized the author.
fn compute_passage_attribute(
    bundle: &Bundle,
    name: &str,
    class_labels: &[&str],
    class_of: impl Fn(i64) -> Option<usize>,
) -> AttrResult {
    let dim = bundle.meta.dim;
    let n = bundle.n_authors();
    let k = class_labels.len().max(1);
    let palette = ["#e6194b", "#4363d8", "#3cb44b", "#f58231", "#911eb4", "#008080"];
    let colors: Vec<String> = (0..k).map(|i| palette[i % palette.len()].to_string()).collect();
    let classes: Vec<String> = class_labels.iter().map(|s| s.to_string()).collect();

    // Class of each non-mystery passage (None = not classifiable / prose).
    let pclass: Vec<Option<usize>> = bundle
        .meta
        .passages
        .iter()
        .map(|p| if p.is_mystery { None } else { class_of(p.authored) })
        .collect();

    let centroids = |exclude: Option<usize>| -> Vec<Option<Vec<f32>>> {
        let mut sums = vec![vec![0f32; dim]; k];
        let mut cnts = vec![0usize; k];
        for (i, p) in bundle.meta.passages.iter().enumerate() {
            if Some(p.author_id) == exclude {
                continue;
            }
            let Some(c) = pclass[i] else { continue };
            let v = bundle.vector(i);
            for d in 0..dim {
                sums[c][d] += v[d];
            }
            cnts[c] += 1;
        }
        (0..k).map(|c| centroid_of_sum(&sums[c], cnts[c], dim)).collect()
    };

    // Baselines over the classified passages.
    let mut class_counts = vec![0u32; k];
    for c in pclass.iter().flatten() {
        class_counts[*c] += 1;
    }
    let total: u32 = class_counts.iter().sum();
    let baseline = *class_counts.iter().max().unwrap_or(&0) as f32 / total.max(1) as f32;
    let random_baseline = 1.0 / class_counts.iter().filter(|&&c| c > 0).count().max(1) as f32;

    // Leaky: train on all passages (includes the author's own).
    let full = centroids(None);
    let (mut leaky_c, mut leaky_t) = (0u32, 0u32);
    for (i, _) in bundle.meta.passages.iter().enumerate() {
        let Some(true_c) = pclass[i] else { continue };
        if let Some(pred) = argmax_centroid(&full, bundle.vector(i)) {
            leaky_t += 1;
            if pred == true_c {
                leaky_c += 1;
            }
        }
    }
    let leaky_accuracy = leaky_c as f32 / leaky_t.max(1) as f32;

    // Fair: leave-one-author-out.
    let mut confusion = vec![vec![0u32; k]; k];
    let mut per_class = vec![(0u32, 0u32); k];
    let (mut fair_c, mut fair_t) = (0u32, 0u32);
    let mut tested_authors = 0usize;
    for a in 0..n {
        let cents = centroids(Some(a));
        let mut tested = false;
        for (i, p) in bundle.meta.passages.iter().enumerate() {
            if p.author_id != a {
                continue;
            }
            let Some(true_c) = pclass[i] else { continue };
            if let Some(pred) = argmax_centroid(&cents, bundle.vector(i)) {
                tested = true;
                confusion[true_c][pred] += 1;
                per_class[true_c].1 += 1;
                fair_t += 1;
                if pred == true_c {
                    fair_c += 1;
                    per_class[true_c].0 += 1;
                }
            }
        }
        if tested {
            tested_authors += 1;
        }
    }
    let fair_accuracy = fair_c as f32 / fair_t.max(1) as f32;

    AttrResult {
        name: name.to_string(),
        classes,
        colors,
        baseline,
        random_baseline,
        fair_accuracy,
        leaky_accuracy,
        per_class,
        confusion,
        tested_authors,
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

// ---------------------------------------------------------------------------
// J-lens bundle — precomputed averaged-Jacobian interpretability readouts for the
// `/jlens` view. Everything the browser shows is computed offline by the `jlens`
// crate; the viewer does no linear algebra. Shipped as `person2vec-jlens-minilm.json`.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct JlensBundle {
    pub model: JlensModel,
    /// Style-axis names; index is the axis id used by [`JlensStyleTraj::axis`].
    pub axes: Vec<String>,
    /// Source depths analyzed, e.g. `[0,1,2,3,4,5]`.
    pub layers: Vec<usize>,
    pub structural: JlensStructural,
    pub examples: Vec<JlensExample>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct JlensModel {
    pub name: String,
    pub layers: usize,
    pub d_model: usize,
    pub vocab: usize,
    pub max_len: usize,
    /// Provenance of the pseudo-unembedding, e.g. "tied word embeddings (logit lens)".
    pub w_u_source: String,
    /// How many passages the averaged Jacobians were built from.
    pub passages: usize,
}

/// Depth-wise structural signatures. Each vector is indexed by source layer.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct JlensStructural {
    pub stable_rank: Vec<f32>,
    pub effective_dim: Vec<f32>,
    /// Mean excess kurtosis of the vocab-lens readout (peakier ⇒ more verbalizable).
    pub verbalizability: Vec<f32>,
    /// Linear-CKA of J-lens geometry between layers (`layers × layers`).
    pub cka: Vec<Vec<f32>>,
    /// Lag-1 readout autocorrelation vs. a position-shuffled null (the paper's 4th
    /// Figure-28 signature). Positive ⇒ readout persists across the sequence.
    #[serde(default)]
    pub autocorrelation: Vec<f32>,
}

/// One curated passage, with its per-(layer, position) token readouts and per-axis
/// style trajectory through depth.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct JlensExample {
    pub passage_idx: usize,
    pub author: String,
    pub color: String,
    pub snippet: String,
    /// WordPiece tokens (position labels for the grid).
    pub tokens: Vec<String>,
    pub cells: Vec<JlensCell>,
    pub style: Vec<JlensStyleTraj>,
    /// Rank-vs-depth trajectory for a handful of top concepts (the paper's key figure).
    #[serde(default)]
    pub concepts: Vec<JlensConcept>,
    /// Sparse J-space decomposition of the activation at a representative layer.
    #[serde(default)]
    pub jspace: JlensJspace,
}

/// A concept token and its vocab-lens rank at each layer (1 = surfaced strongest).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct JlensConcept {
    pub tok: String,
    pub per_layer_rank: Vec<u32>,
}

/// J-space: the sparse non-negative set of J-lens (token) directions that reconstruct
/// an activation, and the fraction of its variance they capture.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct JlensJspace {
    pub layer: usize,
    pub captured_variance: f32,
    /// Token directions with their non-negative coefficients (score = coeff).
    pub items: Vec<JlensTok>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct JlensCell {
    pub layer: usize,
    pub pos: usize,
    /// The input token at this position (for the cell label).
    pub token: String,
    pub top: Vec<JlensTok>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct JlensTok {
    pub tok: String,
    pub score: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct JlensStyleTraj {
    pub axis: usize,
    /// Per-layer style-axis loading (length == `layers`).
    pub per_layer: Vec<f32>,
}

/// Fingerprint-presence detector: precomputed identity centroids + sample embeddings, so
/// the browser can compute "known fingerprint vs blank space" live for a chosen sample,
/// and show the tokens each identity's style direction verbalizes. Written by `bin/fingerprint`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FingerprintBundle {
    pub subject: String,
    pub threshold: f32,
    pub authors: Vec<FpAuthor>,
    pub probes: Vec<FpProbe>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FpAuthor {
    pub name: String,
    pub color: String,
    /// L2-normalized identity centroid in the output embedding space.
    pub centroid: Vec<f32>,
    /// Top tokens the identity's style direction aligns with (embedding logit lens).
    pub tokens: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FpProbe {
    pub label: String,
    pub snippet: String,
    /// L2-normalized embedding of the sample.
    pub vec: Vec<f32>,
}

/// Nearest identity by cosine (L2-normalized inputs) → `(author index, cosine)`. The
/// detector's core: if the cosine clears the threshold the fingerprint is "known",
/// otherwise the sample is in "blank space".
pub fn nearest_centroid(v: &[f32], authors: &[FpAuthor]) -> (usize, f32) {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, a) in authors.iter().enumerate() {
        let s: f32 = v.iter().zip(&a.centroid).map(|(x, y)| x * y).sum();
        if s > best.1 {
            best = (i, s);
        }
    }
    best
}

#[cfg(test)]
mod fp_tests {
    use super::*;
    #[test]
    fn nearest_centroid_picks_closest() {
        let authors = vec![
            FpAuthor { name: "A".into(), centroid: vec![1.0, 0.0], ..Default::default() },
            FpAuthor { name: "B".into(), centroid: vec![0.0, 1.0], ..Default::default() },
        ];
        assert_eq!(nearest_centroid(&[0.9, 0.1], &authors), (0, 0.9));
        assert_eq!(nearest_centroid(&[0.1, 0.9], &authors).0, 1);
    }
}

/// "Does the model learn who wrote it?" — nearest-identity accuracy decoded from the
/// *internal* J-lens readout at each layer, versus the output embedding and chance.
/// Written by `bin/steer`; shows identity lives inside the Jacobians, not just the output.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct IdentityBundle {
    pub model: String,
    /// Noun for one identity: "author" / "coder".
    pub subject: String,
    pub identities: usize,
    pub chance: f32,
    pub output_acc: f32,
    /// Per-layer accuracy decoding identity from the internal Jacobian readout.
    pub per_layer: Vec<f32>,
}
