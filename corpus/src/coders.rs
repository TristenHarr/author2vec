//! Code-authorship dataset: attribute source code to the developer who wrote it via
//! git history, then chunk / scrub / dedup it into the same `AuthorMeta` + `Passage`
//! shape the prose pipeline produces. Shared by `bin/coders.rs` (ship) and
//! `bin/coders-validate.rs` (honesty audit, incl. the scrub ablation).
//!
//! The commit is the unit: we walk each repo's non-merge commits, take the ADDED
//! lines from each commit's diff, and attribute them to that commit's (mailmap-
//! resolved) author. Each passage carries `book_title = "<short-sha> <subject>"` and
//! `series = <repo slug>`, so the shared familiarity ladder becomes
//! coder → repo → commit.

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use regex::Regex;
use serde::Deserialize;
use shared::{AuthorMeta, LadderLabels, Passage, Trait};

use crate::{author_color, manifest_dir, mark_mystery, sample_even};

/// Target whitespace-tokens per code passage (smaller than prose: code sub-word
/// tokenizes densely and MiniLM caps at 512 sub-words).
const CODE_PASSAGE_TOKENS: usize = 160;
const MIN_PASSAGE_TOKENS: usize = 60;
const MAX_PER_DEV: usize = 130;
const MYSTERY_PER_DEV: usize = 4;
/// Skip a commit whose kept added-line count exceeds this (vendored / generated /
/// initial-import dumps). Generous so a founder's big hand-written commits still count;
/// truly machine-sized dumps are already caught by `keep_file`.
const MAX_COMMIT_LINES: usize = 6000;
/// Drop a passage if this fraction of its lines have already been seen anywhere
/// (kills templated boilerplate that would inflate every rung).
const NEAR_DUP_OVERLAP: f64 = 0.8;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct Config {
    #[serde(default)]
    languages: Vec<String>,
    devs: Vec<DevCfg>,
}

#[derive(Deserialize)]
struct DevCfg {
    name: String,
    #[serde(default)]
    color: Option<String>,
    #[serde(default)]
    emails: Vec<String>,
    #[serde(default)]
    logins: Vec<String>,
    #[serde(default)]
    traits: TraitsCfg,
    repos: Vec<RepoCfg>,
}

#[derive(Deserialize, Default)]
struct TraitsCfg {
    /// "Systems" or "Scripting" — the one per-dev dimension that's testable at this
    /// roster size (per-passage era/weekend come from commit timestamps instead).
    #[serde(default)]
    paradigm: String,
}

#[derive(Deserialize, Clone)]
struct RepoCfg {
    slug: String,
    url: String,
    #[serde(default)]
    domain: String,
    #[serde(default)]
    paths: Vec<String>,
    #[serde(default)]
    exclude: Vec<String>,
    #[serde(default)]
    depth: Option<usize>,
    #[serde(default)]
    max_commits: Option<usize>,
}

fn load_config(path: &Path) -> Config {
    let text = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    toml::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

pub fn code_ladder_labels() -> LadderLabels {
    LadderLabels {
        blind: ("Blind guessing", "no information"),
        author_out: ("Never seen this coder", "held out every line they authored"),
        series_out: ("Never seen this repo", "has read their code in other repos"),
        book_out: ("Never seen this file", "has read the coder's other files in this repo"),
        seen: ("Read this file", "has seen other lines from the same file"),
    }
}

// ---------------------------------------------------------------------------
// Identity resolution (layered on each repo's .mailmap, applied by git itself)
// ---------------------------------------------------------------------------

struct IdentityMap {
    by_email: HashMap<String, usize>,
    by_login: HashMap<String, usize>,
    by_name: HashMap<String, usize>,
}

impl IdentityMap {
    fn build(devs: &[DevCfg]) -> Self {
        let mut by_email = HashMap::new();
        let mut by_login = HashMap::new();
        let mut by_name = HashMap::new();
        for (i, d) in devs.iter().enumerate() {
            for e in &d.emails {
                by_email.insert(e.trim().to_lowercase(), i);
            }
            for l in &d.logins {
                by_login.insert(l.trim().to_lowercase(), i);
            }
            by_name.insert(d.name.trim().to_lowercase(), i);
        }
        Self { by_email, by_login, by_name }
    }

    /// Resolve a commit's (name, email) to a roster dev, or None.
    fn resolve(&self, name: &str, email: &str) -> Option<usize> {
        let e = email.trim().to_lowercase();
        if let Some(&i) = self.by_email.get(&e) {
            return Some(i);
        }
        if let Some(login) = github_login(&e) {
            if let Some(&i) = self.by_login.get(&login) {
                return Some(i);
            }
        }
        let n = name.trim().to_lowercase();
        if let Some(&i) = self.by_name.get(&n) {
            return Some(i);
        }
        self.by_login.get(&n).copied()
    }
}

/// Extract the login from a GitHub noreply address `12345+login@users.noreply.github.com`
/// (or the older `login@users.noreply.github.com`).
fn github_login(email: &str) -> Option<String> {
    let local = email.strip_suffix("@users.noreply.github.com")?;
    let login = local.rsplit('+').next().unwrap_or(local);
    if login.is_empty() {
        None
    } else {
        Some(login.to_lowercase())
    }
}

fn is_bot(name: &str, email: &str) -> bool {
    let n = name.to_lowercase();
    let e = email.to_lowercase();
    const PAT: &[&str] = &[
        "[bot]", "dependabot", "renovate", "greenkeeper", "snyk-bot", "github-actions",
        "semantic-release", "allcontributors", "mergify", "actions-user", "bors",
    ];
    if PAT.iter().any(|p| n.contains(p) || e.contains(p)) {
        return true;
    }
    e == "noreply@github.com" || e.ends_with("web-flow@users.noreply.github.com")
}

// ---------------------------------------------------------------------------
// Identity scrubbing (so we embed style, not names)
// ---------------------------------------------------------------------------

struct Scrubber {
    email: Regex,
    url: Regex,
    ident: Option<Regex>,
    /// Distinctive lowercased surname/login substrings (len >= 6) for the post-scrub
    /// safety net: any passage still containing one is dropped, so no name can survive
    /// inside an identifier the word-boundary scrub missed.
    residual: Vec<String>,
}

impl Scrubber {
    fn build(devs: &[DevCfg]) -> Self {
        let email = Regex::new(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}").unwrap();
        let url = Regex::new(r#"https?://[^\s'"<>)]+"#).unwrap();
        let mut toks: Vec<String> = Vec::new();
        let mut residual: Vec<String> = Vec::new();
        for d in devs {
            for l in &d.logins {
                if l.len() >= 3 {
                    toks.push(regex::escape(l));
                }
                if l.len() >= 6 {
                    residual.push(l.to_lowercase());
                }
            }
            // Surname / distinctive name parts (>=4 chars avoids nuking common words).
            for part in d.name.split_whitespace() {
                if part.len() >= 4 {
                    toks.push(regex::escape(part));
                }
                if part.len() >= 6 {
                    residual.push(part.to_lowercase());
                }
            }
            toks.push(regex::escape(d.name.trim()));
        }
        toks.sort();
        toks.dedup();
        residual.sort();
        residual.dedup();
        let ident = if toks.is_empty() {
            None
        } else {
            Some(Regex::new(&format!(r"(?i)\b({})\b", toks.join("|"))).unwrap())
        };
        Self { email, url, ident, residual }
    }

    fn scrub(&self, text: &str) -> String {
        let t = self.email.replace_all(text, "EMAIL");
        let t = self.url.replace_all(&t, "URL");
        match &self.ident {
            Some(re) => re.replace_all(&t, "NAME").into_owned(),
            None => t.into_owned(),
        }
    }

    /// True if a roster identity substring survived scrubbing (drop the passage).
    fn leaks(&self, text: &str) -> bool {
        let low = text.to_lowercase();
        self.residual.iter().any(|t| low.contains(t.as_str()))
    }
}

// ---------------------------------------------------------------------------
// Cloning + history traversal (shell out to git; .mailmap applied for free)
// ---------------------------------------------------------------------------

fn clones_dir() -> PathBuf {
    manifest_dir().join(".clones")
}

fn ensure_clone(repo: &RepoCfg) -> Option<PathBuf> {
    let dir = clones_dir();
    let _ = fs::create_dir_all(&dir);
    let path = dir.join(&repo.slug);
    if path.join(".git").exists() {
        return Some(path);
    }
    println!("  cloning {} …", repo.url);
    let mut args: Vec<String> = vec!["clone".into(), "--single-branch".into(), "--no-tags".into()];
    if let Some(d) = repo.depth {
        args.push("--depth".into());
        args.push(d.to_string());
    }
    args.push(repo.url.clone());
    args.push(path.to_string_lossy().into_owned());
    let ok = Command::new("git")
        .args(&args)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if ok {
        Some(path)
    } else {
        eprintln!("  !! clone failed: {}", repo.url);
        None
    }
}

fn log_args(path: &Path, repo: &RepoCfg, patterns: &[String]) -> Vec<String> {
    let mut args: Vec<String> = vec![
        "-C".into(),
        path.to_string_lossy().into_owned(),
        "log".into(),
        "--no-merges".into(),
        "--no-color".into(),
        "-M".into(),
        "-C".into(),
        "--format=%x01%H%x1f%an%x1f%ae%x1f%ct%x1f%s".into(),
        "-p".into(),
    ];
    // Ask git for EXACTLY this developer's commits (matched against "Name <email>").
    // git ORs multiple --author flags; a loose regex match is caught later by an
    // identity double-check. This reaches all their history — old and new — without
    // walking (or capping) the whole repo.
    for p in patterns {
        args.push(format!("--author={p}"));
    }
    // Only cap commits when the repo explicitly asks (the giant repos).
    if let Some(mc) = repo.max_commits {
        args.push("-n".into());
        args.push(mc.to_string());
    }
    if !repo.paths.is_empty() {
        args.push("--".into());
        for p in &repo.paths {
            args.push(p.clone());
        }
    }
    args
}

/// Call `f(line)` for every line of `git log -p` output, streamed (never buffers the
/// whole multi-hundred-MB patch). `patterns` are `--author` filters (empty = all).
/// Lines are lossy-decoded so a non-UTF8 diff can't abort the walk.
fn stream_log(path: &Path, repo: &RepoCfg, patterns: &[String], mut f: impl FnMut(&str)) {
    let mut child = match Command::new("git")
        .args(&log_args(path, repo, patterns))
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  !! git log failed for {}: {e}", repo.slug);
            return;
        }
    };
    let out = child.stdout.take().unwrap();
    let reader = BufReader::new(out);
    for chunk in reader.split(b'\n') {
        match chunk {
            Ok(bytes) => {
                let line = String::from_utf8_lossy(&bytes);
                f(&line);
            }
            Err(_) => break,
        }
    }
    let _ = child.wait();
}

// ---------------------------------------------------------------------------
// Attribution + chunking + dedup
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Collected {
    book_title: String,
    series: String,
    text: String,
    authored: i64,
}

/// Per-commit buffer of ADDED lines grouped by file, plus the resolved author + time.
struct CommitCtx {
    dev: Option<usize>,
    blocks: Vec<(String, Vec<String>)>,
    added: usize,
    time: i64,
}

struct DedupState {
    /// Exact-duplicate passages are dropped globally (kills vendored/copied code).
    seen_exact: HashSet<u64>,
    /// Near-duplicate shingles are tracked PER-CODER, so a coder processed late in the
    /// roster isn't penalized for lines an earlier coder happened to write. Global
    /// tracking here silently starved the last few coders of passages.
    per_dev_shingles: Vec<HashSet<u64>>,
}

fn hash_str(s: &str) -> u64 {
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

fn norm_line(l: &str) -> String {
    l.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn shingles(text: &str) -> HashSet<u64> {
    let mut s = HashSet::new();
    for line in text.lines() {
        let n = norm_line(line);
        if n.len() >= 4 && n.chars().any(|c| c.is_alphanumeric()) {
            s.insert(hash_str(&n));
        }
    }
    s
}

fn is_trivial(text: &str) -> bool {
    if text.trim().len() < 20 {
        return true;
    }
    let mut distinct: HashSet<&str> = HashSet::new();
    for tok in text.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
        if tok.len() >= 3 && tok.chars().next().map_or(false, |c| c.is_alphabetic() || c == '_') {
            distinct.insert(tok);
        }
    }
    distinct.len() < 5
}

/// Path filter: language allowlist + drop vendored / generated / lock / build files.
fn keep_file(path: &str, langs: &HashSet<String>, exclude: &[String]) -> bool {
    let p = path.to_lowercase();
    const BAD_SEG: &[&str] = &[
        "vendor/", "node_modules/", "third_party/", "third-party/", "deps/", "external/",
        "testdata/", "fixtures/", "/dist/", "/build/", "/target/", "generated/", ".min.",
    ];
    if BAD_SEG.iter().any(|b| p.contains(b)) {
        return false;
    }
    if exclude.iter().any(|e| p.contains(&e.to_lowercase())) {
        return false;
    }
    const BAD_SUFFIX: &[&str] = &[".min.js", ".min.css", ".map", ".pb.go", "_pb2.py"];
    if BAD_SUFFIX.iter().any(|s| p.ends_with(s)) {
        return false;
    }
    const LOCKS: &[&str] = &[
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "cargo.lock", "poetry.lock",
        "composer.lock", "go.sum",
    ];
    if LOCKS.iter().any(|s| p.ends_with(s)) {
        return false;
    }
    if !p.contains('.') {
        return false;
    }
    let ext = p.rsplit('.').next().unwrap_or("");
    langs.contains(ext)
}

/// Extract the new-file path from a `+++ b/<path>` diff line (None for /dev/null).
fn plus_path(line: &str) -> Option<&str> {
    let p = line.strip_prefix("+++ ")?;
    let p = p.strip_prefix("b/").unwrap_or(p);
    if p == "/dev/null" {
        None
    } else {
        Some(p)
    }
}

fn flush_block(ctx: &mut Option<CommitCtx>, cur_file: &Option<String>, cur_block: &mut Vec<String>) {
    if cur_block.is_empty() {
        return;
    }
    let block = std::mem::take(cur_block);
    if let (Some(c), Some(f)) = (ctx.as_mut(), cur_file.as_ref()) {
        if c.dev.is_some() {
            c.added += block.len();
            c.blocks.push((f.clone(), block));
        }
    }
}

/// A finalized commit contributes its ADDED lines to the per-(coder, file) pool, unless
/// it's an oversized import/generated dump. We pool across commits because real
/// maintainers commit in small increments — per-commit windows would starve them.
fn accumulate_commit(ctx: CommitCtx, acc: &mut BTreeMap<String, Vec<(String, i64)>>) {
    if ctx.dev.is_none() {
        return;
    }
    if ctx.added == 0 || ctx.added > MAX_COMMIT_LINES {
        return;
    }
    for (file, lines) in ctx.blocks {
        let v = acc.entry(file).or_default();
        for l in lines {
            v.push((l, ctx.time));
        }
    }
}

fn median_time(times: &mut Vec<i64>) -> i64 {
    if times.is_empty() {
        return 0;
    }
    times.sort_unstable();
    times[times.len() / 2]
}

/// Window one coder's pooled added lines (with authored times) for a single file into
/// passages, tagging each passage with the median authored time of its lines.
fn emit_file(
    dev: usize,
    book_title: &str,
    series: &str,
    lines: &[(String, i64)],
    scrub: Option<&Scrubber>,
    per_dev: &mut [Vec<Collected>],
    dedup: &mut DedupState,
) {
    let mut buf: Vec<&str> = Vec::new();
    let mut times: Vec<i64> = Vec::new();
    let mut toks = 0usize;
    for (l, t) in lines {
        toks += l.split_whitespace().count();
        buf.push(l);
        if *t > 0 {
            times.push(*t);
        }
        if toks >= CODE_PASSAGE_TOKENS {
            keep_passage(&buf.join("\n"), median_time(&mut times), dev, book_title, series, scrub, per_dev, dedup);
            buf.clear();
            times.clear();
            toks = 0;
        }
    }
    if toks >= MIN_PASSAGE_TOKENS {
        keep_passage(&buf.join("\n"), median_time(&mut times), dev, book_title, series, scrub, per_dev, dedup);
    }
}

fn keep_passage(
    raw: &str,
    authored: i64,
    dev: usize,
    book_title: &str,
    series: &str,
    scrub: Option<&Scrubber>,
    per_dev: &mut [Vec<Collected>],
    dedup: &mut DedupState,
) {
    let text = match scrub {
        Some(s) => s.scrub(raw),
        None => raw.to_string(),
    };
    // Post-scrub safety net: drop anything where a name survived inside an identifier.
    if let Some(s) = scrub {
        if s.leaks(&text) {
            return;
        }
    }
    if is_trivial(&text) {
        return;
    }
    let exact = hash_str(&norm_line(&text.replace('\n', " ")));
    if !dedup.seen_exact.insert(exact) {
        return; // exact duplicate
    }
    let sh = shingles(&text);
    if !sh.is_empty() {
        let seen = &mut dedup.per_dev_shingles[dev];
        let overlap = sh.iter().filter(|s| seen.contains(s)).count() as f64 / sh.len() as f64;
        if overlap > NEAR_DUP_OVERLAP {
            return; // near-duplicate of this coder's own boilerplate
        }
        for s in &sh {
            seen.insert(*s);
        }
    }
    per_dev[dev].push(Collected {
        book_title: book_title.to_string(),
        series: series.to_string(),
        text,
        authored,
    });
}

/// Attribute ONE developer's code in ONE repo: run `git log --author=<them>`, keep only
/// commits that also pass the identity double-check, pool their added lines per file,
/// and window each file into passages.
fn process_dev_repo(
    path: &Path,
    repo: &RepoCfg,
    dev: usize,
    patterns: &[String],
    ids: &IdentityMap,
    langs: &HashSet<String>,
    scrub: Option<&Scrubber>,
    per_dev: &mut [Vec<Collected>],
    dedup: &mut DedupState,
) {
    let series = repo.slug.clone();
    let mut acc: BTreeMap<String, Vec<(String, i64)>> = BTreeMap::new();
    let mut ctx: Option<CommitCtx> = None;
    let mut cur_file: Option<String> = None;
    let mut cur_block: Vec<String> = Vec::new();

    stream_log(path, repo, patterns, |line| {
        if let Some(rest) = line.strip_prefix('\u{1}') {
            // New commit header — finalize the previous commit into the per-file pool.
            flush_block(&mut ctx, &cur_file, &mut cur_block);
            if let Some(c) = ctx.take() {
                accumulate_commit(c, &mut acc);
            }
            // --author already narrows to ~this dev; double-check the resolved identity
            // so a loose regex match (another "David", a shared first name) can't leak in.
            let mut it = rest.splitn(5, '\u{1f}');
            let _sha = it.next().unwrap_or("");
            let an = it.next().unwrap_or("");
            let ae = it.next().unwrap_or("");
            let time = it.next().and_then(|s| s.parse::<i64>().ok()).unwrap_or(0);
            let theirs = !is_bot(an, ae) && ids.resolve(an, ae) == Some(dev);
            ctx = Some(CommitCtx {
                dev: theirs.then_some(dev),
                blocks: Vec::new(),
                added: 0,
                time,
            });
            cur_file = None;
            cur_block.clear();
            return;
        }
        if line.starts_with("diff --git ") {
            flush_block(&mut ctx, &cur_file, &mut cur_block);
            cur_file = None;
        } else if line.starts_with("+++ ") {
            flush_block(&mut ctx, &cur_file, &mut cur_block);
            cur_file = match plus_path(line) {
                Some(p) if keep_file(p, langs, &repo.exclude) => Some(p.to_string()),
                _ => None,
            };
        } else if line.starts_with("@@") {
            flush_block(&mut ctx, &cur_file, &mut cur_block);
        } else if line.starts_with("+++") {
            // stray, ignore
        } else if let Some(added) = line.strip_prefix('+') {
            let has_dev = ctx.as_ref().map_or(false, |c| c.dev.is_some());
            if cur_file.is_some() && has_dev {
                cur_block.push(added.to_string());
            }
        } else {
            // context / removed / index / '\ No newline' — ends the current added run
            flush_block(&mut ctx, &cur_file, &mut cur_block);
        }
    });
    // Finalize the final commit.
    flush_block(&mut ctx, &cur_file, &mut cur_block);
    if let Some(c) = ctx.take() {
        accumulate_commit(c, &mut acc);
    }

    // Emit passages per file: pool this coder's contributions to each file so
    // maintainers who commit in small increments still yield enough passages.
    for (file, lines) in acc {
        emit_file(dev, &file, &series, &lines, scrub, per_dev, dedup);
    }
}

/// `--author` regex patterns for a developer: their emails, logins, and full name.
fn author_patterns(d: &DevCfg) -> Vec<String> {
    let mut p: Vec<String> = Vec::new();
    p.extend(d.emails.iter().cloned());
    p.extend(d.logins.iter().cloned());
    p.push(d.name.clone());
    p
}

// ---------------------------------------------------------------------------
// Public entry: collect the whole coders corpus into the shared shape
// ---------------------------------------------------------------------------

pub struct CollectResult {
    pub authors: Vec<AuthorMeta>,
    pub passages: Vec<Passage>,
    pub texts: Vec<String>,
}

/// Scan already-scrubbed texts for any roster identity token that SURVIVED scrubbing.
/// Uses a broader **substring** match (len ≥ 6, so distinctive surnames/logins) than
/// the word-boundary scrubber, so it catches names embedded inside identifiers that
/// `\b` scrubbing would have missed. Returns (passages-with-a-hit, distinct tokens).
pub fn residual_identity_hits(config_path: &Path, texts: &[String]) -> (usize, Vec<String>) {
    let cfg = load_config(config_path);
    let mut tokens: Vec<String> = Vec::new();
    for d in &cfg.devs {
        for l in &d.logins {
            let l = l.to_lowercase();
            if l.len() >= 6 {
                tokens.push(l);
            }
        }
        for part in d.name.split_whitespace() {
            let p = part.to_lowercase();
            if p.len() >= 6 {
                tokens.push(p);
            }
        }
    }
    tokens.sort();
    tokens.dedup();
    let mut hits = 0usize;
    let mut found: Vec<String> = Vec::new();
    for t in texts {
        let low = t.to_lowercase();
        if let Some(tok) = tokens.iter().find(|tok| low.contains(tok.as_str())) {
            hits += 1;
            if !found.contains(tok) {
                found.push(tok.clone());
            }
        }
    }
    (hits, found)
}

fn push_trait(v: &mut Vec<Trait>, name: &str, val: &str) {
    if !val.trim().is_empty() {
        v.push(Trait { name: name.into(), value: val.trim().into() });
    }
}

/// Clone repos, attribute code to roster devs, chunk + (optionally) scrub + dedup, and
/// assemble the `AuthorMeta` / `Passage` / parallel `texts`. `scrub_enabled` is false
/// only for the scrub-ablation control in the audit.
pub fn collect(config_path: &Path, scrub_enabled: bool) -> CollectResult {
    let cfg = load_config(config_path);
    let ids = IdentityMap::build(&cfg.devs);
    let scrubber = Scrubber::build(&cfg.devs);
    let scrub = if scrub_enabled { Some(&scrubber) } else { None };
    let langs: HashSet<String> = cfg.languages.iter().map(|s| s.to_lowercase()).collect();
    assert!(!langs.is_empty(), "coders.toml: `languages` allowlist is empty");

    // Clone every unique repo once (a repo shared by two devs is cloned once; each dev's
    // own commits in it are attributed to them via their own `--author` pass).
    let mut clone_path: HashMap<String, PathBuf> = HashMap::new();
    for d in &cfg.devs {
        for r in &d.repos {
            if !clone_path.contains_key(&r.url) {
                if let Some(p) = ensure_clone(r) {
                    clone_path.insert(r.url.clone(), p);
                }
            }
        }
    }

    let n = cfg.devs.len();
    let mut per_dev: Vec<Vec<Collected>> = vec![Vec::new(); n];
    let mut dedup =
        DedupState { seen_exact: HashSet::new(), per_dev_shingles: vec![HashSet::new(); n] };

    for (di, d) in cfg.devs.iter().enumerate() {
        let patterns = author_patterns(d);
        for r in &d.repos {
            let Some(path) = clone_path.get(&r.url) else { continue };
            let domain = if r.domain.is_empty() { "—" } else { r.domain.as_str() };
            println!("  {} · {} ({domain}) …", d.name, r.slug);
            process_dev_repo(path, r, di, &patterns, &ids, &langs, scrub, &mut per_dev, &mut dedup);
        }
    }

    // Keep only devs with attributed passages; assign dense ids.
    let kept: Vec<usize> = (0..n).filter(|&d| !per_dev[d].is_empty()).collect();
    for (d, dev) in cfg.devs.iter().enumerate() {
        if per_dev[d].is_empty() {
            eprintln!("  !! no passages attributed to {} — dropped", dev.name);
        }
    }

    let mut authors: Vec<AuthorMeta> = Vec::with_capacity(kept.len());
    let mut passages: Vec<Passage> = Vec::new();
    let mut texts: Vec<String> = Vec::new();
    for (new_id, &old) in kept.iter().enumerate() {
        let d = &cfg.devs[old];
        let mut traits = Vec::new();
        push_trait(&mut traits, "Systems or scripting?", &d.traits.paradigm);
        authors.push(AuthorMeta {
            id: new_id,
            name: d.name.clone(),
            color: d.color.clone().unwrap_or_else(|| author_color(new_id)),
            gender: String::new(),
            birth_country: String::new(),
            raised: String::new(),
            educated: String::new(),
            college: String::new(),
            traits,
        });
        let items = sample_even(std::mem::take(&mut per_dev[old]), MAX_PER_DEV);
        let count = items.len();
        for it in items {
            texts.push(it.text.clone());
            passages.push(Passage {
                author_id: new_id,
                book_title: it.book_title,
                series: it.series,
                authored: it.authored,
                text: it.text,
                x: 0.0,
                y: 0.0,
                is_mystery: false,
            });
        }
        println!("  => {}: {} passages", d.name, count);
    }

    assert!(!passages.is_empty(), "no code passages attributed; check identities/logins in coders.toml");
    mark_mystery(&mut passages, MYSTERY_PER_DEV);
    CollectResult { authors, passages, texts }
}

// ---------------------------------------------------------------------------
// Discovery: seed the roster from a repo's top code committers
// ---------------------------------------------------------------------------

/// Print the top code committers (by kept added lines) for every repo in the config —
/// use it to fill in `[[devs]]` blocks. Identities are mailmap-resolved by git.
pub fn discover(config_path: &Path) {
    let cfg = load_config(config_path);
    let langs: HashSet<String> = cfg.languages.iter().map(|s| s.to_lowercase()).collect();
    let mut repos: Vec<RepoCfg> = Vec::new();
    let mut seen = HashSet::new();
    for d in &cfg.devs {
        for r in &d.repos {
            if seen.insert(r.url.clone()) {
                repos.push(r.clone());
            }
        }
    }
    for repo in &repos {
        let Some(path) = ensure_clone(repo) else { continue };
        println!("\n=== {} ({}) — top code committers ===", repo.slug, repo.url);
        // (added_lines, first_year, last_year) keyed by "Name\temail"
        let mut tally: HashMap<String, (u64, i64, i64)> = HashMap::new();
        let mut cur_key: Option<String> = None;
        let mut cur_year: i64 = 0;
        let mut cur_keep = false;
        stream_log(&path, repo, &[], |line| {
            if let Some(rest) = line.strip_prefix('\u{1}') {
                let mut it = rest.splitn(5, '\u{1f}');
                let sha = it.next().unwrap_or("");
                let an = it.next().unwrap_or("");
                let ae = it.next().unwrap_or("");
                let ct = it.next().unwrap_or("");
                let _s = it.next().unwrap_or("");
                let _ = sha;
                if is_bot(an, ae) {
                    cur_key = None;
                    return;
                }
                cur_year = ct.parse::<i64>().map(|t| 1970 + t / 31_557_600).unwrap_or(0);
                cur_key = Some(format!("{an}\t{ae}"));
            } else if line.starts_with("+++ ") {
                cur_keep = matches!(plus_path(line), Some(p) if keep_file(p, &langs, &repo.exclude));
            } else if line.starts_with("diff --git ") {
                cur_keep = false;
            } else if line.starts_with("+++") {
                // ignore
            } else if line.starts_with('+') && cur_keep {
                if let Some(k) = &cur_key {
                    let e = tally.entry(k.clone()).or_insert((0, cur_year, cur_year));
                    e.0 += 1;
                    if cur_year != 0 {
                        e.1 = e.1.min(cur_year);
                        e.2 = e.2.max(cur_year);
                    }
                }
            }
        });
        let mut rows: Vec<(String, (u64, i64, i64))> = tally.into_iter().collect();
        rows.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
        for (k, (lines, first, last)) in rows.into_iter().take(40) {
            let (name, email) = k.split_once('\t').unwrap_or((k.as_str(), ""));
            println!("  {lines:>8}  {first}-{last}  {name}  <{email}>");
        }
    }
}
