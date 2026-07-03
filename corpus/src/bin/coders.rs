//! Generate the `coders` (code-authorship) dataset.
//!
//!   cargo run -p corpus --bin coders --release                # generate the bundle
//!   cargo run -p corpus --bin coders --release -- --discover   # seed the roster
//!   cargo run -p corpus --bin coders --release -- --config path/to/roster.toml
//!
//! Clones the repos in `coders.toml`, attributes ADDED lines per commit to the roster
//! dev who wrote them, scrubs identity tokens, chunks + dedups into ~160-token code
//! passages, embeds them with a code-specialized model, and writes
//! `web/assets/person2vec-coders.{json,bin}` + the datasets manifest.

use std::path::PathBuf;

use fastembed::EmbeddingModel;

use corpus::{coders, embed_and_write, manifest_dir, write_datasets_manifest, ModelSpec};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let config = arg_value(&args, "--config")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir().join("coders.toml"));

    if args.iter().any(|a| a == "--discover") {
        coders::discover(&config);
        return;
    }

    let c = coders::collect(&config, true);

    if args.iter().any(|a| a == "--dry") {
        println!("\nDRY RUN — {} passages from {} coders:", c.passages.len(), c.authors.len());
        for a in &c.authors {
            let n = c.passages.iter().filter(|p| p.author_id == a.id).count();
            let m = c.passages.iter().filter(|p| p.author_id == a.id && p.is_mystery).count();
            println!("  {:<28} {n:>4} passages ({m} mystery)", a.name);
        }
        if let Some(p) = c.passages.iter().find(|p| !p.is_mystery) {
            let snip: String = p.text.chars().take(320).collect();
            println!(
                "\n--- sample: {} · {}\n{snip}",
                c.authors[p.author_id].name, p.book_title
            );
        }
        return;
    }
    println!(
        "\nCollected {} code passages from {} coders. Embedding…",
        c.passages.len(),
        c.authors.len()
    );

    let labels = coders::code_ladder_labels();
    // Prefer the code-specialized model; fall back to MiniLM if it can't load.
    let jina = [ModelSpec {
        key: "coders".into(),
        model: EmbeddingModel::JinaEmbeddingsV2BaseCode,
        name: "jina-embeddings-v2-base-code".into(),
        size: "code · 161M · 768-d".into(),
    }];
    let mut infos = embed_and_write(&jina, &c.authors, &c.passages, &c.texts, &labels);
    if infos.is_empty() {
        eprintln!("jina code model unavailable; falling back to all-MiniLM-L6-v2.");
        let mini = [ModelSpec {
            key: "coders".into(),
            model: EmbeddingModel::AllMiniLML6V2,
            name: "all-MiniLM-L6-v2".into(),
            size: "general · 23M · 384-d".into(),
        }];
        infos = embed_and_write(&mini, &c.authors, &c.passages, &c.texts, &labels);
    }
    assert!(!infos.is_empty(), "no embedding model available for coders");

    write_datasets_manifest();
    println!("\nDone. Wrote web/assets/person2vec-coders.{{json,bin}}.");
}

fn arg_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}
