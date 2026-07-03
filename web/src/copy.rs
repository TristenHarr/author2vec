//! Per-dataset editorial copy. Rung labels, trait names, colors, and every number are
//! data-driven (shipped in the bundle); only the voice/framing lives here, selected by
//! the active dataset key so the same views narrate prose authors or code coders.

pub struct DatasetCopy {
    pub tagline: &'static str,
    pub tab_predict: &'static str,
    pub source_prefix: &'static str,
    pub source_name: &'static str,
    pub source_href: &'static str,
    /// Singular noun for one subject: "author" / "coder".
    pub subject: &'static str,
    /// Plural: "authors" / "coders".
    pub entities: &'static str,
    /// Legend heading: "Authors" / "Coders".
    pub entities_title: &'static str,
    /// The held-out unit for the leave-one-out reference line: "book" / "commit".
    pub unit: &'static str,
    pub map_explainer: &'static str,
    pub gradient_caption: &'static str,
    pub gradient_axis: &'static str,
    /// The one clause that is genuinely prose- vs code-specific in the reel superpower line.
    pub reel_super_clause: &'static str,
    pub dims_explainer: &'static str,
    pub story_proof: &'static str,
    /// The "how it works + why you can trust it" note on the Why page.
    pub methodology: &'static str,
}

static AUTHORS: DatasetCopy = DatasetCopy {
    tagline: "Every author writes in a slightly different corner of vector space. We embedded \
              thousands of passages from famous public-domain books with a local AI model. Here \
              is what their fingerprints look like, and how well a computer can tell them apart.",
    tab_predict: "🎯 Can we predict the author?",
    source_prefix: "Texts: ",
    source_name: "Project Gutenberg",
    source_href: "https://www.gutenberg.org",
    subject: "author",
    entities: "authors",
    entities_title: "Authors",
    unit: "book",
    map_explainer: "Each dot is one ~200-word passage, placed by PCA of its embedding and colored \
                    by author. Hollow rings are the held-out \"mystery\" passages. Hover a dot to \
                    read it; click an author in the legend to hide/show them.",
    gradient_caption: "Give the model a book it has never seen, then vary how many passages of the \
                       author's OTHER writing it has read. Recognition climbs with exposure.",
    gradient_axis: "passages per author the model has read",
    reel_super_clause: "you are human and write in English",
    dims_explainer: "Trippy question: can writing style alone reveal an author's gender or birth \
                     country, for an author the model has NEVER read? We build each trait's profile \
                     from the OTHER authors and guess (leave-one-author-out), so it can't cheat by \
                     recognizing who wrote it. Compare it against blind guessing and against always \
                     guessing the biggest group.",
    story_proof: "I used these famous books and texts, but extrapolate for programming and programmers.",
    methodology: "How it works: passages from public-domain books are embedded once, offline, with \
                  a small local model, and it is never shown an author's name. Accuracy is \
                  leave-one-out, and a label-shuffle control confirms the score isn't a leak. The \
                  honest number is what survives when an author's whole book is held out — style, \
                  not per-book vocabulary.",
};

static CODERS: DatasetCopy = DatasetCopy {
    tagline: "Every coder writes in a slightly different corner of vector space. We attributed \
              open-source code to the developers who wrote it — line by line, from git history — \
              scrubbed the names out, and embedded it with a code model. Here is what their \
              fingerprints look like, and how well a computer can tell them apart.",
    tab_predict: "🎯 Can we predict the coder?",
    source_prefix: "Code: ",
    source_name: "public GitHub repositories",
    source_href: "https://github.com",
    subject: "coder",
    entities: "coders",
    entities_title: "Coders",
    unit: "file",
    map_explainer: "Each dot is one code passage — a run of a coder's added lines (attributed \
                    commit-by-commit, then pooled per file) — placed by PCA of its embedding and \
                    colored by coder. Hollow rings are the held-out \"mystery\" passages. Hover a \
                    dot to read the code; click a coder in the legend to hide/show them.",
    gradient_caption: "Give the model a commit it has never seen, then vary how many passages of \
                       the coder's OTHER code it has read. Recognition climbs with exposure.",
    gradient_axis: "passages per coder the model has read",
    reel_super_clause: "you write in the same handful of languages and idioms as everyone else",
    dims_explainer: "Trippy question: what can code style alone reveal — about a snippet, or a \
                     coder, the model was NEVER trained on? We try to guess WHEN a passage was \
                     written and whether it was a weekend commit (from the code alone), and \
                     whether a coder is a systems or a scripting programmer — always holding that \
                     coder out, so it can't cheat by recognizing who wrote it. It nails some and \
                     whiffs on others; compare each against blind guessing and the majority-class \
                     baseline.",
    story_proof: "Here it is pointed at code: public repositories and the commits of well-known \
                  developers, with every name scrubbed out. The same fingerprint is in your own commits.",
    methodology: "How it works: added lines are attributed to the developer who committed them via \
                  git history, and every name, email, and login is scrubbed out before embedding. \
                  A label-shuffle null, a residual-name scan, and a scrub-on/off ablation confirm we \
                  measure style, not identity. The honest number is what survives when an entire \
                  repo is held out: if it holds, personal style is real; if it collapses to chance, \
                  the model was mostly reading the project's vocabulary — either way, an honest result.",
};

/// The editorial copy for the active dataset key (falls back to the authors voice).
pub fn copy_for(key: &str) -> &'static DatasetCopy {
    match key {
        "coders" => &CODERS,
        _ => &AUTHORS,
    }
}
