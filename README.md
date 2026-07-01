# person2vec

*Can we tell who wrote what, from AI embeddings alone?*

A small Rust + [Dioxus](https://dioxuslabs.com) website that embeds thousands of
passages from famous public-domain books, then lets you explore authorship in
vector space:

- **🗺 Vector-space map** — every ~200-word passage as a dot (PCA of its 384-dim
  embedding), colored by author. Hover to read; toggle authors.
- **🎯 Can we predict the author?** — leave-one-out cross-validation accuracy and a
  confusion matrix, computed live in your browser.
- **✍️ Who do you write like?** — pick a held-out "mystery" passage and watch the
  vectors rank the authors by stylistic similarity.

Everything is **embedded once, offline**, and shipped as a static data asset — the
site is pure client-side WASM with no server and no model at runtime. All the
similarity math runs in the browser.

## How it works

| Crate | What it does |
|-------|--------------|
| `shared` | Data types + vector math (cosine, centroids, LOOCV, k-NN, confusion matrix). Pure Rust, compiles for both native and wasm. |
| `corpus` | **Offline** pipeline: fetch books → strip boilerplate → chunk → embed with [`fastembed`](https://github.com/Anush008/fastembed-rs) (all-MiniLM-L6-v2) → PCA to 2D → write `web/assets/person2vec.{json,bin}`. |
| `web`    | The Dioxus/WASM single-page app that loads the precomputed bundle. |

The bundle is two files: `person2vec.json` (author + passage metadata, 2D coords,
mystery flags) and `person2vec.bin` (the packed little-endian f32 vectors, one row
per passage). Embeddings are L2-normalized, so cosine similarity is a dot product.

## Regenerate the data

Needs network the first time (downloads the ~90 MB embedding model and ~26 books
from Project Gutenberg mirrors; both are cached under `corpus/.cache/`):

```bash
cargo run -p corpus --release
```

Edit `corpus/authors.toml` to change the author roster (name, color, Gutenberg book
IDs). The run prints per-author passage counts and a sanity-check LOOCV accuracy.

## Run the site

```bash
cd web
dx serve            # dev server with hot reload at http://localhost:8080
```

## Build a static site

```bash
cd web
dx bundle --release  # static output for any static host
```

Deploying to a **GitHub Pages project site** (served from a subpath): set
`base_path = "/<repo-name>"` in `web/Dioxus.toml`, build, and copy the built
`index.html` to `404.html` so client-side routes resolve.

## Deploy (Cloudflare Workers)

Production is served from Cloudflare's edge as an assets-only Worker — no server
code, just the static WASM bundle. Config lives in `wrangler.jsonc`; live at
**[author2vec.com](https://author2vec.com)**.

Wrangler needs **Node 22+** (pinned via `.nvmrc`):

```bash
nvm use            # selects Node 22 (see .nvmrc)
npm install        # first time only — installs wrangler locally
npx wrangler login # one-time browser auth to your Cloudflare account
npm run deploy     # clean rebuild (dx bundle --release) + wrangler deploy
```

`npm run deploy` wipes the output dir, rebuilds the release bundle, then uploads
it. The `routes` in `wrangler.jsonc` provision the `author2vec.com` and
`www.author2vec.com` custom domains and their DNS records automatically (the zone
already lives on Cloudflare). Client-side deep links (`/accuracy`, `/dimensions`,
`/why`) resolve via `not_found_handling: "single-page-application"`, and
`npm run dev` serves the built bundle locally through wrangler.

## Credits

- Texts: [Project Gutenberg](https://www.gutenberg.org) (public domain).
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` via `fastembed`.
- A note on honesty: PCA is a *linear* projection, so clusters overlap — that's real,
  not a bug. Translated works are excluded so the signal is the author's style, not a
  translator's.
