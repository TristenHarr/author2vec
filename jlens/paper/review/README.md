# Peer-review harness

Executable quality gate for `jlens/PAPER.md`. Turns "no hallucinations, no slop" into invariants a
machine checks, with empirical thresholds derived from landmark papers rather than taste.

```
make -C jlens/paper review     # run the gate (invariants + self-test)
make -C jlens/paper gate       # run the gate, then build the PDF only if it passes
```

## Invariants (`check_paper.py`)

Each is a named check; the process exits non-zero if any **HARD** check fails.

| Check | Hard | What it enforces |
|-------|------|------------------|
| `citations_resolve` | ✓ | every `[@key]` exists in `refs.bib` |
| `no_orphan_refs` | — | every `refs.bib` entry is actually cited |
| `figures_exist` | ✓ | every `figures/figN.png` referenced exists on disk |
| `no_placeholders` | ✓ | no TODO/TKTK/FIXME/`\cite{`/`???` debris |
| `section_refs_resolve` | ✓ | every `§X.Y` points at a real heading |
| `ledger_fresh` | ✓ | committed `ledger.json` equals a fresh `build_ledger.py` run |
| `numbers_traceable` | ✓ | **every result number is in the ledger or `derivations.json`** — nothing invented |
| `no_hype` | ✓ | zero hype superlatives (landmark papers use them 0× in 69k words) |
| `intensifier_budget` | ✓ | ≤ 1.5 intensifiers / 1k words (landmark max) |
| `novelty_budget` | ✓ | ≤ 0.6 "novel" / 1k words (landmark ~0.24) |
| `number_density` | — | ≥ 15 numbers / 1k words (evidence, not vibes) |

The crown jewel is **`numbers_traceable`**: it scans every number inside `$…$` math in the
Abstract→§5 region and requires each to appear in the machine-extracted `ledger.json`, or in
`derivations.json` (documented method constants / derivations-from-ledger with a written
justification). A fabricated statistic cannot pass.

## The gate is proven non-vacuous (`selftest.py`)

A quality gate that never fails is worthless. `selftest.py` poisons a copy of the paper with one of
each defect (hallucinated number, hype word, intensifier flood, novelty overuse, missing figure,
bogus citation, dangling `§`, placeholder) and asserts the harness (a) fails and (b) names each
defect — then asserts the real paper passes. If you add a check, add a poison.

## Files

- `check_paper.py` — the invariants
- `selftest.py` — proves the harness catches what it claims
- `thresholds.json` — empirical caps (from `exemplar_norms.md`)
- `derivations.json` — the allowlist of documented non-ledger numbers, each with a justification
- `exemplar_profiles.json` / `exemplar_norms.md` — style metrics from 8 landmark papers
- `references/` — cloned paper source (git-ignored; regenerate with the exemplar profiler)
