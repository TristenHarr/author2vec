#!/usr/bin/env python3
"""Peer-review test harness for jlens/PAPER.md.

Turns "don't write slop / don't hallucinate" into executable invariants. Every check either
PASSES or lists concrete violations with line numbers; the process exits non-zero if any HARD
check fails. Run:  python3 jlens/paper/review/check_paper.py

The crown-jewel invariant is `numbers_traceable`: every *result* number in the paper must be
either present in the machine-extracted ledger (jlens/paper/ledger.json) or explicitly listed in
derivations.json with a formula that reduces to ledger values. Nothing may be invented.
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.normpath(os.path.join(HERE, ".."))
ROOT = os.path.normpath(os.path.join(PAPER_DIR, "..", ".."))
PAPER = os.environ.get("PAPER_MD", os.path.join(ROOT, "jlens", "PAPER.md"))
LEDGER = os.path.join(PAPER_DIR, "ledger.json")
REFS = os.path.join(PAPER_DIR, "refs.bib")
FIGDIR = os.path.join(ROOT, "jlens", "figures")
DERIV = os.path.join(HERE, "derivations.json")
THRESH = os.path.join(HERE, "thresholds.json")

RESULT = []  # (name, hard, passed, failures[])


def check(name, hard=True):
    def deco(fn):
        fails = fn()
        RESULT.append((name, hard, len(fails) == 0, fails))
        return fn
    return deco


def lines(text):
    return text.split("\n")


with open(PAPER) as f:
    PAPER_TEXT = f.read()
PL = lines(PAPER_TEXT)


def numbered(pattern, flags=0, where=None):
    """Yield (lineno, matchtext) for a regex over the paper (optionally a section slice)."""
    out = []
    for i, ln in enumerate(PL, 1):
        for m in re.finditer(pattern, ln, flags):
            out.append((i, m.group(0)))
    return out


# ---------- load ledger + build the set of legitimate numeric strings ----------
with open(LEDGER) as f:
    LEDGER_ROWS = json.load(f)


def flatten_nums(v, acc):
    if isinstance(v, bool):
        return
    if isinstance(v, (int, float)):
        acc.append(float(v))
    elif isinstance(v, list):
        for x in v:
            flatten_nums(x, acc)
    elif isinstance(v, dict):
        for x in v.values():
            flatten_nums(x, acc)


_raw = []
for row in LEDGER_ROWS:
    flatten_nums(row.get("value"), _raw)

# normalized string forms a ledger number can legitimately appear as in prose
LEDGER_FORMS = set()
for v in _raw:
    for s in (f"{v:.3f}", f"{v:.2f}", f"{v:.1f}", f"{v:g}", f"{round(v)}"):
        LEDGER_FORMS.add(s.lstrip("+"))          # incl. round-to-integer (prose rounds 205.2->205)
    # percentage forms (value stored as a fraction 0..1)
    if 0 <= abs(v) <= 1:
        for s in (f"{v*100:.1f}", f"{v*100:.0f}", f"{v*100:g}"):
            LEDGER_FORMS.add(s.lstrip("+"))

DERIVATIONS = json.load(open(DERIV)) if os.path.exists(DERIV) else {}
DERIVED_OK = set(str(k) for k in DERIVATIONS.keys())

# constants that are not "results": model dims, layer counts, roster sizes, years, small ints
WHITELIST = set("""0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 20 22 24 40 55 60
127 130 138 220 384 440 512 600 768 1000 1690 2004 2006 2020 2021 2022 2023 2024 2025 2026
1999 1988 2011 2017 2019 8 50 45 220 7084 2467""".split())


def norm(tok):
    return tok.lstrip("+").rstrip("%").rstrip(".")


# ---------- CHECK: every citation resolves ----------
@check("citations_resolve", hard=True)
def _citations():
    bib = open(REFS).read()
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    fails = []
    for lineno, cite in numbered(r"@[A-Za-z0-9]+"):
        k = cite[1:]
        if k not in keys:
            fails.append(f"PAPER.md:{lineno}  [@{k}] not in refs.bib")
    return fails


# ---------- CHECK: every refs.bib entry is actually cited ----------
@check("no_orphan_refs", hard=False)
def _orphans():
    bib = open(REFS).read()
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    used = set(c[1:] for _, c in numbered(r"@[A-Za-z0-9]+"))
    return [f"refs.bib: '{k}' defined but never cited" for k in sorted(keys - used)]


# ---------- CHECK: every referenced figure exists on disk ----------
@check("figures_exist", hard=True)
def _figs():
    fails = []
    for lineno, ref in numbered(r"figures/fig[0-9A-Za-z_]+\.png"):
        if not os.path.exists(os.path.join(ROOT, "jlens", ref)):
            fails.append(f"PAPER.md:{lineno}  missing {ref}")
    return fails


# ---------- CHECK: no placeholder / draft debris ----------
@check("no_placeholders", hard=True)
def _placeholders():
    pat = r"\b(TODO|TKTK|FIXME|XXX|TBD|lorem ipsum)\b|\?\?\?|\\cite\{|\bplaceholder\b"
    fails = []
    for lineno, tok in numbered(pat, re.IGNORECASE):
        fails.append(f"PAPER.md:{lineno}  placeholder/debris: '{tok}'")
    return fails


# ---------- CHECK: every §X.Y cross-reference resolves to a real heading ----------
@check("section_refs_resolve", hard=True)
def _secrefs():
    heads = set()
    for ln in PL:
        m = re.match(r"#{1,3}\s+(?:Appendix\s+)?([0-9]+(?:\.[0-9]+)?)", ln)
        if m:
            heads.add(m.group(1))
        m2 = re.match(r"#{1,3}\s+Appendix\s+([A-D])", ln)
        if m2:
            heads.add(m2.group(1))
    fails = []
    for lineno, ref in numbered(r"§\s?([0-9]+(?:\.[0-9]+)?)"):
        num = ref.replace("§", "").strip()
        if num not in heads:
            fails.append(f"PAPER.md:{lineno}  dangling section ref §{num} (no such heading)")
    return fails


# ---------- CHECK: ledger is fresh (numbers come from a reproducible extraction) ----------
@check("ledger_fresh", hard=True)
def _fresh():
    tmp = os.path.join(HERE, ".ledger_check.json")
    try:
        # build_ledger writes ledger.json in place; capture current, rebuild, diff, restore
        cur = open(LEDGER).read()
        r = subprocess.run([sys.executable, os.path.join(PAPER_DIR, "build_ledger.py")],
                           capture_output=True, text=True, cwd=ROOT)
        if r.returncode != 0:
            return [f"build_ledger.py failed: {r.stderr[-200:]}"]
        rebuilt = open(LEDGER).read()
        if cur.strip() != rebuilt.strip():
            return ["committed ledger.json differs from a fresh build_ledger.py run "
                    "(numbers may be stale — regenerate + re-audit prose)"]
        return []
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ---------- CHECK (crown jewel): every result number is traceable ----------
@check("numbers_traceable", hard=True)
def _numbers():
    # Result numbers are typeset as math ($58.3\%$, $p<0.001$); section/figure refs are plain
    # text (§5.7, Fig 6). Scan ONLY inside $...$ spans in the claim-bearing region so that
    # cross-references never masquerade as untraceable results.
    start = next(i for i, ln in enumerate(PL) if ln.startswith("## Abstract"))
    end = next(i for i, ln in enumerate(PL) if ln.startswith("## 6."))
    fails = []
    numpat = re.compile(r"[-+]?\d+\.\d{1,3}|\b\d{1,3}\\?%|\b\d{2,}")
    for lineno in range(start, end):
        ln = PL[lineno]
        for span in re.findall(r"\$[^$]*\$", ln):
            span = span.replace("{,}", "")   # LaTeX thousands separator 2{,}467 -> 2467
            for m in numpat.finditer(span):
                tok = m.group(0)
                n = norm(tok.replace("\\%", "").replace("%", ""))
                if not n or n in WHITELIST or n in LEDGER_FORMS or n in DERIVED_OK:
                    continue
                if "." not in n and n.split(".")[0] in WHITELIST:
                    continue
                fails.append(f"PAPER.md:{lineno+1}  untraceable result-number '{tok}' in "
                             f"'{span.strip()}' (not in ledger or derivations.json)")
    return fails


TH = json.load(open(THRESH)) if os.path.exists(THRESH) else {}
_WORDS = re.findall(r"[A-Za-z][A-Za-z'\-]*", PAPER_TEXT)
_PER1K = len(_WORDS) / 1000.0
_LOW = [w.lower() for w in _WORDS]


def _rate(vocab):
    return sum(_LOW.count(w) for w in vocab) / _PER1K


def _lines_with(vocab):
    out = []
    for i, ln in enumerate(PL, 1):
        toks = set(re.findall(r"[A-Za-z][A-Za-z'\-]*", ln.lower()))
        hit = toks & set(vocab)
        if hit:
            out.append((i, ", ".join(sorted(hit))))
    return out


# ---------- CHECK: zero hype superlatives (exemplars: 0 across 69k words) ----------
@check("no_hype", hard=True)
def _hype():
    vocab = TH.get("hype_words", ["revolutionary", "groundbreaking", "unprecedented",
            "world-class", "cutting-edge", "paradigm-shifting", "game-changing", "seminal"])
    return [f"PAPER.md:{ln}  hype superlative: {h}" for ln, h in _lines_with(vocab)]


# ---------- CHECK: intensifier rate (exemplars: median 0.86, max 1.5 /1k) ----------
@check("intensifier_budget", hard=True)
def _intens():
    vocab = TH.get("intensifier_words", ["very", "really", "extremely", "incredibly", "hugely",
            "massively", "truly", "absolutely", "remarkably", "super", "vastly", "enormously"])
    cap = TH.get("intensifier_per_1k", 1.5)
    rate = _rate(vocab)
    if rate > cap:
        return [f"intensifier rate {rate:.2f}/1k exceeds cap {cap}/1k "
                f"(offenders: {', '.join(sorted(set(w for w in _LOW if w in vocab)))})"]
    return []


# ---------- CHECK: "novel" is rare (exemplars ~0.24/1k) ----------
@check("novelty_budget", hard=True)
def _novel():
    cap = TH.get("novel_per_1k", 0.6)
    vocab = {"novel", "novelty"}
    rate = _rate(vocab)
    if rate > cap:
        return [f"'novel/novelty' rate {rate:.2f}/1k exceeds cap {cap}/1k "
                f"({int(rate*_PER1K)} uses in {int(_PER1K*1000)} words) — landmark papers use it ~once/4000 words"]
    return []


# ---------- CHECK: sentence length (exemplars mean 15-24.5 words) — flag run-ons ----------
@check("sentence_length", hard=False)
def _sentlen():
    cap = TH.get("max_sentence_words", 48)
    mean_cap = TH.get("mean_sentence_words", 26.0)
    # prose only: drop headings, image lines, code fences, table rows
    prose = []
    in_code = False
    for ln in PL:
        if ln.strip().startswith("```"):
            in_code = not in_code; continue
        if in_code or ln.startswith("#") or ln.strip().startswith("![") or ln.lstrip().startswith("|"):
            continue
        prose.append(ln)
    text = " ".join(prose)
    sents = re.split(r"(?<=[.!?])\s+", text)
    lens = [(len(re.findall(r"[A-Za-z]+", s)), s) for s in sents]
    lens = [(n, s) for n, s in lens if n > 3]
    mean = sum(n for n, _ in lens) / max(len(lens), 1)
    fails = []
    if mean > mean_cap:
        fails.append(f"mean sentence length {mean:.1f} words > {mean_cap} (landmark median ~21)")
    runons = sorted([x for x in lens if x[0] > cap], reverse=True)
    for n, s in runons[:8]:
        fails.append(f"run-on ({n}w): {s.strip()[:90]}...")
    return fails


# ---------- CHECK: number density (evidence, not vibes) — exemplars median 30/1k ----------
@check("number_density", hard=False)
def _density():
    floor = TH.get("number_density_floor_per_1k", 15.0)
    nums = len(re.findall(r"\d+\.?\d*", PAPER_TEXT))
    rate = nums / _PER1K
    if rate < floor:
        return [f"number density {rate:.1f}/1k below floor {floor}/1k — reads as under-evidenced"]
    return []


# ---------- report ----------
def main():
    print("=" * 72)
    print("PAPER PEER-REVIEW HARNESS  —  jlens/PAPER.md")
    print("=" * 72)
    hard_fail = False
    for name, hard, passed, fails in RESULT:
        tag = "PASS" if passed else ("FAIL" if hard else "WARN")
        print(f"[{tag}] {name}" + ("" if passed else f"  ({len(fails)})"))
        for f in fails[:25]:
            print(f"        - {f}")
        if len(fails) > 25:
            print(f"        ... and {len(fails)-25} more")
        if hard and not passed:
            hard_fail = True
    print("-" * 72)
    hp = sum(1 for _, h, p, _ in RESULT if h and p)
    ht = sum(1 for _, h, _, _ in RESULT if h)
    print(f"HARD checks: {hp}/{ht} passing")
    print("RESULT:", "FAIL — fix violations above" if hard_fail else "PASS — paper meets invariants")
    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
