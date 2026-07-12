#!/usr/bin/env python3
"""Meta-test: prove the peer-review harness is not vacuous.

A quality gate that never fails is worthless. This poisons a copy of the paper with one instance
of each defect the harness claims to catch, runs the harness against the poisoned copy, and asserts
(a) it exits non-zero and (b) each specific defect is named in the output. Then it asserts the real
paper passes. Run:  python3 jlens/paper/review/selftest.py
"""
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
REAL_PAPER = os.path.join(ROOT, "jlens", "PAPER.md")
HARNESS = os.path.join(HERE, "check_paper.py")

# Each poison line injects one defect; the value is a substring the harness output must contain.
POISONS = [
    (r"We achieve a $87.3\%$ breakthrough on every metric.", "untraceable result-number", "hallucinated number"),
    # regression guard for the red-team hole: a fabricated round cosine in math must NOT slip
    # through by coincidentally matching a real 0.897 rounded to one decimal.
    (r"Raw cross-model cosine is high ($\approx 0.9$).", "untraceable result-number", "fabricated round cosine (0.9)"),
    ("This is a groundbreaking, revolutionary result.", "hype superlative", "hype words"),
    # rate checks are document-level budgets, so the poison must breach the rate, not just appear once
    (("It is " + "very really extremely truly absolutely incredibly hugely massively vastly enormously " * 4
      + "good."), "intensifier rate", "intensifier flood"),
    ("See our novel novel novel novel novel novel novel novel novel approach.", "novel", "novelty overuse"),
    ("As shown in ![x](figures/fig999_ghost.png) the ghost figure.", "missing figures/fig999_ghost.png", "missing figure"),
    ("This builds on prior work [@nonexistent_citation_xyz].", "not in refs.bib", "bogus citation"),
    ("We prove this in §9.9 later.", "dangling section ref", "dangling section ref"),
    ("Remaining task: TODO finish this section.", "placeholder", "placeholder debris"),
]


def run(paper_path):
    env = dict(os.environ, PAPER_MD=paper_path)
    r = subprocess.run([sys.executable, HARNESS], capture_output=True, text=True, env=env, cwd=ROOT)
    return r.returncode, r.stdout + r.stderr


def main():
    with open(REAL_PAPER) as f:
        body = f.read()
    # inject poisons inside the results region (after "## 5." and before "## 6.") so the
    # number/slop scanners (which scope to the claim region) actually see them.
    inject = "\n\n### 5.99 Poison block (selftest only)\n\n" + "\n\n".join(p[0] for p in POISONS) + "\n"
    poisoned = body.replace("## 6. Discussion", inject + "\n## 6. Discussion", 1)

    fails = []
    with tempfile.NamedTemporaryFile("w", suffix=".md", dir=HERE, delete=False) as tf:
        tf.write(poisoned)
        poisoned_path = tf.name
    try:
        code, out = run(poisoned_path)
        if code == 0:
            fails.append("harness returned PASS on a poisoned paper (should FAIL)")
        for _line, needle, label in POISONS:
            if needle not in out:
                fails.append(f"harness did NOT catch: {label} (expected substring '{needle}')")
    finally:
        os.remove(poisoned_path)

    # and the real paper must pass cleanly
    code_real, out_real = run(REAL_PAPER)
    if code_real != 0:
        fails.append("harness FAILS on the real paper (self-test expects it clean at commit time)")

    print("=" * 64)
    print("HARNESS SELF-TEST")
    print("=" * 64)
    if fails:
        for f in fails:
            print("  [X]", f)
        print(f"\nSELF-TEST FAILED ({len(fails)} problem(s)) — the harness is not trustworthy.")
        sys.exit(1)
    print(f"  [OK] harness FAILS on {len(POISONS)}/{len(POISONS)} injected defects")
    print("  [OK] harness PASSES on the real paper")
    print("\nSELF-TEST PASSED — the gate catches what it claims to.")
    sys.exit(0)


if __name__ == "__main__":
    main()
