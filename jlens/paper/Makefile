# Build the paper PDF from the canonical Markdown source.
#
# Markdown (../PAPER.md) is the source of truth and renders on GitHub as-is.
# This target is optional: it produces a typeset arXiv-style PDF via pandoc.
#
# Requires: pandoc + a LaTeX engine (xelatex/pdflatex) or tectonic.
#   macOS:  brew install pandoc tectonic   (ENGINE defaults to tectonic)
#
# `make review` runs the peer-review test harness (invariants + self-test) — the quality gate.
# `make gate`   runs review, then builds the PDF only if the paper passes.

PAPER   := ../PAPER.md
OUT     := paper.pdf
BIB     := refs.bib
ENGINE  ?= tectonic

.PHONY: pdf clean check review gate

pdf: $(OUT)

review:
	python3 review/check_paper.py
	python3 review/selftest.py

gate: review pdf

$(OUT): $(PAPER) $(BIB)
	pandoc $(PAPER) \
	  --from=gfm+tex_math_dollars \
	  --citeproc --bibliography=$(BIB) \
	  --pdf-engine=$(ENGINE) \
	  --resource-path=..:. \
	  -H header.tex \
	  -V geometry:margin=1in -V fontsize=10pt -V linkcolor=blue \
	  -V colorlinks=true \
	  --toc \
	  -o $(OUT)
	@echo "built $(OUT)"

check:
	@command -v pandoc >/dev/null 2>&1 && echo "pandoc: OK" || echo "pandoc: MISSING (build blocked)"
	@command -v $(ENGINE) >/dev/null 2>&1 && echo "$(ENGINE): OK" || echo "$(ENGINE): MISSING (build blocked)"

clean:
	rm -f $(OUT)
