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
	  --from=markdown+tex_math_dollars \
	  --citeproc --bibliography=$(BIB) \
	  -M reference-section-title=References \
	  --shift-heading-level-by=-1 \
	  -M date=" " \
	  --pdf-engine=$(ENGINE) \
	  --resource-path=..:. \
	  -H header.tex \
	  -V geometry:margin=1in -V fontsize=10pt \
	  -V colorlinks=true -V linkcolor=black -V citecolor=black -V urlcolor=Blue4 \
	  -o $(OUT)
	@echo "built $(OUT)"
	@# build-integrity gate: citations must resolve and figures must be captioned
	@pandoc $(PAPER) --from=markdown+tex_math_dollars --citeproc --bibliography=$(BIB) \
	  --shift-heading-level-by=-1 -M reference-section-title=References -t latex 2>/dev/null \
	  | awk 'BEGIN{c=0;f=0} /\[@[a-zA-Z]/{c++} /\\caption\{/{f++} END{\
	    if(c>0){print "PDF INTEGRITY FAIL: "c" unresolved [@cite] markers"; exit 1}\
	    if(f<8){print "PDF INTEGRITY FAIL: only "f" figure captions (<8)"; exit 1}\
	    print "PDF integrity OK: citations resolve, "f" figure captions"}'

check:
	@command -v pandoc >/dev/null 2>&1 && echo "pandoc: OK" || echo "pandoc: MISSING (build blocked)"
	@command -v $(ENGINE) >/dev/null 2>&1 && echo "$(ENGINE): OK" || echo "$(ENGINE): MISSING (build blocked)"

clean:
	rm -f $(OUT)
