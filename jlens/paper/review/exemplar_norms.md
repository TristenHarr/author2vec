# Empirical style norms from 8 landmark ML papers

Corpus (~69,400 words of prose): Attention Is All You Need, BERT, Sentence-BERT,
Tuned Lens, Representation Engineering, GPT-3, RoBERTa, ELMo. All metrics are on
LaTeX bodies crudely detex'd to prose. Rates are per 1,000 words.

## Empirical bands (min – median – max)

| Metric | min | median | max |
|---|---|---|---|
| mean_sentence_len (words) | 15.3 | 21.4 | 24.5 |
| hedge_rate | 0.0 | 0.84 | 2.29 |
| intensifier_rate | 0.26 | 0.86 | 1.50 |
| superlative_slop_rate | 0.54 | 2.28 | 4.62 |
| first_person_rate (we/our/us) | 16.9 | 22.4 | 35.9 |
| number_density (numeric tokens) | 10.9 | 30.3 | 37.3 |
| citation_count (whole paper) | 49 | 92 | 203 |
| figure_count | 0 | 5 | 89 |

**Superlative breakdown (raw, all 8 papers):** state-of-the-art 74, best 58,
novel 17, greatest 1 — and revolutionary 0, groundbreaking 0, unprecedented 0,
world-class 0. The "superlative_slop_rate" is almost entirely *technical
comparatives* (state-of-the-art / best) tied to measured results, NOT hype.

## Testable rules for our paper

1. **Zero hype superlatives.** {revolutionary, groundbreaking, unprecedented,
   world-class} = 0 across 69k landmark words. HARD CAP: 0 in ours.
2. **"novel" is rare (~0.24/1k, ~1 per 4,000 words).** Use it ≤1–2 times total;
   otherwise state concretely what is new. Flag if novel_rate > 0.5/1k.
3. **Superlatives only as measured comparisons.** "state-of-the-art"/"best" are
   fine *only* when the sentence also carries a number. If a superlative has no
   adjacent numeric token, it's slop — rewrite it.
4. **Intensifiers near zero.** intensifier_rate median 0.86, max 1.50; incredibly
   /hugely /massively never appear. CAP ours at ≤1.5/1k, target <1.0.
5. **Hedge, don't waffle.** hedge_rate 0–2.3/1k (median 0.84). Some hedging is
   normal and honest; CAP at ≤2.5/1k so we don't drown claims in "seems/might".
6. **Write in the first person, actively.** first_person_rate 17–36/1k (median
   22). Target 15–35/1k; don't hide behind passive "it is shown that".
7. **Be number-dense.** number_density 11–37/1k (median 30). Target ≥15/1k;
   every empirical claim should sit next to a figure. Below ~11 reads as
   under-evidenced.
8. **Sentences 15–25 words on average.** Target mean 18–23; no long runs of
   40+ word sentences. And cite generously: a full paper carries ≥50 citations
   (min observed 49), i.e. roughly 1 per major claim.
