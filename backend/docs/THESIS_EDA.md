# Chapter 3 (Part A) — Exploratory Data Analysis

Status date: 2026-05-31
Generated artifact: `backend/results/eda/eda_report.md` (+ `.json`)
Reproduce: `python research/analysis/eda_report.py --test data/test.csv --sample 50000`

This section summarises the exploratory data analysis of the corpus. All figures
are produced by `research/analysis/eda_report.py` and stored under
`results/eda/`. Numbers below are from the test split (length/lexical statistics
computed on a fixed 50,000-row sample, seed 42).

## 3A.1 Class Distribution

The corpus is approximately balanced across all three splits, so macro-F1 and
accuracy are directly comparable and class imbalance is not a primary confound.

| Split | Total | Negative | Neutral | Positive |
|-------|------:|---------:|--------:|---------:|
| Train | 516,886 | 184,266 (35.7%) | 160,744 (31.1%) | 171,876 (33.3%) |
| Val | 128,854 | 47,023 (36.5%) | 39,037 (30.3%) | 42,794 (33.2%) |
| Test | 165,110 | 59,614 (36.1%) | 50,540 (30.6%) | 54,956 (33.3%) |

Neutral is the smallest class in every split (~31%) but not so small as to
constitute an imbalance problem; its weaker performance (Chapter 4) is driven by
ambiguity and length, not frequency.

## 3A.2 Comment-Length Distribution

Comments are short, which is the central modelling challenge — each instance
provides little lexical context.

| Metric | Mean | Median | P90 | P99 | Max |
|--------|-----:|-------:|----:|----:|----:|
| Characters | 116.6 | 78 | 230 | 668 | 9,001 |
| Words | 21.1 | 14 | 41 | 117 | 1,495 |

**Length by class (words):**

| Class | Mean | Median | P90 |
|-------|-----:|-------:|----:|
| Negative | 22.8 | 16 | 44 |
| Neutral | 18.8 | **12** | 36 |
| Positive | 21.5 | 15 | 43 |

**Key finding.** Neutral comments are the *shortest* (median 12 words vs 16 for
Negative and 15 for Positive). This couples the two hardest factors — short text
and label ambiguity — precisely on the Neutral class, and provides a
data-grounded explanation for the consistently lower Neutral F1 observed across
every model. This finding is referenced directly in the Neutral-class analysis
(Chapter 4) and the threats-to-validity discussion.

## 3A.3 Lexical Statistics

On the 50,000-comment sample, the corpus exhibits the high type/token diversity
characteristic of informal social text (large vocabulary relative to tokens,
driven by typos, slang, elongations, and named entities). The full token
frequency table and vocabulary size are in `results/eda/eda_report.json`. The
most frequent tokens are dominated by function words and platform-generic terms,
confirming that discriminative sentiment signal is sparse and distributed —
which is why TF-IDF weighting (down-weighting ubiquitous tokens) is an effective
classical representation here.

## 3A.4 Language Distribution

The corpus was language-filtered to English during preprocessing (138,590
non-English rows removed; see split provenance). The retained corpus is
therefore English-only by construction. This is a deliberate scope decision and
a stated **external-validity limitation**: the system is neither trained nor
evaluated on code-mixed or non-English comments, and its conclusions should not
be extended to multilingual deployment without further work (the multilingual
encoder presets `xlm_v` and `mdeberta_v3` are scaffolded but untrained, and
remain future work).

## 3A.5 Category and Country Distribution (Domain Metadata)

The main split retains only `text` and `label`, but a metadata-bearing 10k
domain split (`data/route_a_domain_10k/`) preserves `CategoryID`, `CountryCode`,
`VideoID`, and `PublishedAt`. This supports the domain-shift slice evaluation.

**Top YouTube categories** (by comment count in the 1,641-row domain test slice):
25, 27, 17, 26, 24, 15, 28, 2, 22, 20 — i.e. a spread across News/Politics (25),
Education (27), Sports (17), Howto (26), Entertainment (24), and others.

**Top countries:** US (496), AU (248), CA (226), GB (193), IN (163), NZ (113),
IE (109), DE (83), PH (10) — an English-speaking-majority but internationally
spread audience.

This breadth is the basis for the **domain-shift slice evaluation**
(`results/domain_shift/category_domain_shift.md` and `country_domain_shift.md`),
which measures per-category and per-country performance spread. That analysis
shows a meaningful accuracy spread across categories (e.g. logistic regression
ranges from F1 ≈ 0.62 on the weakest category to ≈ 0.83 on the strongest),
quantifying the model's sensitivity to topical domain shift.

## 3A.6 Label-Noise Discussion

Because the source labels are automated (see `LABEL_PROVENANCE.md`), some
proportion of "errors" measured against them are in fact *label* noise rather
than *model* error. Two pieces of evidence bound this:

1. The **human gold set** (Chapter 4) shows strong inter-annotator agreement
   (Krippendorff's α = 0.9547) but only ~70% model agreement with reconciled
   human labels — indicating that a substantial share of the gap between
   automated-label accuracy and human-label accuracy is attributable to the
   automated labelling scheme and to genuine ambiguity, not solely to the model.
2. The 9 of 300 gold-set items with no human majority ("disputed") quantify
   irreducible ambiguity, concentrated in the Neutral region.

This is treated as a construct-validity limitation in the threats-to-validity
section, and is the reason the thesis reports human-gold metrics alongside the
automated-label metrics rather than relying on either alone.
