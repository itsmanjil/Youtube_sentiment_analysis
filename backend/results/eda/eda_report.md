# Exploratory Data Analysis — YouTube Sentiment Corpus

Source dataset: `AmaanP314/youtube-comment-sentiment` (HuggingFace Hub).
Labels are automated (not human-annotated); see Label Provenance.

## Class Distribution (from split provenance)

| Split | Total | Negative | Neutral | Positive |
|-------|------:|---------:|--------:|---------:|
| train | 516,886 | 184,266 (35.65%) | 160,744 (31.1%) | 171,876 (33.25%) |
| val | 128,854 | 47,023 (36.49%) | 39,037 (30.3%) | 42,794 (33.21%) |
| test | 165,110 | 59,614 (36.11%) | 50,540 (30.61%) | 54,956 (33.28%) |

The corpus is approximately balanced (each class 30–37% in every split),
so macro-F1 and accuracy are directly comparable and class imbalance is not
a primary confound.

## Comment-Length Distribution

Computed on 50,000 test-split comments.

### Overall (characters / words)

| Metric | Mean | Median | P90 | P99 | Max |
|--------|-----:|-------:|----:|----:|----:|
| Characters | 116.63 | 78.0 | 230.0 | 668.0 | 9001 |
| Words | 21.1 | 14.0 | 41.0 | 117.01 | 1495 |

### Word count by class

| Class | n | Mean words | Median words | P90 words |
|-------|--:|-----------:|-------------:|----------:|
| Negative | 17,951 | 22.75 | 16.0 | 44.0 |
| Neutral | 15,451 | 18.76 | 12.0 | 36.0 |
| Positive | 16,598 | 21.49 | 15.0 | 43.0 |

Comments are short (median ~14 words, P90 ~41), which is the central
modelling challenge: limited lexical context per instance. The Neutral
class contains the shortest comments (median 12 words vs 16 Negative /
15 Positive), which partly explains its lower separability (see the
Neutral-class analysis section).

## Lexical Statistics

- Total tokens (test split): 1,054,939
- Vocabulary size (unique tokens): 40,634
- Type/token ratio: 0.03852

### 20 most frequent tokens

| Token | Count |
|-------|------:|
| `the` | 43,044 |
| `to` | 26,372 |
| `and` | 22,832 |
| `of` | 19,607 |
| `is` | 18,133 |
| `you` | 15,334 |
| `with` | 13,903 |
| `in` | 13,134 |
| `this` | 12,761 |
| `for` | 11,991 |
| `it` | 11,717 |
| `that` | 11,600 |
| `face` | 11,314 |
| `on` | 7,403 |
| `are` | 7,052 |
| `not` | 6,150 |
| `so` | 6,101 |
| `was` | 6,087 |
| `be` | 6,008 |
| `have` | 5,950 |

## Language Distribution

The corpus was language-filtered to English during preprocessing.
During filtering, 138,590 non-English rows were
removed (see split provenance). The retained corpus is therefore
English-only by construction. This is a deliberate scope decision and a
stated external-validity limitation: the system is not evaluated on
code-mixed or non-English comments.

## Category and Country Distribution (metadata split)

Computed on the metadata-bearing domain split (1,641 comments) which
retains `CategoryID` and `CountryCode` columns dropped from the main split.

### Top 10 YouTube CategoryIDs

| CategoryID | Comments |
|-----------:|---------:|
| 25 | 499 |
| 27 | 437 |
| 17 | 144 |
| 26 | 135 |
| 24 | 85 |
| 15 | 80 |
| 28 | 77 |
| 2 | 65 |
| 22 | 44 |
| 20 | 43 |

### Top 10 CountryCodes

| CountryCode | Comments |
|------------:|---------:|
| US | 496 |
| AU | 248 |
| CA | 226 |
| GB | 193 |
| IN | 163 |
| NZ | 113 |
| IE | 109 |
| DE | 83 |
| PH | 10 |

Category and country breadth is the basis for the domain-shift slice
evaluation (see `results/domain_shift/`).
