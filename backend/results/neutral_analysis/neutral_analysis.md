# Neutral-Class Weakness Analysis and Intervention

- Model: `logreg`
- Val/Test sample size: 8,000 / 8,000
- Intervention: scale Neutral probability by `alpha` before argmax
  (prior adjustment / threshold tuning; no retraining).
- `alpha` selected on validation, reported on held-out test.

## 1. Error-Direction Analysis (baseline, test split)

Of 2,450 true-Neutral comments, the baseline `logreg` model:

| Outcome | Count | Share |
|---------|------:|------:|
| Correct (Neutral) | 1,508 | 61.6% |
| Misread as Negative | 599 | 24.4% |
| Misread as Positive | 343 | 14.0% |

Neutral errors are split between both polar classes, i.e. the model
tends to over-commit short, low-signal comments to a polarity rather
than abstaining to Neutral. This motivates a Neutral-favouring prior.

## 2. Alpha Sweep (validation split)

| alpha | Macro-F1 | Neutral-F1 | Neutral-P | Neutral-R |
|------:|---------:|-----------:|----------:|----------:|
| 0.8 | 0.6847 | 0.5937 | 0.6428 | 0.5516 |
| 0.9 | 0.6892 | 0.6097 | 0.6279 | 0.5925 |
| 1.0 | 0.6887 | 0.6153 | 0.6090 | 0.6217 |
| 1.1 | 0.6881 | 0.6189 | 0.5941 | 0.6459 |
| 1.2 | 0.6906 | 0.6269 | 0.5842 | 0.6764 |
| 1.3 | 0.6920 | 0.6336 | 0.5755 | 0.7048 |
| 1.4 | 0.6912 | 0.6362 | 0.5645 | 0.7286 |
| 1.5 | 0.6893 | 0.6388 | 0.5570 | 0.7486 |
| 1.6 | 0.6872 | 0.6405 | 0.5496 | 0.7674 |  <-- selected
| 1.7 | 0.6841 | 0.6395 | 0.5425 | 0.7787 |
| 1.8 | 0.6812 | 0.6387 | 0.5356 | 0.7908 |
| 1.9 | 0.6755 | 0.6342 | 0.5258 | 0.7987 |
| 2.0 | 0.6714 | 0.6328 | 0.5185 | 0.8117 |

Selected `alpha = 1.6` (maximises validation Neutral-F1 subject
to macro-F1 dropping no more than 0.005).

## 3. Held-Out Test Result (baseline vs intervention)

| Metric | Baseline (alpha=1.0) | Intervention (alpha=1.6) | Delta |
|--------|---------------------:|-------------------------:|------:|
| Macro-F1 | 0.6922 | 0.6814 | -0.0108 |
| Neutral-F1 | 0.6175 | 0.6326 | +0.0151 |
| Neutral-Precision | 0.6196 | 0.5486 | — |
| Neutral-Recall | 0.6155 | 0.7469 | — |

## 4. Verdict

**Mixed result.** Neutral-F1 improved by +0.0151 but macro-F1
dropped by -0.0108, exceeding tolerance. The Neutral/macro trade-off
means the intervention is only justified when Neutral recall is the priority.

## 5. Recommendation

- The Neutral class is intrinsically hardest: it has the shortest comments
  (EDA: median 12 words vs 16/15) and the lowest inter-annotator separability.
- Threshold tuning is reported here as a transparent, training-free option.
- Stronger future remedies: class-weighted retraining, Neutral-vs-rest
  cascade classifier, or richer contextual features (encoder embeddings).

