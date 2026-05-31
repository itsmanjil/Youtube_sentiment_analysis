# Neutral-Class Weakness Analysis and Intervention

- Model: `logreg`
- Val/Test sample size: 2,000 / 2,000
- Intervention: scale Neutral probability by `alpha` before argmax
  (prior adjustment / threshold tuning; no retraining).
- `alpha` selected on validation, reported on held-out test.

## 1. Error-Direction Analysis (baseline, test split)

Of 632 true-Neutral comments, the baseline `logreg` model:

| Outcome | Count | Share |
|---------|------:|------:|
| Correct (Neutral) | 378 | 59.8% |
| Misread as Negative | 153 | 24.2% |
| Misread as Positive | 101 | 16.0% |

Neutral errors are split between both polar classes, i.e. the model
tends to over-commit short, low-signal comments to a polarity rather
than abstaining to Neutral. This motivates a Neutral-favouring prior.

## 2. Alpha Sweep (validation split)

| alpha | Macro-F1 | Neutral-F1 | Neutral-P | Neutral-R |
|------:|---------:|-----------:|----------:|----------:|
| 0.8 | 0.6820 | 0.5806 | 0.6136 | 0.5510 |
| 0.9 | 0.6882 | 0.5966 | 0.6093 | 0.5844 |
| 1.0 | 0.6892 | 0.6006 | 0.5945 | 0.6067 |
| 1.1 | 0.6872 | 0.6025 | 0.5705 | 0.6382 |
| 1.2 | 0.6888 | 0.6076 | 0.5573 | 0.6679 |
| 1.3 | 0.6886 | 0.6117 | 0.5459 | 0.6957 |
| 1.4 | 0.6869 | 0.6121 | 0.5314 | 0.7217 |  <-- selected
| 1.5 | 0.6825 | 0.6115 | 0.5209 | 0.7403 |
| 1.6 | 0.6768 | 0.6073 | 0.5088 | 0.7532 |
| 1.7 | 0.6739 | 0.6068 | 0.5031 | 0.7644 |
| 1.8 | 0.6681 | 0.6030 | 0.4941 | 0.7737 |
| 1.9 | 0.6621 | 0.5999 | 0.4862 | 0.7829 |
| 2.0 | 0.6630 | 0.6083 | 0.4861 | 0.8126 |

Selected `alpha = 1.4` (maximises validation Neutral-F1 subject
to macro-F1 dropping no more than 0.005).

## 3. Held-Out Test Result (baseline vs intervention)

| Metric | Baseline (alpha=1.0) | Intervention (alpha=1.4) | Delta |
|--------|---------------------:|-------------------------:|------:|
| Macro-F1 | 0.6832 | 0.6866 | +0.0034 |
| Neutral-F1 | 0.6107 | 0.6398 | +0.0291 |
| Neutral-Precision | 0.6238 | 0.5787 | — |
| Neutral-Recall | 0.5981 | 0.7152 | — |

## 4. Verdict

**The intervention helps.** Neutral-F1 improved by +0.0291 on the
held-out test set while macro-F1 changed by +0.0034 (within the
accepted tolerance). The gain comes from recovering Neutral recall on short,
low-signal comments that the baseline over-committed to a polarity. This is a
cheap, training-free, deployment-ready adjustment.

## 5. Recommendation

- The Neutral class is intrinsically hardest: it has the shortest comments
  (EDA: median 12 words vs 16/15) and the lowest inter-annotator separability.
- Threshold tuning is reported here as a transparent, training-free option.
- Stronger future remedies: class-weighted retraining, Neutral-vs-rest
  cascade classifier, or richer contextual features (encoder embeddings).

