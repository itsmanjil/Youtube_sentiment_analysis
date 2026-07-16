# Domain-Shift / Robustness Slice Evaluation

- Created at: `2026-05-17T07:57:27Z`
- Dataset: `C:\Users\itsma\OneDrive\Documents\GitHub\Youtube_sentiment_analysis\backend\data\test.csv`
- Slice column: `_domain_slice`
- Slice source: `text_length_proxy`
- Samples: `3000`

> The input dataset has no channel/topic/time metadata. Results below are
> text-length robustness slices, not a full cross-domain validation.

## Overall

| Model | Accuracy | Macro-F1 | Worst Slice | Worst Slice F1 | Spread |
| --- | ---: | ---: | --- | ---: | ---: |
| logreg | 0.692000 | 0.687691 | medium | 0.634168 | 0.103649 |
| svm | 0.680333 | 0.675705 | medium | 0.625609 | 0.097311 |
| tfidf | 0.662000 | 0.652884 | medium | 0.596328 | 0.093530 |
| ensemble_nsga2 | 0.693333 | 0.688666 | medium | 0.636747 | 0.095706 |
| meta_learner | 0.689667 | 0.686559 | medium | 0.634055 | 0.105053 |

## logreg Slices

| Slice | Samples | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| medium | 714 | 0.645658 | 0.634168 |
| long | 743 | 0.694482 | 0.676649 |
| short | 774 | 0.687339 | 0.687309 |
| very_short | 769 | 0.737321 | 0.737817 |

## svm Slices

| Slice | Samples | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| medium | 714 | 0.634454 | 0.625609 |
| long | 743 | 0.674293 | 0.656319 |
| short | 774 | 0.686047 | 0.685590 |
| very_short | 769 | 0.723017 | 0.722920 |

## tfidf Slices

| Slice | Samples | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| medium | 714 | 0.609244 | 0.596328 |
| long | 743 | 0.671602 | 0.658679 |
| short | 774 | 0.666667 | 0.660431 |
| very_short | 769 | 0.697009 | 0.689858 |

## ensemble_nsga2 Slices

| Slice | Samples | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| medium | 714 | 0.648459 | 0.636747 |
| long | 743 | 0.699865 | 0.681471 |
| short | 774 | 0.689922 | 0.689821 |
| very_short | 769 | 0.732120 | 0.732453 |

## meta_learner Slices

| Slice | Samples | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| medium | 714 | 0.642857 | 0.634055 |
| long | 743 | 0.691790 | 0.675748 |
| short | 774 | 0.682171 | 0.682590 |
| very_short | 769 | 0.738622 | 0.739108 |

## Interpretation

This is a proxy robustness check because the selected dataset has no channel/topic/time metadata. Add those columns and rerun with `--slice_column` for a true domain-shift evaluation.
