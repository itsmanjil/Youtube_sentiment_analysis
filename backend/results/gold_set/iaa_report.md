# Inter-Annotator Agreement Report

- Generated: 2026-05-29T04:35:01.540022+00:00
- Gold set: `data\gold_set_human_reconciled.csv`
- Annotators: annotator_1, annotator_2
- Items: 300  |  Fully annotated: 300  |  Disputed: 9

## Agreement Metrics

| Metric | Value | Interpretation |
| --- | --- | --- |
| Percent agreement | 0.9700 | — |
| Krippendorff alpha | 0.9547 | strong agreement (alpha >= 0.80) |
| Fleiss kappa | 0.9546 | - |
| Cohen kappa (annotator_1 vs annotator_2) | 0.9546 | - |

## Gold Label Distribution (after reconciliation)

| Label | Count |
| --- | --- |
| Negative | 79 |
| Neutral | 108 |
| Positive | 104 |
| *Disputed (no majority)* | 9 |

## Per-Annotator Label Distribution

| Label | annotator_1 | annotator_2 |
| --- | --- | --- |
| Negative | 80 | 85 |
| Neutral | 113 | 110 |
| Positive | 107 | 105 |

## Interpreting Krippendorff's Alpha

| Range | Interpretation |
| --- | --- |
| alpha >= 0.80 | Strong agreement - gold labels are reliable |
| 0.67 <= alpha < 0.80 | Tentative agreement - Krippendorff's minimum for tentative conclusions |
| 0.40 <= alpha < 0.67 | Moderate agreement - treat gold labels with caution |
| alpha < 0.40 | Poor agreement - adjudication required |

> **Note:** 9 items had no majority label.
> These are marked `is_disputed=True` in the gold set and should be
> reviewed and adjudicated before use as ground truth.

