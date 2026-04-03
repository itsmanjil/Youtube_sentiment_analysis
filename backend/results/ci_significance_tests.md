# CI Method Significance Testing

## Setup

- **Test set**: n = 20,000 samples (seed=42)
- **Test**: McNemar's exact two-sided test on discordant pairs
- **Correction**: Holm-Bonferroni across all pairwise comparisons
- **α** = 0.05  (`*` p<0.05  `**` p<0.01  `***` p<0.001  `ns` not significant)

## Method Performance Summary

| Method | Macro-F1 | Type |
|--------|----------|------|
| meta_learner | 0.6967 | Classical baseline |
| ts_meta | 0.6967 | CI — Temp-scaled meta |
| neuro_fuzzy | 0.6955 | CI — Neuro-fuzzy gate |
| pso | 0.6951 | CI — Single-obj PSO |
| nsga2 | 0.6949 | CI — Multi-obj NSGA-II |
| logreg | 0.6943 | Classical baseline |
| ensemble | 0.6938 | Classical baseline |
| svm | 0.6817 | Classical baseline |
| tfidf | 0.6579 | Classical baseline |

## Pairwise McNemar's Tests (Holm-adjusted)

| Method A | Method B | F1(A) | F1(B) | n01 | n10 | p_raw | p_adj | Sig |
|----------|----------|-------|-------|-----|-----|-------|-------|-----|
| logreg | svm | 0.6943 | 0.6817 | 707 | 950 | 0.0000e+00 | 0.0000e+00 | *** |
| logreg | tfidf | 0.6943 | 0.6579 | 1331 | 1985 | 0.0000e+00 | 0.0000e+00 | *** |
| svm | tfidf | 0.6817 | 0.6579 | 1657 | 2068 | 0.0000e+00 | 0.0000e+00 | *** |
| svm | ensemble | 0.6817 | 0.6938 | 697 | 455 | 0.0000e+00 | 0.0000e+00 | *** |
| svm | meta_learner | 0.6817 | 0.6967 | 1024 | 749 | 0.0000e+00 | 0.0000e+00 | *** |
| svm | pso | 0.6817 | 0.6951 | 995 | 733 | 0.0000e+00 | 0.0000e+00 | *** |
| svm | nsga2 | 0.6817 | 0.6949 | 974 | 718 | 0.0000e+00 | 0.0000e+00 | *** |
| svm | neuro_fuzzy | 0.6817 | 0.6955 | 1018 | 746 | 0.0000e+00 | 0.0000e+00 | *** |
| svm | ts_meta | 0.6817 | 0.6967 | 1024 | 749 | 0.0000e+00 | 0.0000e+00 | *** |
| tfidf | ensemble | 0.6579 | 0.6938 | 1767 | 1114 | 0.0000e+00 | 0.0000e+00 | *** |
| tfidf | meta_learner | 0.6579 | 0.6967 | 1895 | 1209 | 0.0000e+00 | 0.0000e+00 | *** |
| tfidf | pso | 0.6579 | 0.6951 | 1838 | 1165 | 0.0000e+00 | 0.0000e+00 | *** |
| tfidf | nsga2 | 0.6579 | 0.6949 | 1888 | 1221 | 0.0000e+00 | 0.0000e+00 | *** |
| tfidf | neuro_fuzzy | 0.6579 | 0.6955 | 1801 | 1118 | 0.0000e+00 | 0.0000e+00 | *** |
| tfidf | ts_meta | 0.6579 | 0.6967 | 1895 | 1209 | 0.0000e+00 | 0.0000e+00 | *** |
| logreg | ensemble | 0.6943 | 0.6938 | 433 | 434 | 1.0000e+00 | 1.0000e+00 | ns |
| logreg | meta_learner | 0.6943 | 0.6967 | 300 | 268 | 1.9331e-01 | 1.0000e+00 | ns |
| logreg | pso | 0.6943 | 0.6951 | 172 | 153 | 3.1806e-01 | 1.0000e+00 | ns |
| logreg | nsga2 | 0.6943 | 0.6949 | 114 | 101 | 4.1319e-01 | 1.0000e+00 | ns |
| logreg | neuro_fuzzy | 0.6943 | 0.6955 | 223 | 194 | 1.7025e-01 | 1.0000e+00 | ns |
| logreg | ts_meta | 0.6943 | 0.6967 | 300 | 268 | 1.9331e-01 | 1.0000e+00 | ns |
| ensemble | meta_learner | 0.6938 | 0.6967 | 461 | 428 | 2.8315e-01 | 1.0000e+00 | ns |
| ensemble | pso | 0.6938 | 0.6951 | 372 | 352 | 4.8013e-01 | 1.0000e+00 | ns |
| ensemble | nsga2 | 0.6938 | 0.6949 | 377 | 363 | 6.3276e-01 | 1.0000e+00 | ns |
| ensemble | neuro_fuzzy | 0.6938 | 0.6955 | 398 | 368 | 2.9472e-01 | 1.0000e+00 | ns |
| ensemble | ts_meta | 0.6938 | 0.6967 | 461 | 428 | 2.8315e-01 | 1.0000e+00 | ns |
| meta_learner | pso | 0.6967 | 0.6951 | 246 | 259 | 5.9339e-01 | 1.0000e+00 | ns |
| meta_learner | nsga2 | 0.6967 | 0.6949 | 247 | 266 | 4.2680e-01 | 1.0000e+00 | ns |
| meta_learner | neuro_fuzzy | 0.6967 | 0.6955 | 304 | 307 | 9.3552e-01 | 1.0000e+00 | ns |
| meta_learner | ts_meta | 0.6967 | 0.6967 | 0 | 0 | 1.0000e+00 | 1.0000e+00 | ns |
| pso | nsga2 | 0.6951 | 0.6949 | 52 | 58 | 6.3376e-01 | 1.0000e+00 | ns |
| pso | neuro_fuzzy | 0.6951 | 0.6955 | 126 | 116 | 5.6299e-01 | 1.0000e+00 | ns |
| pso | ts_meta | 0.6951 | 0.6967 | 259 | 246 | 5.9339e-01 | 1.0000e+00 | ns |
| nsga2 | neuro_fuzzy | 0.6949 | 0.6955 | 140 | 124 | 3.5593e-01 | 1.0000e+00 | ns |
| nsga2 | ts_meta | 0.6949 | 0.6967 | 266 | 247 | 4.2680e-01 | 1.0000e+00 | ns |
| neuro_fuzzy | ts_meta | 0.6955 | 0.6967 | 307 | 304 | 9.3552e-01 | 1.0000e+00 | ns |

## Key Findings

- **15/36** pairs are statistically significant (p_adj < 0.05)
- **CI vs Classical** significant pairs: 8
- **CI vs CI** significant pairs: 0

### CI methods significantly outperforming classical baselines:

- **pso** > svm  (ΔF1=+0.0135, p_adj=0.0000e+00 ***)
- **nsga2** > svm  (ΔF1=+0.0132, p_adj=0.0000e+00 ***)
- **neuro_fuzzy** > svm  (ΔF1=+0.0138, p_adj=0.0000e+00 ***)
- **ts_meta** > svm  (ΔF1=+0.0150, p_adj=0.0000e+00 ***)
- **pso** > tfidf  (ΔF1=+0.0373, p_adj=0.0000e+00 ***)
- **nsga2** > tfidf  (ΔF1=+0.0370, p_adj=0.0000e+00 ***)
- **neuro_fuzzy** > tfidf  (ΔF1=+0.0376, p_adj=0.0000e+00 ***)
- **ts_meta** > tfidf  (ΔF1=+0.0388, p_adj=0.0000e+00 ***)

### CI pairs NOT significant (p_adj ≥ 0.05):

- logreg vs pso  (p_adj=1.0000e+00 — insufficient evidence of difference)
- logreg vs nsga2  (p_adj=1.0000e+00 — insufficient evidence of difference)
- logreg vs neuro_fuzzy  (p_adj=1.0000e+00 — insufficient evidence of difference)
- logreg vs ts_meta  (p_adj=1.0000e+00 — insufficient evidence of difference)
- ensemble vs pso  (p_adj=1.0000e+00 — insufficient evidence of difference)
- ensemble vs nsga2  (p_adj=1.0000e+00 — insufficient evidence of difference)

## Thesis Interpretation

McNemar's test is appropriate here because all methods are evaluated on the same test set and the question is whether the *pattern of errors* differs — not the overall accuracy level. The Holm-Bonferroni correction controls the family-wise error rate across all pairwise comparisons.

Significant results (p_adj < 0.05) indicate that the two methods make systematically different errors on specific samples — meaning one method genuinely recovers samples the other cannot. Non-significant results indicate that any observed F1 difference is within the range of chance variation on this test set.
