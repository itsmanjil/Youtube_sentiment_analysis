# Per-Class F1 Breakdown

Test set: n = 20,000  (seed=42)

## F1 Score per Class

| Method | Type | Macro-F1 | F1 Positive | F1 Neutral | F1 Negative |
|--------|------|----------|-------------|------------|-------------|
| meta_learner | Classical | 0.6967 | 0.7532 | 0.6322 | 0.7046 |
| ts_meta | CI — Temp-scaled | 0.6967 | 0.7532 | 0.6322 | 0.7046 |
| neuro_fuzzy | CI — Neuro-fuzzy | 0.6955 | 0.7524 | 0.6249 | 0.7092 |
| pso | CI — PSO | 0.6951 | 0.7518 | 0.6260 | 0.7076 |
| nsga2 | CI — NSGA-II | 0.6949 | 0.7517 | 0.6251 | 0.7078 |
| logreg | Classical | 0.6943 | 0.7504 | 0.6250 | 0.7073 |
| ensemble | Classical | 0.6938 | 0.7496 | 0.6231 | 0.7087 |
| svm | Classical | 0.6817 | 0.7401 | 0.6081 | 0.6968 |
| tfidf | Classical | 0.6579 | 0.7237 | 0.5650 | 0.6849 |

## Precision per Class

| Method | Prec Positive | Prec Neutral | Prec Negative |
|--------|---------------|--------------|---------------|
| meta_learner | 0.7766 | 0.6158 | 0.7023 |
| ts_meta | 0.7766 | 0.6158 | 0.7023 |
| neuro_fuzzy | 0.7839 | 0.6297 | 0.6819 |
| pso | 0.7802 | 0.6284 | 0.6842 |
| nsga2 | 0.7791 | 0.6260 | 0.6863 |
| logreg | 0.7758 | 0.6239 | 0.6889 |
| ensemble | 0.7774 | 0.6303 | 0.6817 |
| svm | 0.7584 | 0.6108 | 0.6800 |
| tfidf | 0.7701 | 0.6184 | 0.6188 |

## Recall per Class

| Method | Rec Positive | Rec Neutral | Rec Negative |
|--------|--------------|-------------|--------------|
| meta_learner | 0.7312 | 0.6496 | 0.7070 |
| ts_meta | 0.7312 | 0.6496 | 0.7070 |
| neuro_fuzzy | 0.7234 | 0.6202 | 0.7388 |
| pso | 0.7253 | 0.6236 | 0.7327 |
| nsga2 | 0.7262 | 0.6241 | 0.7306 |
| logreg | 0.7267 | 0.6261 | 0.7268 |
| ensemble | 0.7237 | 0.6162 | 0.7378 |
| svm | 0.7226 | 0.6055 | 0.7144 |
| tfidf | 0.6827 | 0.5201 | 0.7669 |

## Analysis: Neutral Class (Hardest)

The Neutral class consistently achieves the lowest F1 across all methods, reflecting the inherent ambiguity of neutral sentiment in YouTube comments.

| Method | F1 Neutral | vs Best Neutral |
|--------|------------|-----------------|
| meta_learner | 0.6322 | +0.0000 |
| ts_meta | 0.6322 | +0.0000 |
| pso | 0.6260 | -0.0062 |
| nsga2 | 0.6251 | -0.0071 |
| logreg | 0.6250 | -0.0072 |
| neuro_fuzzy | 0.6249 | -0.0073 |
| ensemble | 0.6231 | -0.0091 |
| svm | 0.6081 | -0.0241 |
| tfidf | 0.5650 | -0.0672 |

- **Best Neutral F1**: meta_learner (0.6322)
- **Worst Neutral F1**: tfidf (0.5650)
- **Range**: 0.0672 points across all methods

## CI vs Classical — Per-Class Delta

*(relative to meta_learner as best classical baseline)*

| CI Method | ΔF1 Positive | ΔF1 Neutral | ΔF1 Negative | ΔMacro-F1 |
|-----------|-------------|------------|-------------|-----------|
| pso | -0.0014 | -0.0062 | +0.0030 | -0.0016 |
| nsga2 | -0.0015 | -0.0071 | +0.0032 | -0.0018 |
| neuro_fuzzy | -0.0008 | -0.0073 | +0.0046 | -0.0012 |
| ts_meta | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Thesis Interpretation

Three structural patterns emerge across all methods:

1. **Neutral class bottleneck** — F1 Neutral is 10–15 points below Positive and Negative for every method. This reflects genuine label ambiguity (many YouTube comments express mild or mixed sentiment) and is a known challenge in social media sentiment analysis.

2. **CI methods do not uniformly improve all classes** — the per-class breakdown reveals which classes benefit from adaptive routing. Methods that improve Neutral F1 are particularly valuable because it is the hardest class.

3. **Temperature scaling preserves class balance** — ts_meta shows near-identical per-class F1 to meta_learner (as expected: temperature scaling preserves argmax, only changing confidence).
