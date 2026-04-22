# Effect Sizes for CI vs Baseline Comparisons

Source: `ci_significance_tests.json`  |  n_test = 20,000  |  paired McNemar contingency cells (b=n01, c=n10)

**Why this matters.** With n_test = 20k even a 0.1 pp accuracy gap achieves p < 0.001, so raw p-values are not informative. Effect sizes (Cohen's g for McNemar, risk difference in percentage points) quantify how much the CI methods actually improve on the logreg baseline.

## CI methods vs Logistic Regression baseline

| Comparison | ΔF1 | Acc gain (pp) | Cohen's g | Magnitude | Odds ratio (b/c) | p_adj | Sig? |
|---|---:|---:|---:|:---:|---:|---:|:---:|
| logreg → meta_learner | +0.0025 | +0.16 | +0.028 | negligible | 1.119 | 1.00e+00 | ✗ |
| logreg → ts_meta | +0.0025 | +0.16 | +0.028 | negligible | 1.119 | 1.00e+00 | ✗ |
| logreg → neuro_fuzzy | +0.0013 | +0.14 | +0.035 | negligible | 1.149 | 1.00e+00 | ✗ |
| logreg → pso | +0.0009 | +0.10 | +0.029 | negligible | 1.124 | 1.00e+00 | ✗ |
| logreg → nsga2 | +0.0006 | +0.07 | +0.030 | negligible | 1.129 | 1.00e+00 | ✗ |
| logreg → ensemble | -0.0005 | -0.01 | +0.001 | negligible | 0.998 | 1.00e+00 | ✗ |

## Interpretation

- Best CI method vs logreg: **meta_learner** with a +0.16 pp accuracy gain on the paired test set and Cohen's g = +0.028 (negligible).
- Worst CI method vs logreg: **ensemble** with -0.01 pp and g = +0.001 (negligible).

- By Cohen's (1988) conventions, g < 0.05 is negligible and g < 0.15 is small. Even when the McNemar test is statistically significant (which is nearly automatic at n=20k), a negligible Cohen's g means the practical improvement is below what a human examiner would notice.
- This justifies the thesis's reframing: CI contributions should be defended on **calibration** (ECE reduction), **Pareto trade-offs**, and the **negative result** that fuzzy / NSGA-II / PSO ensembles do not beat a tuned logreg on F1 at scale.

## Full pairwise table

| A | B | n01 (a✗,b✓) | n10 (a✓,b✗) | Cohen's g | Risk diff b-a (pp) | Odds ratio (b/a) | p_adj | Sig |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| logreg | svm | 707 | 950 | +0.073 | -1.21 | 0.744 | 0.00e+00 | ✓ |
| logreg | tfidf | 1331 | 1985 | +0.099 | -3.27 | 0.671 | 0.00e+00 | ✓ |
| svm | tfidf | 1657 | 2068 | +0.055 | -2.05 | 0.801 | 0.00e+00 | ✓ |
| svm | ensemble | 697 | 455 | +0.105 | +1.21 | 1.532 | 0.00e+00 | ✓ |
| svm | meta_learner | 1024 | 749 | +0.078 | +1.38 | 1.367 | 0.00e+00 | ✓ |
| svm | pso | 995 | 733 | +0.076 | +1.31 | 1.357 | 0.00e+00 | ✓ |
| svm | nsga2 | 974 | 718 | +0.076 | +1.28 | 1.357 | 0.00e+00 | ✓ |
| svm | neuro_fuzzy | 1018 | 746 | +0.077 | +1.36 | 1.365 | 0.00e+00 | ✓ |
| svm | ts_meta | 1024 | 749 | +0.078 | +1.38 | 1.367 | 0.00e+00 | ✓ |
| tfidf | ensemble | 1767 | 1114 | +0.113 | +3.26 | 1.586 | 0.00e+00 | ✓ |
| tfidf | meta_learner | 1895 | 1209 | +0.111 | +3.43 | 1.567 | 0.00e+00 | ✓ |
| tfidf | pso | 1838 | 1165 | +0.112 | +3.36 | 1.578 | 0.00e+00 | ✓ |
| tfidf | nsga2 | 1888 | 1221 | +0.107 | +3.33 | 1.546 | 0.00e+00 | ✓ |
| tfidf | neuro_fuzzy | 1801 | 1118 | +0.117 | +3.42 | 1.611 | 0.00e+00 | ✓ |
| tfidf | ts_meta | 1895 | 1209 | +0.111 | +3.43 | 1.567 | 0.00e+00 | ✓ |
| logreg | ensemble | 433 | 434 | +0.001 | -0.01 | 0.998 | 1.00e+00 | ✗ |
| logreg | meta_learner | 300 | 268 | +0.028 | +0.16 | 1.119 | 1.00e+00 | ✗ |
| logreg | pso | 172 | 153 | +0.029 | +0.10 | 1.124 | 1.00e+00 | ✗ |
| logreg | nsga2 | 114 | 101 | +0.030 | +0.07 | 1.129 | 1.00e+00 | ✗ |
| logreg | neuro_fuzzy | 223 | 194 | +0.035 | +0.14 | 1.149 | 1.00e+00 | ✗ |
| logreg | ts_meta | 300 | 268 | +0.028 | +0.16 | 1.119 | 1.00e+00 | ✗ |
| ensemble | meta_learner | 461 | 428 | +0.019 | +0.17 | 1.077 | 1.00e+00 | ✗ |
| ensemble | pso | 372 | 352 | +0.014 | +0.10 | 1.057 | 1.00e+00 | ✗ |
| ensemble | nsga2 | 377 | 363 | +0.009 | +0.07 | 1.039 | 1.00e+00 | ✗ |
| ensemble | neuro_fuzzy | 398 | 368 | +0.020 | +0.15 | 1.082 | 1.00e+00 | ✗ |
| ensemble | ts_meta | 461 | 428 | +0.019 | +0.17 | 1.077 | 1.00e+00 | ✗ |
| meta_learner | pso | 246 | 259 | +0.013 | -0.07 | 0.950 | 1.00e+00 | ✗ |
| meta_learner | nsga2 | 247 | 266 | +0.019 | -0.10 | 0.929 | 1.00e+00 | ✗ |
| meta_learner | neuro_fuzzy | 304 | 307 | +0.002 | -0.01 | 0.990 | 1.00e+00 | ✗ |
| meta_learner | ts_meta | 0 | 0 | +0.000 | +0.00 | 1.000 | 1.00e+00 | ✗ |
| pso | nsga2 | 52 | 58 | +0.027 | -0.03 | 0.897 | 1.00e+00 | ✗ |
| pso | neuro_fuzzy | 126 | 116 | +0.021 | +0.05 | 1.086 | 1.00e+00 | ✗ |
| pso | ts_meta | 259 | 246 | +0.013 | +0.07 | 1.053 | 1.00e+00 | ✗ |
| nsga2 | neuro_fuzzy | 140 | 124 | +0.030 | +0.08 | 1.129 | 1.00e+00 | ✗ |
| nsga2 | ts_meta | 266 | 247 | +0.019 | +0.10 | 1.077 | 1.00e+00 | ✗ |
| neuro_fuzzy | ts_meta | 307 | 304 | +0.002 | +0.01 | 1.010 | 1.00e+00 | ✗ |
