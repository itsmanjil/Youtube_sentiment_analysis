# Thesis Abstract and Chapter 4 Polished Draft

Date: 2026-04-02

This draft is written in a more formal academic style for direct adaptation into
the thesis document.

## Abstract

Sentiment analysis of YouTube comments presents a challenging problem due to
noise, brevity, topic variation, and the need for reliable probabilistic
predictions in practical deployment settings. This thesis develops and evaluates
a reproducible YouTube sentiment analysis system that integrates classical
machine learning models, ensemble methods, and computational-intelligence
components within a unified runtime pipeline. To strengthen reproducibility, the
deployed inference path is tied to a pinned runtime artifact version
(`route_a_live_v1`), including fixed calibration, ensemble-weight, and
neuro-fuzzy configuration files. Evaluation was conducted on a fixed held-out
test set of 165,110 YouTube comments. Under the pinned live runtime, the
stacked meta-learner achieved the highest macro-F1 score (0.6945) with an
accuracy of 0.6953, while the NSGA-II-weighted ensemble achieved the highest
accuracy (0.6959) and the strongest calibration-oriented ensemble performance
(ECE = 0.004601). Logistic regression remained the strongest single-model
calibration baseline (macro-F1 = 0.6928, ECE = 0.003900). A reconciliation
analysis between historical offline benchmark tables and the pinned live runtime
showed that same-name models remained numerically stable, whereas the ensemble
result should now be interpreted through explicit PSO and NSGA-II runtime
variants rather than a single generic ensemble row. The findings support a
benchmark-scoped claim of a reproducible and calibration-aware YouTube sentiment
analysis pipeline, rather than a generic claim of state-of-the-art NLP
performance.

## Chapter 4: Results and Discussion

### 4.1 Evaluation Setting

All final runtime results reported in this chapter are drawn from the pinned
live artifact version `route_a_live_v1`, which defines the exact calibration,
ensemble, and neuro-fuzzy runtime configuration used by the deployed inference
system. The evaluation dataset is a fixed held-out test split
(`backend/data/test.csv`) containing 165,110 labeled YouTube comments. This
setup was chosen to ensure that the reported results reflect the actual runtime
behavior of the system rather than only offline experimental scripts.

The primary runtime benchmark artifact is
`backend/results/runtime/route_a_live_v1/live_runtime_benchmark_full_test.md`,
and the corresponding artifact manifest is stored in
`backend/results/runtime/route_a_live_v1/manifest.json`. These files define the
source of truth for the results reported below.

### 4.2 Full-Test Runtime Performance

Table 4.1 summarizes the performance of the live runtime models on the held-out
test set. The stacked meta-learner achieved the highest macro-F1 score
(0.6945), indicating that it remains the strongest model when the principal
objective is balanced multi-class classification performance. The
NSGA-II-weighted ensemble achieved the highest accuracy (0.6959) and a macro-F1
of 0.6940, while also exhibiting substantially better calibration than the
historical generic ensemble formulation. Logistic regression achieved a
macro-F1 of 0.6928 and the lowest Expected Calibration Error (0.003900) among
the single-model baselines, making it a strong deployment-oriented comparator.

The fuzzy ensemble was active in the runtime path but did not improve full-test
headline performance. Its macro-F1 score was 0.6567, which placed it below the
meta-learner, the NSGA-II ensemble, logistic regression, and SVM. Accordingly,
the fuzzy ensemble should be interpreted in this thesis as an implemented
computational-intelligence component that is experimentally relevant, but not as
the strongest full-test production model in the current pinned runtime.

### 4.3 Interpretation of the Runtime Headline

The results indicate that the appropriate runtime headline depends on the
evaluation objective. If the thesis emphasizes overall classification
performance, the meta-learner should be cited as the principal runtime model
because it achieved the highest macro-F1. If the emphasis is on deployment
reliability and confidence quality, the NSGA-II-weighted ensemble provides the
stronger operational profile because it combines the best runtime accuracy with
a substantially lower calibration error than the historical ensemble baseline.

This distinction is methodologically important. A single headline number is not
sufficient to characterize the quality of a deployed sentiment analysis system,
particularly when probabilistic outputs are intended to support confidence-aware
decision-making. For that reason, both macro-F1 and calibration quality should
be reported together when presenting the final system.

### 4.4 Reconciliation of Historical Offline and Live Runtime Results

To verify consistency between earlier thesis tables and the current deployed
stack, a direct offline-versus-live reconciliation was performed using
`backend/results/runtime/route_a_live_v1/offline_vs_live_reconciliation.md`.
This comparison showed that the same-name models were effectively stable across
the two paths. TF-IDF, logistic regression, and SVM were unchanged on accuracy
and macro-F1. The meta-learner differed only minimally from the historical
offline table (Δ accuracy = -0.0002; Δ macro-F1 = -0.0001), indicating that the
runtime wiring did not materially alter its predictive behavior.

The most important difference concerned the ensemble result. The historical
offline table reported a single generic `ensemble` row, whereas the live runtime
now exposes two distinct variants: `ensemble_pso` and `ensemble_nsga2`. Relative
to the historical offline ensemble, the live NSGA-II ensemble improved accuracy
by +0.0027, macro-F1 by +0.0031, and reduced ECE by -0.014459. By contrast, the
PSO ensemble underperformed the historical ensemble on macro-F1 while still
improving calibration. This finding confirms that the generic historical
ensemble row should no longer be treated as the definitive runtime ensemble
reference.

### 4.5 Statistical Interpretation

The repository already contains paired McNemar significance results for the
historical offline benchmark family in `backend/results/thesis_mcnemar.md`. In
that setting, the historical meta-learner differed significantly from the
historical ensemble (`p_adj = 4.15e-05`) and from logistic regression
(`p_adj = 0.045`). These results remain useful for understanding the behavior of
the original benchmark family. However, paired significance testing has not yet
been recomputed specifically for the pinned live `ensemble_nsga2` runtime
variant against the live meta-learner. Therefore, the live NSGA-II result should
be interpreted as a strong descriptive runtime result rather than as a fully
established inferential improvement over the meta-learner.

### 4.6 Discussion

The evidence supports three main conclusions. First, the repository now contains
a reproducible live runtime benchmark tied to explicit artifact versions,
substantially improving thesis credibility and engineering validity. Second, the
current best full-test runtime performance remains classical-first rather than
transformer-first, with the meta-learner and NSGA-II ensemble outperforming the
currently wired fuzzy system on the large held-out set. Third, calibration-aware
evaluation materially changes the interpretation of model quality: although the
meta-learner provides the best macro-F1, the NSGA-II ensemble provides a more
attractive accuracy-calibration trade-off for deployment.

These conclusions lead to a more defensible thesis position than a broad
“state-of-the-art” framing. The current system should instead be presented as a
benchmark-scoped, deployment-aware, and reproducible YouTube sentiment analysis
pipeline whose strongest validated results arise from the meta-learner and the
NSGA-II ensemble under the pinned runtime configuration.

### 4.7 Limitations

Several limitations should be acknowledged. The strongest live runtime results
are not yet transformer-led, so the Route A objective of demonstrating a
validated transformer-centered computational-intelligence advance remains
unfinished. The fuzzy ensemble is active in the runtime path, but it does not
currently improve the full-test headline metrics. In addition, the live NSGA-II
ensemble has not yet been subjected to a dedicated paired significance test
against the live meta-learner, which constrains inferential claims. Finally,
although artifact pinning substantially improves reproducibility, future thesis
updates must continue to cite the exact runtime artifact version used to produce
each benchmark table.

### 4.8 Chapter Summary

This chapter has shown that the current pinned live runtime achieves stable and
reproducible full-test performance on 165,110 YouTube comments. The stacked
meta-learner is the best model by macro-F1, the NSGA-II-weighted ensemble is the
best deployment-oriented ensemble by accuracy and calibration, and logistic
regression remains a strong single-model calibration baseline. The runtime
results are consistent with the historical offline benchmark, but the ensemble
interpretation has changed and must now be expressed through explicit runtime
variants. Consequently, the appropriate thesis claim is that the project
delivers a benchmark-scoped and calibration-aware runtime system, not that it
has already established a general state-of-the-art sentiment analysis model.

## Suggested Citation Notes

When writing the thesis, cite these repository artifacts directly in the
supporting text:

- `backend/results/runtime/route_a_live_v1/live_runtime_benchmark_full_test.md`
- `backend/results/runtime/route_a_live_v1/offline_vs_live_reconciliation.md`
- `backend/results/runtime/route_a_live_v1/manifest.json`
- `backend/results/thesis_mcnemar.md`
