# Thesis Chapter 5 Polished Draft

Date: 2026-04-02

This document provides a thesis-ready Chapter 5 draft aligned with the pinned
runtime evidence in `backend/results/runtime/route_a_live_v1/`.

## Chapter 5: Conclusion and Future Work

### 5.1 Conclusion

This thesis set out to design and evaluate a sentiment analysis system for
YouTube comments that is not only predictive, but also reproducible and
deployment-aware. The work combined classical machine learning models, ensemble
methods, calibration analysis, and computational-intelligence components within
a single runtime pipeline. A central objective was to ensure that the reported
results could be traced to concrete executable artifacts rather than remaining
as isolated offline experiments.

The final repository state supports this objective. The deployed inference path
is now tied to a pinned runtime artifact version, `route_a_live_v1`, which
records the exact calibration, ensemble, and neuro-fuzzy artifacts used during
evaluation. On the held-out test set of 165,110 YouTube comments, the stacked
meta-learner achieved the highest macro-F1 score (0.6945), while the
NSGA-II-weighted ensemble achieved the highest accuracy (0.6959) and the
strongest calibrated ensemble profile (ECE = 0.004601). Logistic regression
remained a strong single-model baseline and the best-calibrated non-ensemble
model. These findings establish that the system is capable of stable, benchmark-
scoped performance under a fully reproducible runtime configuration.

An important contribution of this work is methodological rather than purely
architectural. The thesis does not only report model metrics; it also resolves
the gap between historical offline evaluation and current deployed runtime
behavior. The reconciliation between the historical thesis table and the pinned
live benchmark showed that same-name models remained numerically stable, while
the ensemble behavior changed in a meaningful way once the runtime began to
distinguish explicit PSO and NSGA-II variants. This makes the final claims more
technically defensible, because the headline results are now tied to the actual
runtime path rather than to an abstract or outdated experimental summary.

At the same time, the findings narrow the appropriate thesis claim. The present
evidence does not justify a broad claim that the system is a generic
state-of-the-art NLP solution, nor does it support the claim that the fuzzy
ensemble is the strongest model on the full held-out test set. Instead, the
defensible conclusion is that this project delivers a reproducible,
calibration-aware YouTube sentiment analysis pipeline whose strongest validated
runtime results are currently achieved by the meta-learner and the NSGA-II
ensemble.

In summary, the main contributions of the thesis are:

1. the construction of an end-to-end YouTube sentiment analysis pipeline with
   reproducible runtime artifacts;
2. the integration of calibration-aware and computational-intelligence
   components into the live inference path;
3. the production of a pinned full-test runtime benchmark on a large held-out
   dataset; and
4. the reconciliation of historical offline benchmark tables with current live
   runtime behavior.

These contributions provide a solid engineering and evaluation foundation for a
Master’s thesis in Data Science and Computational Intelligence, even though the
transformer-centered Route A objective remains incomplete.

### 5.2 Limitations of the Present Work

Several limitations remain and should be stated clearly. First, the strongest
runtime results are still classical-first rather than transformer-first. This
means that the intended Route A goal — demonstrating a validated
computational-intelligence advance built on top of strong pretrained encoders —
has not yet been fully achieved. The current transformer infrastructure is in
place, but the full-scale evidence needed to make it the core thesis headline is
still pending.

Second, although the fuzzy ensemble is now wired into the live runtime, it does
not currently outperform the strongest full-test models. It therefore should be
interpreted as an implemented and experimentally relevant CI component, but not
as the final headline result of the system.

Third, the full-test statistical comparison has not yet been rerun specifically
for the pinned live `ensemble_nsga2` variant against the live meta-learner.
Consequently, the current runtime superiority claims for `ensemble_nsga2` should
be expressed in descriptive and calibration-oriented terms rather than as final
inferential superiority claims.

Fourth, while runtime artifact pinning substantially improves reproducibility,
the thesis still depends on disciplined reporting. Any updated claim must cite
the exact artifact version and associated manifest. Without this discipline, the
system could drift away from the documented thesis results.

### 5.3 Future Work

The most important next step is to complete the transformer-centered Route A
agenda. This requires training and evaluating stronger encoder baselines on the
larger transformer-profile splits, ideally on GPU-backed infrastructure. In
particular, `ModernBERT` and `DeBERTa-v3` should be benchmarked under the same
held-out protocol as the classical models, and their calibrated outputs should
then be integrated into the live runtime comparison framework.

Once a competitive transformer baseline is established, the computational-
intelligence contribution can be strengthened in a more meaningful way. Rather
than relying on weak-model fuzzy fusion, the next stage should focus on
instance-adaptive, uncertainty-aware decision layers built over strong encoder
outputs. This includes model disagreement features, entropy-based routing,
selective prediction, abstention policies, and multi-objective optimization over
macro-F1, calibration, coverage, and latency.

Another important direction is evaluation under distribution shift. Future work
should include cross-channel and temporal testing, as well as evaluation on a
human-labeled gold subset of YouTube comments. This would strengthen construct
validity and make the thesis claims more robust to noise and label ambiguity.

Additional future work may include:

- rerunning paired significance tests directly on the pinned live runtime model
  family;
- adding conformal prediction or selective classification for more rigorous
  uncertainty-aware deployment;
- conducting domain-adaptive pretraining on large volumes of unlabeled YouTube
  text;
- replacing keyword-level aspect aggregation with a true ABSA/ASTE pipeline; and
- extending the system beyond text-only sentiment by incorporating metadata such
  as titles, descriptions, or transcripts in a controlled multimodal setting.

### 5.4 Final Closing Statement

The central outcome of this thesis is not the claim that one model has achieved
an absolute best-in-class score across all sentiment-analysis settings. Rather,
the main result is that a complex research codebase has been turned into a
traceable, reproducible, and calibration-aware runtime system with benchmark-
scoped evidence on a large held-out YouTube dataset. This is a stronger and more
credible contribution than a loosely supported headline claim. It provides a
clear platform for future transformer-centered CI research while already meeting
the standard of a defensible, technically coherent Master’s thesis.

## Suggested Final Paragraph

This thesis demonstrates that rigorous runtime validation, artifact pinning, and
calibration-aware evaluation are essential for making sentiment-analysis claims
that are both scientifically credible and practically useful. The present system
achieves reproducible full-test performance through the meta-learner and
NSGA-II ensemble under a pinned runtime configuration, while also establishing a
clear roadmap toward stronger transformer-centered computational-intelligence
extensions in future work.
