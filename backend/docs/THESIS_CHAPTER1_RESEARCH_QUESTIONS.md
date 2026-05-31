# Chapter 1 — Introduction, Framing, and Research Questions

Status date: 2026-05-31

This document provides the formal Chapter 1 framing for the thesis. It states
the problem, the thesis position, the explicit research questions, and the
contribution claims. It is written for direct adaptation into the thesis
document.

## 1.1 Problem Statement

Sentiment analysis of YouTube comments is a high-volume, noisy, short-text
classification problem. Comments are brief (median ~14 words in this corpus),
informal, topically diverse, and frequently ambiguous — particularly for the
Neutral class. In deployment settings such as brand monitoring, content
moderation triage, and audience analytics, a sentiment system must not only be
*accurate* but also produce *reliable probability estimates*: a prediction of
"90% Positive" should be correct about 90% of the time. Raw accuracy alone is
therefore an insufficient characterisation of system quality.

## 1.2 Thesis Position (Scoped Claim)

This thesis does **not** claim a new state-of-the-art accuracy result on YouTube
sentiment. Instead, it makes a narrower and more defensible claim:

> This work delivers a **reproducible, uncertainty-aware, calibration-aware
> YouTube sentiment analysis pipeline** in which computational-intelligence
> ensemble methods (PSO- and NSGA-II-weighted ensembles, stacked meta-learning,
> and neuro-fuzzy gating) are combined with post-hoc calibration and selective
> prediction, and in which every reported result maps to a pinned, re-runnable
> artifact.

The contribution is the **integration and honest evaluation** of these
computational-intelligence components under a single artifact-pinned runtime,
with calibration and uncertainty treated as first-class evaluation targets
alongside macro-F1.

## 1.3 Research Questions

**RQ1 — Multi-objective ensemble optimisation.**
Can multi-objective evolutionary optimisation (NSGA-II) produce ensemble
weights that improve *calibration* (Expected Calibration Error, Brier score)
relative to single-objective Particle Swarm Optimisation (PSO) and to a logistic
regression baseline, without sacrificing macro-F1?

**RQ2 — Selective prediction.**
Does entropy-based selective prediction (abstention on low-confidence comments)
improve the effective accuracy of the sentiment models on the high-confidence
subset, and what is the accuracy-versus-coverage trade-off across architectures?

**RQ3 — Calibration across model families.**
How does calibration quality (ECE, Brier) vary across classical single models,
PSO/NSGA-II ensembles, and stacked meta-learning on a large held-out test set,
and is post-hoc temperature scaling uniformly beneficial?

**RQ4 — Human agreement and error structure.**
How well do the automated models agree with independent human judgement on a
hand-annotated gold set (with inter-annotator reliability quantified via
Krippendorff's alpha), and what comment characteristics — especially in the
Neutral class — explain systematic model–human disagreement?

## 1.4 Contributions

1. An **artifact-pinned runtime** (`route_a_live_v1`) in which the deployed
   inference path is tied to fixed calibration, ensemble-weight, and
   neuro-fuzzy configuration files, with a live-vs-offline reconciliation
   proving the deployed system reproduces the offline benchmark
   (RQ1, RQ3; `results/runtime/route_a_live_v1/`).

2. A **comparative computational-intelligence evaluation** of PSO vs NSGA-II
   ensemble weighting, stacked meta-learning, and neuro-fuzzy gating, reported
   on 165,110 held-out comments with both macro-F1 and calibration metrics
   (RQ1, RQ3).

3. A **selective-prediction analysis** (coverage–accuracy curves, entropy
   gating) quantifying the confidence quality of each architecture
   (RQ2; `results/route_a_benchmark_cpu_ci/`).

4. A **human gold-set evaluation** with two independent annotators
   (Krippendorff's α = 0.9547, strong agreement) that separates label error
   from model error and grounds the headline metrics in human judgement
   (RQ4; `results/gold_set/`).

5. A **Neutral-class error analysis** with a transparent, training-free
   threshold-tuning intervention whose trade-offs are reported honestly
   (RQ4; `results/neutral_analysis/`).

## 1.5 Scope and Explicit Non-Claims

- **Transformers are future work.** Only a smoke/CPU benchmark of DeBERTa-v3
  exists; no full-dataset, fine-tuned transformer result is claimed. The
  deployed headline is classical-and-ensemble-first. (See
  `ROUTE_A_ENCODER_POSITION.md`.)
- **Labels are automated, not human-origin.** The source corpus is
  automatically labelled; reported metrics measure agreement with that scheme,
  with the human gold set providing an independent reliability check. (See
  `LABEL_PROVENANCE.md`.)
- **Aspect analysis is keyword-level**, not full aspect-based sentiment
  analysis (ABSA).
- The macro-F1 advantage of the best model over logistic regression is **small**
  and is discussed honestly; the thesis argument rests on calibration and
  uncertainty quality, not on raw F1 superiority.
