# Thesis Final Wording

Date: 2026-04-04

This file contains short thesis-safe wording aligned with the pinned runtime
evidence in `backend/results/runtime/route_a_live_v1/`.

## One-Sentence Thesis Position

This thesis presents a reproducible, calibration-aware YouTube sentiment
analysis runtime evaluated on a fixed held-out test set, where the live
meta-learner is the best macro-F1 model and the NSGA-II ensemble is the
strongest calibrated ensemble under pinned artifact version `route_a_live_v1`.

## Contribution Paragraph

The main contribution of this work is not a generic state-of-the-art claim.
Rather, it is the design and validation of a thesis-facing runtime pipeline in
which sentiment models, calibration artifacts, ensemble-weight variants, and
computational-intelligence components are wired into the deployed inference
path and evaluated under a pinned artifact configuration. This makes the final
results easier to trace, reproduce, and defend than a workflow based only on
historical offline experiments.

## Results Paragraph

On the pinned live runtime benchmark over 165,110 held-out YouTube comments,
the stacked meta-learner achieved the highest macro-F1 score of 0.6945, while
the NSGA-II-weighted ensemble achieved the highest accuracy of 0.6959 and the
strongest calibrated ensemble performance with an Expected Calibration Error of
0.004601. Logistic regression remained the strongest single-model calibration
baseline with macro-F1 0.6928 and ECE 0.003900.

## Limitations Paragraph

The current repository should not be presented as a full aspect-based sentiment
analysis system or as a fully transformer-led computational-intelligence
solution. The strongest validated runtime evidence is still classical/ensemble
first, the fuzzy ensemble is implemented but not the best full-test model, the
current aspect feature is a keyword-level proxy rather than full ABSA, and the
`hybrid_dl` runtime is wired but not calibrated in the pinned thesis
configuration because no dedicated temperature row exists for that model.

## Safe Conclusion Paragraph

The defensible final conclusion is that the project delivers a benchmark-scoped,
artifact-pinned, and calibration-aware YouTube sentiment analysis runtime with
stable offline-to-live benchmark behavior. The thesis claim should therefore be
framed around reproducibility, runtime validity, and deployment-oriented model
comparison rather than a broad claim of universally improved calibration or
state-of-the-art sentiment analysis performance.

## Do Not Say

- The system is state-of-the-art.
- The project solves full ABSA.
- All models improved after calibration.
- `hybrid_dl` is calibrated in the pinned runtime.
- The fuzzy ensemble is the best overall runtime model.
