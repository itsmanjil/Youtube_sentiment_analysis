# Chapter 3B: System Design and Implementation

Status date: 2026-07-02

This chapter describes the architecture and configuration of the deployed
system: the classical base classifiers, the three computational-intelligence
combination methods (PSO, NSGA-II, stacked meta-learning), the neuro-fuzzy
gate, the calibration procedure, and the backend/frontend platform that
serves them. Every hyperparameter below is quoted directly from the
implementation rather than described qualitatively, so that the
configuration is independently checkable against the cited source file.

## 3B.1 Architecture Overview

The system is a three-tier application. A Django backend (`backend/`) hosts
the inference pipeline and a REST API; a React frontend (`frontend/`)
provides the analyst-facing dashboard; a research layer
(`backend/research/`) trains and evaluates the classical, ensemble, and
computational-intelligence models offline and exports the pinned runtime
artifacts (`results/runtime/route_a_live_v1/`) that the API loads at
inference time. The base representation for every classical model is TF-IDF
over the cleaned comment text, chosen for its sub-millisecond inference cost
at YouTube scale (Chapter 2).

## 3B.2 Base Classifiers (Level 0)

Three base learners are trained on TF-IDF features and used both directly
and as inputs to every downstream combination method:

| Model | Algorithm | Key configuration |
|-------|-----------|--------------------|
| `logreg` | Logistic Regression | `C=1.0`, `max_iter=200`, `solver="saga"` |
| `svm` | Linear SVM | `LinearSVC` (default regularisation) |
| `tfidf` | Multinomial Naive Bayes | `alpha=0.5` |

These three are the level-0 learners for both the metaheuristic ensembles
(§3B.3–3B.4) and the stacked meta-learner (§3B.5).

## 3B.3 Particle Swarm Optimisation (PSO) Ensemble Weighting

**Source:** `research/optimize_ensemble.py` (production driver);
`research/computational_intelligence/metaheuristics/pso.py` (generic PSO
class, used for the standalone metaheuristics module rather than the
production ensemble pipeline).

PSO searches over one continuous weight per base model
(`logreg`, `svm`, `tfidf`), initialised uniformly in `[0, 1]`, clipped to be
non-negative after each velocity update, and re-normalised to sum to one
before every fitness evaluation. Fitness is the validation-set macro-F1 of
the resulting weighted-average ensemble's argmax predictions (maximised).

| Parameter | Value |
|-----------|-------|
| Particles | 20 (script default) / 30 (pipeline default, `--ensemble_particles`) |
| Iterations | 30 (script default) / 50 (pipeline default, `--ensemble_iterations`) |
| Inertia (`w`) | 0.7 |
| Cognitive coefficient (`c1`) | 1.4 |
| Social coefficient (`c2`) | 1.4 |
| Seed | 42 |
| Objective | Maximise validation macro-F1 (argmax of weighted probability blend) |

The optimiser converged to an SVM-dominant blend (`logreg=0.3087`,
`svm=0.6913`, `tfidf=0.0`), achieving 0.7617 macro-F1 on validation. This
weight vector is written to `results/pso_ensemble_weights.json` and pinned
into the live runtime as `ensemble_pso`.

## 3B.4 NSGA-II Multi-Objective Ensemble Weighting

**Source:** `research/ci/multi_objective_ensemble.py` (driver);
`research/computational_intelligence/metaheuristics/nsga2.py` (algorithm).

Where PSO optimises a single objective, NSGA-II searches the same
ensemble-weight space for a Pareto front trading off three simultaneously
minimised objectives:

1. Negative macro-F1 (i.e. maximise macro-F1)
2. Expected Calibration Error (ECE) (minimise)
3. Negative coverage at a 0.70 confidence threshold (i.e. maximise the
   fraction of predictions with max-class probability ≥ 0.70)

| Parameter | Value |
|-----------|-------|
| Population | 60 |
| Generations | 80 |
| Crossover | Simulated Binary Crossover (SBX), probability 0.9, distribution index η=20 |
| Mutation | Polynomial mutation, distribution index η=20, probability 1/n (n = number of decision variables) |
| Selection | Binary tournament on non-dominated rank + crowding distance |
| Seed | 42 |

Ranking uses standard fast non-dominated sorting with crowding-distance
diversity preservation. The final ensemble configuration
(`ensemble_nsga2`) is selected from the returned Pareto front as the
**knee point**: the solution minimising the normalised Chebyshev
(L∞) distance to the ideal point (the per-objective best value across the
front). This is a principled, parameter-free way to pick a single
balanced operating point from a multi-objective search rather than
arbitrarily choosing the best-F1 or best-ECE extreme. The full Pareto
front, the selected weights, and validation/test metrics for the knee
point are written to `results/multi_objective_ensemble.json`.

## 3B.5 Stacked Meta-Learner

**Source:** `research/meta_learner.py`, class `MetaLearnerEnsemble`.

The stacked meta-learner is a two-level model. Level 0 is the same
`logreg`/`svm`/`tfidf` base classifiers as above; Level 1 is trained on
their out-of-fold class-probability predictions rather than their
in-sample predictions, to avoid the meta-learner over-fitting to base
models that have already seen the training labels.

- **Out-of-fold generation:** `StratifiedKFold(n_splits=5, shuffle=True,
  random_state=42)`. Base models are retrained inside each fold
  (`per_fold` mode, the thesis-grade default) rather than loaded
  pre-trained, so no base model ever sees the labels of the fold it is
  predicting on.
- **Meta-features:** each base model contributes its 3-class probability
  vector (Negative/Neutral/Positive), giving 9 features total
  (`feature_type="probs"`).
- **Meta-model:** `LogisticRegression(C=1.0, max_iter=1000,
  class_weight="balanced", solver="lbfgs")`, trained on the 9-dimensional
  OOF probability features against the true labels.
- **Persistence:** the fitted meta-model plus base-model/vectorizer
  configuration is serialised to `meta_learner.pkl` for reproducible
  inference.

At inference time the three base models are run on the input comment, and
their concatenated probability vectors are passed through the trained
logistic-regression meta-model to produce the final class probabilities —
a learned, data-driven combination rule, in contrast to the fixed linear
weights of PSO/NSGA-II.

## 3B.6 Neuro-Fuzzy Gate

**Source:** `research/ci/neuro_fuzzy_gate.py`, class `NeuroFuzzyGate`.

The neuro-fuzzy gate is a simplified ANFIS-style module that learns a
*per-sample* (rather than fixed) ensemble weighting based on each base
model's confidence. A full three-model, three-set ANFIS would require
3³ = 27 cross-model rules; this implementation instead fits 27 total
parameters via a per-model linear combination of single-model fuzzy sets,
which is tractable to fit on the available validation data while
retaining the fuzzy-inference structure:

- **Input (linguistic variable):** per-model confidence, defined as
  1 − normalised predictive entropy.
- **Fuzzification:** three Gaussian membership functions per model — Low,
  Medium, High — `μ(c) = exp(−0.5·((c − center)/width)²)`. Initial
  centers are {0.25, 0.50, 0.75}, initial width 0.20.
- **Rule/consequent structure:** for each model *m*, a gate value
  `gate_m = Σ_k α_{m,k} · μ_{m,k}(c_m)` is computed as a learned linear
  combination of that model's three membership activations (3 models ×
  3 sets × {center, log-width, consequent weight} = 27 parameters).
- **Defuzzification:** the three per-model gates are passed through a
  softmax to obtain per-sample ensemble weights, which are then used to
  combine the base models' probability vectors.
- **Fitting:** all 27 parameters (MF centers, log-widths, consequent
  weights) are jointly optimised by minimising the negative
  log-likelihood of the gated ensemble's predictions on the validation
  set, via `scipy.optimize.minimize(method="L-BFGS-B")` with bounds
  `center ∈ [0.01, 0.99]`, `log_width ∈ [log 0.05, log 0.50]`,
  `alpha ∈ [-5, 5]`, `ftol=1e-8`, `gtol=1e-6`, for up to `maxiter=200`
  iterations (CLI default `--maxiter 200`).

**Deployment note.** The fitted parameters are consumed at inference time
by a second implementation, `FuzzyEnsembleSentimentEngine`
(`src/sentiment/engines/fuzzy_engine.py`, method `_nf_gate_blend`), which
loads `neuro_fuzzy_gate.json` from the pinned runtime manifest whenever its
configured base models match the fitted gate's `{logreg, svm, tfidf}`
exactly. This deployed blend re-purposes the fitted `alpha` as a Gaussian
sharpness term, `exp(−alpha·(c − center)²/(2·width²))`, rather than the
linear consequent weight (`alpha · μ(c)`) used during fitting in
`neuro_fuzzy_gate.py`. The two formulas are not algebraically equivalent,
but both route through the same softmax-normalised, per-model gate
structure, and the deployed engine is the one that produces the
`fuzzy_ensemble` row in Chapter 4. If `base_models` does not match the
fitted gate's model set, `FuzzyEnsembleSentimentEngine` falls back to a
separate, independently implemented static fuzzy-inference system
(`research/computational_intelligence/fuzzy/engine_integration.py` —
Gaussian membership functions, centroid defuzzification, min/max
t-norm/t-conorm) rather than the learned gate described above; the
`route_a_live_v1` configuration always uses the three matching base
models, so this fallback path is not exercised in the reported results.

As reported in Chapter 4, the deployed gate rarely overrides the
underlying base classifier's argmax on this corpus: a direct ablation on
a 40,000-comment sample (seed 42) shows the argmax changes on only 0.18%
of comments (71/40,000 — 33 corrections, 21 regressions, 17
wrong-to-wrong flips), so the resulting `fuzzy_ensemble` behaves close to
a pass-through of its base model; this is reported as an honest negative
result rather than concealed (§4.1; `research/ci/fuzzy_gate_ablation.py`).

## 3B.7 Temperature Scaling and Calibration

**Source:** `research/ci/temperature_scaling.py`.

Each of the five deployed model configurations (`logreg`, `svm`, `tfidf`,
the static ensemble, `meta_learner`) is independently post-hoc calibrated
by fitting a single scalar temperature *T*:

- Because the classical models expose class probabilities rather than
  raw logits, pseudo-logits are computed as `z = log(p + 1e-10)` from the
  model's output probability vector.
- *T* is fit by minimising the negative log-likelihood of the
  temperature-scaled softmax, `softmax(z / T)`, on the **validation**
  split, using `scipy.optimize.minimize_scalar(method="bounded")` (a
  bounded 1-D Brent-style search) over `T ∈ [0.1, 10.0]`.
- At inference, the fitted *T* is applied to rescale the model's output
  probabilities before they are used downstream (by the API, by the
  ensembles, or by the entropy-gated selective predictor).

Because rescaling by a positive scalar before softmax does not change the
argmax, temperature scaling never changes accuracy or macro-F1 — only the
confidence values — which is why Chapter 4 reports it as a calibration
mechanism strictly orthogonal to the label-accuracy metrics.

## 3B.8 Deployment Platform

**Backend** (`backend/`, Django): three apps.

- `app` — the sentiment-analysis core: YouTube comment fetching
  (`youtube_fetcher.py`, `youtube_scraper.py`), preprocessing
  (`youtube_preprocessor.py`), the sentiment engines
  (`sentiment_engines.py`), deep-learning model support
  (`deep_models.py`), and keyword-level aspect mining
  (`aspect_mining.py`). Exposed endpoints (`app/urls.py`) include
  `youtube/analyze/` (submit a video for analysis),
  `youtube/analysis/<video_id>/` and `youtube/analyses/` (retrieve
  results), and `youtube/health/` (health check).
- `app_api` — authentication: JWT issuance and refresh
  (`token/`, `token/refresh/`) via `djangorestframework-simplejwt`.
- `users` — user account models, serializers, and views.

**Frontend** (`frontend/src/`, React): the analyst-facing pages live under
`Views/Pages/`, including `Dashboard.jsx` (result overview),
`Monitoring.jsx` (live/ongoing analysis tracking), `Report.jsx`,
`Search.jsx` (video/query submission), and `Tables.jsx`, alongside
account pages for authentication. Shared UI, application state, and the
authentication context live in `Components/`, `context/`, and `utils/`.

**Communication.** The frontend authenticates against `app_api`'s JWT
endpoints, then makes authenticated REST calls to `app`'s
`youtube/analyze/` and `youtube/analysis*` endpoints to submit videos and
retrieve calibrated, uncertainty-annotated sentiment results, which the
Dashboard and Monitoring views render alongside confidence and
calibration metadata (verified by `Dashboard.test.jsx`,
`Monitoring.test.jsx`; see Appendix A).

## 3B.9 Chapter Summary

Every computational-intelligence component in this thesis — PSO,
NSGA-II, the stacked meta-learner, and the neuro-fuzzy gate — operates
over the same three TF-IDF-based classical learners, differing only in
*how* their outputs are combined: fixed single-objective weights (PSO),
fixed multi-objective knee-point weights (NSGA-II), a learned non-linear
combiner trained on out-of-fold predictions (meta-learner), or a
per-sample adaptive fuzzy-inference weighting (neuro-fuzzy gate). All
four are calibrated identically via per-model temperature scaling, and
all are served through the same Django/React deployment so that the
Chapter 4 evaluation is a like-for-like comparison of combination
strategies rather than of different underlying data or preprocessing.
