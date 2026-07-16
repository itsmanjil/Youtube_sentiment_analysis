# PSO Convergence Analysis

## 1. Overview

Particle Swarm Optimization (PSO) is a bio-inspired metaheuristic that
iteratively improves a population of candidate solutions (particles) by
combining individual memory (personal best) with collective memory (global best).
Here it optimises the weight vector of a soft-voting ensemble over three
base classifiers (LogReg, SVM, TF-IDF) to maximise Macro-F1 on the validation set.

## 2. Convergence History

| Iteration | Global Best F1 | Mean Personal Best F1 | Improvement |
|-----------|---------------|----------------------|-------------|
|   1 | 0.690356 | 0.684352 | 0.000000 |
|   2 | 0.690651 | 0.687458 | +0.000295 |
|   3 | 0.690659 | 0.689727 | +0.000008 |
|   4 | 0.691326 | 0.690079 | +0.000667 |
|   5 | 0.691326 | 0.690230 | 0.000000 |
|   6 | 0.691326 | 0.690374 | 0.000000 |
|   7 | 0.691326 | 0.690544 | 0.000000 |
|   8 | 0.691342 | 0.690785 | +0.000016 |
|   9 | 0.691342 | 0.690922 | 0.000000 |
|  10 | 0.691374 | 0.691005 | +0.000032 |
|  11 | 0.691486 | 0.691100 | +0.000112 |
|  12 | 0.691486 | 0.691122 | 0.000000 |
|  13 | 0.691486 | 0.691246 | 0.000000 |
|  14 | 0.691486 | 0.691255 | 0.000000 |
|  15 | 0.691551 | 0.691293 | +0.000065 |
|  16 | 0.691551 | 0.691302 | 0.000000 |
|  17 | 0.691551 | 0.691348 | 0.000000 |
|  18 | 0.691551 | 0.691380 | 0.000000 |
|  19 | 0.691551 | 0.691383 | 0.000000 |
|  20 | 0.691551 | 0.691406 | 0.000000 |
|  21 | 0.691551 | 0.691406 | 0.000000 |
|  22 | 0.691551 | 0.691406 | 0.000000 |
|  23 | 0.691551 | 0.691406 | 0.000000 |
|  24 | 0.691551 | 0.691408 | 0.000000 |
|  25 | 0.691551 | 0.691423 | 0.000000 |
|  26 | 0.691551 | 0.691437 | 0.000000 |
|  27 | 0.691551 | 0.691447 | 0.000000 |
|  28 | 0.691551 | 0.691448 | 0.000000 |
|  29 | 0.691554 | 0.691452 | +0.000003 |
|  30 | 0.691554 | 0.691455 | 0.000000 |
|  31 | 0.691554 | 0.691459 | 0.000000 |
|  32 | 0.691554 | 0.691459 | 0.000000 |
|  33 | 0.691554 | 0.691487 | 0.000000 |
|  34 | 0.691554 | 0.691488 | 0.000000 |
|  35 | 0.691554 | 0.691502 | 0.000000 |
|  36 | 0.691554 | 0.691507 | 0.000000 |
|  37 | 0.691554 | 0.691510 | 0.000000 |
|  38 | 0.691554 | 0.691510 | 0.000000 |
|  39 | 0.691554 | 0.691510 | 0.000000 |
|  40 | 0.691554 | 0.691510 | 0.000000 |

> **Convergence:** PSO reaches 99% of its final score by **iteration 1**,
> demonstrating rapid convergence in this low-dimensional (3D) weight space.

## 3. Optimised Weights

| Model | PSO Weight | Uniform Weight | Random Search Weight |
|-------|-----------|----------------|---------------------|
| logreg | 0.880485 | 0.333333 | 0.789913 |
| svm | 0.000000 | 0.333333 | 0.112978 |
| tfidf | 0.119515 | 0.333333 | 0.097109 |

## 4. Method Comparison

| Method | Val Macro-F1 | Description |
|--------|-------------|-------------|
| Uniform weights | 0.6875 | Equal weight to all models |
| Random search (200 trials) | 0.6913 | Random weight sampling |
| **PSO (20 particles, 40 iters)** | **0.6916** | Bio-inspired optimisation |

**PSO test Macro-F1:** 0.6941

> PSO beats uniform weighting by +0.0040 val Macro-F1. Against 200-trial random search the margin is +0.0002 — small enough that no significance test was run to claim PSO reliably beats random search on this 3-parameter simplex; the honest reading is that PSO matches or slightly exceeds random search here, and its role in the thesis is as the single-objective baseline that NSGA-II's multi-objective search is compared against (see results/runtime/route_a_live_v1/multi_objective_ensemble.md), not as a method proven superior to random search in isolation.

## 5. Why PSO Over Grid Search?

| Aspect | Grid Search | PSO |
|--------|------------|-----|
| Search space coverage | Discrete, finite | Continuous [0, 1]³ |
| Scalability to more models | Exponential | Linear in particles |
| Exploitation of landscape | None (blind) | Guided by personal+global best |
| Convergence guarantee | Exhaustive | Asymptotic (probabilistic) |
| Reproducibility | Full | Seed-controlled |

For a 3-model continuous weight space, grid search at resolution 0.1
requires (10+1)³ = 1,331 evaluations vs PSO's 20×40 = 800 — fewer
evaluations while searching a finer-grained continuous space.

## 6. PSO Hyperparameter Justification

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Particles | 20 | Standard for low-dimensional problems (Kennedy & Eberhart 1995) |
| Iterations | 40 | Convergence observed at ~10 iterations (Table 2) |
| Inertia (ω) | 0.7 | Balances exploration/exploitation (Shi & Eberhart 1998) |
| Cognitive (c1) | 1.4 | Standard value from literature |
| Social (c2) | 1.4 | Standard value from literature |

## 7. Thesis Framing

> We apply Particle Swarm Optimization (Kennedy & Eberhart, 1995) to the
> ensemble weight optimisation problem, treating the three-dimensional weight
> vector (LogReg, SVM, TF-IDF) as a continuous search space and maximising
> validation Macro-F1. A swarm of 20 particles over 40 iterations converges to
> a stable optimum (see the convergence table above), yielding weights of
> LogReg=0.880, SVM=0.000,
> TF-IDF=0.120. PSO achieves a validation Macro-F1
> of 0.6916, a +0.0040 improvement over uniform
> weighting (0.6875). Against 200-trial random search (0.6913) the
> margin is +0.0002; PSO's role in this thesis is as the
> single-objective baseline against which the multi-objective NSGA-II ensemble
> (§4.3 / multi_objective_ensemble.md) is compared, not as a method independently
> proven superior to random search on this low-dimensional problem.

## 8. References

- Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
  *Proceedings of ICNN*, 4, 1942–1948.
- Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer.
  *Proceedings of IEEE ICEC*, 69–73.