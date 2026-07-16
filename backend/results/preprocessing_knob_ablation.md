# Preprocessing Knob Ablation

Data: balanced subsample of `train.csv`  (train=14,400, val=3,600)  |  seed=42  |  pipeline=TF-IDF(1,2) + LogReg(balanced)

> Full 2³ ablation of `ClassicalPreprocessConfig` knobs. Each row trains an independent TF-IDF + LogReg pipeline. This complements `thesis_preprocess_ablation.md`, which ablates dataset-level cleaning stages, not preprocessing knobs.

| Expand neg | Negation tag | Remove stopw | |V| | Accuracy | F1-macro | ΔF1 vs baseline |
|:---:|:---:|:---:|---:|---:|---:|---:|
| · | · | · | 30,000 | 0.6339 | 0.6349| +0.0000 |
| · | · | ✓ | 26,605 | 0.6350 | 0.6355| +0.0006 |
| · | ✓ | · | 30,000 | 0.6286 | 0.6300| -0.0049 |
| · | ✓ | ✓ | 26,217 | 0.6256 | 0.6264| -0.0085 |
| ✓ | · | · | 30,000 | 0.6336 | 0.6346| -0.0003 |
| ✓ | · | ✓ | 26,595 | 0.6361 | 0.6367 **←** | +0.0018 |
| ✓ | ✓ | · | 30,000 | 0.6317 | 0.6329| -0.0020 |
| ✓ | ✓ | ✓ | 26,275 | 0.6258 | 0.6268| -0.0081 |

## Interpretation

- **Baseline** (all knobs off): F1 = 0.6349.
- **Best configuration**: expand=✓, neg_tag=·, rm_stop=✓, F1 = 0.6367 (**+0.0018** over baseline).

### Main effects (mean F1 with knob on − mean F1 with knob off)

- expand_negation_contractions: **+0.0010**
- negation_tag: **-0.0064**
- remove_stopwords: **-0.0018**

If the improvements are ≤ 0.005 F1 the knobs should be described in the thesis as *behaviour-preserving* rather than accuracy-enhancing — they stabilise preprocessing across training and inference (no train/inference skew) at essentially zero cost, which is the real contribution documented in `preprocessing_consistency_audit.md`.
