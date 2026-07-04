# Neuro-Fuzzy Gate Ablation

- Dataset: `data/test.csv`
- Sample: 40,000 comments, seed 42
- Base model compared against: `tfidf`

## Result

The gate changes the base classifier's argmax label on **71 of 40,000 comments (0.18%)**:

| Outcome | Count |
|---------|------:|
| Corrections (base wrong -> gate correct) | 33 |
| Regressions (base correct -> gate wrong) | 21 |
| Wrong-to-wrong flips (both wrong, different label) | 17 |
| **Total changed** | **71** |

This reproduces and quantifies the thesis claim that the neuro-fuzzy gate behaves as a near pass-through of its base classifier on this corpus: corrections and regressions are of comparable magnitude, and a share of the changed labels are wrong-to-wrong flips that affect neither accuracy nor macro-F1 materially.