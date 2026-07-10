# Neuro-Fuzzy Gate Ablation

- Dataset: `data/test.csv`
- Sample: 40,000 comments, seed 42
- Base model compared against: `logreg`

## Result

The gate changes the base classifier's argmax label on **1096 of 40,000 comments (2.74%)**:

| Outcome | Count |
|---------|------:|
| Corrections (base wrong -> gate correct) | 456 |
| Regressions (base correct -> gate wrong) | 412 |
| Wrong-to-wrong flips (both wrong, different label) | 228 |
| **Total changed** | **1096** |

This reproduces and quantifies the thesis claim that the neuro-fuzzy gate behaves as a near pass-through of its base classifier on this corpus: corrections and regressions are of comparable magnitude, and a share of the changed labels are wrong-to-wrong flips that affect neither accuracy nor macro-F1 materially.