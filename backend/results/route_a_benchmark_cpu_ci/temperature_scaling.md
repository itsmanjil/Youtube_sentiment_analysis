# Temperature Scaling — Route A (DeBERTa + Classical)

| Model | T | ECE Before | ECE After | Δ ECE% | Macro-F1 |
|-------|---|------------|-----------|--------|----------|
| deberta_v3 | 1.000 | 0.1084 | 0.1084 | -0.0% | 0.6579 |
| logreg | 0.861 | 0.0789 | 0.0621 | +21.2% | 0.7544 |
| svm | 0.735 | 0.1004 | 0.0649 | +35.3% | 0.7863 |