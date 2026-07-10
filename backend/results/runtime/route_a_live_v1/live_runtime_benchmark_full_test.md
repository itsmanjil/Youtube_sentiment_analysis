# Live Runtime Benchmark

- Runtime artifact version: `route_a_live_v1`
- Dataset: `C:\Users\itsma\OneDrive\Documents\GitHub\Youtube_sentiment_analysis\backend\data\test.csv`
- Text column: `text`
- Samples: `165110`

| Model | Accuracy | Macro-F1 | ECE | Brier | Calibrated | Temp | Weights | NF Gate | ms/sample |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | ---: |
| meta_learner | 0.6955 | 0.6946 | 0.018303 | 0.411778 | yes | 1.0000 | — | no | 0.1865 |
| ensemble_pso | 0.6961 | 0.6941 | 0.006103 | 0.409001 | yes | 1.0000 | pso | no | 0.1816 |
| ensemble_nsga2 | 0.6959 | 0.6940 | 0.003919 | 0.409162 | yes | 1.0000 | nsga2 | no | 0.1810 |
| fuzzy_ensemble | 0.6960 | 0.6940 | 0.002987 | 0.409288 | no | — | — | yes | 0.2035 |
| logreg | 0.6946 | 0.6928 | 0.003900 | 0.410009 | yes | 1.0311 | — | no | 0.0570 |
| svm | 0.6801 | 0.6780 | 0.015690 | 0.429094 | yes | 1.0000 | — | no | 0.0549 |
| tfidf | 0.6622 | 0.6567 | 0.017889 | 0.449058 | yes | 1.1306 | — | no | 0.0599 |
