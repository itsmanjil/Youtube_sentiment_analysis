# Live Runtime Benchmark

- Runtime artifact version: `route_a_live_v1`
- Dataset: `/Users/deadshot/Documents/GitHub/Youtube_sentiment_analysis/backend/data/test.csv`
- Text column: `text`
- Samples: `100`

| Model | Accuracy | Macro-F1 | ECE | Brier | Calibrated | Temp | Weights | NF Gate | ms/sample |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | ---: |
| logreg | 0.7000 | 0.6964 | 0.092070 | 0.443187 | yes | 1.0311 | — | no | 0.1287 |
| ensemble_nsga2 | 0.6900 | 0.6865 | 0.081799 | 0.443767 | yes | 0.9348 | nsga2 | no | 0.4110 |
| ensemble_pso | 0.6600 | 0.6536 | 0.114782 | 0.466338 | yes | 0.9348 | pso | no | 0.4889 |
| meta_learner | 0.6500 | 0.6500 | 0.140027 | 0.452315 | yes | 0.9835 | — | no | 0.6334 |
| svm | 0.6400 | 0.6308 | 0.153811 | 0.480375 | yes | 1.0326 | — | no | 0.1365 |
| tfidf | 0.5600 | 0.5503 | 0.147944 | 0.489045 | yes | 1.1306 | — | no | 0.2095 |
| fuzzy_ensemble | 0.5600 | 0.5488 | 0.147861 | 0.488374 | no | — | — | yes | 0.4520 |
