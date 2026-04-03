# Live Runtime Benchmark

- Runtime artifact version: `route_a_live_v1`
- Dataset: `/Users/deadshot/Documents/GitHub/Youtube_sentiment_analysis/backend/data/test.csv`
- Text column: `text`
- Samples: `165110`

| Model | Accuracy | Macro-F1 | ECE | Brier | Calibrated | Temp | Weights | NF Gate | ms/sample |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | ---: |
| meta_learner | 0.6953 | 0.6945 | 0.015711 | 0.411713 | yes | 0.9835 | — | no | 0.4790 |
| ensemble_nsga2 | 0.6959 | 0.6940 | 0.004601 | 0.409204 | yes | 0.9348 | nsga2 | no | 0.4891 |
| logreg | 0.6946 | 0.6928 | 0.003900 | 0.410009 | yes | 1.0311 | — | no | 0.1259 |
| ensemble_pso | 0.6872 | 0.6852 | 0.011272 | 0.419490 | yes | 0.9348 | pso | no | 0.3514 |
| svm | 0.6801 | 0.6780 | 0.016953 | 0.429259 | yes | 1.0326 | — | no | 0.1058 |
| tfidf | 0.6622 | 0.6567 | 0.017889 | 0.449058 | yes | 1.1306 | — | no | 0.1188 |
| fuzzy_ensemble | 0.6622 | 0.6567 | 0.018516 | 0.448920 | no | — | — | yes | 0.5233 |
