# Research Utilities

This folder contains thesis-grade experiment utilities for model evaluation and
computational-intelligence ensemble optimization.

## Dataset format

Provide a CSV file with the following columns:

- `text`: the comment text
- `label`: sentiment label (`Positive`, `Neutral`, `Negative`)

## Run experiments

```bash
python backend/research/experiment_runner.py --data path/to/labeled.csv
```

Optional arguments:
- `--models logreg,svm,tfidf,ensemble`
- `--ensemble-models logreg,svm,tfidf`
- `--ensemble-weights '{"logreg": 0.3, "svm": 0.5, "tfidf": 0.2}'`
- `--output results.json`

## Optimize ensemble weights (PSO)

```bash
python backend/research/optimize_ensemble.py --data path/to/labeled.csv
```

Optional arguments:
- `--models logreg,svm,tfidf`
- `--particles 20`
- `--iterations 30`
- `--output optimized_weights.json`
