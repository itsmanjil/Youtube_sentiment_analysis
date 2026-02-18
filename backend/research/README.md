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

## Evaluate a human gold set

```bash
python backend/research/evaluate_gold_set.py --data backend/data/gold_set_labeled.csv
```

Optional arguments:
- `--models tfidf,logreg,svm,ensemble,meta_learner`
- `--output backend/results/gold_set_evaluation.json`
- `--summary_md backend/results/gold_set_evaluation.md`

## Optimize ensemble weights (PSO)

```bash
python backend/research/optimize_ensemble.py --data path/to/labeled.csv
```

Optional arguments:
- `--models logreg,svm,tfidf`
- `--particles 20`
- `--iterations 30`
- `--output optimized_weights.json`

## Create reproducibility bundle

```bash
cd backend
python research/create_repro_bundle.py \
  --command_file results/experiment_command_log.txt \
  --artifact data/split_metadata.json \
  --artifact results/testset_significance_youtube_filtered.json
```

Optional arguments:
- `--artifact path/or/glob` (repeat to include more artifacts)
- `--bundle_name thesis_run_v1`
- `--output_dir results/repro_bundles`
- `--notes "Any thesis annotation notes"`
