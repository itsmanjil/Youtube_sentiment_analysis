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

## Train a transformer encoder baseline

```bash
cd backend
python research/transformers/train_encoder.py \
  --model_preset modernbert \
  --train_csv data/train.csv \
  --val_csv data/val.csv \
  --test_csv data/test.csv \
  --overwrite_output_dir
```

Notes:
- Defaults to the canonical `text` column from the split builder.
- If you regenerate splits for transformer training, prefer:
  `python scripts/prepare/prepare_hf_dataset.py --youtube_preprocess --primary_text_profile transformer`
- Artifacts are written to `backend/models/transformers/<preset>/`.

## Calibrate a trained encoder

```bash
cd backend
python research/transformers/calibrate_encoder.py \
  --model_preset modernbert \
  --val_csv data/val.csv \
  --test_csv data/test.csv
```

Notes:
- Writes `temperature_scaling.json` into the encoder artifact directory.
- Runtime inference loads that file automatically when the API uses the matching transformer preset.

## Export a probability cube for CI experiments

```bash
cd backend
python research/transformers/export_prob_cube.py \
  --data_csv data/test.csv \
  --models modernbert,deberta_v3 \
  --calibration_profile auto
```

Notes:
- Writes a compressed `.npz` cube plus a companion `.json` summary under `backend/results/prob_cubes/` by default.
- The cube stores `prob_cube`, `model_names`, `labels`, `y_true`, and optional `logits_cube`.
- With mixed model families, `--text_column auto` uses `text_transformer` for transformer presets and `text_classical` for classical models when those columns exist.
- CI scripts can now consume those artifacts directly:
  - `python research/ci/multi_objective_ensemble.py --val_cube results/prob_cubes/val_cube.npz --test_cube results/prob_cubes/test_cube.npz`
  - `python research/ci/neuro_fuzzy_gate.py --val_cube results/prob_cubes/val_cube.npz --test_cube results/prob_cubes/test_cube.npz`

## Run the full Route A benchmark pipeline

```bash
cd backend
python research/route_a/run_benchmark_pipeline.py \
  --split_dir data/route_a_benchmark_cpu \
  --model_preset deberta_v3 \
  --run_tag route_a_benchmark_cpu \
  --overwrite_output_dir
```

Notes:
- Expects a prepared split directory with `train.csv`, `val.csv`, and `test.csv`.
- Runs encoder training, temperature scaling, mixed-model probability-cube export, NSGA-II, neuro-fuzzy gating, and paired significance testing.
- Writes grouped artifacts under `backend/results/route_a_runs/<run_tag>/`.
- For GPU or long-running jobs, point `--split_dir` at the larger transformer-profile split and raise `--epochs`, `--nsga_pop`, and `--nsga_gen`.

## Sweep multiple Route A encoders on the same split

```bash
cd backend
python research/route_a/run_encoder_sweep.py \
  --split_dir data/route_a_transformer_10k \
  --models deberta_v3,modernbert \
  --run_prefix route_a_10k \
  --epochs 3 \
  --batch_size 16 \
  --eval_batch_size 32 \
  --nsga_pop 32 \
  --nsga_gen 32 \
  --overwrite_output_dir
```

Notes:
- Runs the full Route A pipeline once per encoder preset, then writes an aggregate summary to `backend/results/route_a_sweeps/<run_prefix>/`.
- The summary highlights encoder strength, calibrated ECE, CI knee-point metrics, neuro-fuzzy metrics, encoder weight usage inside CI, and significance against the strongest classical baseline on that split.

## Benchmark the pinned live runtime stack

```bash
cd backend
python research/ci/live_runtime_benchmark.py \
  --data data/test.csv \
  --text_column text \
  --label_column label
```

Notes:
- Uses the currently pinned runtime artifact manifest under `backend/results/runtime/<version>/manifest.json`.
- Evaluates the live inference engines, not offline-only research objects.
- Writes thesis-facing outputs to `backend/results/runtime/<version>/live_runtime_benchmark_full_test.json` and `.md`.
- The default model set is `tfidf,logreg,svm,ensemble:pso,ensemble:nsga2,meta_learner,fuzzy_ensemble`.

## Reconcile historical offline and pinned live benchmarks

```bash
cd backend
python research/ci/reconcile_live_vs_offline.py
```

Notes:
- Compares `backend/results/thesis_model_performance_youtube_filtered.md` with the pinned live benchmark JSON.
- Writes `offline_vs_live_reconciliation.json` and `.md` into `backend/results/runtime/<version>/`.
- This is the thesis-facing artifact to cite when explaining differences between historical offline tables and current live runtime behavior.

## Create a CPU-feasible benchmark subset

```bash
cd backend
python scripts/prepare/create_split_subset.py \
  --input_dir data/route_a_transformer \
  --output_dir data/route_a_benchmark_cpu \
  --per_label_train 1000 \
  --per_label_val 250 \
  --per_label_test 250
```

Notes:
- Preserves all columns from the parent split, including `text_transformer`.
- Writes `subset_metadata.json` so pilot benchmarks remain reproducible.

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
