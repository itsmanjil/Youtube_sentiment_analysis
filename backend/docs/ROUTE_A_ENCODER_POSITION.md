# Route A Encoder Position

Status date: 2026-05-18

Route A remains implemented as a runnable research path, but it should not be
the main thesis claim unless a full encoder run is executed in an environment
with `transformers`, `torch`, and suitable compute.

## Current Position

- The final thesis headline is classical/ensemble-first.
- Encoder tooling exists under `research/transformers/` and
  `research/route_a/`.
- The current local environment does not include the transformer runtime
  dependencies needed to rerun DeBERTa/ModernBERT end to end.
- Existing encoder artifacts may be cited only as stored experimental evidence,
  not as a freshly reproduced full Route A result from this pass.

## Future Work Command Path

```bash
cd backend
python research/route_a/run_benchmark_pipeline.py \
  --split_dir data/route_a_transformer_10k \
  --model_preset deberta_v3 \
  --run_tag route_a_10k_deberta_v3 \
  --epochs 3 \
  --overwrite_output_dir
```

For a stronger encoder-first thesis, run the sweep:

```bash
cd backend
python research/route_a/run_encoder_sweep.py \
  --split_dir data/route_a_transformer_10k \
  --models deberta_v3,modernbert \
  --run_prefix route_a_10k \
  --epochs 3 \
  --overwrite_output_dir
```

