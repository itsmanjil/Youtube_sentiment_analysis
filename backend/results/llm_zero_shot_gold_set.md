# LLM Zero-Shot Baseline (Gold Set)

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Data: `data\gold_set_human_reconciled.csv`
- Samples: 291
- Generated at (UTC): 2026-07-19T02:32:09.716440+00:00

| Model | Accuracy | Macro-F1 | ECE | Brier |
| --- | --- | --- | --- | --- |
| llm_zero_shot (Qwen/Qwen2.5-1.5B-Instruct) | 0.6460 | 0.6074 | 0.299284 | 0.623435 |

Compare against the fine-tuned models' rows in `results/master_results_table.md` / `results/gold_set_evaluation.md` — this baseline has seen none of the project's training data.
