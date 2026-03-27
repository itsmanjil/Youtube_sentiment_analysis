# YouTube Sentiment Analysis - Master's Thesis Edition

## Overview

This is a thesis-grade sentiment analysis system for YouTube comments, implementing state-of-the-art methods in Natural Language Processing and Computational Intelligence.

## Key Features

### 🎯 Multiple Model Architectures
- **Classical ML**: TF-IDF + Naive Bayes, Logistic Regression, SVM
- **Deep Learning**: Hybrid CNN-BiLSTM-Attention (2.5M parameters)
- **Transformers**: Encoder baselines via `research/transformers/train_encoder.py` (`ModernBERT`, `DeBERTa-v3`, `XLM-V`, `mDeBERTa-v3`, `BERT`)
- **Ensemble Methods**: Weighted voting, Meta-learner stacking

### 🔬 Research-Grade Evaluation
- **Statistical Tests**: McNemar's, Wilcoxon, Friedman with post-hoc
- **Cross-Validation**: Stratified 10-fold CV
- **Confidence Intervals**: Bootstrap 95% CI
- **Ablation Studies**: Systematic component contribution analysis

### 🔍 Explainability (XAI)
- **SHAP**: Shapley value-based feature importance
- **LIME**: Local interpretable model-agnostic explanations
- **Attention Visualization**: Attention weight heatmaps

### 📊 Advanced Analytics
- Keyword-level sentiment aggregation (token-frequency aspect proxy; see `backend/app/aspect_mining.py`)
- Temporal sentiment dynamics
- Engagement-weighted analysis
- Ethical Bias Analysis (e.g., Dialect, Demographics)
- Confidence scoring (entropy-based)

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd Youtube_sentiment_analysis/backend

# Install dependencies
pip install -r requirements.txt

# Optional: deep learning engines (Hybrid-DL / Transformers)
# PyTorch wheels can be sensitive to the NumPy ABI. Prefer a separate env:
#   python -m venv venv-dl
#   source venv-dl/bin/activate
#   pip install -r requirements-dl.txt
#   pip install -r requirements.txt

# Install explainability tools
pip install shap lime
```

### Training Models

```bash
# Train classical ML models
python train_logreg_youtube.py --data data/train.csv --test_data data/test.csv
python train_svm_youtube.py --data data/train.csv --test_data data/test.csv
python train_tfidf_youtube.py --data data/train.csv --test_data data/test.csv

# Train deep learning model
python research/train_hybrid_dl.py --config research/config/hybrid_dl_config.yaml

# Train transformer encoder baseline
python research/transformers/train_encoder.py \
  --model_preset modernbert \
  --train_csv data/train.csv \
  --val_csv data/val.csv \
  --test_csv data/test.csv \
  --overwrite_output_dir

# Calibrate the trained encoder
python research/transformers/calibrate_encoder.py \
  --model_preset modernbert \
  --val_csv data/val.csv \
  --test_csv data/test.csv
```

### Dataset Leakage Check (Important)

Exact-duplicate texts across `train/val/test` can inflate reported metrics. This repo includes a checker:

```bash
python scripts/prepare/check_split_leakage.py \
  --train data/train.csv --val data/val.csv --test data/test.csv
```

For a prioritized thesis "risks/gaps" checklist (threats to validity + what to report), see:
`backend/docs/THESIS_RISKS_GAPS.md`.

When generating splits with `scripts/prepare/prepare_hf_dataset.py`, the pipeline:
- Drops texts with conflicting labels
- Optionally applies the same YouTube preprocessing used by the API (`--youtube_preprocess`)
- Deduplicates by final model-input text before splitting
- Uses group-aware splitting by `VideoID` when available

Split provenance is written to `data/split_metadata.json`.

To generate leakage-safe, API-aligned splits:

```bash
python scripts/prepare/prepare_hf_dataset.py --youtube_preprocess
```

For production-aligned filtering (slower), add `--filter_spam` and/or `--filter_language`.

### Significance Testing (Thesis-Grade)

After you have a fixed held-out `test.csv`, you can run paired significance tests and
bootstrap confidence intervals:

```bash
backend/venv/bin/python research/testset_significance.py \
  --data data/test.csv \
  --models tfidf,logreg,svm,ensemble,meta_learner \
  --ensemble-models logreg,svm,tfidf \
  --bootstrap 2000 --p_adjust holm --write_tables \
  --output results/testset_significance_youtube_filtered.json
```

This writes:
- `results/testset_significance_youtube_filtered.json` (McNemar + bootstrap CIs)
- `results/thesis_mcnemar.md` and `results/thesis_bootstrap_ci.md` (copy/paste tables)

### Gold Set (Human-Labeled Test)

```bash
# Create a small annotation template
python scripts/prepare/create_gold_set.py --input_csv data/train.csv --sample_size 300

# After manual labeling, use it as a held-out test set
python prepare_youtube_training_data.py \
  --video_list videos.txt \
  --label_method auto \
  --heldout_labeled_csv gold_set_labeled.csv

# Evaluate models on the human-labeled gold set and report agreement
python research/evaluate_gold_set.py \
  --data data/gold_set_labeled.csv \
  --models tfidf,logreg,svm,ensemble,meta_learner \
  --output results/gold_set_evaluation.json
```

This writes:
- `results/gold_set_evaluation.json` (model metrics + human-vs-source agreement when `source_label` exists)
- `results/gold_set_evaluation.md` (thesis-friendly summary table)

If you used `scripts/prepare/fill_gold_set_labels.py`, treat that output as a pipeline sanity check only, not as a thesis-grade human gold set.

### Using the API

```python
from src.sentiment import get_sentiment_engine

# Use classical model (fast)
engine = get_sentiment_engine('svm')
result = engine.analyze("This video is amazing!")
print(f"{result.label}: {result.score:.2f}")

# Use BERT transformer (if fine-tuned artifact is available)
engine = get_sentiment_engine('bert')
result = engine.analyze("This video is terrible")
print(f"{result.label}: {result.score:.2f}")
```

### Explainability

```python
from research.explainability import SHAPExplainer, LIMEExplainer

# SHAP explanation
explainer = SHAPExplainer(model, vectorizer)
explanation = explainer.explain("This video changed my life!")
explainer.visualize(explanation, save_path='shap_plot.png')

# LIME explanation
explainer = LIMEExplainer.from_sklearn_model(model, vectorizer)
explanation = explainer.explain("Terrible content, waste of time")
explainer.visualize(explanation, save_path='lime_plot.png')
```

### Statistical Comparison

```python
from research.evaluation import StatisticalSignificanceTester

tester = StatisticalSignificanceTester(alpha=0.05)

# Compare two models with McNemar's test
result = tester.mcnemars_test(y_true, pred_model_a, pred_model_b)
print(result['interpretation'])

# Compare 3+ models with Friedman test
scores = {
    'LogReg': [0.74, 0.75, 0.73, 0.76, 0.74],
    'SVM': [0.75, 0.76, 0.74, 0.77, 0.75],
    'BERT': [0.85, 0.86, 0.84, 0.87, 0.85],
}
result = tester.friedman_test(scores)
print(result['interpretation'])
```

### Ablation Studies

```python
from research.evaluation import AblationStudyFramework

ablation = AblationStudyFramework(
    base_model_fn=create_full_model,
    evaluation_fn=evaluate_on_test,
    metric_name='f1_macro'
)

# Add component ablations
ablation.add_component_ablation(
    'no_attention',
    lambda: create_model(use_attention=False),
    description="Remove multi-head attention mechanism"
)

# Run all experiments
results = ablation.run()
ablation.generate_report('ablation_results/')
```

## Model Performance

### Reproducible Benchmark (Current Repository Snapshot)

| Model | Accuracy | F1-Macro | F1-Pos | F1-Neu | F1-Neg |
|-------|----------|----------|--------|--------|--------|
| TF-IDF + NB | 66.22% | 65.67% | 72.18% | 56.21% | 68.62% |
| TF-IDF + LogReg | 69.46% | 69.28% | 75.10% | 62.02% | 70.74% |
| TF-IDF + SVM | 68.01% | 67.80% | 73.85% | 60.12% | 69.43% |
| Ensemble | 69.32% | 69.09% | 74.98% | 61.51% | 70.79% |
| Meta-learner | **69.55%** | **69.46%** | **75.25%** | **62.68%** | 70.45% |

Source artifacts:
- `results/thesis_model_performance_youtube_filtered.md`
- `results/thesis_per_class_f1_youtube_filtered.md`

These numbers are from a fixed held-out test set (`data/test.csv` in this snapshot), not 10-fold CV.

### Statistical Significance

From `results/thesis_mcnemar.md` (Holm-adjusted McNemar):

- LogReg vs SVM: `p_adj = 1.81e-92` (significant)
- LogReg vs Meta-learner: `p_adj = 0.045` (significant at α = 0.05)
- Ensemble vs Meta-learner: `p_adj = 4.15e-05` (significant)

Bootstrap confidence intervals are in `results/thesis_bootstrap_ci.md`.

### Reproducibility Notes

- Treat `results/thesis_*` files as the thesis-ready, source-of-truth tables.
- If you retrain or change preprocessing, regenerate tables before updating thesis text.
- Avoid reporting BERT/Hybrid-DL headline metrics unless their held-out test artifacts are saved alongside the classical/ensemble results.
- Use `research/run_thesis_pipeline.py --command_log results/experiment_command_log.txt` to append exact run commands.
- Package each finalized run with:

```bash
python research/create_repro_bundle.py \
  --command_file results/experiment_command_log.txt \
  --artifact data/split_metadata.json \
  --artifact results/testset_significance_youtube_filtered.json \
  --artifact results/thesis_model_performance_youtube_filtered.md \
  --artifact results/thesis_per_class_f1_youtube_filtered.md
```

This writes a timestamped bundle in `results/repro_bundles/` with `manifest.json`,
`commands.sh`, `pip_freeze.txt`, and `artifacts.sha256`.

## Architecture

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed system architecture.

### High-Level Overview

```
Input → Preprocessing → Model Selection → Inference → Analytics → Output
  ↓         ↓              ↓                ↓            ↓          ↓
YouTube   Spam      Classical/DL/    Probability   Aspects/   JSON
  API    Detection   Transformer      + Confidence  Timeline   Response
```

## Directory Structure

```
backend/
├── src/                          # Core application
│   ├── sentiment/                # Sentiment engines
│   ├── preprocessing/            # Text preprocessing
│   ├── services/                 # Business logic
│   └── utils/                    # Utilities
├── research/                     # Thesis components
│   ├── architectures/            # Neural architectures
│   ├── evaluation/               # Statistical tests
│   ├── explainability/           # XAI module
│   └── training/                 # Training infrastructure
├── scripts/                      # Training scripts
├── tests/                        # Test suite
├── models/                       # Trained models
└── docs/                         # Documentation
```

## Research Contributions

### 1. Novel Architecture
Hybrid CNN-BiLSTM-Attention model combining:
- Multi-scale CNN for n-gram pattern extraction
- Bidirectional LSTM for sequential dependencies
- Multi-head attention for focus mechanism

### 2. Rigorous Evaluation
- Statistical significance testing (McNemar's, Wilcoxon, Friedman)
- Bootstrap confidence intervals
- Ablation studies proving component contributions
- Cross-domain validation

### 3. Explainability
- SHAP and LIME for model transparency
- Attention weight visualization
- Token-level importance scoring

### 4. Production-Ready System
- RESTful API with Django
- Multiple model backends
- Efficient batch processing
- Comprehensive error handling

## Thesis Structure

### Recommended Chapters

1. **Introduction**
   - Background on sentiment analysis
   - YouTube-specific challenges
   - Research objectives

2. **Literature Review**
   - Classical ML approaches (TF-IDF, SVM)
   - Deep learning for NLP
   - Transformers (BERT, RoBERTa)
   - Explainability in NLP

3. **Methodology**
   - System architecture
   - Model formulations (mathematical)
   - Evaluation framework
   - Implementation details

4. **Experiments**
   - Dataset description
   - Baseline comparison
   - Ablation studies
   - Statistical analysis

5. **Results and Discussion**
   - Performance analysis
   - Explainability insights
   - Limitations

7. **Ethical Considerations**
   - Data Privacy and Anonymization
   - Algorithmic Bias Analysis
   - Potential for Misuse and Mitigation

8. **Conclusion**
   - Contributions
   - Future work

## Citation

If you use this system in your research, please cite:

```bibtex
@mastersthesis{your_thesis,
  author  = {FILL IN: Your Full Name},
  title   = {FILL IN: Your Thesis Title},
  school  = {FILL IN: Your University},
  year    = {2026},
  type    = {Master's thesis}
}
```

## References

1. Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT.

2. Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.

3. Ribeiro et al. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. KDD.

4. Demsar (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. JMLR.

5. Vaswani et al. (2017). Attention Is All You Need. NeurIPS.

## License

This project is intended for academic research purposes.

## Contact

For questions or collaboration, refer to the GitHub repository linked at the top of this file.

## Acknowledgments

- Pre-trained BERT models from HuggingFace
- GloVe embeddings from Stanford NLP Group
- SHAP library from Scott Lundberg
- LIME library from Marco Tulio Ribeiro
