# YouTube Sentiment Analysis

A research project analyzing sentiment in YouTube comments: classical ML baselines (TF-IDF, Logistic Regression, SVM), ensemble optimization (NSGA-II/PSO), neuro-fuzzy gating, a hybrid CNN-BiLSTM-Attention model, and a DeBERTa-v3 transformer route — with statistical significance testing, seed/fold variance analysis, and error analysis. The narrative lives in `notebooks/`; the underlying research code lives in `backend/research/` and `backend/src/`.

## Stack

- Research: scikit-learn, NLTK, PyTorch/Transformers (optional, `requirements-dl.txt`)
- Presentation: Jupyter notebooks (`notebooks/`)
- CI: GitHub Actions in `.github/workflows/ci.yml`

## Repository Layout

```text
.
|- backend/
|  |- data/                 # Datasets and split artifacts
|  |- docs/                 # Thesis and project documentation
|  |- figures/              # Generated plots and figures
|  |- files/                # Text resources used by preprocessing
|  |- models/               # Trained model artifacts
|  |- research/             # Research/experiment code (baselines, ensembles, evaluation)
|  |- results/              # Generated reports and benchmark outputs
|  |- scripts/              # Data-prep utilities
|  |- src/                  # Reusable preprocessing and sentiment engines
|  |- requirements.txt      # Research + notebook dependencies
|  `- requirements-dl.txt   # Deep-learning extras (Hybrid-DL / Transformers)
|- notebooks/                # The research narrative — start here
|  |- 01_dataset_and_preprocessing.ipynb
|  |- 02_baselines.ipynb
|  |- 03_proposed_model.ipynb
|  |- 04_evaluation_and_significance.ipynb
|  `- 05_error_analysis.ipynb
|- .github/workflows/ci.yml
`- README.md
```

## Quick Start

### Prerequisites

Python 3.11–3.12 (the pinned `numpy==1.26.4` ships wheels only up to CPython 3.12, so `pip install -r requirements.txt` will fail to resolve on 3.13+).

### Install and launch

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
cd ../notebooks
jupyter lab
```

Open any notebook in `notebooks/` — each one loads already-computed artifacts from `backend/results/` and `backend/figures/`, so they run in seconds without retraining anything.

For deep-learning/transformer work (training the hybrid model or DeBERTa-v3 route), additionally install `backend/requirements-dl.txt`. See `backend/research/README.md` for the underlying experiment scripts (baselines, ensemble optimization, transformer training, significance testing, etc.).

## Notebooks

| Notebook | Research question |
| --- | --- |
| `01_dataset_and_preprocessing.ipynb` | What data underlies this study, how was it labelled/cleaned/split, and how reliable are the labels? |
| `02_baselines.ipynb` | How do TF-IDF, Logistic Regression, SVM, and a simple ensemble perform? |
| `03_proposed_model.ipynb` | Does ensemble weighting, neuro-fuzzy gating, or a hybrid DL model improve on the baselines? |
| `04_evaluation_and_significance.ipynb` | Are the differences between models statistically significant and stable across seeds/folds? |
| `05_error_analysis.ipynb` | Where does the system fail, and does it generalize beyond the training distribution? |

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on every `push`/`pull_request`:

- `research-smoke` — split-leakage check and a temperature-scaling smoke run against the committed models/data.
- `notebooks-smoke` — executes every notebook end-to-end (`jupyter nbconvert --execute`), so a broken path or import fails CI instead of surfacing during a defense.

## Related Docs

- `backend/research/README.md` — how to run the underlying experiments
- `backend/docs/ARCHITECTURE.md`
- `backend/docs/THESIS_EXPERIMENT_GUIDE.md`
- `backend/README_THESIS.md`

## License

MIT
