# Visualization and Analysis Scripts

This directory contains scripts for generating publication-quality visualizations and comprehensive evaluation reports for the YouTube Sentiment Analysis thesis project.

## Scripts Overview

### 1. `plot_training_curves.py`
Generates training progress visualizations:
- Training/validation loss curves
- Training/validation accuracy curves
- Training/validation F1-macro curves
- Per-class F1 scores over epochs

**Usage:**
```bash
python plot_training_curves.py \
  --history ../output/thesis_full_gpu/training_history.json \
  --output ./plots
```

**Outputs:**
- `training_curves.png` - Combined 2x2 grid of all metrics
- `loss_curve.png` - Detailed loss plot
- `accuracy_curve.png` - Detailed accuracy plot
- `f1_curve.png` - Detailed F1-macro plot

---

### 2. `plot_confusion_matrix.py`
Generates confusion matrices for all models:
- Individual confusion matrices (normalized and counts)
- Comparative 2x2 grid showing all models

**Usage:**
```bash
python plot_confusion_matrix.py \
  --test_csv ../data/test.csv \
  --models logreg,svm,hybrid_dl,ensemble \
  --output ./plots
```

**Outputs:**
- `confusion_matrix_{model}.png` - Normalized confusion matrix
- `confusion_matrix_{model}_counts.png` - Raw count matrix
- `confusion_matrix_comparison.png` - Side-by-side comparison

---

### 3. `plot_model_comparison.py`
Creates comparative visualizations across models:
- Accuracy comparison bar chart
- F1-Macro comparison bar chart
- Per-class F1 comparison
- Model rankings table

**Usage:**
```bash
python plot_model_comparison.py \
  --results_dir ../results \
  --output ./plots
```

**Outputs:**
- `comparison_accuracy.png` - Accuracy comparison
- `comparison_f1_macro.png` - F1-macro comparison
- `comparison_dashboard.png` - Comprehensive dashboard

---

### 4. `generate_report.py`
Generates a comprehensive markdown evaluation report:
- Executive summary
- Training statistics
- Model comparison tables
- Key findings and insights

**Usage:**
```bash
python generate_report.py \
  --results_dir ../results \
  --experiment thesis_full_gpu \
  --output ../EVALUATION_REPORT.md
```

**Output:**
- `EVALUATION_REPORT.md` - Formatted markdown report

---

### 5. `generate_all.py` (Master Script)
Runs all visualization scripts in sequence.

**Usage:**
```bash
python generate_all.py \
  --experiment thesis_full_gpu \
  --output ./plots
```

This will generate ALL visualizations and the evaluation report in one command.

---

## Requirements

Install required packages:
```bash
pip install matplotlib seaborn numpy scipy pandas scikit-learn
```

## Typical Workflow

1. **After training completes**, run the master script:
```bash
cd research/visualization
python generate_all.py
```

2. **Check outputs:**
   - Plots: `./plots/`
   - Report: `../../EVALUATION_REPORT.md`

3. **For individual visualizations**, run specific scripts as needed.

## Output Directory Structure

```
plots/
├── training_curves.png
├── loss_curve.png
├── accuracy_curve.png
├── f1_curve.png
├── confusion_matrix_logreg.png
├── confusion_matrix_svm.png
├── confusion_matrix_hybrid_dl.png
├── confusion_matrix_comparison.png
├── comparison_accuracy.png
├── comparison_f1_macro.png
└── comparison_dashboard.png

backend/
└── EVALUATION_REPORT.md
```

## Tips

- Use `--help` on any script to see all available options
- All plots are saved at 300 DPI for publication quality
- Confusion matrices work best with ~1500 samples for speed
- The master script will skip tasks if required files are missing

## Example: Complete Pipeline

```bash
# 1. Train model (if not done already)
cd ../..
python research/train_hybrid_dl.py \
  --train_csv data/train_full_filtered.csv \
  --val_csv data/val_full_filtered.csv \
  --test_csv data/test_full_filtered.csv \
  --max_epochs 10 \
  --experiment_name thesis_full_gpu

# 2. Generate all visualizations
cd research/visualization
python generate_all.py

# 3. View results
start plots/comparison_dashboard.png
start ../../EVALUATION_REPORT.md
```
