# YouTube Sentiment Analysis - Thesis Evaluation Report
Generated: 2026-01-20 15:22:43
---

## Executive Summary

## Training Statistics

- Total Epochs: 1
- Best Epoch: 1
- Best Val F1-Macro: 67.11%
- Average Epoch Time: 12.5 minutes
- Total Training Time: 12.5 minutes

## Model Comparison

| Model | Accuracy | F1-Macro | F1-Neg | F1-Neu | F1-Pos |
|-------|----------|----------|--------|--------|--------|
| LOGREG | 74.27% | 74.34% | 75.12% | 70.25% | 77.65% |
| SVM | 75.08% | 75.14% | 76.10% | 70.94% | 78.39% |
| TFIDF | 67.71% | 67.70% | 68.26% | 61.99% | 72.83% |
| ENSEMBLE | 75.08% | 75.14% | - | - | - |

**Best Model:** SVM with F1-Macro = 75.14%

## Key Findings

1. **Model Performance:**

2. **Training Observations:**
   - Model generalizes well (train-val gap: -3.07%)
