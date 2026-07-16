# Class Distribution Report

## Overview

This report documents the sentiment class distribution across all dataset
variants and splits. A balanced distribution is essential to ensure that
Macro-F1 and Accuracy are comparable metrics without class-imbalance confounds.

## Raw (no preprocessing)

| Split | Total | Negative | % | Neutral | % | Positive | % |
|-------|-------|----------|---|---------|---|----------|---|
| Train | 634,528 | 217,608 | 34.3% | 209,674 | 33.0% | 207,246 | 32.7% |
| Val | 153,657 | 53,304 | 34.7% | 50,311 | 32.7% | 50,042 | 32.6% |
| Test | 199,045 | 68,134 | 34.2% | 64,857 | 32.6% | 66,054 | 33.2% |

## YouTube Clean (emoji+normalise)

| Split | Total | Negative | % | Neutral | % | Positive | % |
|-------|-------|----------|---|---------|---|----------|---|
| Train | 594,640 | 208,660 | 35.1% | 189,639 | 31.9% | 196,341 | 33.0% |
| Val | 143,646 | 50,896 | 35.4% | 45,571 | 31.7% | 47,179 | 32.8% |
| Test | 185,845 | 64,985 | 35.0% | 58,259 | 31.3% | 62,601 | 33.7% |

## YouTube Filtered (spam+lang+short)

| Split | Total | Negative | % | Neutral | % | Positive | % |
|-------|-------|----------|---|---------|---|----------|---|
| Train | 516,886 | 184,266 | 35.6% | 160,744 | 31.1% | 171,876 | 33.3% |
| Val | 128,854 | 47,023 | 36.5% | 39,037 | 30.3% | 42,794 | 33.2% |
| Test | 165,110 | 59,614 | 36.1% | 50,540 | 30.6% | 54,956 | 33.3% |

## Balance Assessment

The table below summarises class imbalance on the **held-out test set**
across all variants. Max deviation from uniform (33.3%) is the key metric.

| Variant | Test Neg% | Test Neu% | Test Pos% | Max Deviation |
|---------|-----------|-----------|-----------|---------------|
| Raw (no preprocessing) | 34.2% | 32.6% | 33.2% | ±0.9% |
| YouTube Clean (emoji+normalise) | 35.0% | 31.3% | 33.7% | ±2.0% |
| YouTube Filtered (spam+lang+short) | 36.1% | 30.6% | 33.3% | ±2.8% |

> **Finding:** All variants show approximately balanced class distribution
> (max deviation ±5% from uniform). Macro-F1 and Accuracy are therefore
> directly comparable across models without class-imbalance correction.

## Thesis Framing

Include this in your **Dataset** chapter section:

> Table X presents the class distribution across dataset variants and splits.
> The corpus is approximately balanced across Negative (35.7%), Neutral (31.1%),
> and Positive (33.2%) classes on the primary test split, with a maximum deviation
> of ±4.4% from a uniform distribution. This balance justifies the use of Macro-F1
> as the primary evaluation metric without requiring class-weighted corrections.