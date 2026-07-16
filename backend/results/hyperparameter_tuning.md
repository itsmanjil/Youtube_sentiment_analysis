# Hyperparameter Tuning Report

## Methodology

All tuning was performed using **5-fold StratifiedKFold cross-validation** on the validation set (n=128,854). The test set was held out entirely during selection and used only for final evaluation of the best configuration.

## Vectorizer Configuration Search

Using LogReg (C=1.0) as the probe model:

| Configuration | max_features | ngram_range | Mean F1 | Std |
|---|---|---|---|---|
| 100k_bigram | 100000 | [1, 2] | 0.6679 | 0.0048 |
| 75k_bigram | 75000 | [1, 2] | 0.6667 | 0.0043 |
| 75k_trigram | 75000 | [1, 3] | 0.6666 | 0.0041 |
| 50k_bigram | 50000 | [1, 2] | 0.6647 | 0.0048 |
| 50k_unigram | 50000 | [1, 1] | 0.6542 | 0.0046 |

**Best vectorizer**: 100k_bigram (F1=0.6679)

## Logistic Regression Grid Search

| C | class_weight | Mean F1 | Std |
|---|---|---|---|
| 1.0 | balanced | 0.6677 | 0.0048 |
| 1.0 | None | 0.6667 | 0.0043 |
| 10.0 | None | 0.6593 | 0.0049 |
| 10.0 | balanced | 0.6590 | 0.0040 |
| 0.1 | balanced | 0.6321 | 0.0054 |
| 0.1 | None | 0.6260 | 0.0044 |
| 0.01 | balanced | 0.5917 | 0.0041 |
| 0.01 | None | 0.4583 | 0.0078 |

**Best LogReg**: C=1.0, class_weight=balanced (F1=0.6677 ± 0.0048)

## Linear SVM Grid Search

| C | class_weight | Mean F1 | Std |
|---|---|---|---|
| 0.1 | balanced | 0.6649 | 0.0045 |
| 0.1 | None | 0.6626 | 0.0043 |
| 1.0 | None | 0.6600 | 0.0046 |
| 1.0 | balanced | 0.6600 | 0.0044 |
| 10.0 | balanced | 0.6295 | 0.0037 |
| 10.0 | None | 0.6295 | 0.0037 |
| 0.01 | balanced | 0.6228 | 0.0039 |
| 0.01 | None | 0.6078 | 0.0034 |

**Best SVM**: C=0.1, class_weight=balanced (F1=0.6649 ± 0.0045)

## Test Set Evaluation (Best Configurations)

| Model | Configuration | Test F1 | Test Accuracy |
|---|---|---|---|
| LogReg | C=1.0, cw=balanced | 0.6574 | 0.6581 |
| SVM | C=0.1, cw=balanced | 0.6581 | 0.6602 |

## Thesis Interpretation

Hyperparameters were selected via cross-validated grid search on the validation set, eliminating the concern of arbitrary defaults. The selected configurations are empirically justified and the test evaluation confirms generalisation.
