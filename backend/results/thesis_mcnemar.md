| Model A | Model B | n01 (A wrong, B correct) | n10 (A correct, B wrong) | p | p_adj | sig |
| --- | --- | --- | --- | --- | --- | --- |
| tfidf | logreg | 16235 | 10884 | 4.94e-233 | 3.95e-232 | yes |
| tfidf | svm | 16852 | 13895 | 7.37e-64 | 2.95e-63 | yes |
| tfidf | ensemble | 14367 | 9257 | 3.55e-244 | 3.2e-243 | yes |
| tfidf | meta_learner | 15485 | 9993 | 2.3e-261 | 2.3e-260 | yes |
| logreg | svm | 5668 | 8062 | 3.63e-93 | 1.81e-92 | yes |
| logreg | ensemble | 3439 | 3680 | 0.00445 | 0.00889 | yes |
| logreg | meta_learner | 2510 | 2369 | 0.045 | 0.045 | yes |
| svm | ensemble | 5840 | 3687 | 1.22e-108 | 8.52e-108 | yes |
| svm | meta_learner | 8705 | 6170 | 2.44e-96 | 1.47e-95 | yes |
| ensemble | meta_learner | 4034 | 3652 | 1.38e-05 | 4.15e-05 | yes |
