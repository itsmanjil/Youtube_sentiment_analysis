| Model | Accuracy | Macro-F1 | Acc CI | F1 CI |
| --- | ---: | ---: | --- | --- |
| deberta_v3 | 0.6556 | 0.6579 | (0.5833, 0.7222) | (0.5850, 0.7231) |
| logreg | 0.7556 | 0.7544 | (0.6889, 0.8222) | (0.6851, 0.8175) |
| svm | 0.7889 | 0.7863 | (0.7278, 0.8444) | (0.7251, 0.8428) |
| static_uniform | 0.7778 | 0.7770 | (0.7167, 0.8389) | (0.7127, 0.8350) |
| nsga_knee | 0.7833 | 0.7826 | (0.7222, 0.8389) | (0.7161, 0.8393) |
| neuro_fuzzy | 0.8056 | 0.8046 | (0.7444, 0.8611) | (0.7432, 0.8588) |

| Model A | Model B | n01 | n10 | p | p_adj | sig |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| deberta_v3 | logreg | 30 | 12 | 0.007916 | 0.08707 | no |
| deberta_v3 | svm | 41 | 17 | 0.002233 | 0.02679 | yes |
| deberta_v3 | static_uniform | 30 | 8 | 0.000472 | 0.006608 | yes |
| deberta_v3 | nsga_knee | 34 | 11 | 0.0008241 | 0.01071 | yes |
| deberta_v3 | neuro_fuzzy | 37 | 10 | 9.849e-05 | 0.001477 | yes |
| logreg | svm | 13 | 7 | 0.2632 | 1 | no |
| logreg | static_uniform | 7 | 3 | 0.3438 | 1 | no |
| logreg | nsga_knee | 6 | 1 | 0.125 | 1 | no |
| logreg | neuro_fuzzy | 10 | 1 | 0.01172 | 0.1172 | no |
| svm | static_uniform | 9 | 11 | 0.8238 | 1 | no |
| svm | nsga_knee | 6 | 7 | 1 | 1 | no |
| svm | neuro_fuzzy | 7 | 4 | 0.5488 | 1 | no |
| static_uniform | nsga_knee | 5 | 4 | 1 | 1 | no |
| static_uniform | neuro_fuzzy | 7 | 2 | 0.1797 | 1 | no |
| nsga_knee | neuro_fuzzy | 4 | 0 | 0.125 | 1 | no |
