| Model | Accuracy | Macro-F1 | Acc CI | F1 CI |
| --- | ---: | ---: | --- | --- |
| deberta_v3 | 0.6667 | 0.6629 | (0.5158, 0.7667) | (0.4940, 0.7645) |
| logreg | 0.7167 | 0.7146 | (0.5904, 0.8333) | (0.5820, 0.8289) |
| svm | 0.7667 | 0.7622 | (0.6658, 0.8754) | (0.6442, 0.8686) |
| static_uniform | 0.7667 | 0.7625 | (0.6500, 0.8675) | (0.6330, 0.8644) |
| nsga_knee | 0.7667 | 0.7646 | (0.6579, 0.8833) | (0.6567, 0.8752) |
| neuro_fuzzy | 0.8000 | 0.7988 | (0.6992, 0.8921) | (0.6955, 0.8876) |

| Model A | Model B | n01 | n10 | p | p_adj | sig |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| deberta_v3 | logreg | 6 | 3 | 0.5078 | 1 | no |
| deberta_v3 | svm | 10 | 4 | 0.1796 | 1 | no |
| deberta_v3 | static_uniform | 6 | 0 | 0.03125 | 0.4688 | no |
| deberta_v3 | nsga_knee | 8 | 2 | 0.1094 | 1 | no |
| deberta_v3 | neuro_fuzzy | 10 | 2 | 0.03857 | 0.54 | no |
| logreg | svm | 6 | 3 | 0.5078 | 1 | no |
| logreg | static_uniform | 5 | 2 | 0.4531 | 1 | no |
| logreg | nsga_knee | 4 | 1 | 0.375 | 1 | no |
| logreg | neuro_fuzzy | 6 | 1 | 0.125 | 1 | no |
| svm | static_uniform | 4 | 4 | 1 | 1 | no |
| svm | nsga_knee | 2 | 2 | 1 | 1 | no |
| svm | neuro_fuzzy | 2 | 0 | 0.5 | 1 | no |
| static_uniform | nsga_knee | 3 | 3 | 1 | 1 | no |
| static_uniform | neuro_fuzzy | 4 | 2 | 0.6875 | 1 | no |
| nsga_knee | neuro_fuzzy | 2 | 0 | 0.5 | 1 | no |
