| Model | Accuracy | Macro-F1 | Acc CI | F1 CI |
| --- | ---: | ---: | --- | --- |
| deberta_v3 | 0.6667 | 0.6629 | (0.5413, 0.7921) | (0.5287, 0.7791) |
| logreg | 0.7167 | 0.7146 | (0.6079, 0.8167) | (0.6058, 0.8175) |
| svm | 0.7667 | 0.7622 | (0.6833, 0.8667) | (0.6727, 0.8621) |
| static_uniform | 0.7667 | 0.7625 | (0.6667, 0.8833) | (0.6633, 0.8809) |
| nsga_knee | 0.7000 | 0.6971 | (0.5746, 0.8254) | (0.5647, 0.8185) |
| neuro_fuzzy | 0.7833 | 0.7811 | (0.7000, 0.8754) | (0.6981, 0.8740) |

| Model A | Model B | n01 | n10 | p | p_adj | sig |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| deberta_v3 | logreg | 6 | 3 | 0.5078 | 1 | no |
| deberta_v3 | svm | 10 | 4 | 0.1796 | 1 | no |
| deberta_v3 | static_uniform | 6 | 0 | 0.03125 | 0.4688 | no |
| deberta_v3 | nsga_knee | 2 | 0 | 0.5 | 1 | no |
| deberta_v3 | neuro_fuzzy | 10 | 3 | 0.09229 | 1 | no |
| logreg | svm | 6 | 3 | 0.5078 | 1 | no |
| logreg | static_uniform | 5 | 2 | 0.4531 | 1 | no |
| logreg | nsga_knee | 3 | 4 | 1 | 1 | no |
| logreg | neuro_fuzzy | 6 | 2 | 0.2891 | 1 | no |
| svm | static_uniform | 4 | 4 | 1 | 1 | no |
| svm | nsga_knee | 4 | 8 | 0.3877 | 1 | no |
| svm | neuro_fuzzy | 1 | 0 | 1 | 1 | no |
| static_uniform | nsga_knee | 0 | 4 | 0.125 | 1 | no |
| static_uniform | neuro_fuzzy | 4 | 3 | 1 | 1 | no |
| nsga_knee | neuro_fuzzy | 8 | 3 | 0.2266 | 1 | no |
