# Step-1 numerics audit

- ce32: cuda Tesla T4, torch 2.11.0+cu130, 3863.5 s, complete True, non-finite loss at logged step None
- ce32_ulp: cuda Tesla T4, torch 2.11.0+cu130, 3790.0 s, complete True, non-finite loss at logged step None
- margin32: cuda Tesla T4, torch 2.11.0+cu130, 4134.0 s, complete True, non-finite loss at logged step None
- ce64: cuda Tesla T4, torch 2.11.0+cu130, 4770.5 s, complete True, non-finite loss at logged step None
- ref64: cuda Tesla T4, torch 2.11.0+cu130, 9393.3 s, complete True, non-finite loss at logged step None

| variant | wd | eps | grokked (≥0.95 by budget) | median t_first | median t_stable | dips | min test after first | final test | max fp32-collapse share* | final fp32-collapse share* | max fp64-collapse share* | final margin (median) | max margin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ce32 | 0 | 1e-12 | 0/4 | >budget | >budget | 0 | – | 0.010 | 0.626 | 0.414 | 0.001 | 15.9 | 48.5 |
| ce32_ulp | 0 | 1e-12 | 0/4 | >budget | >budget | 0 | – | 0.011 | 0.629 | 0.451 | 0.001 | 16.2 | 46.3 |
| margin32 | 0 | 1e-12 | 0/4 | >budget | >budget | 0 | – | 0.000 | 1.000 | 1.000 | 0.000 | 29.5 | 30.9 |
| ce64 | 0 | 1e-12 | 0/4 | >budget | >budget | 0 | – | 0.001 | 1.000 | 1.000 | 0.000 | 29.5 | 31.0 |
| ce32 | 0 | 1e-08 | 0/4 | >budget | >budget | 0 | – | 0.028 | 0.765 | 0.106 | 0.000 | 15.5 | 24.9 |
| ce32_ulp | 0 | 1e-08 | 0/4 | >budget | >budget | 0 | – | 0.028 | 0.764 | 0.107 | 0.000 | 15.5 | 22.6 |
| margin32 | 0 | 1e-08 | 0/4 | >budget | >budget | 0 | – | 0.000 | 1.000 | 1.000 | 0.000 | 19.9 | 21.2 |
| ce64 | 0 | 1e-08 | 0/4 | >budget | >budget | 0 | – | 0.000 | 1.000 | 1.000 | 0.000 | 19.9 | 21.2 |
| ref64 | 0 | 1e-08 | 0/4 | >budget | >budget | 0 | – | 0.000 | 1.000 | 1.000 | 0.000 | 19.9 | 21.1 |
| ce32 | 0 | 1e-06 | 0/4 | >budget | >budget | 0 | – | 0.000 | 0.000 | 0.000 | 0.000 | 14.9 | 16.0 |
| ce32_ulp | 0 | 1e-06 | 0/4 | >budget | >budget | 0 | – | 0.000 | 0.001 | 0.001 | 0.000 | 14.9 | 16.0 |
| margin32 | 0 | 1e-06 | 0/4 | >budget | >budget | 0 | – | 0.000 | 0.001 | 0.001 | 0.000 | 15.0 | 16.1 |
| ce64 | 0 | 1e-06 | 0/4 | >budget | >budget | 0 | – | 0.000 | 0.001 | 0.001 | 0.000 | 15.0 | 16.2 |
| ce32 | 0.1 | 1e-12 | 0/4 | >budget | >budget | 0 | – | 0.341 | 0.577 | 0.147 | 0.000 | 15.4 | 40.6 |
| ce32_ulp | 0.1 | 1e-12 | 0/4 | >budget | >budget | 0 | – | 0.340 | 0.581 | 0.139 | 0.000 | 15.4 | 37.5 |
| margin32 | 0.1 | 1e-12 | 4/4 | 29250 | 29250 | 0 | 0.950 | 1.000 | 1.000 | 1.000 | 0.000 | 23.7 | 25.7 |
| ce64 | 0.1 | 1e-12 | 4/4 | 29400 | 29400 | 0 | 0.950 | 1.000 | 1.000 | 1.000 | 0.000 | 23.7 | 25.6 |
| ce32 | 0.1 | 1e-08 | 4/4 | 33950 | 33950 | 0 | 0.950 | 1.000 | 0.000 | 0.000 | 0.000 | 14.4 | 16.1 |
| ce32_ulp | 0.1 | 1e-08 | 4/4 | 33800 | 33800 | 0 | 0.950 | 1.000 | 0.000 | 0.000 | 0.000 | 14.4 | 16.2 |
| margin32 | 0.1 | 1e-08 | 4/4 | 34250 | 34250 | 0 | 0.950 | 1.000 | 0.001 | 0.000 | 0.000 | 14.5 | 16.5 |
| ce64 | 0.1 | 1e-08 | 4/4 | 34250 | 34250 | 0 | 0.950 | 1.000 | 0.001 | 0.000 | 0.000 | 14.5 | 16.5 |
| ref64 | 0.1 | 1e-08 | 4/4 | 34150 | 34150 | 0 | 0.950 | 1.000 | 0.001 | 0.000 | 0.000 | 14.5 | 16.7 |
| ce32 | 0.1 | 1e-06 | 4/4 | 39150 | 39150 | 0 | 0.950 | 1.000 | 0.000 | 0.000 | 0.000 | 9.9 | 11.3 |
| ce32_ulp | 0.1 | 1e-06 | 4/4 | 38950 | 38950 | 0 | 0.950 | 1.000 | 0.000 | 0.000 | 0.000 | 9.9 | 11.4 |
| margin32 | 0.1 | 1e-06 | 4/4 | 38950 | 38950 | 0 | 0.950 | 1.000 | 0.000 | 0.000 | 0.000 | 9.9 | 11.4 |
| ce64 | 0.1 | 1e-06 | 4/4 | 38950 | 38950 | 0 | 0.950 | 1.000 | 0.000 | 0.000 | 0.000 | 9.9 | 11.4 |
| ce32 | 1 | 1e-12 | 4/4 | 3550 | 3550 | 0 | 0.952 | 0.991 | 0.108 | 0.094 | 0.000 | 15.3 | 21.5 |
| ce32_ulp | 1 | 1e-12 | 4/4 | 3600 | 3600 | 0 | 0.953 | 0.994 | 0.109 | 0.095 | 0.000 | 15.3 | 22.1 |
| margin32 | 1 | 1e-12 | 4/4 | 4100 | 4100 | 0 | 0.953 | 1.000 | 1.000 | 1.000 | 0.000 | 20.9 | 22.8 |
| ce64 | 1 | 1e-12 | 4/4 | 4050 | 4050 | 0 | 0.951 | 1.000 | 1.000 | 1.000 | 0.000 | 20.9 | 22.6 |
| ce32 | 1 | 1e-08 | 4/4 | 4450 | 4450 | 0 | 0.955 | 1.000 | 0.000 | 0.000 | 0.000 | 11.8 | 13.1 |
| ce32_ulp | 1 | 1e-08 | 4/4 | 4450 | 4450 | 0 | 0.954 | 1.000 | 0.000 | 0.000 | 0.000 | 11.8 | 13.0 |
| margin32 | 1 | 1e-08 | 4/4 | 4450 | 4450 | 0 | 0.955 | 1.000 | 0.000 | 0.000 | 0.000 | 11.8 | 13.0 |
| ce64 | 1 | 1e-08 | 4/4 | 4450 | 4450 | 0 | 0.953 | 1.000 | 0.000 | 0.000 | 0.000 | 11.8 | 12.9 |
| ref64 | 1 | 1e-08 | 4/4 | 4450 | 4450 | 0 | 0.950 | 1.000 | 0.000 | 0.000 | 0.000 | 11.8 | 13.0 |
| ce32 | 1 | 1e-06 | 4/4 | 5150 | 5150 | 0 | 0.950 | 1.000 | 0.000 | 0.000 | 0.000 | 7.3 | 8.4 |
| ce32_ulp | 1 | 1e-06 | 4/4 | 5150 | 5150 | 0 | 0.951 | 1.000 | 0.000 | 0.000 | 0.000 | 7.3 | 8.5 |
| margin32 | 1 | 1e-06 | 4/4 | 5150 | 5150 | 0 | 0.951 | 1.000 | 0.000 | 0.000 | 0.000 | 7.3 | 8.3 |
| ce64 | 1 | 1e-06 | 4/4 | 5200 | 5200 | 0 | 0.954 | 1.000 | 0.000 | 0.000 | 0.000 | 7.3 | 8.4 |

Figures: curves.png (test accuracy), collapse.png (share of train examples whose float32 softmax_y == 1), margin.png (median train margin; dotted line = margin 15.4, where float32 collapse was seen to start).

*Collapse shares are computed from each run's logits. Only for ce32/ce32_ulp do they describe the gradient the run actually used; for the other variants they say what standard cross-entropy would have done with those logits. Medians count runs not grokked by the budget as +infinity.
