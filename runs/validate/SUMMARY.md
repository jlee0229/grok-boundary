# Step-1 numerics audit

- ce32: cpu arm, torch 2.15.0, 443.1 s, non-finite loss at step None

| variant | wd | eps | grokked (≥0.95 by budget) | median t_first | median t_stable | dips | min test after first | final test | max collapse | final collapse | final margin (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ce32 | 0 | 1e-08 | 0/1 | – | – | 0 | – | 0.000 | 0.725 | 0.720 | 16.1 |
| ce32 | 0.1 | 1e-06 | 0/1 | – | – | 0 | – | 0.001 | 0.000 | 0.000 | 9.3 |
| ce32 | 1 | 1e-12 | 1/1 | 3900 | 3900 | 0 | 0.952 | 0.967 | 0.070 | 0.056 | 15.0 |
| ce32 | 1 | 1e-08 | 1/1 | 4400 | 4400 | 0 | 0.955 | 1.000 | 0.000 | 0.000 | 11.6 |

Figures: curves.png (test accuracy), collapse.png (share of train examples whose float32 softmax_y == 1), margin.png (median train margin; dotted line = float32 cliff 16.64).
