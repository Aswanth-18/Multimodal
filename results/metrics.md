## Cross-validated performance (5-fold x 3 repeats, mean ± std)

| Target | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| CAD | LogisticRegression | 0.856 ± 0.037 | 0.904 ± 0.042 | 0.895 ± 0.042 | 0.898 ± 0.025 | 0.922 ± 0.025 |
| CAD | RandomForest ★ | 0.859 ± 0.044 | 0.868 ± 0.036 | 0.947 ± 0.033 | 0.906 ± 0.029 | 0.926 ± 0.033 |
| CAD | XGBoost | 0.871 ± 0.037 | 0.892 ± 0.040 | 0.935 ± 0.027 | 0.912 ± 0.024 | 0.917 ± 0.037 |
| LAD | LogisticRegression | 0.736 ± 0.046 | 0.780 ± 0.052 | 0.770 ± 0.081 | 0.772 ± 0.045 | 0.816 ± 0.056 |
| LAD | RandomForest ★ | 0.787 ± 0.053 | 0.782 ± 0.057 | 0.889 ± 0.061 | 0.830 ± 0.039 | 0.857 ± 0.047 |
| LAD | XGBoost | 0.782 ± 0.043 | 0.796 ± 0.045 | 0.847 ± 0.063 | 0.819 ± 0.037 | 0.835 ± 0.054 |
| LCX | LogisticRegression | 0.629 ± 0.054 | 0.544 ± 0.099 | 0.450 ± 0.076 | 0.488 ± 0.065 | 0.644 ± 0.058 |
| LCX | RandomForest | 0.673 ± 0.040 | 0.656 ± 0.109 | 0.373 ± 0.063 | 0.471 ± 0.064 | 0.730 ± 0.064 |
| LCX | XGBoost ★ | 0.688 ± 0.049 | 0.637 ± 0.096 | 0.502 ± 0.076 | 0.557 ± 0.066 | 0.739 ± 0.070 |
| RCA | LogisticRegression | 0.667 ± 0.039 | 0.566 ± 0.051 | 0.510 ± 0.131 | 0.528 ± 0.082 | 0.704 ± 0.048 |
| RCA | RandomForest | 0.669 ± 0.046 | 0.633 ± 0.147 | 0.310 ± 0.109 | 0.406 ± 0.105 | 0.712 ± 0.048 |
| RCA | XGBoost ★ | 0.696 ± 0.053 | 0.630 ± 0.103 | 0.489 ± 0.140 | 0.540 ± 0.098 | 0.719 ± 0.061 |

★ = selected (highest mean ROC-AUC). Selected models are then probability-calibrated (sigmoid).

## Calibrated out-of-fold performance of selected models (5-fold)

| Target | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| CAD | RandomForest | 0.868 | 0.889 | 0.931 | 0.910 | 0.921 |
| LAD | RandomForest | 0.769 | 0.774 | 0.853 | 0.812 | 0.851 |
| LCX | XGBoost | 0.686 | 0.654 | 0.429 | 0.518 | 0.735 |
| RCA | XGBoost | 0.667 | 0.603 | 0.333 | 0.429 | 0.699 |
