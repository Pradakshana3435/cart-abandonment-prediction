# Model Comparison Report

## Setup
5-fold stratified cross-validation on training data (9,764 sessions,
~15.6% purchase rate). Scored on ROC-AUC and PR-AUC (PR-AUC prioritized
given class imbalance).

## Results

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression | 0.900 ± 0.012 | 0.647 ± 0.024 |
| Random Forest | 0.930 ± 0.004 | 0.740 ± 0.019 |
| XGBoost | 0.929 ± 0.007 | 0.748 ± 0.025 |

## Decision
Random Forest and XGBoost are statistically tied (gap within CV
variance). **XGBoost selected** for its marginal PR-AUC edge and
native SHAP TreeExplainer support for explainability.

## Calibration check
Initial models used class-weighting (`class_weight='balanced'` /
`scale_pos_weight`) to improve minority-class detection. This was
tested and **rejected**: weighting distorted predicted probabilities
(Brier score 0.099 vs 0.069 unweighted) without improving ranking
(PR-AUC 0.746 vs 0.749). Final model uses no class weighting.

## Final test-set performance (unweighted XGBoost)
- ROC-AUC: 0.934
- PR-AUC: 0.746
- See `notebooks/01_eda.ipynb` for full classification report and
  confusion matrix.

## PageValues sensitivity
Removing `PageValues` drops PR-AUC from 0.749 to 0.377 — it is the
dominant feature. Documented as a limitation: a production system
must confirm this analytics-derived field is available at prediction
time.