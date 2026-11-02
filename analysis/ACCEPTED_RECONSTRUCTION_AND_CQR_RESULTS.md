# Accepted Track A reconstruction and CQR research results — 2026-11-01

## Exact accepted Track A reconstruction
GitHub action: https://github.com/leonmarienga/ML-thesis/actions/runs/41397678781
Reconstructed accepted 912 per-disaster prediction CSV by combining four original immutable historical artifacts with the exact original overlay source `analysis/nonbio_final_50_200m_immutable_overlay.py`. Strict assertions reproduced:

- **912** non-Biological records
- Frozen router **830 / 912**
- Final global MAE **$5,014,819.011614494**
- Final R² **0.8867765710842239**
- RMSE **$64,653,747.252143376**

The artifact `exact-accepted-912-predictions` contains all 912 rows, original router predictions, targets, intermediate model versions and accepted final `challenger_stack_prediction`. No accepted model parameters were changed.

**Important:** These 912 historical point predictions originate from strict *leave-fiscal-year-out* experiments. A held-year prediction can train on *later years*. Chronologically calibrating these residuals as if a past model could have issued them at that time is invalid. Proper Track A prospective interval validation requires reconstructing the exact specialist training per forward fiscal year and generating calibration/test predictions from strictly earlier years, not merely using LFYO predictions and a past-date quantile.

## First CQR experiment
GitHub action: https://github.com/leonmarienga/ML-thesis/actions/runs/41397857911
Separately trained **HistGradientBoostingRegressor quantile models** with log1p-dollar 10/90 or 5/95 quantiles and split conformal calibration. Train 536 FY2010–2018 cases, calibrate 68 FY2019 cases, test 308 FY2020–2024 Track B cases.

| Nominal | Observed coverage | Median width | Mean width |
|---|---:|---:|---:|
| 80% | 89.29% | $0.167M | $24.954M |
| 90% | 95.13% | $0.717M | $140.499M |

**Catastrophic true-band 90% coverage**: $50–200M 4/5; $200–500M 1/1; $500M+ 2/3. However median interval widths are approximately **$2.1 billion** for high-value records. This is not an economically useful or validated tail-risk bound. The sample includes only 9 high-value test cases and ordinary chronological exchangeability is unproven.

**Important distinction:** CQR calibrates separately trained quantile models, not the accepted Track A point estimates. It is a research challenger, not an interval wrapper that automatically centers on the $5.015M system's predictions.

## Research decision
- Retain exact Track A 912 prediction file as immutable baseline/reproducibility artifact.
- Retain CQR only as exploratory uncertainty model.
- Do not claim 90% guaranteed conditional catastrophic coverage or deploy intervals for disaster allocation.
- Future experiments should use fresh, held-out catastrophes, strictly earlier-year fixed model training, and tail-aware interval-width/coverage reporting.
