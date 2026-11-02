# Decision usefulness of calibrated funding intervals — completed fixed audit

Completed run: https://github.com/leonmarienga/ML-thesis/actions/runs/41399416843

## Controlled comparison
The audit reuses **exactly the existing, immutable** split-conformal and CQR interval predictions for 308 FY2020–2024 disasters. No model fit, recalibration, threshold optimization, or selection from the test outcomes is performed. It verifies that actual funding values match the reconstructed accepted Track A 912-row artifact, but **does not use Track A predictions to calibrate or produce these Track B intervals**.

All methods use 2019 calibration and pre-2019 model training, although CQR uses different models from the residual-conformal comparisons. The nine high-value test disasters above $50M are the same cases for all methods.

## 90% interval results
| Method | Overall interval / upper coverage | Coverage of 9 above $50M | Total upper-limit shortfall on nine high-value disasters | Median two-sided width |
|---|---:|---:|---:|---:|
| Pooled two-sided | 94.48% | 0/9 | $6.378B | $5.095M |
| Hierarchical two-sided | 92.86% | 0/9 | $6.378B | $1.436M |
| Pooled upper | 95.45% | 1/9 | $6.387B | not applicable |
| Hierarchical upper | 94.16% | 1/9 | $6.387B | not applicable |
| CQR | 95.13% | 7/9 | $0.407B | $0.717M |

The upper-limit shortfall equals sum(max(actual obligation - interval upper,0)); **it is not unmet humanitarian need or funding required to allocate**.

For CQR the **mean** two-sided width is $140.499M despite a median width of $0.717M, showing extreme, economically problematic interval variability.

## Threshold-alert stress tests
A descriptive rule flags a possible catastrophic event when its interval **upper >= $500M**, using a threshold specified by the existing six-band model design.

- CQR: flags **19 / 308** disasters, catches **3 / 3** above $500M; **16 false-positive alerts** (precision 3/19=15.8%).
- Pooled and hierarchical conformal variants: flag **0 / 308**, missing **3 / 3** $500M+ disasters.

This comparison illustrates a severe false-alarm versus missed-catastrophe tradeoff. It does not validate an operational alert policy or forecasted value of intervention. Different models also use different point/quantile estimation architectures.

## Methodological warnings
- These are historical, repeatedly studied samples; do not tune alert thresholds using them.
- The test-set high-dollar n=9, with only three >$500M. No reliable conditional catastrophic coverage or warning accuracy can be concluded.
- Conformal exchangeability is uncertain under fiscal-year drift.
- The all-current OpenFEMA metadata lacks authenticated contemporaneous version snapshots.
- Funding obligation is an accounting target, not humanitarian demand; reserve planning requires independent allocation utility and loss modelling.
- Any downstream dashboard must prominently label large intervals as *insufficient catastrophic calibration evidence* and distinguish measured retrospective coverage from guaranteed prospective coverage.
- Accepted Track A unchanged: 912 rows, frozen router 830 correct, final MAE $5,014,819.0116, R² 0.8867765711.
