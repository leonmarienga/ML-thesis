# $100K-$1M dedicated ML specialist decision

## Decision

**Accept the dedicated nested-LFYO ML amount specialist for the $100K-$1M band.**

The frozen 830/912 six-band router is unchanged. This decision changes only the amount estimator invoked after the router outputs `100K-1M`.

## Dedicated strict-LFYO audit

True $100K-$1M cases: **n=108**.

| Metric | Outer-training band median | Dedicated ML | Change |
|---|---:|---:|---:|
| MAE | $203,087 | $187,157 | -7.84% |
| RMSE | $265,957 | $247,802 | -6.83% |
| Median absolute error | $146,048 | $122,700 | improved |
| log-MAE | 0.5986 | 0.5448 | improved |
| within 20% | 18.52% | 20.37% | improved |
| within 30% | 27.78% | 29.63% | improved |
| within factor 2 | 53.70% | 67.59% | improved |
| unique predictions | 8 fold medians | 108 | individualized |

All 15 outer folds selected a nonzero ML solution that beat the corresponding inner-LFYO median on inner MAE. The selected shrinkage weight was **1.0 in all 15 folds**, so the validation preferred the individualized ML prediction without pullback toward the median.

Selected model families by outer fold:
- ExtraTrees log target, current19: 6 folds
- LightGBM log target, semantic features: 4 folds
- ExtraTrees log target, semantic features: 4 folds
- Random Forest log target, semantic features: 1 fold

Candidate selection and feature/model fitting use only outer-training data with inner LFYO.

## Frozen-router deployable audit

Rows routed by the frozen router to $100K-$1M: **n=130**.

| Metric | Median amount | Dedicated ML amount |
|---|---:|---:|
| MAE | $433,993 | $415,768 |
| RMSE | $878,411 | $869,533 |
| log-MAE | 1.3617 | 1.2969 |
| within factor 2 | 33.85% | 44.62% |

This includes router errors, so the absolute metrics are worse than the direct true-band audit. Even so, ML improves the deployable amount estimate.

## Isolated full-stack replacement

The accepted 912-row amount-stack predictions from run 34796746603 were reused unchanged for every other routed band. Only the 130 rows routed to $100K-$1M were replaced by the new strict-LFYO ML predictions.

Overall 912-row metrics:

| Metric | Previous accepted stack | ML-upgraded stack |
|---|---:|---:|
| MAE | $5,131,360 | $5,128,762 |
| RMSE | $64,714,795 | $64,714,778 |
| R2 | 0.88656265 | 0.88656271 |
| log-MAE | 1.53748 | 1.52825 |
| log-R2 | 0.66203 | 0.66224 |

The overall raw-dollar change is necessarily small because this band is tiny relative to billion-dollar cases, but proportional/log performance improves.

For the 108 actual $100K-$1M cases inside the complete routed system:
- MAE: $283,160 -> $265,790
- RMSE: $437,449 -> $427,670
- R2: -2.1244 -> -1.9863
- log-MAE: 2.8254 -> 2.7675

For the **80 correctly routed** actual $100K-$1M cases:
- MAE: $199,383 -> **$175,933**
- RMSE: $263,242 -> **$240,574**
- R2: -0.1870 -> **0.0086**

## Final architecture change

Old:

```
$100K-$1M
-> outer-training band median
```

New:

```
$100K-$1M
-> nested-LFYO model-family selection
-> dedicated target-blind ML regressor
-> individualized dollar prediction
-> clip to $100K-$1M
```

The router remains frozen. No held-out fiscal-year funding values are used in model selection or fitting.
