# $0-$100K dedicated two-stage ML specialist decision

## Decision

**Accept the dedicated two-stage ML amount specialist for the $0-$100K band.**

The frozen 830/912 router remains unchanged. This changes only the amount estimator used after the router outputs `0-100K`.

## Why this band needs two stages

This band contains many exact zero-funding cases. A single regressor would tend to blur zeros into arbitrary positive amounts.

The accepted architecture is therefore:

```
router -> 0-100K
        |
        v
funded-vs-zero classifier
        |
        +-- zero -> predict $0
        |
        +-- funded -> positive-amount regressor
                         |
                         v
                  individualized amount
```

Classifier family, classifier threshold, positive-amount regressor, and shrinkage weight are selected only from inner LFYO inside each outer-training fold.

## Direct strict-LFYO audit

True $0-$100K cases: **n=699**.

| Metric | Outer-training band median | Two-stage ML | Change |
|---|---:|---:|---:|
| MAE | $4,483 | $2,936 | -34.52% |
| RMSE | $15,444 | $10,585 | -31.46% |
| R2 | -0.0920 | 0.4870 | large improvement |
| log-MAE | 1.7513 | 0.5605 | improved |
| zero accuracy | 80.54% | 95.28% | improved |
| predicted zero share | 100.00% | 79.26% | individualized |
| actual zero share | 80.54% | 80.54% | reference |
| unique predictions | 1 | 145 | individualized |

The old band median is effectively zero in every outer fold, so it predicts $0 for every case in this band.

All 15 outer folds selected an ML configuration that beat the corresponding inner-LFYO median on inner MAE.

Selected classifier families:
- Random Forest with semantic features: 8 folds
- ExtraTrees with current19: 7 folds

Selected positive-amount regressors:
- Random Forest log-target + semantic features: 12 folds
- ExtraTrees log-target + semantic features: 2 folds
- LightGBM log-target + semantic features: 1 fold

Selected funded-probability thresholds:
- 0.50: 4 folds
- 0.60: 8 folds
- 0.70: 3 folds

Positive-amount shrinkage:
- weight 1.00: 14 folds
- weight 0.75: 1 fold

## Frozen-router deployable audit

Rows routed to $0-$100K by the frozen router: **n=686**.

| Metric | Median amount | Two-stage ML amount |
|---|---:|---:|
| MAE | $11,821 | $10,336 |
| RMSE | $68,635 | $66,130 |
| R2 | -0.0306 | 0.0433 |
| log-MAE | 1.6912 | 0.5852 |
| zero accuracy | 81.63% | 95.48% |

This includes router errors. The specialist still improves deployable performance.

## Correctly routed true-band cases

Among the **666** true $0-$100K cases correctly routed to this band:

- MAE: $2,723 -> **$1,766**
- RMSE: $11,203 -> **$7,385**
- R2: -0.0628 -> **0.5381**

This is the cleanest measure of the amount specialist once routing is correct.

## Cumulative low-band replacement

Starting from the accepted 912-row amount-stack predictions, only these routed bands were replaced:
- $0-$100K -> new two-stage ML specialist
- $100K-$1M -> previously accepted dedicated ML specialist

Every other routed-band prediction was left unchanged.

Overall 912-row metrics:

| Metric | Previous accepted stack | Both low-band ML specialists |
|---|---:|---:|
| MAE | $5,131,360 | $5,127,644 |
| RMSE | $64,714,795 | $64,714,776 |
| R2 | 0.88656265 | 0.88656272 |
| log-MAE | 1.53748 | 0.69633 |
| log-R2 | 0.66203 | 0.90423 |

The raw-dollar full-system metrics barely move because billion-dollar cases dominate raw variance, while log metrics improve dramatically because the low-dollar cases are no longer collapsed to a constant zero/median prediction.

## Final architecture change

Old:

```
$0-$100K
-> outer-training median (usually $0)
```

New:

```
$0-$100K
-> nested-LFYO funded-vs-zero classifier
-> nested-selected probability threshold
-> if funded: nested-selected positive-amount regressor
-> individualized amount below $100K
```
