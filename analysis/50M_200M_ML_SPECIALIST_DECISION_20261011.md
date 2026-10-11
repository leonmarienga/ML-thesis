# $50M-$200M individualized specialist decision

## Decision

**Reject the new individualized specialist for final integration. Keep the strict outer-training band median for this band.**

The frozen six-band router remains unchanged.

## Strict nested-LFYO direct audit

True non-Biological `$50M-$200M` cases: **n=13**.

| Metric | Outer-training median | Nested individualized specialist |
|---|---:|---:|
| MAE | $35.468M | $35.825M |
| RMSE | $48.016M | $48.051M |
| R2 | -0.3573 | -0.3593 |
| log-MAE | 0.3453 | 0.3433 |
| within 20% | 38.46% | 46.15% |
| within 30% | 46.15% | 61.54% |
| within 50% | 76.92% | 76.92% |
| unique predictions | 5 fold medians | 12 individualized |

The individualized specialist improves tolerance accuracy and log-MAE slightly, but it is **worse on the primary direct dollar MAE and RMSE criteria**:
- MAE change vs median: **+1.01%**
- RMSE change vs median: **+0.07%**

Because there are only 13 true cases, this is not enough evidence to replace the simpler median.

## Deployable routed-to-band diagnostic

Rows the frozen router sends to `$50M-$200M`: **n=13**.

| Metric | Median | Individualized specialist |
|---|---:|---:|
| MAE | $40.492M | $34.313M |
| RMSE | $52.051M | $45.351M |
| R2 | -0.1577 | 0.1212 |
| log-MAE | 0.5272 | 0.4726 |

This deployable diagnostic improves because it includes router errors and cross-band cases. However, specialist acceptance is based on the **direct true-band audit**, since the amount specialist should be judged on the regime it is designed to estimate.

## Final amount architecture after current audits

```text
$0-$100K
-> ACCEPTED nested-LFYO two-stage ML
   zero/funded classifier
   + positive amount regressor

$100K-$1M
-> ACCEPTED nested-LFYO dedicated ML regressor

$1M-$50M
-> preserved 900-tree ExtraTrees log-dollar regressor

$50M-$200M
-> KEEP outer-training band median

$200M-$500M
-> preserved target-free specialist stack

$500M+
-> nested historical analogue specialist
```

## Caution

Only 13 true non-Biological `$50M-$200M` cases exist. The individualized specialist produced 12 distinct predictions and better tolerance accuracy, but the direct MAE/RMSE evidence does not justify replacing the median.
