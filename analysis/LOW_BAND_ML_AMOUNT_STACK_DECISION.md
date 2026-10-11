# Combined low-band ML amount-stack upgrade

## Accepted changes

The frozen 830/912 six-band router is unchanged.

Amount-stage changes:

- `$0-$100K`: replace outer-training band median with strict nested-LFYO two-stage ML: funded-vs-zero classifier + positive-amount regressor.
- `$100K-$1M`: replace outer-training band median with strict nested-LFYO dedicated regression specialist.
- All higher-band amount methods remain unchanged in this comparison.

## Full 912-row deployable comparison

| Metric | Previous accepted stack | Low-band ML upgraded stack |
|---|---:|---:|
| MAE | $5,131,360 | $5,127,644 |
| RMSE | $64,714,795 | $64,714,776 |
| R2 | 0.88656265 | 0.88656272 |
| log-MAE | 1.53748 | 0.69633 |
| log-RMSE | 3.64967 | 1.94284 |
| log-R2 | 0.66203 | 0.90423 |
| within 20% of positive targets | 10.60% | 16.05% |
| within 30% | 17.48% | 25.50% |
| within 50% | 27.79% | 42.12% |
| within factor 2 | 31.23% | 48.71% |
| within factor 3 | 43.84% | 61.89% |

Raw-dollar MAE/RMSE change only slightly because the global raw-dollar metrics are dominated by the five billion-dollar cases. The low-band upgrades materially improve proportional/log-scale fit.

## Correctly routed low-band cases

### `$0-$100K` (666 correctly routed cases)

- MAE: $2,723 -> **$1,766**
- RMSE: $11,203 -> **$7,385**
- R2: -0.063 -> **0.538**
- log-MAE: 1.369 -> **0.493**

### `$100K-$1M` (80 correctly routed cases)

- MAE: $199,383 -> **$175,933**
- RMSE: $263,242 -> **$240,574**
- R2: -0.187 -> **0.009**
- log-MAE: 0.577 -> **0.499**

## Updated amount architecture

```text
$0-$100K
-> nested-LFYO two-stage ML
   zero/funded classifier
   + positive-amount regressor

$100K-$1M
-> nested-LFYO dedicated ML regressor

$1M-$50M
-> preserved ExtraTrees log-dollar regressor

$50M-$200M
-> outer-training median (next band to revisit)

$200M-$500M
-> preserved target-free specialist stack

$500M+
-> nested historical analogue specialist
```
