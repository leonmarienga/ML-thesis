# Coverage-aware funding forecasts

The observability audit preserved all 912 historical declarations. The 223 precoverage workbook zeros are unavailable as documented obligations. Within public coverage, 313 declarations returned no actions and one had unusable action dates. Another 51 have not passed the 36-month and 90-day reporting screen. Among 324 elapsed and key-complete declaration histories, 27 net to zero with at least one action by the horizon and 297 have positive obligations. A recorded net zero describes the retrieved signed ledger scope. The audit cannot establish a declaration-wide administrative zero from a missing API result. [Observability table](results/label_observability.csv).

The eventual workbook proxy uses 376 FY2019–24 cases with 13 catastrophic rows from 11 episode groups. It reproduces the retained case-level real-only and 20× forecasts. The signed 36-month proxy has 135 FY2019–23 cases with 11 catastrophic rows from 10 groups. At each origin, only earlier groups with a release-review date before the earliest held event enter training. No FY2024 row cleared the held signed-outcome screen. Both analyses use already examined historical populations. The elapsed rule screens likely outcome availability. The current archive cannot establish that those signed values and amendments were historically published by each origin. [Protocol](../BATCH11_PROTOCOL.md), [fold support](results/fold_support.csv).

| Endpoint and arm | Overall MAE ($M) | $50M+ MAE ($M) |
| --- | ---: | ---: |
| Eventual proxy, real-only | 19.92 | 547.57 |
| Eventual proxy, fixed 20× | 22.29 | 477.77 |
| Eventual proxy, fixed 0.10 hazard gate | 22.85 | 483.55 |
| Eventual proxy, nested selected | 21.24 | 513.51 |
| Signed elapsed, real-only | 29.84 | 347.66 |
| Signed elapsed, fixed 20× | 27.95 | 291.80 |
| Signed elapsed, fixed 0.10 hazard gate with real-only fallback | 29.35 | 310.80 |
| Signed elapsed, nested selected | 29.76 | 345.84 |

The positive-only hurdle changed overall MAE by less than $0.18M on either endpoint and did not pass nested selection. The blended hazard specialist ranks groups more effectively on the eventual proxy in the fixed exploratory arm, with average precision 0.384 versus 0.282 and 24 versus 37 false alerts at the 0.10 threshold. Its overall MAE increased by $2.92M, with a paired episode bootstrap interval of $0.34M to $5.72M. The nested selector never chose either hazard gate. It selected 20× for two of six eventual-proxy origins and one of five signed-proxy origins. On signed outcomes, the nested tail-MAE change was −$1.81M, with a 95% interval of −$22.60M to +$15.78M. The larger fixed signed 20× tail-MAE change was −$55.86M, with an interval of −$143.28M to +$2.50M. Nested selection missed the fixed arm's best exploratory gains and gives no basis for advancing a specialist. These amount models do not establish calibrated catastrophic dollar intervals. [Matched scores](results/scores.csv), [training-only choices](results/inner_selection.csv), [paired episodes](results/paired_changes.csv).

The largest eligible hazard-state-funding-band cell had only two earlier tail groups across both endpoints. The broadest late-origin eventual fitting population had only three $500M+ groups and the signed endpoint had at most two. The real-data hazard-to-global tail median uses a fixed eight-episode shrinkage weight and generates no labels. Its mixed results leave conditional funding simulation and complex generators unsupported. Physics and service simulations proceed under independently defined endpoints. [Independent training support](results/fold_support.csv).
