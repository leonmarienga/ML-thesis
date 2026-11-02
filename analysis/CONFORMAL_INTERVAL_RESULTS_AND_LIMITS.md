# Split-conformal 80%/90% funding interval research — 2026-11-01

## Sources and immutable baselines

Two distinct interval studies were completed:

1. A **retrospective** historical-residual interval audit in local downloadable results, using the old accepted LFYO Track A saved in GitHub Actions run 34796746603 and the newer Track B earlier-year forecasts from run 41393614710.
2. A **disjoint model/calibration/test split** executed in [Actions run 41395641308](https://github.com/leonmarienga/ML-thesis/actions/runs/41395641308) with source `analysis/conformal_frozen_split_2019.py`.

The *old* accepted point prediction archive has MAE $5.131M, and **is not** the newer $5.015M final accepted stack. No claim of calibrating that newer stack is warranted until its precise per-disaster prediction file is recovered.

## Disjoint split experiment

- Point model trained **once** on 536 non-Biological FY2010–FY2018 cases, frozen.
- FY2019: **68** calibration-only cases.
- FY2020–FY2024: **308** test-only cases.
- No test targets accessed to fit point forecasts or quantiles. The six-band accepted router is not changed.
- Both absolute-dollar and absolute-log1p nonconformity scores evaluated at nominal 80% and 90%. Finite-sample quantile uses order statistic ceil((n_cal + 1) × coverage), while lower bounds are truncated at zero (funding cannot be negative).
- This is a **Track B newly frozen early-response model**, not the user's accepted retrospective full six-band model. Present OpenFEMA feature versions and storm name link are still not verified as-of declaration.

### Overall empirical coverage (2020–2024)

| Predictor | Interval type | Nominal | Coverage | Median interval width |
|---|---|---:|---:|---:|
| Metadata only | Dollar scale | 80% | 87.66% | $1.475M |
| Metadata only | Dollar scale | 90% | 93.18% | $5.198M |
| T+72 missions + recent NHC | Dollar scale | 80% | 88.96% | $1.437M |
| T+72 missions + recent NHC | Dollar scale | 90% | **94.81%** | **$5.095M** |
| T+72 missions + recent NHC | Log scale | 80% | 86.69% | $0.448M |
| T+72 missions + recent NHC | Log scale | 90% | 95.13% | $2.658M |

Log-scale 90% **mean** interval width is $35,234M because several catastrophically oversized predictions render the output useless. Do not recommend the log interval based on median width alone.

### Rare high-value disaster failure, T+72 model 90% dollar intervals

| True funding band | Test cases | Covered |
|---|---:|---:|
| $50M–$200M | 5 | **1** |
| $200M–$500M | 1 | **0** |
| $500M+ | 3 | **0** |

The point model misses the catastrophic tail and the pooled 2019 residual quantile is not large enough to cover those disasters. Broad overall 94.81% coverage therefore **does not imply safety for catastrophic budgets**.

## Retrospective LFYO-prequential diagnostic

- An older Track A archive has 912 LFYO predictions, but each earlier-year prediction may have been trained on subsequent fiscal years. Therefore chronological calibration of its residuals is **not** prospectively valid.
- Its historical 90% dollar intervals achieve only 77.79% empirical coverage for 833 evaluable cases (2011–2024), and zero coverage in all three upper true funding bands. Do not present as rigorous conformal validity.
- Track B expanding-year residual calibration (2019 seeds, 2020–2024 test years) achieved 90.91% empirical 90% dollar coverage for 308 cases with the updated T+72 model. However, models refit yearly; the classical fixed split-conformal guarantee does not directly apply.

## Critical methodological boundaries

- The 2019 calibration errors and 2020–2024 future errors are **not assumed exchangeable**. Strong secular changes in disaster severity and FEMA programs can invalidate formal finite-sample split conformal coverage guarantees.
- Rare bands have fewer than 10 future tests and 2019 calibration lacks enough large disasters. Conditional 90% coverage cannot be claimed in high bands.
- A valid interval for a routed final dollar prediction may cross the predicted funding band; never clip it to the router's range.
- The target is historical obligated dollars, not objectively known humanitarian resource requirement.
- The strongest next research stage is **rare-event-aware Mondrian / hierarchical conformal** with fallback to pooled scores and strict minimum-group sizes; it cannot claim exact high-band conditional guarantees without adequate calibration data.
- **Conformalized quantile regression (CQR) has not yet been implemented or tested**; neither these results nor the repository are evidence of CQR performance.
- Exact final $5.015M point-estimate per-row artifact must be located/created before applying calibration to the accepted deployed point estimates.

## Decision

Retain this as research-only uncertainty-quantification work. Do **not** claim that a 90% interval has 90% catastrophic-disaster coverage. Do not alter the accepted 830/912 router or reported $5.015M MAE, R² 0.8868.
