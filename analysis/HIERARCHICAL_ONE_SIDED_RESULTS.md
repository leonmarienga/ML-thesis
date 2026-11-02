# Hierarchical and one-sided prediction intervals — completed research audit

Run: https://github.com/leonmarienga/ML-thesis/actions/runs/41396350227

## Test design
Same disjoint experiment as prior split conformal: fit 536 FY2010–2018 cases, calibrate on 68 FY2019 cases, evaluate 308 FY2020–2024 cases. Frozen early-response Track B point model, with archived NHC wind/pressure filtered by <=72-hour issue time and reconstructed mission action-date proxies.

**Four interval designs** at 80% and 90%:
1. Pooled symmetric absolute residuals.
2. Hierarchical symmetric absolute residuals grouped by **predicted** funding amount, with minimum 15 calibration records per group and pooled fallback.
3. Pooled one-sided upper bounds using signed errors (actual minus prediction).
4. Hierarchical one-sided upper bounds, same group policy.

Conditional groups cannot use actual funding bands at inference; the grouping is based on predicted funding amounts.

## 90% results: T+72 + recent NHC, n=308
| Method | Observed coverage | Median interval width / upper bound | Rare true-band coverage |
|---|---:|---:|---|
| Pooled symmetric | 94.48% | $5.095M median two-sided width | 0/9 for >=$50M |
| Hierarchical symmetric | 92.86% | $1.436M median two-sided width | 0/9 for >=$50M |
| Pooled one-sided upper | 95.45% (Y <= upper) | $4.020M median upper bound | 1/9 for >=$50M |
| Hierarchical one-sided upper | 94.16% (Y <= upper) | $1.436M median upper bound | 1/9 for >=$50M |

A much smaller median interval is **not** evidence of better disaster-tail protection.

## Why hierarchical fallback fails on high cases

Of 68 FY2019 calibration records, the T+72 point model predicts:
- 57 at $0–$100K,
- 6 at $100K–$1M,
- 3 at $1M–$50M,
- 2 above $50M.

Only the predicted $0–$100K group has >=15 calibration points. All other groups fall back to the pooled 68-case quantile, leaving catastrophic cases unprotected. This procedure does *not* provide guaranteed conditional 90% coverage by predicted group or by true funding band.

**Mathematical constraint:** Under ordinary split conformal with five calibration observations, the 90% finite-sample rank would be ceil(6×.9)=6, larger than the available five observations. Nontrivial finite 90% intervals cannot be obtained from that isolated sample without further assumptions or pooling.

## One-sided upper limit vs two-sided interval

An upper prediction limit answers a risk-management question: how large could historical eventual FEMA obligations be at the chosen nominal level? The 90% pooled upper bound covers 294/308 outcomes in this experiment, but it misses eight of nine catastrophically funded disasters, even though global coverage is high.

This is *funding* risk, not necessarily *humanitarian need*. These upper bounds must not automatically determine aid allocation.

## Research interpretation and safeguards
- The accepted 830/912 six-band router and ~$5.015M global point MAE remain unchanged.
- Frozen-model train/cal/test separation is valid, but future-year exchangeability remains unverified and current OpenFEMA event versions may be post-event.
- This is an exploratory historical sample; repeatedly inspected outcomes are not an untouched external validation set.
- Disaster-event clusters and thin catastrophic tails require broader independent testing.
- No conformalized quantile regression has yet been implemented.
- The exact newest accepted 912 per-case point predictions can be **reconstructed** from the four immutable input artifacts identified by the immutable overlay workflow (runs 34796746603, 38153347912, 38152554081, 38162966192), but their final 38166372260 workflow artifact is no longer available directly. The run log confirms global MAE $5,014,819.01 and R² .88677657. Use the exact per-case overlay before an acceptance Track A calibration.

## Recommendation
Preserve pooled one-sided bounds as a transparent baseline and label catastrophic intervals **INSUFFICIENT TAIL EVIDENCE**. Do not present hierarchical intervals as reliable protection for extremely high-value emergencies until fresh external calibration data include enough large cases.
