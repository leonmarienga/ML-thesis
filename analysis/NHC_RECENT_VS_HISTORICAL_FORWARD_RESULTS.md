# NHC recent vs historical feature: earlier-fiscal-year-only comparison

Run: https://github.com/leonmarienga/ML-thesis/actions/runs/41338100996

## Experimental design
- Frozen router's 830/912 accepted output is read but not retrained.
- The test specialist is an experimental **fixed** five-neighbour $50M–$200M analogue with support $10M–$500M, 75% neighbour prediction and 25% training-band median shrinkage.
- Outer evaluation trains only on fiscal years **strictly earlier** than the test year; enough training support exists for held-out tests in fiscal years 2019, 2020, 2022, 2023, and 2024.
- Compared three candidate distances: (A) operational-only, (B) recent NHC wind/pressure issued ≤72 hours before the date-based T0, (C) latest archived NHC wind/pressure plus log advisory age, any age. Hazard-distance weight fixed at **0.25**.
- Original all-tropical NHC records (123) matched **without using FEMA funding amounts**. 113 have archived observations and 41 have recent (≤72h) T0 advisories. Preprocessing is fit on the training pool only.
- Models are **not** the accepted specialist's exact pipeline and are not an acceptance-test replacement for it.

## Results: 13-case rare-band, earlier-year-supported subsets

| Scope | n | Variant | MAE $M | RMSE $M | R² | within 30% |
|---|---:|---|---:|---:|---:|---:|
| True $50–200M | 9 | Operational | **21.736** | 26.669 | 0.6093 | 66.67% |
| True $50–200M | 9 | Recent NHC ≤72h | **20.521** | **22.772** | **0.7151** | **77.78%** |
| True $50–200M | 9 | Any-age NHC | 23.915 | 26.497 | 0.6143 | 66.67% |
| Frozen-router assigned $50–200M | 10 | Operational | 31.999 | 46.763 | 0.1989 | 60.00% |
| Frozen-router assigned $50–200M | 10 | Recent NHC ≤72h | **29.500** | **41.030** | **0.3833** | **70.00%** |
| Frozen-router assigned $50–200M | 10 | Any-age NHC | 31.315 | 39.881 | 0.4173 | 60.00% |

Direct MAE improvement for recent variant: **$1.215M (5.59%)**.
Routed MAE improvement for recent variant: **$2.499M (7.81%)**.

## Uncertainty
Storm-grouped paired bootstrap is strictly exploratory, not an independent CI because the research architecture has been developed over similar cases.
- Direct n=9, 8 storm clusters, *recent minus operational* MAE delta: **−1.215M**, interval **[−9.512M, +3.453M]**.
- Routed n=10, 8 storm clusters, *recent minus operational* MAE delta: **−2.499M**, interval **[−10.752M, +3.453M]**.
- Both intervals span zero. **No statistically secure reduction in MAE is demonstrated.**

## Critical interpretation limits
1. Existing operational features include potentially **future-leaky mission totals, disaster duration, and response-complexity measurements**. Even though the NHC additional signals use historical archive timestamps, the **full model remains retrospective, not declaration-time-safe**.
2. Current FEMA declaration titles were used to link storm identity and may have been modified after the original declaration. Their availability at T0 is unverified.
3. `declarationDate` is a date-only field treated as 00:00 UTC; the exact real-time decision point is not verified.
4. Some storm names recur and disaster rows share events. Cluster diagnostics account for some repetition, but N=8 distinct storm clusters in tested rare-band cases is tiny.
5. Only six of the nine true-band cases and seven of ten routed cases are tropical; other cases remain unmodified in the NHC distance comparison.
6. Weight 0.25 was used in an earlier related pilot and is not an untouched pre-registered hyperparameter. There is **no independent untouched future-year holdout**.
7. The frozen 91.01% router and accepted ~$5.015M full-system MAE are **not improved by this research run**; no accepted specialist was replaced.

## Decision
Retain **recent ≤72-hour NHC** as the preferred research *candidate*, while rejecting any claim that the end-to-end six-band system has improved. Next work: timestamped FEMA and mission feature snapshots, independent tests with robust storm clustering, and source issue-time validation across all hazard types.
