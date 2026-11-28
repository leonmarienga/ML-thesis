# Continuous Batches 5–10: independent replay status

Reference: `b0b6edb229d112dd6d11896a512bf1112414e4f6`. Run: 2026-11-28. Isolated Linux x86-64 CPython 3.12.14 with the 76 dependency versions from the successful reference CI jobs. Used pinned, retained inputs and the [archived workflow](../../../.github/workflows/thesis-continuous-batches.yml); no new acquisition and no FY2025 outcome access.

| Batch | Compared tables | Result |
| --- | ---: | --- |
| 5, complete geography, early fields, winds, ablation | 11 | Nine source/feature tables matched; monetary ablation `metrics.csv` and `paired.csv` differed from the archived run. Repeating the ablation in the same local environment produced byte-identical local outputs. |
| 6, announcement vintage and duration | 14 | Workflow comparison passed. |
| 7, documentary and outage support | 14 | Workflow comparison passed. |
| 8, precoverage evidence and outage transfer | 25 | Workflow comparison passed. |
| 9, component accounting and wind exposure | 8 | Workflow comparison passed. |
| 10, legacy ledger, storm panel, humanitarian sources | 23 | Workflow comparison passed. |

Seven Batch 5 MAE entries differed beyond the recorded relative and absolute comparison tolerances. The largest absolute difference was about $12,321 within a fiscal-year subgroup and about $1,409 for a pooled comparison. All 14 paired uncertainty intervals still contained zero in both outputs. The discrepancy is unresolved; do not mark Batch 5 reproduced. The archived results and frozen Track A are unchanged. The reproduced tables establish computational agreement only for the listed analyses, not new predictive validity.
