# Continuous Batches 11–14: independent replay status

Pinned commit `b0b6edb229d112dd6d11896a512bf1112414e4f6`; isolated CPython 3.12.14 Linux x86-64, 76 pinned dependency versions. Used the retained archives and archived workflow without changing source or relaxing tolerances. No FY2025 outcomes accessed.

| Batch | Results |
| --- | --- |
| 11: Coverage-aware forecasting and essential-service decisions | Expected sets matched. Eleven of twelve output tables matched archived values. The sole discrepancy was `coverage_specialists/inner_selection.csv`: six of 26 inner MAE and six inner tail-MAE entries exceeded the 1e-7 comparison threshold. Maximum absolute differences were $99.36 and $2,072.67 respectively. Selection flags were unchanged in all 26 rows; the other six coverage-specialist tables and all five service-scenario tables matched. Leave Batch 11 open for exact local reproduction. |
| 12: Coastal observation gates and minimum service | 11 tables passed output-set and numerical checks. |
| 13: Signed weighting influence and coastal transfer | 16 tables passed output-set and numerical checks. |
| 14: Transport allocation and frontier | Six CSV tables matched under tolerance and two PNGs passed readability checks. This is not a pixel-equality check. |

The archived CI run and this independent local run are separate observations. Batch 11's discrepancy does not establish an improvement, and nothing was overwritten. Do not rerun passed batches absent a material change. Future diagnosis may compare numerical/library build metadata instead of modifying the frozen comparator or thresholds.
