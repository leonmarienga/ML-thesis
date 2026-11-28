# Retrospective uncertainty and review-flag audit

**Design.** Audit the fixed `nested_metadata`, `dollar`, nominal 90% intervals in the pinned FY2020–24 forward artifact at `b0b6edb2`. Recalculate interval inclusion without refitting probabilities, cutoffs, forecasts or flags. The $50 million threshold is used only for descriptive grouping by realized amount; it is not known at decision time. The audit is on previously examined historical outcomes, not independent prospective confirmation.

| Stratum | Covered / n | Misses flagged for review | Median interval width |
| --- | ---: | ---: | ---: |
| All | 281 / 308 (91.23%) | 23 / 27 | $4.28M |
| Below $50M | 281 / 299 (93.98%) | 16 / 18 | $4.28M |
| At least $50M | 0 / 9 | 7 / 9 | $4.44M |

Coverage differs across years, from 85.96% in FY2021 to 94.32% in FY2024. The stored review flag marks 159 of 308 cases (51.62%). It would have missed two of nine extreme interval failures, and a flagged interval remains a failed interval, not a successful forecast. The tail cases are few, the years are examined and repeated episode dependence is possible. No claim of conditional coverage, improved uncertainty, or operational threshold validity follows.

**Replay.** Run `python audit.py /path/to/pinned/checkout` in the pinned NumPy/pandas environment. The script verifies FY2025 exclusion, unique observations and strictly earlier calibration-year labels, recomputes inclusions and emits aggregate per-year and tail results with the interval score. It does not certify feature availability dates or independently rebuild the calibration dataset. Next, predefine an abstention/tail calibration protocol based only on training availability and predicted signals, and test once on genuinely independent eligible episodes. Do not tune a review cutoff on these nine tail outcomes.
