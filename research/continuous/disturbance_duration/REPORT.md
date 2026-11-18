# Synthetic duration prior for a measured electrical-service endpoint

The DOE-417 annual summaries provided seven authentic spreadsheets for 2018–2024. The analysis selected reports that list severe weather or natural disaster and the alert criterion for loss of electric service to more than 50,000 customers. It predicts whether the reported qualifying disturbance lasts longer than 24 hours. The form defines the restoration clock using the reporting criteria. Household restoration, service demand and FEMA Mission Assignment obligations require separate observations. [DOE-417 reporting and annual summaries](https://doe417.energy.gov/), [reporting instructions](https://doe417.energy.gov/files/DOE417_Form_Instructions.pdf).

## Design and support

The locked comparison uses 2021–2024 chronological origins, the same held public reports for every variant and season as the only forecast feature. It omits the preceding calendar year and requires a further 14 days between both training clocks and that year. The most recent annual summaries provide a retrospective archive. Individual publication and revision timestamps are missing, so the delay rule approximates vintage. The analysis ends before October 1, 2024. Bootstrap blocks of 14 and 28 calendar days account for some dependence between reports. The blocks approximate episodes because there is no shared storm ID in the model. [Locked protocol](../BATCH6_PROTOCOL.md), [origin support](results/origin_support.csv).

The 607 criterion-selected rows include 19 after the calendar cutoff. Among the remaining 588, usable durations number 402. Missing or unparseable endpoints account for 175 rows and 11 rows lack an onset date. The 2024 workbook records no end date or time for 80 of 92 selected rows before the cutoff. Complete-case estimates therefore may be biased. Nine usable records had civil-clock durations below one hour, including two in the held comparison. The reports and alert criteria do not resolve which timestamps refer to what portion of the outage. One pair of 2021 reports shares both clocks. The audit preserves each report and groups their errors together. These durations cannot estimate customer-hours lost or verified hours of community restoration. [Duration summary](results/duration_summary.csv), [row-level audit](results/duration_audit.csv).

The empirical comparators use observed training durations pooled or separated by season. The parametric prior draws 1,024 fixed quantiles from an origin-local seasonal normal distribution on log(1 + hours), truncated to viable durations. A fixed weight of 20 shrinks the first two seasonal moments toward the pooled training moments. The hybrid assigns equal mass to the empirical seasonal distribution and the parametric prior. Every scored row is an authentic held report. The simulated grid only represents distribution mass.

## Results and decision

| Model | 24-hour Brier | Log pinball at 50% | Log pinball at 90% | Upper 90% coverage | Median upper bound, hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pooled empirical | 0.25161 | 0.43275 | 0.17388 | 91.28% | 76.27 |
| Seasonal empirical | 0.25131 | 0.44577 | 0.18211 | 90.12% | 74.15 |
| Seasonal synthetic prior | 0.24387 | 0.42587 | 0.17941 | 92.44% | 88.90 |
| Equal mixture | 0.24729 | 0.43330 | 0.18102 | 90.70% | 85.78 |

Across 172 held reports, including 82 longer than 24 hours, the mixture improved Brier by 0.00401 or 1.60% against the matched seasonal baseline. Its paired 95% intervals for the Brier change were [−0.00717, −0.00134] over 64 two-week blocks and [−0.00782, −0.00096] over 40 four-week blocks. The 163-report sensitivity outside 23–25 hours yielded a 1.47% improvement. The point estimate improved in each origin and the latest origin contributes only 12 usable reports. The mixture passed the coverage and upper pinball criteria relative to the seasonal baseline. The required 5% Brier gain was absent. Its median upper bound was 11.63 hours wider. The purely parametric comparator improved Brier by 2.96% relative to the seasonal baseline and the pooled empirical comparator had the lowest upper pinball loss. [Scores and per-origin results](results/scores.csv), [paired Brier intervals](results/paired_brier_intervals.csv).

The hybrid is retained as an exploratory auxiliary result. A stronger follow-up would require authentic end clocks with known collection vintages, adjudicated event grouping and endpoint completion in a second population. The evidence supports no allocation or monetary model change.
