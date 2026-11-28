# Michigan Bar: a worked example of forecast revision

The two public bulletins share 43 valid hours, 2022-12-31 18:00 UTC to 2023-01-02 12:00 UTC, on the same earlier-verified gauge series. These hours are a subsequence of the original 230 matched pairs, not additional evidence or independent events. The matched observation at every valid hour agrees across the versions.

| Bulletin issued (UTC) | MAE (ft) | Bias (ft; forecast minus observed) | Historical 12-ft hits | False exceedances |
| --- | ---: | ---: | ---: | ---: |
| 2022-12-28 17:13 | 1.637 | −1.637 | 11/21 | 0/22 |
| 2022-12-31 17:00 | 0.646 | +0.261 | 21/21 | 0/22 |

For the first common valid time, 2022-12-31 18:00 UTC, USGS observed 16.47 ft. The early bulletin predicted 10.10 ft 72 hours 47 minutes beforehand. The later bulletin predicted 16.20 ft one hour beforehand, but that bulletin was issued eight hours after the first measured flood-stage crossing. The common hours control which valid timestamps are scored; they do not control for warning lead, assimilated observations or the timing of initial flooding. These counts do not support operational alert reliability, lead-adjusted superiority, or generalization to other storms.

Replay from this folder with `python -I -S common_hours.py stage_pairs.csv`. The comparison checks the pinned pair digest, exact report-time and lead identities, identical observations at the common times, and the historical 12.0 ft boundary. No model rerun, source changes, new fiscal outcomes or external action were needed. See the [original stage verification](REPORT.md) for observation provenance and the distinct unresolved QINE discharge clock.
