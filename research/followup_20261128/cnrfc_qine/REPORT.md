# CNRFC QINE / USGS discharge compatibility probe

## Decision

**Michigan Bar (`MHBC1`) passes the endpoint gate for a retrospective same-site discharge comparison; the timestamp gate is still open for an exact operational lead-time comparison.** Do not generalize either conclusion to other locations simply because they use the same archive column code. No fitted model, funding data, FY2025 outcomes or policy performance claim.

[NOAA's hydrologic time-series codebook](https://www.weather.gov/media/owp/oh/hrl/docs/14datatype_ts.pdf) differentiates adjusted instantaneous river discharge (`QINE`), observed instantaneous discharge (`QIN`) and simulated instantaneous discharge (`SQIN`). The [CNRFC deterministic export catalog](https://www.cnrfc.noaa.gov/deterministicHourlyProductCSV.php) gives QINE and SQIN in kcfs. Its group-specific [North San Joaquin definitions](https://www.cnrfc.noaa.gov/documentation/North_SanJoaquin_definitions.pdf) determine the physical target:

| CNRFC field | Meaning at this site | Official gauge use |
|---|---|---|
| MHBC1 QINE | Michigan Bar streamflow estimate adjusted to river observations | USGS `11335000`, parameter `00060` (instantaneous discharge, cfs), is a comparable future target after an explicit release-time gate. Pre-cycle values are likely influenced by that gauge and are not independent observations. |
| CMPC1 QINE | Pardee Dam natural inflow adjusted to daily full-natural flow observations | Do not use raw regulated river discharge; require a definition-, clock- and location-compatible reconstructed natural-flow series. |
| MCNC1 QINE | McConnell total-flow adjusted estimate | CNRFC documents no empirical stage–discharge rating at this site during the event. Stage measurements cannot serve as discharge targets. |
| MCNC1L | McConnell local component, not in either ZIP | Excludes Michigan Bar and Scott Road inputs; a total-flow gauge would be the wrong target. |

The [CNRFC storm analysis](https://www.cnrfc.noaa.gov/storm_summaries/dec2022Jan2023storms.php) gives the `MHBC1/MHB/11335000` identifier crosswalk and documents major flooding at Michigan Bar amid the New Year's 2022–23 event. CNRFC reports no major dams upstream there; an [older USGS flood record](https://pubs.usgs.gov/wsp/1686/report.pdf) notes potential diversion effects. This is a same-site stream-discharge comparison, not proof that the readings are full-natural flow.

## Acquired and inspected

The fixed observation window is `[2022-12-27 00:00Z, 2023-01-03 00:00Z)`. Both December 31 nominal 12Z and 18Z CNRFC deterministic cycle ZIPs, the USGS continuous response, and CNRFC group definitions returned HTTP 200. The [USGS observation endpoint documentation](https://api.waterdata.usgs.gov/docs/ogcapi/) and the capture manifest record the versioned requests. Both ZIPs hold one CSV with a GMT clock and explicit `QINE` subheaders including MHBC1 and CMPC1. Convert kcfs to cfs exactly by multiplying by 1,000. No time shifts, interpolation, gauge substitution or discharge-from-stage reconstruction were performed. USGS returned 673 approved, unqualified observations from one discharge series, with 672 in the half-open window and one right-endpoint observation excluded.

| Nominal cycle, UTC | ZIP HTTP Last-Modified, UTC | Pre-cycle exact pairs / MAE (cfs) | Cycle to archive-clock pairs withheld | After archive-clock exact pairs |
|---|---|---:|---:|---:|
| Dec 31 12:00 | Dec 31 16:56:47 | 108 / 3.241 | 5 | 55 |
| Dec 31 18:00 | Dec 31 20:58:13 | 114 / 4.119 | 3 | 51 |

The 222 pre-cycle pair instances corroborate site, variables, units and clock, not forecast skill or independent gauge sources. The 106 after-archive-clock pair instances contain only 55 distinct valid hours and come from one event. HTTP Last-Modified is a retrospective file metadata check, **not** a verified first public release time. Capturing it helps expose why nominal hours are insufficient. The Dec 31 18Z file's 18:00 row is near the gauge value (63,627.48 versus 63,600 cfs), but it predates the archive timestamp and is withheld from follow-on skill analysis. The 12Z file's row at that same hour is 60,579.75 cfs; it is an exact-hour descriptive comparison only. The approved trace's highest 15-minute sample is 67,900 cfs at 17:45 UTC; the report's rounded 18:00 peak time cannot replace the exact-hour gauge value. The 18Z file was modified after that measured crest; do not use its close apparent match as an advance prediction.

## Replay and next gate

- From this directory `python -B replay.py` verifies response hashes in `raw/MANIFEST.json` and reproduces `results/summary.json` and `results/exact_hour_matches.csv`. Then run `sha256sum -c results/REPLAY_SHA256SUMS`. To re-query, copy `acquire.py` to a clean directory and run it; HTTP metadata can change and overwritten captures are forbidden.
- The replay deliberately exposes rows in three timing bands so excluded values are not silently lost. Within-band residual summaries are descriptive diagnostics, not model comparisons. A qualifying future benchmark must add dated issuance or dissemination evidence for a representative cohort, isolate values verified *after* that origin, and choose non-overlapping storm units. Treat naturally regulated/inflow/local-volume mismatches as exclusions, retain the reason, and do not fabricate a naturalized reference.
