# USGS flood gauge normalization audit — completed

Successful action: https://github.com/leonmarienga/ML-thesis/actions/runs/41402706099

## Aim and population
Twelve selected 2019–2024 FEMA Flood disasters from the earlier, outcome-blind NOAA county pilot. Six have USGS daily-mean streamflow candidates. USGS site metadata gives separate `state_cd` (2 digits) and `county_cd` (3 digits); the first check incorrectly compared only the 3-digit county component against a 5-digit FIPS. Rechecking concatenated full state+county FIPS verified **6/6 station counties**.

## Retrospective seasonal streamflow anomalies
For each selected station, the method retrieves USGS NWIS daily mean streamflow (00060 / statistic 00003) for the FEMA event window, plus up to ten years of **pre-event** daily mean observations. The peak event-window daily mean flow is compared to historical prior-year daily means within 30 calendar days of the peak calendar day. Reference eligibility: >=60 seasonal days from >=3 distinct years. Six of six available candidates passed.

| FEMA record | USGS station | Peak daily mean (cfs) | Prior-season median (cfs) | Peak/median | Prior seasonal daily-mean percentile |
|---|---|---:|---:|---:|---:|
| 4430 | 05458900 | 9,740 | 795 | 12.25× | 100.00% |
| 4435 | 06904500 | 14,800 | 497.5 | 29.75× | 99.34% |
| 4405 | 06207500 | 7,810 | 2,510 | 3.11× | 96.23% |
| 4424 | 03237500 | 11,900 | 311 | 38.26× | 99.67% |
| 4416 | 0817887350 | 1,480 | 11.55 | 128.14× | 100.00% |
| 4402 | 04026390 | 182 | 0.38 | 478.95× | 99.18% |

These are strong *station-level hydrological anomaly candidates*, **not** certified flood severity for those FEMA records. Ratios can be unusually large for very low seasonal median streamflow; do not rank disasters by ratios alone. The percentile is a rank among *daily means*, not an annual peak-flood return period or recurrence probability.

## Critical limitations
- Being in the same county does not demonstrate the gauge is in a causally relevant flooded watershed or affected flood footprint.
- Current USGS daily records may be revised after the disaster; the use of pre-event calendar years in the reference does not establish versioned t0 availability.
- A 21-day cap on event windows, and sampling one county/station per FEMA disaster, can mischaracterize complex flood geography.
- 12-case pilot was selected deterministically, not to represent all US flood disasters; six missing gauges remain missing and are **not** zero flow.
- NASA IMERG rainfall, river gauge stage, satellite flood masks, exposed housing and geographically aligned hurricane damage are still missing.
- No funding model was trained or updated. The accepted 830/912 router and $5.015M overall MAE remain frozen.

## Next experimental gate
Validate USGS gauge catchment hydrology and event footprint using station coordinates/HUC and verified county/watershed relationships; broaden station selection for all 87 county-matched recent floods; then acquire independently georeferenced rainfall and exposure before six-band prediction ablation.
