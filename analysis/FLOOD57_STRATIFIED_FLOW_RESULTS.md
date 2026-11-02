# FY2019–2024: 24-disaster stratified USGS flood-flow audit

Successful GitHub Actions run: https://github.com/leonmarienga/ML-thesis/actions/runs/41406256247

## Experiment design
Previous USGS NLDI screening identified 57 of 87 geolocated FEMA Flood disasters with a NOAA report point inside at least one nearby gauge upstream basin. This new research-only pilot takes 24 of those 57, selecting four disasters per fiscal year FY2019–2024 in ascending disaster-number order, independent of funding amount. At most two NLDI-screened candidate stations per disaster (sorted by proximity). Station-flow availability determined by the current USGS NWIS Daily Values API.

Event-time daily **mean** streamflow (USGS 00060 / statistic 00003) is compared with daily mean observations in the preceding up to ten years, selecting historical days within 30 calendar days of event peak. Baseline needs >=60 seasonal observations and >=3 years. This is a *retrospective physical-feature pilot*, not as-of-time source validation.

## Empirical findings
- 24 sampled FEMA floods, 4 per year FY2019–2024.
- **11/24** floods yielded complete selected-gauge event-window daily mean data and adequate seasonal reference.
- **13/24** had no usable daily-flow records from attempted stations; these are missing, *not* zero severity.
- 32 station–disaster attempts.
- There is substantial selection bias toward gauged rivers. Three selected 2021 flood disasters? Do not infer absent year-specific counts unless read the case CSV.
- Example candidate 4397 (2019), station 01503000, peak daily mean 26,300 cfs, 23.80x prior seasonal median, seasonal daily percentile 99.18%.
- Example candidate 4676 (2023), station 03377500, peak daily mean 53,200 cfs, 2.88x prior seasonal median, seasonal daily percentile 87.87%.
- Example candidate 4755 (2024), station 01310500, peak daily mean 203 cfs, 87.88x prior median, seasonal daily percentile 100%; ratios are sensitive to small baseline flow.

## Essential limitations
Upstream-basin intersection is a hydrologic plausibility screen, not proof that reported damaging flood passed the station. One NOAA report point is not a satellite-mapped inundation polygon. Historical USGS daily mean discharge can be revised. Flood events can be shared by multiple FEMA declarations. Daily mean values may fail to capture flash floods. This is not a 57/87 coverage estimate of fully attributed disaster severity. No funding-model comparison or accuracy gains tested, and no NASA rainfall, inundation extent or property footprint exposure retrieved.

## Next defensible steps
1. Establish station-event hydrological relevance through documented stream reach and affected-area validation, and keep an explicit match-confidence class.
2. Retrieve NASA IMERG event precipitation and verify geospatial alignment for the 24-case subset; add exposure only with documented footprint.
3. Test feature availability and distribution across missing cases; report 11/24 coverage and nonrandom missingness.
4. Expand flood observations only after match-quality protocol is fixed.
5. Compare enrichment on paired, strictly held-out year disasters; preserve frozen 830/912 router and final MAE $5,014,819, R² 0.8868.
