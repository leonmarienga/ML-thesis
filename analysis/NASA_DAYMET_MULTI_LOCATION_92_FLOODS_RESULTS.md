# Phase 23 — Multi-location Daymet rainfall profiles for FY2019–2024 FEMA Floods

Completed workflow: https://github.com/leonmarienga/ML-thesis/actions/runs/41411613949

## Design
Original cohort: all **92** FY2019–2024 FEMA Flood declarations. **87** have matched NOAA Flood/Flash Flood report points. For each geolocated disaster, choose the prior medoid-nearest *actual* NOAA report point first, then at most two additional actual NOAA report coordinates chosen to maximize distance from already selected points, with a minimum 10-km spacing. Selection is independent of USGS station availability and FEMA funding.

NASA ORNL Daymet 1-km station-informed gridded precipitation (NOT IMERG) was retrieved at all selected coordinates using the existing <=21-day event-window and prior-year historical seasonal baseline protocol. Verify point-1 observations reproduce the exact previous one-location run before aggregating. No target or accepted model used.

## Results
- **248** distinct selected NOAA report points across 87 geolocated disasters.
- **79** geolocated disasters had three selected points; **3** had two; **5** had one. Five additional FEMA floods have no NOAA geolocation.
- **201/248** rainfall point records passed the fixed eligibility rule (>=3 daily measurements and >=80% of the incident window), 47 failed the rule.
- **69/92** FEMA floods had >=1 eligible rainfall measurement; previous single-point coverage **69/92**. **No gain in number of covered disasters**.
- **67/92** had >=2 eligible locations and **65/92** had all three.
- Among floods with >=2 eligible locations, median across-point event rainfall total range **58.55 mm**, 90th percentile range **182.766 mm**.
- Across the 69 with any rain, median of each flood's across-location median event rainfall total **67.67 mm**.
- Source point-1 measurements and statuses reproduced the earlier 87-flood output exactly.
- Fiscal-year coverage with >=2 eligible points: 2019 22/27, 2020 4/6, 2021 5/8, 2022 8/10, 2023 14/20, 2024 14/21.

## Interpretation
The change gives **spatial variability measurements** and demonstrates that a single NOAA report location can be unrepresentative. The larger set does NOT improve case-level rainfall availability with the original >=3-day requirement. Three selected points are not a valid proxy for an entire flood polygon; sampled maxima are not physical event-wide rainfall maxima. Observations are retrospective and source versions not verified at declaration/T+72.

## Future test
Before claims of improved funding prediction: test 1–2-day rainfall eligibility separately for short incidents, construct gridded footprint and exposure weighting, and run same-case strictly year-held-out ablations of FEMA-only vs rainfall vs flow vs combined features. The accepted 830/912 six-band router and $5.015M MAE / R² 0.8868 remain frozen.
