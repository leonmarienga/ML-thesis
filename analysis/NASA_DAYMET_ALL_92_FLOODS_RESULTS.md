# NASA Daymet 92-FEMA-flood geographic-independent rainfall study

Successful GitHub Actions run: https://github.com/leonmarienga/ML-thesis/actions/runs/41410421735

## Fixed sample and method
FEMA Flood declarations FY2019–2024: 92. NOAA event-to-FEMA linkage provided 87 geolocated flood cases. For each FEMA case, the selected rainfall location is an actual NOAA Flood/Flash Flood report point closest to the median coordinates of its event-linked reports. No USGS gauge or FEMA funding outcome is used for location choice. The aim is to avoid selection conditioned on river gauge coverage in the earlier 24-case pilot.

NASA ORNL DAAC **Daymet** single-pixel 1-km station-informed gridded daily precipitation (not NASA IMERG satellite rainfall). Event-time exposure limited to <=21-day FEMA incident window, with prior-year seasonal precipitation context drawn from up to ten pre-event calendar years. Stored features include event rain sum, maximum daily precipitation, trailing 3/7-day maxima ending during the event window, antecedent rain, rainy-day counts and seasonal peak 3-day percentile. No funding model fit.

## Observed completeness
- **69/92** FEMA Flood cases yielded usable rainfall under >=3 event days and >=80% event-window completeness.
- **87/92** had at least one NOAA report location; 18 geolocated cases were excluded by the event window completeness rule, and five lacked NOAA location.
- By fiscal year, usable rain: 2019 23/27; 2020 5/6; 2021 5/8; 2022 8/10; 2023 14/20; 2024 14/21.
- Event rainfall median **83.43 mm** among 69; 90th percentile **315.626 mm**.
- Median maximum trailing 3-day accumulation **71.44 mm**.
- Among FEMA disasters, 90th percentile of the selected location's report-point spread is **393.68 km**. This illustrates how unrepresentative a single NOAA report pixel can be for large declarations.
- The previous 24-flood pilot had 22/24 cases but used gauge-informed location selection. The resulting point-specific rainfall values are not directly interchangeable.

## Crucial limitations before modelling
- Single-point rain is not impacted-area rainfall or realized economic loss. Multi-location spatial aggregation and exposure-aware geometry are needed.
- NOAA final report point geographies and Daymet retrospective daily fields have not been authenticated as information available at declaration or T+72.
- Short FEMA incidents can fail a >=3 observation threshold despite valid 1- or 2-day rainfall. Separate sensitivity to the eligibility rule *without changing frozen model evaluation*.
- NOAA event matching can link reports from many places, and physical severity should not simply be inferred from one report location. Inspect NOAA point spread.
- Daymet is **not** GPM IMERG; do not describe the extracted features as satellite IMERG.
- Future model comparison needs comparable earlier-year data and a strictly held-out split, not a 69-case cherry-picked retrospective comparison. Include the whole disaster cohort and model feature missingness explicitly.

Frozen 830/912 router, global ~$5.015M accepted amount model, and all accepted metrics remain unchanged.
