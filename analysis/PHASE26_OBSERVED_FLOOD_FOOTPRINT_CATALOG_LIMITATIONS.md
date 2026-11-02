# Phase 26 — Observed flood-footprint availability: initial catalog audit

GitHub Actions run (completed): https://github.com/leonmarienga/ML-thesis/actions/runs/41418259711

## Research goal
Identify **observed event-specific flood inundation polygons** matched to 92 FEMA Flood cases (FY2019–2024), not FEMA static flood-hazard/risk-zone maps. Preserve flood intensity vs geographic exposure distinction. Later verified polygons could be intersected with housing/population data.

## Results, rigorously interpreted
- FEMA Flood cohort 92, geolocated NOAA event locations 87.
- Copernicus EMS public activation endpoint responded without request errors, but the version-1 extraction/filters produced **zero candidate event records**.
- USFIMR homepage responded, but the version-1 HTML-table parser produced **zero catalog rows** for 2019–2024.
- Therefore the number of events **with actual observed maps is UNKNOWN, not zero**. A zero parser result is not evidence of absent source coverage. A later source-schema validation, alternative archives, and explicit 2019–2024 spatial index are required.
- Event-specific observed inundation polygons successfully ingested: **0**.
- Polygon-to-FEMA event overlap and exposed households: **not computed**.
- No FEMA funding model trained; accepted 830/912 router and MAE ~$5.015M unchanged.

## Immediate methodological action
Validate the actual endpoint response schemas and find original event polygon download URLs for both catalogs. Only once those source formats and timestamps are verified should a coverage percentage be reported. Distinguish Copernicus mapped AOI, satellite surface water, and event-specific flood extent; all three are different spatial concepts. Do not substitute FEMA floodplain risk layers for observed event water.

## Alternative source timing
NASA OPERA DSWx products are primarily suitable for recent events, with 2023+ HLS coverage and later radar products; converting water occurrence to *flood* requires quality masking and background/permanent-water comparisons. Product retrieval and spatiotemporal validation remain pending.

## Files
Action artifact contains audit CSV and summary JSON. Source: `analysis/flood26_observed_footprint_catalog_audit.py`.
