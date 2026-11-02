# NLDI basin screening for geolocated FEMA floods — 2026-11-02

Run: https://github.com/leonmarienga/ML-thesis/actions/runs/41404741366

## Outcome-blind geometric screen

All 92 FEMA Flood declarations from FY2019–2024 were considered; 87 had geolocated NOAA event reports. Earlier proximity screen found 84/87 disasters with a USGS gauge <=25 km, 85/87 <=50 km, 3,247 candidate station pairs across 137 counties.

The next screen selected at most **two closest <=25 km stations per FEMA flood**, producing **168 disaster–station pairs**, with **158 unique stations**, then obtained their USGS NLDI upstream-basin geometries and intersected NOAA *report points*.

- Valid NLDI basin polygon: **153/168 pairs**, representing **80 disasters**.
- NOAA report point inside gauge's upstream basin polygon: **83/168 pairs**, representing **57 disasters**.
- 13 station queries not indexed (HTTP 404), 2 basin geometries unusable; absent information must not be called negative attribution.
- No disaster funding target or funding band used in candidate selection or screening.
- All 87 records remain in the source population; NLDI fail/missing cases are not converted to zero physical severity.

**Interpretation:** The 57 disasters have preliminary drainage-connectivity evidence that is stronger than county or distance alone, but *not* verified flood causation, affected-property exposure, or decision-time available evidence. A NOAA point may describe an affected locality rather than an inundated river reach. Basin polygons may exclude valid downstream or coastal situations. The algorithm screens only two candidates per case, so 57/87 is not the fraction of floods with genuinely usable gauges.

## Next data step

For a limited, quality-reviewed sample of the 57, download event-time and strictly prior-year USGS discharge; calculate normalized peak daily-mean discharge and seasonal percentile using the pre-specified reference calculation. Determine station's operational parameter support and sample completeness. Separately add NASA IMERG rainfall tied to location/time and Census tract housing intersecting verified footprints.

No model training completed in this phase; Track A frozen 830/912, $5.015M MAE and R² .8868 unchanged.
