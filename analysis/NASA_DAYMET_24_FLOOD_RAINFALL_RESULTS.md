# NASA Daymet precipitation feasibility: 24 flood disasters

Successful run: https://github.com/leonmarienga/ML-thesis/actions/runs/41409555437

## Completed measurements
- Existing fixed cohort: 24 FEMA Flood disaster IDs, four each from fiscal years 2019–2024, with preliminary NLDI upstream-basin NOAA point screening.
- NASA ORNL DAAC **Daymet Single Pixel** daily 1-km gridded station-informed precipitation. **NOT GPM IMERG** and not direct satellite-only precipitation.
- 22/24 flood cases yielded complete-enough event-time rainfall (>=3 observations and >=80% of capped <=21-day incident window).
- 11/24 had previously normalized USGS flow; all 11 also have rainfall. Thus: 11 both, 11 rainfall-only, 0 flow-only, 2 neither.
- Missing rainfall in FEMA 4609 and 4678 (short incident windows, fewer than three daily observations; may be a filter artifact rather than genuine precipitation unavailable).
- Rainfall quantities: daily maximum, 3-day and 7-day trailing rolling maxima, event-window sum, >=1mm rainy days, 7-day antecedent accumulation, and peak three-day total rank compared against up to ten **earlier-year** seasonal comparator records.
- 3-day and 7-day maxima are trailing precipitation windows **ending within** the capped incident window; therefore they can incorporate days before the incident begins. They are not always contained entirely within the incident window. These values must be so labeled.
- Source and match provenance are retained for each disaster: NOAA event report point, NOAA EVENT_ID, USGS station ID and proximity, incident window and requested Daymet years.
- No disaster funding target used in feature extraction, no model trained or modified.

## Selected event rainfall examples
| FEMA disaster | Event-period Daymet rainfall sum (mm) | Trailing 3-day max ending during event (mm) | Historical seasonal percentile |
|---|---:|---:|---:|
| 4397 | 123.59 | 123.59 | 99.51% |
| 4402 | 312.45 | 140.95 | 99.84% |
| 4466 | 585.24 | 553.91 | 99.67% |
| 4477 | 4.49 | 4.49 | 70.92% |
| 4606 | 322.37 | 257.32 | 100% |
| 4755 | 179.36 | 179.36 | 100% |

## Interpretive cautions
- One NOAA reported point and Daymet pixel do not measure the entire disaster footprint, flood extent, exposure, damage, or causation.
- The rainfall source is retrospective and cannot yet be claimed available as-of FEMA declaration time.
- 10-year historical comparison is based on same-location Daymet precipitation, not a formal rainfall return period.
- Seasonal three-day windows overlap; percentiles are empirical among historical rolling windows, not independent storms.
- The two sub-three-day incident windows were excluded from completeness even if available shorter-period rainfall; retain this exclusion rule transparently.
- The current extraction focuses on flood samples with preliminary upstream-basin candidate gauges, which may preferentially exclude other affected regions.
- No improvement in FEMA funding predictions has yet been tested.

## Next experiment
Expand the rainfall extraction across the 87 geolocated floods (including ungauged cases) using an explicit cutoff and data-coverage protocol, then test physical features in a time-split paired model comparison while distinguishing retrospective Track A and early response Track B.
