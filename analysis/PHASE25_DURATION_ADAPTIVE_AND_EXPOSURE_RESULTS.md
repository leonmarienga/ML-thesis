# Phase 25: Duration-aware Daymet rainfall and county-population proxy ablation

Completed GitHub run: https://github.com/leonmarienga/ML-thesis/actions/runs/41417102317

## Data correction
The original Daymet flood rainfall eligibility criterion required >=3 event days and >=80% daily observation coverage. This unnecessarily excluded 18 otherwise fully observed 1–2-day incidents. Phase25 introduces an *opt-in* >=1 event day threshold only for shorter incidents, retaining >=80% completeness and unchanged >=3-day threshold for longer incidents.

From the original 92 FY2019–24 FEMA Flood cohort:
- 69 -> **87** disasters with eligible single-point rainfall.
- 69 -> **87** disasters with >=1 eligible multi-location point.
- **82** disasters with >=2 eligible rainfall points.
- **79** disasters with >=3 eligible points.
- 18 original short incidents and 47 excluded point-level results were recomputed from NASA Daymet; all short point results passed the duration-aware rule.
- Five records still lack independently linked NOAA report coordinates.
- Original baseline ablation remains reproducible; this is a separate experiment.

## Forward-year paired funding comparison
59 flood disasters in FY2021–FY2024; train exclusively on earlier-year FEMA Floods FY2019 onwards. Identical ExtraTrees settings and fiscal-year folds, same target and 59 test cases across six challenger configurations. All metrics experimental; only retrospective covariate timelines are available.

| Configuration | MAE (USD) | R² | Six-band accuracy |
|---|---:|---:|---:|
| FEMA-only | 537,057.01 | .46768 | 76.27% |
| FEMA + duration-adaptive single-point rainfall | **530,415.23** | **.48189** | 74.58% |
| FEMA + duration-adaptive multi-point rainfall | 538,104.88 | .47067 | 76.27% |
| FEMA + multi-point rainfall + scarce USGS flow | 536,317.50 | .47572 | 76.27% |
| FEMA + NOAA-reporting-county population proxy | 576,054.65 | .31252 | 77.97% |
| FEMA + multi-point rain + population proxy | 585,961.56 | .27373 | 76.27% |

MAE delta vs FEMA-only for single-point rain: **-$6,641.78** (~-1.24%). Paired case-bootstrap exploratory 95% range of MAE differences **[-$54,473, +$37,520]**, crossing zero; no reliable improvement established. Multi rain+flow: -$739.52; proxy-only: +$38,997.63; multi+proxy: +$48,904.55. These results do NOT demonstrate benefit from simply adding physical/environmental covariates.

County population here means summed county population for counties with NOAA reports, not people flooded, displaced, economically impacted, or residing in verified inundation footprint.

## Key limitations
1. Post-disaster FEMA operational counts and finalized NOAA/Daymet features are used; although model target training respects earlier fiscal years, inputs are not independently authenticated as available at T0/T+72.
2. Only 59 held-out flood cases, with small training sets and 11 normalized USGS flow cases; rare high-dollar band conclusions are unstable.
3. The same 2021–24 data were evaluated and inspected repeatedly, so this is exploratory rather than untouched confirmatory test.
4. Disaster IDs can share meteorological events and the bootstrap does not cluster by storm.
5. One or three NOAA report points do not represent gridded flood footprint, rainfall area, exposure or humanitarian needs.
6. The accepted 912-row six-band router 830/912, final global MAE ~$5.015M and R² .8868 remain unchanged.

## Next
Obtain genuine footprint-based rainfall/exposure with independent timestamp provenance, and establish an external or untouched temporal holdout before any acceptance of enriched models. Do not optimize continuously against the same 59 cases.
