# Phase 24 — Chronological flood-only funding ablation and short-event eligibility audit

Successful runs:
- [FEMA-only vs physical features](https://github.com/leonmarienga/ML-thesis/actions/runs/41415807900)
- [Short-event eligibility audit](https://github.com/leonmarienga/ML-thesis/actions/runs/41416042429)

## Model comparison

Evaluate 59 FEMA Floods in FY2021–2024 with fiscal-year-forward training on 33 (FY2019–20), 41, 51, and 71 prior-year floods. Same ExtraTrees log-dollar regression (350 trees, min_samples_leaf=2) and six-band classifier in each arm. Source records are retrospective and *not* certified available at declaration time. The accepted Track A 912-row model is not used or changed.

| Model | MAE | R² | 6-band accuracy |
|---|---:|---:|---:|
| FEMA only | $537,057.01 | 0.46768 | 76.27% |
| FEMA + single-location Daymet | $558,919.07 | 0.43600 | 74.58% |
| FEMA + multi-location Daymet | $550,304.94 | 0.45786 | 74.58% |
| FEMA + multi-location Daymet + USGS | $559,296.21 | 0.43567 | 74.58% |

Paired overall additional MAE vs FEMA-only: single rainfall +$21,862.06 (exploratory case-resampling 95% range -$20,134 to +$72,992); multi rainfall +$13,247.93 (range -$21,068 to +$54,913); multi+flow +$22,239.20 (range -$8,432 to +$57,914). Those descriptive ranges span zero and are not storm-cluster-adjusted. **No empirically supported improvement over FEMA-only.**

## Data coverage
92 flood records FY2019–2024. Daymet rainfall available for 69 under >=3-day protocol; river discharge for only 11 in the year-balanced earlier screened gauge subset. The extra features are retrospective and may not correspond to the physical area generating actual FEMA obligations.

## Why 18 rainfall cases were excluded

From the exact original single-location Daymet artifact, the 18 NOAA-geolocated missing-rain cases all had **100% complete daily rainfall observation coverage** but a FEMA incident window of 1 or 2 days: five 1-day and thirteen 2-day cases. The fixed eligibility rule required 3 observations. This is a procedural exclusion and not missing source observations. Consequently 87/92 would *potentially* have rainfall if duration-aware rules were used and features were recomputed; actual eligible rainfall remains 69/92. This audit does NOT recompute those features or retest model accuracy.

## Research decision

Retain the frozen six-band router 830/912 (91.01%), global accepted MAE $5,014,819 and R² 0.8868. Keep the exploratory models as negative findings. Before any improved severe-flood funding claim, consider properly area-weighted rainfall, time-authenticated inputs, dataset size, and duration-aware feature eligibility with independently specified temporal comparisons.
