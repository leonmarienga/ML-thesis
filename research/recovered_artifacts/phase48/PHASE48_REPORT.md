# Phase 48 — Outcome-blind, geographically distributed disaster-level flood source screen

The cohort is 36 FEMA Flood disasters with 2023/2024 incident starts and strict designated counties. Up to two separated county interiors per disaster were selected without using NOAA impact reports, radar signatures or dollar targets. A clipped 4km square was evaluated from 2023/24 monthly Landsat JRC surface-water classifications, using the same calendar month of the preceding year as a control. **This is a limited sampled-water source audit, not a disaster-wide flood estimate.**

## Aggregate measures

{
  "fema_FY2023_24_flood_declarations": 41,
  "incident_2023_24": 37,
  "eligible_with_strict_county": 36,
  "fixed_county_samples": 63,
  "unique_counties_sampled": 58,
  "pilot_disasters_included": 12,
  "new_nonpilot_disasters_included": 24,
  "real_monthly_raster_reads_attempted": 189,
  "real_monthly_raster_reads_successful": 183,
  "event_month_readable_sites": 61,
  "event_month_sites_30pct_observed": 57,
  "paired_event_baseline_sites": 61,
  "paired_event_baseline_sites_both_observed_30pct": 54,
  "cases_with_any_paired_event_baseline_site": 35,
  "cases_with_any_new_water_cells": 20,
  "observed_geographic_window_km2_before_cloud_censor": 1007.8440652783196,
  "matched_pair_new_water_within_sample_windows_km2_NOT_DISASTER_EXTENT": 1.092584725086823,
  "verified_disaster_specific_inundation_footprints": 0,
  "population_or_building_exposure_estimates": 0,
  "funding_models_fitted": 0,
  "frozen_accepted_Track_A_unchanged": true,
  "source_admin": "Official FEMA designation joined to US Census 2023 county polygons",
  "source_satellite": "JRC Global Surface Water v1.5 monthly Landsat C2 via public source.coop"
}

## Cases (prior-year same-month vs incident month)

|   disasterNumber | comparison_phase   |   sampled_counties_with_paired_data |   both_observed_pixels |   sample_candidate_water_gain_km2 |   sample_candidate_water_loss_km2 |
|-----------------:|:-------------------|------------------------------------:|-----------------------:|----------------------------------:|----------------------------------:|
|             4708 | event              |                                   2 |                  30296 |                       0           |                       0           |
|             4709 | event              |                                   1 |                  18987 |                       0.309483    |                       0.00969299  |
|             4711 | event              |                                   2 |                  24420 |                       0           |                       0           |
|             4717 | event              |                                   2 |                  29836 |                       0.00264531  |                       0.0116394   |
|             4718 | event              |                                   2 |                  56626 |                       0.391504    |                       0.00271029  |
|             4719 | event              |                                   2 |                  28446 |                       0           |                       0.00555681  |
|             4720 | event              |                                   2 |                  57252 |                       0.00386674  |                       0.00387889  |
|             4722 | event              |                                   2 |                  37658 |                       0           |                       0           |
|             4726 | event              |                                   2 |                  60669 |                       0.00206025  |                       0           |
|             4731 | event              |                                   2 |                  52523 |                       0.0375205   |                       0           |
|             4732 | event              |                                   2 |                  54644 |                       0.0070502   |                       0           |
|             4736 | event              |                                   1 |                  28797 |                       0           |                       0           |
|             4737 | event              |                                   1 |                  28715 |                       0           |                       0           |
|             4739 | event              |                                   1 |                  26219 |                       0.000566158 |                       0           |
|             4740 | event              |                                   2 |                  57254 |                       0           |                       0.0387355   |
|             4744 | event              |                                   1 |                  28647 |                       0.00222693  |                       0.00111346  |
|             4745 | event              |                                   2 |                  60406 |                       0.00217861  |                       0.000544652 |
|             4749 | event              |                                   1 |                  27242 |                       0.0317245   |                       0.0323013   |
|             4752 | event              |                                   2 |                  52019 |                       0           |                       0           |
|             4755 | event              |                                   2 |                  54158 |                       0.00116581  |                       0           |
|             4756 | event              |                                   2 |                  52704 |                       0           |                       0           |
|             4757 | event              |                                   2 |                  36498 |                       0.00341479  |                       0.0256188   |
|             4758 | event              |                                   1 |                      0 |                       0           |                       0           |
|             4780 | event              |                                   2 |                  55039 |                       0.0034468   |                       0.0234474   |
|             4781 | event              |                                   2 |                  25270 |                       0           |                       0.00196895  |
|             4787 | event              |                                   2 |                  28602 |                       0           |                       0           |
|             4789 | event              |                                   2 |                  45389 |                       0           |                       0.00104707  |
|             4793 | event              |                                   1 |                  22165 |                       0           |                       0           |
|             4797 | event              |                                   2 |                  45458 |                       0.125247    |                       0.164852    |
|             4805 | event              |                                   2 |                  43364 |                       0           |                       0           |
|             4807 | event              |                                   2 |                  56538 |                       0.12536     |                       0           |
|             4814 | event              |                                   2 |                  55137 |                       0.0011113   |                       0           |
|             4816 | event              |                                   1 |                  29041 |                       0.00110259  |                       0.0077181   |
|             4819 | event              |                                   2 |                  53870 |                       0.0265932   |                       0.0816549   |
|             4826 | event              |                                   2 |                  57341 |                       0.014317    |                       0           |

## Observability limitations

- Sampling is deterministic and outcome-blind, but only up to two counties per disaster; not a statistically weighted disaster-wide extent estimate.
- The sampled 4km squares may include tributaries or permanent lakes. Prior-year changes can reflect seasonality, management or drought unrelated to FEMA event.
- The month in which the incident began is not an event-day flood map; short-lived flooding may be missed entirely.
- A JRC class 0 pixel is UNOBSERVED and must not enter the dry denominator or paired water transition.
- Spatial JRC pixels are approximately 25-30m; 4km windows are clipped using generalized 2023 county geometry.
- A change from prior-year dry to event-month water is candidate seasonal surface-water difference, not a validated flood polygon.
- Post-event monthly imagery and final county designations are retrospective, not available as T0/T+72 features.
- This is an eligibility and coverage study; it does NOT produce a full disaster footprint, loss estimate, or an operational AI decision-support feature.
- No FEMA funding targets accessed and accepted Track A not modified.