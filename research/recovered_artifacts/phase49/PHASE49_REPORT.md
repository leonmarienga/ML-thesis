# Phase 49 — Outcome-blind, geographically distributed disaster-level flood source screen

The cohort is 36 FEMA Flood disasters with 2023/2024 incident starts and strict designated counties. Phase49 expands the same Phase48 anchor sites to up to three widely separated fixed windows per selected county. Up to two separated county interiors per disaster were selected without using NOAA impact reports, radar signatures or dollar targets. A clipped 4km square was evaluated from 2023/24 monthly Landsat JRC surface-water classifications, using the same calendar month of the preceding year as a control. **This is a limited sampled-water source audit, not a disaster-wide flood estimate.**

## Aggregate measures

{
  "fema_FY2023_24_flood_declarations": 41,
  "incident_2023_24": 37,
  "eligible_with_strict_county": 36,
  "fixed_county_samples": 187,
  "unique_counties_sampled": 58,
  "pilot_disasters_included": 12,
  "new_nonpilot_disasters_included": 24,
  "real_monthly_raster_reads_attempted": 561,
  "real_monthly_raster_reads_successful": 543,
  "event_month_readable_sites": 181,
  "event_month_sites_30pct_observed": 169,
  "paired_event_baseline_sites": 181,
  "paired_event_baseline_sites_both_observed_30pct": 158,
  "cases_with_any_paired_event_baseline_site": 35,
  "cases_with_any_new_water_cells": 33,
  "observed_geographic_window_km2_before_cloud_censor": 2838.6536840477565,
  "matched_pair_new_water_within_sample_windows_km2_NOT_DISASTER_EXTENT": 8.53364965857704,
  "verified_disaster_specific_inundation_footprints": 0,
  "population_or_building_exposure_estimates": 0,
  "funding_models_fitted": 0,
  "frozen_accepted_Track_A_unchanged": true,
  "source_admin": "Official FEMA designation joined to US Census 2023 county polygons",
  "source_satellite": "JRC Global Surface Water v1.5 monthly Landsat C2 via public source.coop",
  "phase49_sampling_design": "same Phase48 county roster, original anchor plus up to two interior maximin 6.5km-separated 4km squares per county",
  "sites_added_outside_phase48_anchor": 124,
  "original_phase48_site_count": 63,
  "counties_with_three_sites": 62,
  "counties_with_fewer_than_three_sites": 1,
  "paired_sites_quality30pct": 158,
  "anchor_sites_quality30pct": 54,
  "additional_sites_quality30pct": 104,
  "paired_cases_both_anchor_and_expanded": 34,
  "case_positive_anchor_quality30pct": 20,
  "case_positive_expanded_quality30pct": 31,
  "case_newly_positive_only_after_spatial_expansion": 11,
  "quality30pct_anchor_new_water_rate_per_100_observed_km2": 0.13446923010060371,
  "quality30pct_expanded_new_water_rate_per_100_observed_km2": 0.37999192079607286,
  "geographic_heterogeneity_case_rate_median_absolute_change": 0.03907780193383989,
  "case_rank_spearman_anchor_vs_expanded": 0.5043196821283099,
  "not_a_probabilistic_or_population_weighted_disaster_sampling_design": true,
  "phase49_predeclared_before_funding_test": true,
  "phase48_anchor_new_water_reproducibility_cases": 61
}

## Cases (prior-year same-month vs incident month)

|   disasterNumber | comparison_phase   |   sampled_counties_with_paired_data |   both_observed_pixels |   sample_candidate_water_gain_km2 |   sample_candidate_water_loss_km2 |
|-----------------:|:-------------------|------------------------------------:|-----------------------:|----------------------------------:|----------------------------------:|
|             4708 | event              |                                   2 |                  86482 |                       0.01021     |                       0           |
|             4709 | event              |                                   1 |                  58886 |                       0.399366    |                       0.0753769   |
|             4711 | event              |                                   2 |                  54699 |                       0.383571    |                       0.0123462   |
|             4717 | event              |                                   2 |                  92831 |                       0.00795792  |                       0.0253562   |
|             4718 | event              |                                   2 |                 153239 |                       0.775445    |                       0.0141492   |
|             4719 | event              |                                   2 |                 112500 |                       0.01548     |                       0.169403    |
|             4720 | event              |                                   2 |                 158878 |                       0.0549551   |                       0.274678    |
|             4722 | event              |                                   2 |                  97524 |                       3.93809     |                       0.00111809  |
|             4726 | event              |                                   2 |                 185238 |                       0.0569817   |                       0.0015296   |
|             4731 | event              |                                   2 |                 154300 |                       0.0446613   |                       0.00119236  |
|             4732 | event              |                                   2 |                 161373 |                       0.00994051  |                       0           |
|             4736 | event              |                                   1 |                  86823 |                       0.0132293   |                       0.112472    |
|             4737 | event              |                                   1 |                  73994 |                       0.00382091  |                       0.00927935  |
|             4739 | event              |                                   1 |                  81041 |                       0.000566158 |                       0           |
|             4740 | event              |                                   2 |                 151066 |                       0.00391881  |                       0.213997    |
|             4744 | event              |                                   1 |                  71233 |                       0.135548    |                       0.026784    |
|             4745 | event              |                                   2 |                 155799 |                       0.00902431  |                       0.000544652 |
|             4749 | event              |                                   1 |                  81468 |                       0.0363245   |                       0.0949537   |
|             4752 | event              |                                   2 |                 148940 |                       0.0110531   |                       0           |
|             4755 | event              |                                   2 |                 134656 |                       0.0489633   |                       0.0518143   |
|             4756 | event              |                                   2 |                 142000 |                       0.000599902 |                       0           |
|             4757 | event              |                                   2 |                 117948 |                       0.0119273   |                       0.147282    |
|             4758 | event              |                                   1 |                      0 |                       0           |                       0           |
|             4780 | event              |                                   2 |                 161531 |                       0.190492    |                       0.0378678   |
|             4781 | event              |                                   2 |                  75773 |                       0.0144656   |                       0.00196895  |
|             4787 | event              |                                   2 |                  68895 |                       0.00587322  |                       0.00352393  |
|             4789 | event              |                                   2 |                  88727 |                       0           |                       0.00104707  |
|             4793 | event              |                                   1 |                  60143 |                       0.0457636   |                       0.00357607  |
|             4797 | event              |                                   2 |                 132893 |                       0.985943    |                       0.184118    |
|             4805 | event              |                                   2 |                  66610 |                       0.00146338  |                       0.00219508  |
|             4807 | event              |                                   2 |                 170163 |                       0.180061    |                       0.00836963  |
|             4814 | event              |                                   2 |                 167873 |                       0.0523576   |                       0.228818    |
|             4816 | event              |                                   1 |                  71724 |                       0.00110259  |                       0.00937608  |
|             4819 | event              |                                   2 |                 153723 |                       0.774023    |                       0.186157    |
|             4826 | event              |                                   2 |                 168527 |                       0.310467    |                       0           |

## Phase49 anchor vs expanded samples: normalized new-water rate per 100 observed km²

|   disasterNumber |   anchor_quality_sites |   anchor_observed_area_km2 |   anchor_new_water_km2 |   anchor_rate_per_100_observed_km2 | anchor_any_detected   |   expanded_quality_sites |   expanded_observed_area_km2 |   expanded_new_water_km2 |   expanded_rate_per_100_observed_km2 | expanded_any_detected   |   expanded_minus_anchor_rate |
|-----------------:|-----------------------:|---------------------------:|-----------------------:|-----------------------------------:|:----------------------|-------------------------:|-----------------------------:|-------------------------:|-------------------------------------:|:------------------------|-----------------------------:|
|             4708 |                      1 |                    15.9088 |            0           |                         0          | False                 |                        3 |                      44.4888 |              0           |                          0           | False                   |                  0           |
|             4709 |                      1 |                    13.1458 |            0.309483    |                         2.35424    | True                  |                        3 |                      40.7323 |              0.399366    |                          0.980466    | True                    |                 -1.37378     |
|             4711 |                      1 |                    15.0444 |            0           |                         0          | False                 |                        3 |                      33.7051 |              0.383571    |                          1.13802     | True                    |                  1.13802     |
|             4717 |                      1 |                    15.7851 |            0.00264531  |                         0.0167583  | True                  |                        3 |                      47.5073 |              0.00583208  |                          0.0122762   | True                    |                 -0.0044821   |
|             4718 |                      2 |                    30.8387 |            0.391504    |                         1.26952    | True                  |                        6 |                      83.4943 |              0.775445    |                          0.92874     | True                    |                 -0.340781    |
|             4719 |                      1 |                    15.8069 |            0           |                         0          | False                 |                        5 |                      62.1699 |              0.01548     |                          0.0248995   | True                    |                  0.0248995   |
|             4720 |                      2 |                    31.7003 |            0.00386674  |                         0.0121978  | True                  |                        6 |                      87.9321 |              0.0549551   |                          0.0624972   | True                    |                  0.0502993   |
|             4722 |                      2 |                    20.8266 |            0           |                         0          | False                 |                        5 |                      53.9224 |              3.93809     |                          7.30326     | True                    |                  7.30326     |
|             4726 |                      2 |                    31.1423 |            0.00206025  |                         0.00661559 | True                  |                        6 |                      94.7675 |              0.0569817   |                          0.0601279   | True                    |                  0.0535123   |
|             4730 |                      0 |                     0      |          nan           |                       nan          |                       |                        0 |                       0      |            nan           |                        nan           |                         |                nan           |
|             4731 |                      2 |                    31.7772 |            0.0375205   |                         0.118074   | True                  |                        6 |                      93.4989 |              0.0446613   |                          0.0477667   | True                    |                 -0.0703072   |
|             4732 |                      2 |                    31.4325 |            0.0070502   |                         0.0224297  | True                  |                        6 |                      92.7341 |              0.00994051  |                          0.0107194   | True                    |                 -0.0117103   |
|             4736 |                      1 |                    15.7507 |            0           |                         0          | False                 |                        3 |                      47.4911 |              0.0132293   |                          0.0278563   | True                    |                  0.0278563   |
|             4737 |                      1 |                    15.8425 |            0           |                         0          | False                 |                        3 |                      40.7451 |              0.00382091  |                          0.00937759  | True                    |                  0.00937759  |
|             4739 |                      1 |                    14.8441 |            0.000566158 |                         0.00381403 | True                  |                        3 |                      46.1121 |              0.000566158 |                          0.00122779  | True                    |                 -0.00258624  |
|             4740 |                      2 |                    31.8276 |            0           |                         0          | False                 |                        6 |                      84.0568 |              0.00391881  |                          0.0046621   | True                    |                  0.0046621   |
|             4744 |                      1 |                    15.9487 |            0.00222693  |                         0.0139631  | True                  |                        3 |                      39.6479 |              0.135548    |                          0.341878    | True                    |                  0.327915    |
|             4745 |                      2 |                    31.8524 |            0.00217861  |                         0.00683969 | True                  |                        5 |                      79.8382 |              0.00847725  |                          0.010618    | True                    |                  0.00377834  |
|             4749 |                      1 |                    15.7134 |            0.0317245   |                         0.201894   | True                  |                        3 |                      46.994  |              0.0363245   |                          0.077296    | True                    |                 -0.124598    |
|             4752 |                      2 |                    31.0597 |            0           |                         0          | False                 |                        6 |                      88.8748 |              0.0110531   |                          0.0124367   | True                    |                  0.0124367   |
|             4755 |                      2 |                    31.6789 |            0.00116581  |                         0.0036801  | True                  |                        6 |                      78.669  |              0.0489633   |                          0.0622396   | True                    |                  0.0585595   |
|             4756 |                      2 |                    31.8318 |            0           |                         0          | False                 |                        6 |                      85.7731 |              0.000599902 |                          0.000699406 | True                    |                  0.000699406 |
|             4757 |                      2 |                    20.7783 |            0.00341479  |                         0.0164344  | True                  |                        6 |                      67.038  |              0.0119273   |                          0.0177919   | True                    |                  0.00135746  |
|             4758 |                      0 |                     0      |          nan           |                       nan          |                       |                        0 |                       0      |            nan           |                        nan           |                         |                nan           |
|             4780 |                      2 |                    31.6177 |            0.0034468   |                         0.0109015  | True                  |                        6 |                      92.8305 |              0.190492    |                          0.205205    | True                    |                  0.194303    |
|             4781 |                      1 |                    15.8317 |            0           |                         0          | False                 |                        3 |                      47.5184 |              0           |                          0           | False                   |                  0           |
|             4787 |                      1 |                    15.9951 |            0           |                         0          | False                 |                        3 |                      37.7556 |              0.00587322  |                          0.0155559   | True                    |                  0.0155559   |
|             4789 |                      2 |                    24.0107 |            0           |                         0          | False                 |                        4 |                      43.3596 |              0           |                          0           | False                   |                  0           |
|             4793 |                      1 |                    15.8346 |            0           |                         0          | False                 |                        3 |                      42.974  |              0.0457636   |                          0.106491    | True                    |                  0.106491    |
|             4797 |                      2 |                    24.6816 |            0.125247    |                         0.507451   | True                  |                        6 |                      71.9904 |              0.985943    |                          1.36955     | True                    |                  0.862096    |
|             4805 |                      2 |                    31.7225 |            0           |                         0          | False                 |                        4 |                      48.735  |              0.00146338  |                          0.00300274  | True                    |                  0.00300274  |
|             4807 |                      2 |                    31.7775 |            0.12536     |                         0.394493   | True                  |                        6 |                      95.4607 |              0.180061    |                          0.188624    | True                    |                 -0.205869    |
|             4814 |                      2 |                    31.0294 |            0.0011113   |                         0.00358143 | True                  |                        6 |                      94.5075 |              0.0523576   |                          0.0554005   | True                    |                  0.051819    |
|             4816 |                      1 |                    16.0101 |            0.00110259  |                         0.00688682 | True                  |                        3 |                      39.5986 |              0.00110259  |                          0.0027844   | True                    |                 -0.00410241  |
|             4819 |                      2 |                    31.8355 |            0.0265932   |                         0.0835333  | True                  |                        6 |                      90.663  |              0.774023    |                          0.853736    | True                    |                  0.770203    |
|             4826 |                      2 |                    31.6334 |            0.014317    |                         0.0452594  | True                  |                        6 |                      92.9617 |              0.310467    |                          0.333973    | True                    |                  0.288713    |

## Coverage sensitivity

|   threshold | group             |   sites |   cases_covered |   new_water_area_sample_km2 |   observed_sample_area_km2 |   water_change_km2_per_100_observed_km2 |
|------------:|:------------------|--------:|----------------:|----------------------------:|---------------------------:|----------------------------------------:|
|         0   | anchor_only       |      61 |              35 |                     1.09258 |                    816.295 |                                0.133847 |
|         0   | all_spatial_sites |     181 |              35 |                     8.53365 |                   2257.78  |                                0.377966 |
|         0.3 | anchor_only       |      54 |              34 |                     1.09258 |                    812.517 |                                0.134469 |
|         0.3 | all_spatial_sites |     158 |              34 |                     8.5063  |                   2238.55  |                                0.379992 |
|         0.6 | anchor_only       |      50 |              34 |                     1.06138 |                    785.239 |                                0.135167 |
|         0.6 | all_spatial_sites |     150 |              34 |                     8.4751  |                   2185.19  |                                0.387842 |
|         0.8 | anchor_only       |      50 |              34 |                     1.06138 |                    785.239 |                                0.135167 |
|         0.8 | all_spatial_sites |     143 |              34 |                     7.59247 |                   2116     |                                0.358813 |

## Observability limitations

- Sampling is deterministic and outcome-blind, but only up to two counties per disaster and three sample windows per selected county; not a statistically weighted disaster-wide extent estimate.
- The sampled 4km squares may include tributaries or permanent lakes. Prior-year changes can reflect seasonality, management or drought unrelated to FEMA event.
- The month in which the incident began is not an event-day flood map; short-lived flooding may be missed entirely.
- A JRC class 0 pixel is UNOBSERVED and must not enter the dry denominator or paired water transition.
- Spatial JRC pixels are approximately 25-30m; 4km windows are clipped using generalized 2023 county geometry.
- A change from prior-year dry to event-month water is candidate seasonal surface-water difference, not a validated flood polygon.
- Post-event monthly imagery and final county designations are retrospective, not available as T0/T+72 features.
- This is an eligibility and coverage study; it does NOT produce a full disaster footprint, loss estimate, or an operational AI decision-support feature.
- No FEMA funding targets accessed and accepted Track A not modified.