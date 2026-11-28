# Thesis completion checklist

**Study:** AI-Assisted Decision-Support Framework for Disaster Relief Resource Allocation Under Uncertainty  
**Evidence baseline:** [b0b6edb2](https://github.com/leonmarienga/ML-thesis/commit/b0b6edb229d112dd6d11896a512bf1112414e4f6) · **Checklist date:** 2026-11-28

## Status and release rules

- **Drafted** = text exists; **Implemented** = runnable source exists; **Tested** = scoped execution passed; **Independently verified** = finding reproduced or checked independently within its stated scope; **Submission-ready** = integrated, formatted, independently checked, and cleared for the applicable academic review. An item can have several statuses; an independently verified historical result is not prospective validation.
- **Outstanding** = executable next step; **Blocked** = named dependency prevents th…6089 tokens truncated…n- ablation_geography · archived exploratory candidate; no promotion over historical benchmark
- ablation_geography_physical · archived exploratory candidate; no promotion over historical benchmark
- ablation_metadata · archived exploratory candidate; no promotion over historical benchmark
- ablation_physical · archived exploratory candidate; no promotion over historical benchmark
- ablation_report_proxy · archived exploratory candidate; no promotion over historical benchmark
- ablation_retrospective_metadata · archived exploratory candidate; no promotion over historical benchmark
- ablation_verified_link_only · archived exploratory candidate; no promotion over historical benchmark
- equal_blend_verified · archived exploratory candidate; no promotion over historical benchmark
- hard_funding_band · archived exploratory candidate; no promotion over historical benchmark
- hazard_experts_verified · archived exploratory candidate; no promotion over historical benchmark
- hybrid_hazard_band · archived exploratory candidate; no promotion over historical benchmark
- nested_combined · archived exploratory candidate; no promotion over historical benchmark
- nested_metadata · retained exploratory reference; strongest observed macro F1
- nested_verified_link_only · archived exploratory candidate; no promotion over historical benchmark
- oof_weighted_blend_verified · archived exploratory candidate; no promotion over historical benchmark
- soft_funding_band · archived exploratory candidate; no promotion over historical benchmark
- forward_residual_dollar_and_log1p · archived; limited tail coverage
- synthetic_sequential_policy_suite · research simulation with independent need
- ablation_real_only · Matched real-only reference
- ablation_cost_sensitive · Exploratory control; tail MAE improvement and no tail interval coverage
- ablation_random_repeats · Exploratory control; tail MAE gain with poorer probability calibration
- ablation_smote · Earlier tuning support gate closed; fixed FY2024 runs retained
- ablation_smotenc · Exploratory; limited same-state same-hazard support and no promotion
- ablation_smoter · Exploratory; no improvement in approved outcomes
- ablation_smogn · Exploratory; no convincing gain and feature-vector repetition
- ablation_gaussian_copula · Support gate closed; no generator fit
- ablation_hybrid · Exploratory; no convincing gain
- ablation_ctgan · Support gate closed; no generator fit
- usace_flood_impact_pilot · Economic demonstration with no FEMA obligation labels
- coverage_screened_extra_trees · Exploratory; training eligibility correction retained
- coverage_screened_extra_trees_weight_20 · Exploratory; no model promotion
- pga_bjf97_hybrid_ridge · Exploratory; below integration threshold
- historical_horizon_audit · Data audit; no fitted funding model
- clean_water_recourse_cvar · Synthetic policy comparator; below integration gate
- complete_pages_verified_physical · Exploratory; recovered input version is retained
- older_announcement_maturity_audit · Authentic data audit
- doe417_pooled_empirical · Matched real-data baseline
- doe417_seasonal_empirical · Primary matched real-data baseline
- doe417_seasonal_duration_prior_v1 · Exploratory synthetic auxiliary comparator
- doe417_half_mixture_v1 · Exploratory mixture below practical gain threshold
- documentary_timing_mature_group_gate · Feasibility audit. General and tail design gates closed.
- eaglei_24h_threshold_observability_v1 · Auxiliary source audit. One sampled year.
- precoverage_signed_obligation_gate_v1 · Eight verified documents. No new target labels.
- eaglei_sustained_real_logistic_v1 · Auxiliary chronological research comparator
- eaglei_sustained_markov_hybrid_v1 · Negative result. Below locked integration threshold.
- component_record_label_gate_v1 · Source audit with zero eligible declaration labels
- county_wind_recovery_support_v1 · Auxiliary feasibility audit. Independent storm screen closed.
- legacy_exact_ledger_gate_v1 · No new comparable signed declaration outcomes
- weather_storm_outage_panel_gate_v1 · No generator fit. New held group gate failed.
- independent_humanitarian_source_gate_v1 · Empirical benchmark context. No causal allocation estimate.
- batch11_eventual_coverage_nested · exploratory nested historical selector, no promotion
- batch11_signed_elapsed_nested · small uncertain signed-endpoint result
- batch11_service_recourse · simulation trade-offs, no operational promotion
- batch12_tide_residual_persistence · Retrospective physical comparator
- batch12_service_priority_floor · Transparent simulation option
- batch13_signed_weight_fixed20 · exploratory amount candidate; original predictions preserved
- batch13_coastal_ridge · retrospective statistical comparator
- batch13_coastal_synthetic_prior · retained negative standalone and hybrid gain
- batch14_transport_multi_objective · simulation only

## Authentic and derived datasets

Verify provenance, coverage, units, rights, target exclusion and restrictions for each item used in the final release. [DATA_SOURCE_REGISTRY.csv](research/DATA_SOURCE_REGISTRY.csv).

- original_workbook · Original thesis repository
- openfema · FEMA
- openfema_target_free · FEMA
- census_geometry · US Census Bureau
- census_popest · US Census Bureau
- noaa_stormevents · NOAA NCEI and National Weather Service
- nhc_hurdat · NOAA National Hurricane Center
- nhc_advisories · NOAA National Hurricane Center
- mtbs · USGS and USDA Forest Service
- usgs_comcat · USGS
- daymet · ORNL DAAC and NASA
- usgs_legacy_flow · USGS
- jrc_monthly · European Commission JRC and Google
- usgs_daily_ogc · USGS
- nasa_gibs · NASA Earthdata GIBS
- nasa_laads · NASA LAADS DAAC
- allocation_simulation · This study
- augmentation_historical_input · This study and FEMA
- augmentation_generated_features · This study
- usace_nsi_22063 · USACE
- usace_egm0103 · USACE
- usace_ec_loss_scenarios · This study
- coverage_dds_dr1995_2024 · FEMA
- coverage_ma_actions_historical · FEMA
- pga_usgs_station_records · USGS and networks
- bjf97_synthetic_prior · This study, based on Boore et al.
- fema_fr_notice_public_inspection · Office of the Federal Register and FEMA
- noaa_ncei_significant_earthquake_impacts · NOAA NCEI and WDS
- historical_complete_ma_horizons · FEMA
- who_emergency_safe_water · WHO
- synthetic_safe_water_recourse · This study
- nara_biden_original_approvals · White House and National Archives
- nhc_atcf_operational_radii · NOAA NHC
- coops_coastal_winds · NOAA CO-OPS
- openfema_complete_pages_original_snapshot · OpenFEMA
- nara_documentary_program_fields · White House and National Archives
- nhc_coops_wind_validation_extension · NOAA NHC and NOAA CO-OPS
- nara_2017_2020_early_notice_versions · White House and National Archives
- internet_archive_presidential_early_replay · Internet Archive
- doe_417_annual_summary · U.S. Department of Energy
- internet_archive_presidential_early_extension · Internet Archive and NARA
- ornl_eaglei_figshare_2018_v1 · Oak Ridge National Laboratory and authors
- mesa_appa_ereliability_historical_sample · City of Mesa electric utility
- federal_precoverage_ma_documentary · GAO, DHS OIG and United States Senate
- ornl_eaglei_annual_transfer_v2 · Oak Ridge National Laboratory and authors
- federal_component_audits_2005_gulf · DHS and NASA Inspectors General and U.S. House
- census_points_2017_wind_outage · U.S. Census Bureau and NOAA NHC
- openfema_exact_precoverage_five · FEMA and DataLumos preservation catalogue
- nhc_eaglei_geographic_panel_v1 · NOAA NHC, U.S. Census and ORNL
- census_hps_disaster_weeks_52_61_62 · U.S. Census Bureau
- cdc_casper_usvi_2017_2018 · CDC and US Virgin Islands Department of Health
- fema_shelter_tempo_schema_only · FEMA and American Red Cross
- batch11_signed_labels · FEMA OpenFEMA
- batch11_casper_anchors · CDC and USVI Department of Health
- rockaway_sandy_casper · Subaiya, Moussavi, Velasquez and Stillman
- michael_casper_cross_sections · Florida Department of Health
- eia_pr_maria · EIA
- noaa_ida_pilots · NOAA National Ocean Service
- coops_tidal_residual · NOAA CO-OPS
- batch13_noaa_coastal_forcing · NOAA CO-OPS
- batch14_transport_demand_anchors · CDC and USVI Department of Health

## Synthetic dataset and simulator registries

- [`continuous/coastal_transfer/SYNTHETIC_DATA_REGISTRY.csv`](research/continuous/coastal_transfer/SYNTHETIC_DATA_REGISTRY.csv)
- [`continuous/disturbance_duration/SYNTHETIC_DATA_REGISTRY.csv`](research/continuous/disturbance_duration/SYNTHETIC_DATA_REGISTRY.csv)
- [`continuous/service_floors/SYNTHETIC_DATA_REGISTRY.csv`](research/continuous/service_floors/SYNTHETIC_DATA_REGISTRY.csv)
- [`continuous/service_scenarios/SYNTHETIC_DATA_REGISTRY.csv`](research/continuous/service_scenarios/SYNTHETIC_DATA_REGISTRY.csv)
- [`continuous/sustained_outage/SYNTHETIC_DATA_REGISTRY.csv`](research/continuous/sustained_outage/SYNTHETIC_DATA_REGISTRY.csv)
- [`continuous/transport_pareto/SYNTHETIC_DATA_REGISTRY.csv`](research/continuous/transport_pareto/SYNTHETIC_DATA_REGISTRY.csv)

## References to reconcile and verify

Each entry needs metadata, identifier, primary-source and claim-to-citation check in the final manuscript; add missing foundational and newly used sources rather than citing this list by itself. [REFERENCE_REGISTRY.csv](research/REFERENCE_REGISTRY.csv).

- pekel2016 · 2016 · High-resolution mapping of global surface water and its long-term changes
- tellman2021 · 2021 · Satellite imaging reveals increased proportion of population exposed to floods
- barber2023 · 2023 · Conformal prediction beyond exchangeability
- holguin2013 · 2013 · On the appropriate objective function for post-disaster humanitarian logistics models
- jrcdoc · 2026 · Global Surface Water data access and product notes
- nws2023 · 2023 · Mississippi River Flood of 2023
- state2024 · 2024 · Disaster assistance legislative report
- chawla2002 · 2002 · SMOTE: Synthetic Minority Over-sampling Technique
- torgo2013 · 2013 · SMOTE for Regression
- branco2017 · 2017 · SMOGN: A Pre-processing Approach for Imbalanced Regression
- xu2019 · 2019 · Modeling Tabular Data using Conditional GAN
- usace2000 · 2000 · Generic Depth-Damage Relationships
- usace2026 · 2026 · National Structure Inventory technical documentation
- femama2024 · 2024 · Mission Assignments Policy 104-010-3
- bjf1997 · 1997 · Equations for estimating horizontal response spectra and peak acceleration from western North American earthquakes
- marano2026 · 2026 · ShakeMap Atlas V4 and AtlasCat
- brelsford2024 · 2024 · A dataset of recorded electricity outages by United States county 2014–2022
- gao2006gulf · 2006 · Disaster Relief: Governmentwide Framework Needed to Collect and Consolidate Information to Report on Billions in Federal Funding for the 2005 Gulf Coast Hurricanes
- gao2008cost · 2008 · Disaster Cost Estimates: FEMA Can Improve Its Learning from Past Experience and Management of Disaster-Related Resources
- dhs_oig_2009_gulf · 2009 · Consolidated Report on DHS’ Management of 2005 Gulf Coast Hurricanes Mission Assignment Funding
- nasa_oig_2007_katrina · 2007 · Final Memorandum on the Audit of Federal Emergency Management Agency Mission Assignments for Hurricane Katrina Disaster Relief
- schnall2019usvicasper · 2019 · Addressing Community Needs During the Hurricane Response and Recovery Efforts Through Community Assessments for Public Health Emergency Response (CASPER)—United States Virgin Islands, 2017–2018
- omb2007missionaccounting · 2007 · Year-end Accounting Guidance for Disaster Relief Fund Transactions
- subaiya2014 · 2014 · A Rapid Needs Assessment of the Rockaway Peninsula in New York City After Hurricane Sandy and the Relationship of Socioeconomic Status to Recovery
- rockaway_casper_30488477 · 2019 · A modified Community Assessment for Public Health Emergency Response (CASPER) four months after Hurricane Sandy
- michael_casper_34081673 · 2022 · Community Assessment for Public Health Emergency Response (CASPER) Following Hurricane Michael, Bay and Gulf Counties, Florida, 2019
- michael_casper_38189201 · 2023 · Community Assessment for Public Health Emergency Response (CASPER) following Hurricane Michael: Gadsden, Calhoun, and Jackson Counties, Florida, 2020
- noaa_hourly_height_flags · The Data API Response Help
- jason3_inverse_barometer · Jason-3 Products Handbook
- wfp_road_logistics · Sending Goods by Road

