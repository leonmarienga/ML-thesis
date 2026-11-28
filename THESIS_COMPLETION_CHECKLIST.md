# Thesis completion checklist

**Study:** AI-Assisted Decision-Support Framework for Disaster Relief Resource Allocation Under Uncertainty  
**Evidence baseline:** [b0b6edb2](https://github.com/leonmarienga/ML-thesis/commit/b0b6edb229d112dd6d11896a512bf1112414e4f6) · **Checklist date:** 2026-11-28

## Status and release rules

- **Drafted** = text exists; **Implemented** = runnable source exists; **Tested** = scoped execution passed; **Independently verified** = finding reproduced or checked independently within its stated scope; **Submission-ready** = integrated, formatted, independently checked, and cleared for the applicable academic review. An item can have several statuses; an independently verified historical result is not prospective validation.
- **Outstanding** = executable next step; **Blocked** = named dependency prevents the specified claim or deliverable. Missing independent monetary records block promotion of a model, not a defensible supervisor-review package.
- **Objective 1:** Deliver a fixed, dated supervisor-review release. Include defensible negative results and limitations; do not move the release gate whenever a new experiment begins. **Objective 2:** Develop and test independent research candidates separately. Integrate only results with preserved protocols, provenance, reproducible logs, and correctly scoped verification into a new version.
- Do not alter frozen Track A files or expose outcomes for the 143 reserved FY2025 declarations. Separate historical obligations, economic losses, humanitarian need, simulated service, and confirmed receipt. No force-push or changes to the default branch. Omit embargoed, restricted, credential-bearing and unpublished third-party material from public artifacts.

### Historical benchmark interpretation

The 830/912 frozen count is an exactly reproducible historical artifact. It is **not a target-blind validation score**. In the final historical router lineage, the [825→826 Severe Ice rule](analysis/nonbio_825_severe_ice_operational_dominance.py#L70-L79), the [826→827 Fire rule](analysis/nonbio_826_fire_duration_floor.py#L30-L36), the [827→829 rule](analysis/nonbio_827_hurricane_zero_mission_prereq.py#L15-L40) and [829→830 rule](analysis/nonbio_829_hurricane_amendment_floor.py#L19-L36) select the applicable hazard rows with actual cleaned obligations below $50M before making the held-year split. Because that filter applies to held-year examples, their outcomes enter postprocessing eligibility even if rule parameters are estimated on training years. Preserve the original predictions and metrics as historical evidence; do not deploy the rules or subtract their recorded gains to claim an independently validated alternative. Discuss this separately from nonchronological LFYO and decision-time feature availability.

## Research objectives and questions

| ID | Item | State and evidence | Acceptance action |
| --- | --- | --- | --- |
| O1 / RQ1 | Can historical FEMA obligations and six funding bands be reconstructed, and does time-valid physical/exposure evidence improve forward prediction? | Independently verified historical routing; forward comparisons tested with mixed results. [Core results](research/FINAL_METRICS.csv) | Describe estimands and six-band lineage, including target-conditioned routing; report accuracy, macro F1, recall, dollar errors and matched forward comparisons without conflating protocols. |
| O2 / RQ2 | Which authentic and synthetic sources improve physical severity or rare-funding estimates on independent episodes? | Documented acquisitions, unsuccessful trials and exclusions; 20× exploratory. [Augmentation](research/augmentation/REPORT.md), [source registry](research/DATA_SOURCE_REGISTRY.csv), [coastal](research/continuous/coastal_transfer/REPORT.md) | Retain negative findings; continue episode-disjoint, chronology-correct source work; count acquisition and matched observations separately from skill. |
| O3 / RQ3 | How well do uncertainty intervals and review flags communicate risk across years and rare bands? | Tested historical forward calibration; extreme-tail coverage weak. [Core results](research/FINAL_METRICS.csv) | Document calibration windows, finite sample ranks and coverage/width by band; mark abstention and limits in UI. |
| O4 / RQ4 | How do costs, risks, minimum service, fairness, travel time and vulnerability trade off in simulated allocation? | Tested simulation and frontier; not observed delivery impacts. [Transport](research/continuous/transport_pareto/REPORT.md) | Preserve scenario assumptions, sensitivity and denominator. No superiority claim from the exploratory legacy RL or interface. |
| O5 / RQ5 | Can a reproducible human-in-the-loop prototype explain uncertainty and compare policies without treating obligations as need? | Implemented baseline CLI and private offline prototype; final integration and user workflow QA outstanding. [Technical guide](research/TECHNICAL_GUIDE.md) | Verify complete reviewer workflow in a clean environment, including evidence warnings, simulation provenance and exported comparison. No claim of human-participant evaluation without verified records and approvals. |

## Six chapters and front/back matter

| Section | State | Evidence and next gate |
| --- | --- | --- |
| Front matter: title; degree/school; declaration/disclosure; acknowledgements; abstract; keywords; contents; lists | Abstract drafted; official institutional front matter outstanding | Verify current EUL title format, required abstracts/languages, disclosure wording and supervisor process. No signed attestations on the author’s behalf. |
| 1 Introduction: 1.1 operational context; 1.2 problem; 1.3 aims; 1.4 research questions; 1.5 scope and significance; 1.6 contributions and roadmap | Drafted | Tie all questions to results and limitations; distinguish financial and service estimands explicitly. |
| 2 Literature: 2.1 disaster response/MA finance; 2.2 hazard, exposure and provenance; 2.3 hierarchical and rare-event prediction; 2.4 synthetic and physics-informed learning; 2.5 calibration; 2.6 optimization, RL and human oversight; 2.7 research gap | Drafted; expanded synthesis outstanding | Replace source lists with comparison of designs, evidence gaps and why each thesis method is appropriate. Verify every reference below. |
| 3 Methodology: 3.1 cohort and outcomes; 3.2 original model and six-band routing; 3.3 specialists and alternatives; 3.4 authentic linkage/time of availability; 3.5 synthetic generation and transfer; 3.6 temporal and episode validation; 3.7 uncertainty; 3.8 allocation, routing and RL; 3.9 software/provenance/ethics | Drafted; six-band and DQN lineage integrated and archive counts independently verified | [Working manuscript](research/submission_review/THESIS.md), [technical crosswalk](research/submission_review/SIX_BAND_METHODOLOGY.md). Complete original two-stage development, full alternative architecture and Django audit; do not infer participant results. |
| 4 Results: 4.1 baseline and variants; 4.2 forward and hazard evidence; 4.3 synthetic/catastrophic findings; 4.4 uncertainty and sensitivity; 4.5 allocation, routing and RL; 4.6 prototype and replay | Drafted; core numbers independently verified; tables outstanding | Include purpose, implementation, design, population, metrics and scientific interpretation for each named experiment. Preserve CIs, misses and failed trials, not just rankings. |
| 5 Discussion: 5.1 question-by-question interpretation; 5.2 generalization, selection and leakage; 5.3 uncertainty, equity and human oversight; 5.4 external validity and operational limits; 5.5 comparison to prior literature | Drafted | Explain why modeled reachability is not aid delivered, why administrative history is not severity/need and why repeated tuning is not external validation. |
| 6 Conclusions: 6.1 supported contributions; 6.2 unsupported claims and boundaries; 6.3 deployment prerequisites; 6.4 future research | Drafted | Keep prospective validation, ledger recovery, transfer and field evaluation on the future track; do not present them as completed. |
| Back matter: references; experiment cards; provenance; data dictionary; software guide; release manifest; reproduction notebook | Partial; outstanding | Cross-reference every numbered artifact, record final environment and log, and package evidence safe to disseminate. Supervisor review remains an academic gate. |

## Figure, table and equation inventory

- **Figures F1–F5 outstanding for the integrated version:** evidence/decision firewall architecture, six-band router and specialist lineage, chronological and episode-disjoint evaluation timeline, hazard linkage and source availability, data/prototype/replay architecture. Source: [methods](research/THESIS_METHODS_RESULTS_DISCUSSION.md), [technical guide](research/TECHNICAL_GUIDE.md).
- **Figures F6–F10:** confusion matrix / six-band support, historical versus forward errors and tail, uncertainty coverage and interval width, simulated Pareto frontier (baseline plot exists), transport rerouting and prototype screenshots (outstanding final QA). Source: [core metrics](research/FINAL_METRICS.csv), [transport results](research/continuous/transport_pareto/REPORT.md).
- **Tables T1–T6:** data lineage/coverage/missingness; cohorts and horizons; model lineage; Track A per-band recalls and three amount stacks; matched forward ablations; authentic hazards and timing evidence. **T7–T11:** synthetic/transfer and tail gates; uncertainty by band/year; allocation and logistics trade-offs; experiment/reproduction map; submission/software manifest. Existing source tables need final numbering, units and visual checks.
- **Formulations E1–E7:** exact half-open band intervals; historical clipped amount versus signed 36-month estimand; chronology and episode constraints; log-dollar loss/back-transform and paired group uncertainty; conformal rank/empirical coverage; simulated budgets, service/loss and fairness; transport flow/travel/capacity and legacy RL state/actions/reward. Confirm equations against the corresponding implementation rather than filling from generic formulas.

## Software, artifacts and clean-run acceptance

| Item | State | Verification or next action |
| --- | --- | --- |
| Frozen original router and amount archives; reported overlay; accepted two-low specialist distinction | Independently verified historical baseline | 830/912 routes; original about $5.131M MAE, two-low about $5.128M, reported historical overlay about $5.015M. [Core results](research/FINAL_METRICS.csv). Guard paths and keep versions distinct. |
| Original/legacy Django, two-stage XGBoost and DQN artifacts | DQN notebook source and reporting checked; two-stage XGBoost and Django audit outstanding | [Legacy DQN crosswalk](research/submission_review/SIX_BAND_METHODOLOGY.md#dqn-was-an-allocation-experiment-not-the-archived-router). Do not describe Django as the current verified prediction UI. |
| Historical source snapshots, signed action audits, target-free roster, source provenance | Existing checks passed on baseline | Record 24 retained baseline hashes previously verified; check all final artefact hashes and date, units, observation quality, scope, rights and attribution for added sources. |
| Phase 50 and baseline notebook; continuous Batches 1–14 | Tested in two passing CI jobs | [Baseline notebook replay](https://github.com/leonmarienga/ML-thesis/actions/runs/45478602031), [continuous replay](https://github.com/leonmarienga/ML-thesis/actions/runs/45478602006). Final expanded executed notebook outstanding. |
| Clean environment and file-completeness checks | Reference environment captured; final run outstanding | Use Python/OS metadata, 19 direct pins and scoped resolver capture with hashes where available; check filename sets, CSV tolerances, notebook assertions, stdout and exit status. [Requirements](research/requirements.txt) |
| Historical CLI and allocation application | Implemented; staged standalone CLI tested in isolated Python; 1,424 offline JS checks repeated; actual browser and final combined demo outstanding | [Staged QA](research/submission_review/demo_qa/REPORT.md). Run the historical CLI entry point and browser UI against fixed scenarios. Verify missingness, intervals, alternative policies, accessibility, and reproducible export; keep authentic/hindcast/synthetic states visible. |
| Rebuild guide, source package, release identity, review copy | Drafted / outstanding | Name exact version and hashes; include executed notebook, sources, methods, known exclusions, and one-command instructions, then obtain appropriate supervisor review. |
| Historical precoverage 36-month signed actions | Blocked for model confirmation | No eligible public outcomes for five exact declarations. [Ledger gate](research/continuous/legacy_ledger_gate/REPORT.md). Preserve the missing-data finding rather than imputing totals or awaiting records indefinitely. |
| FY2025 outcomes | Embargoed; excluded | Never acquire, open, train on, score, or disclose outcomes; do not make their release a submission requirement. |

## Research queue separate from manuscript release

1. **Authentic decision-time severity:** verify archive availability, variable semantics and timestamp provenance before scoring; require independent events and exact scope. Record no-fit and failed paths. [Michigan Bar QINE feasibility](https://github.com/leonmarienga/ML-thesis/blob/thesis/research-followup-20261128/research/followup_20261128/cnrfc_qine/REPORT.md) passed same-site variable matching in one storm; public issue time and generalization remain unverified.
2. **Physics-informed synthetic transfer:** extend eligible independently matched wet-site episodes; lock a datum fit independent of the held storms, training-only simulation and real-only controls. Preserve exceptions; no operational claim from retrospective best-track forcing.
3. **Catastrophic obligations:** keep fixed 20× exploratory unless genuinely comparable independent signed histories are available. New features alone do not recover missing labels; quantify false alerts and uncertainty.
4. **Allocation and logistics:** use pinned scenarios and explicit capacities, held disruption patterns, independent sensitivity and equity metrics; continue routing/prepositioning while maintaining the planned/dispatch/receipt distinction. No real delivery claims from simulation.
5. **Review loop:** pick the highest-priority feasible release task plus a bounded research task; log code/data versions, evidence, outcome including failures, next gate and artifact. Promote findings only to a new version after validity review; do not rewrite historical results.

## Significant experiments, protocols and evidence reports

Baseline evidence index; existence is not independent replay of each analysis. For the final appendix, add a purpose, method, implementation, result and interpretation card per experiment.

- [`augmentation/PROTOCOL.md`](research/augmentation/PROTOCOL.md)
- [`augmentation/REPORT.md`](research/augmentation/REPORT.md)
- [`continuous/BATCH10_PROTOCOL.md`](research/continuous/BATCH10_PROTOCOL.md)
- [`continuous/BATCH11_PROTOCOL.md`](research/continuous/BATCH11_PROTOCOL.md)
- [`continuous/BATCH12_PROTOCOL.md`](research/continuous/BATCH12_PROTOCOL.md)
- [`continuous/BATCH13_PROTOCOL.md`](research/continuous/BATCH13_PROTOCOL.md)
- [`continuous/BATCH14_PROTOCOL.md`](research/continuous/BATCH14_PROTOCOL.md)
- [`continuous/BATCH2_PROTOCOL.md`](research/continuous/BATCH2_PROTOCOL.md)
- [`continuous/BATCH3_PROTOCOL.md`](research/continuous/BATCH3_PROTOCOL.md)
- [`continuous/BATCH4_PROTOCOL.md`](research/continuous/BATCH4_PROTOCOL.md)
- [`continuous/BATCH5_PROTOCOL.md`](research/continuous/BATCH5_PROTOCOL.md)
- [`continuous/BATCH6_PROTOCOL.md`](research/continuous/BATCH6_PROTOCOL.md)
- [`continuous/BATCH7_PROTOCOL.md`](research/continuous/BATCH7_PROTOCOL.md)
- [`continuous/BATCH8_PROTOCOL.md`](research/continuous/BATCH8_PROTOCOL.md)
- [`continuous/BATCH9_PROTOCOL.md`](research/continuous/BATCH9_PROTOCOL.md)
- [`continuous/EARTHQUAKE_PILOT_PROTOCOL.md`](research/continuous/EARTHQUAKE_PILOT_PROTOCOL.md)
- [`continuous/GEOGRAPHY_PAGINATION_PROTOCOL.md`](research/continuous/GEOGRAPHY_PAGINATION_PROTOCOL.md)
- [`continuous/PROTOCOL.md`](research/continuous/PROTOCOL.md)
- [`continuous/announcement_fields/REPORT.md`](research/continuous/announcement_fields/REPORT.md)
- [`continuous/announcements/REPORT.md`](research/continuous/announcements/REPORT.md)
- [`continuous/coastal_transfer/REPORT.md`](research/continuous/coastal_transfer/REPORT.md)
- [`continuous/complete_geography/REPORT.md`](research/continuous/complete_geography/REPORT.md)
- [`continuous/component_accounting/REPORT.md`](research/continuous/component_accounting/REPORT.md)
- [`continuous/county_outages/REPORT.md`](research/continuous/county_outages/REPORT.md)
- [`continuous/county_wind_exposure/REPORT.md`](research/continuous/county_wind_exposure/REPORT.md)
- [`continuous/coverage/REPORT.md`](research/continuous/coverage/REPORT.md)
- [`continuous/coverage_specialists/REPORT.md`](research/continuous/coverage_specialists/REPORT.md)
- [`continuous/disturbance_duration/REPORT.md`](research/continuous/disturbance_duration/REPORT.md)
- [`continuous/documentary_extension/REPORT.md`](research/continuous/documentary_extension/REPORT.md)
- [`continuous/earthquake/REPORT.md`](research/continuous/earthquake/REPORT.md)
- [`continuous/filings/REPORT.md`](research/continuous/filings/REPORT.md)
- [`continuous/horizons/REPORT.md`](research/continuous/horizons/REPORT.md)
- [`continuous/humanitarian_source/REPORT.md`](research/continuous/humanitarian_source/REPORT.md)
- [`continuous/impacts/REPORT.md`](research/continuous/impacts/REPORT.md)
- [`continuous/legacy_ledger_gate/REPORT.md`](research/continuous/legacy_ledger_gate/REPORT.md)
- [`continuous/new_humanitarian_evidence/REPORT.md`](research/continuous/new_humanitarian_evidence/REPORT.md)
- [`continuous/older_announcements/REPORT.md`](research/continuous/older_announcements/REPORT.md)
- [`continuous/precoverage_documentary/REPORT.md`](research/continuous/precoverage_documentary/REPORT.md)
- [`continuous/service_floors/REPORT.md`](research/continuous/service_floors/REPORT.md)
- [`continuous/service_scenarios/REPORT.md`](research/continuous/service_scenarios/REPORT.md)
- [`continuous/storm_outage_panel/REPORT.md`](research/continuous/storm_outage_panel/REPORT.md)
- [`continuous/sustained_outage/REPORT.md`](research/continuous/sustained_outage/REPORT.md)
- [`continuous/tidal_residual/REPORT.md`](research/continuous/tidal_residual/REPORT.md)
- [`continuous/transport_pareto/REPORT.md`](research/continuous/transport_pareto/REPORT.md)
- [`continuous/water_recourse/REPORT.md`](research/continuous/water_recourse/REPORT.md)
- [`continuous/weight_influence/REPORT.md`](research/continuous/weight_influence/REPORT.md)
- [`continuous/wind_extension/REPORT.md`](research/continuous/wind_extension/REPORT.md)
- [`continuous/winds/REPORT.md`](research/continuous/winds/REPORT.md)
- [`follow_up/EXPERIMENT_PROTOCOL.md`](research/follow_up/EXPERIMENT_PROTOCOL.md)
- [`follow_up/REPORT.md`](research/follow_up/REPORT.md)

## Model and experiment registry

Listed status is the pinned registry's classification; verify any claim used in the final text against the specific artifact and validation design. [MODEL_REGISTRY.csv](research/MODEL_REGISTRY.csv).

- track_a_original_stack · preserved frozen benchmark
- track_a_reported_historical_50_200m_overlay · reported historical amount overlay preserved
- track_a_two_accepted_low_specialists · preserved frozen benchmark
- phase36_report_density · archived; repeatedly examined cohort
- phase37_physical_rainflow · archived; did not displace density
- early_response_t0_t72 · archived; original source vintage unverified
- archived_track_b_cqr · archived; high-value intervals too wide
- ablation_combined · archived exploratory candidate; no promotion over historical benchmark
- ablation_geography · archived exploratory candidate; no promotion over historical benchmark
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

