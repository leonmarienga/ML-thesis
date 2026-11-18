# Reproduce the first continuous research batch

Run from the repository root on x86-64 Linux with Python 3.12 and `research/requirements.txt`. The original source snapshots allow offline replay. Source-page retrieval in the examples verifies cached content and source URLs before parsing. [Numeric runtime details](../augmentation/REPRODUCE.md).

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_CORETYPE=Haswell
export NPY_DISABLE_CPU_FEATURES=X86_V4,AVX512_ICL,AVX512_SPR
sha256sum -c research/continuous/coverage/inputs/SHA256SUMS
sha256sum -c research/continuous/earthquake/inputs/SHA256SUMS
mkdir -p work/continuous-replay/{dds,usgs,coverage,ablation,pga-input,pga}
tar -xzf research/continuous/coverage/inputs/openfema_public_responses.tar.gz -C work/continuous-replay/dds
tar -xzf research/continuous/earthquake/inputs/usgs_public_responses.tar.gz -C work/continuous-replay/usgs
python analysis/thesis_historical_coverage.py work/continuous-replay/dds work/continuous-replay/coverage
python analysis/thesis_precoverage_ablation.py work/continuous-replay/ablation
python analysis/thesis_earthquake_auxiliary.py work/continuous-replay/pga-input work/continuous-replay/pga --fetch work/continuous-replay/usgs
jupyter execute research/THESIS_REPRODUCTION.ipynb
```

For the earthquake model alone, pass the retained `research/continuous/earthquake/inputs` folder without the paired fetch argument. The archive also retains the initial narrow query and data to support the recorded coverage diagnostic. The replay never joins the locked FY2025 cohort to obligations.

## Publication and impact evidence replay

```bash
sha256sum -c research/continuous/BATCH2_SHA256SUMS
mkdir -p work/continuous-replay/{filing-snapshot,impact-snapshot,filings,impacts}
tar xf research/continuous/filings/inputs/federal_register_public_responses.tar.gz -C work/continuous-replay/filing-snapshot
tar xf research/continuous/impacts/inputs/noaa_impact_public_responses.tar.gz -C work/continuous-replay/impact-snapshot
python analysis/thesis_declaration_publication.py work/continuous-replay/filing-snapshot work/continuous-replay/filings
python analysis/thesis_earthquake_impact_audit.py work/continuous-replay/impact-snapshot work/continuous-replay/impacts
```

## Obligation horizons and water simulation replay

```bash
sha256sum -c research/continuous/horizons/inputs/SHA256SUMS
mkdir -p work/continuous-replay/{actions-snapshot,horizons,water-recourse}
tar xf research/continuous/horizons/inputs/historical_mission_actions_responses.tar.gz -C work/continuous-replay/actions-snapshot
python analysis/thesis_obligation_horizon_audit.py work/continuous-replay/actions-snapshot work/continuous-replay/horizons
python analysis/thesis_water_recourse.py work/continuous-replay/water-recourse
```

## Original announcement and tropical wind replay

```bash
sha256sum -c research/continuous/BATCH4_SHA256SUMS
mkdir -p work/continuous-replay/{announcement-snapshot,wind-snapshot,announcements,winds}
tar xf research/continuous/announcements/inputs/nara_public_html_responses.tar.gz -C work/continuous-replay/announcement-snapshot
tar xf research/continuous/winds/inputs/noaa_public_wind_responses.tar.gz -C work/continuous-replay/wind-snapshot
python analysis/thesis_whitehouse_announcements.py work/continuous-replay/announcement-snapshot work/continuous-replay/announcements
python analysis/thesis_tropical_wind_verification.py work/continuous-replay/wind-snapshot work/continuous-replay/winds
```

## Full-page geography, documentary fields and independent storm replay

```bash
sha256sum -c research/continuous/BATCH5_SHA256SUMS
mkdir -p work/continuous-replay/{dds-complete,wind-extension-source,complete-geography,announcement-fields,wind-extension,geography-ablation}
tar xf research/continuous/complete_geography/inputs/openfema_all_36_verified_pages.tar.gz -C work/continuous-replay/dds-complete
tar xf research/continuous/wind_extension/inputs/noaa_additional_wind_responses.tar.gz -C work/continuous-replay/wind-extension-source
python analysis/thesis_full_page_geography.py work/continuous-replay/dds-complete work/continuous-replay/complete-geography
python analysis/thesis_announcement_fields.py work/continuous-replay/announcement-snapshot work/continuous-replay/announcement-fields
python analysis/thesis_tropical_wind_verification.py work/continuous-replay/wind-extension-source work/continuous-replay/wind-extension --independent-extension
python analysis/thesis_full_geography_ablation.py research/continuous/complete_geography/hazard_replay/cohort_features.csv work/continuous-replay/geography-ablation
```

To rebuild observed hazard features, pass the original complete source cache and the earlier extracted hazard artifacts to `thesis_full_page_geography.py` with `--artifacts`. The cache includes the originally registered NOAA, Census, USGS and vegetation inputs. The model replay above uses the retained feature matrix and the archive retains source-level linkage outputs. The original hazard assembly code and old results remain in place. [Original registered inputs](../inputs/RAW_SOURCE_MANIFEST.csv).
