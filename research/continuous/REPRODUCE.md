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

## Earlier announcement vintages and short-horizon disturbance duration

Use the archived bytes and their acquisition manifests to replay both Batch 6 workstreams without network access.

```bash
sha256sum -c research/continuous/BATCH6_SHA256SUMS
mkdir -p work/continuous-replay/{older-source,doe-source,older-announcements,disturbance-duration}
tar xf research/continuous/older_announcements/inputs/older_announcement_responses.tar.gz -C work/continuous-replay/older-source
tar xf research/continuous/disturbance_duration/inputs/doe_annual_summaries.tar.gz -C work/continuous-replay/doe-source
python analysis/thesis_older_announcement_archive.py work/continuous-replay/older-source work/continuous-replay/older-announcements
python analysis/thesis_disturbance_duration.py work/continuous-replay/doe-source work/continuous-replay/disturbance-duration
```

The announcement replay includes the civil-day claims, independent timestamped archived bodies and origin-maturity bounds. The DOE replay preserves the complete-case screen, held report predictions and paired block bootstrap under the published protocol. The seven DOE files require both XLS and XLSX readers. [Batch 6 protocol](BATCH6_PROTOCOL.md).

## Expanded declaration text coverage and independent outage observations

```bash
sha256sum -c research/continuous/BATCH7_SHA256SUMS
mkdir -p work/continuous-replay/{older-extension,outage-source,documentary-extension,county-outages}
tar xf research/continuous/documentary_extension/inputs/extended_announcement_responses.tar.gz -C work/continuous-replay/older-extension
tar xf research/continuous/county_outages/inputs/eaglei_and_mesa_evidence.tar.gz -C work/continuous-replay/outage-source
python analysis/thesis_older_announcement_archive.py work/continuous-replay/older-extension work/continuous-replay/documentary-extension --extend-quarterly-sample
python analysis/thesis_documentary_support.py work/continuous-replay/documentary-extension/document_identity.csv work/continuous-replay/documentary-extension
python analysis/thesis_outage_observability.py work/continuous-replay/outage-source work/continuous-replay/county-outages
```

The archived outage inputs include the exact rows for the fixed 24 counties and the filtered utility CSV. To reconstruct the slice, obtain file 42547879 from the pinned author-release URL and invoke the outage script with `--national /path/to/eaglei_outages_2018.csv`. It reads the retained release metadata, verifies the 999,210,542-byte file against its published MD5 and records SHA-256 for the source and extracted slice. The extracted data are enough for the offline analysis. [Locked protocol](BATCH7_PROTOCOL.md), [author distribution](https://doi.org/10.6084/m9.figshare.24237376.v1).

## Precoverage obligation evidence and sustained county outage transfer

Requires `pdftotext` from Poppler for the original government PDFs. The outage analysis verifies the retained county rows from the four source years.

```bash
sha256sum -c research/continuous/BATCH8_SHA256SUMS
mkdir -p work/continuous-replay/{precoverage-source,annual-outage-source,precoverage-documents,sustained-outages,zero-sensitivity}
tar xf research/continuous/precoverage_documentary/inputs/public_document_archive.tar.gz -C work/continuous-replay/precoverage-source
tar xf research/continuous/sustained_outage/inputs/eaglei_annual_slice_archive.tar.gz -C work/continuous-replay/annual-outage-source
tar xf research/continuous/county_outages/inputs/eaglei_and_mesa_evidence.tar.gz -C work/continuous-replay/annual-outage-source eaglei_2018_24_counties.csv.gz national_slice_manifest.json
cp work/continuous-replay/annual-outage-source/national_slice_manifest.json work/continuous-replay/annual-outage-source/slice_manifest_2018.json
python analysis/thesis_precoverage_documentary.py work/continuous-replay/precoverage-source research/continuous/precoverage_documentary/inputs/curated_evidence.csv work/continuous-replay/precoverage-documents
python analysis/thesis_sustained_outage_transfer.py work/continuous-replay/annual-outage-source work/continuous-replay/sustained-outages
python analysis/thesis_sustained_outage_transfer.py work/continuous-replay/annual-outage-source work/continuous-replay/zero-sensitivity --exclude-zero-counts
```

To rebuild the 2020, 2022 and 2023 slices, acquire their original files from the public links in the archived Figshare v2 metadata and supply the directory to the sustained-outage script with `--national`. Each original file matches its published MD5 before the selected rows are used. [Locked design](BATCH8_PROTOCOL.md), [publisher release](https://doi.org/10.6084/m9.figshare.24237376.v2).

## Recipient-side audits and county wind geometry

```bash
sha256sum -c research/continuous/BATCH9_SHA256SUMS
mkdir -p work/continuous-replay/{component-source,county-wind-source,component-accounting,county-wind-exposure}
tar xf research/continuous/component_accounting/inputs/public_component_reports.tar.gz -C work/continuous-replay/component-source
tar xf research/continuous/county_wind_exposure/inputs/census_2017_county_points.tar.gz -C work/continuous-replay/county-wind-source
tar xf research/continuous/winds/inputs/noaa_public_wind_responses.tar.gz -C work/continuous-replay/county-wind-source ./aal132020.dat.gz ./aal132020.dat.gz.json ./aal092022.dat.gz ./aal092022.dat.gz.json
tar xf research/continuous/wind_extension/inputs/noaa_additional_wind_responses.tar.gz -C work/continuous-replay/county-wind-source ./aal142018.dat.gz ./aal142018.dat.gz.json ./aal102023.dat.gz ./aal102023.dat.gz.json
python analysis/thesis_component_accounting.py work/continuous-replay/component-source research/continuous/component_accounting/inputs/curated_records.csv work/continuous-replay/component-accounting
python analysis/thesis_county_wind_exposure.py work/continuous-replay/county-wind-source work/continuous-replay/county-wind-exposure
```

The scanner uses Poppler plus hash-verified archived OCR sidecars. To recreate the sidecars, render all 12 NASA report pages at 200 dpi and the first eight CBP report pages at 140 dpi with `pdftoppm -gray -png`. Apply Tesseract at its default page segmentation and concatenate the per-page text, in order, with form feeds. The retained sidecar hashes identify the original OCR runs. [Batch 9 protocol](BATCH9_PROTOCOL.md).

## Expanded hurricane panel, exact legacy ledgers, and independently measured need

```bash
sha256sum -c research/continuous/BATCH10_SHA256SUMS
mkdir -p work/continuous-replay/batch10/{legacy-source,wind-source,storm-slices,survey-source,legacy,panel,primary,zero,needs}
work=work/continuous-replay/batch10
base=research/continuous
panel="$base/storm_outage_panel/inputs"
tar xf "$base/legacy_ledger_gate/inputs/public_legacy_ledger_routes.tar.gz" -C "$work/legacy-source"
tar xf "$base/humanitarian_source/inputs/public_survey_and_shelter_sources.tar.gz" -C "$work/survey-source"
for input in "$panel"/forecast_and_county_sources.tar.gz "$panel"/nhc_reports_*.tar.gz; do
    tar xf "$input" -C "$work/wind-source"
done
tar xf "$panel/verified_eaglei_event_slices.tar.gz" -C "$work/storm-slices"
python analysis/thesis_public_ledger_gate.py "$work/legacy-source" "$work/legacy"
python analysis/thesis_humanitarian_source_gate.py "$work/survey-source" "$base/humanitarian_source/inputs/casper_extract.csv" "$work/needs"
python analysis/thesis_storm_outage_panel.py panel "$work/wind-source" "$work/panel" --tcr-markers "$panel/tcr_markers.json"
python analysis/thesis_storm_outage_panel.py evaluate "$work/wind-source" "$work/primary" --slices "$work/storm-slices" --panel "$panel/storm_county_panel.csv"
python analysis/thesis_storm_outage_panel.py evaluate "$work/wind-source" "$work/zero" --slices "$work/storm-slices" --panel "$panel/storm_county_panel.csv" --exclude-zeros
```

To regenerate the exact outage event slices, place the four previously verified EAGLE-I national CSVs in one directory. Run the panel script with `extract`, the locked panel, an output directory, `--slices`, `--national` and `--file-card "$panel/source_cards.json"`. The extraction revalidates the complete public annual file size, MD5 and SHA-256 for each year. The offline replay uses only the verified subsets. [Locked design](BATCH10_PROTOCOL.md).

## Batch 11 coverage-aware funding and sequential services

Run at the repository root with the pinned research requirements and Poppler. Compare the CSV tables with the retained results. The funding loader reads only the frozen historical cohort, the retained coverage ablation and the stored signed-outcome archive. The services script verifies the reported medication, utility access, food and water values directly in the archived survey table before simulating.

```bash
sha256sum -c research/continuous/BATCH11_SHA256SUMS
mkdir -p work/batch11/{coverage_specialists,service_scenarios,survey-source}
tar xf research/continuous/humanitarian_source/inputs/public_survey_and_shelter_sources.tar.gz -C work/batch11/survey-source ./casper_usvi.pdf
python analysis/thesis_coverage_specialists.py work/batch11/coverage_specialists
python analysis/thesis_service_scenarios.py work/batch11/survey-source/casper_usvi.pdf work/batch11/service_scenarios
```

The independent humanitarian crosswalk preserves manually verified published estimates with source locations. The official electricity and tide pages are retained for review. The scripts do not use the crosswalk to select or score a policy.

## Batch 12 coastal tide and service floors

```bash
sha256sum -c research/continuous/BATCH12_SHA256SUMS
base=research/continuous/tidal_residual
mkdir -p work/batch12/tide-raw
tar xf "$base/inputs/noaa_hourly_stage_archive.tar.gz" -C work/batch12/tide-raw
python analysis/thesis_tidal_residual.py "$base/inputs/panel.csv" work/batch12/tide-raw work/batch12/tide
python analysis/thesis_tidal_residual.py "$base/inputs/panel.csv" work/batch12/tide-raw work/batch12/unflagged --unflagged-only
python analysis/thesis_service_floor.py work/batch11/survey-source/casper_usvi.pdf work/batch12/service_floors
```

The normal replay reads retained files. The bounded `--acquire` option records new response versions and timestamps for the fixed station panel.
