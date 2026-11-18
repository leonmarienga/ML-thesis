# Reproduce the research

Run commands at the repository root on the continuation branch. Python 3.12 and the pinned packages in [requirements.txt](requirements.txt) reproduce the documented runs. The notebook can read and verify the shipped results without querying a remote source. The full enrichment rebuild requires public network access and disk space.

## 1. Prepare the environment

```bash
python3.12 -m venv .thesis-env
source .thesis-env/bin/activate
python -m pip install -r research/requirements.txt
mkdir -p work/{artifacts,phase50,raw,track_a,follow_up,forward,uncertainty,allocation,phase51}
for archive in research/inputs/actions/*.zip; do
  name=$(basename "$archive" .zip)
  mkdir -p "work/artifacts/$name"
  unzip -oq "$archive" -d "work/artifacts/$name"
done
sha256sum -c research/inputs/SHA256SUMS
```

The zip files are original Actions archives. [Archive IDs, names and hashes](inputs/ACTION_ARCHIVE_MANIFEST.csv).

## 2. Reconstruct Track A

The historical overlay script uses fixed input paths. In a clean replay checkout, extract its four archives to these directories.

```bash
mkdir -p audit_inputs/nonbio_final_50_200_overlay/{baseline,low0,low1,mid}
for pair in track_a_base:baseline track_a_low0:low0 track_a_low1:low1 track_a_mid:mid; do
  unzip -oq "research/inputs/actions/${pair%:*}.zip" -d "audit_inputs/nonbio_final_50_200_overlay/${pair#*:}"
done
python analysis/reconstruct_accepted_912_exact.py
tar -xzf research/inputs/snapshots/openfema_fy2025_target_free.tar.gz -C work/follow_up
python analysis/thesis_audit.py \
  --track-a work/artifacts/track_a/all_912_immutable_overlay.csv \
  --track-a-rebuilt audit_outputs/nonbio_final_50_200_overlay/all_912_immutable_overlay.csv \
  --output work/track_a --target-free work/follow_up
```

Check 912 records, 830 correct routes, the half-open band rule, $5,014,819.01 MAE and R² 0.8867765711. The exact archive hash is independent of the tolerance used for results reproduced with a different CSV serializer. The audit reserves an FY2025 roster with metadata only.

## 3. Replay Phase 50 and the independent cross-check

```bash
tar -xzf research/inputs/snapshots/phase50_sampled_pixels.tar.gz -C work/phase50
python analysis/phase50_temporal_placebo.py --artifacts work/artifacts --output work/phase50
tar -xzf research/inputs/snapshots/dr4722_gauge_responses.tar.gz -C work/phase51
python analysis/phase51_flood_context.py work/phase50 work/phase51
python analysis/phase51_modis_crosscheck.py \
  work/artifacts/phase49/phase49_county_windows_2023_2024.geojson work/phase51 --phase50-pixels work/phase50/pixels
```

The first command replays the original pixel windows. To retrieve source windows again, point the script at a new output directory. The source URLs, ETags and pixel hashes support point-by-point comparisons. The MODIS step uses GIBS public raster access. An Earthdata account is required to use the LAADS raw-product route.

## 4. Rebuild hazard features and models

```bash
tar -xzf research/inputs/snapshots/openfema_fy2010_2024_pages.tar.gz -C work/raw
python analysis/thesis_retrieve_sources.py work/raw --manifest research/inputs/RAW_SOURCE_MANIFEST.csv
python analysis/thesis_hazard_features.py --raw work/raw --artifacts work/artifacts --output work/hazards
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python analysis/thesis_forward_models.py work/hazards/cohort_features.csv work/forward --workers 4
python analysis/thesis_uncertainty.py work/forward/forward_predictions.csv work/uncertainty
python analysis/thesis_measurement_sensitivity.py work/hazards/cohort_features.csv work/uncertainty
python analysis/thesis_allocation.py work/allocation
```

The manifest verifies exact raw-release hashes. Revised data or retired releases require a new research source version. The committed feature table is sufficient to rerun model fitting, comparisons and interval calibration independently of those data services.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python analysis/thesis_forward_models.py research/results/hazards/cohort_features.csv work/forward --workers 4
python analysis/thesis_uncertainty.py work/forward/forward_predictions.csv work/uncertainty
python analysis/thesis_demo.py 4722 --scenario-event 1
jupyter execute research/THESIS_REPRODUCTION.ipynb
```

The demonstration reads retrospective predictions and a separate synthetic scenario. The historical display flags each missing evidence family and shows the six-band probability distribution. The simulation compares human-selected priorities under the same need and budgets. No allocation consumes the historical dollar prediction.

## 5. Read the study

[Thesis methods, results and discussion](THESIS_METHODS_RESULTS_DISCUSSION.md) explains the estimands, decision-time assumptions, matching, and policy loss. [Final metrics](FINAL_METRICS.csv) contains the core signed-off measurements. [Research status](STATUS.md) records the next confirmatory experiment and its dependencies.
