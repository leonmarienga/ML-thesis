# Reproduce the follow-up

Use Python 3.12 and the pinned [requirements](../requirements.txt). Run from the repository root. Historical benchmarks, the FY2025 roster and the preceding thesis notebook remain immutable.

## Replay model comparisons from retained evidence

```bash
sha256sum -c research/inputs/SHA256SUMS
sha256sum -c research/follow_up/inputs/SHA256SUMS
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p work/thesis_followup_replay
python analysis/thesis_decision_time_followup.py work/thesis_followup_replay/timing \
  --mission-archive research/follow_up/inputs/mission_asof.zip
python analysis/thesis_physical_model_followup.py \
  --weather research/follow_up/results/weather \
  --spatial research/follow_up/results/spatial work/thesis_followup_replay/physical
python analysis/thesis_rare_event_followup.py \
  --physical work/thesis_followup_replay/physical/physical_model_inputs.csv \
  work/thesis_followup_replay/rare_event
```

The exposure input uses the committed, adjudicated coverage snapshot. No network access to data sources is needed for model replays. The matched 376-case prevalence, scores and model support should match within platform-level floating-point tolerance. The timing analysis reads the original workbook and retained archives and does not query FEMA.

## Rebuild physical evidence

1. Use the original [technical guide](../TECHNICAL_GUIDE.md) and locked source releases to prepare the TIGER county polygons, MTBS inputs, NHC best tracks, and archived advisory names. The corresponding raw files appear in the original raw-source manifest.
2. Obtain the public IBTrACS WP and SP archive revisions by the URLs in [IBTrACS source manifest](inputs/ibtracs_sources.csv) and verify SHA-256. Generate adjacent acquisition metadata with the existing source acquisition helper. The retained subset contains matched identifiers, coordinates and source fixes.
3. Rebuild the derived physical inputs with the commands below. Full spatial re-retrieval is susceptible to current Census ArcGIS delivery and upstream corrections. Compare file identity, retrieval status and bytes against the manifests and record any differences as a new evidence version. The retained perimeter responses, study geometries, block-group intersection inputs and derived populations support reproduction without full raw census exports.

```bash
python analysis/thesis_followup_precipitation.py \
  research/results/hazards/cohort_features.csv research/results/hazards/county_keys.csv \
  work/thesis_followup_raw/climate work/thesis_followup_replay/weather
python analysis/thesis_followup_spatial.py \
  --raw work/raw --ibtracs work/thesis_followup_raw/ibtracs \
  --advisories work/artifacts/nhc/storm_matching_and_timestamp_audit.csv \
  --event-names research/follow_up/event_name_evidence.csv \
  --cache work/thesis_followup_raw/spatial --output work/thesis_followup_replay/spatial
```

To replay the FY2025 check offline, extract the retained metadata-only export to the cache directory first. The guard uses a fixed seven-field metadata whitelist. It has no outcome retrieval code.

```bash
mkdir -p work/thesis_followup_raw/fy2025_metadata
tar -xzf research/follow_up/inputs/fy2025_metadata_only.tar.gz -C work/thesis_followup_raw/fy2025_metadata
python analysis/thesis_fy2025_metadata_guard.py \
  research/results/follow_up/fy2025_target_free_roster.csv \
  work/thesis_followup_raw/fy2025_metadata work/thesis_followup_replay/reservation
```
