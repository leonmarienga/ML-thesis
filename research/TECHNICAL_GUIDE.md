# Reproduce and continue the research

## Checkout and installation

Use the research child branch; do not merge or modify main.

```bash
git clone --branch experiment/thesis-continuation-20261117 https://github.com/leonmarienga/ML-thesis.git
cd ML-thesis
python3.12 -m venv work/venv
work/venv/bin/pip install -r requirements-research.txt
```

The continuation was tested with Python3.12.14. Library pins reproduce the
tested environment; they are separate from the original model environment.
The exact original training artifacts may require their historical package
versions. Do not refit or overwrite frozen models with new library versions.

## Recompute the completed observational experiment offline

```bash
work/venv/bin/python -m unittest discover -s analysis -p test_phase50_temporal_placebo.py
work/venv/bin/python analysis/phase50_temporal_placebo.py --offline-inputs research/phase50 --out work/reproduced-phase50
```

This reads saved real cropped pixels and fixed polygons. It does not contact
external services. Compare site/case/aggregate tables with research/phase50;
allow1e-9 absolute/relative tolerance for geodesic floating-point values after
CSV coordinate serialization. Source-grid masks, pixel counts and the 65
successful historical-log matches should agree. The two Alaska anchors have
no usable comparator in the recovered pre-correction log. The frozen original
342-file SHA256 manifest is research/frozen_source_checksums.json.

Source retrieval mode is separate and uses current service responses:

```bash
work/venv/bin/python analysis/phase50_temporal_placebo.py --out work/fresh-phase50
work/venv/bin/python analysis/continuation_source_audit.py
work/venv/bin/python analysis/continuation_nasa_map_audit.py
```

Fresh source responses may be revised. Compare source checksums and source
edition metadata rather than silently treating new data as the original snapshot.
The source registry records exact downloaded file URLs; retrieval scripts
preserve version filenames and source response checksums. Complete TIFF hashes
are not available for range-read tiles: registry hashes refer to saved cropped
NPZ files. Current FEMA metadata and its original record identifiers are saved
under research/phase50/source_snapshots.

Exact source-response replay is available with:

```bash
work/venv/bin/python analysis/replay_source_manifest.py --manifest research/source_audit/source_retrieval_log.csv --out work/exact-source-replay
```

This requests the recorded URLs rather than selecting newer archive versions.
Current dynamic API responses or HTML pages may differ; any checksum mismatch
is reported as a failed exact reproduction. Original NOAA/HURDAT raw downloads
and earthquake responses are retained under source_snapshots.

## Replay exact accepted prediction reconstruction when archives become accessible

The source implementation is unchanged. The previous run 41397678781 succeeded,
and its aggregate log has been recovered. This environment receivedHTTP403 from
redirected artifact storage, so the commands below are a recovery path, not a
claim of present successful per-row recovery.

```bash
mkdir -p audit_inputs/nonbio_final_50_200_overlay/{baseline,low0,low1,mid}
gh run download 34796746603 -R leonmarienga/ML-thesis -n nonbio-final-830-accepted-amount-stack -D audit_inputs/nonbio_final_50_200_overlay/baseline
gh run download 38153347912 -R leonmarienga/ML-thesis -n nonbio-0-100k-dedicated-ml-specialist -D audit_inputs/nonbio_final_50_200_overlay/low0
gh run download 38152554081 -R leonmarienga/ML-thesis -n nonbio-100k-1m-dedicated-ml-specialist -D audit_inputs/nonbio_final_50_200_overlay/low1
gh run download 38162966192 -R leonmarienga/ML-thesis -n nonbio-50-200m-routed-acceptance-audit -D audit_inputs/nonbio_final_50_200_overlay/mid
work/venv/bin/python analysis/reconstruct_accepted_912_exact.py
```

The last command must assert912 rows,830 routing successes, MAE 5014819.011614494
and R² 0.8867765710842239. Preserve actual archive checksums. Do not replace
unavailable artifacts with guessed model outputs.

For the original Phase 49 archive use run 41581240598, artifact
phase49-spatial-representativeness-audit; Alaska correction is run 41581789674,
artifact phase49b-alaska-raster-window-correction. Replace only DR 4730 rows and
assert187 unique sites,374 comparisons,561 monthly records and 36 disasters.
Compare the original integer gain counts to the new fixed-site reads.

## Allocation and human-review demonstration

```bash
work/venv/bin/python analysis/continuation_allocation_simulation.py
work/venv/bin/python analysis/research_decision_support_demo.py
```

The former generates clearly labelled synthetic needs, vulnerability, releases,
allocations and policy metrics. Seed20261117 and fixed policies reproduce the
7,200 evaluations. No historical funding is used as need. The latter displays
actual satellite evidence and a separately labelled synthetic allocation episode;
it abstains from unsupported case funding/interval predictions. Optionally pass
the recovered exact accepted CSV with --accepted-predictions; its schema and
aggregate benchmark must verify before use.

## Notebook, documentation and figures

Open research/REPRODUCIBLE_RESEARCH.ipynb with the work/venv Python kernel.
It is saved with executed cells and outputs. It replays the offline raster audit,
checks source integrity, reexecutes the simulation and explains the mathematics,
source quality, historical evidence and remaining limitations.

analysis/build_continuation_deliverables.py rebuilds registries, figures and the
unexecuted notebook from committed result tables. It must be followed by notebook
execution to regenerate verified cell outputs. Figures use actual tables; NASA
renders retain the official legend and are not treated as ground truth.

research/THESIS_METHODS_RESULTS_DISCUSSION.md is a draft suitable for revision
into the thesis; it clearly labels uncompleted work. All prediction/model records
in MODEL_REGISTRY.csv retain their validation status. Read STATUS.md before
resuming. Avoid repeating completed experiments unless checking reproduction
or resolving a documented new discrepancy.
