# Replay the rare-disaster augmentation study

Run from the repository root in a Python 3.12 environment with the pinned `research/requirements.txt` dependencies. The benchmark uses the retained FY2024-and-earlier inputs and runs without network access. Four workers are the upper parallelism used for this study. Replay fits fresh models within every fold and compares the tabular evidence to the archived results.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
sha256sum -c research/augmentation/inputs/SHA256SUMS
git diff --exit-code 45eeaaf HEAD -- research/results research/inputs research/follow_up
mkdir -p /tmp/thesis-augmentation
python analysis/thesis_augmentation_study.py /tmp/thesis-augmentation --workers 4
python analysis/thesis_augmentation_verify.py /tmp/thesis-augmentation --reference research/augmentation/results
python analysis/thesis_physics_impact_pilot.py --retained-inventory-summary research/augmentation/inputs/physics_inventory_summary.json /tmp/thesis-augmentation
cmp /tmp/thesis-augmentation/physics_impact_pilot.csv research/augmentation/results/physics_impact_pilot.csv
jupyter execute research/THESIS_REPRODUCTION.ipynb
```

The physics replay uses the retained aggregate of the byte-identified USACE structure inventory. To independently compute the aggregate from the same payload, obtain the study-vintage county GeoJSON and pass its path with `--inventory-api-file`. The public API is mutable, so a later download may have a different digest. The original payload digest and size are recorded in [the NSI source record](inputs/nsi_22063.geojson.json). The `physics_impact_pilot.csv` table is independent of the supervised benchmark.

`membership.csv.gz` contains real, dated fold membership. `registry.csv.gz` registers each fold, method, ratio, stratum, seed, donor set, generation attempt and outcome. `synthetic.csv.gz` contains synthetic feature rows, method branch, lineage and screening flags. `generator_distributions.csv` summarizes uniqueness, near-duplicates and concentration. `predictions.csv.gz`, `inner_predictions.csv.gz` and `sensitivity.csv.gz` contain real-case predictions and outcomes. `selections.csv` records the earlier-window dose decisions. `tuning_grid.csv` shows the earlier-window evidence for those choices. `metrics.csv`, `per_band.csv`, `reliability.csv`, `intervals.csv` and `paired_episode_bootstrap.csv` hold matched comparisons. `sensitivity_summary.csv` holds the episode-omission diagnostics.
