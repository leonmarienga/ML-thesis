# Recoverable research checkpoint — 2026-11-17

Branch: `experiment/thesis-continuation-20261117`.
Parent research commit: `041afba8a10da9c2a962222e0a2644284a84fb7b`.
First continuation experiment checkpoint, pushed and remotely verified:
`5a1907e5f46b629b6c30bee148fd3412957bdacf`.
The commit containing this status file is the final continuation release;
resolve its exact SHA with `git rev-parse HEAD`. The final chat reply links
the remotely verified release. GitHub's branch/commit ZIP includes the complete
tracked checkpoint without requiring this environment's unavailable output mount.
Main, the original workbook and all 342 original tracked sources remain unchanged.

## Last completed work and findings

Phase 50 seasonal/interannual placebo audit completed: 289/289 planned reads,
67 fixed windows, all same-edition JRC v1.5. At 30% common observability the
primary seasonal comparison covers 44 anchors / 27 cases, excess gain
0.053073 km² per 100 observed km². Adding four MN sites raises the excess to
0.687342; no stable disaster-wide extent interpretation is supported.
All 65 sites with usable historical log observations match observability and
rounded gain. Two Alaska anchors lack a successful pre-correction log entry.
The original archives were subsequently recovered and byte-verified. All 187
reconstructed geometries match exactly, all 63 Phase48 anchors agree, and all
67 Phase50 integer gains and event observability fractions match the originals.
The Alaska-only merge verifies 561 monthly rows and 374 comparisons.

DR4816's original workbook August date disagrees with official FEMA's June date.
Only the new observational audit was corrected; historical models/data were not
rewritten. The superseded initial summary is retained with a rejection reason.

Actual multi-hazard source retrieval and support inventory completed as a bounded
pilot. NOAA2024, Atlantic HURDAT2 and eight USGS earthquake responses were saved.
MTBS source pages succeeded; advertised map service failed. No unverified hazard
feature was promoted to an accepted funding model.

Eighteen independently sensed NASA MODIS rendered maps for six MN sites were
retrieved. They show some flood-class colors and extensive insufficient-data
areas. Raw QA and surveyed truth remain absent, so no flood accuracy or certified
exposure was calculated.

Synthetic allocation completed: 300 episodes, eight arrivals, three budget ratios,
7,200 policy evaluations and 57,600 allocation rows. All cumulative cash and
oracle-regret invariants passed. No historical funding was used as need.

The frozen 912-case prediction archive was also recovered and recomputed:
830/912 routing, MAE $5,014,819.011614494, R² 0.8867765710842239, balanced accuracy
0.895451 and macro F1 0.878561. Per-case outputs, confusion matrices, hazard/year/
band errors and descriptive year-block difference uncertainty are saved.

Six scientific-mask/date tests, complete offline pixel-table reproduction,
original-source checksums and the full executed notebook passed. Numeric
reproduction tolerance is 1e-9 for geodesic round-trip arithmetic. No newly
independent predictive improvement or operational validation is established.

## Durable artifacts

- `research/phase50/`: source reads, fixed roster/geometries, cropped real pixels,
  site/case/aggregate comparisons, log replay, correction and source snapshots.
- `research/source_audit/`: hazard support, actual source records, retrieval logs,
  original NOAA/HURDAT raw archives and earthquake responses.
- `research/flood_validation/`: 18 dated NASA renders, layer metadata, source logs,
  official legend and the original 2023 NASA meeting presentation.
- `research/allocation/`: clearly synthetic scenarios, every allocation, policy
  metrics, paired Monte Carlo differences and generator/version description.
- `research/recovered_artifacts/`: original accepted prediction and Phase48/49/49b
  ZIPs with matching GitHub digests, extracted originals and corrected merges.
- `research/frozen_prediction_audit/`: all recomputed frozen metrics and exact
  integer/geographic replication checks.
- `research/REPRODUCIBLE_RESEARCH.ipynb`, figures, thesis draft, technical guide,
  prospective protocol, conclusion and root CSV registries.

## Genuine blockers and outstanding objectives

CLI Actions ZIP downloads still return HTTP403. The GitHub connector's
download_workflow_artifact operation successfully recovered the exact archives;
that blocker is resolved. Do not repeat the failing CLI route. Separate trained
model weights were not verified, but original training procedures and all three
frozen prediction versions are preserved. The chat's designated /codex output
directory is not writable; repository files and branch ZIP are the delivery route.

Full certified multi-hazard severity/exposure assembly, raw historical flood QA,
genuinely untouched comparable outcomes, reliable rare-event intervals and
real-need allocation validation remain incomplete. No claim is made that the
physical datasets do not exist. Do not manufacture successful completion of these
gates or select another complex model from the examined 59 flood cases.

## Exact next justified action and commands

Advance raw MODIS historical product/QA acquisition and independently verified event linkage;
rendered colors cannot substitute for truth. These are data-validation gates,
not authorization to refit the accepted framework.

Do not repeat the completed research to resume it. Verify the checkpoint first:

```bash
git switch experiment/thesis-continuation-20261117
git rev-parse HEAD
work/venv/bin/python analysis/validate_continuation_checkpoint.py
work/venv/bin/python analysis/continuation_archive_recovery_audit.py
```

For a deliberate offline reproduction:

```bash
work/venv/bin/python analysis/phase50_temporal_placebo.py --offline-inputs research/phase50 --out work/reproduced-phase50
```

The full thesis research goal remains incomplete. FINAL_CONCLUSION.md supplies the
bounded scientific conclusion and the exact limits of every completed result.
