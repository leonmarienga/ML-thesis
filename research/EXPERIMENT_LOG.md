# Experiment log (append only)

## Continuation setup

Question: can the interrupted Phase50 be resumed with preserved source lineage?
Input version: git 041afba8a10da9c2a962222e0a2644284a84fb7b, original master workbook.
No model training or seeds yet. Existing checkout was clean on branch work;
fetched research head and created isolated child branch. Read handoff completely.
Artifact ZIP retrieval returned HTTP403; decoded log recovery succeeded through
GitHub connector. Negative finding: original artifact CSV byte checksums cannot
yet be independently verified. Next action: reconstruct only the fixed logged
site keys, verify raster observations against historical log, then seasonal audit.

## Phase50: same-edition temporal placebos — executed

Question: does sampled water gain persist beyond corresponding historical
monthly/annual transitions? Hypothesis: a broad stable event-specific signal
would be robust to seasonal controls, observability thresholds and site expansion.
Primary metric and bounded variants were fixed in STATUS.md and the script
before the full execution. Inputs: original log site keys, fixed Census2023
geometry, actual JRC v1.5 2022–2024 pixels; source SHA/URLs in registry.
Model version: none; no funding training, seed or funding-driven selection.
Code checkpoint: initial execution5a1907e5f46b629b6c30bee148fd3412957bdacf;
later offline-replay improvements are in this checkpoint's source tree.

Protocol:63 anchor keys,4 extra DR4722 keys as separate sensitivity; seasonal
four-date common-pixel and2024 interannual three-date common-pixel masks;
30/60/80% thresholds; pair-specific denominators as sensitivity. Unobserved
class0 is never dry. All289 planned reads succeeded. Primary seasonal44 sites,
27 cases: excess gain0.053073km² per100 observed km². Including MN extras:
46 sites,27 cases,0.687342. Primary interannual17 sites,11 cases:0.086369.
Interpretation: strong dependence on geographic sampling; no representative
disaster extent, flood-truth metric or exposure estimate. Hypothesis of a stable
broad severity interpretation is unsupported. Next justified action: raw
independent-product QA and validated event linkage, not funding-test tuning.

Failure/correction: initial workbook-based DR4816 start was August22 rather
than official June22. Historical-log replay failed for that one site. Retrieved
all36 current FEMA event metadata records, compared dates, found this single
discrepancy, reran the affected months and the full aggregate audit. Original
workbook and benchmark unchanged. All65 comparable log entries now agree;
2 Alaska anchors lack successful pre-correction log data. Superseded initial
summary is retained and labelled under phase50. The original ZIP byte lineage
and full integer comparator remain blocked byHTTP403.

## Multi-hazard source and support audit — executed, bounded pilot

Question: which represented hazards have enough records and usable physical
fields? Hypothesis: source availability differs by hazard and cannot be inferred
from a generic disaster label. Inputs:971-row original workbook, official
NOAA2024 gzip, official Atlantic HURDAT2 file, eight USGS regional/time queries,
MTBS source documentation. Versions and actual raw-response SHA256 in registry.
Model/version/seed: none; source audit only. Protocol: count all hazard/year/band
support; one-year NOAA availability pilot; eight fixed earthquake queries with
regional boxes and48h windows; no automatic nearest-event acceptance.

Results:20 hazard labels,912 non-Biological cases, no drought cases. Actual
wind/hail/tornado fields retrieved; numeric snow/ice fields absent in this
archive. Eight earthquake queries succeeded but remain attribution candidates.
NHC best-track source retrieved; final rather than contemporaneous estimates.
MTBS catalog access succeeded, map service503; direct candidate ZIP route404.
No wildfire measurements fabricated, no standalone8-case earthquake model fit.
Interpretation: valid source feasibility evidence, incomplete multi-hazard
feature assembly. Next: independently certify event geography/dates and acquire
fire/other-hazard physical records before accepting predictors.

## Independent MODIS rendered-map audit — executed

Question: can independently dated coarse flood-product maps be inspected for
the six exact Minnesota windows? Hypothesis: independently sensed classification
can offer context while exposing data gaps. Input: official NASA GIBS advertised
MODIS_Combined_Flood_3-Day layer, fixed polygons and NWS seasonal chronology.
Dates fixed to2023-04-12,2023-04-26 and2023-05-10 before render inspection.
Model: external satellite classification product, no fitted thesis model or seed.
Actual source URLs, returned map checksums, layer metadata and official legend
saved. Results:18/18 renders decoded. Flood colors occur inside two added
Houston County polygons; inland views are often insufficient-data gray.
Interpretation: contextual corroboration and missingness, not surveyed labels.
No precision/recall/IoU/FPR or area inferred from enlarged display pixels.
Next: authorized raw historical MODIS class/QA access and an independent-label
protocol. Do not repeat endless small-patch searches or claim all archives absent.

## Allocation under uncertainty — synthetic experiment executed

Question: how do budget feasibility, total service and worst-served-disaster
service trade off without treating historical funding as need? Hypothesis:
efficiency, vulnerability priority and service equity can conflict. Existing
reinforcement_learning.ipynb reviewed: historical log-funding error drives its
reward, so it does not independently validate humanitarian utility.

Inputs/version: code-defined synthetic generator and separately saved scenario
table; no FEMA targets. Seed20261117. Protocol fixed:300 episodes,8 arrivals,
3 budget ratios,5 simultaneous and3 sequential policies; uncertain second cash
tranche; same random scenarios for every policy; no policy tuning. Need and
uncertainty equations are in source and notebook. Metrics: unmet synthetic need,
fulfilled fraction, worst-disaster service, vulnerable-group gap, wasted units,
cash utilization and same-cash-constraint oracle regret. Monte Carlo paired
difference intervals describe generator sampling error only.

Results:7,200 policy evaluations,57,600 event allocations. Every cash and regret
invariant passed. At ratio0.7, sequential greedy: fulfilled0.5785, minimum service
0.0604; reserve:0.5583 and0.2717. Interpretation: scenario trade-off, not proven
real-world welfare improvement. Negative result: no single policy universally
dominates both objectives. Next: need labels and human-governed utility/fairness
definitions before any real allocation application.

## Reproducibility and documentation validation — executed

All6 scientific-mask/date tests passed. Complete Phase50 tables reproduced
offline from saved pixels within1e-9 numeric tolerance; integer counts and
log-match flags agree. The executed23-cell notebook includes12 code cells and
reexecutes the full synthetic experiment as well as offline pixel analysis.
Original342 source checksums verified. Historical metric evidence, new
observations and synthetic outcomes are separated in FINAL_METRICS.csv.
No new model improvement, prospective validation or complete-thesis status is
claimed. The first research checkpoint was pushed and remote SHA verified.

## Archive-recovery correction and exact frozen evaluation — executed

The earlier artifact-access blocker is superseded: GitHub's connector
download_workflow_artifact operation succeeds even though CLI redirects fail.
Recovered original archives for runs 41397678781, 41578898108, 41581240598 and
41581789674. Every ZIP SHA256 matches GitHub's recorded digest. Signed transient
download references were used only in ignored work files, never committed.

Question: do recovered original artifacts verify the reconstructed geography,
pixels and reported frozen funding metrics? No new model fit or parameter
selection. Inputs: original accepted 912 CSV and original Phase48/49/49b files.
Protocol: replace only Alaska DR4730 rows, assert 187 sites, 374 pairs and 561
monthly rows, compare all original polygons and primary anchors, then compare
all 67 Phase50 integer gains and observation fractions. Recompute classification
and monetary scores for all three immutable prediction columns on the same
912 records, with hazard/year/band/zero/rare breakdowns.

Results: all 187 geometries are exactly equal (maximum Hausdorff distance 0),
all 63 original anchors match, all 67 integer gains and event observation
fractions match, including Alaska. The accepted predictions exactly reproduce
830/912, MAE 5014819.011614494, RMSE 64653747.252143376 and R² 0.8867765710842239.
Balanced accuracy 0.8954508207; macro precision 0.8672919107; macro F1 0.8785607328.
Full predictions, confusion matrix and metric/influence tables preserved.

Descriptive fixed-prediction fiscal-year block bootstrap (seed 20261117,
2,000 replicates) gives accepted-minus-original MAE -$116,540.53, percentile
95% interval [-$301,156.44, +$28,174.46]. It crosses zero; fiscal-year clusters
are not independently verified physical-event clusters. This is historical
uncertainty, not a new untouched comparison, operational validity or evidence
that the continuation improved the frozen model. Previous reports of archive
unavailability describe the earlier failed route and are not the final status.

Next justified action: raw flood-product QA and verified multi-hazard event
linkage, followed by the prospectively frozen evaluation protocol. Do not rerun
completed source/metric experiments merely because a new session begins.
