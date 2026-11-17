# Bounded scientific conclusion and completion status

The accepted historical Track A benchmark is preserved. This continuation has
**not established a fully validated operational multi-hazard allocation model**
or a validated improvement over Track A. It has completed the interrupted Phase 50
temporal placebo analysis using real satellite pixels, exposed a material metadata
error and strong spatial sensitivity, and implemented reproducible evidence and
allocation audits. Unsupported model replacement would overstate the evidence.

## Results supported by executed work

- The workbook contains 971 unique disaster records,912 non-Biological records,
  and 20 incident-type labels. The largest groups are430 severe storms,198 floods,
  108 hurricanes and 59 Biological records. There are33 fires,27 tornadoes and 8
  earthquakes. There are no Drought records. One negative net-funding target is
  retained in the workbook; the original band function clamps it to zero.
- The exact original 912-case prediction archive was recovered through the
  GitHub connector and its ZIP checksum matches GitHub's digest. Recomputing
  those fixed predictions gives 830/912 routing, MAE $5,014,819.011614494,
  RMSE $64,653,747.252143376 and R² 0.8867765710842239. Balanced accuracy is
  0.895451 and macro F1 is 0.878561. These remain historical LFYO results,
  not a new independent or prospective evaluation. Complete per-case predictions,
  confusion matrices, hazard/year/band errors and influence tables are saved.
- All342 files tracked at the starting research commit retain their original
  SHA256 checksums. Main has not been modified or merged.
- Phase 50 completed289 planned monthly raster reads successfully over 67 fixed
  windows:63 anchor keys and 4 additional Minnesota keys. The source edition is
  JRC v1.5 for2022–2024. The primary seasonal comparison uses44 sufficiently
  observable anchors covering27/36 eligible disasters; not all63 anchors qualify.
- At30% common observability, the seasonal primary event-minus-placebo gain is
  **0.053073 km² per100 observed km²**. Event and placebo gain rates are0.080857
  and 0.027784. Excluding Minnesota DR 4722 yields0.054403. Adding its four extra
  sites raises the pooled difference to**0.687342**, with 46 qualifying sites.
  This large dependence on site placement rejects a stable disaster-wide
  interpretation. The primary case-level median is zero; see the actual case
  table for the signed differences and the coverage sensitivity tables.
- The2024 interannual primary comparison covers17 sites and 11 disasters and
  yields0.086369 km² excess gain per100 common observed km². Neither placebo
  design identifies known flood-free truth, so neither estimates a false-positive
  rate or causal disaster effect.
- DR 4816 is dated2024-08-22 in the workbook but2024-06-22 in current official
  FEMA metadata. Using June reproduces its historical raster log. The original
  workbook was not changed. All65 sites with usable historical log observations
  now reproduce the logged observability and rounded gain; the two Alaska
  anchors have no comparable successful entry in that pre-correction log.
- Original Phase48, Phase49 and Phase49b ZIPs were subsequently recovered and
  byte-verified. All 187 reconstructed geometries match exactly; all 63 original
  anchor coordinates agree; and all 67 Phase50 integer water-gain counts and
  event observation fractions match the corrected original CSVs, including Alaska.
  The Alaska-only merge verifies 187 sites, 374 comparisons and 561 monthly rows.
- Eighteen NASA MODIS rendered maps were obtained for the exact six Minnesota
  windows on2023-04-12,2023-04-26 and 2023-05-10. Some added Houston County
  windows show the product’s flood color. Many inland views show insufficient
  data. These maps strengthen sensor-level context, but they are coarse rendered
  classifications without raw QA or independent surveyed labels. No flood IoU,
  precision, recall, certified footprint or footprint-derived exposure is claimed.
- An official2024 NOAA storm archive, an official Atlantic HURDAT2 archive and
  eight USGS earthquake-query responses were retrieved. They establish useful
  physical-field availability, not verified disaster attribution. MTBS catalog
  documentation was retrieved; its advertised map service returned503. No
  accepted model consumes the unverified candidate features.
- A separate synthetic allocation experiment executed300 episodes,8 arrivals,
  3 budget ratios and 8 policy configurations:7,200 policy evaluations. All
  cumulative cash constraints and hindsight-oracle regret bounds passed. At
  planned budget ratio 0.7, sequential greedy allocation has mean fulfilled
  fraction0.5785 and mean minimum-disaster service0.0604; forecast reserve has
  0.5583 and 0.2717. These are trade-offs under one synthetic generator, not real
  humanitarian outcomes. Historical funding was not used as need.

## Research decisions

Retain Track A as the frozen **retrospective accounting-prediction benchmark**.
Keep physical-evidence enrichment separate until event attribution, source timing,
coverage and independent validation are established. Do not train another funding
model merely to optimize the already examined59 flood tests. Existing comparisons
and uncertainty studies are preserved as exploratory historical evidence.

The final defensible research architecture separates: source evidence and quality
gates; historical funding prediction; prediction uncertainty and abstention; an
independent need/utility definition; constrained allocation; human review. Its
components have different validation levels. The supplied demonstration runs this
separation and abstains when exact case predictions or credible need inputs are
missing. It is not an operational deployment.

The original reconstruction uses LFYO, which can train on future fiscal years.
New operational comparisons must use strictly earlier training years and
training-only nested selection. Current metadata snapshots do not establish
historical T0/T+72 availability. Excellent retrospective R² cannot repair that
limitation. The handoff’s shorthand about upper-band median specialists is also
incomplete: source code and reconstruction evidence retain a50–200M hybrid and
upper-band specialists. They have not been overwritten with reconstructed medians.

Previously reported interval results show the difference between pooled coverage
and catastrophic reliability. CQR reported7/9 high-value cases covered at90%
nominal, with mean interval width about$140.5M. That is too little tail support
and too much variability for a reliable catastrophic allocation guarantee.
Those are separate Track B models, not intervals fitted around the accepted
Track A predictions. No new interval claim is made here.

## What remains incomplete and why

1. Complete validated multi-hazard physical/exposure dataset: physical sources
   were retrieved in bounded pilots, but broad event linkage, perimeters,
   hydrological relevance, population/building vintage and uncertainty remain
   unverified. Thirty-three fire records do not justify claiming absent wildfire
   data; acquisition and matching remain incomplete, not scientifically impossible.
2. Independent flood labels and raw MODIS historical product QA: rendered maps
   are accessible, but survey truth and raw-product ingestion were not completed.
   Current NASA documentation advertises historical products; do not claim no
   archive exists merely because older documentation described rolling retention.
3. A new untouched evaluation population and comparable finalized targets:
   no such targets were selected or inspected here. A prospective protocol is
   supplied; realized future outcomes cannot be generated within this session.
4. New integration/ensemble selection, measurement-uncertainty propagation,
   real-need allocation validation and reliable rare-event intervals: these
   depend on the preceding data and validation gates. They were not executed
   and must not appear as successful results in a thesis.
5. Comprehensive novelty review and full text verification of all literature:
   a bounded primary-source search was performed. Existing surface-water and
   conformal methods are not novel. No “first” or globally novel framework claim
   is established. A publisher search lead on disaster DSS remains incompletely
   verified and is excluded from a formal verified bibliography.

These gaps mean the full requested thesis research objective remains incomplete.
This report is a substantive, recoverable research checkpoint and a defensible
conclusion about the available evidence, not a declaration that all objectives
have been met. Prediction and geospatial archive access is resolved through the
connector despite persistent CLI403. Separate trained-model weights were not
verified; original training procedures remain preserved. A descriptive fiscal-year
block bootstrap of accepted-minus-original historical MAE gives -$116,541,
with percentile interval [-$301,156, +$28,174]; it crosses zero and is not an
independent validation claim. The next work should ingest raw flood QA and certify
hazard linkage before introducing more models. See STATUS.md and TECHNICAL_GUIDE.md
for exact commands and remaining gates.
