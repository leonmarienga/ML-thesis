# Preregistered exploratory RFC guidance versus historical stage

**Design:** RFC operational guidance, v1. **Status:** frozen before retrieval of historical USGS truth. Acquisition remains locked pending root review and a separate instruction. This is not a public-forecast study. This does not reopen the prior public clock gate, which remains at **0 of 24 eligible targets**. No FY2025 or historical funding data are requested or part of the design.

**Question.** In three geographically and meteorologically distinct, purposively identified historical flood episodes, what errors do the exact 11 previously screened archive-compatible RFC river-stage guidance values show against the same-site approved gage stage at their locked valid instants? This is a narrow retrospective agreement study between an operational guidance archive and gauge observations, conditional on exact archival availability. It neither tests publicly available decision-time skill nor implies that operators or the public saw the RFC guidance at the archive clock.

## Locked source and selection

The five parent source files are exact bytes from [commit e40aa61](https://github.com/leonmarienga/ML-thesis/tree/e40aa61d005df96db353c99bcd297edf7d490425/research/followup_20261128/stage_fixed_lead); the read-only Git tree object is `f578254637a599ccfc10a743130380f6ddb23a39`. `SOURCE_SHA256SUMS` validates those files, including the earlier strict blocked plan. All source files are forecast, provenance or station-series metadata only; no observed-stage outcomes.

The earlier outcome-blind selection searched a common 24 to under 30 hour band, after taking the later of RFC WMO header, explicitly UTC SHEF DC and trace-corroborated HML issue. When multiple vintages fit, it picked the most recent eligible time; an unresolved stage conflict at that time would have blocked selection. The selected vintage, stage value, clock and number of candidates are all fixed by source. Eleven candidates are selected from 24 previously locked times and 13 are not eligible; never promote unavailable times for the new estimand. HML corroboration means exact equal forecast trace values at at least eight shared times and a unique qualifying issue close to both the WMO and DC clocks. IEM processes the HML and archives the RFC; this is trace corroboration, not independent model or publication-clock validation.

| Episode and fixed site | Exact planned UTC observation instants | Pinned series |
|---|---|---|
| KY2022, Jackson JKNK2 / USGS-03280000 | Jul 28 12:00, 18:00; Jul 29 06:00, 18:00 (2022) | `aeec7ccf0dfd418a8180011c8f127f97` |
| OH2024, Wheeling WLGW2 / USGS-03112500 | Apr 4 12:00, 18:00 (2024) | `b25321fb288542249dce1b9826f2eb4a` |
| HELENE2024, Asheville AVLN7 / USGS-03451500 | Sep 27 06:00, 12:00; Sep 28 00:00, 06:00, 18:00 (2024) | `ace6cac9d0464163b5acecbda4979855` |

There is one quantitative physical endpoint: instantaneous **site-local gage height in feet**, not geographic elevation, discharge, damages, humanitarian outcomes or dollars. Current NOAA and USGS metadata pin the same active monitoring point and the instantaneous 00065/00011 series at each site. Before numerical scoring, record documentary event-date station/gauge lineage and a compatible local stage reference for each site. Evidence of relocation, ambiguous reference or unresolved datum offset withholds the event's score. Do not estimate offsets from the held records or silently equate a geodetic reference to gauge zero.

## Future-only acquisition rules

Only after separate authorization and freeze verification, acquire 11 exact-instant responses using the locked prospective query list, no more dates or sites. [USGS continuous schema](https://api.waterdata.usgs.gov/ogcapi/v0/collections/continuous/schema?f=html) defines the date-time filter and continuous stage fields. If retrieval is allowed, retain per-call exact query, timestamp, HTTP status and SHA-256 of returned bytes; failures stay explicit (a bounded transport retry does not change the selection). An alternative endpoint or schema may be considered only as a separately documented transport adaptation, never to select a different observation. Preserve distinct counts for no feature, failed transport, incompatible datum, wrong identity, ambiguous/multiple or paginated result, non-exact timestamp, unapproved/qualified, and invalid value. An eligible response is exactly one complete feature, the specified monitoring location and time-series ID, instantaneous 00065/00011 gage height in ft, approved, explicitly unqualified, finite numeric value, with matching UTC instant. Never interpolate, take nearest, replace an outage with a later value, use rated discharge, resolve ambiguity from plausible values, or query an entire event window.

## Descriptive analysis and stop rule

Define signed error as guidance stage minus approved observed stage for each event and pair. Positive error means guidance was above the observation. The primary summary, **only if all 11 pairs and all three event-level reference checks pass**, is the average of the three event-level mean absolute errors: each event gets equal weight. The secondary summaries in the complete cohort are equal-event mean signed bias and pooled MAE, explicitly weighted by different number of target pairs. Report pair horizon, coverage and exact reason per target regardless of result. For incomplete coverage, the primary remains not evaluable; list only valid-row event summaries with explicit eligible/locked denominators and describe them as observed subsets. Do not reweight the remaining events into a revised primary or add an event to replace a failure.

Correlation among each event's repeated lead-time rows and purposive coverage preclude treating 11 pairs as independent trials. Three episode clusters are too few for inferential claims; no tests, intervals, fitted models, threshold/extremes assessment, independent baseline, accuracy endorsement, hazard ranking, operational dissemination claim or funding inference. Report magnitudes descriptively and retain all unsuccessful rows. This is a follow-up to a negative public-forecast feasibility result, not a substitute that rescues the original claim.

## Verify this freeze

- `sha256sum -c SOURCE_SHA256SUMS`
- `sha256sum -c FREEZE_SHA256SUMS`
- `python -B build_plan.py && python -B test_design.py`
- `sha256sum -c FREEZE_SHA256SUMS`

`build_plan.py` makes no network requests; the emitted URLs are prospective only. `outcome_rules.py` is future offline scoring logic, with no data acquisition or CLI. Tests use fictitious, in-memory envelopes only, confirm exact pairing and equal-event weighting, and ensure that upstream alteration, wrong clock, wrong series/unit, duplicate, pagination, unapproved/qualified, nonnumeric and absent datum cases are rejected. The incomplete fixture cannot emit a primary score. No real observations, numerical scores, or branch writes were made in this phase.
