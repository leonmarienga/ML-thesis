# Forward-time and decision-time validity audit (2026-11-01)

## Scope
The accepted six-band router at 830/912 remains frozen. The hazard-exposure
experiment uses a **retrospective** 971-row FEMA enrichment and a fixed
5-neighbour 50–200M specialist. The chronological pilot changes **only** the
fiscal-year training rule: training cases must have `fyDeclared < test year`.

**This is NOT a declaration-time or genuine as-of feature evaluation.**

## Source availability and action

| Family | Previous pilot | Available at declaration? | Next valid approach |
|---|---|---|---|
| Census county population | 2019/2024 county population estimates retrospectively matched to event-report counties | Historical population can be available, but the chosen revisions may postdate each disaster; event-county membership comes from finalized NOAA reports | Census 2010/2020 snapshots with documented publication dates, plus declared-area FIPS list available as of t0; do not use final event-counties |
| NOAA Storm Events | Finalized event list, IDs, dates and county membership | No: many NOAA final reports are released weeks/months later | Time-stamped preliminary NWS LSRs, rainfall gauges, radar or warnings published before t0 |
| NHC HURDAT2 | Final best-track hurricane wind, pressure and tracks | No: final best-track can be revised after the event | Archived advisories with `issue_time <= t0`, storm positions and warnings issued by t0 |
| CAL FIRE perimeters | Retrospective acreage and post-fire inspections | No: final mapped perimeter and inspections may be post-event | Archived satellite hotspot detections with acquisition time <= t0; operational perimeter snapshots |
| EAGLE-I | Full-event outage trajectories | No: later trajectory unavailable at t0 | Truncate telemetry strictly at t0; normalize only by already-known customer estimates |
| FEMA missions/agencies | Eventual mission totals and response complexity | Not always | Retain only missions with effective record and observation timestamps <= t0; otherwise exclude |

## Build a genuinely as-of evaluation
1. Fix each disaster's t0: FEMA declaration timestamp (separate optional t+72 h).
2. Extract at-declaration designated county FIPS from FEMA record version history. Distinguish declaration amendments added later.
3. Join historical population data **as released by t0**; never join via later NOAA event membership.
4. For each hazard, join issued/observed versions with `published_at <= t0`, and permit missing values. Do not backfill post-t0 observations.
5. Exclude future missions, future NOAA official damage estimates, final HURDAT best tracks, and post-event assessment data.
6. Train on earlier fiscal years only, using a historical source snapshot where possible; test annually and report selection-independent confidence intervals.
7. Evaluate all 912 routed cases and rare-band recall before replacing accepted model. Check coverage and false event matches.
8. Maintain two independent claims: (a) retrospective physical-impact research; (b) decision-time available predictive research.

## Caveats
Training on earlier historical fiscal years while using feature data collected or
finalized later is only a **chronological generalization experiment**. It does
not eliminate temporal leakage due to feature publication delays. The pilot
uses only 13 high-band observations and cannot establish general applicability.
