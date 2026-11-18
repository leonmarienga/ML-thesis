# FY2025 target-free reservation and candidate evaluation protocol

**Reservation locked. Outcome access embargoed.** The existing 143 non-Biological declaration IDs and metadata are preserved byte-for-byte. The archived roster SHA-256 is b8ad33116b3fd12b5c410b3774bff687a35ee817652350088157e66c26b3a2ed. The follow-up queried official declaration type under a nonfinancial field whitelist. No FY2025 obligations, costs, payouts, awards or outcomes were downloaded. [Locked roster](../results/follow_up/fy2025_target_free_roster.csv).

## Population and model lock

- Retain all 143 IDs and report eligibility, source coverage and applicability across the full roster. All 912 non-Biological historical training records have major disaster type DR. Use the FY2025 DR stratum as the primary comparable population. EM and FM cases remain reserved for separately defined applicability checks, with their own coverage and labels if comparable. Eligibility uses policy and metadata available before outcome access.
- Lock separate T0 and T+72 evaluations. T0 is the original, documented declaration issuance instant in UTC. A day-only value cannot certify eligibility of features issued during that day. Verify event identity and strict geography from preserved contemporaneous records.
- Freeze one metadata point baseline and one regularized exceedance-risk specification before outcome access. The baseline uses the forward-only metadata design. The challenger uses the regularized model described in the experiment protocol at the fixed 10% review threshold. Any time-valid physical ablations must be fixed before outcome access. Score all arms on identical primary cases, purging shared episodes in training and calibration. Keep probability review separate from amount selection and Track A.
- Report overall monetary MAE and at-least-$50M MAE, with Brier score and precision/recall for the fixed alert. Also report baseline difference, marginal and tail 80%/90% interval coverage and widths, false alerts, missingness, and episode-cluster uncertainty. For fewer than 20 independent tail episodes, flag limited evidence for conditional inference. Exactly $50M enters the upper stratum.

## Required inputs and comparability

| Input | Eligibility at cutoff | Current status |
| --- | --- | --- |
| Disaster metadata, issue timestamp, programs and geography | Original issuance or an archived version publicly available by the cutoff, with record ID, timestamp and hash | Exact issue timestamps and snapshots unresolved |
| Mission Assignment actions | Complete historical ledger captured by cutoff, with immutable amendment IDs and source and action timestamps. Distinguish complete observed zero from source absence. Exclude financial fields at inference. | Current rows and dateReceived provide retrospective proxies. Original historical versions unresolved |
| Cyclone forecasts | Archived official advisory with issuance UTC, known storm identity, wind and track fields at cutoff | Archive identified. Exact FEMA title and issuance bindings incomplete |
| Rain, winds and fires | Documented observations or forecasts issued no later than cutoff, joined using eligible geometry | Final nClimGrid-Daily, HURDAT2, IBTrACS, NIFC reviewed perimeters and post-event county amendments are retrospective inputs |
| Population and housing | Census version released by cutoff and contemporaneously scoped footprint | Census vintages identified. Footprint timing and coverage unresolved |
| Outcomes | Versioned Mission Assignments with confirmed aggregation method and same elapsed horizon for training and FY2025 | Comparable historical ledger and maturation audit pending |

## Outcome maturity and release conditions

Proposed fixed horizon is 36 calendar months after T0, followed by 90 elapsed days for auditing reporting completeness. The latest candidate metadata-only review date is **December 25, 2028**, calculated from declaration dates. This is a conservative design pending validation on historical vintages. Closeout or current refresh dates alone cannot establish historical maturity.

Before FY2025 outcome access, require the following.

1. Lock hashes and issue dates for historical and FY2025 feature versions, allowed missingness, event grouping, roster and analysis code. Complete the outcome-free match with those versions and the locked metadata.
2. Reconcile how obligationAmount, maId, amendments, reversals, and zero or unavailable records produced the original workbook target. A candidate standardized label is net nominal Mission Assignment obligations for the declaration within the fixed horizon. Freeze aggregation, currency, completeness and timing rules, and validate on historical cases only. Apply the same formula to historical and FY2025 outcomes. If it differs from the workbook, rerun the historical comparator on the standardized outcome and preserve both definitions.
3. Record an approved, complete, pre-unblinding package covering the roster, feature vintages, historical maturity evidence, model and evaluation code, metrics, and alert threshold. Outcome access requires all gates to pass.

The funding measure is federal Mission Assignment obligations. Humanitarian need, total disaster damage and program-specific awards are separate quantities. [FEMA Mission Assignments policy](https://www.fema.gov/sites/default/files/documents/fema_policy-104-010-3_mission-assignments_2024.pdf).
