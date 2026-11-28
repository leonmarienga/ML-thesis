# Appendix C. Prospective tabletop comparison protocol (draft)

**Status:** Supervisor-review design and synthetic instrumentation check only. No participants, recruitment, consent, participant logs or field recommendations. This is not a prerequisite for reviewing the current simulator thesis.

## Research question and arms

Can an assisted decision-maker produce a better feasible allocation than an unaided decision-maker and the same automated comparator given identical scenario information? Evaluate gain over unaided decisions separately from gain over both standalone arms. The importance of this distinction is documented in [Vaccaro, Almaatouq and Malone (2024)](https://doi.org/10.1038/s41562-024-02024-1). Neither that evidence nor the thesis's 48-world scripted-veto fixture demonstrates a real human effect in disaster logistics.

| Arm | Plan produced |
| --- | --- |
| Unaided | Participant sees the common brief, commits a final plan without advice. |
| Automated comparator | The same frozen linear optimizer used to produce the assisted recommendation, with deterministic inputs, tie handling, version and audit metadata. Call this optimization, not a trained AI model. |
| Assisted | Participant commits an initial plan before seeing the fixed recommendation, then accepts it, edits it or rejects it. Store initial and final separately. |

The initial assisted plan is a within-session revision measure, not an unaided control, because expecting advice may change initial behavior. The commit-before-advice design has precedent in [Buçinca, Malaya and Gajos (2021)](https://kgajos.seas.harvard.edu/papers/bucinca21trust.pdf), not a validated benefit in this task.

## Setting and allocation

Use a single-dispatch tabletop task with fixed synthetic requests, vulnerable zones, stock, route capacities and time constraints. All information shown at decision time and scoring constraints must be the same across conditions in the initial single-stage design. Freeze two matched scenario sets before the pilot. Each set contains normal, delayed and closed-route cases. Give neutral practice cases not used for evaluation. Conceal recommendations in the unaided condition and before the initial assisted commitment. Withhold outcome feedback until the end; don't share another participant's response.

Use four counterbalanced sequences in randomized blocks; within each set randomize case order according to a concealed sequence generated before each approved session. No participant sees the same scenario twice. Example sequence cells are unaided A then assisted B; unaided B then assisted A; assisted A then unaided B; and assisted B then unaided A. The optimizer runs once on every fixed case using the inputs that participants see. Repeated deterministic optimizer runs are not sample-size units. Account for the possibility of period and carryover effects: counterbalancing mitigates them; it does not eliminate them. If pilot evidence suggests strong learning, consider a randomized between-participant design before confirmatory registration.

## Outcomes and analysis

The primary proposed score is minimum coverage across all zones with positive demand, with an infeasible submitted plan assigned zero and retained as a separate failure indicator. An inaccessible zone remains in the denominator; do not drop an infeasible plan or score silence as an observed decision. Missing and timed-out plans are separate, reported by arm, with prespecified missing-data sensitivity for any larger study. Score feasible plans with the pinned evaluator and common constraint definitions. Report the lack of discrimination when hard closures force the primary score to zero for every feasible policy; keep those cases and report the regime split, without choosing a replacement outcome after data are seen. Secondary outcomes include overall and high-vulnerability coverage, infeasibility causes, time, advice acceptance/edit/rejection, and score changes from assisted initial to final. Never infer lives saved from tabletop outcomes.

A feasibility pilot reports flow counts, recording integrity, time and usability barriers, without an efficacy claim or a fixed participant target before approval. An adequately sized confirmatory evaluation predefines the smallest meaningful difference, recruitment population, analysis model and operating characteristics. Analyze assisted versus unaided by randomized assignment, allowing participant and scenario dependence and accounting for period. Compare the assisted plan against the optimizer on the same cases. Report the two contrasts and uncertainty; reserve any claim of improvement over both for evidence on both prespecified comparisons. Do not count several dispatches by one person as independent participants, and do not treat an optimizer's constant per-case result as independent replication.

## Data handling and gates

Only after the supervisor's scope decision and the applicable written institutional ethics/data-protection determination, finalize approved materials, recruitment, consent, withdrawal, storage and retention. EUL identifies the [University Research and Publication Ethics Committee](https://www.eul.edu.tr/en/university-research-publication-ethics-committee/) as its ethics and data-protection review body. This draft does not imply approval or exemption. Any approved future study should use coded study IDs, access-controlled linkage kept separately if needed, schema/version identifiers, and structured override categories without requesting sensitive narrative. Keep participant-level files and allocation lists out of public GitHub. Use generated cases only and keep the FY2025 outcome embargo intact.

Before any pilot, implement and verify full scenario/brief equivalence, advice reveal ordering, record locking, feasibility parity against the exact pinned evaluator, withdrawal/missing pathways, and browser/keyboard access. Study software must be separate from the read-only research viewer. Freeze the scenario bank, randomization, optimizer and scorecard before the confirmatory study. No implementation may silently correct infeasible plans after the decision has been logged.

## Current synthetic dry-run

Run `python dry_run.py` and compare with `dry_run.json`. This small standard-library test verifies four schedule slots, six synthetic scenario IDs, balanced exposures, no within-slot repeats, simple stock/route/quantity rejection, preservation of demand in a closed zone and basic initial/recommendation/final record consistency. The test uses toy rule checks rather than the full study evaluator. It does not collect data, establish browser behavior, certify a real study build, measure usability or estimate comparative performance.
