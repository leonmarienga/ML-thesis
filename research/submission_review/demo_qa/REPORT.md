# Offline decision-support QA, November 28, 2026

**Decision:** browser smoke/keyboard QA remains blocked by the absence of a runnable local browser engine. A separate standard-library CLI candidate passes local process-level replay on archived values. Nothing was deployed or written to GitHub.

## Browser artifact

Restored the unchanged private working checkpoint version 1; checked its SHA-256 against the release record (b4279001858503685d3fe024b6a22095438866b1fd4181cb4ac6fa437a6681af). Built the prototype from the pinned research checkout below. The regenerated HTML is byte-identical to the archived file (SHA-256 15e6c2390f2f0e727a213d1574c5856dd68429d2d54e77fd87f702c51a978482). Re-ran all 1,424 JavaScript interactions through the Node DOM-mock harness. The installed Chromium launcher points to an absent browser binary; Firefox and Chrome executables were not found. No substitute browser install or remote browser launch was attempted. This proof covers functionality in the mock, not rendered correctness, focus order, screen-reader behavior or usability.

## Archived CLI

The original command imports the hazard feature builder (and thus its geospatial dependencies), even when it only reads and displays archived CSVs. The isolated local environment had no geopandas. The candidate preserves the original arguments and numerical meaning while using only the Python standard library, checking required fields, requiring an unambiguous archived disaster/prediction, refusing historical years outside FY2010–2024 before interpreting an amount, enforcing finite numeric records and valid probability and interval fields, and producing user-readable errors for absent or malformed inputs. The historical classification and separately archived regression amount are intentionally not forcibly reconciled by the presentation layer. For DR-4722 they are respectively the USD 1M–50M band and a USD 5,677.77 amount estimate. Both are retrospective archive outputs, not decision-time estimates. The scenario selected by the user remains a separate synthetic comparison in service units.

Smoke tests used Python isolated/no-site mode, compared labels and availability keys with the pinned builder source, compared the display values with the four pinned CSVs, changed the synthetic scenario with the disaster held fixed, covered an event outside the forward cohort, and tested unknown event and scenario, missing and malformed inputs, a duplicate event, and an out-of-scope dummy date. Evidence files include the actual subprocess test output and one generated briefing. The UI/CLI checks do not fit models, replicate the full research notebook, validate forecast skill or measure decision impact. No FY2025 outcomes were queried or used.

## Reproduce

Get an isolated checkout of commit [b0b6edb229d112dd6d11896a512bf1112414e4f6](https://github.com/leonmarienga/ML-thesis/commit/b0b6edb229d112dd6d11896a512bf1112414e4f6) of the thesis research repository. Set REPO to that checkout path. Run:
    
    python test_cli.py "$REPO"
    python -I -S analysis/thesis_demo.py 4722 --scenario-event 1 --results "$REPO/research/results"

This is a delivery candidate in an isolated package, not a replacement committed to the thesis repository. The browser artifact remains the separate, previously preserved Batch 14 interactive simulation viewer.
