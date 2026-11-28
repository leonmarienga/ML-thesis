# Supervisor review: start here

**Status: review draft, not a submission or an operational release.** This guide is a route through the evidence, not an assertion that every experiment or academic requirement has passed.

1. **Read:** Download the [editable Word draft](review_draft/Disaster%20Relief%20Thesis%20Supervisor%20Review%20Draft.docx) or open the [six-chapter source](THESIS.md). The [Word build notes](review_draft/BUILD.md) describe the rendering requirements. Use the [six-band methodology](SIX_BAND_METHODOLOGY.md) for the distinction between the original amount stack, the accepted low-band overlays and the exploratory mid-band overlay.
2. **Try:** Download [the offline viewer](decision_support_v2/index.html) as a file, then open it in Chromium. GitHub's source preview is not the app. Use the [walkthrough](decision_support_v2/README.md) and [browser acceptance report](decision_support_v2/browser_qa/REPORT.md). The viewer is a demonstration of archived estimates and simulated policies, not a live forecast, dispatch authority, or measurement of need.
3. **Check:** Use the [claim-to-evidence map](#table-t10-claim-to-evidence-map) below and [completion checklist](../../THESIS_COMPLETION_CHECKLIST.md). Read the open items before relying on any result. In particular, notebook replay and external validation are different requirements.

## Exact evidence snapshots

| Role | Immutable checkpoint |
| --- | --- |
| Frozen benchmark and baseline notebook | [b0b6edb229d112dd6d11896a512bf1112414e4f6](https://github.com/leonmarienga/ML-thesis/tree/b0b6edb229d112dd6d11896a512bf1112414e4f6) |
| Manuscript, offline viewer, T4 and T8 artifacts covered by the review snapshot | [b951c7a4883400c0d69a04d71ec91290351267e3](https://github.com/leonmarienga/ML-thesis/tree/b951c7a4883400c0d69a04d71ec91290351267e3/research/submission_review) |
| Original follow-up experiments, including preserved negative results | [6bd74e7dceaf339df2b9f911141a8a7281428d5f](https://github.com/leonmarienga/ML-thesis/tree/6bd74e7dceaf339df2b9f911141a8a7281428d5f/research/followup_20261128) |

## Table T10. Claim-to-evidence map

| Manuscript question | Inspect this evidence | What it does not establish |
| --- | --- | --- |
| What exactly does 91.01% mean? (Ch. 3.2, 4) | [Frozen metrics](https://github.com/leonmarienga/ML-thesis/blob/b0b6edb229d112dd6d11896a512bf1112414e4f6/research/results/track_a/model_metrics.csv), [F6 counts and errors](figures/F6-T4-band-evidence.csv), [T4 amount versions](figures/T4-amount-versions.csv) | An operational or target-blind score. Four gates used the held amount to decide rule eligibility. The three amount-stack MAEs describe different archived stacks. |
| Did an outcome-blind replacement improve routing? (Ch. 4) | [Fixed target-blind trial](https://github.com/leonmarienga/ML-thesis/tree/6bd74e7dceaf339df2b9f911141a8a7281428d5f/research/followup_20261128/target_blind_router_v1) | The prespecified gate did not pass. A one-case net difference in previously examined years is not independent confirmation. |
| How wide were the intervals, and what failed in the tail? (Ch. 4) | [Matched T8](figures/uncertainty/T8-matched-intervals.csv), [build and scope](figures/uncertainty/REPORT.md), [review-flag audit](https://github.com/leonmarienga/ML-thesis/tree/6bd74e7dceaf339df2b9f911141a8a7281428d5f/research/followup_20261128/uncertainty_audit) | Equal coverage above $50M, superiority attributable solely to interval construction, or added coverage from a review flag. Only nine tail cases were available. |
| Did the 20× challenger justify replacing the baseline? (Ch. 4–5) | [Episode influence and weight study](https://github.com/leonmarienga/ML-thesis/blob/b0b6edb229d112dd6d11896a512bf1112414e4f6/research/continuous/weight_influence/REPORT.md) | A confirmed improvement. The tail interval includes no change and the overall advantage reverses when the largest qualifying training episode is removed. |
| Can physical evidence be joined to the decision timeline? (Ch. 3.4, 4–5) | [Coastal transfer report](https://github.com/leonmarienga/ML-thesis/blob/b0b6edb229d112dd6d11896a512bf1112414e4f6/research/continuous/coastal_transfer/REPORT.md) and [follow-up archive](https://github.com/leonmarienga/ML-thesis/tree/6bd74e7dceaf339df2b9f911141a8a7281428d5f/research/followup_20261128) | Certified real-time availability, comparable historical obligation labels, or additional independent disasters. A physical simulation is not a funding record or authentic demand. |
| Which policy trade-offs were observed? (Ch. 4, App. B–C) | [Transport experiment](https://github.com/leonmarienga/ML-thesis/blob/b0b6edb229d112dd6d11896a512bf1112414e4f6/research/continuous/transport_pareto/REPORT.md), [offline viewer](decision_support_v2/README.md), [prospective tabletop protocol](tabletop/PROTOCOL.md) | Relief effectiveness or a participant comparison. The three arms are scripted; proxy, optimizer and scripted veto are not observations of humans. |
| What was mechanically replayed? (App. A) | [Baseline clean run](replay/BASELINE_CLEAN_REPLAY_REPORT.md); batches [1–4](replay/CONTINUOUS_BATCHES_1_4.md), [5–10](replay/CONTINUOUS_BATCHES_5_10.md), [11–14](replay/CONTINUOUS_BATCHES_11_14.md) | All archived outputs matching exactly. Two Batch 5 tables differ, and Batch 11 has an inner-selection discrepancy despite unchanged final scores. The expanded end-to-end notebook remains open. |

## What to resolve before final submission

- Reconcile sources and citations with the primary references, finish the figure and table register, and re-render the complete draft after revisions.
- Complete the clean expanded notebook and portable verification handoff. Keep both locally nonmatching experiment scopes visible rather than claiming all archive tables reproduce.
- Apply the university's current template and author declaration requirements after institutional confirmation. Do not mark the work academically approved before it is reviewed.
- Keep unavailable comparable FEMA obligations as a scientific limitation. Do not use the 143 reserved FY2025 outcomes to fill evidence gaps. If the real-participant tabletop is pursued, obtain separate approval before recruitment; the current thesis only has simulated comparisons and a protocol.

## Re-run the review smoke check

[Recorded results](review_verification/smoke-results.json) and the [checker](review_verification/reviewer_smoke.py) cover the named review snapshot only. Use a repository-shaped folder containing the pinned `research/submission_review` files. No data, predictions, model weights, or Git credentials are required. The checker compares exact bytes with the pinned Git tree before executing scoped tests.

```sh
curl -fsS 'https://api.github.com/repos/leonmarienga/ML-thesis/git/trees/b951c7a4883400c0d69a04d71ec91290351267e3?recursive=1' -o review-tree.json
python -I -S reviewer_smoke.py /path/to/review-root review-tree.json
```

This performs the core digest, Word container, HTML structure and synthetic tabletop checks. With Node installed, it runs the two JavaScript harnesses. To add actual Chromium automation, supply `--browser-node-modules /path/to/node_modules` with the exact linked [browser dependency lock](decision_support_v2/browser_qa/package-lock.json). The optional checks print `not run` when not selected or not available. A smoke pass does not rerun or validate the predictive science or replace final manuscript rendering.

**Recorded clean-snapshot audit.** The published SHA-256 manifests match 14 viewer files and one Word file. The Word package also passes its CRC, XML and three internal image checks. The tabletop result matched fresh output byte-for-byte (no participants). Node passed 1,448 scripted interactions plus the two levels. Headless Chromium 158.0.8127.0 repeated the two offline viewports and passed the three keyboard workflows without horizontal overflow, HTTP requests or page errors. The chapter's F6 historical totals, T4 dollars and T8 numerators, denominators and displayed medians also matched the published figure tables. The local repository-relative link destinations and follow-up tree paths were checked, not every external URL or fragment. This run did not validate Word desktop clients, assistive technology, user behavior, or field effectiveness.
