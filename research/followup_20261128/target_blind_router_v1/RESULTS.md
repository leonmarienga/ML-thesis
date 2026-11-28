# Target-blind six-band replacement v1: no promotion

**Disposition:** fixed metadata interactions yielded one extra correct route, but the paired confidence interval includes zero and support is concentrated in two episode groups. The candidate did not meet the pre-specified independent-replication priority gate. Preserve it as a falsifiable research experiment; do not promote, retune on these same outcomes, or compare its accuracy numerically with the frozen retrospective router as if they were equivalent.

## Primary result

| Measure | Fixed fresh baseline | Target-blind candidate |
|---|---:|---:|
| Correct routes (376 declarations) | 273 (72.61%) | 274 (72.87%) |
| Pooled six-class macro-F1 | 0.21555 | 0.22096 |
| Mean ordinal absolute error (bands) | 0.43617 | 0.42287 |
| Multiclass Brier (sum over classes; lower is better) | 0.41094 | 0.40812 |
| ≥$50M threshold recall | 1/13 | 2/13 |
| ≥$200M threshold recall | 0/4 | 0/4 |

Paired episode-group bootstrap (10,000 replicates): candidate-minus-baseline macro-F1 **+0.005415**, 95% percentile interval **[0.000000, +0.016941]**. This interval is descriptive uncertainty for the already examined historical sample, not independent confirmation. Of the three changed classifications, one became correct and two remained incorrect. Neither model correctly routed any of the four actual ≥$200M cases into its exact band; the sole candidate prediction of $200M–<$500M is not a correct ≥$200M threshold classification. The 266/376 support in the lowest band explains why accuracy alone is uninformative.

## Coverage, checks and scope

All 376 held cases had valid, nonnegative declaration elapsed time. The gate allowed 73 of 75 held cases in the three target hazards: 9/9 Severe Ice Storm, 14/16 Fire, and 50/50 Hurricane. All three route changes were Hurricanes, spanning two source-connected groups. Ten actual ≥$50M cases were eligible and two changed; the scoring shows this because **actual dollars are not used to protect a row**. Predicted ≥$50M baseline routes and other hazards never change, by definition. The gate fails two criteria: the primary interval's lower bound is not strictly positive, and only two rather than ten distinct episode groups changed. Upper-threshold recall did not decline, but sparsity makes that guard weak.

No training row meets or follows the earliest held start/declaration date, or shares a disclosed group with the corresponding held fold. Time purged 92 earlier-FY rows across origin fits; applying the group purge after time exclusion removed no additional rows. Across all six origins, 160 held label positions changed under permutation and 315 under deterministic class reassignment. Both models' outputs and probabilities were exactly invariant in every origin for both interventions, with sealed output hashes. A second local process replay reproduced all nine evidence files byte for byte. This is a leakage test for **held target dependence**, not a certification that vintage, episode search coverage or historical label maturity is fully known.

Final mission counts, reported amendment burden, and post-event duration were explicitly excluded. The replacement uses declaration type and elapsed time as modest metadata proxies, not measurements of strength, severity or operational activity. Snapshot metadata and final accumulated historical outcomes are not authenticated as-at-time sources. No FY2025 outcome/metadata was accessed; no frozen Track A predictions were inputs; there was no label-dependent experiment selection. Treat this as a matched chronological archive backtest on 325 previously examined episode groups, not an independent disaster validation, and do not infer operational accuracy or allocation effectiveness.

If this lane continues, prioritize a provenance-first pre-registration for independently archived timestamps and more severe episodes, not another rule search on this historical holdout. The direct historical mission/duration mechanisms remain ineligible without separately verified time-of-availability replacements. The present candidate offers proof of outcome-blind construction, not evidence of a material improvement.

## Reproduce

From this folder, run `./run.sh <pinned-checkout> <venv-python> <new-output-directory>`. Requires Git, Python 3.12, NumPy 2.5.5, pandas 2.3.3 and scikit-learn 1.9.1 (observed versions in `evidence/summary.json`). The launcher verifies the pre-scoring protocol SHA and the script verifies pinned commit and source bytes. `evidence/forward_fold_checks.csv` contains per-origin label stress tests and data separation; `evidence/predictions_without_labels.csv` contains sealed predictions without outcomes. See `evidence/eligibility_and_changes.csv`, `evidence/band_metrics.csv`, `evidence/confusion.csv` and `evidence/bootstrap_episode_delta.csv` for the diagnostics. No archived feature table or funding label dump is included in this bundle.
