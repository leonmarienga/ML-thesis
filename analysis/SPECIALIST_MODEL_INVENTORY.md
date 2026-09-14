# Funding specialist model inventory

Purpose: document which pre-existing specialist models can defensibly sit behind the frozen non-Biological 830/912 router, and distinguish amount predictors from routing/boundary helpers.

## Frozen upstream router

- Non-Biological six-band router: 830/912 = 91.0088% exact band accuracy.
- Macro recall: 89.5451%.
- True >=$50M root recall: 23/23 = 100%.
- This router is frozen for specialist integration.

## Recovered specialists

### $0-$100K

Status: no dedicated preserved continuous-dollar specialist was found.

The current lower architecture treats this primarily as a routing / zero-or-small-funding regime. Historical two-stage work had a funded-vs-not-funded classifier, but the specialist-regime work on the current branch does not contain a standalone final amount regressor specifically for $0-$100K.

Integration status: **needs a frozen amount rule/model if exact dollar prediction is required; router band itself is already validated.**

### $100K-$1M and $1M-$50M

Preserved implementation:
- `analysis/nonbio_cross_1m_rescue.py`
- `analysis/nonbio_1m_regression_cutoff.py`

Amount model:
- `ExtraTreesRegressor`
- 900 trees
- trained on rows with $100K <= funding < $50M
- target = `log1p(totalObligatedFunding)`
- target-blind current/mission-semantic features
- strict outer/inner LFYO use in the low-router audits

Role in the historical architecture:
- produces a dollar estimate inside the $100K-$50M support window
- the predicted amount is used around the $1M boundary to separate `$100K-$1M` from `$1M-$50M`

Strict direct amount audit (run 34762965578, n=190):
- overall `$100K-$50M`: R2=0.3532, MAE=$1.347M, RMSE=$3.850M, median AE=$0.396M, log-MAE=0.6732, 58.95% within factor 2, 80.53% within factor 3.
- `$100K-$1M`: R2=-2.2951, MAE=$274.6K, RMSE=$449.2K, log-MAE=0.6243.
- `$1M-$50M`: R2=0.1584, MAE=$2.759M, RMSE=$5.838M, log-MAE=0.7376.

Matched strict-LFYO band-median baselines calculated from the same held-out predictions:
- overall band-median baseline: R2=0.1661, MAE=$1.598M, RMSE=$4.372M, log-MAE=0.6566.
- `$100K-$1M` band-median: MAE=$203.1K, RMSE=$266.0K, log-MAE=0.5986. The preserved ExtraTrees amount model is worse here on the main error metrics, so **do not freeze it as the final `$100K-$1M` amount specialist**.
- `$1M-$50M` band-median: R2=-0.0912, MAE=$3.436M, RMSE=$6.648M, median AE=$1.855M, log-MAE=0.7329. The preserved ExtraTrees model improves MAE by about 19.7%, RMSE by about 12.2%, and median AE by about 21.0%, although log-MAE is essentially unchanged/slightly worse.

Important distinction:
- the later `$1M-$50M` internal sub-band experiments (`analysis/nonbio_mid_subbands.py`) are **routing classifiers**, not amount regressors.

Integration decision:
- `$100K-$1M`: use a simple strict training-band baseline unless a better preserved specialist is recovered/built; do not use the shared ExtraTrees regressor as the frozen amount model.
- `$1M-$50M`: the preserved ExtraTrees log-dollar regressor adds useful dollar-error reduction and remains a viable specialist candidate.

### $50M-$200M

Preserved implementation:
- `fine_split_regime_experiment.py` on branch `experiment/mission-composition-20260829`

Internal amount specialists:
- `$50M-$100M`: 50/50 blend of weighted ExtraTrees raw-target and LightGBM log-target models trained on broader $20M-$200M support.
- `$100M-$200M`: 75/25 blend of weighted ExtraTrees raw-target and Ridge raw-target models trained on broader $50M-$300M support.

Old development results from `fine-split-regime-results` were oracle-routed and therefore not deployable.

Strict target-free audit run `34764085040` on the 13 non-Biological `$50M-$200M` cases:
- internal sub-router accuracy: 61.54%; balanced accuracy 61.25%; AUC 0.60.
- broad-band LFYO median: R2=-0.3573, MAE=$35.47M, RMSE=$48.02M.
- routed sub-band LFYO medians: R2=-0.3715, MAE=$36.06M, RMSE=$48.27M.
- target-free routed preserved specialists: R2=-0.7118, MAE=$42.00M, RMSE=$53.92M.
- oracle sub-band LFYO medians: R2=0.7272, MAE=$17.07M, RMSE=$21.53M.
- oracle preserved specialists: R2=0.4414, MAE=$23.15M, RMSE=$30.80M.

The deployable specialist stack increased MAE by 18.42% versus the broad median and 16.47% versus routed sub-band medians. Even with perfect internal routing, simple sub-band medians beat the preserved ML specialists.

Integration decision: **reject the current `$50M-$100M` / `$100M-$200M` ML specialist stack. Use a training-derived `$50M-$200M` baseline unless a materially better direct amount model is found.**

### $200M-$500M

Preserved implementations:
- `boundary_weighted_specialist_experiment.py`
- `temporal_event_hierarchical_specialist_v2_experiment.py`
- `target_free_router_experiment.py`
- branch `experiment/mission-composition-20260829`

Amount architecture:
- `$200M-$300M`: direct overlapping-window expert + pooled within-band-position expert.
- `$300M-$500M`: high-value direct/static expert with event-aware fallback for sparse/unseen families.
- no standalone model trained only on five or four tiny target-band observations; broader high-value support is used.

Preserved strict LFYO specialist benchmark (`boundary-weighted-specialist-results`):
- overall `$200M-$500M`: R2=0.8847, MAE=$16.96M, RMSE=$26.39M.
- `$200M-$300M`: R2=0.3144, MAE=$10.89M, RMSE=$15.42M.
- `$300M-$500M`: R2=0.5026, MAE=$24.55M, RMSE=$35.63M.
- LOO development diagnostics were higher (overall R2=0.9800) but are not the conservative headline.

Target-free internal router:
- CatBoost classifier trained on $100M-$1B non-held-year support.
- router label: funding > $300M on training data only.
- hyperparameters selected by inner LFYO log loss.
- router accuracy: 8/9 = 88.89%.
- router balanced accuracy: 87.5%.

End-to-end target-free routed funding result (`target-free-router-results`):
- R2=0.92246
- MAE=$13.99M
- RMSE=$21.64M
- one internal routing miss: disaster 4340, $301.825M routed to `$200M-$300M`.
- oracle specialist reference: R2=0.97953, MAE=$7.99M, RMSE=$11.12M.

Integration status: **best-preserved deployable specialist stack; ready to sit behind the frozen broad `$200M-$500M` router.**

### $500M+

Historical project state:
- prior development work referred to a billion-dollar / `$500M+` Hybrid Expert and reported development performance around R2 ~0.90, but the exact original implementation could not be recovered.

Strict non-Biological extreme audit run `34765630016` evaluated five `$500M+` cases with whole held-fiscal-year exclusion and nested training-only component selection.

Simple baselines:
- extreme median: R2=-0.7088, MAE=$1.547B, RMSE=$1.733B, log-MAE=0.7784.
- extreme geometric mean: R2=-0.5339, MAE=$1.387B, RMSE=$1.642B, log-MAE=0.6935.
- extreme mean: R2=-0.5625, MAE=$1.510B, RMSE=$1.657B, log-MAE=0.7335.

Model results:
- nested selected direct LightGBM: R2=0.0400, MAE=$1.226B, RMSE=$1.299B, log-MAE=0.6614.
- nested selected historical analogue: **R2=0.5790, MAE=$675.8M, RMSE=$860.1M, log-MAE=0.2752, log-R2=0.7396**.
- nested selected hybrid: R2=0.4997, MAE=$801.5M, RMSE=$937.6M, log-MAE=0.3554.
- legacy reconstructed hybrid: R2=0.2566, MAE=$902.9M, RMSE=$1.143B, log-MAE=0.4158.

The nested analogue reduced MAE by about 51.3% versus the best simple baseline (geometric mean), 44.9% versus the nested direct ML model, 15.7% versus the nested hybrid, and 25.2% versus the earlier reconstructed hybrid. It placed all five cases within factor 2 and factor 3, and all five within 50% relative error.

Integration decision: **use the nested-selected historical analogue as the current defensible `$500M+` specialist. Do not call it the original unrecovered Billion-dollar Hybrid Expert.** The five-case sample is extremely small, so these metrics remain high-variance development-validation estimates.

## Routing-only specialist work that must not be confused with amount ML

The following are primarily band/boundary routing experiments, not final continuous funding specialists:
- `analysis/nonbio_mid_subbands.py`
- `analysis/nonbio_low_router.py`
- `analysis/nonbio_low_thresholds.py`
- `analysis/nonbio_low_vote.py`
- `analysis/nonbio_flood_1m_specialist.py`
- hazard-specific rescue/veto/verifier scripts used to reach the frozen 830 router.

They may improve which specialist is selected, but they do not replace the amount-prediction specialist layer.

## Diagnostic integration run (not final)

Run 34745067053 used the frozen 830 router plus a mixture of preserved and reconstructed amount specialists.

Artifact `nonbio-830-router-amount-specialists`:
- router-band median baseline: R2=0.5509, MAE=$10.02M, RMSE=$128.77M, log-MAE=1.5617.
- router + mixed specialist layer: R2=0.80185, MAE=$6.48M, RMSE=$85.53M, log-MAE=0.8875.
- oracle-band mixed-specialist ceiling: R2=0.80383, MAE=$6.11M, RMSE=$85.10M, log-MAE=0.7469.

This run demonstrates that the router->specialist architecture is promising, but it is **not the final thesis result** because lower-band models were partly reconstructed/generic and the original `$500M+` Hybrid Expert was unavailable.

## Final integration target

The final non-Biological stack should be:

`disaster -> frozen 830/912 broad router -> accepted band amount rule/specialist -> dollar estimate`

with strict outer fiscal-year exclusion for every supervised amount fit.

Current accepted amount choices:
1. `$0-$100K`: unresolved exact-dollar specialist; use a simple training-derived baseline until improved.
2. `$100K-$1M`: simple training-band baseline; shared ExtraTrees rejected for final amount prediction.
3. `$1M-$50M`: preserved ExtraTrees log-dollar specialist remains useful.
4. `$50M-$200M`: reject split ML specialists; use a training-derived broad-band baseline unless a better direct model is found.
5. `$200M-$500M`: use the preserved target-free specialist stack unchanged.
6. `$500M+`: use the nested-selected historical analogue from run `34765630016`.

Next step: rerun one end-to-end frozen 830 router -> accepted amount choices evaluation with common baselines and metrics.