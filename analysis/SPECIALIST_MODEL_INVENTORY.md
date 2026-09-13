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

Important distinction:
- the later `$1M-$50M` internal sub-band experiments (`analysis/nonbio_mid_subbands.py`) are **routing classifiers**, not amount regressors.

Integration status: **preserved amount model available.**

### $50M-$200M

Preserved implementation:
- `fine_split_regime_experiment.py` on branch `experiment/mission-composition-20260829`

Internal amount specialists:
- `$50M-$100M`: 50/50 blend of weighted ExtraTrees raw-target and LightGBM log-target models trained on broader $20M-$200M support.
- `$100M-$200M`: 75/25 blend of weighted ExtraTrees raw-target and Ridge raw-target models trained on broader $50M-$300M support.

Development results from preserved artifact `fine-split-regime-results`:
- `$50M-$100M`: n=20, R2=-0.3575, MAE=$10.92M, RMSE=$14.09M.
- `$100M-$200M`: n=19, R2=-0.5239, MAE=$28.59M, RMSE=$33.73M.

Validation caveat:
- true internal band selected the specialist in this development experiment.
- therefore the fine split is **oracle-routed** and is not directly deployable behind the broad `$50M-$200M` router without a target-free internal sub-router.

Integration status: **amount specialists preserved, but internal 50/100M routing is not yet deployable.**

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
- prior development work referred to a billion-dollar / `$500M+` Hybrid Expert and reported development performance around R2 ~0.90.

Recovery result:
- the exact original Hybrid Expert implementation and exact validation artifact were **not found** in the preserved Git branch/history searched here.
- the newer `analysis/nonbio_830_router_amount_specialists.py` contains an explicitly reconstructed high-quantile LightGBM + historical-analogue hybrid, but this must **not** be represented as the original Hybrid Expert.

Integration status: **original specialist must be recovered or faithfully rebuilt and revalidated before final end-to-end claims.**

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

`disaster -> frozen 830/912 broad router -> preserved internal sub-router where needed -> preserved amount specialist -> dollar estimate`

with strict outer fiscal-year exclusion for every supervised amount fit.

Before final scoring:
1. freeze and validate the preserved `$100K-$50M` amount regressor as an amount predictor, not only as a $1M boundary tool;
2. decide whether `$50M-$200M` should use a broad specialist or build/freeze a target-free `$50M-$100M` vs `$100M-$200M` internal router;
3. use the preserved target-free `$200M-$500M` stack unchanged;
4. recover or faithfully rebuild the original `$500M+` Hybrid Expert and validate it under the same outer protocol;
5. then rerun a single end-to-end router + specialists evaluation.