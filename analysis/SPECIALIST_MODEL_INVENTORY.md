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

Integration status: **use strict outer-training band median for the current final non-Biological integration.**

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

Strict direct amount audit (run 34762965578, n=190):
- overall `$100K-$50M`: R2=0.3532, MAE=$1.347M, RMSE=$3.850M.
- `$100K-$1M`: R2=-2.2951, MAE=$274.6K, RMSE=$449.2K.
- `$1M-$50M`: R2=0.1584, MAE=$2.759M, RMSE=$5.838M.

Matched strict-LFYO band-median comparison:
- `$100K-$1M` band-median MAE=$203.1K, RMSE=$266.0K: the shared ExtraTrees model is worse, so reject it for this band.
- `$1M-$50M` band-median MAE=$3.436M, RMSE=$6.648M: the ExtraTrees model improves MAE by about 19.7% and RMSE by about 12.2%.

Integration decision:
- `$100K-$1M`: outer-training band median.
- `$1M-$50M`: preserved ExtraTrees log-dollar specialist.

### $50M-$200M

Strict target-free audit run `34764085040` on 13 non-Biological cases:
- internal sub-router accuracy 61.54%; balanced accuracy 61.25%; AUC 0.60.
- broad-band LFYO median: R2=-0.3573, MAE=$35.47M, RMSE=$48.02M.
- target-free routed preserved specialists: R2=-0.7118, MAE=$42.00M, RMSE=$53.92M.
- oracle sub-band LFYO medians: R2=0.7272, MAE=$17.07M, RMSE=$21.53M.
- oracle preserved specialists: R2=0.4414, MAE=$23.15M, RMSE=$30.80M.

Integration decision: **reject the split ML specialists; use the outer-training broad-band median.**

### $200M-$500M

Preserved implementations:
- `boundary_weighted_specialist_experiment.py`
- `temporal_event_hierarchical_specialist_v2_experiment.py`
- `target_free_router_experiment.py`

Preserved full development benchmark across the original 9 `$200M-$500M` cases:
- target-free routed funding: R2=0.92246, MAE=$13.99M, RMSE=$21.64M.

Important non-Biological-only final integration result on the current 5 true `$200M-$500M` cases:
- R2=-0.8390
- MAE=$56.41M
- RMSE=$81.56M
- 60% within 20%, 80% within 30%, 100% within 50%.

This difference matters: the earlier 0.922 result was a broader development evaluation; the five-case non-Biological-only subset is much smaller and gives unstable within-band R2. The amount stack is retained because it improves MAE versus the current router-band median on these five cases ($56.41M vs $66.11M), despite worse RMSE and R2.

Integration status: **retain the preserved target-free stack, with the five-case non-Biological caveat explicitly reported.**

### $500M+

Strict non-Biological extreme audit run `34765630016` evaluated five `$500M+` cases.

Best model:
- nested selected historical analogue: **R2=0.5790, MAE=$675.8M, RMSE=$860.1M, log-MAE=0.2752, log-R2=0.7396**.
- it beat the extreme median/geometric-mean/mean baselines, direct LightGBM, nested hybrid, and earlier reconstructed hybrid.
- all 5 were within 50%, factor 2, and factor 3.

Integration decision: **use the nested-selected historical analogue. Do not call it the unrecovered original Billion-dollar Hybrid Expert.**

## Final non-Biological end-to-end result

Run `34796746603`, artifact `nonbio-final-830-accepted-amount-stack`, evaluated all 912 non-Biological rows using the frozen 830/912 broad router and only the accepted band amount choices.

### Overall

- Frozen router accuracy: **830/912 = 91.01%**.
- Router-band median baseline: **R2=0.55090, MAE=$10.016M, RMSE=$128.765M, log-MAE=1.5617**.
- Final accepted amount stack: **R2=0.88656, MAE=$5.131M, RMSE=$64.715M, log-MAE=1.5375, log-R2=0.6620**.
- Oracle-band accepted-stack diagnostic: **R2=0.88818, MAE=$4.783M, RMSE=$64.251M, log-MAE=1.4711**.

Relative to the router-band median baseline, the accepted stack:
- reduces MAE by **48.77%**;
- reduces RMSE by **49.74%**;
- improves R2 by **+0.3357**.

The gap between deployable and oracle-band accepted stacks is small in raw-dollar fit (R2 0.8866 vs 0.8882), indicating that broad-band routing is no longer the main driver of raw-dollar error. Router mistakes still matter strongly on proportional/log metrics and raise MAE by about **7.27%** versus the oracle-band accepted stack.

### Final accepted choices

1. `$0-$100K`: outer-training actual-band median.
2. `$100K-$1M`: outer-training actual-band median.
3. `$1M-$50M`: preserved 900-tree ExtraTrees log-dollar regressor, whole held year excluded and output clipped to routed band.
4. `$50M-$200M`: outer-training actual-band median; split ML specialists rejected.
5. `$200M-$500M`: preserved target-free specialist stack.
6. `$500M+`: nested training-selected historical analogue.

### Final by-actual-band deployable metrics

- `$0-$100K` n=699: R2=-24.2419, MAE=$16.1K, RMSE=$74.3K.
- `$100K-$1M` n=108: R2=-2.1244, MAE=$283.2K, RMSE=$437.4K.
- `$1M-$50M` n=82: R2=-1.4357, MAE=$3.677M, RMSE=$9.932M.
- `$50M-$200M` n=13: R2=-2.5824, MAE=$51.95M, RMSE=$78.01M.
- `$200M-$500M` n=5: R2=-0.8390, MAE=$56.41M, RMSE=$81.56M.
- `$500M+` n=5: **R2=0.5790, MAE=$675.8M, RMSE=$860.1M**.

Negative within-band R2 values do not contradict the strong overall R2: within-band variance is narrow and sample sizes are small, while the full-system raw-dollar R2 is dominated by the model correctly separating funding scales spanning zero to billions. For thesis reporting, pair R2 with MAE, RMSE, median error, log error, and tolerance accuracy rather than using within-band R2 alone.

## Routing-only work

The following remain routing/boundary experiments rather than final continuous amount models:
- `analysis/nonbio_mid_subbands.py`
- `analysis/nonbio_low_router.py`
- `analysis/nonbio_low_thresholds.py`
- `analysis/nonbio_low_vote.py`
- `analysis/nonbio_flood_1m_specialist.py`
- hazard-specific rescue/veto/verifier scripts used to reach the frozen 830 router.
