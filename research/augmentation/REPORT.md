# Chronological augmentation for rare FEMA obligations

The experiment supports keeping the existing candidate framework. No novel synthetic generator improved the nested comparisons across rare-event ranking, dollar error and interval coverage. Reweighting and random repeats reduced upper-tail dollar error in the exploratory replay. The resulting estimates still missed high obligations by hundreds of millions of dollars, and their intervals missed every held-out case in the upper three bands.

## Matched evaluation

The study used the retained 912 real non-Biological declarations and twenty-four earlier internal evaluation folds within six outer origins. All 33 training configurations were evaluated on the same 376 real FY2019–24 cases across 325 episode groups. The three high-funding bands contain nine, one and three declarations respectively, spanning 11 episode groups overall. Two earlier real years drove ratio selection. Two older real years supplied probability and dollar-residual calibration after removing records from episode groups also represented in tuning. Training sources had to precede each held year, pass the existing incident-date purge and have no episode overlap. Each augmentation fit ran independently inside its own training partition. FY2019–24 outcomes were previously examined, so all comparisons are exploratory. [Protocol](PROTOCOL.md), [fold membership](results/membership.csv.gz), [selected ratios](results/selections.csv).

The ablation fixed the learner, feature encoding and chronological partitions and varied only the training weights or examples. The features were incident, state and declaration type, month, declaration delay and declared program flags. These are retrospective lean metadata and lack a complete decision-time version audit. The amount regressor trained on log1p obligations and all predictions were bounded by the maximum eligible real training amount. The real-only ablation obtained $19.88M overall MAE. The archived matched nested metadata model obtained $19.91M overall and $547.79M for the same $50M+ cases, with different encoding and training settings. Track A remained frozen. [Full ablation metrics](results/metrics.csv), [archived matched metrics](../results/forward/model_metrics.csv), [archived matched predictions](../results/forward/forward_predictions.csv).

## Matched results after real-only probability calibration

Average precision (AP) in the table relates to obligations of at least $50M. Review alerts use the fixed 10% probability threshold. MAE and RMSE are in millions of US dollars. Reference AP for a constant ranking is the observed 13/376 prevalence, or 0.035.

| Earlier-fold policy | Six-band accuracy | Balanced accuracy | Macro F1 | $50M+ AP | False alerts | MAE | RMSE | $50M+ MAE | R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Real only | 69.15% | 21.33% | .195 | .350 | 53 | 19.88 | 190.34 | 553.27 | −.007 |
| Cost-sensitive weights | 69.41% | 21.10% | .189 | .346 | 52 | 21.64 | 182.95 | 500.60 | .070 |
| Random repeats | 68.62% | 23.97% | .233 | .219 | 43 | 19.97 | 187.61 | 540.61 | .022 |
| SMOTE | 69.15% | 21.33% | .195 | .350 | 53 | 19.88 | 190.34 | 553.27 | −.007 |
| SMOTENC | 70.21% | 20.37% | .196 | .245 | 47 | 19.88 | 190.34 | 553.27 | −.007 |
| SMOTER variant | 70.21% | 20.37% | .196 | .231 | 47 | 19.88 | 190.45 | 553.60 | −.008 |
| SMOGN variant | 69.68% | 19.96% | .190 | .298 | 55 | 19.88 | 190.20 | 552.93 | −.005 |
| Downweighted real–synthetic hybrid | 69.41% | 21.39% | .196 | .350 | 52 | 19.88 | 190.32 | 553.26 | −.007 |

SMOTE never qualified for earlier-fold selection. The Gaussian copula support gate also remained closed in every partition. Both returned the real-only result under the nested policy. CTGAN stayed outside the model fits. [Generator audit](GENERATOR_AUDIT.md), [every ratio and calibration scope](results/metrics.csv), [all six band recalls](results/per_band.csv).

## Improvements worth retaining as exploratory controls

- **Cost-sensitive weights** reduced $50M+ MAE by **$52.67M** (95% paired episode-bootstrap interval for change **−$91.56M to −$12.35M**). Overall MAE rose by $1.76M. Raw $50M+ AP rose from .292 to .535 (paired interval for change −.060 to +.429). In the pooled, calibrated predictions it was .346, compared with .350 for real only (paired interval for change −.160 to +.117). Ranking and probability transport require separate validation. The calibrated alert rule identified 12 of the 13 tail cases and raised 52 false alerts. The reference identified 11 with 53 false alerts. [Paired estimates](results/paired_episode_bootstrap.csv), [probability results](results/reliability.csv).
- **Random repeats** reduced $50M+ MAE by **$12.67M** (paired interval **−$38.46M to −$0.65M**) and increased calibrated balanced accuracy from 21.33% to 23.97%. It recognized one of three $500M+ cases by exact class, zero of one $200M–$500M cases and zero of nine $50M–$200M cases. The real-only classifier recognized 0/3, 0/1 and 2/9 respectively. Repeat sampling worsened calibrated $50M+ AP and Brier score, with one fewer true alert and ten fewer false alerts. Every repeat retained its original donor. [Per-band recalls](results/per_band.csv), [full metrics](results/metrics.csv).
- **Fixed-ratio observations** include a $16.70M overall MAE at 20× random repeats, a .440 calibrated $500M+ AP at 10× repeats and a .353 calibrated $50M+ AP at 2× hybrid. They are results from the inspected grid. The earlier-fold policies did not reproduce those metrics. The $500M+ AP comparison is based on three held declarations. [Full fixed-ratio grid](results/metrics.csv), [selection scores](results/tuning_grid.csv).

The paired comparisons used 800 resamples of complete episode groups. These intervals describe variation over the observed episodes, conditional on the fitted models and explored years. They cannot measure the performance on future catastrophe types or recover information about unobserved episodes. [Paired estimates](results/paired_episode_bootstrap.csv).

## Probability and interval limitations

The real holdout prevalence was 3.46% for $50M+. The real-only raw mean risk was 2.90%, with Brier score .0277. The intercept fit on earlier real cases increased the mean risk to 7.66% and Brier score to .0429. A natural-prevalence calibration population prevents calibration on an oversampled prior. Changes in the prevalence and model score distributions across years still matter. The reliability tables retain each natural test bin, observed rate and alert count for both scopes. [Reliability tables](results/reliability.csv), [case predictions](results/predictions.csv.gz).

Real-only 80% and 90% dollar intervals covered 305/376 (81.1%) and 337/376 (89.6%) overall, with median widths $0.77M and $3.74M. The corresponding tail coverage was **0/13 at both levels for every configuration**, including weights, repeats and all trained generators. Weights increased the 90% tail median width to $8.34M under nested selection. The pooled earlier residuals and the log-dollar learner do not provide valid uncertainty for the tail under temporal shift. In FY2024, omitting the $4.38B peak training episode shifted estimates for the two high-funding held cases by a mean absolute $1.69M for the real-only model, $15.24M with 5× weighting and $2.63M with 5× repeats. The omitted episode was also the most-used SMOTER donor. The refit removed the only compatible $500M+ donor pair. [Coverage and width](results/intervals.csv), [episode sensitivity](results/sensitivity_summary.csv).

The three target bands require additional independent, comparably mature and genuinely time-available real examples to test transferable improvement and calibration. Repeating or interpolating the existing rare examples did not resolve that empirical limit. Existing tail rankings remain exploratory, and FY2025 remains an independent target-free cohort with 143 declarations under its existing embargo. [Generator audit](GENERATOR_AUDIT.md), [FY2025 lock](../follow_up/FY2025_LOCK.md).
