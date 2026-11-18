# Chronological transfer for sustained observed county outage recovery

The analysis retains the fixed 24-county sample from the observability pilot. It adds author-released observations from 2020, 2022 and 2023 and checks all 3.56 GB of new source CSVs against the published file-level MD5 values. The replay preserves unchanged original source rows for these counties, with SHA-256 and a pinned author release. The 2023 file names its aggregate customer count `sum`. The analysis harmonizes the header and retains the original row values. [ORNL author release](https://doi.org/10.6084/m9.figshare.24237376.v2), [source documentation](https://doi.org/10.1038/s41597-024-03095-5), [locked protocol](../BATCH8_PROTOCOL.md).

## Endpoint and data support

Each entry requires an observed transition from below 1,000 to at least 1,000 reported customers out. The 48-hour county cooldown applies before checking outcome completeness. The target requires all 12 quarter-hour counts in the final three hours of a fully observed 24-hour window to be below 1,000. A missing quarter prevents scoring. Unobserved clocks or customer counts were never filled. The predictors use calendar season and counts at entry and 15 minutes earlier. County windows may span several grid disturbances or utilities. There is no row-level proof of historical public-release vintages or a cause label for a specific weather event.

| Year | Retained entries | Complete windows | Sustained below threshold |
| --- | ---: | ---: | ---: |
| 2018 | 1,717 | 866 | 601 |
| 2020 | 1,780 | 902 | 630 |
| 2022 | 1,798 | 1,117 | 816 |
| 2023 | 1,753 | 1,022 | 750 |

Across 3,551 retained held entries, 2,139 windows had complete outcomes. The result conditions on that observed subset. The large number of county windows does not supply independent storms. The paired intervals use fixed 14- and 28-day blocks stratified by held year as dependence proxies. [Window outcomes](results/window_outcomes.csv), [clock diagnostics](results/county_clock_audit.csv), [entry selection](results/entry_selection.csv).

## Training-only synthetic test

Two earlier-only origin comparisons used the same held observations and the same fixed L2 logistic specification. The generator estimated phase-specific transitions among eight outage-count bins from complete earlier trajectories, with a 20-count shrinkage prior toward across-phase transitions. It generated eight trajectories per real training entry within each origin and preserved the donor's observed entry predictors. The total synthetic weight equals one quarter of the real training weight. Simulated transitions between states present in training were restricted to observed pairs. The generation can represent recurrence and different elapsed-time transition phases. Its within-phase Markov assumption omits physical repair mechanisms, shared weather shocks and utility reporting changes. [Generator registry](SYNTHETIC_DATA_REGISTRY.csv), [source membership](results/training_trajectory_membership.csv), [transition checks](results/transition_support.csv).

| Model | Brier | Log loss | Average precision |
| --- | ---: | ---: | ---: |
| Pooled prevalence | 0.19703 | 0.58334 | 0.73295 |
| Real-only logistic | 0.18641 | 0.55556 | 0.82822 |
| Weighted hybrid logistic | 0.18661 | 0.55649 | 0.82752 |

The real-only model beat pooled prevalence in both origins. The hybrid increased pooled Brier by **0.000205**. The paired 95% intervals were **[−0.000622, +0.001009]** over 52 two-week blocks and **[−0.000620, +0.000990]** over 26 four-week blocks. The hybrid was slightly better in 2022 and worse in 2023. It did not meet the locked practical gain or uncertainty criteria. Calibration summaries show that, in the main 0.6–0.8 bin, the real-only and hybrid predicted rates averaged 0.695 and 0.693 against observed rates of 0.732 and 0.719 on their respective bin memberships. [Matched metrics](results/scores.csv), [paired comparisons](results/paired_brier_intervals.csv), [calibration](results/calibration.csv).

## Source-version sensitivity

The author paper states that the released data omit zero outages. The selected 2018 and 2020 rows contain no zero counts, while the 2022 and 2023 files include 2,668 and 3,075 explicit numeric zeros. Their provenance is unresolved. The primary design preserved values as recorded. A separate diagnostic reran entry selection, training, generation and scoring with all explicit zeros treated as unknown observations. It retained 1,978 held windows. Hybrid Brier again increased, by **0.000353**, and the paired intervals still included zero. This diagnostic preserves the same model settings and does not resolve the true clock state of the excluded observations. [Sensitivity metrics](zero_sensitivity/scores.csv), [sensitivity paired intervals](zero_sensitivity/paired_brier_intervals.csv).

The real-only fit is retained as a research comparator and the simulated augmentation is kept as a negative result. The measured county threshold endpoint supplies no evidence about individual restoration, verified humanitarian need or FEMA funding amounts.
