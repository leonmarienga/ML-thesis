# F9 and T9: transport policy trade-offs (synthetic)

Figure F9 projects all 89 archived settings onto two axes for readability. Point status is independently recalculated using **all three** mean objectives (total service, high-vulnerability service, least-served zone), applying the recorded 1e-9 comparison tolerance. No 2D dominance rule or line interpolation is used. The 46 nondominated settings are the empirical frontier of the tested settings, not the frontier of all feasible logistics policies. Labels A–D match the four fixed preference profiles in T9. Their weighted-score maxima are checked against all 46 frontier settings, including scores and exact metric identity.

Panel B uses regime-specific mean least-served-zone service for the same vulnerability rule at 0% and 100% of the feasible floor. The displayed one-decimal differences are differences of those means; the pooled paired interval in Chapter 4 comes from the separate paired report. The 48 cases are synthetic (15 normal, 18 delayed, 15 closed). The extrema and apparent trade-offs are not operational effects, actual routes or participant responses.

The input files are the unchanged Batch 14 summaries at [frozen source checkpoint](https://github.com/leonmarienga/ML-thesis/tree/b0b6edb229d112dd6d11896a512bf1112414e4f6/research/continuous/transport_pareto/results): `pareto_points.csv`, `preference_choices.csv`, `disruption_sensitivity.csv`. Download only those three files. The standard-library builder checks their pinned SHA-256 values and refuses changed inputs. From the repository root run:

```sh
python research/submission_review/figures/transport/build_transport_summary.py \
  research/continuous/transport_pareto/results /tmp/transport-figures
diff research/submission_review/figures/F9-simulated-transport.svg /tmp/transport-figures/F9-simulated-transport.svg
diff research/submission_review/figures/transport/T9-preference-summary.csv /tmp/transport-figures/transport/T9-preference-summary.csv
```

Two clean output directories regenerated both files byte-for-byte. SVG rendered locally for review. This audit independently recalculates the frontier and preference decisions **from the archived setting/condition summaries**; no simulator or historical training is rerun. See [independent Batch 14 replay](../../replay/CONTINUOUS_BATCHES_11_14.md) for the complementary simulator check.
