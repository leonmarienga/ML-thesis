# Appendix D equation verification

The manuscript's equations are traced to the [frozen implementation](https://github.com/leonmarienga/ML-thesis/commit/b0b6edb229d112dd6d11896a512bf1112414e4f6). See [Appendix D](../THESIS.md#appendix-d-implemented-formulations) for formulas, scope and exact source links. E1–E7 describe distinct studies. In particular, the signed 36-month action sum is not the clipped historical workbook target, and fixed-split conformalized quantile regression is not the episode-purged residual pipeline.

Two isolated runs on synthetic fixtures produced [identical results](toy_results.json). Tests cover exact band edges, signed eligibility and maturity, fold cutoff and grouping, log-to-USD clipping and paired group weighting, finite-sample ranks and residual coverage, service balance and fairness, eight transport objectives and the legacy RL release and hard cap. The driver loads selected function definitions and constants from pinned Git objects, not the archival outcomes. It fits no model. Front-end usability, independent predictive validity, route sequencing and real-world humanitarian effectiveness are outside this audit.

From this directory, supply a checkout containing the pinned commit. The test uses the baseline NumPy, pandas and SciPy environment, documented in [the repository requirements](../../requirements.txt).

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python validate_toy_equations.py \
  /path/to/pinned-checkout /path/to/new-output.json
```

The Word builder uses native Office Math nodes and separate plain currency markers; see [integrated build and visual QA instructions](../review_draft/BUILD_INTEGRATED.md).
