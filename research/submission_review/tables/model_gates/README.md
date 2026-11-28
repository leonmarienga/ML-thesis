# Model lineage (T3) and synthetic/catastrophic gates (T7)

The CSV extracts preserve 12 model versions, 14 distinct gate dispositions and 66 typed metrics. Start with [the review tables](T3_T7_MANUSCRIPT_BLOCKS.md) or explore [T3](T3-model-lineage.csv), [T7 gates](T7-synthetic-catastrophic-gates.csv) and [T7 metrics](T7-gate-metrics.csv). The [source manifest](SOURCE_MANIFEST.csv) traces inputs to 40 immutable Git blob identities and SHA-256 digests. Each disposition distinguishes an evaluated comparison, a feasibility screen or a blocked fit.

The builder reads only named, version-pinned historical evidence objects from an existing Git cache and the four staged follow-up manifest/files. It was verified with network lazy fetching disabled and two clean output directories.

```sh
GIT_NO_LAZY_FETCH=1 python -I -B build_tables.py /path/to/local/repo-cache /tmp/t3-t7-rebuilt
```

No model fitting, held FY2025 acquisition or raw historic truth replay occurs. A table can verify a reported aggregate without certifying training parity, provenance before decision time or an independent benefit.
