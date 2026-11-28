# Scoped reviewer reproduction notebook

This companion reproduces selected manuscript evidence from 18 exact GitHub artifacts pinned to three immutable source commits. It needs no training data, model weights, credentials, paid services, or FY2025 outcomes. It deliberately does not rerun or refit the full historical model pipeline. The frozen benchmark and the later experimental results are separate.

## Review and replay

- [Notebook source](thesis-review.ipynb) and [executed notebook](thesis-review.executed.ipynb), with a [machine-readable ledger](claims.json).
- [Evidence manifest](manifest.json) gives source paths, commit IDs, byte counts, SHA-256 and Git blob IDs.
- Python 3.10+ runs the offline verifier with the standard library. Python 3.12.14 was used for the verification.

From this folder, stage only the allowlisted artifacts:

```sh
python -I -S fetch_evidence.py evidence
```

The fetcher verifies each staged file and does not request unlisted data. A previously staged folder can be checked entirely offline:

```sh
python -I -S fetch_evidence.py evidence --check-only
python -I -S notebook_tool.py run thesis-review.ipynb --evidence evidence \
  --executed reviewer-local.executed.ipynb --claims reviewer-local.claims.json
```

Alternatively start Jupyter in this folder (or set `THESIS_REVIEW_EVIDENCE` to the absolute staged path) and run all cells. Use a trusted copy of the notebook and scripts. The verifier authenticates the manifest before any evidence is parsed and rejects modified inputs. New output names are required. Rebuilding source from the [annotated cells](cells/) is optional:

```sh
python -I -S notebook_tool.py build 006773bfcb065f426fd79384c7d275fe102eda2c0b1450d286a4a82a9510cb32 rebuilt.ipynb
```

## Verification and limits

A fresh 18-file retrieval matched the original evidence byte-for-byte. An isolated CPython 3.12.14 environment without pip executed all eight code cells using only the standard library and regenerated the exact claim ledger and stdlib-executed file. The included [CLI tests](test_cli.py) reject a changed manifest or member before export. The archived Jupyter output validates against notebook format 4; an in-process Jupyter kernel executed the same eight cells and returned the identical nine-claim ledger. Separate kernels could not start with either TCP or Unix sockets in the test environment. This is explicitly identified by output metadata; the in-process path is [replayable](validate_inprocess.py) with `nbformat` and `ipykernel`.

The notebook verifies historical route arithmetic, matched forward confusion and the same-cohort uncertainty counts, the transport-floor and synthetic-veto findings, the 20× and target-blind limits, USGS stage residuals from published matched pairs, and the no-participant tabletop dry-run record. It distinguishes recomputed summaries from archived bootstrap intervals or MAEs that are not recalculated here. It neither verifies independent operational validity nor claims that all original tables match. Batch 5 still has two archive-comparison mismatches; Batch 11's inner-selection values still differ while recorded final outputs agree. A complete model-to-report clean pipeline and signed-off final handoff are separate requirements.
