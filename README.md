# AI-assisted disaster relief decision support research

Research branch: `experiment/thesis-continuation-20261117`, based on
`experiment/noaa-storm-impact-20261102` at commit
`041afba8a10da9c2a962222e0a2644284a84fb7b`. Main and the original benchmark are
preserved. This is a research checkpoint, not a validated operational product.

The continuation executed the interrupted Phase 50 temporal placebo experiment,
retrieved real multi-hazard source data, audited source/decision-time limitations,
and compared allocation policies on explicitly synthetic needs. It found strong
spatial sensitivity and a material incident-date error. It does not establish
an improvement over the frozen historical funding predictor.

Start with the [scientific conclusion](research/FINAL_CONCLUSION.md),
[executed notebook](research/REPRODUCIBLE_RESEARCH.ipynb),
[technical guide](research/TECHNICAL_GUIDE.md), and
[current checkpoint](research/STATUS.md). The
[thesis draft](research/THESIS_METHODS_RESULTS_DISCUSSION.md) separates completed
work, historical evidence, simulation and unresolved objectives.

Provenance and results are recorded in DATA_SOURCE_REGISTRY.csv,
SOURCE_AND_FEATURE_MANIFEST.csv, MODEL_REGISTRY.csv, REFERENCE_REGISTRY.csv and
FINAL_METRICS.csv. Every source tracked at the initial research commit is covered
by research/frozen_source_checksums.json. Saved classified raster windows permit
offline reproduction. The original workbook and historical scripts remain intact.

```bash
python3.12 -m venv work/venv
work/venv/bin/pip install -r requirements-research.txt
work/venv/bin/python -m unittest discover -s analysis -p test_phase50_temporal_placebo.py
work/venv/bin/python analysis/phase50_temporal_placebo.py --offline-inputs research/phase50 --out work/reproduced-phase50
work/venv/bin/python analysis/continuation_allocation_simulation.py
work/venv/bin/python analysis/research_decision_support_demo.py
```

Historical funding obligations are accounting outcomes, not humanitarian need.
Missing satellite observations are not dry land. Local water changes are not
certified disaster footprints. Historical LFYO scores are not prospective
operational accuracy. Exact frozen prediction/model artifacts and several
validation/data gates remain unresolved; see the conclusion before making claims.
