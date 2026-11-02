#!/usr/bin/env python3
"""Reproduce EXACT final accepted 912 outputs from four immutable source artifacts.

Never refit or replace router. Reproduce the old immutable overlay script and
assert MAE 5,014,819.0116 and R2 0.88677657108 before any interval experiment.
"""
from pathlib import Path
import importlib.util,json
ROOT=Path(__file__).resolve().parents[1]
module=ROOT/"analysis"/"nonbio_final_50_200m_immutable_overlay.py"
spec=importlib.util.spec_from_file_location("original_immutable_overlay",module)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.main()
summary=json.loads((ROOT/"audit_outputs"/"nonbio_final_50_200m_overlay"/"summary.json").read_text())
res=summary["overall"]["candidate_with_50_200M_hybrid"]
assert abs(res["MAE_USD"]-5014819.011614494)<.01,res
assert abs(res["R2"]-.8867765710842239)<1e-9,res
print("TRACK_A_EXACT_RECONSTRUCTION_OK",json.dumps(res))
