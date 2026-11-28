"""Retrospective audit of the existing fixed metadata dollar intervals; no refit."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

repo=Path(sys.argv[1]).resolve()
src=repo/'research/results/uncertainty/forward_prediction_intervals.csv'
d0=pd.read_csv(src)
assert d0.fyDeclared.between(2010,2024).all()
d=d0.query("variant == 'nested_metadata' and method == 'dollar' and level == 0.9").copy()
assert len(d)==d.disasterNumber.nunique()==308 and d.fyDeclared.between(2020,2024).all()
assert (d.last_calibration_year<d.fyDeclared).all()
d['cover_check']=(d.target>=d.lower_usd)&(d.target<=d.upper_usd)
assert (d.cover_check==d.covered).all()
d["tail"]=d.target.ge(50_000_000)
assert d["tail"].sum()==9
d["score90"]=d.width_usd+20*((d.lower_usd-d.target).clip(lower=0)+(d.target-d.upper_usd).clip(lower=0))
groups=[("all",d),("below_50m",d[~d["tail"]]),("at_least_50m",d[d["tail"]])]
groups+= [(f"fy{year}",g) for year,g in d.groupby("fyDeclared")]
rows=[]
for group,g in groups:
 rows.append(dict(scope=group,n=len(g),covered=int(g.covered.sum()),coverage_pct=100*g.covered.mean(),
                  flagged=int(g.review.sum()),misses=int((~g.covered).sum()),
                  flagged_misses=int((~g.covered & g.review).sum()),
                  median_width_usd=g.width_usd.median(),mean_interval_score_usd=g.score90.mean()))
out=pd.DataFrame(rows)
out.to_csv(Path(__file__).with_name("audit.csv"),index=False,float_format="%.6f",lineterminator="\n")
print(out.to_string(index=False))
