#!/usr/bin/env python3
"""Fixed-model, single 2019 holdout calibration for early-response intervals.

Train once: FY <=2018. Calibrate on FY2019. Test FY2020-2024. No
post-2018 model refit; conformal ranking is based only on the separate 2019
calibration targets. Time drift still invalidates exchangeability guarantees.

Original current FEMA records NOT authenticated historic version snapshots;
therefore no prospective-deployment claim despite strict cutoff filtering.
"""
from pathlib import Path
import math,json
import numpy as np,pandas as pd
import early_response_912_t0_t72_forward as base
OUT=base.ROOT/"audit_outputs"/"conformal_frozen_2019"
OUT.mkdir(parents=True,exist_ok=True)
ARMS=["metadata","t72_missions_nhc72"]
METHODS=["absolute_dollar","absolute_log1p"]
LEVELS=[0.80,0.90]
def residual(df,method):
 y=df.target.to_numpy(float);p=df.pred_point.to_numpy(float)
 if method=="absolute_dollar":return abs(y-p)
 return abs(np.log1p(y)-np.log1p(p))
def conformal_q(e,level):
 n=len(e)
 k=math.ceil((n+1)*level)
 return float(np.sort(e)[k-1]) if k<=n else float("inf"),int(k)
def bounds(p,q,method):
 if method=="absolute_dollar":return max(0.,p-q),p+q
 log=np.log1p(p)
 return max(0.,float(np.expm1(max(0.,log-q)))),float(np.expm1(min(710.,log+q)))
def metrics(d):
 return {"n":len(d),"empirical_coverage":float(d.covered.mean()),
   "mean_width_M":float(d.width.mean()/1e6),
   "median_width_M":float(d.width.median()/1e6),
   "above_upper_n":int(d.above_upper.sum()),
   "below_lower_n":int(d.below_lower.sum())}
def main():
 d=base.load()
 train=d[d.fyDeclared<=2018];cal=d[d.fyDeclared==2019]
 te=d[d.fyDeclared.between(2020,2024)]
 assert len(train)==536 and len(cal)==68 and len(te)==308,(len(train),len(cal),len(te))
 rows=[];fit=[]
 for arm in ARMS:
  c_pred,c_band=base.evaluate_fold(train,cal,arm,2019)
  t_pred,t_band=base.evaluate_fold(train,te,arm,2019)
  cp=cal[["disasterNumber","target"]].copy()
  cp["pred_point"]=c_pred
  tp=te[["disasterNumber","fyDeclared","target","band","incidentType"]].copy()
  tp["pred_point"]=t_pred
  tp["pred_band"]=t_band
  fit.append({"variant":arm,"train_n":len(train),"calibration_n":len(cal),
    "test_n":len(te),"calibration_MAE":float(abs(cp.target-cp.pred_point).mean()),
    "test_MAE":float(abs(tp.target-tp.pred_point).mean())})
  for method in METHODS:
   e=residual(cp,method)
   for level in LEVELS:
    q,k=conformal_q(e,level)
    for _,r in tp.iterrows():
     lo,hi=bounds(float(r.pred_point),q,method)
     rows.append({"variant":arm,"method":method,"level":level,
       "disasterNumber":int(r.disasterNumber),"fyDeclared":int(r.fyDeclared),
       "incidentType":r.incidentType,"actual_band":r.band,"pred_band":r.pred_band,
       "actual":float(r.target),"point":float(r.pred_point),
       "calibration_n":len(cp),"q":q,"q_rank":k,
       "lower":lo,"upper":hi,"width":hi-lo,
       "covered":int(lo<=r.target<=hi),
       "above_upper":int(r.target>hi),"below_lower":int(r.target<lo)})
 res=pd.DataFrame(rows)
 # Each 90%-level interval must enclose the corresponding 80% interval.
 for _,g in res.groupby(["variant","method"]):
  a=g[g.level==.8].set_index("disasterNumber").sort_index()
  b=g[g.level==.9].set_index("disasterNumber").sort_index()
  assert a.index.equals(b.index)
  assert (b.lower<=a.lower).all() and (b.upper>=a.upper).all()
 grouped=[]
 for names,part in res.groupby(["variant","method","level"]):
  grouped.append(dict(zip(["variant","method","level"],names),**metrics(part)))
 by_band=[]
 for names,part in res.groupby(["variant","method","level","actual_band"]):
  by_band.append(dict(zip(["variant","method","level","actual_band"],names),**metrics(part)))
 by_hazard=[]
 for names,part in res.groupby(["variant","method","level","incidentType"]):
  by_hazard.append(dict(zip(["variant","method","level","incidentType"],names),**metrics(part)))
 by_year=[]
 for names,part in res.groupby(["variant","method","level","fyDeclared"]):
  by_year.append(dict(zip(["variant","method","level","fyDeclared"],names),**metrics(part)))
 meta={"study":"Frozen 2010-2018 point models / FY2019 calibration / FY2020-24 test",
  "point_model_frozen_across_calibration_and_test":True,
  "n_train":len(train),"n_cal":len(cal),"n_test":len(te),
  "model_fit":fit,
  "overall":grouped,
  "by_true_funding_band":by_band,
  "by_hazard":by_hazard,"by_year":by_year,
  "qualifications":[
   "This is true disjoint training/calibration/test by fiscal year. Source inputs are current reconstructed proxies, not authenticated historical record versions.",
   "Strict finite-sample split-conformal validity requires exchangeability of calibration and future test errors; temporal drift may violate it.",
   "The target is final historical obligated funding, not humanitarian need.",
   "Only 68 FY2019 calibration cases, extremely few catastrophic events; pooled coverage can conceal rare-band failure.",
   "No accepted six-band router or point estimation system is altered; this is separate Track B.",
   "Storm identities from current FEMA titles, not certified at original issuance."
  ]}
 res.to_csv(OUT/"fixed_split_case_intervals.csv",index=False)
 pd.DataFrame(grouped).to_csv(OUT/"fixed_split_overall.csv",index=False)
 pd.DataFrame(by_band).to_csv(OUT/"fixed_split_by_band.csv",index=False)
 pd.DataFrame(by_hazard).to_csv(OUT/"fixed_split_by_hazard.csv",index=False)
 pd.DataFrame(by_year).to_csv(OUT/"fixed_split_by_year.csv",index=False)
 (OUT/"summary.json").write_text(json.dumps(meta,indent=2))
 print("BEGIN_FROZEN_SPLIT_CONFORMAL")
 print(json.dumps({"fit":fit,"overall":grouped,
  "catastrophic_band_90pct":[r for r in by_band if r["level"]==.9 and
    r["actual_band"] in ["50M-200M","200M-500M","500M+"]]},indent=2))
 print("END_FROZEN_SPLIT_CONFORMAL")
if __name__=="__main__":main()
