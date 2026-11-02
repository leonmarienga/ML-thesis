#!/usr/bin/env python3
"""Research-only CQR: frozen pre-2019 quantile models, 2019 calibration, 2020-24 test.

Conformalized quantile regression in LOG1P dollar space; all features use
historically screened proxies, not independently certified t0 source versions.
Point model/accepted router never modified. Quantile models fit only pre-2019.
"""
import json,math
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import early_response_912_t0_t72_forward as early
OUT=early.ROOT/"audit_outputs"/"cqr_track_b"
OUT.mkdir(parents=True,exist_ok=True)
LEVELS=[.8,.9]
def qstat(e,level):
 k=int(math.ceil((len(e)+1)*level))
 return float(np.sort(e)[k-1]) if k<=len(e) else float("inf")
def metrics(df):
 return {"n":len(df),"coverage":float(df.covered.mean()),
         "median_width_M":float(df.width.median()/1e6),
         "mean_width_M":float(df.width.mean()/1e6),
         "above_upper":int((df.actual>df.upper).sum())}
def main():
 d=early.load()
 train=d[d.fyDeclared<=2018]
 cal=d[d.fyDeclared==2019]
 test=d[d.fyDeclared.between(2020,2024)]
 assert len(train)==536 and len(cal)==68 and len(test)==308
 arm="t72_missions_nhc72"
 xcal,xte=early.transform(train,cal,arm)
 xtrain,_=early.transform(train,train,arm)
 # The original train-derived column layout must be identical.
 assert list(xcal.columns)==list(xtrain.columns)==list(xte.columns)
 yt=np.log1p(train.target.to_numpy(float))
 predictions=[]
 fit=[]
 for level in LEVELS:
  a=(1-level)/2
  lo=HistGradientBoostingRegressor(loss="quantile",quantile=a,max_iter=140,
      max_leaf_nodes=9,min_samples_leaf=20,l2_regularization=2,random_state=20261101)
  hi=HistGradientBoostingRegressor(loss="quantile",quantile=1-a,max_iter=140,
      max_leaf_nodes=9,min_samples_leaf=20,l2_regularization=2,random_state=20261101)
  lo.fit(xtrain,yt);hi.fit(xtrain,yt)
  lc,uc=lo.predict(xcal),hi.predict(xcal)
  lt,ut=lo.predict(xte),hi.predict(xte)
  # Non-crossing enforce sorted raw endpoints at each observation.
  lc,uc=np.minimum(lc,uc),np.maximum(lc,uc)
  lt,ut=np.minimum(lt,ut),np.maximum(lt,ut)
  scores=np.maximum(lc-np.log1p(cal.target.to_numpy(float)),
    np.log1p(cal.target.to_numpy(float))-uc)
  q=qstat(scores,level)
  lower=np.expm1(np.clip(lt-q,0,700)).clip(0)
  upper=np.expm1(np.clip(ut+q,0,700)).clip(0)
  fit.append({"level":level,"calibration_n":len(cal),"q_log":q,
    "raw_median_width_M":float(np.median(np.expm1(np.clip(ut,0,700))-
                                        np.expm1(np.clip(lt,0,700)))/1e6)})
  for (_,row),l,u in zip(test.iterrows(),lower,upper):
   predictions.append({"disasterNumber":int(row.disasterNumber),
       "fyDeclared":int(row.fyDeclared),"incidentType":row.incidentType,
       "true_band":early.band(row.target),"level":level,
       "actual":float(row.target),"lower":float(l),"upper":float(u),
       "width":float(u-l),"covered":int(l<=row.target<=u)})
 out=pd.DataFrame(predictions)
 summ=[{"level":level,**metrics(g)} for level,g in out.groupby("level")]
 bands=[{"level":level,"band":label,**metrics(g)}
   for (level,label),g in out.groupby(["level","true_band"])]
 out.to_csv(OUT/"cqr_case_intervals.csv",index=False)
 pd.DataFrame(bands).to_csv(OUT/"cqr_by_true_band.csv",index=False)
 obj={"method":"CQR log1p quantile HistGradientBoosting; quantile-models trained <=2018; calibrate 2019; test 2020-24",
      "train_n":len(train),"calibration_n":len(cal),"test_n":len(test),
      "fit":fit,"overall":summ,"by_band":bands,
      "limitations":["No exchangeability guarantee across chronological drift.",
       "The input source versions are not genuinely certified as-of.",
       "Quantile models are newly trained research models; the accepted Track A output is immutable.",
       "Extreme-dollar log intervals can become economically absurd. Report mean and median widths.",
       "Only nine catastrophic true-funding cases in 2020-24 test period."]}
 (OUT/"summary.json").write_text(json.dumps(obj,indent=2))
 print("BEGIN_CQR_SUMMARY")
 print(json.dumps({"fit":fit,"overall":summ,"high":[x for x in bands if x["band"] in
     ["50M-200M","200M-500M","500M+"]]},indent=2))
 print("END_CQR_SUMMARY")
if __name__=="__main__":main()
