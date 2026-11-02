#!/usr/bin/env python3
"""Predicted-risk hierarchical and one-sided conformal intervals (research-only).

Single disjoint split: pre-2019 point-model training, FY2019 residual calibration,
FY2020-2024 untouched by fit or calibration. Split by *predicted* dollar band,
not true funding band, which would be unavailable at inference.

Hierarchy:
  fine predicted amount band -> coarse predicted <1M vs >=1M -> pooled
  A group is used only with >=15 calibration observations, otherwise fallback.
This pooled fallback is not a formal finite-sample conditional guarantee.
Future-year non-exchangeability and current FEMA record-version drift remain.

No accepted point prediction model/router changed.
"""
from pathlib import Path
import math,json
import numpy as np
import pandas as pd
import early_response_912_t0_t72_forward as base
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"audit_outputs"/"conformal_hierarchical_upper"
OUT.mkdir(parents=True,exist_ok=True)
LEVELS=[.8,.9]
ARMS=["metadata","t72_missions_nhc72"]
MIN_GROUP=15
METHODS=["pooled_symmetric","hierarchical_symmetric","pooled_upper","hierarchical_upper"]
def fine(x):
    return ["0-100K","100K-1M","1M-50M","50M+"][np.searchsorted([0,1e5,1e6,50e6,np.inf],max(0,float(x)),side="right")-1]
def broad(x):
    return "pred_below_1M" if float(x)<1e6 else "pred_1M_plus"
def get_q(scores,level):
    v=np.sort(np.asarray(scores,float))
    k=int(math.ceil((len(v)+1)*level))
    return (float(v[k-1]) if k<=len(v) else float("inf"),k)
def calibrator(cal,method,level):
    kind="upper" if method.endswith("_upper") else "symmetric"
    hierarchy=method.startswith("hierarchical")
    c=cal.copy()
    c["fine"]=c.point.map(fine)
    c["broad"]=c.point.map(broad)
    c["resid"]=c.target-c.point if kind=="upper" else (c.target-c.point).abs()
    stats={}
    q_pool,k=get_q(c.resid,level)
    stats["pooled"]={"n":len(c),"q":q_pool,"rank":k}
    for groupname,col in [("fine","fine"),("broad","broad")]:
        for key,g in c.groupby(col):
            if len(g)<MIN_GROUP:continue
            q,k=get_q(g.resid,level)
            stats[f"{groupname}:{key}"]={"n":len(g),"q":q,"rank":k}
    def choose(point):
        if hierarchy:
            for key in [f"fine:{fine(point)}",f"broad:{broad(point)}"]:
                if key in stats:
                    return stats[key]["q"],key,stats[key]["n"]
        return q_pool,"pooled",len(c)
    return choose,stats
def estimate(point,q,kind):
    if kind=="upper":
        return 0.,float(max(0,point+q))
    return max(0,float(point-q)),float(point+q)
def evaluate(g,upper=False):
    y=g.target.to_numpy(float)
    h=g.upper.to_numpy(float)
    low=g.lower.to_numpy(float)
    miss=y>h
    if upper:
        return {"n":len(g),"upper_coverage":float(np.mean(~miss)),
          "above_upper":int(np.sum(miss)),"mean_upper_M":float(np.mean(h)/1e6),
          "median_upper_M":float(np.median(h)/1e6),
          "median_headroom_M":float(np.median(h-g.point.to_numpy(float))/1e6),
          "group_fallback_rate":float(np.mean(g.calibration_group=="pooled"))}
    return {"n":len(g),"interval_coverage":float(np.mean((y>=low)&(y<=h))),
      "above_upper":int(np.sum(miss)),"below_lower":int(np.sum(y<low)),
      "mean_width_M":float(np.mean(h-low)/1e6),
      "median_width_M":float(np.median(h-low)/1e6),
      "group_fallback_rate":float(np.mean(g.calibration_group=="pooled"))}
def run():
    d=base.load()
    train=d[d.fyDeclared<=2018]
    cal=d[d.fyDeclared==2019]
    test=d[d.fyDeclared.between(2020,2024)]
    assert (len(train),len(cal),len(test))==(536,68,308)
    predictions=[]
    fits=[]
    for arm in ARMS:
        pc,_=base.evaluate_fold(train,cal,arm,2019)
        pt,_=base.evaluate_fold(train,test,arm,2019)
        c=cal[["disasterNumber","target"]].copy()
        c["point"]=pc
        t=test[["disasterNumber","target","fyDeclared","incidentType","band"]].copy()
        t["point"]=pt
        t["predicted_fine_band"]=t.point.map(fine)
        t["predicted_coarse_band"]=t.point.map(broad)
        fits.append({"variant":arm,"training_n":len(train),"calibration_n":len(cal),
          "test_n":len(test),"calibration_predictions_min":float(c.point.min()),
          "calibration_predictions_max":float(c.point.max()),
          "calibration_fine_counts":c.point.map(fine).value_counts().to_dict(),
          "calibration_coarse_counts":c.point.map(broad).value_counts().to_dict()})
        for method in METHODS:
            kind="upper" if method.endswith("_upper") else "symmetric"
            for level in LEVELS:
                use,calibration_groups=calibrator(c,method,level)
                for _,r in t.iterrows():
                    q,key,n=use(r.point)
                    lower,upper=estimate(float(r.point),q,kind)
                    predictions.append({"variant":arm,"method":method,"level":level,
                       "disasterNumber":int(r.disasterNumber),"fyDeclared":int(r.fyDeclared),
                       "incidentType":r.incidentType,"true_band":r.band,
                       "predicted_fine_band":r.predicted_fine_band,
                       "predicted_coarse_band":r.predicted_coarse_band,
                       "target":float(r.target),"point":float(r.point),
                       "lower":lower,"upper":upper,"width":upper-lower,
                       "quantile":q,"calibration_group":key,"calibration_group_n":n,
                       "covered":int(lower<=r.target<=upper),
                       "upper_covers":int(r.target<=upper)})
    result=pd.DataFrame(predictions)
    assert len(result)==len(test)*len(ARMS)*len(METHODS)*len(LEVELS)
    summary=[]
    band=[]
    hazard=[]
    yearly=[]
    groups=[]
    for keys,g in result.groupby(["variant","method","level"]):
        values=dict(zip(["variant","method","level"],keys))
        summary.append({**values,**evaluate(g,keys[1].endswith("_upper"))})
        for col,output in [("true_band",band),("incidentType",hazard),
                           ("fyDeclared",yearly),("calibration_group",groups)]:
            for name,part in g.groupby(col):
                output.append({**values,col:name,**evaluate(part,keys[1].endswith("_upper"))})
    # Pre-specified nested upper & symmetric checks.
    for (_,method),group in result.groupby(["variant","method"]):
        a=group[group.level==.8].set_index("disasterNumber").sort_index()
        b=group[group.level==.9].set_index("disasterNumber").sort_index()
        assert a.index.equals(b.index)
        assert (b.upper>=a.upper-1e-5).all()
        assert (b.lower<=a.lower+1e-5).all()
    obj={"purpose":"Group-aware / one-sided split-conformal fixed-model diagnostic",
      "training":[2010,2018],"calibration":[2019,2019],"testing":[2020,2024],
      "n_test":308,"calibration_n":68,"group_min_n":MIN_GROUP,
      "group_policy":"fine point-estimate band >=15, else broad point estimate under/above $1M >=15, else pooled; no true-band calibration group at inference.",
      "finite_quantile":"ceil((n_cal+1)*coverage) order statistic",
      "models":fits,"overall":summary,"by_true_band":band,"by_hazard":hazard,
      "by_year":yearly,"by_calibration_group":groups,
      "caveats":["Conditional coverage not guaranteed by hierarchical fallback or chronological drift.",
        "Point model uses current FEMA records with historical action-time filters, not original versioned snapshot.",
        "Only 68 calibration cases and 9 catastrophic test disasters.",
        "Predicted groups are observable at inference; true band only evaluated post hoc.",
        "No CQR implemented yet, and latest Track A model is not modified.",
        "One-sided upper bounds are ranges for historical obligated dollars, not humanitarian allocation advice."]}
    result.to_csv(OUT/"all_case_intervals.csv",index=False)
    pd.DataFrame(summary).to_csv(OUT/"overall.csv",index=False)
    pd.DataFrame(band).to_csv(OUT/"coverage_by_true_band.csv",index=False)
    pd.DataFrame(groups).to_csv(OUT/"coverage_by_calibration_group.csv",index=False)
    pd.DataFrame(hazard).to_csv(OUT/"coverage_by_incident_type.csv",index=False)
    pd.DataFrame(yearly).to_csv(OUT/"coverage_by_year.csv",index=False)
    (OUT/"summary.json").write_text(json.dumps(obj,indent=2,default=str))
    print("BEGIN_HIERARCHICAL_CONFORMAL_RESULTS")
    print(json.dumps({"models":fits,"overall":summary,
      "high_band_90pct":[x for x in band if x["level"]==.9 and x["true_band"] in
       ["50M-200M","200M-500M","500M+"]]},indent=2))
    print("END_HIERARCHICAL_CONFORMAL_RESULTS")
if __name__=="__main__":run()
