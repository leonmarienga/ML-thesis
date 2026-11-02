#!/usr/bin/env python3
"""Decision usefulness audit of EXISTING CQR and split-conformal predictions.

This file does NOT train/recalibrate/select interval models or use test funding
amounts for cutoff choices. It audits immutable Track B artifacts from:
  hierarchical: GitHub Actions 41396350227
  CQR:          GitHub Actions 41397857911
and compares test IDs and final outcomes against reconstructed accepted Track A:
  exact Track A: GitHub Actions 41397678781

FY2010-2018 train, FY2019 calibrate, FY2020-2024 test, 308 cases.
Tail and warning thresholds: $50M, $200M and $500M (existing band edges).
A warning upper>=threshold is a descriptive sensitivity flag, NOT a funding
allocation instruction. No model tuned on the 9 catastrophic test disasters.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
INP=ROOT/"audit_inputs"/"uncertainty_decision_audit"
OUT=ROOT/"audit_outputs"/"uncertainty_decision_audit"
OUT.mkdir(parents=True,exist_ok=True)
METHOD_LABELS={
  "pooled_symmetric":"Pooled conformal two-sided",
  "hierarchical_symmetric":"Hierarchical conformal two-sided",
  "pooled_upper":"Pooled conformal upper",
  "hierarchical_upper":"Hierarchical conformal upper",
  "CQR":"Conformalized quantile regression"}
LEVELS=[.8,.9]
THRESHOLDS=[50e6,200e6,500e6]
WIDTH_THRESHOLDS=[1e6,10e6,50e6,200e6,500e6]
CATASTROPHIC=50e6

def wilson(num,total,z=1.95996398454):
    if not total:return [None,None]
    p=num/total
    d=1+z*z/total
    center=(p+z*z/(2*total))/d
    half=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [float(max(0,center-half)),float(min(1,center+half))]

def clean():
    h=pd.read_csv(INP/"hierarchical"/"all_case_intervals.csv")
    c=pd.read_csv(INP/"cqr"/"cqr_case_intervals.csv")
    track=pd.read_csv(INP/"accepted"/"all_912_immutable_overlay.csv")
    assert len(track)==912 and track.disasterNumber.nunique()==912
    assert len(h)==308*4*2*2 and len(c)==308*2
    h=h[h.variant=="t72_missions_nhc72"].copy()
    h=h.rename(columns={"target":"actual","true_band":"actual_band"})
    h["method_id"]=h.method
    c=c.rename(columns={"true_band":"actual_band"})
    c["method_id"]="CQR"
    cols=["disasterNumber","fyDeclared","incidentType","actual_band",
          "level","method_id","actual","lower","upper","width"]
    d=pd.concat([h[cols],c[cols]],ignore_index=True)
    assert len(d)==308*5*2
    for (level,method),group in d.groupby(["level","method_id"]):
        assert len(group)==308 and group.disasterNumber.nunique()==308,(level,method)
        assert group.fyDeclared.between(2020,2024).all()
        assert np.isfinite(group[["actual","lower","upper","width"]].to_numpy(float)).all()
        assert (group.lower>=0).all() and (group.upper>=group.lower).all()
        assert np.allclose(group.width,group.upper-group.lower,atol=.001)
    accepted=track[["disasterNumber","target","state","router_pred"]]
    assert accepted.disasterNumber.nunique()==912
    d=d.merge(accepted,on="disasterNumber",how="left",validate="many_to_one")
    assert d.target.notna().all()
    assert np.allclose(d.actual.to_numpy(float),d.target.to_numpy(float),atol=.01,rtol=0)
    # Audit only. Accepted Track A is NOT used to calibrate Track B intervals.
    d=d.drop(columns="target")
    same=d.sort_values(["method_id","level","disasterNumber"])
    assert same["method_id"].nunique()==5 and same["level"].nunique()==2
    d["is_two_sided"]=~d.method_id.str.endswith("_upper")
    d["covered_two_sided"]=(d.lower<=d.actual)&(d.actual<=d.upper)
    d["upper_covers"]=d.actual<=d.upper
    d["shortfall"]=np.maximum(d.actual-d.upper,0.0)
    d["is_catastrophic"]=d.actual>=CATASTROPHIC
    d["method_label"]=d.method_id.map(METHOD_LABELS)
    assert d.method_label.notna().all()
    return d,track

def aggregate(g):
    if g.empty:
        return {"n":0}
    n=len(g)
    correct=(g.covered_two_sided if bool(g.is_two_sided.iloc[0]) else g.upper_covers)
    hits=int(correct.sum())
    tail=g[g.is_catastrophic]
    tail_cover=(tail.covered_two_sided if bool(g.is_two_sided.iloc[0]) else tail.upper_covers)
    return {
      "n":int(n),
      "coverage_type":"two-sided" if bool(g.is_two_sided.iloc[0]) else "upper",
      "coverage":float(correct.mean()),
      "coverage_Wilson_95_lower":wilson(hits,n)[0],
      "coverage_Wilson_95_upper":wilson(hits,n)[1],
      "above_upper_count":int((g.shortfall>0).sum()),
      "upper_shortfall_total_M":float(g.shortfall.sum()/1e6),
      "upper_shortfall_max_M":float(g.shortfall.max()/1e6),
      "median_width_M":float(g.width.median()/1e6) if g.is_two_sided.iloc[0] else None,
      "mean_width_M":float(g.width.mean()/1e6) if g.is_two_sided.iloc[0] else None,
      "p90_width_M":float(g.width.quantile(.9)/1e6) if g.is_two_sided.iloc[0] else None,
      "p95_width_M":float(g.width.quantile(.95)/1e6) if g.is_two_sided.iloc[0] else None,
      "median_upper_M":float(g.upper.median()/1e6),
      "p90_upper_M":float(g.upper.quantile(.9)/1e6),
      "catastrophic_n":int(len(tail)),
      "catastrophic_covered":int(tail_cover.sum()),
      "catastrophic_coverage":float(tail_cover.mean()) if len(tail) else None,
      "catastrophic_Wilson_95":wilson(int(tail_cover.sum()),len(tail)),
      "catastrophic_above_upper_n":int((tail.shortfall>0).sum()),
      "catastrophic_shortfall_total_M":float(tail.shortfall.sum()/1e6),
      "catastrophic_shortfall_max_M":float(tail.shortfall.max()/1e6),
      "catastrophic_median_width_M":float(tail.width.median()/1e6) if len(tail) and g.is_two_sided.iloc[0] else None,
      "catastrophic_median_upper_M":float(tail.upper.median()/1e6) if len(tail) else None,
      **{f"interval_width_gt_{int(t/1e6)}M_n":int((g.width>t).sum()) if g.is_two_sided.iloc[0] else None
         for t in WIDTH_THRESHOLDS},
    }

def warning_table(g,threshold):
    alert=g.upper>=threshold
    actual=g.actual>=threshold
    tp=int((alert&actual).sum())
    fp=int((alert&~actual).sum())
    fn=int((~alert&actual).sum())
    tn=int((~alert&~actual).sum())
    return {
      "threshold_M":threshold/1e6,"n":len(g),
      "true_above_threshold":int(actual.sum()),"flagged":int(alert.sum()),
      "true_positives":tp,"false_positives":fp,
      "false_negatives":fn,"true_negatives":tn,
      "sensitivity":tp/(tp+fn) if tp+fn else None,
      "precision":tp/(tp+fp) if tp+fp else None,
      "alert_rate":float(alert.mean()),
      "false_alert_fraction":fp/(tp+fp) if tp+fp else None,
      "not_flagged_but_true_high":fn}

def main():
    d,accepted=clean()
    summary=[]
    by_band=[]
    by_hazard=[]
    by_year=[]
    warning=[]
    for (method,level),g in d.groupby(["method_id","level"]):
        k={"method_id":method,"method":METHOD_LABELS[method],"nominal_coverage":float(level)}
        summary.append({**k,**aggregate(g)})
        for col,target in [("actual_band",by_band),("incidentType",by_hazard),
                           ("fyDeclared",by_year)]:
            for name,part in g.groupby(col):
                target.append({**k,col:name,**aggregate(part)})
        for threshold in THRESHOLDS:
            warning.append({**k,**warning_table(g,threshold)})
    overall=pd.DataFrame(summary)
    flags=pd.DataFrame(warning)
    tail=d[(d.level==.9)&(d.is_catastrophic)].copy()
    assert tail.disasterNumber.nunique()==9 and len(tail)==9*5
    wide=tail[["disasterNumber","fyDeclared","state","incidentType","actual_band",
               "actual","method_id","upper","width","shortfall","upper_covers"]].copy()
    wide=wide.sort_values(["disasterNumber","method_id"])
    # Check all 5 interval methods refer to the same disaster and target.
    for disaster,grp in wide.groupby("disasterNumber"):
        assert len(grp)==5 and grp.actual.nunique()==1,(disaster,len(grp))
    source={
      "Track_A_exact_rows":int(len(accepted)),
      "Track_A_router_correct":int((accepted.actual_band==accepted.router_pred).sum()),
      "Track_A_verified_MAE":float((accepted.target-accepted.challenger_stack_prediction).abs().mean()),
      "Track_A_not_used_for_Track_B_intervals":True,
      "Track_B_test_rows":308,
      "Track_B_test_years":[2020,2021,2022,2023,2024],
      "existing_interval_variants":list(METHOD_LABELS),
      "new_interval_fit_or_calibration":False,
      "true_high_value_test_rows":9,
      "warning_cutoffs_M":[50,200,500],
      "level_values":LEVELS,
      "source_GitHub_runs":{"hierarchical":"41396350227",
                            "CQR":"41397857911","exact Track A":"41397678781"},
      "comparability":"Same 308 test IDs/targets, but CQR uses its own separately trained point/quantile models.",
      "limitations":[
        "True funding bands are only for post hoc auditing, not for constructing prediction groups.",
        "Original FEMA data snapshots and event identity are not fully time-version authenticated.",
        "All methods were previously evaluated on the same 9 large disasters; diagnostic audit only, no fresh independent holdout.",
        "Extreme cases can share meteorological events; Wilson intervals ignore such clustering.",
        "The target is historical obligations, not actual humanitarian needs.",
        "Threshold flagging is not an allocation recommendation or fully specified decision policy."
      ]}
    d.to_csv(OUT/"all_case_decision_usefulness.csv",index=False)
    overall.to_csv(OUT/"method_summary.csv",index=False)
    flags.to_csv(OUT/"threshold_warning_metrics.csv",index=False)
    pd.DataFrame(by_band).to_csv(OUT/"coverage_by_true_band.csv",index=False)
    pd.DataFrame(by_hazard).to_csv(OUT/"coverage_by_hazard.csv",index=False)
    pd.DataFrame(by_year).to_csv(OUT/"coverage_by_year.csv",index=False)
    wide.to_csv(OUT/"nine_high_value_case_shortfalls.csv",index=False)
    (OUT/"provenance_and_limitations.json").write_text(json.dumps(source,indent=2))
    print("BEGIN_DECISION_USEFULNESS_AUDIT")
    print(json.dumps({"source":source,
       "overall90":[x for x in summary if x["nominal_coverage"]==.9],
       "threshold90_50M":[x for x in warning if x["nominal_coverage"]==.9 and x["threshold_M"]==50],
       "threshold90_500M":[x for x in warning if x["nominal_coverage"]==.9 and x["threshold_M"]==500]
    },indent=2))
    print("END_DECISION_USEFULNESS_AUDIT")

if __name__=="__main__":
    main()
