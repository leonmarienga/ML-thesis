#!/usr/bin/env python3
"""Immutable-artifact overlay: frozen six-band router, accepted baseline stack,
accepted two low-band specialists, and one RESEARCH-ONLY $50M-$200M candidate.

No models are retrained; no router or baseline results are modified.
All four inputs are OUTER leave-fiscal-year-out prediction artifacts.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/"audit_inputs"/"nonbio_final_50_200_overlay"
OUT=ROOT/"audit_outputs"/"nonbio_final_50_200_overlay"
BASE=INPUT/"baseline"/"predictions.csv"
LOW0=INPUT/"low0"/"routed_predictions.csv"
LOW1=INPUT/"low1"/"routed_predictions.csv"
MID=INPUT/"mid"/"routed_predictions.csv"
BANDS=["0-100K","100K-1M","1M-50M","50-200M","200-500M","500M+"]
ROUTE_SIZES={"0-100K":686,"100K-1M":130,"50-200M":13}

def numbers(x):
    return pd.to_numeric(x,errors="raise").to_numpy(float)

def measure(a,p):
    y=numbers(a);q=numbers(p)
    diff=np.abs(y-q)
    pos=y>0
    return {
       "n":int(len(y)),
       "MAE_USD":float(diff.mean()),
       "RMSE_USD":float(np.sqrt(np.mean((y-q)**2))),
       "MedAE_USD":float(np.median(diff)),
       "R2":float(1-np.sum((y-q)**2)/np.sum((y-np.mean(y))**2)) if len(y)>1 and np.var(y)>0 else None,
       "log_MAE":float(np.abs(np.log1p(y)-np.log1p(q)).mean()),
       "log_R2":float(1-np.sum((np.log1p(y)-np.log1p(q))**2)/np.sum((np.log1p(y)-np.log1p(y).mean())**2)) if len(y)>1 and np.var(np.log1p(y))>0 else None,
       "within20_positive":float(np.mean((diff[pos]/y[pos])<=.2)) if np.any(pos) else None,
       "within30_positive":float(np.mean((diff[pos]/y[pos])<=.3)) if np.any(pos) else None,
       "within50_positive":float(np.mean((diff[pos]/y[pos])<=.5)) if np.any(pos) else None,
    }

def unique_ids(data,label):
    assert len(data)==data.disasterNumber.nunique(),f"{label}: repeated disaster IDs"
    assert data.disasterNumber.notna().all()
    data["disasterNumber"]=pd.to_numeric(data.disasterNumber,errors="raise").astype(int)
    return data

def inject(base, specialist, band, column, label, multiplier=1.0):
    specialist=unique_ids(specialist,label)
    candidates=base.loc[base.router_pred==band,["disasterNumber","fyDeclared","target"]]
    assert len(candidates)==ROUTE_SIZES[band],f"Unexpected routed count {band}: {len(candidates)}"
    assert len(specialist)==len(candidates),f"{label} count does not match frozen router"
    assert set(candidates.disasterNumber)==set(specialist.disasterNumber), f"{label} routed ID disagreement"
    joined=candidates.merge(specialist,on="disasterNumber",how="inner",validate="one_to_one",suffixes=("","_specialist"))
    assert len(joined)==len(candidates)
    assert (pd.to_numeric(joined.fyDeclared)==pd.to_numeric(joined.fyDeclared_specialist)).all(),f"{label} FY mismatch"
    assert np.allclose(numbers(joined.target),numbers(joined["actual" if "actual" in joined.columns else "actual_m"])*(1.0 if "actual" in joined.columns else 1e6),rtol=0,atol=0.05),f"{label} actual funding mismatch"
    vals=pd.Series(numbers(specialist[column])*multiplier,index=specialist.disasterNumber).reindex(base.disasterNumber)
    assert np.isfinite(vals[base.router_pred==band]).all(),f"{label} missing amount"
    lo,hi={"0-100K":(0,1e5),"100K-1M":(1e5,1e6),"50-200M":(50e6,200e6)}[band]
    zz=vals[base.router_pred==band]
    assert (zz>=lo-0.01).all() and (zz<=hi+0.01).all(),f"{label} out-of-band predictions"
    return vals

def main():
    base=unique_ids(pd.read_csv(BASE),"baseline")
    assert len(base)==912, len(base)
    assert int((base.router_pred==base.actual_band).sum())==830
    assert base.router_pred.isin(BANDS).all()
    for band,count in ROUTE_SIZES.items():assert int((base.router_pred==band).sum())==count
    base=base.sort_values("disasterNumber").reset_index(drop=True)

    low0=inject(base,pd.read_csv(LOW0),"0-100K","selected_ml_prediction","0-100K")
    low1=inject(base,pd.read_csv(LOW1),"100K-1M","selected_ml_prediction","100K-1M")
    mid=inject(base,pd.read_csv(MID),"50-200M","hybrid_m","50-200M",multiplier=1e6)

    base["frozen_stack_prediction"]=numbers(base.accepted_stack_prediction)
    base["accepted_low_band_prediction"]=base.frozen_stack_prediction.copy()
    base.loc[base.router_pred=="0-100K","accepted_low_band_prediction"]=low0[base.router_pred=="0-100K"].to_numpy()
    base.loc[base.router_pred=="100K-1M","accepted_low_band_prediction"]=low1[base.router_pred=="100K-1M"].to_numpy()
    base["challenger_stack_prediction"]=base.accepted_low_band_prediction.copy()
    base.loc[base.router_pred=="50-200M","challenger_stack_prediction"]=mid[base.router_pred=="50-200M"].to_numpy()

    no_change=~base.router_pred.isin(["0-100K","100K-1M","50-200M"])
    assert np.array_equal(base.loc[no_change,"frozen_stack_prediction"],base.loc[no_change,"challenger_stack_prediction"])
    assert np.array_equal(base.loc[base.router_pred!="50-200M","accepted_low_band_prediction"],base.loc[base.router_pred!="50-200M","challenger_stack_prediction"])
    assert int(np.sum(base.accepted_low_band_prediction!=base.challenger_stack_prediction))==13, "Not exactly 13 updated rows"

    base["delta_abs_error_from_mid"]=abs(base.target-base.accepted_low_band_prediction)-abs(base.target-base.challenger_stack_prediction)
    target=base[base.router_pred=="50-200M"]
    gains={
        "improved_routed_cases":int((target.delta_abs_error_from_mid>0).sum()),
        "worsened_routed_cases":int((target.delta_abs_error_from_mid<0).sum()),
        "absolute_error_saved_on_13_USD":float(target.delta_abs_error_from_mid.sum()),
        "global_MAE_saved_USD":float(target.delta_abs_error_from_mid.sum()/912),
        "only_13_predictions_changed":True,
        "frozen_router_correct":830,
        "frozen_router_total":912,
    }
    summary={
       "provenance":{
          "baseline_stack":"run 34796746603, artifact nonbio-final-830-accepted-amount-stack",
          "0-100K":"run 38153347912, artifact nonbio-0-100k-dedicated-ml-specialist",
          "100K-1M":"run 38152554081, artifact nonbio-100k-1m-dedicated-ml-specialist",
          "50-200M":"run 38162966192, artifact nonbio-50-200m-routed-acceptance-audit",
          "study_note":"Frozen historical/router predictions; not a newly independent holdout."
       },
       "overall":{
          "frozen_original_stack":measure(base.target,base.frozen_stack_prediction),
          "accepted_two_low_specialists":measure(base.target,base.accepted_low_band_prediction),
          "candidate_with_50_200M_hybrid":measure(base.target,base.challenger_stack_prediction)},
       "routed_50_200M":{
          "accepted_training_median":measure(target.target,target.accepted_low_band_prediction),
          "individualized_hybrid":measure(target.target,target.challenger_stack_prediction)},
       "by_actual_band":{},
       "by_routed_band":{},
       "delta":gains
    }
    for band in BANDS:
        for label,key in [("by_actual_band","actual_band"),("by_routed_band","router_pred")]:
            d=base[base[key]==band]
            summary[label][band]={
                 "n":len(d),
                 "before":measure(d.target,d.accepted_low_band_prediction),
                 "after":measure(d.target,d.challenger_stack_prediction)}
    OUT.mkdir(parents=True,exist_ok=True)
    cols=["disasterNumber","fyDeclared","state","incidentType","target","actual_band","router_pred","frozen_stack_prediction","accepted_low_band_prediction","challenger_stack_prediction","delta_abs_error_from_mid"]
    base[cols].to_csv(OUT/"all_912_immutable_overlay.csv",index=False)
    target[cols].to_csv(OUT/"routed_13_changes.csv",index=False)
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("BEGIN_IMMUTABLE_OVERLAY_RESULTS",flush=True)
    print(json.dumps(summary,indent=2),flush=True)
    print("END_IMMUTABLE_OVERLAY_RESULTS",flush=True)

if __name__=="__main__":main()
