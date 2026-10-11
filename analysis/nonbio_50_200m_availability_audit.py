#!/usr/bin/env python3
"""Locked-architecture sensitivity tests for the $50M-$200M candidate.
No fitting to held-out-year targets; no router modifications; fixed k=5, alpha=.75.
These tests are exploratory checks, NOT selection of a new deployment winner.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from nonbio_50_200m_fast_hybrid import load, LO, HI

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"audit_outputs"/"nonbio_50_200m_availability_audit"
FROZEN=ROOT/"audit_inputs"/"50_200m_ml_specialist"/"accepted"/"candidate_predictions.csv"
K=5
ALPHA=.75
SETS={
 "full_operational": ["durationDays","declarationDelayDays","expectedResourceScore","missionAssignmentCount",
  "uniqueAgencyCount","uniqueMaTypeCount","uniquePriorityCount","responseComplexityScore",
  "missionDensity","agencyDensity"],
 "without_mission": ["durationDays","declarationDelayDays","expectedResourceScore"],
 "declaration_proxy": ["declarationDelayDays","ihProgramDeclared","paProgramDeclared","hmProgramDeclared"]
}

def analogue_fixed(train, tests, cols):
    pool=train[(train.target_m>=10)&(train.target_m<500)].copy()
    assert len(pool)>K, f"Only {len(pool)} training analogues"
    A=pool[cols].copy().apply(pd.to_numeric,errors="coerce")
    B=tests[cols].copy().apply(pd.to_numeric,errors="coerce")
    # All missing-value estimates and scales are learned using training alone.
    med=A.median().fillna(0)
    A=A.fillna(med).to_numpy(float)
    B=B.fillna(med).to_numpy(float)
    scaler=RobustScaler(quantile_range=(10,90)).fit(A)
    A=np.clip(scaler.transform(A),-8,8)
    B=np.clip(scaler.transform(B),-8,8)
    dist=np.mean((B[:,None,:]-A[None,:,:])**2,axis=2)
    dist+=2.0*(tests.incidentType.to_numpy()[:,None]!=pool.incidentType.to_numpy()[None,:])
    dist+=0.3*(tests.state.to_numpy()[:,None]!=pool.state.to_numpy()[None,:])
    y=pool.target_m.to_numpy(float)
    z=[]
    for row in dist:
        indices=np.argsort(row,kind="stable")[:K]
        weights=1.0/(0.2+np.sqrt(row[indices]))
        z.append(float(np.average(y[indices],weights=weights)))
    return np.clip(z,LO,HI-1e-6)

def metric(actual,pred):
    y=np.asarray(actual,float); z=np.asarray(pred,float)
    a=np.abs(y-z)
    return {"n":int(len(y)),"MAE_M":float(a.mean()),"RMSE_M":float(np.sqrt(np.mean((y-z)**2))),
     "MedAE_M":float(np.median(a)),
     "R2":float(1-((y-z)**2).sum()/((y-y.mean())**2).sum()) if len(y)>1 and np.var(y)>0 else None,
     "log_MAE":float(np.mean(abs(np.log1p(y*1e6)-np.log1p(z*1e6)))),
     "within20":float(np.mean(a/np.maximum(y,1e-9)<=.20)),
     "within30":float(np.mean(a/np.maximum(y,1e-9)<=.30)),
     "within50":float(np.mean(a/np.maximum(y,1e-9)<=.50)),
     "distinct_usd":int(len(np.unique(np.round(z*1e6))))}

def run():
    d=load()
    f=pd.read_csv(FROZEN)
    assert len(f)==912 and int((f.actual_band.astype(str)==f.candidate_pred_new.astype(str)).sum())==830
    routed=f.loc[f.candidate_pred_new.astype(str)=="50-200M","disasterNumber"].astype(int)
    assert len(routed)==13
    d["is_routed"]=d.disasterNumber.astype(int).isin(routed)
    direct=[];assigned=[];forward=[]
    band=d[(d.target_m>=LO)&(d.target_m<HI)]
    for fy in sorted(band.fyDeclared.unique()):
        train=d[d.fyDeclared!=fy]
        test=d[(d.fyDeclared==fy)&(d.target_m>=LO)&(d.target_m<HI)]
        routed_test=d[(d.fyDeclared==fy)&d.is_routed]
        assert len(train[(train.target_m>=LO)&(train.target_m<HI)])>0
        median=float(train.loc[(train.target_m>=LO)&(train.target_m<HI),"target_m"].median())
        for setting,cols in SETS.items():
            for kind,te,dest in [("direct",test,direct),("routed",routed_test,assigned)]:
                if te.empty:continue
                # Target mask to ensure predictions cannot reference test amounts.
                masked=te.copy();masked["target_m"]=np.nan
                pred=analogue_fixed(train,masked,cols)
                pred=np.clip(median+ALPHA*(pred-median),LO,HI-1e-6)
                for (_,rec),z in zip(te.iterrows(),pred):
                    dest.append({"fy":int(fy),"disasterNumber":int(rec.disasterNumber),"kind":kind,"features":setting,
                                 "actual_m":float(rec.target_m),"median_m":median,"hybrid_m":float(z)})
    # More demanding forward-time stress test: only years STRICTLY PRIOR to the evaluated FY.
    # Earliest F.Y. without prior examples cannot be scored in a target-band median comparison.
    for fy in sorted(band.fyDeclared.unique()):
        train=d[d.fyDeclared<fy]
        te=d[(d.fyDeclared==fy)&(d.target_m>=LO)&(d.target_m<HI)]
        prior_band=train[(train.target_m>=LO)&(train.target_m<HI)]
        if prior_band.empty:continue
        med=float(prior_band.target_m.median())
        for setting,cols in SETS.items():
            masked=te.copy();masked["target_m"]=np.nan
            pred=analogue_fixed(train,masked,cols)
            pred=np.clip(med+ALPHA*(pred-med),LO,HI-1e-6)
            for (_,rec),z in zip(te.iterrows(),pred):
                forward.append({"fy":int(fy),"disasterNumber":int(rec.disasterNumber),"kind":"forward",
                                "features":setting,"actual_m":float(rec.target_m),
                                "median_m":med,"hybrid_m":float(z),"prior_band_n":int(len(prior_band))})
    frames={"direct":pd.DataFrame(direct),"routed":pd.DataFrame(assigned),"forward":pd.DataFrame(forward)}
    metrics={}
    for name,frame in frames.items():
        metrics[name]={}
        for setting,g in frame.groupby("features"):
            metrics[name][setting]={"baseline":metric(g.actual_m,g.median_m),
                                   "individualized":metric(g.actual_m,g.hybrid_m)}
            print(name,setting, "n=",len(g),"median_mae=",metrics[name][setting]["baseline"]["MAE_M"],
                  "candidate_mae=",metrics[name][setting]["individualized"]["MAE_M"],flush=True)
    # Compare early-only and all-operational on the same held-out FY and same 13 target cases.
    assert frames["direct"].groupby("features").size().eq(13).all()
    assert frames["routed"].groupby("features").size().eq(13).all()
    OUTPUT={"method":"Fixed k=5/alpha=.75; years held out; no per-experiment selection; training-target-bounded analogues",
            "scope":"Research sensitivity only; prospective feature availability must be documented",
            "feature_sets":SETS,
            "warning":"Outer LFYO includes later fiscal years in training; forward test excludes future data but covers fewer cases.",
            "metrics":metrics,
            "forward_scored_unique_disasters":int(frames["forward"].disasterNumber.nunique())}
    OUT.mkdir(parents=True,exist_ok=True)
    for name,frame in frames.items():frame.to_csv(OUT/f"{name}_predictions.csv",index=False)
    (OUT/"summary.json").write_text(json.dumps(OUTPUT,indent=2),encoding="utf-8")
    print("BEGIN_AVAILABILITY_AUDIT",flush=True)
    print(json.dumps(OUTPUT,indent=2),flush=True)
    print("END_AVAILABILITY_AUDIT",flush=True)

if __name__=="__main__":run()
