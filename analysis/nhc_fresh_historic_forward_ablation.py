#!/usr/bin/env python3
"""Fixed, comparable *forward-fiscal-year* NHC T0 feature ablation.

Uses all-tropical (123 cases) outcome-blind archive features, and existing frozen
830/912 router to evaluate 50-200M direct/routed specialist cases.
No model is replaced. This is NOT a declaration-time evaluation of the FULL
system: baseline operational inputs may have been updated after declaration,
and cyclone identity comes from current FEMA titles.
"""
import json, math
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import nonbio_50_200m_fast_hybrid as old

ROOT=Path(__file__).resolve().parents[1]
IN=ROOT/"audit_inputs"/"nhc_feature_ablation"
OUT=ROOT/"audit_outputs"/"nhc_fresh_vs_historic"
OUT.mkdir(parents=True,exist_ok=True)
FRESH_HOURS=72  # Pre-specified using source-timing reasoning, not funded test targets.
K=5
ALPHA=0.75
WEIGHT=0.25  # Experimental fixed sensitivity weight; chosen based on prior research history.
SUPPORT_LOW,SUPPORT_HIGH=10.0,500.0
B1,B2=50.0,200.0
OP=[
    "durationDays","declarationDelayDays","expectedResourceScore",
    "missionAssignmentCount","uniqueAgencyCount","uniqueMaTypeCount",
    "uniquePriorityCount","responseComplexityScore","missionDensity",
    "agencyDensity"
]
RAW=["nhc_t0_wind_mph","nhc_t0_pressure_mb","nhc_t0_hours_before_cutoff"]
VARIANTS=["operational","recent_72h","historical_archive"]
def tropical(x):
    return str(x).strip().lower() in {"hurricane","tropical storm"}
def qscale(pool,test,cols):
    a=pool[cols].apply(pd.to_numeric,errors="coerce")
    b=test[cols].apply(pd.to_numeric,errors="coerce")
    med=a.median().fillna(0.0)
    x=a.fillna(med).to_numpy(float)
    y=b.fillna(med).to_numpy(float)
    norm=RobustScaler(quantile_range=(10,90)).fit(x)
    return np.clip(norm.transform(x),-8,8),np.clip(norm.transform(y),-8,8)
def operator_dist(pool,test):
    a,b=qscale(pool,test,OP)
    d=np.mean((b[:,None,:]-a[None,:,:])**2,axis=2)
    d+=2.0*(test.incidentType.astype(str).to_numpy()[:,None]!=pool.incidentType.astype(str).to_numpy()[None,:])
    d+=0.30*(test.state.astype(str).to_numpy()[:,None]!=pool.state.astype(str).to_numpy()[None,:])
    return d
def source_dist(pool,test,variant):
    if variant=="operational":
        return np.zeros((len(test),len(pool)),float),np.zeros((len(test),len(pool)),bool)
    a=pool[RAW].apply(pd.to_numeric,errors="coerce").copy()
    b=test[RAW].apply(pd.to_numeric,errors="coerce").copy()
    ha=a.nhc_t0_hours_before_cutoff
    hb=b.nhc_t0_hours_before_cutoff
    # No negative advisory age is admissible, timestamps already constrained at t0.
    ea=ha.notna()&(ha>=0)
    eb=hb.notna()&(hb>=0)
    if variant=="recent_72h":
        ea&=(ha<=FRESH_HOURS)
        eb&=(hb<=FRESH_HOURS)
    if variant=="historical_archive":
        a["log_age"]=np.log1p(ha.clip(lower=0))
        b["log_age"]=np.log1p(hb.clip(lower=0))
        cols=["nhc_t0_wind_mph","nhc_t0_pressure_mb","log_age"]
    else:
        cols=["nhc_t0_wind_mph","nhc_t0_pressure_mb"]
    eligible=(np.array([tropical(t) for t in test.incidentType])[:,None]
        &np.array([tropical(t) for t in pool.incidentType])[None,:]
        &eb.to_numpy()[:,None]&ea.to_numpy()[None,:])
    x=a[cols].to_numpy(float);y=b[cols].to_numpy(float)
    obs_a=np.isfinite(x)&ea.to_numpy()[:,None]
    obs_b=np.isfinite(y)&eb.to_numpy()[:,None]
    # Train-derived scaling only. Constant/all-missing columns have unit scale.
    median=np.nanmedian(np.where(obs_a,x,np.nan),axis=0)
    median=np.where(np.isfinite(median),median,0)
    train=np.where(obs_a,x,median)
    testvals=np.where(obs_b,y,median)
    p10=np.percentile(train,10,axis=0)
    p90=np.percentile(train,90,axis=0)
    spread=np.where((p90-p10)>1e-9,p90-p10,1)
    train=np.clip((train-median)/spread,-8,8)
    testvals=np.clip((testvals-median)/spread,-8,8)
    both=obs_b[:,None,:]&obs_a[None,:,:]
    common=both.sum(axis=2)
    e=(((testvals[:,None,:]-train[None,:,:])**2)*both).sum(axis=2)
    d=np.divide(e,common,out=np.zeros((len(test),len(pool)),float),where=common>0)
    available=eligible&(common>0)
    d=np.where(available,d,0.0)
    return d,available
def predict(train,test,variant):
    pool=train[(train.target_m>=SUPPORT_LOW)&(train.target_m<SUPPORT_HIGH)].copy()
    assert len(pool)>K
    ops=operator_dist(pool,test)
    ext,avail=source_dist(pool,test,variant)
    dist=ops+(WEIGHT*ext)
    y=pool.target_m.to_numpy(float)
    pred=[];counts=[];neigh_storm_ids=[]
    for n,line in enumerate(dist):
        idx=np.argsort(line,kind="stable")[:K]
        w=1.0/(0.2+np.sqrt(line[idx]))
        value=float(np.average(y[idx],weights=w))
        pred.append(float(np.clip(value,B1,B2-1e-6)))
        counts.append(int(avail[n,idx].sum()))
        neigh_storm_ids.append("|".join(str(x) for x in pool.iloc[idx].disasterNumber))
    return np.array(pred),counts,neigh_storm_ids

def metrics(df):
    y=df.actual_m.to_numpy(float)
    p=df.pred_m.to_numpy(float)
    e=abs(y-p)
    return {"n":len(df),
      "MAE_M":float(np.mean(e)),
      "RMSE_M":float(np.sqrt(np.mean((y-p)**2))),
      "R2":float(1-sum((y-p)**2)/sum((y-y.mean())**2)) if len(y)>1 and np.var(y)>0 else None,
      "within30":float(np.mean(e/np.maximum(y,1e-8)<=.3)),
      "cases_using_source_analogue_pair":int((df.valid_source_pairs>0).sum()),
      "n_tropical_test_cases":int(df.incidentType.isin(["Hurricane","Tropical Storm"]).sum()),
      "test_fiscal_years":sorted(int(x) for x in df.fy.unique().tolist())
      }
def cluster_bootstrap(base,other,repeats=3000):
    # Stress test with clustered resampling of meteorological events.
    # Not an independent confidence interval due to earlier model development.
    a=base.set_index("disasterNumber").sort_index()
    b=other.set_index("disasterNumber").sort_index()
    assert a.index.equals(b.index)
    ids=a["storm_cluster"].fillna("").astype(str)
    errors_a=(a.actual_m-a.pred_m).abs().to_numpy(float)
    errors_b=(b.actual_m-b.pred_m).abs().to_numpy(float)
    clusters=ids.to_numpy()
    keys=np.unique(clusters)
    rng=np.random.default_rng(20261101)
    result=[]
    for _ in range(repeats):
        sample=rng.choice(keys,size=len(keys),replace=True)
        idx=np.concatenate([np.flatnonzero(clusters==k) for k in sample])
        result.append(float(np.mean(errors_b[idx]-errors_a[idx])))
    return {"different_storm_clusters":int(len(keys)),
       "experimental_minus_operational_MAE_delta_M":float(np.mean(errors_b-errors_a)),
       "exploratory_cluster_bootstrap_delta_95pct_M":[float(x) for x in np.quantile(result,[.025,.975])],
       "bootstrap_repetitions":repeats,
       "caveat":"Highly selected 50-200M cases; storms can overlap across disaster rows. Interval is diagnostic, not independent confirmation."}

def main():
    base=old.load()
    assert len(base)==912
    path=IN/"tropical"/"nhc_tropical_features_target_blind.csv"
    feat=pd.read_csv(path)
    assert len(feat)==123 and feat.disasterNumber.nunique()==123
    assert all(c in feat.columns for c in RAW)
    aud=pd.read_csv(IN/"tropical"/"storm_matching_and_timestamp_audit.csv")
    assert len(aud)==123
    frozen=pd.read_csv(IN/"frozen"/"candidate_predictions.csv")
    assert len(frozen)==912
    assert int((frozen.actual_band==frozen.candidate_pred_new).sum())==830
    d=base.merge(feat,on="disasterNumber",how="left",validate="one_to_one")
    d=d.merge(aud[["disasterNumber","named_storm"]],on="disasterNumber",how="left",validate="one_to_one")
    assert len(d)==912
    # Non-tropical unmatched records stay missing (not zero meteorological intensity).
    assert d.nhc_t0_hours_before_cutoff.notna().sum()==113
    assert (d.nhc_t0_hours_before_cutoff.dropna()>=0).all()
    # The storm cluster is only used for assessment; never in predictors or analogues.
    d["incident_year"]=pd.to_datetime(pd.read_excel(ROOT/"master_openfema_40plus.xlsx")
       .set_index("disasterNumber").reindex(d.disasterNumber).incidentBeginDate,
       utc=True,errors="coerce").dt.year.to_numpy()
    d["storm_cluster"]=np.where(
       d.named_storm.notna(),d.named_storm.fillna("").astype(str)+"|"+d.incident_year.astype("Int64").astype(str),
       "fema_"+d.disasterNumber.astype(str))
    routed=set(frozen.loc[frozen.candidate_pred_new=="50-200M","disasterNumber"].astype(int))
    assert len(routed)==13
    d["routed"]=d.disasterNumber.isin(routed)
    out=[]
    for year in sorted(d.fyDeclared.unique()):
        tr=d[d.fyDeclared<year].copy()
        # Require enough earlier-year specialist support; no future-year data.
        if tr.target_m.between(SUPPORT_LOW,SUPPORT_HIGH,inclusive="left").sum()<6:
            continue
        inband=tr.target_m.between(B1,B2,inclusive="left")
        if inband.sum()<3:continue
        med=float(tr.loc[inband,"target_m"].median())
        for scope,te in [
            ("direct",d[(d.fyDeclared==year)&d.target_m.between(B1,B2,inclusive="left")]),
            ("routed",d[(d.fyDeclared==year)&d.routed])]:
            if len(te)==0:continue
            test=te.copy()
            test["target_m"]=np.nan
            for variant in VARIANTS:
                estimates,shared,neighbors=predict(tr,test,variant)
                estimate=np.clip(med+ALPHA*(estimates-med),B1,B2-1e-6)
                for (_,r),p,n,nn in zip(te.iterrows(),estimate,shared,neighbors):
                    out.append({"scope":scope,"variant":variant,"fy":int(year),
                      "disasterNumber":int(r.disasterNumber),"incidentType":r.incidentType,
                      "storm_cluster":r.storm_cluster,"actual_m":float(r.target_m),
                      "pred_m":float(p),"median_m":med,"valid_source_pairs":int(n),
                      "neighbor_ids":nn,"nhc_age_hours":r.nhc_t0_hours_before_cutoff,
                      "nhc_wind_mph":r.nhc_t0_wind_mph,"nhc_pressure_mb":r.nhc_t0_pressure_mb})
    pred=pd.DataFrame(out)
    if pred.empty:raise AssertionError("No eligible earlier-year test cases")
    result={}
    for scope,group in pred.groupby("scope"):
        result[scope]={}
        for variant,part in group.groupby("variant"):
            result[scope][variant]=metrics(part)
        comparison={}
        for variant in VARIANTS[1:]:
            comparison[variant]=cluster_bootstrap(
                group[group.variant=="operational"],
                group[group.variant==variant])
        result[scope]["paired_diagnostic"]=comparison
    summary={
      "experiment":"Earlier-fiscal-year-only frozen analogue sensitivity",
      "frozen_router":"830/912 unchanged",
      "models":"50-200M five-nearest historical support 10M-500M, alpha 0.75",
      "comparison":["operational","recent_72h","historical_archive"],
      "fixed_weight":WEIGHT,
      "source_eligibility":{"tropical_all":len(feat),
         "t0_archived_advisories":int(d.nhc_t0_wind_mph.notna().sum()),
         "recent_72h":int((d.nhc_t0_hours_before_cutoff<=72).fillna(False).sum())},
      "scope":result,
      "major_limitations":[
        "Only 13 actual cases and 13 routed cases in rare band; earlier-year support reduces counts.",
        "Current FEMA titles are retrospective storm identity proxies, not proven as-of.",
        "FEMA declaration time conservatively set to midnight UTC from a date-only source.",
        "Operational baseline includes post-declaration mission counts and possibly duration: FULL system NOT t0 safe.",
        "Prior experiment used similar data for choosing severity weight: no independent untouched holdout.",
        "No source group represents local physical damage or exposed population.",
        "No accepted amount specialist or router was replaced."
      ]
    }
    OUT.mkdir(parents=True,exist_ok=True)
    pred.sort_values(["scope","fy","disasterNumber","variant"]).to_csv(OUT/"paired_forward_year_predictions.csv",index=False)
    d[["disasterNumber","incidentType","fyDeclared","storm_cluster"]+RAW].to_csv(
       OUT/"tropical_asof_feature_coverage_nonbio912.csv",index=False)
    (OUT/"results.json").write_text(json.dumps(summary,indent=2,default=str))
    print("BEGIN_NHC_FORWARD_ABLATION\n"+json.dumps(summary,indent=2,default=str)+"\nEND_NHC_FORWARD_ABLATION",flush=True)
if __name__=="__main__":main()
