#!/usr/bin/env python3
"""Research-only physical-severity ablation for the frozen 50-200M analogue.

Inputs are immutable Sept 2026 external features (971 FEMA rows), the 971-row
original workbook and the 830/912 frozen-router prediction artifact. This does
not update the accepted router or any deployed specialist.

WARNING: NOAA final storm reports, NHC HURDAT2 and CAL FIRE perimeter records
are retrospective. No claim of declaration-time availability is justified.
Model hyperparameters are FIXED before looking at this experiment's scores.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import nonbio_50_200m_fast_hybrid as old

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/"audit_inputs"/"severity_physical_pilot"
OUT=ROOT/"audit_outputs"/"severity_physical_pilot"
GROUPS={
    # Reported event footprints (not direct measurements of rainfall/depth).
    "noaa_footprint":["noaa_event_count","noaa_episode_count",
        "noaa_unique_event_types","noaa_unique_counties"],
    "nhc_intensity":["nhc_min_distance_km","nhc_storm_max_wind_kt",
        "nhc_storm_min_pressure_mb","nhc_ace",
        "nhc_closest_wind_kt","nhc_wind_distance_index"],
    "calfire_acres":["calfire_fire_count","calfire_acres_total","calfire_acres_max"]
}
# No NOAA property/crop damage, structure damage, fatality or funding fields.
OPERATIONAL=[
    "durationDays","declarationDelayDays","expectedResourceScore",
    "missionAssignmentCount","uniqueAgencyCount","uniqueMaTypeCount",
    "uniquePriorityCount","responseComplexityScore","missionDensity",
    "agencyDensity"
]
K=5
ALPHA=0.75
SUPPORT_LO,SUPPORT_HI=10.0,500.0
# Pre-specified sensitivity weights, not selected using the outer cases.
VARIANTS={"operational":0.0,"severity_weight_035":0.35,"severity_weight_075":0.75}

def squared_scaled_distance(a,b,cols):
    xa=a[cols].apply(pd.to_numeric,errors="coerce")
    xb=b[cols].apply(pd.to_numeric,errors="coerce")
    med=xa.median().fillna(0.0)
    train=xa.fillna(med).to_numpy(float)
    test=xb.fillna(med).to_numpy(float)
    scale=RobustScaler(quantile_range=(10,90)).fit(train)
    train=np.clip(scale.transform(train),-8,8)
    test=np.clip(scale.transform(test),-8,8)
    return np.mean((test[:,None,:]-train[None,:,:])**2,axis=2)

def missing_aware_group_distance(pool,test,cols):
    a=pool[cols].apply(pd.to_numeric,errors="coerce").to_numpy(float)
    b=test[cols].apply(pd.to_numeric,errors="coerce").to_numpy(float)
    valid_a=np.isfinite(a)
    valid_b=np.isfinite(b)
    # Scale each severity measurement using training-only nonmissing values.
    median=np.nanmedian(a,axis=0)
    median=np.where(np.isfinite(median),median,0)
    a2=np.where(valid_a,a,median)
    b2=np.where(valid_b,b,median)
    s=RobustScaler(quantile_range=(10,90)).fit(a2)
    a2=np.clip(s.transform(a2),-8,8)
    b2=np.clip(s.transform(b2),-8,8)
    both=valid_b[:,None,:]&valid_a[None,:,:]
    differences=(b2[:,None,:]-a2[None,:,:])**2
    n=both.sum(axis=2)
    score=np.divide((differences*both).sum(axis=2),n,
        out=np.zeros((len(test),len(pool))),where=n>0)
    return score,n>0

def pairwise(pool,test):
    physical=np.zeros((len(test),len(pool)),float)
    observed=np.zeros_like(physical)
    for name,cols in GROUPS.items():
        group,available=missing_aware_group_distance(pool,test,cols)
        physical+=group*available
        observed+=available
    physical=np.divide(physical,observed,
        out=np.zeros_like(physical),where=observed>0)
    # Original operational distance, unchanged when severity weight == 0.
    core=squared_scaled_distance(pool,test,OPERATIONAL)
    core+=2.0*(test.incidentType.to_numpy()[:,None]!=pool.incidentType.to_numpy()[None,:])
    core+=0.30*(test.state.to_numpy()[:,None]!=pool.state.to_numpy()[None,:])
    return core,physical,observed>0

def predict(train,te,weight):
    pool=train[(train.target_m>=SUPPORT_LO)&(train.target_m<SUPPORT_HI)].copy()
    assert len(pool)>K
    core,physical,available=pairwise(pool,te)
    # Only similarity to known physical observations can affect distance.
    distance=core+weight*physical
    values=pool.target_m.to_numpy(float)
    result=[]
    neighbour_obs=[]
    for j,row in enumerate(distance):
        nearest=np.argsort(row,kind="stable")[:K]
        w=1/(0.2+np.sqrt(row[nearest]))
        estimate=float(np.average(values[nearest],weights=w))
        result.append(np.clip(estimate,old.LO,old.HI-1e-6))
        neighbour_obs.append(int(available[j,nearest].sum()))
    return np.asarray(result,float),neighbour_obs

def metrics(actual,pred):
    y=np.asarray(actual,float);p=np.asarray(pred,float)
    error=np.abs(y-p); relative=error/np.maximum(y,1e-9)
    result={"n":len(y),"MAE_M":float(error.mean()),
      "RMSE_M":float(np.sqrt(np.mean((y-p)**2))),
      "MedAE_M":float(np.median(error)),
      "R2":float(1-((y-p)**2).sum()/((y-y.mean())**2).sum()) if len(y)>1 and np.var(y)>0 else None,
      "log_MAE":float(np.abs(np.log1p(y*1e6)-np.log1p(p*1e6)).mean()),
      "within20":float(np.mean(relative<=.2)),
      "within30":float(np.mean(relative<=.3)),
      "within50":float(np.mean(relative<=.5))}
    return result

def main():
    d=old.load()
    ext=pd.read_csv(INPUT/"external"/"external_features_971.csv")
    frozen=pd.read_csv(INPUT/"frozen"/"candidate_predictions.csv")
    assert len(ext)==971 and ext.disasterNumber.nunique()==971
    assert len(frozen)==912 and int((frozen.actual_band==frozen.candidate_pred_new).sum())==830
    for key,cols in GROUPS.items():
        assert set(cols)<=set(ext.columns),(key,set(cols)-set(ext.columns))
    d=d.merge(ext,on="disasterNumber",how="left",validate="one_to_one")
    assert len(d)==912
    routed_ids=set(frozen.loc[frozen.candidate_pred_new=="50-200M","disasterNumber"].astype(int))
    assert len(routed_ids)==13
    d["routed"]=d.disasterNumber.astype(int).isin(routed_ids)

    coverage={}
    for group,cols in GROUPS.items():
        present=d[cols].notna().any(axis=1)
        coverage[group]={"nonbio_912":int(present.sum()),
           "true50_200":int(present[(d.target_m>=old.LO)&(d.target_m<old.HI)].sum()),
           "routed50_200":int(present[d.routed].sum())}
    records=[]
    held_years=sorted(set(d.loc[(d.target_m>=old.LO)&(d.target_m<old.HI),"fyDeclared"])|
        set(d.loc[d.routed,"fyDeclared"]))
    for year in held_years:
        tr=d[d.fyDeclared!=year].copy()
        cases=[("direct",d[(d.fyDeclared==year)&(d.target_m>=old.LO)&(d.target_m<old.HI)]),
               ("routed",d[(d.fyDeclared==year)&(d.routed)])]
        m=float(tr.loc[(tr.target_m>=old.LO)&(tr.target_m<old.HI),"target_m"].median())
        assert np.isfinite(m)
        for scope,te in cases:
            if te.empty:continue
            masked=te.copy()
            masked["target_m"]=np.nan  # Test targets invisible at inference.
            for variant,weight in VARIANTS.items():
                raw,physical_neighbours=predict(tr,masked,weight)
                estimates=np.clip(m+ALPHA*(raw-m),old.LO,old.HI-1e-6)
                for (_,case),guess,num in zip(te.iterrows(),estimates,physical_neighbours):
                    records.append({"scope":scope,"disasterNumber":int(case.disasterNumber),
                       "fyDeclared":int(year),"actual_m":float(case.target_m),
                       "variant":variant,"estimate_m":float(guess),"band_median_m":m,
                       "neighbours_with_physical_data":num,
                       "true_band":bool(old.LO<=case.target_m<old.HI)})
    result=pd.DataFrame(records)
    scores={}
    for scope in ["direct","routed"]:
        test=result[result.scope==scope]
        scores[scope]={}
        for variant,part in test.groupby("variant"):
            assert len(part)==13,(scope,variant,len(part))
            scores[scope][variant]=metrics(part.actual_m,part.estimate_m)
            scores[scope][variant]["neighbours_with_any_physical_evidence"]=int(
                (part.neighbours_with_physical_data>0).sum())
        print(scope,[(name,round(val["MAE_M"],3)) for name,val in scores[scope].items()],flush=True)
    summary={
       "purpose":"Research-only, fixed architecture external severity feature ablation",
       "funding_router_unchanged":"830/912",
       "data_record_count":len(d),
       "original_external_source":"GitHub Actions 34017327286 artifact external-severity-ablation",
       "historical_match_provenance":"Target-blind state/date/event heuristics; some event attribution uncertain",
       "prediction_timing":"RETROSPECTIVE, NOT DECLARATION-TIME VALIDATED",
       "excluded_fields":["noaa_property_damage_total","noaa_crop_damage_total",
          "noaa_deaths_direct","calfire_destroyed_structures"],
       "feature_groups":GROUPS,"variant_weights":VARIANTS,
       "frozen_candidate":{"k":K,"alpha":ALPHA,"historical_support_M":[SUPPORT_LO,SUPPORT_HI]},
       "nonbiological_feature_coverage":coverage,
       "metrics":scores,
       "statistical_warning":"13 cases; candidate design informed by previous held-year results. Not a new independent holdout."
    }
    OUT.mkdir(parents=True,exist_ok=True)
    result.sort_values(["scope","fyDeclared","disasterNumber","variant"]).to_csv(
        OUT/"predictions_13_direct_and_routed.csv",index=False)
    d[["disasterNumber","state","incidentType","fyDeclared"]+
      sum(GROUPS.values(),[])].to_csv(OUT/"nonbio_912_physical_features.csv",index=False)
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("BEGIN_PHYSICAL_SEVERITY_SUMMARY",flush=True)
    print(json.dumps(summary,indent=2),flush=True)
    print("END_PHYSICAL_SEVERITY_SUMMARY",flush=True)
if __name__=="__main__":main()
