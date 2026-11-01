#!/usr/bin/env python3
"""Research-only low-leakage proxy comparison, earlier-fiscal-year-only.

Uses stable event metadata + date-of-declaration, then action-time-screened
MissionAssignments, then archive NHC wind when <=72h of declaration day.
FEMA current records do NOT prove historical field versions. No feature
constructed from final mission totals/duration/response complexity/financials.
"""
import json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.preprocessing import OneHotEncoder,RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
ROOT=Path(__file__).resolve().parents[1];IN=ROOT/"audit_inputs"/"mission_asof_compare";OUT=ROOT/"audit_outputs"/"mission_asof_compare";OUT.mkdir(parents=True,exist_ok=True)
BANDS=[0,1e5,1e6,50e6,200e6,500e6,float("inf")]
LABELS=["0-100K","100K-1M","1M-50M","50-200M","200-500M","500M+"]
CATS=["state","incidentType"]
BASE_NUM=["fyDeclared","incident_delay_days"]
MISSION_NUM=["ma_t0_mission_count","ma_t0_agency_count","ma_t0_esf_count"]
NHC_NUM=["nhc_t0_wind_mph","nhc_t0_pressure_mb"]
def data():
 d=pd.read_excel(ROOT/"master_openfema_40plus.xlsx")
 assert len(d)==971
 d=d[d.incidentType!="Biological"].copy()
 assert len(d)==912
 d["target_m"]=pd.to_numeric(d.totalObligatedFunding).clip(lower=0)/1e6
 start=pd.to_datetime(d.incidentBeginDate,errors="coerce",utc=True)
 dec=pd.to_datetime(d.declarationDate,errors="coerce",utc=True)
 d["incident_delay_days"]=(dec-start).dt.total_seconds()/86400
 m=pd.read_csv(IN/"mission"/"mission_actions_t0_t72_all971.csv")
 n=pd.read_csv(IN/"tropical"/"nhc_tropical_features_target_blind.csv")
 assert len(m)==971 and len(n)==123
 d=d.merge(m[["disasterNumber"]+MISSION_NUM],on="disasterNumber",how="left",validate="one_to_one")
 d=d.merge(n[["disasterNumber"]+NHC_NUM+["nhc_t0_hours_before_cutoff"]],on="disasterNumber",how="left",validate="one_to_one")
 assert len(d)==912
 # Exact t0 source age criterion; do not fill NHC absent values as zero.
 stale=~d.nhc_t0_hours_before_cutoff.between(0,72)
 d.loc[stale,NHC_NUM]=np.nan
 return d
def embed(train,test,nums):
 cat=OneHotEncoder(handle_unknown="ignore",sparse_output=False)
 a=cat.fit_transform(train[CATS].fillna("MISSING").astype(str))
 b=cat.transform(test[CATS].fillna("MISSING").astype(str))
 X=train[nums].apply(pd.to_numeric,errors="coerce")
 Y=test[nums].apply(pd.to_numeric,errors="coerce")
 med=X.median().fillna(0)
 known=X.notna().astype(float).to_numpy()
 testknown=Y.notna().astype(float).to_numpy()
 v=X.fillna(med).to_numpy(float);z=Y.fillna(med).to_numpy(float)
 scale=RobustScaler(quantile_range=(10,90)).fit(v)
 v=np.clip(scale.transform(v),-7,7)
 z=np.clip(scale.transform(z),-7,7)
 # Missingness indicators only when source set contains conditional NHC values.
 if "nhc_t0_wind_mph" in nums:
  v=np.column_stack([v,known])
  z=np.column_stack([z,testknown])
 return np.column_stack([a*2,v]),np.column_stack([b*2,z])
def get_nearest(tr,te,features,k=5):
 a,b=embed(tr,te,features)
 distance=np.sqrt(np.mean((b[:,None,:]-a[None,:,:])**2,axis=2))
 ix=np.argsort(distance,axis=1,kind="stable")[:,:k]
 val=tr.target_m.to_numpy()[ix]
 weights=1/(0.2+np.take_along_axis(distance,ix,axis=1))
 return (val*weights).sum(axis=1)/weights.sum(axis=1)
def metrics(d):
 y=d.actual_m.to_numpy();p=d.pred_m.to_numpy()
 e=abs(y-p);return {"n":len(d),"MAE_M":float(e.mean()),"RMSE_M":float(np.sqrt(np.mean((y-p)**2))),"within30":float(np.mean(e/np.maximum(y,1e-6)<=.3))}
def main():
 d=data()
 frozen=pd.read_csv(IN/"frozen"/"candidate_predictions.csv")
 assert len(frozen)==912 and (frozen.actual_band==frozen.candidate_pred_new).sum()==830
 routed=set(frozen.loc[frozen.candidate_pred_new=="50-200M","disasterNumber"].astype(int))
 assert len(routed)==13
 sets={"metadata_only":BASE_NUM,"metadata_t0_missions":BASE_NUM+MISSION_NUM,"metadata_t0_missions_recent_nhc":BASE_NUM+MISSION_NUM+NHC_NUM}
 rec=[]
 for year in sorted(d.fyDeclared.unique()):
  tr=d[d.fyDeclared<year]
  pool=tr[tr.target_m.between(10,500,inclusive="left")]
  support=tr[tr.target_m.between(50,200,inclusive="left")]
  if len(pool)<6 or len(support)<3:continue
  median=float(support.target_m.median())
  testset={"direct":d[(d.fyDeclared==year)&d.target_m.between(50,200,inclusive="left")],
    "routed":d[(d.fyDeclared==year)&d.disasterNumber.isin(routed)]}
  for scope,te in testset.items():
   if len(te)==0:continue
   for name,cols in sets.items():
    pred=get_nearest(pool,te,cols)
    pred=np.clip(median+.75*(pred-median),50,200-1e-6)
    for (_,row),value in zip(te.iterrows(),pred):
     rec.append({"scope":scope,"year":int(year),"disasterNumber":int(row.disasterNumber),"incidentType":row.incidentType,
       "variant":name,"actual_m":float(row.target_m),"pred_m":float(value),
       "has_early_missions":bool(row.ma_t0_mission_count>0),"has_recent_nhc":bool(pd.notna(row.nhc_t0_wind_mph))})
 p=pd.DataFrame(rec)
 result={scope:{name:metrics(g) for name,g in group.groupby("variant")} for scope,group in p.groupby("scope")}
 s={"router":"830/912 unchanged",
   "evaluation":"Forward fiscal year, new reduced metadata-based kNN; NOT exact accepted specialist",
   "data_limit":"Current FEMA metadata not proven archived as-of T0, date-only midnight UTC cutoff; current MissionAssignments rows are not historical versions",
   "excluded_final_features":["missionAssignmentCount","uniqueAgencyCount","durationDays","responseComplexityScore","missionDensity","agencyDensity","expectedResourceScore","funding"],
   "trained_on_future_fiscal_years":False,
   "asof_inputs":{"missions_by_t0":int((d.ma_t0_mission_count>0).sum()),"recent_NHC_t0":int(d.nhc_t0_wind_mph.notna().sum())},
   "metrics":result,
   "interpretation":"Small, repeatedly studied high-dollar sample. Descriptive ablation only; no frozen-router replacement."}
 p.to_csv(OUT/"paired_reduced_feature_predictions.csv",index=False)
 (OUT/"summary.json").write_text(json.dumps(s,indent=2))
 print("BEGIN_ASOF_REDUCED_COMPARE\n"+json.dumps(s,indent=2)+"\nEND_ASOF_REDUCED_COMPARE",flush=True)
if __name__=="__main__":main()
