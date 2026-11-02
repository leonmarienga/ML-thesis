#!/usr/bin/env python3
"""Early-response research-only six-band and funding experiments, 912 cases.

All 2019-2024 held-out fiscal years train ONLY on preceding fiscal years.
Accepted 830/912 retrospective router and amount specialists are not touched.

CAVEAT: current FEMA declaration metadata and MissionAssignments are not
historic versioned records. dateReceived filters mission action time only.
NHC storm identity uses contemporary-current FEMA titles. Cutoffs T0 and T72
use conservative midnight UTC of declarationDate. This is an as-of PROXY,
not a genuinely prospective system or a validated decision-time model.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor,ExtraTreesClassifier
from sklearn.metrics import (mean_absolute_error,mean_squared_error,
 accuracy_score,balanced_accuracy_score,precision_recall_fscore_support,r2_score)

ROOT=Path(__file__).resolve().parents[1]
INP=ROOT/"audit_inputs"/"early_response_912"
OUT=ROOT/"audit_outputs"/"early_response_912_t0_t72"
OUT.mkdir(parents=True,exist_ok=True)
LABELS=["0-100K","100K-1M","1M-50M","50M-200M","200M-500M","500M+"]
BINS=[0,1e5,1e6,50e6,200e6,500e6,float("inf")]
YEARS=list(range(2019,2025))
CATS=["state","incidentType"]
BASE=["declaration_year","declaration_month"]
MA=["mission_count","agency_count","esf_count","ma_type_count","priority_count"]
NH=["wind_mph","pressure_mb"]
ARM={
 "metadata":(None,None),
 "t0_missions":("t0",None),
 "t0_missions_nhc72":("t0","t0"),
 "t72_missions":("t72",None),
 "t72_missions_nhc72":("t72","t72")
}
def band(y):
    return LABELS[int(np.searchsorted(BINS,max(0,float(y)),side="right")-1)]
def load():
    data=pd.read_excel(ROOT/"master_openfema_40plus.xlsx")
    assert len(data)==971 and data.disasterNumber.nunique()==971
    data=data[data.incidentType!="Biological"].copy()
    assert len(data)==912
    data["target"]=pd.to_numeric(data.totalObligatedFunding,errors="raise").clip(lower=0)
    data["band"]=data.target.map(band)
    data["fyDeclared"]=pd.to_numeric(data.fyDeclared).astype(int)
    dt=pd.to_datetime(data.declarationDate,errors="coerce",utc=True)
    data["declaration_year"]=dt.dt.year
    data["declaration_month"]=dt.dt.month
    missions=pd.read_csv(INP/"missions"/"mission_actions_t0_t72_all971.csv")
    cyclone=pd.read_csv(INP/"nhc"/"nhc_tropical_features_target_blind.csv")
    assert len(missions)==971 and missions.disasterNumber.nunique()==971
    assert len(cyclone)==123 and cyclone.disasterNumber.nunique()==123
    macols=[f"ma_{t}_{x}" for t in ["t0","t72"] for x in MA]
    nhcols=[f"nhc_{t}_{x}" for t in ["t0","t72"] for x in NH+["hours_before_cutoff"]]
    assert set(macols)<=set(missions.columns)
    assert set(nhcols)<=set(cyclone.columns)
    data=data.merge(missions[["disasterNumber"]+macols],on="disasterNumber",how="left",validate="one_to_one")
    data=data.merge(cyclone[["disasterNumber"]+nhcols],on="disasterNumber",how="left",validate="one_to_one")
    assert len(data)==912
    for c in macols:
        data[c]=pd.to_numeric(data[c],errors="coerce").fillna(0)
    for t in ["t0","t72"]:
        age=f"nhc_{t}_hours_before_cutoff"
        data[age]=pd.to_numeric(data[age],errors="coerce")
        recent=data[age].between(0,72)
        for f in NH:
            col=f"nhc_{t}_{f}"
            data[col]=pd.to_numeric(data[col],errors="coerce")
            data.loc[~recent,col]=np.nan
    return data
def cols(arm):
    mt,nt=ARM[arm]
    c=list(BASE)
    if mt:c += [f"ma_{mt}_{x}" for x in MA]
    if nt:c += [f"nhc_{nt}_{x}" for x in NH]
    return c
def transform(train,test,arm):
    num=cols(arm)
    a=pd.get_dummies(train[CATS].fillna("MISSING").astype(str),dtype=float)
    b=pd.get_dummies(test[CATS].fillna("MISSING").astype(str),dtype=float)
    b=b.reindex(columns=a.columns,fill_value=0)
    x=train[num].apply(pd.to_numeric,errors="coerce")
    y=test[num].apply(pd.to_numeric,errors="coerce")
    missing_x=x.isna().astype(float).add_prefix("missing_")
    missing_y=y.isna().astype(float).add_prefix("missing_")
    fill=x.median().fillna(0)
    a=pd.concat([a.reset_index(drop=True),x.fillna(fill).reset_index(drop=True),
                 missing_x.reset_index(drop=True)],axis=1)
    b=pd.concat([b.reset_index(drop=True),y.fillna(fill).reset_index(drop=True),
                 missing_y.reset_index(drop=True)],axis=1)
    assert list(a.columns)==list(b.columns)
    return a,b
def evaluate_fold(train,test,arm,year):
    x,y=transform(train,test,arm)
    reg=ExtraTreesRegressor(n_estimators=250,min_samples_leaf=3,max_features=.9,
       random_state=20261101+year,n_jobs=-1)
    reg.fit(x,np.log1p(train.target.to_numpy(float)))
    amounts=np.expm1(reg.predict(y)).clip(0)
    cls=ExtraTreesClassifier(n_estimators=250,min_samples_leaf=2,max_features=.9,
       class_weight="balanced_subsample",random_state=20261201+year,n_jobs=-1)
    cls.fit(x,train.band.to_numpy())
    bands=cls.predict(y)
    return amounts,bands
def calculate(d):
    a=d.target.to_numpy(float);p=d.pred_amount.to_numpy(float)
    scores=precision_recall_fscore_support(d.band,d.pred_band,labels=LABELS,zero_division=0)
    out={"n":len(d),"MAE":float(mean_absolute_error(a,p)),
      "RMSE":float(np.sqrt(mean_squared_error(a,p))),
      "R2":float(r2_score(a,p)) if np.var(a)>0 else None,
      "log_MAE":float(np.mean(abs(np.log1p(a)-np.log1p(p)))),
      "accuracy":float(accuracy_score(d.band,d.pred_band)),
      "balanced_accuracy":float(balanced_accuracy_score(d.band,d.pred_band)),
      "macro_F1":float(np.mean(scores[2])),
      "per_band":{label:{"support":int(scores[3][i]),"recall":float(scores[1][i]),
         "precision":float(scores[0][i])} for i,label in enumerate(LABELS)}}
    return out
def paired(a,b):
    a=a.sort_values("disasterNumber");b=b.sort_values("disasterNumber")
    assert a.disasterNumber.tolist()==b.disasterNumber.tolist()
    diff=abs(b.target-b.pred_amount).to_numpy()-abs(a.target-a.pred_amount).to_numpy()
    rng=np.random.default_rng(20261101)
    bootstrap=[float(np.mean(diff[rng.integers(0,len(diff),len(diff))])) for _ in range(1500)]
    return {"new_minus_old_MAE":float(np.mean(diff)),"exploratory_95pct_interval":
       [float(t) for t in np.quantile(bootstrap,[.025,.975])]}
def main():
    d=load()
    records=[]
    for year in YEARS:
        tr=d[d.fyDeclared<year]
        te=d[d.fyDeclared==year]
        if not len(te):continue
        assert len(tr)>100 and (tr.fyDeclared<year).all()
        for arm in ARM:
            amounts,bands=evaluate_fold(tr,te,arm,year)
            for (_,row),v,c in zip(te.iterrows(),amounts,bands):
                records.append({"disasterNumber":int(row.disasterNumber),
                  "fyDeclared":int(year),"state":row.state,"incidentType":row.incidentType,
                  "arm":arm,"target":float(row.target),"band":row.band,
                  "pred_amount":float(v),"pred_band":str(c)})
        print("COMPLETE_FISCAL_YEAR",year,"train",len(tr),"test",len(te),flush=True)
    p=pd.DataFrame(records)
    assert len(p)>0 and p.disasterNumber.nunique()>100
    n=p.disasterNumber.nunique()
    assert (p.groupby("arm").size()==n).all()
    model_scores={k:calculate(p[p.arm==k]) for k in ARM}
    years={int(y):{k:calculate(p[(p.fyDeclared==y)&(p.arm==k)])
       for k in ARM} for y in sorted(p.fyDeclared.unique())}
    ref=p[p.arm=="metadata"]
    comparisons={k:paired(ref,p[p.arm==k]) for k in ARM if k!="metadata"}
    comparisons["t72_vs_t0_with_recent_nhc"]=paired(
       p[p.arm=="t0_missions_nhc72"],p[p.arm=="t72_missions_nhc72"])
    cover={cut:{"missions":int((d[f"ma_{cut}_mission_count"]>0).sum()),
       "nhc_recent":int(d[f"nhc_{cut}_wind_mph"].notna().sum())} for cut in ["t0","t72"]}
    answer={"purpose":"non-production, experimental earlier-year response model",
       "train_rule":"train on only fiscal years earlier than each held-out 2019-2024 fiscal year",
       "full_912_source_coverage":cover,"test_n":n,"test_years":YEARS,
       "test_band_count":ref.band.value_counts().reindex(LABELS,fill_value=0).astype(int).to_dict(),
       "model_specs":"ExtraTrees log-dollar regressor 250 trees + independent six-class ExtraTreesClassifier 250 trees",
       "models":model_scores,"by_year":years,"paired_diagnostics":comparisons,
       "caveats":["Future-leaky full-event predictors excluded; target remains FINAL obligation total.",
        "Current FEMA declaration title and mission action versions are not certified as contemporaneous.",
        "FEMA declarationDate interpreted as midnight UTC, not precise approval hour.",
        "Current data has been repeatedly explored and estimates are not an independent holdout.",
        "Bootstrap is exploratory and does not cluster multi-state hurricanes.",
        "No change to frozen 830/912 router or accepted $5.015M overall MAE model."]}
    p.sort_values(["disasterNumber","arm"]).to_csv(OUT/"paired_forward_predictions.csv",index=False)
    (OUT/"summary.json").write_text(json.dumps(answer,indent=2))
    slim={k:{f:v[f] for f in ["n","MAE","RMSE","R2","log_MAE","accuracy",
            "balanced_accuracy","macro_F1","per_band"]}for k,v in model_scores.items()}
    print("BEGIN_EARLY_RESPONSE_912")
    print(json.dumps({"n":n,"coverage":cover,"test_band_count":answer["test_band_count"],
       "models":slim,"paired":comparisons},indent=2))
    print("END_EARLY_RESPONSE_912")
if __name__=="__main__":main()
