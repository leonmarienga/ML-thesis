from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, recall_score, balanced_accuracy_score, f1_score

import external_physical_severity_high_router_experiment as base
import hurricane_local_impact_signal_audit as hlocal

OUT = Path("nested_hazard_first_nonbio_results")
OUT.mkdir(exist_ok=True)

BANDS=[3,4,5]
SAFE_CATS=[c for c in base.BASE_CAT if c!="eventScale"]
HUR_TYPES={"HURRICANE","TROPICAL STORM","TYPHOON"}

CFG=dict(
    iterations=220,
    depth=3,
    learning_rate=.03,
    l2_leaf_reg=20,
    loss_function="MultiClass",
    verbose=False,
    random_seed=42,
    allow_writing_files=False,
    auto_class_weights="Balanced",
)

HUR_LOCAL=[
    "min_track_distance_km","wind_at_closest_kt","pressure_at_closest_mb",
    "hours_within_100km","hours_within_200km","hours_within_300km",
    "max_wind_within_100km","max_wind_within_200km","max_wind_within_300km",
    "distance_weighted_wind","distance_weighted_wind2","local_impact_integral",
    "local_peak_proxy","local_major_hours_300km","local_hurricane_hours_300km",
    "population_x_local_impact","population_x_local_peak2",
]

def metric(y,p):
    y=np.asarray(y,int); p=np.asarray(p,int)
    rec=recall_score(y,p,labels=BANDS,average=None,zero_division=0)
    cm=confusion_matrix(y,p,labels=BANDS)
    return {
        "correct3":int(cm[0,0]),"n3":int((y==3).sum()),"r3":float(rec[0]),
        "correct4":int(cm[1,1]),"n4":int((y==4).sum()),"r4":float(rec[1]),
        "correct5":int(cm[2,2]),"n5":int((y==5).sum()),"r5":float(rec[2]),
        "min_recall":float(rec.min()),
        "balanced_accuracy":float(balanced_accuracy_score(y,p)),
        "macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),
        "confusion_matrix":cm.tolist(),
        "pass80":bool(np.all(rec>=.8)),
    }

def prep(df,nums,cats):
    X=df[nums+cats].copy()
    for c in nums:
        X[c]=pd.to_numeric(X[c],errors="coerce")
    for c in cats:
        X[c]=X[c].astype(str).fillna("MISSING")
    return X

def fit_multiclass(train,test,nums,cats):
    classes=sorted(train.band.astype(int).unique())
    if len(classes)==1:
        return np.full(len(test),classes[0],int)
    m=CatBoostClassifier(**CFG)
    m.fit(prep(train,nums,cats),train.band.astype(int),cat_features=cats)
    return np.asarray(m.predict(prep(test,nums,cats))).reshape(-1).astype(int)

def build_hurricane_features(d):
    source=d[
        d.incidentType.astype(str).str.upper().str.strip().isin(HUR_TYPES)
        & d.band.isin(BANDS)
    ].copy()
    if source.empty:
        return pd.DataFrame(columns=["disasterNumber"]+HUR_LOCAL+["hurricaneMatched"])

    titles=hlocal.fetch_titles(source.disasterNumber.astype(int).tolist())
    cents=hlocal.fetch_state_centroids()
    storms,points=hlocal.parse_hurdat_points()

    source=source.merge(
        titles[["disasterNumber","declarationTitle"]],
        on="disasterNumber",how="left"
    )
    source=source.merge(
        cents[["state","centroid_lat","centroid_lon"]],
        on="state",how="left"
    )

    rows=[]
    for _,r in source.iterrows():
        q={"disasterNumber":int(r.disasterNumber),"hurricaneMatched":0}
        s=hlocal.exact_named_match(r,storms)
        if s is not None:
            f=hlocal.local_features(
                str(s.storm_id),points,
                float(r.centroid_lat) if pd.notna(r.centroid_lat) else np.nan,
                float(r.centroid_lon) if pd.notna(r.centroid_lon) else np.nan,
                float(base.POP.get(str(r.state),np.nan)),
            )
            if f:
                q["hurricaneMatched"]=1
                for c in HUR_LOCAL:
                    q[c]=f.get(c,np.nan)
        rows.append(q)
    out=pd.DataFrame(rows)
    for c in HUR_LOCAL:
        if c not in out: out[c]=np.nan
    out.to_csv(OUT/"hurricane_local_features.csv",index=False)
    return out

def hazard_group(s):
    s=str(s).upper().strip()
    if s in HUR_TYPES:return "hurricane_like"
    if s=="FIRE":return "fire"
    return "other"

def main():
    d=base.load_master()
    hf=build_hurricane_features(d)
    d=d.merge(hf,on="disasterNumber",how="left")
    d["hurricaneMatched"]=pd.to_numeric(d.get("hurricaneMatched",0),errors="coerce").fillna(0).astype(int)

    h=d[
        d.band.isin(BANDS)
        & ~d.incidentType.astype(str).str.upper().str.strip().eq("BIOLOGICAL")
    ].copy().reset_index(drop=True)
    h["hazard_group"]=h.incidentType.map(hazard_group)

    base_nums=[c for c in base.BASE_NUM if c in h.columns]
    base_cats=[c for c in SAFE_CATS if c in h.columns]
    hur_nums=base_nums+[c for c in HUR_LOCAL if c in h.columns]+["hurricaneMatched"]
    hur_nums=list(dict.fromkeys(hur_nums))

    pred=np.full(len(h),-99,int)
    generic=np.full(len(h),-99,int)
    rows=[]
    folds=[]

    for yr in sorted(h.fyDeclared.astype(int).unique()):
        trm=h.fyDeclared.astype(int)!=yr
        tem=~trm
        tr=h.loc[trm].copy()
        te=h.loc[tem].copy()

        # Generic fallback for comparison and unsupported specialist folds.
        gp=fit_multiclass(tr,te,base_nums,base_cats)
        generic[np.flatnonzero(tem.to_numpy())]=gp

        hp=np.full(len(te),-99,int)
        routes=[]
        for j,(_,r) in enumerate(te.iterrows()):
            group=r.hazard_group
            specialist_train=tr[tr.hazard_group==group].copy()

            if group=="hurricane_like" and len(specialist_train)>=6 and specialist_train.band.nunique()>=2:
                p=fit_multiclass(specialist_train,r.to_frame().T,hur_nums,base_cats)[0]
                route="hurricane_specialist"
            elif group=="fire" and len(specialist_train)>=3 and specialist_train.band.nunique()>=2:
                p=fit_multiclass(specialist_train,r.to_frame().T,base_nums,base_cats)[0]
                route="fire_specialist"
            else:
                p=gp[j]
                route="generic_fallback"
            hp[j]=int(p)
            routes.append(route)

        idx=np.flatnonzero(tem.to_numpy())
        pred[idx]=hp

        for j,ii in enumerate(idx):
            r=h.iloc[ii]
            rows.append({
                "outer_year":int(yr),
                "disasterNumber":int(r.disasterNumber),
                "state":str(r.state),
                "incidentType":str(r.incidentType),
                "hazard_group":str(r.hazard_group),
                "target":float(r.target),
                "band":int(r.band),
                "generic_prediction":int(gp[j]),
                "hazard_first_prediction":int(hp[j]),
                "route":routes[j],
            })

        folds.append({
            "outer_year":int(yr),
            "train_hurricane_like":int((tr.hazard_group=="hurricane_like").sum()),
            "train_fire":int((tr.hazard_group=="fire").sum()),
            "test_rows":int(len(te)),
        })

    y=h.band.astype(int).to_numpy()
    out=pd.DataFrame(rows).sort_values(["outer_year","disasterNumber"])
    out.to_csv(OUT/"nested_hazard_first_oof.csv",index=False)

    actual5=out[out.band==5].copy()
    summary={
        "purpose":"Strict leave-fiscal-year-out hazard-first router on non-Biological high-value cases, using target-free HURDAT2 local-impact features for hurricane-like events and a dedicated Fire specialist.",
        "counts":{str(b):int((h.band==b).sum()) for b in BANDS},
        "hazard_counts":h.hazard_group.value_counts().to_dict(),
        "metrics":{
            "generic_multiclass_fallback":metric(y,generic),
            "hazard_first":metric(y,pred),
        },
        "actual_500m_plus":actual5[
            ["disasterNumber","state","incidentType","target","generic_prediction","hazard_first_prediction","route"]
        ].to_dict("records"),
        "hazard_first_correct_generic_wrong":out[
            (out.hazard_first_prediction==out.band)&(out.generic_prediction!=out.band)
        ][["disasterNumber","state","incidentType","target","band","generic_prediction","hazard_first_prediction","route"]].to_dict("records"),
        "hazard_first_wrong_generic_correct":out[
            (out.hazard_first_prediction!=out.band)&(out.generic_prediction==out.band)
        ][["disasterNumber","state","incidentType","target","band","generic_prediction","hazard_first_prediction","route"]].to_dict("records"),
        "outer_folds":folds,
        "guardrails":[
            "Biological disasters are excluded from both training and evaluation.",
            "Every outer fiscal year is completely excluded from the specialist fitting used to predict that year.",
            "Hurricane local-impact features come only from HURDAT2/Census/FEMA declaration-title matching and contain no funding information.",
            "Funding-derived eventScale is excluded.",
            "Fire uses only the existing target-free FEMA feature family in this first hazard-first diagnostic.",
            "If a hazard specialist lacks sufficient training support in a fold, prediction falls back to the generic target-free model rather than using the held-out outcome."
        ]
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str),flush=True)

if __name__=="__main__":
    main()
