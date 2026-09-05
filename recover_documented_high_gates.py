from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import external_physical_severity_high_router_experiment as base

OUT = Path("recovered_documented_high_gates_results")
OUT.mkdir(exist_ok=True)

EXPOSURE = Path("recovered_gate_inputs/exposure/declared_population_exposure_features.csv")
MISSION = Path("recovered_gate_inputs/mission/mission_composition_features.csv")

LOW_NUM = [
    "durationDays","declarationDelayDays","expectedResourceScore",
    "missionAssignmentCount","uniqueAgencyCount","uniqueMaTypeCount","uniquePriorityCount",
    "responseComplexityScore","missionDensity","agencyDensity","population2010","eventSize",
    "logPopulation","logMission","logComplexity","logAgency","logDuration","logEventSize",
    "missionAssignmentCount_event_pct_rank","responseComplexityScore_event_pct_rank",
    "uniqueAgencyCount_event_pct_rank","population2010_event_pct_rank",
    "missionAssignmentCount_relative_event_avg","responseComplexityScore_relative_event_avg",
    "uniqueAgencyCount_relative_event_avg","population2010_relative_event_avg",
    "ihProgramDeclared","iaProgramDeclared","paProgramDeclared","hmProgramDeclared",
    "declaredCountyMatchedCount","declaredUniquePlaceCount","declaredGeographyRowCount",
    "declaredCountyMatchRatio","declaredPopulation2010","logDeclaredPopulation2010",
    "declaredPopulationShareState","declaredCountyPopulationMean","declaredCountyPopulationMedian",
    "declaredCountyPopulationMax","declaredLandSqMi","logDeclaredLandSqMi",
    "declaredLandShareState","declaredPopulationDensity","statewideDeclarationFlag",
]
LOW_CAT = ["incidentType","state","expectedResourceLevel","disasterCategory","durationClass"]

LOW_CONFIGS = [
    dict(n_estimators=400,max_depth=d,min_samples_leaf=leaf,max_features=mf,
         class_weight="balanced",random_state=42,n_jobs=-1)
    for d in [3,4,5,None] for leaf in [1,2] for mf in [0.7,1.0]
]

TOP_WEIGHTS = ["uniform","distance"]
TOP_ALPHAS = [0.25,0.50,0.75,1.0]
TOP_YEAR_MODE = ["logistic","knn3"]


def num(x):
    return pd.to_numeric(x, errors="coerce").replace([np.inf,-np.inf],np.nan)


def best_threshold(y,p):
    y=np.asarray(y,int); p=np.asarray(p,float)
    fpr,tpr,thr=roc_curve(y,p); tnr=1-fpr
    feasible=np.where((tpr>=.8)&(tnr>=.8))[0]
    cand=feasible if len(feasible) else np.arange(len(thr))
    mins=np.minimum(tpr[cand],tnr[cand])
    best=mins.max()
    c2=cand[mins==best]
    j=int(c2[np.argmax((tpr[c2]+tnr[c2])/2)])
    return dict(
        threshold=float(thr[j]), lower_recall=float(tnr[j]),
        upper_recall=float(tpr[j]), min_recall=float(min(tnr[j],tpr[j])),
        pass80=bool(len(feasible))
    )


def side_metrics(y,pred):
    y=np.asarray(y,int); pred=np.asarray(pred,int)
    n0=int((y==0).sum()); n1=int((y==1).sum())
    c0=int(((y==0)&(pred==0)).sum()); c1=int(((y==1)&(pred==1)).sum())
    return dict(lower_correct=c0,lower_n=n0,lower_recall=c0/max(n0,1),
                upper_correct=c1,upper_n=n1,upper_recall=c1/max(n1,1))


def low_pipe(cfg,nums,cats):
    pre=ColumnTransformer([
        ("num",Pipeline([("imp",SimpleImputer(strategy="median"))]),nums),
        ("cat",Pipeline([
            ("imp",SimpleImputer(strategy="most_frequent")),
            ("oh",OneHotEncoder(handle_unknown="ignore",sparse_output=False))
        ]),cats),
    ])
    return Pipeline([("pre",pre),("model",ExtraTreesClassifier(**cfg))])


def low_prob(train,test,cfg,nums,cats):
    y=(train.band==4).astype(int)
    m=low_pipe(cfg,nums,cats)
    m.fit(train[nums+cats],y)
    return m.predict_proba(test[nums+cats])[:,1]


def low_inner_oof(train,cfg,nums,cats):
    pair=train[train.band.isin([3,4])]
    out=pd.Series(np.nan,index=pair.index,dtype=float)
    for yr in sorted(pair.fyDeclared.astype(int).unique()):
        tr=pair[pair.fyDeclared.astype(int)!=yr]
        te=pair[pair.fyDeclared.astype(int)==yr]
        if te.empty or (tr.band==4).astype(int).nunique()<2: continue
        out.loc[te.index]=low_prob(tr,te,cfg,nums,cats)
    return out


def select_low(train,nums,cats):
    pair=train[train.band.isin([3,4])]
    y=(pair.band==4).astype(int).to_numpy()
    best=None
    for i,cfg in enumerate(LOW_CONFIGS):
        p=low_inner_oof(train,cfg,nums,cats).loc[pair.index].to_numpy(float)
        ok=np.isfinite(p)
        if ok.sum()<5 or len(np.unique(y[ok]))<2: continue
        op=best_threshold(y[ok],p[ok])
        key=(int(op["pass80"]),op["min_recall"],(op["lower_recall"]+op["upper_recall"])/2)
        if best is None or key>best[0]: best=(key,i,cfg,op)
    if best is None: raise RuntimeError("No lower gate selectable")
    return best[1],best[2],best[3]


def top_xy(df,normal):
    return pd.DataFrame({
        "logMission":np.log1p(num(df.missionAssignmentCount).fillna(0).clip(lower=0)),
        "normalShare":num(df[normal]).fillna(0).clip(0,1),
    },index=df.index)


def top_components(train,test,normal,weights,year_mode):
    X=top_xy(train,normal); Xe=top_xy(test,normal)
    y=(train.band==5).astype(int).to_numpy()

    k=min(3,len(train))
    knn=Pipeline([("s",StandardScaler()),
                  ("m",KNeighborsClassifier(n_neighbors=k,weights=weights))])
    knn.fit(X,y)
    pk=knn.predict_proba(Xe)[:,list(knn.named_steps["m"].classes_).index(1)]

    if year_mode=="logistic":
        ym=Pipeline([("s",StandardScaler()),
                     ("m",LogisticRegression(class_weight="balanced",C=1,max_iter=5000))])
    else:
        ym=Pipeline([("s",StandardScaler()),
                     ("m",KNeighborsClassifier(n_neighbors=min(3,len(train)),weights="distance"))])
    ym.fit(train[["fyDeclared"]].astype(float),y)
    py=ym.predict_proba(test[["fyDeclared"]].astype(float))[:,list(ym.named_steps["m"].classes_).index(1)]
    return pk,py


def top_inner_oof(train,normal,weights,alpha,year_mode):
    pair=train[train.band.isin([4,5])]
    out=pd.Series(np.nan,index=pair.index,dtype=float)
    for yr in sorted(pair.fyDeclared.astype(int).unique()):
        tr=pair[pair.fyDeclared.astype(int)!=yr]
        te=pair[pair.fyDeclared.astype(int)==yr]
        if te.empty or (tr.band==5).astype(int).nunique()<2: continue
        pk,py=top_components(tr,te,normal,weights,year_mode)
        out.loc[te.index]=alpha*pk+(1-alpha)*py
    return out


def select_top(train,normal):
    pair=train[train.band.isin([4,5])]
    y=(pair.band==5).astype(int).to_numpy()
    best=None
    for w in TOP_WEIGHTS:
      for a in TOP_ALPHAS:
       for ym in TOP_YEAR_MODE:
        p=top_inner_oof(train,normal,w,a,ym).loc[pair.index].to_numpy(float)
        ok=np.isfinite(p)
        if ok.sum()<5 or len(np.unique(y[ok]))<2: continue
        op=best_threshold(y[ok],p[ok])
        key=(int(op["pass80"]),op["min_recall"],(op["lower_recall"]+op["upper_recall"])/2)
        if best is None or key>best[0]: best=(key,w,a,ym,op)
    if best is None: raise RuntimeError("No top gate selectable")
    return best[1],best[2],best[3],best[4]


def compose(low_prob_arr,low_t,top_prob_arr,top_t):
    highside=np.asarray(low_prob_arr)>=low_t
    out=np.full(len(highside),3,int)
    out[highside]=4
    out[highside & (np.asarray(top_prob_arr)>=top_t)]=5
    return out


def main():
    if not EXPOSURE.exists() or not MISSION.exists():
        raise FileNotFoundError("Archived gate feature snapshots were not downloaded by workflow")

    d=base.load_master()
    exposure=pd.read_csv(EXPOSURE)
    mission=pd.read_csv(MISSION)

    normal_candidates=[c for c in mission.columns if c.lower().startswith("priority_share__") and "normal" in c.lower()]
    if not normal_candidates:
        raise RuntimeError("Normal priority share not found in archived mission snapshot")
    normal=sorted(normal_candidates,key=lambda c:(len(c),c))[0]

    d=d.merge(exposure.drop(columns=["state"],errors="ignore"),on="disasterNumber",how="left")
    d=d.merge(mission[["disasterNumber",normal]],on="disasterNumber",how="left")
    d[normal]=num(d[normal]).fillna(0)
    nums=[c for c in LOW_NUM if c in d.columns]
    cats=[c for c in LOW_CAT if c in d.columns]

    high=d[d.band.isin([3,4,5])].copy().reset_index(drop=True)
    years=sorted(high.fyDeclared.astype(int).unique())

    rows=[]; selections=[]
    for outer in years:
        train=high[high.fyDeclared.astype(int)!=outer].copy()
        test=high[high.fyDeclared.astype(int)==outer].copy()

        li,lcfg,lop=select_low(train,nums,cats)
        ltrain=train[train.band.isin([3,4])]
        lp=low_prob(ltrain,test,lcfg,nums,cats)

        tw,ta,tym,topop=select_top(train,normal)
        ttrain=train[train.band.isin([4,5])]
        pk,py=top_components(ttrain,test,normal,tw,tym)
        tp=ta*pk+(1-ta)*py

        pred=compose(lp,lop["threshold"],tp,topop["threshold"])
        for j,(_,r) in enumerate(test.iterrows()):
            rows.append({
                "outer_year":int(outer),"disasterNumber":int(r.disasterNumber),"state":str(r.state),
                "incidentType":str(r.incidentType),"target":float(r.target),"actual_band":int(r.band),
                "low_prob_middle":float(lp[j]),"low_threshold":float(lop["threshold"]),
                "low_highside":int(lp[j]>=lop["threshold"]),
                "top_prob_extreme":float(tp[j]),"top_threshold":float(topop["threshold"]),
                "top_extreme":int(tp[j]>=topop["threshold"]),
                "composed_prediction":int(pred[j]),
            })
        selections.append({
            "outer_year":int(outer),"low_config_index":int(li),
            "low_config":json.dumps(lcfg,default=str),"low_threshold":float(lop["threshold"]),
            "low_inner_lower_recall":float(lop["lower_recall"]),
            "low_inner_upper_recall":float(lop["upper_recall"]),
            "top_weights":tw,"top_alpha_knn":float(ta),"top_year_mode":tym,
            "top_threshold":float(topop["threshold"]),
            "top_inner_middle_recall":float(topop["lower_recall"]),
            "top_inner_extreme_recall":float(topop["upper_recall"]),
        })

    out=pd.DataFrame(rows).sort_values(["outer_year","disasterNumber"])
    out.to_csv(OUT/"nested_oof_all_high_predictions.csv",index=False)
    pd.DataFrame(selections).to_csv(OUT/"outer_fold_selected_configs.csv",index=False)

    low_eval=out[out.actual_band.isin([3,4])].copy()
    low_y=(low_eval.actual_band==4).astype(int).to_numpy()
    low_p=low_eval.low_highside.to_numpy(int)
    top_eval=out[out.actual_band.isin([4,5])].copy()
    top_y=(top_eval.actual_band==5).astype(int).to_numpy()
    top_p=top_eval.top_extreme.to_numpy(int)

    y=out.actual_band.to_numpy(int); pred=out.composed_prediction.to_numpy(int)
    rec=recall_score(y,pred,labels=[3,4,5],average=None,zero_division=0)
    cm=confusion_matrix(y,pred,labels=[3,4,5])

    summary={
        "purpose":"Reproduce the two documented nested local high-value gates from archived feature snapshots and a constrained model-family search, then compose them hierarchically.",
        "archived_inputs":{
            "exposure_run":33647372275,
            "mission_feature_run":33653305090,
            "normal_priority_feature":normal,
        },
        "local_lower_gate":side_metrics(low_y,low_p),
        "local_top_gate":side_metrics(top_y,top_p),
        "documented_reference":{
            "lower":"34/39 (or separately 33/39) and 8/9",
            "top":"8/9 and 6/7",
        },
        "composed_three_band":{
            "band3_correct":int(cm[0,0]),"band3_n":int((y==3).sum()),"band3_recall":float(rec[0]),
            "band4_correct":int(cm[1,1]),"band4_n":int((y==4).sum()),"band4_recall":float(rec[1]),
            "band5_correct":int(cm[2,2]),"band5_n":int((y==5).sum()),"band5_recall":float(rec[2]),
            "min_recall":float(rec.min()),"confusion_matrix":cm.tolist(),
        },
        "local_lower_failures":low_eval[low_p!=low_y][
            ["disasterNumber","state","incidentType","target","actual_band","low_prob_middle","low_threshold"]
        ].to_dict("records"),
        "local_top_failures":top_eval[top_p!=top_y][
            ["disasterNumber","state","incidentType","target","actual_band","top_prob_extreme","top_threshold"]
        ].to_dict("records"),
        "composition_failures":out[out.composed_prediction!=out.actual_band][
            ["disasterNumber","state","incidentType","target","actual_band",
             "low_prob_middle","low_threshold","top_prob_extreme","top_threshold","composed_prediction"]
        ].to_dict("records"),
        "guardrails":[
            "Archived exposure and mission-composition feature snapshots from the historical runs are reused exactly.",
            "No funding-derived eventScale or funding-per-resource predictor is included.",
            "Every outer fiscal year is excluded from all model/configuration/threshold selection.",
            "Lower-gate search is restricted to ExtraTrees within the documented exposure/mission-scale/geographic/event-relative feature family.",
            "Top gate is fixed to 3-nearest-neighbor mission/Normal-priority structure, with only blend/year-context choices selected inside outer training years.",
            "If the documented reference counts are not reproduced, the old figures remain development records rather than final reproducible thesis evidence."
        ],
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str),flush=True)


if __name__=="__main__":
    main()
