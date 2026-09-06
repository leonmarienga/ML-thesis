#!/usr/bin/env python3
"""
Storm-relative hurricane lower-band audit.

All 108 Hurricane declarations are grouped by official FEMA declaration title
+ incident year, without using funding. Relative response features are computed
before selecting high-value cases.

Purpose: distinguish jurisdiction-level funding within the same hurricane
(e.g., Sandy NJ vs Sandy NY) using relative Mission Assignment workload.

Retrospective diagnostic: final mission aggregates are used.
"""

from __future__ import annotations
import json, re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score

from mission_semantic_audit import (
    CURRENT_19, build_semantic_rollup, fetch_all_mission_assignments,
    normalize_master, normalize_model_frame, prep_pipeline,
)
from external_severity_ablation import load_fema_declaration_titles

ROOT=Path(__file__).resolve().parents[1]
MASTER=ROOT/"master_openfema_40plus.xlsx"
OUT=ROOT/"audit_outputs"/"storm_relative_hurricane"
OUT.mkdir(parents=True,exist_ok=True)

def canonical_title(s):
    s=str(s or "").upper().strip()
    s=re.sub(r"\s+"," ",s)
    return s

def add_relative_features(h: pd.DataFrame) -> pd.DataFrame:
    x=h.copy()
    x["incident_year"]=pd.to_datetime(x["incidentBeginDate"],errors="coerce").dt.year
    x["storm_key"]=x["declarationTitle"].map(canonical_title)+"|"+x["incident_year"].astype("Int64").astype(str)

    base=[
        "missionAssignmentCount","uniqueAgencyCount","uniqueMaTypeCount","uniquePriorityCount",
        "responseComplexityScore","sem_unique_missions","sem_unique_agencies","sem_unique_esf",
        "sem_type_dfa_count","sem_esf_3_count","sem_esf_4_count",
        "sem_topic_power_count","sem_topic_engineering_infrastructure_count",
        "sem_topic_logistics_commodities_count","sem_topic_debris_count",
    ]
    base=[c for c in base if c in x.columns]

    g=x.groupby("storm_key",dropna=False)
    x["storm_declaration_count"]=g["disasterNumber"].transform("count")
    for c in base:
        v=pd.to_numeric(x[c],errors="coerce").fillna(0.0)
        total=v.groupby(x["storm_key"]).transform("sum")
        mx=v.groupby(x["storm_key"]).transform("max")
        mean=v.groupby(x["storm_key"]).transform("mean")
        x[f"stormrel_{c}_share"]=np.where(total>0,v/total,np.nan)
        x[f"stormrel_{c}_to_max"]=np.where(mx>0,v/mx,np.nan)
        x[f"stormrel_{c}_to_mean"]=np.where(mean>0,v/mean,np.nan)
    return x

def fit_eval(df: pd.DataFrame, features: List[str], kind: str):
    y=(df["totalObligatedFunding"]>=200_000_000).astype(int).to_numpy()
    pred=np.zeros(len(df),dtype=int)
    prob=np.zeros(len(df),dtype=float)
    for yr in sorted(df["fyDeclared"].astype(int).unique()):
        te=df["fyDeclared"].astype(int).to_numpy()==yr
        tr=~te
        Xtr=normalize_model_frame(df.loc[tr,features])
        Xte=normalize_model_frame(df.loc[te,features])
        if kind=="log":
            model=LogisticRegression(max_iter=5000,class_weight="balanced",C=0.5)
        else:
            model=RandomForestClassifier(
                n_estimators=700,random_state=1000+yr,
                class_weight="balanced_subsample",max_features="sqrt"
            )
        pipe=prep_pipeline(Xtr,model)
        pipe.fit(Xtr,y[tr])
        pred[te]=pipe.predict(Xte)
        if hasattr(pipe[-1],"predict_proba"):
            prob[te]=pipe.predict_proba(Xte)[:,1]
    return {
        "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
        "lower_recall":float(recall_score(y,pred,pos_label=0)),
        "middle_recall":float(recall_score(y,pred,pos_label=1)),
        "confusion_matrix":confusion_matrix(y,pred,labels=[0,1]).tolist(),
        "predictions":pred.tolist(),
        "probabilities":prob.tolist(),
    }

def main():
    master=normalize_master(pd.read_excel(MASTER))
    ma=fetch_all_mission_assignments()
    sem,_=build_semantic_rollup(master,ma)
    df=master.merge(sem,on="disasterNumber",how="left")
    h=df[df["incidentType"]=="Hurricane"].copy()
    titles=load_fema_declaration_titles(master)
    h["declarationTitle"]=h["disasterNumber"].astype(int).map(titles).fillna("")
    h=add_relative_features(h)

    # Features are created over ALL hurricane rows before target filtering.
    rel=[c for c in h.columns if c.startswith("stormrel_")]
    current=[c for c in CURRENT_19 if c in h.columns]
    semcols=[
        c for c in h.columns if (c.startswith("sem_") or c.startswith("ma_"))
        and h[c].notna().sum()>=2 and h[c].nunique(dropna=True)>1
    ]

    # Lower/middle only; extreme handled by separate gate.
    evaldf=h[
        (h["totalObligatedFunding"]>=50_000_000)
        & (h["totalObligatedFunding"]<500_000_000)
    ].copy().reset_index(drop=True)

    sets={
        "current19":current,
        "semantics":current+semcols,
        "semantics_stormrelative":current+semcols+rel,
        "stormrelative_compact":current+[
            c for c in rel if any(k in c for k in [
                "missionAssignmentCount","responseComplexityScore",
                "uniqueAgencyCount","sem_type_dfa_count","sem_esf_3_count",
                "sem_topic_power_count","sem_topic_engineering_infrastructure_count",
                "sem_topic_logistics_commodities_count","sem_topic_debris_count"
            ])
        ],
    }
    results={}
    predtab=evaldf[["disasterNumber","state","fyDeclared","declarationTitle","storm_key","totalObligatedFunding"]].copy()
    predtab["actual_mid"]=(evaldf["totalObligatedFunding"]>=200_000_000).astype(int)

    for name,features in sets.items():
        for kind in ["log","rf"]:
            r=fit_eval(evaldf,features,kind)
            results[f"{name}_{kind}"]={"feature_count":len(features),**r}
            predtab[f"{name}_{kind}_pred"]=r["predictions"]
            predtab[f"{name}_{kind}_prob"]=r["probabilities"]

    predtab.to_csv(OUT/"predictions.csv",index=False)
    h[["disasterNumber","state","fyDeclared","declarationTitle","storm_key","totalObligatedFunding","storm_declaration_count"]+rel].to_csv(
        OUT/"all108_storm_relative_features.csv",index=False
    )
    summary={"n_all_hurricanes":int(len(h)),"n_lower_middle_eval":int(len(evaldf)),"results":results}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

    md=["# Storm-relative hurricane lower router","",f"- All Hurricane declarations enriched: **{len(h)}**",
        f"- $50M-$500M hurricane evaluation cases: **{len(evaldf)}**","","| Model | Lower recall | Middle recall | BA |","|---|---:|---:|---:|"]
    for name,r in results.items():
        md.append(f"| {name} | {r['lower_recall']:.1%} | {r['middle_recall']:.1%} | {r['balanced_accuracy']:.3f} |")
    (OUT/"summary.md").write_text("\n".join(md),encoding="utf-8")
    print("\n".join(md))

if __name__=="__main__":
    main()
