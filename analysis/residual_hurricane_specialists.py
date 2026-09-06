#!/usr/bin/env python3
"""
Residual hurricane specialist audit.

Freezes the confirmed non-Biological hierarchy and studies only hurricane cases.

A) Lower specialist: $50M-$200M vs $200M-$500M among hurricane cases below $500M.
B) Extreme specialist: <$500M vs $500M+ among high-value hurricane cases.

All reported model metrics use outer leave-fiscal-year-out.
Candidate feature families are defined a priori from the mechanism audit.
No funding-derived field is used as a predictor.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score

from mission_semantic_audit import (
    CURRENT_19, build_semantic_rollup, fetch_all_mission_assignments,
    normalize_master, normalize_model_frame, prep_pipeline,
)
from external_severity_ablation import build_external
from nonbio_hazard_hierarchy import initial_mechanism_counts

ROOT=Path(__file__).resolve().parents[1]
MASTER=ROOT/"master_openfema_40plus.xlsx"
OUT=ROOT/"audit_outputs"/"residual_hurricane_specialists"
OUT.mkdir(parents=True,exist_ok=True)

def valid_cols(df, cols):
    return [c for c in cols if c in df.columns and df[c].notna().sum()>=2 and df[c].nunique(dropna=True)>1]

def lfyo_binary(df: pd.DataFrame, features: List[str], target, kind="log") -> Dict:
    y=np.asarray(target,dtype=int)
    pred=np.zeros(len(df),dtype=int)
    prob=np.zeros(len(df),dtype=float)
    fold=[]
    for fy in sorted(df["fyDeclared"].astype(int).unique()):
        te=df["fyDeclared"].astype(int).to_numpy()==fy
        tr=~te
        if len(np.unique(y[tr]))<2:
            pred[te]=0; prob[te]=0.0
            fold.append({"fy":int(fy),"learnable":False,"n":int(te.sum())})
            continue
        Xtr=normalize_model_frame(df.loc[tr,features])
        Xte=normalize_model_frame(df.loc[te,features])
        if kind=="rf":
            model=RandomForestClassifier(
                n_estimators=700,random_state=4200+fy,
                class_weight="balanced_subsample",max_features="sqrt",
                min_samples_leaf=1
            )
        else:
            model=LogisticRegression(max_iter=5000,class_weight="balanced",C=0.5)
        pipe=prep_pipeline(Xtr,model)
        pipe.fit(Xtr,y[tr])
        pred[te]=pipe.predict(Xte)
        prob[te]=pipe.predict_proba(Xte)[:,1]
        fold.append({"fy":int(fy),"learnable":True,"n":int(te.sum())})
    return {
        "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
        "negative_recall":float(recall_score(y,pred,pos_label=0,zero_division=0)),
        "positive_recall":float(recall_score(y,pred,pos_label=1,zero_division=0)),
        "confusion_matrix":confusion_matrix(y,pred,labels=[0,1]).tolist(),
        "predictions":pred.tolist(),
        "probabilities":prob.tolist(),
        "folds":fold,
    }

def main():
    master=normalize_master(pd.read_excel(MASTER))
    ma=fetch_all_mission_assignments()
    sem,_=build_semantic_rollup(master,ma)
    ext,_=build_external(master)
    mech=initial_mechanism_counts(master,ma)
    df=(master.merge(sem,on="disasterNumber",how="left")
              .merge(ext,on="disasterNumber",how="left")
              .merge(mech,on="disasterNumber",how="left"))
    df["initial_usace_esf3_dfa_count"]=df["initial_usace_esf3_dfa_count"].fillna(0)

    h=df[(df["incidentType"]=="Hurricane") & (df["totalObligatedFunding"]>=50_000_000)].copy().reset_index(drop=True)

    current=valid_cols(h,CURRENT_19)
    semcols=valid_cols(h,[c for c in h.columns if c.startswith("sem_") or c.startswith("ma_")])
    extcols=valid_cols(h,[c for c in h.columns if c.startswith("nhc_") or c.startswith("noaa_")])

    # Pre-specified interpretable operational families.
    response_scale=valid_cols(h,[
        "missionAssignmentCount","uniqueAgencyCount","uniqueMaTypeCount","uniquePriorityCount",
        "responseComplexityScore","missionDensity","agencyDensity",
        "sem_unique_missions","sem_unique_agencies","sem_unique_esf",
        "sem_type_dfa_count","sem_type_fos_count",
    ])
    infrastructure=valid_cols(h,[
        "initial_usace_esf3_dfa_count",
        "sem_esf_3_count","sem_esf_3_share","sem_esf_4_count","sem_esf_4_share",
        "sem_topic_power_count","sem_topic_power_share",
        "sem_topic_engineering_infrastructure_count","sem_topic_engineering_infrastructure_share",
        "sem_topic_debris_count","sem_topic_debris_share",
        "sem_topic_emergency_power_count","sem_topic_emergency_power_share",
        "sem_topic_temporary_housing_count","sem_topic_temporary_housing_share",
        "sem_topic_logistics_commodities_count","sem_topic_logistics_commodities_share",
    ])
    assistance_mix=valid_cols(h,[
        "sem_type_dfa_count","sem_type_dfa_share","sem_type_fos_count","sem_type_fos_share",
        "sem_priority_emergency_count","sem_priority_emergency_share",
        "sem_priority_urgent_count","sem_priority_urgent_share",
        "sem_priority_normal_count","sem_priority_normal_share",
    ])
    timing=valid_cols(h,[
        "durationDays","declarationDelayDays",
        "sem_mean_pop_duration_days","sem_median_pop_duration_days","sem_max_pop_duration_days",
        "sem_days_to_first_received","sem_first7_mission_count","sem_first14_mission_count",
    ])
    physical=extcols

    families={
        "current19":current,
        "semantics":list(dict.fromkeys(current+semcols)),
        "response_scale":list(dict.fromkeys(current+response_scale)),
        "infrastructure":list(dict.fromkeys(current+infrastructure)),
        "response_plus_infra":list(dict.fromkeys(current+response_scale+infrastructure)),
        "infra_mix_timing":list(dict.fromkeys(current+infrastructure+assistance_mix+timing)),
        "semantics_external":list(dict.fromkeys(current+semcols+physical)),
        "compact_mechanism":list(dict.fromkeys(
            response_scale+infrastructure+assistance_mix+timing+[
                c for c in physical if any(k in c for k in ["wind","pressure","damage","fatal","event_count"])
            ]
        )),
    }
    families={k:valid_cols(h,v) for k,v in families.items() if len(valid_cols(h,v))>0}

    # A: lower/middle.
    lm=h[h["totalObligatedFunding"]<500_000_000].copy().reset_index(drop=True)
    ylm=(lm["totalObligatedFunding"]>=200_000_000).astype(int).to_numpy()
    lower_results={}
    lower_pred=lm[["disasterNumber","state","fyDeclared","totalObligatedFunding"]].copy()
    lower_pred["actual_mid"]=ylm
    for name,feats in families.items():
        for kind in ["log","rf"]:
            r=lfyo_binary(lm,feats,ylm,kind)
            key=f"{name}_{kind}"
            lower_results[key]={"feature_count":len(feats),**r}
            lower_pred[key+"_pred"]=r["predictions"]
            lower_pred[key+"_prob"]=r["probabilities"]
    lower_pred.to_csv(OUT/"lower_middle_predictions.csv",index=False)

    # B: extreme.
    yex=(h["totalObligatedFunding"]>=500_000_000).astype(int).to_numpy()
    extreme_results={}
    extreme_pred=h[["disasterNumber","state","fyDeclared","totalObligatedFunding"]].copy()
    extreme_pred["actual_extreme"]=yex
    for name,feats in families.items():
        for kind in ["log","rf"]:
            r=lfyo_binary(h,feats,yex,kind)
            key=f"{name}_{kind}"
            extreme_results[key]={"feature_count":len(feats),**r}
            extreme_pred[key+"_pred"]=r["predictions"]
            extreme_pred[key+"_prob"]=r["probabilities"]
    extreme_pred.to_csv(OUT/"extreme_predictions.csv",index=False)

    # Case-control feature table for the three residual errors plus correctly classified comparators.
    audit_cols=valid_cols(h,list(dict.fromkeys(
        ["disasterNumber","state","fyDeclared","totalObligatedFunding"]+
        response_scale+infrastructure+assistance_mix+timing+physical
    )))
    h[audit_cols].to_csv(OUT/"hurricane_feature_audit.csv",index=False)

    # Rank models: balance first, then minimum class recall.
    def rank_table(results):
        rows=[]
        for k,r in results.items():
            rows.append({
                "model":k,
                "balanced_accuracy":r["balanced_accuracy"],
                "negative_recall":r["negative_recall"],
                "positive_recall":r["positive_recall"],
                "min_recall":min(r["negative_recall"],r["positive_recall"]),
                "feature_count":r["feature_count"],
            })
        return pd.DataFrame(rows).sort_values(
            ["min_recall","balanced_accuracy","feature_count"],
            ascending=[False,False,True]
        )
    lrank=rank_table(lower_results); erank=rank_table(extreme_results)
    lrank.to_csv(OUT/"lower_model_ranking.csv",index=False)
    erank.to_csv(OUT/"extreme_model_ranking.csv",index=False)

    summary={
        "hurricane_high_n":int(len(h)),
        "hurricane_lower_middle_n":int(len(lm)),
        "lower_middle_top":lrank.head(10).to_dict(orient="records"),
        "extreme_top":erank.head(10).to_dict(orient="records"),
        "lower_middle_results":lower_results,
        "extreme_results":extreme_results,
        "note":"Model-family comparison is developmental. Families were defined from mechanism hypotheses before this run; no target-derived predictor is included.",
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

    md=["# Residual hurricane specialists","",
        f"- High-value hurricanes: **{len(h)}**",
        f"- Lower/middle hurricanes (<$500M): **{len(lm)}**","",
        "## Best lower-vs-middle models","",
        "| Model | Lower recall | Middle recall | BA |",
        "|---|---:|---:|---:|"]
    for _,r in lrank.head(8).iterrows():
        md.append(f"| {r.model} | {r.negative_recall:.1%} | {r.positive_recall:.1%} | {r.balanced_accuracy:.3f} |")
    md += ["","## Best hurricane extreme models","","| Model | <500M recall | 500M+ recall | BA |","|---|---:|---:|---:|"]
    for _,r in erank.head(8).iterrows():
        md.append(f"| {r.model} | {r.negative_recall:.1%} | {r.positive_recall:.1%} | {r.balanced_accuracy:.3f} |")
    (OUT/"summary.md").write_text("\n".join(md),encoding="utf-8")
    print("\n".join(md))

if __name__=="__main__":
    main()
