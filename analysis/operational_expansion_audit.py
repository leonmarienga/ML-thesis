#!/usr/bin/env python3
"""
Operational-expansion / Mission Assignment amendment audit.

Adds target-free amendment dynamics to ALL 971 disasters before high-band
selection. No obligation amount/date/cost-share dollar field is used.

Focus:
- fraction of missions amended
- repeated amendment depth
- USACE/COE amendment intensity
- ESF-3 + DFA amendment intensity
- USACE x ESF-3 x DFA amendment intensity
- early-vs-late mission record expansion where dateReceived is available

Evaluation:
A. Existing current19 + semantics + corrected external extreme gate
B. A + operational expansion
Then fire verifier is applied identically.
Lower router is tested with and without expansion.

Strict outer leave-fiscal-year-out.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score

from mission_semantic_audit import (
    CURRENT_19,
    build_semantic_rollup,
    fetch_all_mission_assignments,
    normalize_master,
    normalize_model_frame,
    prep_pipeline,
)
from external_severity_ablation import build_external

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "operational_expansion"
OUT.mkdir(parents=True, exist_ok=True)

EXTREME = 500_000_000.0

def band(v: float) -> str:
    if v < 200_000_000: return "50-200M"
    if v < 500_000_000: return "200-500M"
    return "500M+"

def is_coe(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.upper().str.contains(
        r"(^|[^A-Z])(COE|USACE)([^A-Z]|$)", regex=True
    )

def build_expansion(master: pd.DataFrame, ma: pd.DataFrame) -> pd.DataFrame:
    ids = set(master["disasterNumber"].dropna().astype(int))
    x = ma.copy()
    x["disasterNumber"] = pd.to_numeric(x["disasterNumber"], errors="coerce").astype("Int64")
    x = x[x["disasterNumber"].isin(ids)].copy()

    x["amend_num"] = pd.to_numeric(x["maAmendNumber"], errors="coerce").fillna(0).clip(lower=0)
    x["support_num"] = pd.to_numeric(x["supportFunction"], errors="coerce")
    x["is_coe"] = is_coe(x["agencyId"]).astype(int)
    x["is_esf3"] = (x["support_num"] == 3).astype(int)
    x["is_dfa"] = x["maType"].fillna("").astype(str).str.upper().str.contains("DFA").astype(int)
    x["is_coe_esf3"] = (x["is_coe"] & x["is_esf3"]).astype(int)
    x["is_esf3_dfa"] = (x["is_esf3"] & x["is_dfa"]).astype(int)
    x["is_coe_esf3_dfa"] = (x["is_coe"] & x["is_esf3"] & x["is_dfa"]).astype(int)

    # Per mission amendment structure; one row per disasterNumber + maId.
    gm = x.groupby(["disasterNumber","maId"], dropna=False)
    per = gm.agg(
        exp_rows=("maId","size"),
        exp_max_amend=("amend_num","max"),
        exp_coe=("is_coe","max"),
        exp_esf3=("is_esf3","max"),
        exp_dfa=("is_dfa","max"),
        exp_coe_esf3=("is_coe_esf3","max"),
        exp_esf3_dfa=("is_esf3_dfa","max"),
        exp_coe_esf3_dfa=("is_coe_esf3_dfa","max"),
    ).reset_index()
    per["exp_amended"] = (per["exp_max_amend"] >= 1).astype(int)
    per["exp_amend2plus"] = (per["exp_max_amend"] >= 2).astype(int)
    per["exp_amend3plus"] = (per["exp_max_amend"] >= 3).astype(int)
    per["exp_amend5plus"] = (per["exp_max_amend"] >= 5).astype(int)
    per["exp_amend10plus"] = (per["exp_max_amend"] >= 10).astype(int)
    per["exp_extra_rows"] = (per["exp_rows"] - 1).clip(lower=0)

    # Interaction-weighted amendment counts.
    for flag in [
        "exp_coe","exp_esf3","exp_dfa","exp_coe_esf3","exp_esf3_dfa","exp_coe_esf3_dfa"
    ]:
        per[f"{flag}_amended"] = per[flag] * per["exp_amended"]
        per[f"{flag}_amend2plus"] = per[flag] * per["exp_amend2plus"]
        per[f"{flag}_extra_rows"] = per[flag] * per["exp_extra_rows"]
        per[f"{flag}_amend_depth"] = per[flag] * per["exp_max_amend"]

    def q90(s):
        return float(s.quantile(.90)) if len(s) else np.nan

    g = per.groupby("disasterNumber")
    out = g.agg(
        exp_unique_missions=("maId","nunique"),
        exp_amended_missions=("exp_amended","sum"),
        exp_amend2plus_missions=("exp_amend2plus","sum"),
        exp_amend3plus_missions=("exp_amend3plus","sum"),
        exp_amend5plus_missions=("exp_amend5plus","sum"),
        exp_amend10plus_missions=("exp_amend10plus","sum"),
        exp_extra_rows_total=("exp_extra_rows","sum"),
        exp_max_amendment=("exp_max_amend","max"),
        exp_mean_max_amend=("exp_max_amend","mean"),
        exp_p90_max_amend=("exp_max_amend",q90),
    ).reset_index()

    denom = out["exp_unique_missions"].replace(0,np.nan)
    out["exp_amended_share"] = out["exp_amended_missions"]/denom
    out["exp_amend2plus_share"] = out["exp_amend2plus_missions"]/denom
    out["exp_amend3plus_share"] = out["exp_amend3plus_missions"]/denom
    out["exp_amend5plus_share"] = out["exp_amend5plus_missions"]/denom

    # Aggregate each interaction separately.
    for flag in [
        "exp_coe","exp_esf3","exp_dfa","exp_coe_esf3","exp_esf3_dfa","exp_coe_esf3_dfa"
    ]:
        agg = g.agg(
            **{
                f"{flag}_missions": (flag,"sum"),
                f"{flag}_amended_missions": (f"{flag}_amended","sum"),
                f"{flag}_amend2plus_missions": (f"{flag}_amend2plus","sum"),
                f"{flag}_extra_rows_total": (f"{flag}_extra_rows","sum"),
                f"{flag}_amend_depth_total": (f"{flag}_amend_depth","sum"),
            }
        ).reset_index()
        out = out.merge(agg,on="disasterNumber",how="left")
        d = out[f"{flag}_missions"].replace(0,np.nan)
        out[f"{flag}_amended_share"] = out[f"{flag}_amended_missions"]/d
        out[f"{flag}_amend2plus_share"] = out[f"{flag}_amend2plus_missions"]/d
        out[f"{flag}_extra_rows_per_mission"] = out[f"{flag}_extra_rows_total"]/d
        out[f"{flag}_mean_amend_depth"] = out[f"{flag}_amend_depth_total"]/d

    # Timing of mission creation relative to declaration. DateReceived is not
    # financial, but this is still retrospective unless a t0 window is chosen.
    decl = master[["disasterNumber","declarationDate"]].copy()
    decl["decl_dt"] = pd.to_datetime(decl["declarationDate"],errors="coerce",utc=True)
    first = (
        x.sort_values(["disasterNumber","maId","amend_num"])
         .groupby(["disasterNumber","maId"],dropna=False)
         .head(1)
         .merge(decl[["disasterNumber","decl_dt"]],on="disasterNumber",how="left")
    )
    first["recv_dt"] = pd.to_datetime(first["dateReceived"],errors="coerce",utc=True)
    first["days_from_decl"] = (first["recv_dt"]-first["decl_dt"]).dt.total_seconds()/86400.0
    for day in [7,14,30,60,90]:
        c = (
            first[first["days_from_decl"] <= day]
            .groupby("disasterNumber")["maId"].nunique()
            .rename(f"exp_missions_by_{day}d")
            .reset_index()
        )
        out = out.merge(c,on="disasterNumber",how="left")
        out[f"exp_missions_by_{day}d"] = out[f"exp_missions_by_{day}d"].fillna(0)
        out[f"exp_mission_share_by_{day}d"] = (
            out[f"exp_missions_by_{day}d"]/out["exp_unique_missions"].replace(0,np.nan)
        )

    return out.replace([np.inf,-np.inf],np.nan)

def fit_binary(train, features, target):
    X = normalize_model_frame(train[features])
    m = LogisticRegression(max_iter=5000,class_weight="balanced",C=0.5)
    p = prep_pipeline(X,m)
    p.fit(X,target)
    return p

def fire_threshold(train):
    f=train[train["incidentType"]=="Fire"]
    neg=f.loc[f["totalObligatedFunding"]<EXTREME,"sem_esf_4_share"].dropna()
    pos=f.loc[f["totalObligatedFunding"]>=EXTREME,"sem_esf_4_share"].dropna()
    if len(neg) and len(pos):
        mx,mn=float(neg.max()),float(pos.min())
        return (mx+mn)/2 if mx<mn else mn
    return 0.0

def evaluate(high, gate_features, lower_features, use_fire=True):
    rows=[]
    for yr in sorted(high["fyDeclared"].astype(int).unique()):
        tr=high[high["fyDeclared"].astype(int)!=yr].copy()
        te=high[high["fyDeclared"].astype(int)==yr].copy()

        yg=(tr["totalObligatedFunding"]>=EXTREME).astype(int)
        gate=fit_binary(tr,gate_features,yg)
        gp=gate.predict(normalize_model_frame(te[gate_features]))

        lt=tr[tr["totalObligatedFunding"]<EXTREME].copy()
        yl=(lt["totalObligatedFunding"]>=200_000_000).astype(int)
        lower=fit_binary(lt,lower_features,yl)
        lp=lower.predict(normalize_model_frame(te[lower_features]))

        fth=fire_threshold(tr)
        for (_,r),gpred,lpred in zip(te.iterrows(),gp,lp):
            extreme=int(gpred)
            if use_fire and extreme==1 and r["incidentType"]=="Fire":
                share=float(r["sem_esf_4_share"]) if pd.notna(r["sem_esf_4_share"]) else 0.0
                if share < fth: extreme=0
            pred="500M+" if extreme else ("200-500M" if int(lpred) else "50-200M")
            rows.append({
                "disasterNumber":int(r["disasterNumber"]),
                "state":r["state"],"incidentType":r["incidentType"],
                "fyDeclared":int(r["fyDeclared"]),
                "actual_band":r["actual_band"],"pred":pred
            })
    p=pd.DataFrame(rows)
    per={}
    for b in ["50-200M","200-500M","500M+"]:
        m=p["actual_band"]==b
        c=int((p.loc[m,"pred"]==b).sum()); n=int(m.sum())
        per[b]={"correct":c,"total":n,"recall":c/n if n else None}
    return p,per

def main():
    master=normalize_master(pd.read_excel(MASTER))
    ma=fetch_all_mission_assignments()
    sem,_=build_semantic_rollup(master,ma)
    ext,match=build_external(master)
    expansion=build_expansion(master,ma)
    match.to_csv(OUT/"external_match_audit.csv",index=False)
    expansion.to_csv(OUT/"expansion_features_971.csv",index=False)

    df=master.merge(sem,on="disasterNumber",how="left").merge(ext,on="disasterNumber",how="left").merge(expansion,on="disasterNumber",how="left")
    high=df[(df["incidentType"]!="Biological")&(df["totalObligatedFunding"]>=50_000_000)].copy().reset_index(drop=True)
    high["actual_band"]=high["totalObligatedFunding"].map(band)

    current=[c for c in CURRENT_19 if c in high.columns]
    semcols=[c for c in high.columns if (c.startswith("sem_") or c.startswith("ma_")) and high[c].notna().sum()>=2 and high[c].nunique(dropna=True)>1]
    extcols=[c for c in high.columns if (c.startswith("nhc_") or c.startswith("noaa_") or c.startswith("calfire_")) and high[c].notna().sum()>=2 and high[c].nunique(dropna=True)>1]
    expcols=[c for c in high.columns if c.startswith("exp_") and high[c].notna().sum()>=2 and high[c].nunique(dropna=True)>1]

    sets={
        "baseline_fire":{
            "gate":current+semcols+extcols,
            "lower":current+semcols,
        },
        "expansion_gate":{
            "gate":current+semcols+extcols+expcols,
            "lower":current+semcols,
        },
        "expansion_lower":{
            "gate":current+semcols+extcols,
            "lower":current+semcols+expcols,
        },
        "expansion_both":{
            "gate":current+semcols+extcols+expcols,
            "lower":current+semcols+expcols,
        },
    }
    results={}
    for name,s in sets.items():
        pred,per=evaluate(high,s["gate"],s["lower"],use_fire=True)
        pred.to_csv(OUT/f"{name}_predictions.csv",index=False)
        results[name]={
            "overall_correct":int((pred["actual_band"]==pred["pred"]).sum()),
            "overall_accuracy":float((pred["actual_band"]==pred["pred"]).mean()),
            "per_band":per,
            "errors":pred[pred["actual_band"]!=pred["pred"]][
                ["disasterNumber","state","incidentType","actual_band","pred"]
            ].to_dict(orient="records"),
            "gate_feature_count":len(s["gate"]),
            "lower_feature_count":len(s["lower"]),
        }

    focus_ids=[4086,4611,4671,4830,4339,4344,4724,4827]
    focus_cols=["disasterNumber","state","incidentType","fyDeclared","totalObligatedFunding"]+[
        c for c in expcols if any(k in c for k in [
            "amended_share","amend2plus","extra_rows_per_mission","mean_amend_depth",
            "missions_by_30d","missions_by_60d","missions_by_90d"
        ])
    ]
    high[high["disasterNumber"].isin(focus_ids)][focus_cols].to_csv(OUT/"focus_expansion_features.csv",index=False)

    summary={"expansion_feature_count":len(expcols),"results":results}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    md=["# Operational-expansion audit","",f"- Expansion features tested: **{len(expcols)}**","",
        "| Model | 50-200M | 200-500M | 500M+ | Overall |","|---|---:|---:|---:|---:|"]
    for name,r in results.items():
        p=r["per_band"]
        md.append(
            f"| {name} | {p['50-200M']['correct']}/{p['50-200M']['total']} "
            f"({p['50-200M']['recall']:.1%}) | "
            f"{p['200-500M']['correct']}/{p['200-500M']['total']} "
            f"({p['200-500M']['recall']:.1%}) | "
            f"{p['500M+']['correct']}/{p['500M+']['total']} "
            f"({p['500M+']['recall']:.1%}) | "
            f"{r['overall_correct']}/23 ({r['overall_accuracy']:.1%}) |"
        )
    (OUT/"summary.md").write_text("\n".join(md),encoding="utf-8")
    print("\n".join(md))

if __name__=="__main__":
    main()
