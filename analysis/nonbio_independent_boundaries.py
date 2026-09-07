#!/usr/bin/env python3
"""
Strict-LFYO independent-boundary audit.

Goal
----
Remove direct band competition by modeling the two problematic boundaries
with separate mechanisms:

LOW boundary:
  Among true/predicted low cases >=100K and <50M, fit a continuous log-dollar
  regressor and classify at the fixed $1M boundary.

UPPER boundary:
  Keep the validated semext four-band sentinel for <50M vs >=50M.

Stage 1 (<100K vs >=100K) remains the validated semantics RF with inner-LFYO
macro-F1 threshold.

Final reported six bands remain unchanged.
Biological remains excluded/frozen.
High hierarchy remains frozen.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import recall_score, precision_score

from mission_semantic_audit import (
    CURRENT_19, build_semantic_rollup, fetch_all_mission_assignments,
    normalize_master, normalize_model_frame, prep_pipeline,
)
from external_severity_ablation import build_external
from nonbio_hazard_hierarchy import initial_mechanism_counts
from nonbio_outage_rescue import build_eaglei_all
from nonbio_all_ranges import BANDS, funding_band, high22_predict, six_band_metrics, valid_cols
from nonbio_root_recall import fit_log, proba, inner_oof_scores, recall_first_threshold
from nonbio_root_cascade import verifier_oof, threshold_for_recall, fit_verifier
from nonbio_low_thresholds import inner_binary_oof, choose_threshold
from nonbio_fourband_sentinel import (
    sentinel_oof, perfect_high_veto_threshold, fit_multi, low_probability,
)

ROOT=Path(__file__).resolve().parents[1]
MASTER=ROOT/"master_openfema_40plus.xlsx"
OUT=ROOT/"audit_outputs"/"nonbio_independent_boundaries"
OUT.mkdir(parents=True,exist_ok=True)
LOW_BANDS=["0-100K","100K-1M","1M-50M"]


def fit_stage1(train, semfeat, seed):
    low=train[train["target_clean"]<50_000_000].copy()
    y=(low["target_clean"]>=100_000).astype(int)
    X=normalize_model_frame(low[semfeat])
    m=RandomForestClassifier(
        n_estimators=900,random_state=seed,class_weight="balanced_subsample",
        max_features="sqrt",min_samples_leaf=1,n_jobs=-1,
    )
    p=prep_pipeline(X,m); p.fit(X,y); return p


def stage1_prob(m,df,features):
    return m.predict_proba(normalize_model_frame(df[features]))[:,1]


def make_reg(kind, seed):
    if kind=="ridge":
        return Ridge(alpha=1.0)
    if kind=="rf":
        return RandomForestRegressor(
            n_estimators=900,random_state=seed,max_features="sqrt",
            min_samples_leaf=1,n_jobs=-1
        )
    if kind=="extra":
        return ExtraTreesRegressor(
            n_estimators=900,random_state=seed,max_features="sqrt",
            min_samples_leaf=1,n_jobs=-1
        )
    if kind=="gbr":
        return GradientBoostingRegressor(
            random_state=seed,n_estimators=500,learning_rate=0.03,
            max_depth=3,loss="huber"
        )
    if kind=="hist":
        return HistGradientBoostingRegressor(
            random_state=seed,learning_rate=0.05,max_iter=400,
            max_leaf_nodes=15,l2_regularization=1.0
        )
    raise ValueError(kind)


def fit_reg(train,features,kind,seed):
    t=train[(train["target_clean"]>=100_000)&(train["target_clean"]<50_000_000)].copy()
    X=normalize_model_frame(t[features])
    y=np.log1p(t["target_clean"].to_numpy(float))
    m=make_reg(kind,seed)
    p=prep_pipeline(X,m); p.fit(X,y); return p


def predict_reg(m,df,features):
    return np.expm1(m.predict(normalize_model_frame(df[features])))


def low_metrics(df,col):
    pb={}; rs=[]
    for b in LOW_BANDS:
        z=df[df["actual_band"]==b]
        c=int((z[col]==b).sum()); n=int(len(z)); r=c/n if n else None
        pb[b]={"correct":c,"total":n,"recall":r}
        if r is not None: rs.append(r)
    return {
        "accuracy":float((df[col]==df["actual_band"]).mean()),
        "macro_recall":float(np.mean(rs)),
        "per_band":pb,
    }


def main():
    master=normalize_master(pd.read_excel(MASTER))
    master["target_clean"]=pd.to_numeric(
        master["totalObligatedFunding"],errors="coerce"
    ).fillna(0).clip(lower=0)
    master["actual_band"]=master["target_clean"].map(funding_band)

    ma=fetch_all_mission_assignments()
    sem,_=build_semantic_rollup(master,ma)
    ext,ext_audit=build_external(master)
    mech=initial_mechanism_counts(master,ma)
    print("Building EAGLE-I Hurricane features...",flush=True)
    eag=build_eaglei_all(master)

    df=(master.merge(sem,on="disasterNumber",how="left")
        .merge(ext,on="disasterNumber",how="left")
        .merge(mech,on="disasterNumber",how="left")
        .merge(eag,on="disasterNumber",how="left"))
    df["initial_usace_esf3_dfa_count"]=df["initial_usace_esf3_dfa_count"].fillna(0).astype(int)
    nonbio=df[df["incidentType"]!="Biological"].copy().reset_index(drop=True)

    current=valid_cols(nonbio,CURRENT_19)
    semcols=valid_cols(nonbio,[c for c in nonbio.columns if c.startswith("sem_") or c.startswith("ma_")])
    semfeat=list(dict.fromkeys(current+semcols))
    extcols=valid_cols(nonbio,[c for c in nonbio.columns
                               if c.startswith("nhc_") or c.startswith("noaa_") or c.startswith("calfire_")])
    semext=list(dict.fromkeys(semfeat+extcols))

    high_all=nonbio[nonbio["target_clean"]>=50_000_000].copy()
    hc=valid_cols(high_all,CURRENT_19)
    hs=valid_cols(high_all,[c for c in high_all.columns if c.startswith("sem_") or c.startswith("ma_")])
    he=valid_cols(high_all,[c for c in high_all.columns
                            if c.startswith("nhc_") or c.startswith("noaa_") or c.startswith("calfire_")])
    high_gate_features=hc+hs+he
    high_lower_features=hc+hs

    variants={}
    for feats_name,feats in [("current19",current),("semantics",semfeat)]:
        for kind in ["ridge","rf","extra","gbr"]:
            variants[f"{feats_name}_{kind}"]=(feats,kind)
    rows={k:[] for k in variants}
    oracle={k:[] for k in variants}

    for fy in sorted(nonbio["fyDeclared"].astype(int).unique()):
        train=nonbio[nonbio["fyDeclared"].astype(int)!=fy].copy()
        test=nonbio[nonbio["fyDeclared"].astype(int)==fy].copy()
        train_high=train[train["target_clean"]>=50_000_000].copy()

        # Frozen upstream root.
        oof_cur=inner_oof_scores(train,current).rename(columns={"prob":"pcur"})
        oof_sem=inner_oof_scores(train,semext).rename(columns={"prob":"psem"})
        th_cur,_=recall_first_threshold(oof_cur.rename(columns={"pcur":"prob"}))
        th_sem,_=recall_first_threshold(oof_sem.rename(columns={"psem":"prob"}))
        oo=oof_cur[["disasterNumber","pcur"]].merge(
            oof_sem[["disasterNumber","psem"]],on="disasterNumber",how="inner")
        tm=train.merge(oo,on="disasterNumber",how="left")
        tm["candidate"]=(tm["pcur"]>=th_cur)|(tm["psem"]>=th_sem)
        cand_train=tm[tm["candidate"]].copy()
        voof=verifier_oof(cand_train,semext,"log",60000)
        vth,_=threshold_for_recall(voof,0.95)
        verifier=fit_verifier(cand_train,semext,"log",80000+fy)

        cur_root=fit_log(train,current); sem_root=fit_log(train,semext)
        pcur=proba(cur_root,test,current); psem=proba(sem_root,test,semext)
        candidate=(pcur>=th_cur)|(psem>=th_sem)
        upstream=np.zeros(len(test),dtype=int)
        ci=np.flatnonzero(candidate)
        if len(ci):
            vp=verifier.predict_proba(normalize_model_frame(test.iloc[ci][semext]))[:,1]
            upstream[ci]=(vp>=vth).astype(int)

        # Frozen semext sentinel.
        soof=sentinel_oof(train,semext,"log",200000)
        sth,_=perfect_high_veto_threshold(soof)
        strain=train[train["target_clean"]>=1_000_000].copy()
        smodel=fit_multi(strain,semext,"log",220000+fy)
        final_root=upstream.copy()
        si=np.flatnonzero(upstream==1)
        if len(si):
            pp=low_probability(smodel,test.iloc[si],semext)
            final_root[si[pp>=sth]]=0

        # Frozen stage1 threshold/model.
        s1oof=inner_binary_oof(train,semfeat,stage=1,kind="rf",seedbase=110000)
        s1th,_=choose_threshold(s1oof,"macro_f1")
        s1=fit_stage1(train,semfeat,130000+fy)

        pred_high_rows=test.loc[final_root==1].copy()
        high_map=high22_predict(
            train_high,pred_high_rows,high_gate_features,high_lower_features
        ) if not pred_high_rows.empty else {}

        for name,(features,kind) in variants.items():
            reg=fit_reg(train,features,kind,300000+fy)

            def pred_low(dd):
                if dd.empty: return np.array([],dtype=object)
                p1=stage1_prob(s1,dd,semfeat)
                out=np.full(len(dd),"0-100K",dtype=object)
                ii=np.flatnonzero(p1>=s1th)
                if len(ii):
                    dollars=predict_reg(reg,dd.iloc[ii],features)
                    out[ii]=np.where(dollars>=1_000_000,"1M-50M","100K-1M")
                return out.astype(str)

            true_low=test[test["target_clean"]<50_000_000].copy()
            op=pred_low(true_low)
            for (_,r),p in zip(true_low.iterrows(),op):
                oracle[name].append({
                    "disasterNumber":int(r["disasterNumber"]),
                    "funding":float(r["target_clean"]),
                    "actual_band":r["actual_band"],"pred":str(p)
                })

            pred_low_rows=test.loc[final_root==0].copy()
            lp=pred_low(pred_low_rows)
            low_map={int(dn):str(p) for dn,p in zip(pred_low_rows["disasterNumber"],lp)}

            for j,(_,r) in enumerate(test.iterrows()):
                dn=int(r["disasterNumber"]); rp=int(final_root[j])
                final=high_map[dn] if rp else low_map[dn]
                rows[name].append({
                    "disasterNumber":dn,"state":r["state"],"incidentType":r["incidentType"],
                    "fyDeclared":int(r["fyDeclared"]),"actual_band":r["actual_band"],
                    "root_actual_high":int(r["target_clean"]>=50_000_000),
                    "root_pred_high":rp,"final_pred":final
                })

    results={}
    for name in variants:
        p=pd.DataFrame(rows[name]); o=pd.DataFrame(oracle[name])
        p.to_csv(OUT/f"{name}_predictions.csv",index=False)
        o.to_csv(OUT/f"{name}_oracle_low.csv",index=False)
        results[name]={
            "oracle_low":low_metrics(o,"pred"),
            "end_to_end":six_band_metrics(p,"final_pred"),
            "root_high_recall":float(recall_score(
                p["root_actual_high"],p["root_pred_high"],pos_label=1,zero_division=0))
        }

    ext_audit.to_csv(OUT/"external_match_audit.csv",index=False)
    (OUT/"summary.json").write_text(json.dumps({"results":results},indent=2))

    md=["# Independent-boundary continuous low specialist","","| Variant | Oracle low acc | Oracle macro | 100K-1M | 1M-50M | End-to-end | Macro | 50-200M | 200-500M | 500M+ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name,r in results.items():
        o=r["oracle_low"]; e=r["end_to_end"]; pb=e["per_band"]
        md.append(
            f"| {name} | {o['accuracy']:.1%} | {o['macro_recall']:.1%} | "
            f"{o['per_band']['100K-1M']['correct']}/{o['per_band']['100K-1M']['total']} ({o['per_band']['100K-1M']['recall']:.1%}) | "
            f"{o['per_band']['1M-50M']['correct']}/{o['per_band']['1M-50M']['total']} ({o['per_band']['1M-50M']['recall']:.1%}) | "
            f"{e['overall_correct']}/{e['overall_total']} ({e['overall_accuracy']:.1%}) | {e['macro_recall']:.1%} | "
            f"{pb['50-200M']['correct']}/{pb['50-200M']['total']} ({pb['50-200M']['recall']:.1%}) | "
            f"{pb['200-500M']['correct']}/{pb['200-500M']['total']} ({pb['200-500M']['recall']:.1%}) | "
            f"{pb['500M+']['correct']}/{pb['500M+']['total']} ({pb['500M+']['recall']:.1%}) |"
        )
    (OUT/"summary.md").write_text("\n".join(md))
    print("\n".join(md))

if __name__=="__main__": main()
