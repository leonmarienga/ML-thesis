#!/usr/bin/env python3
"""
Strict-LFYO hazard-specific >=$50M boundary sentinels.

Benchmark architecture:
- accepted recall-first upstream >=$50M root
- validated global semext four-band sentinel
- lower boundary: semantics ExtraTrees log-dollar regressor at $1M
- frozen high hierarchy

New test:
Apply an additional one-way hazard-specific veto ONLY to Hurricanes and/or Fires
already selected as >=$50M after the global sentinel.

Each hazard verifier is trained only on outer-training rows of that hazard with
funding >=$1M. Its inner-LFYO threshold is chosen to preserve 100% of true
>=50M hazard cases in outer-training OOF predictions.

Rare high-value hazards (Flood, Typhoon, Tropical Storm, etc.) are left
untouched.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
OUT=ROOT/"audit_outputs"/"nonbio_hazard_boundary"
OUT.mkdir(parents=True,exist_ok=True)
LOW_BANDS=["0-100K","100K-1M","1M-50M"]


def fit_stage1(train, semfeat, seed):
    from sklearn.ensemble import RandomForestClassifier
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


def fit_low_reg(train,features,seed):
    t=train[(train["target_clean"]>=100_000)&(train["target_clean"]<50_000_000)].copy()
    X=normalize_model_frame(t[features])
    y=np.log1p(t["target_clean"].to_numpy(float))
    m=ExtraTreesRegressor(
        n_estimators=900,random_state=seed,max_features="sqrt",
        min_samples_leaf=1,n_jobs=-1
    )
    p=prep_pipeline(X,m); p.fit(X,y); return p


def predict_low_reg(m,df,features):
    return np.expm1(m.predict(normalize_model_frame(df[features])))


def build_low_predictor(train,semfeat,current,outer_fy):
    s1oof=inner_binary_oof(train,semfeat,stage=1,kind="rf",seedbase=110000)
    s1th,_=choose_threshold(s1oof,"macro_f1")
    s1=fit_stage1(train,semfeat,130000+outer_fy)
    reg=fit_low_reg(train,semfeat,300000+outer_fy)

    def pred(dd):
        if dd.empty: return np.array([],dtype=object)
        p1=stage1_prob(s1,dd,semfeat)
        out=np.full(len(dd),"0-100K",dtype=object)
        ii=np.flatnonzero(p1>=s1th)
        if len(ii):
            dollars=predict_low_reg(reg,dd.iloc[ii],semfeat)
            out[ii]=np.where(dollars>=1_000_000,"1M-50M","100K-1M")
        return out.astype(str)
    return pred


def fit_hazard_binary(train,features,kind,seed):
    y=(train["target_clean"]>=50_000_000).astype(int)
    X=normalize_model_frame(train[features])
    if kind=="rf":
        m=RandomForestClassifier(
            n_estimators=900,random_state=seed,class_weight="balanced_subsample",
            max_features="sqrt",min_samples_leaf=1,n_jobs=-1
        )
    else:
        m=LogisticRegression(max_iter=5000,class_weight="balanced",C=0.5)
    p=prep_pipeline(X,m); p.fit(X,y); return p


def hazard_oof(outer_train,hazard,features,kind,seedbase):
    d=outer_train[
        (outer_train["incidentType"]==hazard)
        &(outer_train["target_clean"]>=1_000_000)
    ].copy()
    rows=[]
    for fy in sorted(d["fyDeclared"].astype(int).unique()):
        tr=d[d["fyDeclared"].astype(int)!=fy].copy()
        te=d[d["fyDeclared"].astype(int)==fy].copy()
        ytr=(tr["target_clean"]>=50_000_000).astype(int)
        if te.empty or ytr.nunique()<2: continue
        m=fit_hazard_binary(tr,features,kind,seedbase+fy)
        pp=m.predict_proba(normalize_model_frame(te[features]))[:,1]
        for (_,r),p in zip(te.iterrows(),pp):
            rows.append({
                "disasterNumber":int(r["disasterNumber"]),
                "actual_high":int(r["target_clean"]>=50_000_000),
                "p_high":float(p),
                "fy":int(fy),
            })
    return pd.DataFrame(rows)


def safe_high_threshold(oof):
    if oof.empty or int(oof["actual_high"].sum())==0:
        return 0.0,{"recall":None,"precision":None,"fp":None}
    th=float(oof.loc[oof["actual_high"]==1,"p_high"].min())
    yp=(oof["p_high"]>=th).astype(int)
    return th,{
        "recall":float(recall_score(oof["actual_high"],yp,pos_label=1,zero_division=0)),
        "precision":float(precision_score(oof["actual_high"],yp,pos_label=1,zero_division=0)),
        "fp":int(((oof["actual_high"]==0)&(yp==1)).sum()),
        "tn":int(((oof["actual_high"]==0)&(yp==0)).sum()),
    }


def main():
    master=normalize_master(pd.read_excel(MASTER))
    master["target_clean"]=pd.to_numeric(master["totalObligatedFunding"],errors="coerce").fillna(0).clip(lower=0)
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

    variants={
        "global_only":{},
        "hurricane_log":{"Hurricane":"log"},
        "hurricane_rf":{"Hurricane":"rf"},
        "fire_log":{"Fire":"log"},
        "fire_rf":{"Fire":"rf"},
        "hurr_log_fire_log":{"Hurricane":"log","Fire":"log"},
        "hurr_rf_fire_rf":{"Hurricane":"rf","Fire":"rf"},
        "hurr_log_fire_rf":{"Hurricane":"log","Fire":"rf"},
    }
    rows={k:[] for k in variants}
    diag=[]

    for fy in sorted(nonbio["fyDeclared"].astype(int).unique()):
        train=nonbio[nonbio["fyDeclared"].astype(int)!=fy].copy()
        test=nonbio[nonbio["fyDeclared"].astype(int)==fy].copy()
        train_high=train[train["target_clean"]>=50_000_000].copy()

        # Frozen accepted upstream root.
        oof_cur=inner_oof_scores(train,current).rename(columns={"prob":"pcur"})
        oof_sem=inner_oof_scores(train,semext).rename(columns={"prob":"psem"})
        th_cur,_=recall_first_threshold(oof_cur.rename(columns={"pcur":"prob"}))
        th_sem,_=recall_first_threshold(oof_sem.rename(columns={"psem":"prob"}))
        oo=oof_cur[["disasterNumber","pcur"]].merge(oof_sem[["disasterNumber","psem"]],on="disasterNumber",how="inner")
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

        # Frozen global semext sentinel.
        soof=sentinel_oof(train,semext,"log",200000)
        sth,_=perfect_high_veto_threshold(soof)
        strain=train[train["target_clean"]>=1_000_000].copy()
        smodel=fit_multi(strain,semext,"log",220000+fy)
        global_root=upstream.copy()
        si=np.flatnonzero(upstream==1)
        if len(si):
            pp=low_probability(smodel,test.iloc[si],semext)
            global_root[si[pp>=sth]]=0

        low_predict=build_low_predictor(train,semfeat,current,fy)

        # Precompute hazard models/thresholds once per kind/hazard.
        cache={}
        for hazard in ["Hurricane","Fire"]:
            for kind in ["log","rf"]:
                ho=hazard_oof(train,hazard,semext,kind,400000 if kind=="log" else 410000)
                hth,hdiag=safe_high_threshold(ho)
                htrain=train[(train["incidentType"]==hazard)&(train["target_clean"]>=1_000_000)].copy()
                hy=(htrain["target_clean"]>=50_000_000).astype(int)
                model=None
                if len(htrain)>=6 and hy.nunique()>=2:
                    model=fit_hazard_binary(htrain,semext,kind,420000+fy)
                cache[(hazard,kind)]=(hth,hdiag,model)
                diag.append({
                    "outer_fy":fy,"hazard":hazard,"kind":kind,
                    "threshold":hth,"inner_recall":hdiag.get("recall"),
                    "inner_precision":hdiag.get("precision"),"inner_fp":hdiag.get("fp")
                })

        for name,rules in variants.items():
            final_root=global_root.copy()

            for hazard,kind in rules.items():
                hth,hdiag,model=cache[(hazard,kind)]
                idx=np.flatnonzero(
                    (final_root==1)&(test["incidentType"].to_numpy()==hazard)
                )
                if len(idx) and model is not None:
                    hp=model.predict_proba(normalize_model_frame(test.iloc[idx][semext]))[:,1]
                    final_root[idx[hp<hth]]=0

            pred_high_rows=test.loc[final_root==1].copy()
            high_map=high22_predict(train_high,pred_high_rows,high_gate_features,high_lower_features) if not pred_high_rows.empty else {}
            pred_low_rows=test.loc[final_root==0].copy()
            lp=low_predict(pred_low_rows)
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
        p=pd.DataFrame(rows[name])
        p.to_csv(OUT/f"{name}_predictions.csv",index=False)
        y=p["root_actual_high"].to_numpy(int); yp=p["root_pred_high"].to_numpy(int)
        results[name]={
            "root_high_recall":float(recall_score(y,yp,pos_label=1,zero_division=0)),
            "root_high_precision":float(precision_score(y,yp,pos_label=1,zero_division=0)),
            "root_fp":int(((y==0)&(yp==1)).sum()),
            "root_fp_by_hazard":p[(p["root_actual_high"]==0)&(p["root_pred_high"]==1)].incidentType.value_counts().to_dict(),
            "end_to_end":six_band_metrics(p,"final_pred"),
            "high_false_negatives":p[(p["root_actual_high"]==1)&(p["root_pred_high"]==0)][["disasterNumber","state","incidentType","actual_band"]].to_dict(orient="records")
        }

    pd.DataFrame(diag).to_csv(OUT/"hazard_fold_diagnostics.csv",index=False)
    ext_audit.to_csv(OUT/"external_match_audit.csv",index=False)
    (OUT/"summary.json").write_text(json.dumps({"results":results},indent=2))

    md=["# Hazard-specific $50M boundary sentinel audit","",
        "| Variant | High recall | Precision | Root FP | End-to-end | Macro | 100K-1M | 1M-50M | 50-200M | 200-500M | 500M+ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name,r in results.items():
        e=r["end_to_end"]; pb=e["per_band"]
        md.append(
            f"| {name} | {r['root_high_recall']:.1%} | {r['root_high_precision']:.1%} | {r['root_fp']} | "
            f"{e['overall_correct']}/{e['overall_total']} ({e['overall_accuracy']:.1%}) | {e['macro_recall']:.1%} | "
            f"{pb['100K-1M']['correct']}/{pb['100K-1M']['total']} ({pb['100K-1M']['recall']:.1%}) | "
            f"{pb['1M-50M']['correct']}/{pb['1M-50M']['total']} ({pb['1M-50M']['recall']:.1%}) | "
            f"{pb['50-200M']['correct']}/{pb['50-200M']['total']} ({pb['50-200M']['recall']:.1%}) | "
            f"{pb['200-500M']['correct']}/{pb['200-500M']['total']} ({pb['200-500M']['recall']:.1%}) | "
            f"{pb['500M+']['correct']}/{pb['500M+']['total']} ({pb['500M+']['recall']:.1%}) |"
        )
    md += ["","## High false negatives"]
    for name,r in results.items():
        md.append(f"### {name}")
        if not r["high_false_negatives"]: md.append("- None")
        else:
            for x in r["high_false_negatives"]:
                md.append(f"- FEMA {x['disasterNumber']} {x['state']} {x['incidentType']} {x['actual_band']}")
    (OUT/"summary.md").write_text("\n".join(md))
    print("\n".join(md))

if __name__=="__main__": main()
