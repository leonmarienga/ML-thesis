#!/usr/bin/env python3
"""
Strict LFYO second semantic key for the Severe Storm $1M boundary.

Architecture tested inside each outer fiscal-year fold:
1. Frozen general low router produces base predictions.
2. Accepted Severe Storm CURRENT_19 RF rescue is reproduced from outer-training OOF.
3. ONLY Severe Storm candidates still predicted $100K-$1M after step 2 may be
   promoted by an independent compact semantic model.

The second key uses MissionAssignment semantic features only (no CURRENT_19 features),
so it tests genuinely additional operational signal. Its threshold is selected only
from outer-training held-out predictions. Guards preserve at least 80% or 90% of
residual candidates whose true funding is below $1M.

This is low-branch only. No >=$50M component is touched. Biological is excluded.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from mission_semantic_audit import (
    CURRENT_19, build_semantic_rollup, fetch_all_mission_assignments,
    normalize_master, normalize_model_frame, prep_pipeline,
)
from nonbio_all_ranges import funding_band, valid_cols
from nonbio_low_thresholds import low_metrics
from nonbio_cross_1m_rescue import base_oof_predictions, fit_outer_base
from nonbio_severe_storm_1m_rescue import (
    fit_specialist as fit_first_specialist,
    rescue_oof_scores as first_oof_scores,
    apply_rescue as apply_first_rescue,
    choose_threshold as choose_first_threshold,
)

ROOT=Path(__file__).resolve().parents[1]
MASTER=ROOT/"master_openfema_40plus.xlsx"
OUT=ROOT/"audit_outputs"/"nonbio_storm_second_semantic_key"
OUT.mkdir(parents=True,exist_ok=True)
HAZARD="Severe Storm"
LOW_BANDS=["0-100K","100K-1M","1M-50M"]

COMPACT_SEMANTIC=[
    "sem_topic_security_law_enforcement_count",
    "sem_topic_security_law_enforcement_share",
    "sem_agency_tva_count",
    "sem_agency_tva_share",
    "sem_esf_7_share",
    "sem_agency_gsa_share",
    "sem_unique_priorities",
    "sem_topic_housing_shelter_count",
    "sem_topic_housing_shelter_share",
    "sem_topic_water_wastewater_count",
    "sem_topic_water_wastewater_share",
    "ma_mean_amendment",
    "ma_amendment_share",
]


def fit_semantic(train,features,kind,seed):
    t=train[(train.incidentType==HAZARD)&(train.target_clean>=100_000)&(train.target_clean<50_000_000)].copy()
    y=(t.target_clean>=1_000_000).astype(int)
    X=normalize_model_frame(t[features])
    if kind=="rf":
        model=RandomForestClassifier(n_estimators=800,random_state=seed,class_weight="balanced_subsample",max_features="sqrt",min_samples_leaf=2,n_jobs=1)
    else:
        model=LogisticRegression(max_iter=5000,class_weight="balanced",C=0.5)
    pipe=prep_pipeline(X,model); pipe.fit(X,y); return pipe


def semantic_oof_scores(train,base_oof,features,kind,outer_fy):
    rows=[]
    for fy in sorted(train.fyDeclared.astype(int).unique()):
        tr=train[train.fyDeclared.astype(int)!=fy].copy()
        cand=base_oof[(base_oof.fy.astype(int)==fy)&(base_oof.base_pred=="100K-1M")].copy()
        if cand.empty: continue
        te=cand[["disasterNumber"]].merge(train,on="disasterNumber",how="left")
        te=te[te.incidentType==HAZARD].copy()
        if te.empty: continue
        st=tr[(tr.incidentType==HAZARD)&(tr.target_clean>=100_000)&(tr.target_clean<50_000_000)]
        ytr=(st.target_clean>=1_000_000).astype(int)
        if len(st)<12 or ytr.nunique()<2: continue
        model=fit_semantic(tr,features,kind,(610000 if kind=="log" else 620000)+int(outer_fy)*100+int(fy))
        pp=model.predict_proba(normalize_model_frame(te[features]))[:,1]
        for dn,p in zip(te.disasterNumber.astype(int),pp): rows.append({"disasterNumber":int(dn),"semantic_prob":float(p)})
    return pd.DataFrame(rows)


def apply_second(first_oof,semantic_scores,th):
    d=first_oof.copy()
    smap=dict(zip(semantic_scores.disasterNumber.astype(int),semantic_scores.semantic_prob.astype(float))) if not semantic_scores.empty else {}
    d["semantic_prob"]=d.disasterNumber.astype(int).map(smap)
    d["second_pred"]=d["pred"]
    m=(d["pred"]=="100K-1M")&d.semantic_prob.notna()&(d.semantic_prob>=th)
    d.loc[m,"second_pred"]="1M-50M"
    return d


def score_second(d):
    rec=[]
    for b in LOW_BANDS:
        m=d.actual_band==b
        if m.any(): rec.append(float((d.loc[m,"second_pred"]==b).mean()))
    macro=float(np.mean(rec)); acc=float((d.second_pred==d.actual_band).mean())
    residual=d[(d["pred"]=="100K-1M")&d.semantic_prob.notna()].copy()
    neg=residual[residual.actual_band!="1M-50M"]
    pos=residual[residual.actual_band=="1M-50M"]
    neg_keep=float((neg.second_pred=="100K-1M").mean()) if len(neg) else np.nan
    pos_recall=float((pos.second_pred=="1M-50M").mean()) if len(pos) else np.nan
    promotions=int(((d["pred"]=="100K-1M")&(d.second_pred=="1M-50M")).sum())
    return {"macro":macro,"accuracy":acc,"negative_keep":neg_keep,"residual_mid_recall":pos_recall,"promotions":promotions,"residual_negative_n":len(neg),"residual_positive_n":len(pos)}


def choose_second(first_oof,scores,guard):
    if scores.empty:
        d=apply_second(first_oof,scores,1.0); return 1.0,score_second(d)
    probs=scores.semantic_prob.to_numpy(float)
    grid=np.unique(np.r_[0.10,np.arange(0.15,0.96,0.02),0.99,probs])
    best=None
    for th in grid:
        d=apply_second(first_oof,scores,float(th)); m=score_second(d)
        if np.isfinite(m["negative_keep"]) and m["negative_keep"]<guard: continue
        key=(m["macro"],m["accuracy"],float(th))
        if best is None or key>best[0]: best=(key,float(th),m)
    if best is None:
        d=apply_second(first_oof,scores,1.0); return 1.0,score_second(d)
    return best[1],best[2]


def main():
    master=normalize_master(pd.read_excel(MASTER))
    master["target_clean"]=pd.to_numeric(master.totalObligatedFunding,errors="coerce").fillna(0).clip(lower=0)
    master["actual_band"]=master.target_clean.map(funding_band)
    ma=fetch_all_mission_assignments(); sem,_=build_semantic_rollup(master,ma)
    df=master.merge(sem,on="disasterNumber",how="left")
    df=df[df.incidentType!="Biological"].copy().reset_index(drop=True)
    current=valid_cols(df,CURRENT_19)
    semcols=valid_cols(df,[c for c in df.columns if (c.startswith("sem_") or c.startswith("ma_")) and not any(b in c.lower() for b in ["oblig","fund","cost","amount","dollar"])])
    semfeat=list(dict.fromkeys(current+semcols))
    compact=valid_cols(df,COMPACT_SEMANTIC)
    low=df[df.target_clean<50_000_000].copy()

    variants={
        "accepted_first_key":None,
        "semantic_log_guard80":("log",0.80),
        "semantic_log_guard90":("log",0.90),
        "semantic_rf_guard80":("rf",0.80),
        "semantic_rf_guard90":("rf",0.90),
    }
    rows={k:[] for k in variants}; diagrows=[]

    for outer_fy in sorted(low.fyDeclared.astype(int).unique()):
        train=low[low.fyDeclared.astype(int)!=outer_fy].copy(); test=low[low.fyDeclared.astype(int)==outer_fy].copy()
        base_oof,_=base_oof_predictions(train,semfeat,int(outer_fy))

        first_scores=first_oof_scores(train,base_oof,current,"rf",int(outer_fy))
        first_th,first_diag=choose_first_threshold(base_oof,first_scores,guard=0.75)
        first_oof=apply_first_rescue(base_oof,first_scores,first_th)

        selected={}
        for name,spec in variants.items():
            if name=="accepted_first_key": continue
            kind,guard=spec
            ss=semantic_oof_scores(train,base_oof,compact,kind,int(outer_fy))
            th,dg=choose_second(first_oof,ss,guard)
            selected[name]=(kind,guard,th,dg)
            diagrows.append({"outer_fy":int(outer_fy),"variant":name,"first_threshold":float(first_th),"second_threshold":float(th),**dg})

        outer_predict=fit_outer_base(train,semfeat,int(outer_fy)); base_pred,base_dollars=outer_predict(test)
        st=train[(train.incidentType==HAZARD)&(train.target_clean>=100_000)&(train.target_clean<50_000_000)].copy(); ytr=(st.target_clean>=1_000_000).astype(int)
        first_model=fit_first_specialist(st,current,"rf",710000+int(outer_fy)) if len(st)>=12 and ytr.nunique()>=2 else None
        first_pred=base_pred.copy(); first_prob=np.full(len(test),np.nan)
        idx=np.flatnonzero((base_pred=="100K-1M")&(test.incidentType.to_numpy()==HAZARD))
        if len(idx) and first_model is not None:
            pp=first_model.predict_proba(normalize_model_frame(test.iloc[idx][current]))[:,1]; first_prob[idx]=pp; first_pred[idx[pp>=first_th]]="1M-50M"

        for name in variants:
            final=first_pred.copy(); second_prob=np.full(len(test),np.nan)
            if name!="accepted_first_key":
                kind,guard,th,dg=selected[name]
                smodel=fit_semantic(train,compact,kind,(810000 if kind=="log" else 820000)+int(outer_fy)) if len(st)>=12 and ytr.nunique()>=2 else None
                ridx=np.flatnonzero((first_pred=="100K-1M")&(test.incidentType.to_numpy()==HAZARD))
                if len(ridx) and smodel is not None:
                    sp=smodel.predict_proba(normalize_model_frame(test.iloc[ridx][compact]))[:,1]; second_prob[ridx]=sp; final[ridx[sp>=th]]="1M-50M"
            for (_,r),bp,fp1,fp2,p1,p2,dollars in zip(test.iterrows(),base_pred,first_pred,final,first_prob,second_prob,base_dollars):
                rows[name].append({"disasterNumber":int(r.disasterNumber),"fyDeclared":int(r.fyDeclared),"state":r.state,"incidentType":r.incidentType,"actual_band":r.actual_band,"base_pred":str(bp),"first_pred":str(fp1),"final_pred":str(fp2),"first_prob":None if not np.isfinite(p1) else float(p1),"second_prob":None if not np.isfinite(p2) else float(p2),"base_reg_dollars":float(dollars)})

    results={}
    for name in variants:
        p=pd.DataFrame(rows[name]); p.to_csv(OUT/f"{name}_predictions.csv",index=False)
        lm=low_metrics(p.rename(columns={"final_pred":"pred"}),"pred")
        storm=p[p.incidentType==HAZARD]; sb={}
        for b in ["100K-1M","1M-50M"]:
            m=storm.actual_band==b; n=int(m.sum()); c=int((storm.loc[m,"final_pred"]==b).sum()); sb[b]={"correct":c,"total":n,"recall":c/n if n else None}
        results[name]={"low_metrics":lm,"storm_boundary":sb,"second_promotions":int(((p.first_pred=="100K-1M")&(p.final_pred=="1M-50M")).sum())}

    pd.DataFrame(diagrows).to_csv(OUT/"thresholds.csv",index=False)
    (OUT/"summary.json").write_text(json.dumps({"compact_semantic_features":compact,"results":results},indent=2),encoding="utf-8")
    md=["# Severe Storm second semantic-key audit","","| Variant | Low acc | Macro | 100K-1M | 1M-50M | Storm low | Storm mid | Second promotions |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name,r in results.items():
        lm=r["low_metrics"]; pb=lm["per_band"]; sb=r["storm_boundary"]
        md.append(f"| {name} | {lm['accuracy']:.1%} | {lm['macro_recall']:.1%} | {pb['100K-1M']['correct']}/{pb['100K-1M']['total']} ({pb['100K-1M']['recall']:.1%}) | {pb['1M-50M']['correct']}/{pb['1M-50M']['total']} ({pb['1M-50M']['recall']:.1%}) | {sb['100K-1M']['correct']}/{sb['100K-1M']['total']} ({sb['100K-1M']['recall']:.1%}) | {sb['1M-50M']['correct']}/{sb['1M-50M']['total']} ({sb['1M-50M']['recall']:.1%}) | {r['second_promotions']} |")
    (OUT/"summary.md").write_text("\n".join(md),encoding="utf-8"); print("\n".join(md))

if __name__=="__main__": main()
