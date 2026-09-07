#!/usr/bin/env python3
"""Fast strict-LFYO conditional low audit for hidden $1M-$50M sub-bands."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from mission_semantic_audit import (
    CURRENT_19, build_semantic_rollup, fetch_all_mission_assignments,
    normalize_master, normalize_model_frame, prep_pipeline,
)
from nonbio_all_ranges import funding_band, valid_cols
from nonbio_low_thresholds import inner_binary_oof, choose_threshold

ROOT=Path(__file__).resolve().parents[1]
MASTER=ROOT/"master_openfema_40plus.xlsx"
OUT=ROOT/"audit_outputs"/"nonbio_mid_subbands_lowonly"
OUT.mkdir(parents=True,exist_ok=True)

SCHEMES={
    "current_best":None,
    "split_3_10":[1_000_000,3_000_000,10_000_000,50_000_000],
    "split_5_15":[1_000_000,5_000_000,15_000_000,50_000_000],
    "split_5_10_25":[1_000_000,5_000_000,10_000_000,25_000_000,50_000_000],
}
LOW=["0-100K","100K-1M","1M-50M"]

def fit_rf(df,features,y,seed):
    X=normalize_model_frame(df[features])
    m=RandomForestClassifier(
        n_estimators=700,random_state=seed,class_weight="balanced_subsample",
        max_features="sqrt",min_samples_leaf=1,n_jobs=-1,
    )
    p=prep_pipeline(X,m); p.fit(X,y); return p

def posprob(m,df,features):
    return m.predict_proba(normalize_model_frame(df[features]))[:,1]

def ilabel(v,edges):
    if v<1_000_000: return "LOW_100K_1M"
    for i in range(len(edges)-1):
        if edges[i]<=v<edges[i+1]: return f"MID_{i}"
    raise ValueError(v)

def metrics(rows):
    d=pd.DataFrame(rows)
    pb={}
    for b in LOW:
        x=d[d.actual_band==b]
        pb[b]={
            "correct":int((x.pred==b).sum()),"total":int(len(x)),
            "recall":float((x.pred==b).mean())
        }
    mid=d[d.actual_band=="1M-50M"]
    return {
        "accuracy":float((d.pred==d.actual_band).mean()),
        "macro_recall":float(np.mean([pb[b]["recall"] for b in LOW])),
        "per_band":pb,
        "mid_down":int((mid.pred=="100K-1M").sum()),
        "mid_zero":int((mid.pred=="0-100K").sum()),
    }

def main():
    master=normalize_master(pd.read_excel(MASTER))
    master["target_clean"]=pd.to_numeric(master.totalObligatedFunding,errors="coerce").fillna(0).clip(lower=0)
    master["actual_band"]=master.target_clean.map(funding_band)
    ma=fetch_all_mission_assignments()
    sem,_=build_semantic_rollup(master,ma)
    df=master.merge(sem,on="disasterNumber",how="left")
    df=df[(df.incidentType!="Biological")&(df.target_clean<50_000_000)].copy()
    current=valid_cols(df,CURRENT_19)
    semcols=valid_cols(df,[c for c in df.columns if c.startswith("sem_") or c.startswith("ma_")])
    semfeat=list(dict.fromkeys(current+semcols))

    rows={k:[] for k in SCHEMES}
    counts={}
    mids=df[(df.target_clean>=1_000_000)&(df.target_clean<50_000_000)]
    for name,edges in SCHEMES.items():
        if edges:
            counts[name]=pd.Series([ilabel(v,edges) for v in mids.target_clean]).value_counts().to_dict()

    for fy in sorted(df.fyDeclared.astype(int).unique()):
        tr=df[df.fyDeclared.astype(int)!=fy].copy()
        te=df[df.fyDeclared.astype(int)==fy].copy()

        # Frozen low stage1.
        oof1=inner_binary_oof(tr,semfeat,stage=1,kind="rf",seedbase=110000)
        th1,_=choose_threshold(oof1,"macro_f1")
        y1=(tr.target_clean>=100_000).astype(int)
        s1=fit_rf(tr,semfeat,y1,130000+fy)

        for name,edges in SCHEMES.items():
            upper=tr[tr.target_clean>=100_000].copy()
            if name=="current_best":
                # Binary 100K-1M vs 1M-50M with inner macro-F1 threshold.
                oof2=inner_binary_oof(tr,current,stage=2,kind="rf",seedbase=120000)
                th2,_=choose_threshold(oof2,"macro_f1")
                y2=(upper.target_clean>=1_000_000).astype(int)
                s2=fit_rf(upper,current,y2,140000+fy)
            else:
                th2=None
                y2=upper.target_clean.map(lambda v: ilabel(v,edges))
                s2=fit_rf(upper,current,y2,150000+fy)

            p1=posprob(s1,te,semfeat)
            pred=np.full(len(te),"0-100K",dtype=object)
            ii=np.flatnonzero(p1>=th1)
            if len(ii):
                if name=="current_best":
                    p2=posprob(s2,te.iloc[ii],current)
                    pred[ii]=np.where(p2>=th2,"1M-50M","100K-1M")
                else:
                    raw=s2.predict(normalize_model_frame(te.iloc[ii][current])).astype(str)
                    pred[ii]=np.where(np.char.startswith(raw,"MID_"),"1M-50M","100K-1M")
            for (_,r),p in zip(te.iterrows(),pred):
                rows[name].append({
                    "disasterNumber":int(r.disasterNumber),
                    "funding":float(r.target_clean),
                    "actual_band":r.actual_band,
                    "pred":str(p),
                })

    result={k:metrics(v) for k,v in rows.items()}
    for k,v in rows.items(): pd.DataFrame(v).to_csv(OUT/f"{k}.csv",index=False)
    summary={"counts":counts,"results":result}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2))
    md=["# Fast conditional low sub-band audit","","| Variant | Acc | Macro | 0-100K | 100K-1M | 1M-50M | mid down |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for k,r in result.items():
        p=r["per_band"]
        md.append(
            f"| {k} | {r['accuracy']:.1%} | {r['macro_recall']:.1%} | "
            f"{p['0-100K']['correct']}/{p['0-100K']['total']} ({p['0-100K']['recall']:.1%}) | "
            f"{p['100K-1M']['correct']}/{p['100K-1M']['total']} ({p['100K-1M']['recall']:.1%}) | "
            f"{p['1M-50M']['correct']}/{p['1M-50M']['total']} ({p['1M-50M']['recall']:.1%}) | {r['mid_down']} |"
        )
    (OUT/"summary.md").write_text("\n".join(md))
    print("\n".join(md))

if __name__=="__main__": main()
