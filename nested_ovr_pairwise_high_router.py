from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, recall_score, balanced_accuracy_score, f1_score, roc_curve

import external_physical_severity_high_router_experiment as base

OUT = Path("nested_ovr_pairwise_high_router_results")
OUT.mkdir(exist_ok=True)

BANDS=[3,4,5]
SAFE_CATS=[c for c in base.BASE_CAT if c!="eventScale"]

CFG=dict(
    iterations=150,
    depth=3,
    learning_rate=.035,
    l2_leaf_reg=20,
    loss_function="Logloss",
    verbose=False,
    random_seed=42,
    allow_writing_files=False,
    auto_class_weights="Balanced",
)

def prep(df,nums,cats):
    X=df[nums+cats].copy()
    for c in nums:
        X[c]=pd.to_numeric(X[c],errors="coerce")
    for c in cats:
        X[c]=X[c].astype(str).fillna("MISSING")
    return X

def fit_prob(train,test,y,nums,cats):
    y=np.asarray(y,int)
    if len(np.unique(y))<2:
        return np.full(len(test),float(y[0]) if len(y) else .5)
    m=CatBoostClassifier(**CFG)
    m.fit(prep(train,nums,cats),y,cat_features=cats)
    return m.predict_proba(prep(test,nums,cats))[:,list(m.classes_).index(1)]

def threshold(y,p):
    y=np.asarray(y,int); p=np.asarray(p,float)
    ok=np.isfinite(p); y=y[ok]; p=p[ok]
    if len(y)<4 or len(np.unique(y))<2:
        return .5
    fpr,tpr,thr=roc_curve(y,p); tnr=1-fpr
    score=np.minimum(tpr,tnr)
    best=np.nanmax(score)
    cand=np.where(np.isclose(score,best))[0]
    j=int(cand[np.argmax((tpr[cand]+tnr[cand])/2)])
    return float(thr[j])

def inner_binary_oof(train,mask_fn,label_fn,nums,cats):
    sub=train[mask_fn(train)].copy()
    out=pd.Series(np.nan,index=sub.index,dtype=float)
    for yr in sorted(sub.fyDeclared.astype(int).unique()):
        tr=sub[sub.fyDeclared.astype(int)!=yr]
        va=sub[sub.fyDeclared.astype(int)==yr]
        if va.empty: continue
        y=label_fn(tr).astype(int).to_numpy()
        out.loc[va.index]=fit_prob(tr,va,y,nums,cats)
    y=label_fn(sub).astype(int).to_numpy()
    return out,threshold(y,out.to_numpy(float))

def ovr_predict(train,test,nums,cats):
    probs=[]; thrs=[]
    for b in BANDS:
        mask=lambda d: d.band.isin(BANDS)
        lab=lambda d,b=b: (d.band==b)
        _,t=inner_binary_oof(train,mask,lab,nums,cats)
        y=(train.band==b).astype(int).to_numpy()
        p=fit_prob(train,test,y,nums,cats)
        probs.append(p); thrs.append(t)
    P=np.column_stack(probs)
    T=np.asarray(thrs,float)
    # Normalize by each class's own nested threshold so rare classes can compete fairly.
    score=P/np.clip(T,1e-6,None)
    pred=np.asarray([BANDS[i] for i in np.argmax(score,axis=1)],int)
    return pred,P,T,score

def pair_predict(train,test,nums,cats):
    pairs=[(3,4),(3,5),(4,5)]
    votes=np.zeros((len(test),3),int)
    margin=np.zeros((len(test),3),float)
    meta=[]
    for lo,hi in pairs:
        sub=train[train.band.isin([lo,hi])].copy()
        mask=lambda d,lo=lo,hi=hi: d.band.isin([lo,hi])
        lab=lambda d,hi=hi: (d.band==hi)
        _,t=inner_binary_oof(train,mask,lab,nums,cats)
        p=fit_prob(sub,test,(sub.band==hi).astype(int).to_numpy(),nums,cats)
        choose_hi=p>=t
        for i,is_hi in enumerate(choose_hi):
            winner=hi if is_hi else lo
            votes[i,BANDS.index(winner)]+=1
            # threshold-centered signed support; winner gets positive evidence
            scale=max(t,1-t,1e-6)
            m=(p[i]-t)/scale
            margin[i,BANDS.index(hi)]+=m
            margin[i,BANDS.index(lo)]-=m
        meta.append({"pair":f"{lo}v{hi}","threshold":float(t)})
    pred=[]
    for i in range(len(test)):
        vmax=votes[i].max()
        winners=np.where(votes[i]==vmax)[0]
        if len(winners)==1:
            pred.append(BANDS[int(winners[0])])
        else:
            j=winners[np.argmax(margin[i,winners])]
            pred.append(BANDS[int(j)])
    return np.asarray(pred,int),votes,margin,meta

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

def main():
    d=base.load_master()
    h=d[d.band.isin(BANDS)].copy().reset_index(drop=True)
    nums=[c for c in base.BASE_NUM if c in h.columns]
    cats=[c for c in SAFE_CATS if c in h.columns]

    po=np.full(len(h),-99,int)
    pp=np.full(len(h),-99,int)
    rows=[]; folds=[]

    for yr in sorted(h.fyDeclared.astype(int).unique()):
        tr_mask=h.fyDeclared.astype(int)!=yr
        te_mask=~tr_mask
        tr=h.loc[tr_mask].copy()
        te=h.loc[te_mask].copy()

        op,OP,OT,OS=ovr_predict(tr,te,nums,cats)
        pair,votes,margin,pmeta=pair_predict(tr,te,nums,cats)

        idx=np.flatnonzero(te_mask.to_numpy())
        po[idx]=op; pp[idx]=pair

        for j,ii in enumerate(idx):
            r=h.iloc[ii]
            rows.append({
                "outer_year":int(yr),
                "disasterNumber":int(r.disasterNumber),
                "state":str(r.state),
                "incidentType":str(r.incidentType),
                "target":float(r.target),
                "band":int(r.band),
                "ovr_prediction":int(op[j]),
                "pairwise_prediction":int(pair[j]),
                "ovr_p3":float(OP[j,0]),"ovr_p4":float(OP[j,1]),"ovr_p5":float(OP[j,2]),
                "ovr_t3":float(OT[0]),"ovr_t4":float(OT[1]),"ovr_t5":float(OT[2]),
                "ovr_s3":float(OS[j,0]),"ovr_s4":float(OS[j,1]),"ovr_s5":float(OS[j,2]),
                "pair_votes3":int(votes[j,0]),"pair_votes4":int(votes[j,1]),"pair_votes5":int(votes[j,2]),
                "pair_margin3":float(margin[j,0]),"pair_margin4":float(margin[j,1]),"pair_margin5":float(margin[j,2]),
            })
        folds.append({"outer_year":int(yr),"pair_thresholds":pmeta,
                      "ovr_thresholds":{"3":float(OT[0]),"4":float(OT[1]),"5":float(OT[2])}})

    y=h.band.astype(int).to_numpy()
    out=pd.DataFrame(rows).sort_values(["outer_year","disasterNumber"])
    out.to_csv(OUT/"nested_oof_predictions.csv",index=False)

    actual5=out[out.band==5].copy()
    summary={
        "purpose":"Fully nested One-vs-Rest bank and Pairwise Tournament routers on all 55 high-value cases including Biological.",
        "counts":{str(b):int((h.band==b).sum()) for b in BANDS},
        "metrics":{
            "one_vs_rest":metric(y,po),
            "pairwise_tournament":metric(y,pp),
        },
        "actual_500m_plus":actual5[
            ["disasterNumber","state","incidentType","target","ovr_prediction","pairwise_prediction",
             "ovr_p3","ovr_p4","ovr_p5","pair_votes3","pair_votes4","pair_votes5"]
        ].to_dict("records"),
        "ovr_correct_pair_wrong":out[(out.ovr_prediction==out.band)&(out.pairwise_prediction!=out.band)][
            ["disasterNumber","state","incidentType","target","band","ovr_prediction","pairwise_prediction"]
        ].to_dict("records"),
        "pair_correct_ovr_wrong":out[(out.pairwise_prediction==out.band)&(out.ovr_prediction!=out.band)][
            ["disasterNumber","state","incidentType","target","band","ovr_prediction","pairwise_prediction"]
        ].to_dict("records"),
        "outer_fold_thresholds":folds,
        "guardrails":[
            "All 55 high-value cases including Biological are evaluated.",
            "Every outer fiscal year is excluded from model fitting and threshold selection.",
            "Every OVR and pairwise threshold is selected only from inner leave-fiscal-year-out predictions inside the current outer training set.",
            "Funding-derived eventScale is excluded.",
            "No actual held-out funding value or band is used to route a test case.",
            "FY2020 remains a true Biological cold-start because all high-value Biological cases occur in the held-out year."
        ]
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str),flush=True)

if __name__=="__main__":
    main()
