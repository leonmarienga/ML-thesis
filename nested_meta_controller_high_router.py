from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, recall_score, balanced_accuracy_score, f1_score

import external_physical_severity_high_router_experiment as base
import multi_direction_high_router_ensemble as md

OUT = Path("nested_meta_controller_high_router_results")
OUT.mkdir(exist_ok=True)

BANDS=[3,4,5]

# Keep the same expert architecture but trim iteration counts for the deeper
# nested stacking audit. No hyperparameters are selected from outer OOF.
md.BIN_CFG["iterations"]=110
md.MULTI_CFG["iterations"]=140

META_CFG=dict(
    iterations=140,
    depth=2,
    learning_rate=.03,
    l2_leaf_reg=25,
    loss_function="MultiClass",
    verbose=False,
    random_seed=42,
    allow_writing_files=False,
    auto_class_weights="Balanced",
)

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

def expert_predictions(train,test,nums,cats):
    a,_=md.bottom_up_predict(train,test,nums,cats)
    b,_=md.top_down_predict(train,test,nums,cats)
    c,_=md.sifter_predict(train,test,nums,cats)
    return np.asarray(a,int),np.asarray(b,int),np.asarray(c,int)

def meta_frame(source,a,b,c):
    x=pd.DataFrame(index=source.index)
    x["bottom_up"]=np.asarray(a,int).astype(str)
    x["top_down"]=np.asarray(b,int).astype(str)
    x["sifter"]=np.asarray(c,int).astype(str)
    x["incidentType"]=source["incidentType"].astype(str).to_numpy()
    # Structural disagreement descriptors only; no funding-derived feature.
    arr=np.column_stack([a,b,c]).astype(int)
    x["vote3"]=(arr==3).sum(axis=1)
    x["vote4"]=(arr==4).sum(axis=1)
    x["vote5"]=(arr==5).sum(axis=1)
    x["min_vote"]=arr.min(axis=1)
    x["max_vote"]=arr.max(axis=1)
    x["spread"]=arr.max(axis=1)-arr.min(axis=1)
    x["all_agree"]=((arr[:,0]==arr[:,1])&(arr[:,1]==arr[:,2])).astype(int)
    x["top_is_extreme"]=(arr[:,1]==5).astype(int)
    x["bottom_is_extreme"]=(arr[:,0]==5).astype(int)
    x["sifter_is_extreme"]=(arr[:,2]==5).astype(int)
    return x

def nested_meta_training(outer_train,nums,cats):
    rows=[]; ys=[]
    for meta_year in sorted(outer_train["fyDeclared"].astype(int).unique()):
        meta_tr=outer_train[outer_train["fyDeclared"].astype(int)!=meta_year].copy()
        meta_va=outer_train[outer_train["fyDeclared"].astype(int)==meta_year].copy()
        if meta_va.empty: continue
        a,b,c=expert_predictions(meta_tr,meta_va,nums,cats)
        xf=meta_frame(meta_va,a,b,c)
        xf["row_index"]=meta_va.index.to_numpy()
        rows.append(xf)
        ys.extend(meta_va["band"].astype(int).tolist())
    if not rows:
        raise RuntimeError("No inner OOF rows for meta-controller")
    X=pd.concat(rows,axis=0).sort_values("row_index")
    y=outer_train.loc[X["row_index"].astype(int),"band"].astype(int).to_numpy()
    X=X.drop(columns=["row_index"])
    return X,y

def majority(a,b,c):
    out=[]
    for vals in zip(a,b,c):
        vals=[int(v) for v in vals]
        counts={v:vals.count(v) for v in set(vals)}
        m=max(counts.values())
        winners=[v for v,k in counts.items() if k==m]
        out.append(winners[0] if len(winners)==1 else int(vals[2]))
    return np.asarray(out,int)

def main():
    d=base.load_master()
    h=d[d["band"].isin(BANDS)].copy().reset_index(drop=True)
    nums=[c for c in base.BASE_NUM if c in h.columns]
    cats=[c for c in md.SAFE_CATS if c in h.columns]

    meta_pred=np.full(len(h),-99,int)
    maj_pred=np.full(len(h),-99,int)
    bu_pred=np.full(len(h),-99,int)
    td_pred=np.full(len(h),-99,int)
    sf_pred=np.full(len(h),-99,int)
    rows=[]
    fold_summaries=[]

    meta_cats=["bottom_up","top_down","sifter","incidentType"]

    for outer_year in sorted(h["fyDeclared"].astype(int).unique()):
        tr_mask=h["fyDeclared"].astype(int)!=outer_year
        te_mask=~tr_mask
        outer_train=h.loc[tr_mask].copy()
        outer_test=h.loc[te_mask].copy()

        # Fully nested stacking data: every meta-training row is itself generated
        # by experts that exclude that row's fiscal year AND the outer held year.
        Xmeta,ymeta=nested_meta_training(outer_train,nums,cats)

        meta=CatBoostClassifier(**META_CFG)
        meta.fit(Xmeta,ymeta,cat_features=meta_cats)

        a,b,c=expert_predictions(outer_train,outer_test,nums,cats)
        Xtest=meta_frame(outer_test,a,b,c)
        mp=np.asarray(meta.predict(Xtest)).reshape(-1).astype(int)
        mj=majority(a,b,c)

        idx=np.flatnonzero(te_mask.to_numpy())
        bu_pred[idx]=a; td_pred[idx]=b; sf_pred[idx]=c
        maj_pred[idx]=mj; meta_pred[idx]=mp

        cls=[int(v) for v in meta.classes_]
        prob=meta.predict_proba(Xtest)
        for j,ii in enumerate(idx):
            r=h.iloc[ii]
            rec={
                "outer_year":int(outer_year),
                "disasterNumber":int(r.disasterNumber),
                "state":str(r.state),
                "incidentType":str(r.incidentType),
                "target":float(r.target),
                "band":int(r.band),
                "bottom_up":int(a[j]),
                "top_down":int(b[j]),
                "sifter":int(c[j]),
                "majority_vote":int(mj[j]),
                "meta_prediction":int(mp[j]),
            }
            for band in BANDS:
                rec[f"meta_prob_band{band}"]=float(prob[j,cls.index(band)]) if band in cls else 0.0
            rows.append(rec)

        fold_summaries.append({
            "outer_year":int(outer_year),
            "meta_training_rows":int(len(Xmeta)),
            "meta_training_band_counts":{str(b):int((ymeta==b).sum()) for b in BANDS},
            "outer_test_rows":int(len(outer_test)),
        })

    y=h["band"].astype(int).to_numpy()
    metrics={
        "bottom_up":metric(y,bu_pred),
        "top_down":metric(y,td_pred),
        "sifter":metric(y,sf_pred),
        "majority_vote":metric(y,maj_pred),
        "nested_meta_controller":metric(y,meta_pred),
    }

    out=pd.DataFrame(rows).sort_values(["outer_year","disasterNumber"])
    out.to_csv(OUT/"nested_meta_oof_predictions.csv",index=False)
    pd.DataFrame(fold_summaries).to_csv(OUT/"outer_fold_meta_training_summary.csv",index=False)

    actual5=out[out.band==5].copy()
    meta_false5=out[(out.band!=5)&(out.meta_prediction==5)].copy()
    rescued=out[(out.meta_prediction==out.band)&(out.majority_vote!=out.band)].copy()
    harmed=out[(out.meta_prediction!=out.band)&(out.majority_vote==out.band)].copy()

    summary={
        "purpose":"Fully nested learned consensus controller over bottom-up, top-down, and sifter high-value routers on all 55 high-value cases including Biological.",
        "counts":{str(b):int((h.band==b).sum()) for b in BANDS},
        "metrics":metrics,
        "actual_500m_plus":actual5[
            ["disasterNumber","state","incidentType","target","bottom_up","top_down","sifter",
             "majority_vote","meta_prediction","meta_prob_band3","meta_prob_band4","meta_prob_band5"]
        ].to_dict("records"),
        "meta_false_500m_plus":meta_false5[
            ["disasterNumber","state","incidentType","target","band","bottom_up","top_down","sifter",
             "majority_vote","meta_prediction","meta_prob_band5"]
        ].to_dict("records"),
        "meta_rescued_vs_majority":rescued[
            ["disasterNumber","state","incidentType","target","band","bottom_up","top_down","sifter",
             "majority_vote","meta_prediction"]
        ].to_dict("records"),
        "meta_harmed_vs_majority":harmed[
            ["disasterNumber","state","incidentType","target","band","bottom_up","top_down","sifter",
             "majority_vote","meta_prediction"]
        ].to_dict("records"),
        "outer_folds":fold_summaries,
        "guardrails":[
            "All 55 high-value cases including Biological are evaluated.",
            "The outer fiscal year is excluded from every expert and from every meta-controller training operation.",
            "Meta-controller training features are generated by an additional leave-fiscal-year-out loop inside the outer training set.",
            "Each base router still performs its own inner leave-year threshold selection on its current training subset.",
            "Funding-derived eventScale is excluded from the experts.",
            "The meta-controller sees only expert predictions, disagreement descriptors, and incident type; it receives no funding amount or funding-derived feature.",
            "The meta-controller architecture/hyperparameters are fixed before outer OOF evaluation and are not tuned on outer results.",
            "For outer FY2020, the controller has no high-value Biological examples in its training data, so Biological remains a genuine temporal cold-start test."
        ]
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str),flush=True)

if __name__=="__main__":
    main()
