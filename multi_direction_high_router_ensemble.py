from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, recall_score, balanced_accuracy_score, f1_score, roc_curve

import external_physical_severity_high_router_experiment as base

OUT = Path("multi_direction_high_router_ensemble_results")
OUT.mkdir(exist_ok=True)

BANDS = [3,4,5]
SAFE_CATS = [c for c in base.BASE_CAT if c != "eventScale"]

BIN_CFG = dict(
    iterations=180, depth=3, learning_rate=.035, l2_leaf_reg=20,
    loss_function="Logloss", verbose=False, random_seed=42,
    allow_writing_files=False, auto_class_weights="Balanced"
)
MULTI_CFG = dict(
    iterations=220, depth=3, learning_rate=.03, l2_leaf_reg=20,
    loss_function="MultiClass", verbose=False, random_seed=42,
    allow_writing_files=False, auto_class_weights="Balanced"
)


def prep(df, nums, cats):
    X = df[nums+cats].copy()
    for c in nums:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in cats:
        X[c] = X[c].astype(str).fillna("MISSING")
    return X


def fit_binary(train, test, positive_band_set, nums, cats):
    y = train["band"].isin(positive_band_set).astype(int)
    if y.nunique() < 2:
        return np.full(len(test), float(y.iloc[0]) if len(y) else .5)
    m = CatBoostClassifier(**BIN_CFG)
    m.fit(prep(train,nums,cats), y, cat_features=cats)
    return m.predict_proba(prep(test,nums,cats))[:, list(m.classes_).index(1)]


def best_threshold(y, p):
    y=np.asarray(y,int); p=np.asarray(p,float)
    ok=np.isfinite(p); y=y[ok]; p=p[ok]
    if len(y)<4 or len(np.unique(y))<2:
        return .5, {"lower_recall":0.0,"upper_recall":0.0,"min_recall":0.0}
    fpr,tpr,thr=roc_curve(y,p); tnr=1-fpr
    mins=np.minimum(tpr,tnr)
    best=np.nanmax(mins)
    cand=np.where(np.isclose(mins,best))[0]
    j=int(cand[np.argmax((tpr[cand]+tnr[cand])/2)])
    return float(thr[j]), {
        "lower_recall":float(tnr[j]),
        "upper_recall":float(tpr[j]),
        "min_recall":float(min(tnr[j],tpr[j]))
    }


def inner_gate_threshold(train, gate_train_mask, positive_bands, nums, cats):
    sub=train[gate_train_mask(train)].copy()
    probs=pd.Series(np.nan,index=sub.index,dtype=float)
    for yr in sorted(sub["fyDeclared"].astype(int).unique()):
        tr=sub[sub["fyDeclared"].astype(int)!=yr]
        va=sub[sub["fyDeclared"].astype(int)==yr]
        if va.empty: continue
        probs.loc[va.index]=fit_binary(tr,va,positive_bands,nums,cats)
    y=sub["band"].isin(positive_bands).astype(int).to_numpy()
    th,met=best_threshold(y,probs.to_numpy(float))
    return th,met


def bottom_up_predict(train,test,nums,cats):
    # Gate 1: band3 vs {4,5}. If not band3, Gate 2: band4 vs band5.
    g1_mask=lambda d: d["band"].isin([3,4,5])
    t1,m1=inner_gate_threshold(train,g1_mask,{3},nums,cats)
    p1=fit_binary(train[train.band.isin([3,4,5])],test,{3},nums,cats)

    pred=np.full(len(test),-99,int)
    go_high=p1 < t1
    pred[~go_high]=3

    if go_high.any():
        g2_mask=lambda d: d["band"].isin([4,5])
        t2,m2=inner_gate_threshold(train,g2_mask,{4},nums,cats)
        tr2=train[train.band.isin([4,5])]
        p2=fit_binary(tr2,test.loc[go_high],{4},nums,cats)
        pred[np.flatnonzero(go_high)] = np.where(p2>=t2,4,5)
    else:
        t2=.5; m2={"lower_recall":np.nan,"upper_recall":np.nan,"min_recall":np.nan}

    return pred, {"gate3_threshold":t1,"gate3_inner":m1,"gate4_threshold":t2,"gate4_inner":m2}


def top_down_predict(train,test,nums,cats):
    # Gate 1: band5 vs {3,4}. If not band5, Gate 2: band4 vs band3.
    g1_mask=lambda d: d["band"].isin([3,4,5])
    t1,m1=inner_gate_threshold(train,g1_mask,{5},nums,cats)
    p1=fit_binary(train,test,{5},nums,cats)

    pred=np.full(len(test),-99,int)
    is_extreme=p1>=t1
    pred[is_extreme]=5

    if (~is_extreme).any():
        g2_mask=lambda d: d["band"].isin([3,4])
        t2,m2=inner_gate_threshold(train,g2_mask,{4},nums,cats)
        tr2=train[train.band.isin([3,4])]
        p2=fit_binary(tr2,test.loc[~is_extreme],{4},nums,cats)
        pred[np.flatnonzero(~is_extreme)] = np.where(p2>=t2,4,3)
    else:
        t2=.5; m2={"lower_recall":np.nan,"upper_recall":np.nan,"min_recall":np.nan}

    return pred, {"gate5_threshold":t1,"gate5_inner":m1,"gate4_threshold":t2,"gate4_inner":m2}


def sifter_predict(train,test,nums,cats):
    # Structurally different view: one broad split {3} vs {4,5}, then
    # a direct multiclass model competes across all three bands.
    # Final choice uses the broad split only as a coarse regime filter.
    broad_mask=lambda d: d["band"].isin([3,4,5])
    tb,mb=inner_gate_threshold(train,broad_mask,{3},nums,cats)
    pb=fit_binary(train,test,{3},nums,cats)

    m=CatBoostClassifier(**MULTI_CFG)
    m.fit(prep(train,nums,cats),train["band"].astype(int),cat_features=cats)
    proba=m.predict_proba(prep(test,nums,cats))
    classes=[int(x) for x in m.classes_]
    P=np.zeros((len(test),3),float)
    for j,b in enumerate(BANDS):
        if b in classes:
            P[:,j]=proba[:,classes.index(b)]

    pred=np.zeros(len(test),int)
    for i in range(len(test)):
        if pb[i]>=tb:
            pred[i]=3
        else:
            # sifter only decides between upper two classes after broad rejection
            pred[i]=4 if P[i,1]>=P[i,2] else 5
    return pred, {"broad_band3_threshold":tb,"broad_inner":mb}


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


def majority_vote(a,b,c):
    out=[]
    for x,y,z in zip(a,b,c):
        vals=[int(x),int(y),int(z)]
        counts={v:vals.count(v) for v in set(vals)}
        m=max(counts.values())
        winners=[v for v,k in counts.items() if k==m]
        out.append(winners[0] if len(winners)==1 else int(z))  # sifter breaks 1-1-1 ties
    return np.asarray(out,int)


def extreme_preserving_vote(a,b,c):
    out=[]
    for x,y,z in zip(a,b,c):
        vals=[int(x),int(y),int(z)]
        # Pre-specified extreme-preservation rule:
        # top-down may preserve 500M+ only if another independent router
        # at least agrees that the case is not in the lowest high band.
        if int(y)==5 and (int(x)>=4 or int(z)>=4):
            out.append(5)
            continue
        counts={v:vals.count(v) for v in set(vals)}
        m=max(counts.values())
        winners=[v for v,k in counts.items() if k==m]
        out.append(winners[0] if len(winners)==1 else int(z))
    return np.asarray(out,int)


def main():
    d=base.load_master()
    h=d[
        d["band"].isin(BANDS)
        & ~d["incidentType"].astype(str).str.upper().str.strip().eq("BIOLOGICAL")
    ].copy().reset_index(drop=True)

    nums=[c for c in base.BASE_NUM if c in h.columns]
    cats=[c for c in SAFE_CATS if c in h.columns]

    pa=np.full(len(h),-99,int)
    pb=np.full(len(h),-99,int)
    pc=np.full(len(h),-99,int)
    fold_rows=[]

    for yr in sorted(h["fyDeclared"].astype(int).unique()):
        tr=h[h["fyDeclared"].astype(int)!=yr].copy()
        te_mask=h["fyDeclared"].astype(int)==yr
        te=h[te_mask].copy()

        a,ma=bottom_up_predict(tr,te,nums,cats)
        b,mb=top_down_predict(tr,te,nums,cats)
        c,mc=sifter_predict(tr,te,nums,cats)

        idx=np.flatnonzero(te_mask.to_numpy())
        pa[idx]=a; pb[idx]=b; pc[idx]=c
        fold_rows.append({
            "outer_year":int(yr),
            "bottom_up":ma,
            "top_down":mb,
            "sifter":mc,
        })

    maj=majority_vote(pa,pb,pc)
    ext=extreme_preserving_vote(pa,pb,pc)

    y=h["band"].astype(int).to_numpy()
    metrics={
        "bottom_up":metric(y,pa),
        "top_down":metric(y,pb),
        "sifter":metric(y,pc),
        "majority_vote":metric(y,maj),
        "extreme_preserving_vote":metric(y,ext),
    }

    out=h[["disasterNumber","state","fyDeclared","incidentType","target","band"]].copy()
    out["bottom_up"]=pa
    out["top_down"]=pb
    out["sifter"]=pc
    out["majority_vote"]=maj
    out["extreme_preserving_vote"]=ext
    out["all_three_agree"]=(pa==pb)&(pb==pc)
    out["three_way_disagreement"]=(pa!=pb)&(pa!=pc)&(pb!=pc)
    out.to_csv(OUT/"oof_multi_router_predictions.csv",index=False)
    pd.DataFrame([{"outer_year":r["outer_year"],
                   "bottom_up":json.dumps(r["bottom_up"]),
                   "top_down":json.dumps(r["top_down"]),
                   "sifter":json.dumps(r["sifter"])} for r in fold_rows]
                ).to_csv(OUT/"outer_fold_thresholds.csv",index=False)

    actual5=out[out.band==5].copy()
    false5=out[(out.band!=5)&(
        (out.bottom_up==5)|(out.top_down==5)|(out.sifter==5)|
        (out.majority_vote==5)|(out.extreme_preserving_vote==5)
    )].copy()

    complement={
        "all_three_agreement_rate":float(out.all_three_agree.mean()),
        "three_way_disagreement_count":int(out.three_way_disagreement.sum()),
        "cases_where_top_down_correct_and_bottom_up_wrong":out[
            (out.top_down==out.band)&(out.bottom_up!=out.band)
        ][["disasterNumber","state","incidentType","target","band","bottom_up","top_down","sifter"]].to_dict("records"),
        "cases_where_bottom_up_correct_and_top_down_wrong":out[
            (out.bottom_up==out.band)&(out.top_down!=out.band)
        ][["disasterNumber","state","incidentType","target","band","bottom_up","top_down","sifter"]].to_dict("records"),
    }

    summary={
        "purpose":"Test complementary routing directions before adopting a voting ensemble for the unresolved non-Biological high-value router.",
        "counts":{str(b):int((h.band==b).sum()) for b in BANDS},
        "metrics":metrics,
        "actual_500m_plus_case_votes":actual5[
            ["disasterNumber","state","incidentType","target","bottom_up","top_down","sifter","majority_vote","extreme_preserving_vote"]
        ].to_dict("records"),
        "false_500m_plus_votes":false5[
            ["disasterNumber","state","incidentType","target","band","bottom_up","top_down","sifter","majority_vote","extreme_preserving_vote"]
        ].to_dict("records"),
        "complementarity":complement,
        "guardrails":[
            "Biological disasters are excluded from both training and evaluation for this diagnostic.",
            "Every prediction is strict leave-fiscal-year-out.",
            "Every binary gate threshold is selected from inner leave-year-out predictions within the outer training years.",
            "Funding-derived eventScale is excluded.",
            "Bottom-up and top-down use different conditional training supports and opposite routing order.",
            "The sifter uses a coarse band3-vs-upper split followed by a direct upper-two comparison.",
            "Voting rules are pre-specified and are not tuned on outer OOF outcomes.",
            "This is a non-Biological high-router diagnostic; the final six-band 80%-per-band requirement remains unchanged."
        ]
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str),flush=True)


if __name__=="__main__":
    main()
