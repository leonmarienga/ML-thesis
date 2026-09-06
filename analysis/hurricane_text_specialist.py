#!/usr/bin/env python3
"""
Hurricane disaster-level Mission Assignment text audit.

Tests whether non-financial initial Mission Assignment wording contains
out-of-year signal for:
A) $50M-$200M vs $200M-$500M hurricanes
B) <$500M vs $500M+ hurricanes

Leakage controls:
- one INITIAL action/amendment per maId
- obligation/cost-share/date-obligated fields excluded
- all digits, currency terms, URLs and emails removed from text
- outer fiscal year excluded from TF-IDF vocabulary and classifier fit
"""

from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from mission_semantic_audit import (
    fetch_all_mission_assignments, normalize_master
)

ROOT=Path(__file__).resolve().parents[1]
MASTER=ROOT/"master_openfema_40plus.xlsx"
OUT=ROOT/"audit_outputs"/"hurricane_text_specialist"
OUT.mkdir(parents=True,exist_ok=True)

def redact(x):
    s="" if pd.isna(x) else str(x).lower()
    s=re.sub(r"https?://\\S+|www\\.\\S+|\\S+@\\S+"," ",s)
    s=re.sub(r"\\$|usd|dollars?|million|billion|thousand|obligation|funding|cost estimate"," ",s)
    s=re.sub(r"\\d+(?:[.,]\\d+)*"," ",s)
    s=re.sub(r"[^a-z\\s/-]"," ",s)
    s=re.sub(r"\\s+"," ",s).strip()
    return s

def build_docs(master, ma):
    x=ma.copy()
    x["disasterNumber"]=pd.to_numeric(x["disasterNumber"],errors="coerce").astype("Int64")
    ids=set(master["disasterNumber"].dropna().astype(int))
    x=x[x["disasterNumber"].isin(ids)].copy()
    x["amend"]=pd.to_numeric(x["maAmendNumber"],errors="coerce").fillna(-1)
    x["action"]=pd.to_numeric(x["actionId"],errors="coerce").fillna(-1)
    init=(x.sort_values(["disasterNumber","maId","amend","action"])
            .groupby(["disasterNumber","maId"],dropna=False).head(1).copy())
    init["txt"]=(init["assistanceRequested"].fillna("").astype(str)+" "+
                 init["statementOfWork"].fillna("").astype(str)).map(redact)
    docs=init.groupby("disasterNumber")["txt"].apply(lambda s:" ".join(v for v in s if v)).rename("ma_text").reset_index()
    docs["doc_chars"]=docs["ma_text"].str.len()
    return docs

def lfyo(df,target, mode):
    y=np.asarray(target,dtype=int)
    pred=np.zeros(len(df),dtype=int)
    prob=np.zeros(len(df),dtype=float)
    folds=[]
    for fy in sorted(df.fyDeclared.astype(int).unique()):
        te=df.fyDeclared.astype(int).to_numpy()==fy
        tr=~te
        if len(np.unique(y[tr]))<2:
            folds.append({"fy":int(fy),"learnable":False})
            continue
        vec=TfidfVectorizer(
            ngram_range=(1,2), min_df=2, max_df=.995,
            sublinear_tf=True, max_features=18000,
            strip_accents="unicode"
        )
        Xt=vec.fit_transform(df.loc[tr,"ma_text"].fillna(""))
        Xv=vec.transform(df.loc[te,"ma_text"].fillna(""))
        if mode=="text_plus_counts":
            cols=["missionAssignmentCount","uniqueAgencyCount","uniqueMaTypeCount","uniquePriorityCount","responseComplexityScore"]
            A=df.loc[tr,cols].apply(pd.to_numeric,errors="coerce").fillna(0).to_numpy(float)
            B=df.loc[te,cols].apply(pd.to_numeric,errors="coerce").fillna(0).to_numpy(float)
            sc=StandardScaler()
            A=sc.fit_transform(A); B=sc.transform(B)
            Xt=sparse.hstack([Xt,sparse.csr_matrix(A)],format="csr")
            Xv=sparse.hstack([Xv,sparse.csr_matrix(B)],format="csr")
        model=LogisticRegression(max_iter=7000,class_weight="balanced",C=.25,solver="liblinear")
        model.fit(Xt,y[tr])
        pred[te]=model.predict(Xv)
        prob[te]=model.predict_proba(Xv)[:,1]
        folds.append({"fy":int(fy),"learnable":True,"vocab":int(len(vec.vocabulary_))})
    return {
        "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
        "negative_recall":float(recall_score(y,pred,pos_label=0,zero_division=0)),
        "positive_recall":float(recall_score(y,pred,pos_label=1,zero_division=0)),
        "confusion_matrix":confusion_matrix(y,pred,labels=[0,1]).tolist(),
        "predictions":pred.tolist(),"probabilities":prob.tolist(),"folds":folds
    }

def main():
    master=normalize_master(pd.read_excel(MASTER))
    ma=fetch_all_mission_assignments()
    docs=build_docs(master,ma)
    h=master[(master.incidentType=="Hurricane")&(master.totalObligatedFunding>=50_000_000)].copy()
    h=h.merge(docs,on="disasterNumber",how="left").reset_index(drop=True)
    h["ma_text"]=h["ma_text"].fillna("")

    results={}
    # lower-middle
    lm=h[h.totalObligatedFunding<500_000_000].copy().reset_index(drop=True)
    ylm=(lm.totalObligatedFunding>=200_000_000).astype(int).to_numpy()
    lp=lm[["disasterNumber","state","fyDeclared","totalObligatedFunding","doc_chars"]].copy()
    lp["actual_mid"]=ylm
    for mode in ["text","text_plus_counts"]:
        r=lfyo(lm,ylm,mode); results["lower_"+mode]=r
        lp[mode+"_pred"]=r["predictions"]; lp[mode+"_prob"]=r["probabilities"]
    lp.to_csv(OUT/"lower_predictions.csv",index=False)

    # extreme
    yex=(h.totalObligatedFunding>=500_000_000).astype(int).to_numpy()
    ep=h[["disasterNumber","state","fyDeclared","totalObligatedFunding","doc_chars"]].copy()
    ep["actual_extreme"]=yex
    for mode in ["text","text_plus_counts"]:
        r=lfyo(h,yex,mode); results["extreme_"+mode]=r
        ep[mode+"_pred"]=r["predictions"]; ep[mode+"_prob"]=r["probabilities"]
    ep.to_csv(OUT/"extreme_predictions.csv",index=False)

    # Save redacted texts for audit, not financial fields.
    h[["disasterNumber","state","fyDeclared","ma_text","doc_chars"]].to_csv(OUT/"redacted_documents.csv",index=False)

    (OUT/"summary.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    md=["# Hurricane Mission Assignment text specialist","","## Lower vs middle"]
    for mode in ["text","text_plus_counts"]:
        r=results["lower_"+mode]
        md.append(f"- {mode}: lower={r['negative_recall']:.1%}, middle={r['positive_recall']:.1%}, BA={r['balanced_accuracy']:.3f}")
    md += ["","## Extreme"]
    for mode in ["text","text_plus_counts"]:
        r=results["extreme_"+mode]
        md.append(f"- {mode}: <500M={r['negative_recall']:.1%}, 500M+={r['positive_recall']:.1%}, BA={r['balanced_accuracy']:.3f}")
    (OUT/"summary.md").write_text("\n".join(md),encoding="utf-8")
    print("\n".join(md))

if __name__=="__main__":
    main()
