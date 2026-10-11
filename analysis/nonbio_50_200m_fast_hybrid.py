#!/usr/bin/env python3
"""Research-only: nested LFYO amount specialization for 50M <= target < 200M.
Only 912 non-Biological disasters. No router changes, target-derived predictors,
or post-hoc ID exceptions. All selections are per-fold inner LFYO.
Reported outer-test scores are exploratory after repeated architecture development.
"""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "audit_outputs" / "nonbio_50_200m_fast_hybrid"
LO, HI = 50.0, 200.0  # Millions USD
CATS = ["state", "incidentType", "expectedResourceLevel", "disasterCategory", "durationClass"]
NUMS = ["durationDays", "declarationDelayDays", "expectedResourceScore",
        "missionAssignmentCount", "uniqueAgencyCount", "uniqueMaTypeCount",
        "uniquePriorityCount", "responseComplexityScore", "missionDensity",
        "agencyDensity", "ihProgramDeclared", "paProgramDeclared",
        "hmProgramDeclared"]
SEED = 20261011
ALPHAS = [0.10, 0.25, 0.50, 0.75, 1.0]
SPECS = {
  "ridge_raw_20_300": ("ridge", 20,300, "raw", 35),
  "ridge_log_20_300": ("ridge", 20,300, "log", 35),
  "ridge_raw_10_500": ("ridge", 10,500, "raw", 80),
  "et_raw_20_300": ("extra", 20,300, "raw", 2),
  "et_log_20_300": ("extra", 20,300, "log", 2),
  "rf_raw_20_300": ("forest", 20,300, "raw", 3),
  "analogue_k3_20_300": ("analogue", 20,300, "raw", 3),
  "analogue_k5_10_500": ("analogue", 10,500, "raw", 5),
  "ordinal_20_300": ("ordinal",20,300,"raw",1)
}
MIXES = {
  "ridge_tree": ("ridge_raw_20_300", "et_raw_20_300"),
  "ridge_analogue": ("ridge_raw_20_300", "analogue_k3_20_300"),
  "tree_analogue": ("et_raw_20_300", "analogue_k3_20_300")
}

def load():
    f = ROOT / "master_openfema_40plus.xlsx"
    d = pd.read_excel(f)
    assert len(d)==971, f"Unexpected original dataset size: {len(d)}"
    d = d[~d["incidentType"].astype(str).str.contains("biolog",case=False,na=False)].copy()
    assert len(d)==912, f"Unexpected non-biological size: {len(d)}"
    d["target_m"] = pd.to_numeric(d["totalObligatedFunding"],errors="raise").clip(lower=0) / 1e6
    d["fyDeclared"] = pd.to_numeric(d["fyDeclared"],errors="raise").astype(int)
    for c in CATS:
        if c not in d: raise KeyError(c)
        d[c] = d[c].fillna("MISSING").astype(str)
    for c in NUMS:
        if c not in d: raise KeyError(c)
        if d[c].dtype=="object":
            d[c]=d[c].astype(str).str.lower().replace(
              {"true":"1","false":"0","yes":"1","no":"0","nan":np.nan})
        d[c] = pd.to_numeric(d[c],errors="coerce")
    assert int(((d.target_m>=LO)&(d.target_m<HI)).sum())==13
    return d.reset_index(drop=True)

def preprocessor():
    return ColumnTransformer([
      ("num", Pipeline([
         ("imp",SimpleImputer(strategy="median",keep_empty_features=True)),
         ("scale",RobustScaler(quantile_range=(10,90)))]), NUMS),
      ("cat", Pipeline([
         ("imp",SimpleImputer(strategy="constant",fill_value="MISSING")),
         ("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]), CATS)
    ],sparse_threshold=0)

def metrics(actual,pred):
    a=np.asarray(actual,dtype=float); p=np.asarray(pred,dtype=float)
    e=np.abs(a-p); rel=e/np.maximum(a,1e-8)
    return {"n":int(len(a)),"MAE_M":float(e.mean()),
       "RMSE_M":float(np.sqrt(np.mean((a-p)**2))),
       "MedAE_M":float(np.median(e)),
       "R2":float(r2_score(a,p)) if len(a)>=2 and np.var(a)>0 else None,
       "log_MAE":float(np.abs(np.log1p(a)-np.log1p(p)).mean()),
       "within20":float(np.mean(rel<=0.20)),
       "within30":float(np.mean(rel<=0.30)),
       "within50":float(np.mean(rel<=0.50)),
       "unique_predictions":int(len(np.unique(np.round(p,6))))}

def analogue(tr,te,k):
    cols=["durationDays","declarationDelayDays","expectedResourceScore",
        "missionAssignmentCount","uniqueAgencyCount","uniqueMaTypeCount",
        "uniquePriorityCount","responseComplexityScore","missionDensity",
        "agencyDensity"]
    a=tr[cols].copy();b=te[cols].copy()
    med=a.median().fillna(0)
    a=a.fillna(med).to_numpy(float);b=b.fillna(med).to_numpy(float)
    scale=RobustScaler(quantile_range=(10,90)).fit(a)
    a=np.clip(scale.transform(a),-8,8);b=np.clip(scale.transform(b),-8,8)
    dist=np.mean((b[:,None,:]-a[None,:,:])**2,axis=2)
    dist+=2.0*(te.incidentType.to_numpy()[:,None]!=tr.incidentType.to_numpy()[None,:])
    dist+=0.30*(te.state.to_numpy()[:,None]!=tr.state.to_numpy()[None,:])
    y=tr.target_m.to_numpy(float)
    output=[]
    for row in dist:
        j=np.argsort(row,kind="stable")[:min(k,len(row))]
        w=1.0/(0.2+np.sqrt(row[j]))
        output.append(np.average(y[j],weights=w))
    return np.clip(output,LO,HI-1e-6)

def predict_expert(name, train, test):
    kind,low,high,target,param=SPECS[name]
    tr=train[(train.target_m>=low)&(train.target_m<high)]
    if len(tr)<8: raise RuntimeError(f"Not enough training rows {name}: {len(tr)}")
    if kind=="analogue":return analogue(tr,test,int(param))
    t=np.clip(tr.target_m.to_numpy(float),LO,HI-1e-6)
    prep=preprocessor()
    X=tr[CATS+NUMS]; XT=test[CATS+NUMS]
    if kind=="ordinal":
        a=prep.fit_transform(X);b=prep.transform(XT)
        probs=[]
        for boundary in [62.5,87.5,112.5,137.5,162.5,187.5]:
            target_y=(t>=boundary).astype(int)
            if len(np.unique(target_y))==1: pp=np.full(len(test),target_y[0],float)
            else:
                clf=LogisticRegression(C=0.15,solver="liblinear",max_iter=250,random_state=SEED)
                clf.fit(a,target_y);pp=clf.predict_proba(b)[:,1]
            probs.append(pp)
        ps=np.minimum.accumulate(np.asarray(probs),axis=0)
        return np.clip(LO+25.0*ps.sum(axis=0),LO,HI-1e-6)
    if kind=="ridge": reg=Ridge(alpha=param,solver="lsqr")
    elif kind=="extra":reg=ExtraTreesRegressor(n_estimators=140,min_samples_leaf=param,
                             max_features=0.8,n_jobs=-1,random_state=SEED)
    elif kind=="forest":reg=RandomForestRegressor(n_estimators=140,min_samples_leaf=param,
                             max_features=0.8,n_jobs=-1,random_state=SEED)
    else: raise ValueError(kind)
    pipeline=Pipeline([("pre",prep),("reg",reg)])
    pipeline.fit(X,np.log(t) if target=="log" else t)
    pred=pipeline.predict(XT)
    if target=="log":pred=np.exp(np.clip(pred,np.log(LO),np.log(HI)))
    return np.clip(pred,LO,HI-1e-6)

def candidate_preds(train,valid):
    p={name:predict_expert(name,train,valid) for name in SPECS}
    for name,members in MIXES.items():
        p[name]=np.mean([p[m] for m in members],axis=0)
    return p

def select(train):
    years=sorted(train.loc[(train.target_m>=LO)&(train.target_m<HI),"fyDeclared"].unique())
    truths=[];meds=[];preds={name:[] for name in list(SPECS)+list(MIXES)}
    for iy in years:
        t=train[train.fyDeclared!=iy]
        v=train[(train.fyDeclared==iy)&(train.target_m>=LO)&(train.target_m<HI)]
        m=float(t.loc[(t.target_m>=LO)&(t.target_m<HI),"target_m"].median())
        z=candidate_preds(t,v)
        truths.extend(v.target_m.tolist());meds.extend([m]*len(v))
        for name in preds:preds[name].extend(z[name].tolist())
    truth=np.asarray(truths); med=np.asarray(meds)
    rows=[]
    for name,base in preds.items():
        base=np.asarray(base)
        for alpha in ALPHAS:
            pp=np.clip(med+alpha*(base-med),LO,HI-1e-6)
            if np.ptp(pp)<0.00001:continue
            score=metrics(truth,pp)
            rows.append({"candidate":name,"alpha":alpha, "inner_MAE_M":score["MAE_M"],
                         "inner_RMSE_M":score["RMSE_M"],"inner_n":len(truth)})
    rows.sort(key=lambda r:(r["inner_MAE_M"],r["inner_RMSE_M"],r["candidate"],r["alpha"]))
    assert rows, "No non-constant inner candidates"
    return rows[0], rows

def run():
    d=load()
    pred_rows=[];folds=[]
    years=sorted(d.loc[(d.target_m>=LO)&(d.target_m<HI),"fyDeclared"].unique())
    for year in years:
        tr=d[d.fyDeclared!=year]
        te=d[(d.fyDeclared==year)&(d.target_m>=LO)&(d.target_m<HI)]
        m=float(tr.loc[(tr.target_m>=LO)&(tr.target_m<HI),"target_m"].median())
        broad=float(tr.loc[(tr.target_m>=20)&(tr.target_m<300),"target_m"].median())
        selected,_=select(tr)
        pred=candidate_preds(tr,te)[selected["candidate"]]
        final=np.clip(m+selected["alpha"]*(pred-m),LO,HI-1e-6)
        for (_,row),z in zip(te.iterrows(),final):
            pred_rows.append({"disasterNumber":int(row.disasterNumber),"fyDeclared":int(year),
                "state":row.state,"incidentType":row.incidentType,
                "actual_m":float(row.target_m),"median_m":m,"broad_median_m":broad,
                "hybrid_m":float(z),"candidate":selected["candidate"],"alpha":selected["alpha"]})
        folds.append({"outer_fy":int(year),"outer_n":len(te),"outer_training_band_n":int(((tr.target_m>=LO)&(tr.target_m<HI)).sum()),
                      **selected})
        print(f"FOLD FY={year}: selected {selected['candidate']} alpha={selected['alpha']}",flush=True)
    p=pd.DataFrame(pred_rows);f=pd.DataFrame(folds)
    yy=p.actual_m.to_numpy();aa=p.hybrid_m.to_numpy();mm=p.median_m.to_numpy()
    report={"direct_true_band":{"median":metrics(yy,mm),"broad_median":metrics(yy,p.broad_median_m.to_numpy()),
             "selected_individualized":metrics(yy,aa)},
            "conditions":{"outer_LFYO":True,"all_model_and_shrinkage_selection":"inner LFYO",
              "router_unchanged":True,"funding_target_not_in_X":True,"n_true_band":len(p)},
            "acceptance_gates":{"lower_MAE":float(np.mean(abs(yy-aa)))<float(np.mean(abs(yy-mm))),
            "no_worse_RMSE":float(np.mean((yy-aa)**2))<=float(np.mean((yy-mm)**2)),
            "individualized":len(np.unique(np.round(aa,6)))>1},
            "fold_selected_counts":dict(Counter(f.candidate)),
            "caution":"Exploratory nested outer scores; repeat model-family search can still cause research-level selection bias."}
    OUTPUT.mkdir(parents=True,exist_ok=True)
    p.to_csv(OUTPUT/"direct_predictions.csv",index=False)
    f.to_csv(OUTPUT/"fold_selections.csv",index=False)
    (OUTPUT/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("BEGIN_FINAL_METRICS_JSON",flush=True)
    print(json.dumps(report,indent=2),flush=True)
    print("END_FINAL_METRICS_JSON",flush=True)
if __name__=="__main__":run()
