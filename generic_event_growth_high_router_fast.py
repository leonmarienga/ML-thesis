from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, confusion_matrix, balanced_accuracy_score, f1_score

import external_physical_severity_high_router_experiment as base
import cdc_biological_severity_signal_audit as cdc_audit

OUT=Path("generic_event_growth_high_router_fast_results")
OUT.mkdir(exist_ok=True)

GROWTH=[
    "eventGrowth7LogRatio",
    "eventGrowth14LogRatio",
    "recentActivityPerMillionLog",
    "recentFatalityPerMillionLog",
    "onsetAgeDaysLog",
]
SAFE_CATS=[c for c in base.BASE_CAT if c!="eventScale"]
FOCUS=[4480,4482,4485,4486,4489,4515]


def num(x):
    return pd.to_numeric(x,errors="coerce").replace([np.inf,-np.inf],np.nan)


def cdc_one(g,decl,pop):
    if pd.isna(decl) or g.empty or not pop or pop<=0:
        return None
    z=g[g["date"]<=decl].sort_values("date")
    if z.empty:return None
    def s(lo,hi,col):
        if col not in z:return 0.0
        x=z[(z["date"]>lo)&(z["date"]<=hi)]
        return float(num(x[col]).fillna(0).clip(lower=0).sum())
    r7=s(decl-pd.Timedelta(days=7),decl,"new_case")
    p7=s(decl-pd.Timedelta(days=14),decl-pd.Timedelta(days=7),"new_case")
    r14=s(decl-pd.Timedelta(days=14),decl,"new_case")
    p14=s(decl-pd.Timedelta(days=28),decl-pd.Timedelta(days=14),"new_case")
    d7=s(decl-pd.Timedelta(days=7),decl,"new_death")
    first=z[num(z["tot_cases"]).fillna(0)>0]
    onset=float((decl-first["date"].min()).days) if len(first) else 0.0
    return {
        "eventGrowth7LogRatio":float(np.log((r7+1)/(p7+1))),
        "eventGrowth14LogRatio":float(np.log((r14+1)/(p14+1))),
        "recentActivityPerMillionLog":float(np.log1p(r7/pop*1e6)),
        "recentFatalityPerMillionLog":float(np.log1p(d7/pop*1e6)),
        "onsetAgeDaysLog":float(np.log1p(max(onset,0))),
        "growthSourceAudit":"CDC",
        "rawRecent7":r7,"rawPrev7":p7,"rawRecent14":r14,"rawPrev14":p14,
    }


def noaa_one(g,decl,types,pop):
    if pd.isna(decl) or not types or not pop or pop<=0:
        return None
    z=g[
        g["EVENT_TYPE"].isin(types)
        & (g["END_DT"]<=decl)
        & (g["END_DT"]>decl-pd.Timedelta(days=28))
    ].copy()
    if z.empty:
        r7=p7=r14=p14=d7=0.0; onset=0.0
    else:
        def cnt(lo,hi):
            return float(((z["END_DT"]>lo)&(z["END_DT"]<=hi)).sum())
        r7=cnt(decl-pd.Timedelta(days=7),decl)
        p7=cnt(decl-pd.Timedelta(days=14),decl-pd.Timedelta(days=7))
        r14=cnt(decl-pd.Timedelta(days=14),decl)
        p14=cnt(decl-pd.Timedelta(days=28),decl-pd.Timedelta(days=14))
        recent=z[(z["END_DT"]>decl-pd.Timedelta(days=7))&(z["END_DT"]<=decl)]
        d7=float((num(recent["DEATHS_DIRECT"]).fillna(0)+num(recent["DEATHS_INDIRECT"]).fillna(0)).clip(lower=0).sum()) if len(recent) else 0.0
        earliest=z["BEGIN_DT"].min()
        onset=float((decl-earliest).days) if pd.notna(earliest) else 0.0
    return {
        "eventGrowth7LogRatio":float(np.log((r7+1)/(p7+1))),
        "eventGrowth14LogRatio":float(np.log((r14+1)/(p14+1))),
        "recentActivityPerMillionLog":float(np.log1p(r7/pop*1e6)),
        "recentFatalityPerMillionLog":float(np.log1p(d7/pop*1e6)),
        "onsetAgeDaysLog":float(np.log1p(max(onset,0))),
        "growthSourceAudit":"NOAA",
        "rawRecent7":r7,"rawPrev7":p7,"rawRecent14":r14,"rawPrev14":p14,
    }


def build(source):
    forbidden={"target","band","totalObligatedFunding","eventScale","fundingPerMission","fundingPerAgency","fundingPerDay"}
    bad=forbidden.intersection(source.columns)
    if bad:raise RuntimeError(f"Forbidden fields in external builder: {sorted(bad)}")
    years=sorted(pd.to_datetime(source["effectiveDeclarationDate"],errors="coerce").dropna().dt.year.astype(int).unique())
    years=[int(y) for y in years if 2010<=int(y)<=2024]
    noaa=base.fetch_noaa(years)
    cdc=cdc_audit.fetch_cdc()
    noaa_by_state={abbr:noaa[noaa["STATE"]==name].copy() for abbr,name in base.STATE_NAME.items()}
    cdc_by_state={st:g.copy() for st,g in cdc.groupby("state_abbr")}
    rows=[]
    for _,r in source.iterrows():
        dn=int(r["disasterNumber"]); st=str(r["state"]); inc=str(r["incidentType"])
        decl=pd.to_datetime(r["effectiveDeclarationDate"],errors="coerce")
        pop=float(base.POP.get(st,np.nan))
        if inc.upper().strip()=="BIOLOGICAL":
            feat=cdc_one(cdc_by_state.get(st,pd.DataFrame()),decl,pop)
        else:
            feat=noaa_one(noaa_by_state.get(st,pd.DataFrame()),decl,base.noaa_types(inc),pop)
        if feat is None:
            feat={c:0.0 for c in GROWTH}
            feat.update({"growthSourceAudit":"NEUTRAL_UNSUPPORTED","rawRecent7":np.nan,"rawPrev7":np.nan,"rawRecent14":np.nan,"rawPrev14":np.nan})
        rows.append({"disasterNumber":dn,**feat})
    e=pd.DataFrame(rows)
    e.to_csv(OUT/"event_growth_all_disasters.csv",index=False)
    return e


def met(y,p):
    rec=recall_score(y,p,labels=[3,4,5],average=None,zero_division=0)
    cm=confusion_matrix(y,p,labels=[3,4,5])
    return {"r3":float(rec[0]),"r4":float(rec[1]),"r5":float(rec[2]),
            "correct3":int(cm[0,0]),"correct4":int(cm[1,1]),"correct5":int(cm[2,2]),
            "min_recall":float(rec.min()),"balanced_accuracy":float(balanced_accuracy_score(y,p)),
            "macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),
            "confusion_matrix":cm.tolist(),
            "pass_counts":bool(cm[0,0]>=32 and cm[1,1]>=8 and cm[2,2]>=6)}


def pair(high,lo,hi,nums,cats):
    mask=high["band"].isin([lo,hi]).to_numpy()
    p=np.full(len(high),np.nan)
    for yr in sorted(high.loc[mask,"fyDeclared"].astype(int).unique()):
        tr=mask&(high["fyDeclared"].astype(int).to_numpy()!=yr)
        te=mask&(high["fyDeclared"].astype(int).to_numpy()==yr)
        Xtr=base.prep(high.loc[tr],nums,cats); Xte=base.prep(high.loc[te],nums,cats)
        ytr=(high.loc[tr,"band"]==hi).astype(int)
        m=base.fit_cat(Xtr,ytr,Xte,cats); cls=list(m.classes_)
        p[te]=m.predict_proba(Xte)[:,cls.index(1)]
    yy=(high.loc[mask,"band"]==hi).astype(int).to_numpy()
    return p,base.op_point(yy,p[mask])


def main():
    d=base.load_master()
    source=d[["disasterNumber","state","incidentType","effectiveDeclarationDate"]].copy()
    e=build(source)
    d=d.merge(e,on="disasterNumber",how="left")
    for c in GROWTH:d[c]=num(d[c]).fillna(0)
    high=d[d.band.isin([3,4,5])].copy().reset_index(drop=True)
    safe_num=[c for c in base.BASE_NUM if c in high.columns]
    cats=[c for c in SAFE_CATS if c in high.columns]
    variants={"baseline_safe":safe_num,"baseline_plus_event_growth":safe_num+GROWTH}
    mrows=[]; prows=[]; pairs=[]; pp=[]
    years=sorted(high.fyDeclared.astype(int).unique())
    for name,nums in variants.items():
        pred=np.full(len(high),-99,int)
        for yr in years:
            tr=high.fyDeclared.astype(int)!=yr; te=~tr
            Xtr=base.prep(high.loc[tr],nums,cats); Xte=base.prep(high.loc[te],nums,cats)
            m=base.fit_cat(Xtr,high.loc[tr,"band"].astype(int),Xte,cats)
            pred[te]=np.asarray(m.predict(Xte)).reshape(-1).astype(int)
        mm=met(high.band.astype(int).to_numpy(),pred)
        mrows.append({"variant":name,**{k:v for k,v in mm.items() if k!="confusion_matrix"},"confusion_matrix":json.dumps(mm["confusion_matrix"])})
        q=high[["disasterNumber","state","fyDeclared","incidentType","target","band","growthSourceAudit"]].copy()
        q["variant"]=name;q["predicted_band"]=pred;prows.append(q)
        for lo,hi in [(3,4),(4,5)]:
            prob,op=pair(high,lo,hi,nums,cats)
            pairs.append({"variant":name,"pair":f"{lo}vs{hi}",**op})
            mk=high.band.isin([lo,hi])
            qq=high.loc[mk,["disasterNumber","state","fyDeclared","incidentType","target","band","growthSourceAudit"]].copy()
            qq["variant"]=name;qq["pair"]=f"{lo}vs{hi}";qq["prob_upper"]=prob[mk.to_numpy()];pp.append(qq)
    pd.DataFrame(mrows).to_csv(OUT/"multiclass_results.csv",index=False)
    pd.DataFrame(pairs).to_csv(OUT/"pairwise_results.csv",index=False)
    pd.concat(prows,ignore_index=True).to_csv(OUT/"multiclass_predictions.csv",index=False)
    pd.concat(pp,ignore_index=True).to_csv(OUT/"pairwise_predictions.csv",index=False)
    focus=high[high.disasterNumber.isin(FOCUS)][["disasterNumber","state","fyDeclared","incidentType","target","band","growthSourceAudit"]+GROWTH+["rawRecent7","rawPrev7","rawRecent14","rawPrev14"]].sort_values("disasterNumber")
    focus.to_csv(OUT/"focus_2020_growth.csv",index=False)
    summary={
        "purpose":"Fast target-free cross-hazard test of the raw recent-vs-previous severity-growth concept identified in the Biological audit.",
        "high_counts":{str(b):int((high.band==b).sum()) for b in [3,4,5]},
        "multiclass":mrows,"pairwise":pairs,"focus_cases":focus.to_dict("records"),
        "guardrails":[
            "Features are built for all disasters without target, band, funding amounts, funding-per-resource fields, or eventScale.",
            "Biological uses CDC data on or before declaration; physical hazards use NOAA records ending on or before declaration.",
            "HHS hospital variables are excluded.",
            "The original funding-derived eventScale is excluded from the predictive model.",
            "All predictions are leave-fiscal-year-out; pairwise operating thresholds are development OOF diagnostics only."
        ]}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str),flush=True)

if __name__=="__main__":main()
