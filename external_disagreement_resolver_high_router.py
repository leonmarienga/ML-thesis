from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import external_physical_severity_high_router_experiment as master_base
import cdc_biological_severity_signal_audit as cdc_audit
import declared_population_exposure_router_experiment_v2 as exposure_v2
import mission_composition_experiment as mission_exp

OUT = Path("external_disagreement_resolver_high_router_results")
OUT.mkdir(exist_ok=True)

PAIR_LABELS = [3, 4, 5]
ACCEPT_COUNTS = {3: 32, 4: 8, 5: 6}

A_NUM_BASE = [
    "durationDays","declarationDelayDays","expectedResourceScore",
    "missionAssignmentCount","uniqueAgencyCount","uniqueMaTypeCount","uniquePriorityCount",
    "responseComplexityScore","missionDensity","agencyDensity","population2010","eventSize",
    "logPopulation","logMission","logComplexity","logAgency","logDuration","logEventSize",
    "missionAssignmentCount_event_pct_rank","responseComplexityScore_event_pct_rank",
    "uniqueAgencyCount_event_pct_rank","population2010_event_pct_rank",
    "missionAssignmentCount_relative_event_avg","responseComplexityScore_relative_event_avg",
    "uniqueAgencyCount_relative_event_avg","population2010_relative_event_avg",
    "ihProgramDeclared","iaProgramDeclared","paProgramDeclared","hmProgramDeclared",
]
A_EXPOSURE = [
    "declaredCountyMatchedCount","declaredUniquePlaceCount","declaredGeographyRowCount",
    "declaredCountyMatchRatio","declaredPopulation2010","logDeclaredPopulation2010",
    "declaredPopulationShareState","declaredCountyPopulationMean","declaredCountyPopulationMedian",
    "declaredCountyPopulationMax","declaredLandSqMi","logDeclaredLandSqMi",
    "declaredLandShareState","declaredPopulationDensity","statewideDeclarationFlag",
]
A_CATS = ["incidentType","state","expectedResourceLevel","disasterCategory","durationClass"]

A_CONFIGS = [
    {"max_depth":3,"min_samples_leaf":1,"max_features":0.75},
    {"max_depth":4,"min_samples_leaf":1,"max_features":0.80},
    {"max_depth":5,"min_samples_leaf":1,"max_features":0.85},
    {"max_depth":5,"min_samples_leaf":2,"max_features":0.85},
]

B_WEIGHT_OPTIONS = ["uniform","distance"]
B_ALPHAS = [0.0,0.25,0.5,0.75,1.0]

EXT_DIMS = [
    "extPersistencePct","extExtentPct","extFatalityPct",
    "extAcutePct","extIntensityPct","extBurdenPct",
]

NOAA_INDEX = master_base.NOAA_INDEX


def num(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf,-np.inf], np.nan)


def all_metrics(y, pred):
    y=np.asarray(y,int); pred=np.asarray(pred,int)
    rec=recall_score(y,pred,labels=PAIR_LABELS,average=None,zero_division=0)
    cm=confusion_matrix(y,pred,labels=PAIR_LABELS)
    return {
        "r3":float(rec[0]),"r4":float(rec[1]),"r5":float(rec[2]),
        "correct3":int(cm[0,0]),"correct4":int(cm[1,1]),"correct5":int(cm[2,2]),
        "min_recall":float(rec.min()),
        "pass_counts":bool(cm[0,0]>=32 and cm[1,1]>=8 and cm[2,2]>=6),
        "confusion_matrix":cm.tolist(),
    }


def binary_metrics(y, pred):
    y=np.asarray(y,int); pred=np.asarray(pred,int)
    neg=float(((pred==0)&(y==0)).sum()/max((y==0).sum(),1))
    pos=float(((pred==1)&(y==1)).sum()/max((y==1).sum(),1))
    return {"lower_recall":neg,"upper_recall":pos,"min_recall":min(neg,pos)}


def best_binary_threshold(y, p):
    y=np.asarray(y,int); p=np.asarray(p,float)
    fpr,tpr,thr=roc_curve(y,p)
    tnr=1-fpr
    feasible=np.where((tpr>=.8)&(tnr>=.8))[0]
    cand=feasible if len(feasible) else np.arange(len(thr))
    mins=np.minimum(tpr[cand],tnr[cand])
    best=mins.max()
    c2=cand[mins==best]
    j=int(c2[np.argmax((tpr[c2]+tnr[c2])/2)])
    return {
        "threshold":float(thr[j]),
        "lower_recall":float(tnr[j]),
        "upper_recall":float(tpr[j]),
        "min_recall":float(min(tnr[j],tpr[j])),
        "pass80":bool(len(feasible)),
    }


def ensure_mission_features(d):
    path=mission_exp.OUT/"mission_composition_features.csv"
    if path.exists():
        return pd.read_csv(path)
    dns=d["disasterNumber"].astype(int).tolist()
    raw=mission_exp.fetch_openfema_for_disasters(dns)
    feats=mission_exp.composition_features(raw,dns)
    feats.to_csv(path,index=False)
    return feats


def find_normal_priority_share(comp):
    candidates=[c for c in comp.columns if c.lower().startswith("priority_share__") and "normal" in c.lower()]
    if not candidates:
        # Keep the scientific definition explicit. Do not silently substitute another priority.
        raise RuntimeError("No Normal-priority mission-share column found in frozen MissionAssignments features.")
    # If naming variants exist, choose the shortest exact-like label deterministically.
    candidates=sorted(candidates,key=lambda c:(len(c),c))
    return candidates[0]


def fetch_noaa_enhanced(years):
    html=requests.get(NOAA_INDEX,timeout=180,headers={"User-Agent":"ML-thesis-research/1.0"}).text
    frames=[]
    use=[
        "STATE","EVENT_TYPE","BEGIN_DATE_TIME","END_DATE_TIME",
        "INJURIES_DIRECT","INJURIES_INDIRECT","DEATHS_DIRECT","DEATHS_INDIRECT",
        "MAGNITUDE","CZ_TYPE","CZ_FIPS","CZ_NAME","BEGIN_LAT","BEGIN_LON","END_LAT","END_LON",
    ]
    for y in sorted(set(int(x) for x in years)):
        names=re.findall(rf"StormEvents_details-ftp_v1\.0_d{y}_c\d+\.csv\.gz",html)
        if not names:
            continue
        name=sorted(set(names))[-1]
        raw=requests.get(NOAA_INDEX+name,timeout=300,headers={"User-Agent":"ML-thesis-research/1.0"}).content
        df=pd.read_csv(io.BytesIO(raw),compression="gzip",usecols=lambda c:c in use,low_memory=False)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    n=pd.concat(frames,ignore_index=True)
    n["STATE"]=n["STATE"].astype(str).str.upper().str.strip()
    n["EVENT_TYPE"]=n["EVENT_TYPE"].astype(str).str.upper().str.strip()
    n["BEGIN_DT"]=pd.to_datetime(n["BEGIN_DATE_TIME"],errors="coerce",utc=True).dt.tz_convert(None)
    n["END_DT"]=pd.to_datetime(n["END_DATE_TIME"],errors="coerce",utc=True).dt.tz_convert(None)
    for c in ["INJURIES_DIRECT","INJURIES_INDIRECT","DEATHS_DIRECT","DEATHS_INDIRECT",
              "MAGNITUDE","BEGIN_LAT","BEGIN_LON","END_LAT","END_LON"]:
        if c not in n.columns:n[c]=np.nan
        n[c]=num(n[c])
    return n


def rank_pct(ref, col, idx):
    x=num(ref[col]).fillna(0)
    return float(x.rank(pct=True,method="average").loc[idx])


def cdc_dims(cdc, decl, target_state):
    eligible=cdc[cdc["date"]<=decl].copy()
    if eligible.empty:return None
    snap=eligible.sort_values(["state_abbr","date"]).groupby("state_abbr",as_index=False).tail(1).copy()
    snap["population"]=snap["state_abbr"].map(master_base.POP)
    snap=snap[num(snap["population"]).fillna(0)>0].copy()
    if target_state not in set(snap["state_abbr"]):return None

    pop=num(snap["population"])
    snap["cum_cases_abs"]=num(snap["tot_cases"]).fillna(0).clip(lower=0)
    snap["case_rate"]=snap["cum_cases_abs"]/pop*1e5
    snap["death_rate"]=num(snap["tot_death"]).fillna(0).clip(lower=0)/pop*1e5
    snap["new_case_rate"]=num(snap["new_case"]).fillna(0).clip(lower=0)/pop*1e5
    snap["new_death_rate"]=num(snap["new_death"]).fillna(0).clip(lower=0)/pop*1e5
    snap["acute_rate"]=snap["new_case_rate"]+10*snap["new_death_rate"]
    snap["intensity_rate"]=snap["new_case_rate"]
    snap["burden_rate"]=snap["case_rate"]

    first={}
    for st,g in eligible.groupby("state_abbr"):
        gg=g[num(g["tot_cases"]).fillna(0)>0]
        first[st]=gg["date"].min() if len(gg) else pd.NaT
    snap["first_case_date"]=snap["state_abbr"].map(first)
    snap["persistence_days"]=(decl-snap["first_case_date"]).dt.days.clip(lower=0).fillna(0)
    snap["extent_abs"]=snap["cum_cases_abs"]

    zidx=snap.index[snap["state_abbr"]==target_state][-1]
    return {
        "extPersistencePct":rank_pct(snap,"persistence_days",zidx),
        "extExtentPct":rank_pct(snap,"extent_abs",zidx),
        "extFatalityPct":rank_pct(snap,"death_rate",zidx),
        "extAcutePct":rank_pct(snap,"acute_rate",zidx),
        "extIntensityPct":rank_pct(snap,"intensity_rate",zidx),
        "extBurdenPct":rank_pct(snap,"burden_rate",zidx),
        "extSourceAudit":"CDC",
    }


def noaa_dims(noaa, decl, incident, target_state):
    types=master_base.noaa_types(incident)
    if not types or noaa.empty:return None
    lo=decl-pd.Timedelta(days=21)
    n=noaa[(noaa["EVENT_TYPE"].isin(types))&(noaa["END_DT"]<=decl)&(noaa["END_DT"]>=lo)].copy()

    states=[]
    for abbr,pop in master_base.POP.items():
        nm=master_base.STATE_NAME.get(abbr)
        if nm:states.append({"state_abbr":abbr,"STATE":nm,"population":float(pop)})
    ref=pd.DataFrame(states)
    if target_state not in set(ref["state_abbr"]):return None

    if len(n):
        n["deaths"]=(num(n["DEATHS_DIRECT"]).fillna(0)+num(n["DEATHS_INDIRECT"]).fillna(0)).clip(lower=0)
        n["injuries"]=(num(n["INJURIES_DIRECT"]).fillna(0)+num(n["INJURIES_INDIRECT"]).fillna(0)).clip(lower=0)
        n["dur_days"]=((n["END_DT"].fillna(n["BEGIN_DT"])-n["BEGIN_DT"]).dt.total_seconds()/86400).clip(lower=0)
        n["area_key"]=n.get("CZ_TYPE","").astype(str)+"|"+n.get("CZ_FIPS","").astype(str)+"|"+n.get("CZ_NAME","").astype(str)
        g=n.groupby("STATE")
        agg=g.agg(
            record_count=("EVENT_TYPE","size"),
            unique_areas=("area_key","nunique"),
            deaths=("deaths","sum"),
            injuries=("injuries","sum"),
            max_magnitude=("MAGNITUDE","max"),
            persistence=("dur_days","median"),
        ).reset_index()
    else:
        agg=pd.DataFrame(columns=["STATE","record_count","unique_areas","deaths","injuries","max_magnitude","persistence"])

    ref=ref.merge(agg,on="STATE",how="left")
    for c in ["record_count","unique_areas","deaths","injuries","max_magnitude","persistence"]:
        ref[c]=num(ref[c]).fillna(0).clip(lower=0)
    ref["fatality_rate"]=ref["deaths"]/ref["population"]*1e6
    ref["acute_rate"]=ref["injuries"]/ref["population"]*1e6
    ref["burden_rate"]=ref["record_count"]/ref["population"]*1e6
    zidx=ref.index[ref["state_abbr"]==target_state][-1]
    return {
        "extPersistencePct":rank_pct(ref,"persistence",zidx),
        "extExtentPct":rank_pct(ref,"unique_areas",zidx),
        "extFatalityPct":rank_pct(ref,"fatality_rate",zidx),
        "extAcutePct":rank_pct(ref,"acute_rate",zidx),
        "extIntensityPct":rank_pct(ref,"max_magnitude",zidx),
        "extBurdenPct":rank_pct(ref,"burden_rate",zidx),
        "extSourceAudit":"NOAA",
    }


def build_external_dimensions(source):
    forbidden={"target","band","totalObligatedFunding","fundingPerMission","fundingPerAgency","fundingPerDay","eventScale"}
    bad=forbidden.intersection(source.columns)
    if bad:raise RuntimeError(f"Forbidden target/funding-derived columns reached external builder: {sorted(bad)}")

    dates=pd.to_datetime(source["effectiveDeclarationDate"],errors="coerce")
    years=sorted(dates.dropna().dt.year.astype(int).unique().tolist())
    noaa=fetch_noaa_enhanced([y for y in years if 2010<=y<=2024])
    cdc=cdc_audit.fetch_cdc()

    cache={}
    rows=[]
    for _,r in source.iterrows():
        dn=int(r["disasterNumber"]); st=str(r["state"]); inc=str(r["incidentType"])
        decl=pd.to_datetime(r["effectiveDeclarationDate"],errors="coerce")
        feat=None
        if pd.notna(decl):
            key=(str(decl.date()),inc.upper().strip(),st)
            if key in cache:
                feat=cache[key]
            else:
                if inc.upper().strip()=="BIOLOGICAL":
                    feat=cdc_dims(cdc,decl,st)
                else:
                    feat=noaa_dims(noaa,decl,inc,st)
                cache[key]=feat
        if feat is None:
            feat={c:0.5 for c in EXT_DIMS}
            feat["extSourceAudit"]="NEUTRAL_UNSUPPORTED"
        rows.append({"disasterNumber":dn,**feat})
    out=pd.DataFrame(rows)
    out.to_csv(OUT/"external_physical_dimensions_all_disasters.csv",index=False)
    return out


def prep_a_fit(train, features_num, features_cat, cfg):
    X=train[features_num+features_cat].copy()
    pre=ColumnTransformer([
        ("num",Pipeline([("imp",SimpleImputer(strategy="median"))]),features_num),
        ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                         ("oh",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),features_cat),
    ])
    model=ExtraTreesClassifier(
        n_estimators=700,class_weight="balanced",random_state=42,n_jobs=-1,**cfg
    )
    pipe=Pipeline([("pre",pre),("model",model)])
    return pipe


def gate_a_prob(train_pair, test, features_num, features_cat, cfg):
    y=(train_pair["band"]==4).astype(int)
    pipe=prep_a_fit(train_pair,features_num,features_cat,cfg)
    pipe.fit(train_pair[features_num+features_cat],y)
    return pipe.predict_proba(test[features_num+features_cat])[:,1]


def b_frames(df, normal_col):
    X=pd.DataFrame({
        "logMission":np.log1p(num(df["missionAssignmentCount"]).fillna(0).clip(lower=0)),
        "normalShare":num(df[normal_col]).fillna(0).clip(0,1),
    },index=df.index)
    return X


def gate_b_components(train_pair,test,normal_col,weights):
    X=b_frames(train_pair,normal_col); Xe=b_frames(test,normal_col)
    y=(train_pair["band"]==5).astype(int).to_numpy()
    k=min(3,len(train_pair))
    knn=Pipeline([("scale",StandardScaler()),("knn",KNeighborsClassifier(n_neighbors=k,weights=weights))])
    knn.fit(X,y)
    pk=knn.predict_proba(Xe)[:,list(knn.named_steps["knn"].classes_).index(1)]

    yr=Pipeline([("scale",StandardScaler()),("lr",LogisticRegression(class_weight="balanced",C=1.0,max_iter=5000))])
    yy=train_pair[["fyDeclared"]].astype(float)
    yye=test[["fyDeclared"]].astype(float)
    yr.fit(yy,y)
    py=yr.predict_proba(yye)[:,list(yr.named_steps["lr"].classes_).index(1)]
    return pk,py


def inner_oof_a(outer_train, numf, catf, cfg):
    p=pd.Series(np.nan,index=outer_train.index,dtype=float)
    for yr in sorted(outer_train["fyDeclared"].astype(int).unique()):
        tr=outer_train[(outer_train["fyDeclared"].astype(int)!=yr)&outer_train["band"].isin([3,4])]
        te=outer_train[outer_train["fyDeclared"].astype(int)==yr]
        if len(tr) and (tr["band"]==4).nunique()!=1:
            p.loc[te.index]=gate_a_prob(tr,te,numf,catf,cfg)
    return p


def inner_oof_b_components(outer_train, normal_col, weights):
    pk=pd.Series(np.nan,index=outer_train.index,dtype=float)
    py=pd.Series(np.nan,index=outer_train.index,dtype=float)
    for yr in sorted(outer_train["fyDeclared"].astype(int).unique()):
        tr=outer_train[(outer_train["fyDeclared"].astype(int)!=yr)&outer_train["band"].isin([4,5])]
        te=outer_train[outer_train["fyDeclared"].astype(int)==yr]
        if len(tr) and (tr["band"]==5).nunique()!=1:
            a,b=gate_b_components(tr,te,normal_col,weights)
            pk.loc[te.index]=a; py.loc[te.index]=b
    return pk,py


def select_gate_a(outer_train,numf,catf):
    pair=outer_train[outer_train["band"].isin([3,4])]
    y=(pair["band"]==4).astype(int).to_numpy()
    best=None
    for i,cfg in enumerate(A_CONFIGS):
        p=inner_oof_a(outer_train,numf,catf,cfg).loc[pair.index].to_numpy(float)
        ok=np.isfinite(p)
        if ok.sum()<4 or len(np.unique(y[ok]))<2:continue
        met=best_binary_threshold(y[ok],p[ok])
        key=(int(met["pass80"]),met["min_recall"],(met["lower_recall"]+met["upper_recall"])/2)
        if best is None or key>best[0]:
            best=(key,i,cfg,met)
    if best is None:raise RuntimeError("Could not select Gate A inside outer training years.")
    return best[2],best[3],best[1]


def select_gate_b(outer_train,normal_col):
    pair=outer_train[outer_train["band"].isin([4,5])]
    y=(pair["band"]==5).astype(int).to_numpy()
    best=None
    for weights in B_WEIGHT_OPTIONS:
        pk,py=inner_oof_b_components(outer_train,normal_col,weights)
        for alpha in B_ALPHAS:
            p=(alpha*pk+(1-alpha)*py).loc[pair.index].to_numpy(float)
            ok=np.isfinite(p)
            if ok.sum()<4 or len(np.unique(y[ok]))<2:continue
            met=best_binary_threshold(y[ok],p[ok])
            key=(int(met["pass80"]),met["min_recall"],(met["lower_recall"]+met["upper_recall"])/2)
            if best is None or key>best[0]:
                best=(key,weights,alpha,met)
    if best is None:raise RuntimeError("Could not select Gate B inside outer training years.")
    return best[1],best[2],best[3]


def compose(a,b,ta,tb):
    # Ordinal chain: top boundary gets first claim; otherwise lower-high boundary.
    out=np.full(len(a),3,int)
    out[np.asarray(a)>=ta]=4
    out[np.asarray(b)>=tb]=5
    return out


def resolver_features(df,a,b,ta,tb,external):
    z=pd.DataFrame(index=df.index)
    z["gateAProb"]=np.asarray(a,float)
    z["gateBProb"]=np.asarray(b,float)
    z["gateAMargin"]=np.asarray(a,float)-ta
    z["gateBMargin"]=np.asarray(b,float)-tb
    if external:
        for c in EXT_DIMS:z[c]=num(df[c]).fillna(.5).to_numpy(float)
    return z


def resolver_oof(outer_train, a_oof, b_oof, ta, tb, external):
    p=pd.Series(np.nan,index=outer_train.index,dtype=float)
    Xall=resolver_features(outer_train,a_oof.loc[outer_train.index],b_oof.loc[outer_train.index],ta,tb,external)
    yall=(outer_train["band"]==4).astype(int)
    for yr in sorted(outer_train["fyDeclared"].astype(int).unique()):
        tr=outer_train["fyDeclared"].astype(int)!=yr
        te=~tr
        if yall.loc[tr].nunique()<2:continue
        model=Pipeline([("scale",StandardScaler()),("lr",LogisticRegression(class_weight="balanced",C=.5,max_iter=5000))])
        model.fit(Xall.loc[tr],yall.loc[tr])
        p.loc[te]=model.predict_proba(Xall.loc[te])[:,1]
    return p


def choose_rescue_threshold(outer_train, baseline, middle_prob):
    y=outer_train["band"].astype(int).to_numpy()
    b=np.asarray(baseline,int)
    p=np.asarray(middle_prob,float)
    best=None
    for t in [0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90]:
        pred=b.copy()
        rescue=(pred!=4)&np.isfinite(p)&(p>=t)
        pred[rescue]=4
        rec=recall_score(y,pred,labels=PAIR_LABELS,average=None,zero_division=0)
        key=(int(np.all(rec>=.8)),float(rec.min()),float(rec.mean()),-int(rescue.sum()))
        if best is None or key>best[0]:
            best=(key,t,rec,int(rescue.sum()))
    return float(best[1]),[float(x) for x in best[2]],best[3]


def fit_resolver(train,a_oof,b_oof,ta,tb,external):
    X=resolver_features(train,a_oof.loc[train.index],b_oof.loc[train.index],ta,tb,external)
    y=(train["band"]==4).astype(int)
    model=Pipeline([("scale",StandardScaler()),("lr",LogisticRegression(class_weight="balanced",C=.5,max_iter=5000))])
    model.fit(X,y)
    return model


def main():
    d=master_base.load_master()

    # Complete target-free declaration exposure for every master disaster.
    exposure=exposure_v2.build_population_exposure_v2(d)
    d=d.merge(exposure.drop(columns=["state"],errors="ignore"),on="disasterNumber",how="left")

    # Frozen nonfinancial MissionAssignments composition.
    comp=ensure_mission_features(d)
    normal_col=find_normal_priority_share(comp)
    d=d.merge(comp[["disasterNumber",normal_col]],on="disasterNumber",how="left")
    d[normal_col]=num(d[normal_col]).fillna(0)

    # External physical layer receives no target/band/funding-derived variables.
    source=d[["disasterNumber","state","incidentType","effectiveDeclarationDate"]].copy()
    ext=build_external_dimensions(source)
    d=d.merge(ext,on="disasterNumber",how="left")
    for c in EXT_DIMS:d[c]=num(d[c]).fillna(.5).clip(0,1)

    # Explicitly block known funding-derived master features from the gate.
    blocked={"target","band","totalObligatedFunding","eventScale","fundingPerMission","fundingPerAgency","fundingPerDay"}
    numf=[c for c in A_NUM_BASE+A_EXPOSURE if c in d.columns and c not in blocked]
    catf=[c for c in A_CATS if c in d.columns and c not in blocked]

    high=d[d["band"].isin([3,4,5])].copy().reset_index(drop=True)
    years=sorted(high["fyDeclared"].astype(int).unique())

    outer_rows=[]
    for outer_year in years:
        tr=high[high["fyDeclared"].astype(int)!=outer_year].copy()
        te=high[high["fyDeclared"].astype(int)==outer_year].copy()
        if te.empty:continue

        a_cfg,a_sel,a_cfg_idx=select_gate_a(tr,numf,catf)
        b_weights,b_alpha,b_sel=select_gate_b(tr,normal_col)
        ta=float(a_sel["threshold"]); tb=float(b_sel["threshold"])

        # Cross-fitted gate scores for resolver training.
        a_oof=inner_oof_a(tr,numf,catf,a_cfg)
        bk_oof,by_oof=inner_oof_b_components(tr,normal_col,b_weights)
        b_oof=b_alpha*bk_oof+(1-b_alpha)*by_oof

        # Fill any rare inner-year score gaps conservatively with training-pair prevalence.
        a_prev=float((tr["band"]==4).sum()/max(tr["band"].isin([3,4]).sum(),1))
        b_prev=float((tr["band"]==5).sum()/max(tr["band"].isin([4,5]).sum(),1))
        a_oof=a_oof.fillna(a_prev); b_oof=b_oof.fillna(b_prev)

        baseline_train=compose(a_oof.loc[tr.index],b_oof.loc[tr.index],ta,tb)

        score_oof=resolver_oof(tr,a_oof,b_oof,ta,tb,external=False)
        ext_oof=resolver_oof(tr,a_oof,b_oof,ta,tb,external=True)
        score_t,score_inner_rec,score_inner_rescues=choose_rescue_threshold(tr,baseline_train,score_oof.loc[tr.index])
        ext_t,ext_inner_rec,ext_inner_rescues=choose_rescue_threshold(tr,baseline_train,ext_oof.loc[tr.index])

        score_model=fit_resolver(tr,a_oof,b_oof,ta,tb,external=False)
        ext_model=fit_resolver(tr,a_oof,b_oof,ta,tb,external=True)

        # Final pair models fit only on non-held fiscal years.
        a_pair=tr[tr["band"].isin([3,4])]
        a_test=gate_a_prob(a_pair,te,numf,catf,a_cfg)

        b_pair=tr[tr["band"].isin([4,5])]
        bk_test,by_test=gate_b_components(b_pair,te,normal_col,b_weights)
        b_test=b_alpha*bk_test+(1-b_alpha)*by_test

        baseline=compose(a_test,b_test,ta,tb)

        Xs=resolver_features(te,a_test,b_test,ta,tb,external=False)
        Xe=resolver_features(te,a_test,b_test,ta,tb,external=True)
        ps=score_model.predict_proba(Xs)[:,1]
        pe=ext_model.predict_proba(Xe)[:,1]

        score_pred=baseline.copy()
        score_pred[(score_pred!=4)&(ps>=score_t)]=4
        ext_pred=baseline.copy()
        ext_pred[(ext_pred!=4)&(pe>=ext_t)]=4

        for j,(_,r) in enumerate(te.iterrows()):
            outer_rows.append({
                "outer_year":int(outer_year),
                "disasterNumber":int(r["disasterNumber"]),
                "state":str(r["state"]),
                "incidentType":str(r["incidentType"]),
                "target":float(r["target"]),
                "actual_band":int(r["band"]),
                "gateA_prob_middle":float(a_test[j]),
                "gateA_threshold":ta,
                "gateA_predict_highside":int(a_test[j]>=ta),
                "gateA_config_index":int(a_cfg_idx),
                "gateA_inner_lower_recall":float(a_sel["lower_recall"]),
                "gateA_inner_upper_recall":float(a_sel["upper_recall"]),
                "gateB_prob_upper":float(b_test[j]),
                "gateB_threshold":tb,
                "gateB_predict_upper":int(b_test[j]>=tb),
                "gateB_knn_weights":b_weights,
                "gateB_year_blend_alpha_knn":float(b_alpha),
                "gateB_inner_lower_recall":float(b_sel["lower_recall"]),
                "gateB_inner_upper_recall":float(b_sel["upper_recall"]),
                "baseline_prediction":int(baseline[j]),
                "score_only_middle_probability":float(ps[j]),
                "score_only_rescue_threshold":score_t,
                "score_only_prediction":int(score_pred[j]),
                "external_middle_probability":float(pe[j]),
                "external_rescue_threshold":ext_t,
                "external_prediction":int(ext_pred[j]),
                "score_inner_recalls":json.dumps(score_inner_rec),
                "external_inner_recalls":json.dumps(ext_inner_rec),
                "score_inner_rescues":int(score_inner_rescues),
                "external_inner_rescues":int(ext_inner_rescues),
                **{c:float(r[c]) for c in EXT_DIMS},
                "extSourceAudit":str(r["extSourceAudit"]),
            })

    out=pd.DataFrame(outer_rows).sort_values(["outer_year","disasterNumber"])
    out.to_csv(OUT/"nested_outer_leave_year_predictions.csv",index=False)

    y=out["actual_band"].to_numpy(int)
    baseline=out["baseline_prediction"].to_numpy(int)
    score=out["score_only_prediction"].to_numpy(int)
    external=out["external_prediction"].to_numpy(int)

    # Local gate metrics on their intended adjacent pairs.
    ma=out["actual_band"].isin([3,4]).to_numpy()
    ya=(out.loc[ma,"actual_band"]==4).astype(int).to_numpy()
    pa=(out.loc[ma,"gateA_prob_middle"]>=out.loc[ma,"gateA_threshold"]).astype(int).to_numpy()

    mb=out["actual_band"].isin([4,5]).to_numpy()
    yb=(out.loc[mb,"actual_band"]==5).astype(int).to_numpy()
    pb=(out.loc[mb,"gateB_prob_upper"]>=out.loc[mb,"gateB_threshold"]).astype(int).to_numpy()

    middle=out[out["actual_band"]==4].copy()
    middle["gateA_correct_for_middle"]=middle["gateA_predict_highside"]==1
    middle["gateB_correct_for_middle"]=middle["gateB_predict_upper"]==0
    middle["baseline_correct"]=middle["baseline_prediction"]==4
    middle["external_correct"]=middle["external_prediction"]==4
    middle.to_csv(OUT/"nine_middle_band_case_audit.csv",index=False)

    baseline_fail=out[out["baseline_prediction"]!=out["actual_band"]].copy()
    baseline_fail.to_csv(OUT/"baseline_chain_failures.csv",index=False)
    changed=out[out["external_prediction"]!=out["baseline_prediction"]].copy()
    changed.to_csv(OUT/"external_resolver_overrides.csv",index=False)

    summary={
        "objective":"Use separate target-free external physical dimensions only as a middle-band rescue layer over two local high-value gates.",
        "normal_priority_feature":normal_col,
        "gateA_nested_outer":binary_metrics(ya,pa),
        "gateB_nested_outer":binary_metrics(yb,pb),
        "baseline_composed_router":all_metrics(y,baseline),
        "score_only_disagreement_rescue":all_metrics(y,score),
        "external_physical_disagreement_rescue":all_metrics(y,external),
        "middle_band_cases":middle[[
            "disasterNumber","state","fyDeclared" if "fyDeclared" in middle.columns else "outer_year",
            "incidentType","target","gateA_predict_highside","gateB_predict_upper",
            "baseline_prediction","external_prediction","external_middle_probability"
        ]].to_dict("records"),
        "external_override_count":int((external!=baseline).sum()),
        "external_overrides_corrected":int(((external==y)&(baseline!=y)).sum()),
        "external_overrides_harmed":int(((external!=y)&(baseline==y)).sum()),
        "guardrails":[
            "No true funding target or band is used to route any held-year case.",
            "Gate A model/configuration and operating threshold are selected only inside the non-held outer years using inner leave-fiscal-year-out predictions.",
            "Gate B uses the documented 3-nearest-neighbor mission/Normal-priority structure with declaration-year context; weight/blend/threshold are selected only inside the outer training years.",
            "Resolver training uses cross-fitted gate scores for outer-training cases, never in-sample gate scores.",
            "Resolver rescue threshold is selected only from inner leave-year-out resolver predictions inside each outer training set.",
            "The external feature builder receives no target, band, obligation amount, funding-per-resource field, or eventScale.",
            "External source identity and availability are audit fields only and never model predictors.",
            "CDC/NOAA physical dimensions use only observations on or before the FEMA declaration date.",
            "No NOAA property-damage or crop-damage dollar fields are loaded.",
            "Original funding-derived eventScale is explicitly excluded.",
        ],
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str),flush=True)


if __name__=="__main__":
    main()
