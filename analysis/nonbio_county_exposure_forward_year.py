#!/usr/bin/env python3
"""Research-only FORWARD-YEAR county exposure x hazard-aware analogue.

No edits to frozen router or accepted specialists. All match steps ignore funding.
County populations indicate residents of counties reporting NOAA events, NOT
individually harmed persons. Census estimates are retrospectively revised;
NOAA records are finalized post-event. Not t0/declaration-time safe.
"""
from __future__ import annotations
from pathlib import Path
from io import BytesIO
import gzip, json, re, time, hashlib
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import RobustScaler
import nonbio_50_200m_fast_hybrid as original

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"audit_inputs"/"county_exposure_hazard"/"external"
FROZEN=ROOT/"audit_inputs"/"county_exposure_hazard"/"frozen"/"candidate_predictions.csv"
CACHE=ROOT/".cache"/"county_exposure_hazard"
OUT=ROOT/"audit_outputs"/"county_exposure_hazard"
CACHE.mkdir(parents=True,exist_ok=True)
NOAA="https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
CENSUS={
 "vintage2019":"https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv",
 "vintage2024":"https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/totals/co-est2024-alldata.csv",
}
SESSION=requests.Session()
SESSION.headers["User-Agent"]="Academic-research-exposure-audit/1.0"
K=5
ALPHA=.75
LO,HI=original.LO,original.HI
WEIGHTS={"original":0.0,"hazard_0p25":.25,"hazard_0p5":.5}
OP=[
 "durationDays","declarationDelayDays","expectedResourceScore","missionAssignmentCount",
 "uniqueAgencyCount","uniqueMaTypeCount","uniquePriorityCount",
 "responseComplexityScore","missionDensity","agencyDensity"
]
# Hazard groups are only compared when both events belong to same family.
HAZARD={
 "tropical":["nhc_storm_max_wind_kt","nhc_storm_min_pressure_mb","nhc_wind_distance_index",
             "noaa_unique_counties","pop_reported_counties_log","noaa_reported_counties_n"],
 "hydrometeorological":["noaa_event_count","noaa_episode_count","noaa_unique_counties",
             "pop_reported_counties_log","noaa_reported_counties_n"],
 "wildfire":["calfire_acres_log1p","calfire_fire_count",
             "pop_reported_counties_log","noaa_reported_counties_n"],
}
def family(hazard:str)->str:
    h=str(hazard).lower()
    if h in {"hurricane","tropical storm","typhoon"}:return "tropical"
    if h in {"flood","severe storm","tornado","coastal storm","winter storm","snowstorm"}:return "hydrometeorological"
    if h in {"fire","wildfire"}:return "wildfire"
    return "unsupported"

def download(url,path):
    if path.exists() and path.stat().st_size>200:return path
    for attempt in range(4):
        try:
            r=SESSION.get(url,timeout=(15,135))
            r.raise_for_status()
            path.write_bytes(r.content)
            return path
        except Exception as e:
            print("Retry",url,attempt,str(e)[:140],flush=True)
            time.sleep(2**attempt)
    raise RuntimeError("Download unavailable: "+url)

def noaa_county_map(audit):
    wanted=set()
    for raw in audit.noaa_event_ids.fillna(""):
        wanted.update(int(x) for x in str(raw).split("|") if x.isdigit())
    index=download(NOAA,CACHE/"index.html").read_text(errors="replace")
    files={}
    for year in range(2009,2025):
        names=sorted(set(re.findall(rf'StormEvents_details-ftp_v1\.0_d{year}_c\d+\.csv\.gz',index)))
        if names:files[year]=names[-1]
    print(f"NOAA selected event IDs={len(wanted)}, available years={sorted(files)}",flush=True)
    selected=[]
    year_stats=[]
    for year,filename in files.items():
        path=download(NOAA+filename,CACHE/filename)
        cols=["EVENT_ID","STATE_FIPS","CZ_TYPE","CZ_FIPS"]
        data=pd.read_csv(path,compression="gzip",usecols=lambda c:c in cols,low_memory=False)
        ids=pd.to_numeric(data["EVENT_ID"],errors="coerce")
        matched=data.loc[ids.isin(wanted)].copy()
        matched["EVENT_ID"]=pd.to_numeric(matched.EVENT_ID,errors="coerce").astype("Int64")
        if len(matched):
            matched["state_fips"]=pd.to_numeric(matched.STATE_FIPS,errors="coerce")
            matched["county_fips"]=pd.to_numeric(matched.CZ_FIPS,errors="coerce")
            matched=matched[(matched.CZ_TYPE.astype(str).str.upper()=="C")&
                            (matched.state_fips.between(1,79))&
                            (matched.county_fips.between(1,999))].copy()
            matched["fips"]=matched.state_fips.astype(int).astype(str).str.zfill(2)+matched.county_fips.astype(int).astype(str).str.zfill(3)
            selected.append(matched[["EVENT_ID","fips"]])
        year_stats.append({"year":year,"matched_noaa_events":int(ids.isin(wanted).sum()),"file":filename})
        print("NOAA",year,"valid matched counties",len(matched),flush=True)
    if not selected:raise RuntimeError("No NOAA counties matched any existing event ID")
    return pd.concat(selected,ignore_index=True).drop_duplicates(),year_stats

def census_population():
    rows=[]
    for vintage,url in CENSUS.items():
        path=download(url,CACHE/Path(url).name)
        df=pd.read_csv(path,encoding="latin-1",low_memory=False)
        print(vintage,"headers",df.columns.tolist()[:20],flush=True)
        required={"STATE","COUNTY"}
        assert required<=set(df.columns)
        d=df.copy()
        d["state_code"]=pd.to_numeric(d.STATE,errors="coerce")
        d["county_code"]=pd.to_numeric(d.COUNTY,errors="coerce")
        d=d[d.state_code.between(1,79)&d.county_code.between(1,999)].copy()
        d["fips"]=d.state_code.astype(int).astype(str).str.zfill(2)+d.county_code.astype(int).astype(str).str.zfill(3)
        for year in range(2010,2025):
            if vintage=="vintage2019" and year>2019:continue
            if vintage=="vintage2024" and year<2020:continue
            col=f"POPESTIMATE{year}"
            if col not in d.columns:continue
            tmp=d[["fips",col]].rename(columns={col:"population"})
            tmp["population"]=pd.to_numeric(tmp.population,errors="coerce")
            tmp["year"]=year
            tmp=tmp.dropna(subset=["population"])
            rows.append(tmp)
    pop=pd.concat(rows,ignore_index=True)
    assert not pop.duplicated(["fips","year"]).any()
    return pop

def attach_exposure(df,audit,map_events,pop):
    a=audit[["disasterNumber","noaa_event_ids"]].copy()
    a["disasterNumber"]=pd.to_numeric(a.disasterNumber).astype(int)
    a["EVENT_ID"]=a.noaa_event_ids.fillna("").astype(str).str.split("|")
    a=a.explode("EVENT_ID")
    a["EVENT_ID"]=pd.to_numeric(a.EVENT_ID,errors="coerce").astype("Int64")
    a=a.dropna(subset=["EVENT_ID"])
    counties=a.merge(map_events,on="EVENT_ID",how="inner")[["disasterNumber","fips"]].drop_duplicates()
    fy=df[["disasterNumber","incidentBeginDate"]].copy()
    fy["year"]=pd.to_datetime(fy.incidentBeginDate,errors="coerce",utc=True).dt.year.fillna(2010).clip(2010,2024).astype(int)
    counties=counties.merge(fy[["disasterNumber","year"]],on="disasterNumber",validate="many_to_one")
    counties=counties.merge(pop,on=["fips","year"],how="left",validate="many_to_one")
    summary=counties.groupby("disasterNumber").agg(
      pop_reported_counties=("population",lambda x:x.sum(min_count=1)),
      noaa_reported_counties_n=("fips","nunique"),
      noaa_reported_counties_with_census=("population","count"),
    ).reset_index()
    full=df.merge(summary,on="disasterNumber",how="left",validate="one_to_one")
    full["pop_reported_counties_log"]=np.log1p(full.pop_reported_counties)
    return full,counties

def opdistance(a,b):
    p=a[OP].apply(pd.to_numeric,errors="coerce")
    q=b[OP].apply(pd.to_numeric,errors="coerce")
    med=p.median().fillna(0)
    A=p.fillna(med).to_numpy(float)
    B=q.fillna(med).to_numpy(float)
    t=RobustScaler(quantile_range=(10,90)).fit(A)
    A=np.clip(t.transform(A),-8,8)
    B=np.clip(t.transform(B),-8,8)
    result=np.mean((B[:,None,:]-A[None,:,:])**2,axis=2)
    result+=2*(b.incidentType.to_numpy()[:,None]!=a.incidentType.to_numpy()[None,:])
    result+=.3*(b.state.to_numpy()[:,None]!=a.state.to_numpy()[None,:])
    return result

def physicaldistance(a,b):
    out=np.zeros((len(b),len(a)),float)
    availability=np.zeros_like(out,dtype=bool)
    fam_a=np.array([family(x) for x in a.incidentType])
    fam_b=np.array([family(x) for x in b.incidentType])
    for family_name,cols in HAZARD.items():
        pa=a[cols].apply(pd.to_numeric,errors="coerce")
        pb=b[cols].apply(pd.to_numeric,errors="coerce")
        good_a=pa.notna().to_numpy()
        good_b=pb.notna().to_numpy()
        # Robust scales and medians learnt from outer-training pool alone.
        med=pa.median().fillna(0)
        A=pa.fillna(med).to_numpy(float);B=pb.fillna(med).to_numpy(float)
        scaler=RobustScaler(quantile_range=(10,90)).fit(A)
        A=np.clip(scaler.transform(A),-8,8)
        B=np.clip(scaler.transform(B),-8,8)
        both=good_b[:,None,:]&good_a[None,:,:]
        n=both.sum(axis=2)
        dist=np.divide((((B[:,None,:]-A[None,:,:])**2)*both).sum(axis=2),n,
           out=np.zeros_like(out),where=n>0)
        eligible=(fam_b[:,None]==family_name)&(fam_a[None,:]==family_name)&(n>0)
        out+=dist*eligible
        availability|=eligible
    return out,availability

def predict(train,test,weight):
    pool=train[(train.target_m>=10)&(train.target_m<500)]
    assert len(pool)>K
    dist=opdistance(pool,test)
    if weight>0:
        physical,available=physicaldistance(pool,test)
        dist+=weight*physical
    else:
        available=np.zeros_like(dist,dtype=bool)
    y=pool.target_m.to_numpy(float)
    predictions=[]
    valid_pair=[]
    for j,d in enumerate(dist):
        neighbors=np.argsort(d,kind="stable")[:K]
        w=1/(.2+np.sqrt(d[neighbors]))
        predictions.append(float(np.clip(np.average(y[neighbors],weights=w),LO,HI-1e-6)))
        valid_pair.append(int(available[j,neighbors].sum()))
    return np.asarray(predictions),valid_pair

def metrics(data):
    y=data.actual_m.to_numpy(float);p=data.estimate_m.to_numpy(float)
    err=abs(y-p)
    return {"n":len(data),"MAE_M":float(err.mean()),"RMSE_M":float(np.sqrt(((y-p)**2).mean())),
      "R2":float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)) if len(y)>1 and np.var(y)>0 else None,
      "within30":float(np.mean(err/np.maximum(y,1e-9)<=.3)),
      "median_MAE_M":float(abs(y-data.median_m.to_numpy(float)).mean()),
      "cases_with_relevant_analogue_pairs":int((data.valid_pairs>0).sum())}

def run():
    d=original.load()
    audit=pd.read_csv(SRC/"external_match_audit.csv")
    ext=pd.read_csv(SRC/"external_features_971.csv")
    frozen=pd.read_csv(FROZEN)
    assert len(d)==912 and len(audit)==971 and len(ext)==971
    assert (frozen.actual_band==frozen.candidate_pred_new).sum()==830
    matched,year_stats=noaa_county_map(audit)
    census=census_population()
    allmaster=pd.read_excel(ROOT/"master_openfema_40plus.xlsx")
    enriched,counties=attach_exposure(allmaster,audit,matched,census)
    d=d.merge(enriched[["disasterNumber","pop_reported_counties",
       "noaa_reported_counties_n","noaa_reported_counties_with_census",
       "pop_reported_counties_log"]],on="disasterNumber",how="left",validate="one_to_one")
    d=d.merge(ext,on="disasterNumber",how="left",validate="one_to_one")
    for col in set(sum(HAZARD.values(),[]))-set(d.columns):
        raise KeyError(f"Unrecognized physical indicator {col}")
    ids=set(frozen.loc[frozen.candidate_pred_new=="50-200M","disasterNumber"].astype(int))
    assert len(ids)==13
    d["routed"]=d.disasterNumber.astype(int).isin(ids)
    results=[]
    for year in sorted(d.fyDeclared.unique()):
        # ONLY earlier fiscal years. Earlier-year training may still include later-finalized
        # feature versions: chronological holdout != as-of source availability.
        train=d[d.fyDeclared<year]
        support=train.target_m.between(10,500,inclusive="left").sum()
        if support < 6 or train.target_m.between(LO,HI,inclusive="left").sum() < 3:
            print("SKIP year",year,"historical reference pool too small",flush=True)
            continue
        cases=[("direct",d[(d.fyDeclared==year)&d.target_m.between(LO,HI,inclusive="left")]),
          ("routed",d[(d.fyDeclared==year)&d.routed])]
        med=float(train.loc[train.target_m.between(LO,HI,inclusive="left"),"target_m"].median())
        for scope,te in cases:
            if te.empty:continue
            masked=te.copy(); masked["target_m"]=np.nan
            for name,weight in WEIGHTS.items():
                estimates,valid_pairs=predict(train,masked,weight)
                estimates=np.clip(med+ALPHA*(estimates-med),LO,HI-1e-6)
                for (_,rec),pred,pair_n in zip(te.iterrows(),estimates,valid_pairs):
                    results.append({"scope":scope,"variant":name,"disasterNumber":int(rec.disasterNumber),
                        "fy":int(year),"incidentType":rec.incidentType,"actual_m":float(rec.target_m),
                        "estimate_m":float(pred),"median_m":med,"valid_pairs":pair_n,
                        "exposed_county_pop":rec.pop_reported_counties,
                        "reported_counties":rec.noaa_reported_counties_n})
    out=pd.DataFrame(results)
    scores={}
    for scope,g in out.groupby("scope"):
        scores[scope]={}
        for variant,g2 in g.groupby("variant"):
            assert len(g2)>0,(scope,variant,len(g2))
            scores[scope][variant]=metrics(g2)
    report={
        "dataset":len(d),"router_accuracy":"830/912 frozen",
        "validation":"forward_fiscal_year_only_earlier_year_training; no retrospective year used to train an earlier test year",
        "asof_source_validated":False,
        "source_type":"Retrospective NOAAs reported counties + county-level Census population estimates",
        "population_is":"Total residents of distinct counties with matched NOAA county-type events; NOT disaster-affected population",
        "availability":"Final NOAA event records and retrospectively revised Census files: NOT prospective/declaration-time",
        "census_sources":CENSUS,
        "noaa_years":year_stats,
        "matched_NOAA_event_ids_to_counties":int(matched.EVENT_ID.nunique()),
        "matched_disasters_with_county_population":int(d.pop_reported_counties.notna().sum()),
        "direct_with_county_population":int(d.loc[d.target_m.between(LO,HI,inclusive="left"),"pop_reported_counties"].notna().sum()),
        "routed_with_county_population":int(d.loc[d.routed,"pop_reported_counties"].notna().sum()),
        "weights_fixed_no_heldout_selection":WEIGHTS,
        "metrics":scores,
        "risk":"Potential event match errors; broad county overexposure; repeatedly inspected historical high-band cases."
    }
    OUT.mkdir(parents=True,exist_ok=True)
    out.to_csv(OUT/"direct_and_routed_13_prediction_ablation.csv",index=False)
    counties.to_csv(OUT/"event_county_census_matches.csv",index=False)
    enriched[["disasterNumber","state","incidentType","pop_reported_counties",
       "noaa_reported_counties_n","noaa_reported_counties_with_census",
       "pop_reported_counties_log"]].to_csv(OUT/"exposure_features_971.csv",index=False)
    (OUT/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("BEGIN_FORWARD_EXPOSURE_HAZARD_RESULTS",flush=True)
    print(json.dumps(report,indent=2),flush=True)
    print("END_FORWARD_EXPOSURE_HAZARD_RESULTS",flush=True)

if __name__=="__main__":run()
