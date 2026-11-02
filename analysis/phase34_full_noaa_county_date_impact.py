#!/usr/bin/env python3
"""Phase 34/35: NOAA flood county/date candidate universe, attribution audit,
and explicitly exploratory retrospective forward-year ablation.

No claim of causal FEMA–NOAA disaster linkage or true total economic loss:
county/date matching is a candidate screen. NOAA final damage amounts,
contemporary FEMA designated geographies and finished operations are not T0/T+72
information. Frozen Track A is not touched.
"""
from __future__ import annotations
from pathlib import Path
import json, re, hashlib, time, shutil
import numpy as np
import pandas as pd
import requests
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"audit_outputs"/"phase34_noaa_footprint"
OUT.mkdir(parents=True,exist_ok=True)
PHASE25=ROOT/"audit_inputs"/"phase34"/"phase25"
PHASE33=ROOT/"audit_inputs"/"phase34"/"phase33"
NOAA_ROOT="https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
FEMA_API="https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"
CORE={"Flood","Flash Flood","Coastal Flood","Lakeshore Flood"}
CONTEXT={"Heavy Rain","Storm Surge/Tide"}
session=requests.Session()
session.headers.update({"User-Agent":"AI-assisted-disaster-relief-research/Phase34 academic source audit"})

def get(url,params=None,stream=False):
    for i in range(5):
        try:
            x=session.get(url,params=params,timeout=(25,180),stream=stream)
            x.raise_for_status()
            return x
        except Exception as exc:
            if i==4:raise
            print("RETRY",i,url,str(exc)[:200],flush=True)
            time.sleep(3*(i+1))

def fips(x):
    if pd.isna(x):return None
    z=re.sub(r"\D","",str(x))
    return int(z) if z else None

def money(v):
    if pd.isna(v) or not str(v).strip():return np.nan
    z=re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMB]?)\s*",str(v).upper().replace(",","").replace("$",""))
    return float(z.group(1))*{"":1,"K":1e3,"M":1e6,"B":1e9}[z.group(2)] if z else np.nan

def data_fema(ids):
    columns="disasterNumber,state,incidentType,incidentBeginDate,incidentEndDate,declarationDate,fyDeclared,fipsStateCode,fipsCountyCode,placeCode,designatedArea,tribalRequest,declarationType"
    allrows=[]
    for j in range(0,len(ids),8):
        group=ids[j:j+8]
        query=" or ".join("disasterNumber eq "+str(int(n)) for n in group)
        offset=0
        while True:
            p={"$filter":"("+query+")","$select":columns,"$top":"1000","$skip":str(offset)}
            result=get(FEMA_API,params=p).json()
            rows=result.get("DisasterDeclarationsSummaries")
            if rows is None:raise RuntimeError("OpenFEMA schema/query failure: "+str(result)[:1000])
            allrows+=rows
            offset+=len(rows)
            if len(rows)<1000:break
            if offset>12000:raise RuntimeError("Unexpected FEMA pagination")
        print("FEMA batch",j//8+1,"rows",len(allrows),flush=True)
    df=pd.DataFrame(allrows)
    if df.empty:raise RuntimeError("No FEMA designated area data")
    df=df[df.disasterNumber.isin(ids)].copy()
    df["fips_state"]=pd.to_numeric(df.fipsStateCode,errors="coerce")
    df["fips_county"]=pd.to_numeric(df.fipsCountyCode,errors="coerce")
    df["county_fips"]=df.fips_state*1000+df.fips_county
    df["is_strict_county"]=df.fips_county.between(1,998)&df.fips_state.between(1,99)
    df["incident_start"]=pd.to_datetime(df.incidentBeginDate,utc=True,errors="coerce").dt.tz_localize(None).dt.normalize()
    df["incident_end"]=pd.to_datetime(df.incidentEndDate,utc=True,errors="coerce").dt.tz_localize(None).dt.normalize()
    df["declaration_date"]=pd.to_datetime(df.declarationDate,utc=True,errors="coerce").dt.tz_localize(None).dt.normalize()
    df=df.drop_duplicates(subset=["disasterNumber","county_fips","designatedArea","incidentBeginDate","incidentEndDate"])
    df.to_csv(OUT/"fema_official_designated_areas.csv",index=False)
    return df

def data_noaa(years):
    html=get(NOAA_ROOT).text
    wanted=["EVENT_ID","EPISODE_ID","YEAR","STATE_FIPS","STATE","CZ_TYPE","CZ_FIPS","CZ_NAME","EVENT_TYPE","BEGIN_DATE_TIME","END_DATE_TIME","DAMAGE_PROPERTY","DAMAGE_CROPS","DEATHS_DIRECT","DEATHS_INDIRECT","INJURIES_DIRECT","INJURIES_INDIRECT","CZ_TIMEZONE"]
    frames=[];manifest=[]
    for yr in years:
        pat=r"StormEvents_details-ftp_v1\.0_d"+str(yr)+r"_c\d{8}\.csv\.gz"
        files=sorted(set(re.findall(pat,html)))
        if not files:raise RuntimeError("No NOAA year "+str(yr))
        name=files[-1]
        path=OUT/name
        with get(NOAA_ROOT+name,stream=True) as resp,open(path,"wb") as handle:
            for chunk in resp.iter_content(256*1024):
                if chunk:handle.write(chunk)
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        total=0;elig=0
        for block in pd.read_csv(path,dtype=str,compression="gzip",usecols=lambda k:k in wanted,chunksize=25000):
            total+=len(block)
            use=block[block.EVENT_TYPE.isin(CORE|CONTEXT)].copy()
            if len(use):
                use["archive_year"]=yr
                use["source_file"]=name
                frames.append(use)
                elig+=len(use)
        path.unlink()
        print("NOAA",yr,"rows",total,"hazard-scope",elig,flush=True)
        manifest.append({"year":yr,"source":NOAA_ROOT+name,"filename":name,"sha256":sha,"total_rows":total,"flood_context_rows":elig})
    n=pd.concat(frames,ignore_index=True)
    n["EVENT_ID"]=pd.to_numeric(n.EVENT_ID,errors="coerce")
    assert n.EVENT_ID.notna().all()
    n["county_fips"]=pd.to_numeric(n.STATE_FIPS,errors="coerce")*1000+pd.to_numeric(n.CZ_FIPS,errors="coerce")
    n["begin"]=pd.to_datetime(n.BEGIN_DATE_TIME,errors="coerce",format="mixed").dt.normalize()
    n["end"]=pd.to_datetime(n.END_DATE_TIME,errors="coerce",format="mixed").dt.normalize()
    n["property_usd"]=n.DAMAGE_PROPERTY.map(money)
    n["crop_usd"]=n.DAMAGE_CROPS.map(money)
    n["damage_usd"]=n[["property_usd","crop_usd"]].sum(axis=1,min_count=1)
    n["core_hazard"]=n.EVENT_TYPE.isin(CORE)
    n["is_county"]=n.CZ_TYPE.eq("C")
    n=n.drop_duplicates("EVENT_ID")
    n.to_csv(OUT/"all_hazard_scope_noaa_events.csv",index=False)
    pd.DataFrame(manifest).to_csv(OUT/"noaa_release_manifest.csv",index=False)
    return n,manifest

def join(fema,noaa,ids,anchors):
    # NOAA county records and FEMA designated-area county keys are not spatial footprints.
    r=fema[fema.is_strict_county & fema.incident_start.notna()].copy()
    r["incident_end_imputed"]=r.incident_end.isna()
    r["effective_incident_end"]=r.incident_end
    # Missing incident end: do not assume indefinite event duration.
    r=r[r.effective_incident_end.notna()].copy()
    r=r[r.effective_incident_end.ge(r.incident_start)]
    r=r.drop_duplicates(["disasterNumber","county_fips","incident_start","effective_incident_end"])
    n=noaa[noaa.is_county&noaa.begin.notna()&noaa.end.notna()&noaa.county_fips.notna()]
    m=r.merge(n,on="county_fips",how="left",suffixes=("_fema","_noaa"))
    m["overlap"]=m.begin.le(m.effective_incident_end)&m.end.ge(m.incident_start)
    m["overlap_one_day"]=m.begin.le(m.effective_incident_end+pd.Timedelta(days=1))&m.end.ge(m.incident_start-pd.Timedelta(days=1))
    m=m[m.overlap_one_day & m.EVENT_ID.notna()].copy()
    m["core_hazard"]=m.EVENT_TYPE.isin(CORE)
    m["strict_core_overlap"]=m.overlap & m.core_hazard
    m["context_only"]=~m.core_hazard
    m["temporal_edge_only"]=~m.overlap
    # If one NOAA event qualifies for several FEMA declarations, flag ambiguity.
    dup=(m[m.strict_core_overlap].groupby("EVENT_ID").disasterNumber.nunique().rename("n_fema_disasters_sharing_event"))
    m=m.join(dup,on="EVENT_ID")
    m["n_fema_disasters_sharing_event"]=m.n_fema_disasters_sharing_event.fillna(0).astype(int)
    m["ambiguous_event_attribution"]=m.n_fema_disasters_sharing_event.gt(1)
    m["matched_flood_footprint_certified"]=False
    m["available_asof_declaration"]=False
    m["available_asof_t72"]=False
    # Drop identical qualifying NOAA report repeated for multiple FEMA area lines.
    m=m.drop_duplicates(["disasterNumber","EVENT_ID"])
    m.to_csv(OUT/"noaa_candidate_events_by_fema_county_date.csv",index=False)
    a=m[(m.strict_core_overlap)&~m.ambiguous_event_attribution].copy()
    sums=[]
    for disaster in ids:
        src=r[r.disasterNumber==disaster]
        v=m[(m.disasterNumber==disaster)&m.strict_core_overlap]
        unique=a[a.disasterNumber==disaster]
        vgood=unique[unique.damage_usd.notna()]
        sums.append({
            "disasterNumber":int(disaster),
            "eligible_FEMA_county_areas":int(src.county_fips.nunique()),
            "matched_FEMA_counties":int(v.county_fips.nunique()),
            "candidate_noaa_core_count":int(len(v)),
            "candidate_count_with_damage_number":int(v.damage_usd.notna().sum()),
            "candidate_context_count":int(m[(m.disasterNumber==disaster)&m.context_only].shape[0]),
            "candidate_date_edge_count":int(m[(m.disasterNumber==disaster)&m.temporal_edge_only].shape[0]),
            "ambiguous_shared_core_events":int(v.ambiguous_event_attribution.sum()),
            "unambiguous_noaa_core_count":int(len(unique)),
            "unambiguous_nonmissing_damage_reports":int(len(vgood)),
            "unambiguous_positive_damage_reports":int(vgood.damage_usd.gt(0).sum()),
            "unambiguous_reported_property_usd":float(vgood.property_usd.sum(min_count=1)) if len(vgood) else np.nan,
            "unambiguous_reported_crop_usd":float(vgood.crop_usd.sum(min_count=1)) if len(vgood) else np.nan,
            "unambiguous_reported_sum_usd":float(vgood.damage_usd.sum(min_count=1)) if len(vgood) else np.nan,
            "unambiguous_reported_max_usd":float(vgood.damage_usd.max()) if len(vgood) else np.nan,
            "sparse_or_unverifiable":len(src)==0 or len(v)==0,
            "candidate_not_complete_damage":True,
            "not_early_response_safe":True})
    cases=pd.DataFrame(sums)
    assert cases.disasterNumber.nunique()==len(ids)
    cases.to_csv(OUT/"fema_disaster_candidate_impact_features.csv",index=False)
    # Reconcile original sample IDs and date conditions.
    matches=set(zip(m.disasterNumber.astype(int),m.EVENT_ID.astype(int)))
    anchors["in_expanded_county_date_candidate"]= [ (int(x.disasterNumber),int(x.noaa_event_id)) in matches for _,x in anchors.iterrows()]
    anchors.to_csv(OUT/"phase33_anchor_reconciliation.csv",index=False)
    return cases,m,anchors

def model_ablation(cases):
    """Matched FY2021-2024, original Phase25 model definition. Exploratory only."""
    from sklearn.metrics import r2_score
    import flood_forward_model_ablation as ablation
    ablation.IN=PHASE25/"model_inputs"
    ablation.OUT=OUT/"ablation"
    ablation.OUT.mkdir(exist_ok=True)
    d, features, available=ablation.prepare()
    assert d.disasterNumber.nunique()==92
    ff=cases.copy()
    ff["noaa_has_coverage"]=ff.unambiguous_noaa_core_count.gt(0).astype(int)
    ff["noaa_reported_damage_available"]=ff.unambiguous_nonmissing_damage_reports.gt(0).astype(int)
    for key,source in [
        ("noaa_log1p_sum_usd","unambiguous_reported_sum_usd"),
        ("noaa_log1p_max_usd","unambiguous_reported_max_usd"),
        ("noaa_log1p_report_count","unambiguous_noaa_core_count"),
        ("noaa_log1p_reported_damage_count","unambiguous_nonmissing_damage_reports"),
        ("noaa_log1p_eligible_counties","eligible_FEMA_county_areas"),
        ("noaa_log1p_counties_matched","matched_FEMA_counties")]:
        ff[key]=np.log1p(pd.to_numeric(ff[source],errors="coerce").clip(lower=0))
    new_features=["noaa_log1p_sum_usd","noaa_log1p_max_usd","noaa_log1p_report_count",
      "noaa_log1p_reported_damage_count","noaa_log1p_eligible_counties",
      "noaa_log1p_counties_matched","noaa_has_coverage","noaa_reported_damage_available"]
    d=d.merge(ff[["disasterNumber"]+new_features],on="disasterNumber",how="left",validate="one_to_one")
    assert len(d)==92
    arms={
      "fema_only":features["fema_only"],
      "fema_multi_rain":features["fema_multi_rain"],
      "fema_noaa_damage_candidates":features["fema_only"]+new_features,
      "fema_multi_rain_noaa_damage_candidates":features["fema_multi_rain"]+new_features}
    predictions=[];folds=[]
    for fy in ablation.YEARS:
        train=d[d.fyDeclared<fy].copy();test=d[d.fyDeclared==fy].copy()
        assert len(train)>0 and len(test)>0 and train.fyDeclared.max()<fy
        folds.append({"test_fy":int(fy),"n_train":len(train),"n_test":len(test)})
        for arm,cols in arms.items():
            pred,classification=ablation.fit_fold(train,test,cols,fy)
            for (_,case),p,b in zip(test.iterrows(),pred,classification):
                predictions.append({"disasterNumber":int(case.disasterNumber),"fyDeclared":int(fy),
                  "arm":arm,"target":float(case.target),"pred_amount":float(p),
                  "true_band":str(case.band),"pred_band":str(b)})
        print("TEST FY",fy,"trained",len(train),"tested",len(test),flush=True)
    p=pd.DataFrame(predictions)
    p.to_csv(OUT/"ablation"/"exploratory_forward_year_predictions.csv",index=False)
    groups=[]
    for name,g in p.groupby("arm"):
        error=g.pred_amount-g.target
        groups.append({"arm":name,"n":len(g),"MAE_USD":float(error.abs().mean()),
           "RMSE_USD":float(np.sqrt(np.mean(error**2))),
           "R2":float(r2_score(g.target,g.pred_amount)),
           "six_band_accuracy":float((g.pred_band==g.true_band).mean())})
    metrics=pd.DataFrame(groups)
    metrics.to_csv(OUT/"ablation"/"exploratory_model_comparison.csv",index=False)
    pd.DataFrame(folds).to_csv(OUT/"ablation"/"forward_fold_provenance.csv",index=False)
    return metrics,p,folds

def main():
    master=pd.read_excel(ROOT/"master_openfema_40plus.xlsx")
    assert len(master)==971
    flooded=master[(master.incidentType=="Flood") & master.fyDeclared.between(2019,2024)]
    assert len(flooded)==92
    points=pd.read_csv(PHASE25/"all_248_point_duration_adaptive.csv")
    anchors=pd.read_csv(PHASE33/"candidate_event_match_audit.csv")
    assert points.disasterNumber.nunique()==87 and len(points)==248
    ids=sorted(map(int,flooded.disasterNumber.unique()))
    fema=data_fema(ids)
    noaa,source=data_noaa(range(2017,2025))
    cases,m,anchor=join(fema,noaa,ids,anchors)
    metrics,p,folds=model_ablation(cases)
    sums={
      "fema_2019_2024_floods":len(ids),
      "fema_cases_with_api_rows":int(fema.disasterNumber.nunique()),
      "fema_cases_with_eligible_counties":int(cases.eligible_FEMA_county_areas.gt(0).sum()),
      "fema_cases_with_noaa_strict_core":int(cases.candidate_noaa_core_count.gt(0).sum()),
      "fema_cases_with_unambiguous_core":int(cases.unambiguous_noaa_core_count.gt(0).sum()),
      "fema_cases_with_nonmissing_unambiguous_damage":int(cases.unambiguous_nonmissing_damage_reports.gt(0).sum()),
      "fema_cases_with_positive_unambiguous_damage":int(cases.unambiguous_positive_damage_reports.gt(0).sum()),
      "total_unique_fema_noaa_strict_core_links":int(m.strict_core_overlap.sum()),
      "ambiguous_fema_noaa_strict_core_links":int((m.strict_core_overlap&m.ambiguous_event_attribution).sum()),
      "phase33_anchor_present_in_expanded_county_date":int(anchor.in_expanded_county_date_candidate.sum()),
      "phase33_anchor_total":len(anchor),
      "recomputed_holdout_case_count":int(p.disasterNumber.nunique()),
      "model_results":metrics.to_dict(orient="records"),
      "source_manifest":source,
      "limitations":[
        "FEMA current designated areas are not an observed hazard footprint and may reflect later administrative designations.",
        "Same-county same-day storm event is not proof of causation; overlapping unrelated flood events can be included.",
        "NOAA final reported losses are neither comprehensive insured/total economic losses nor availability-certified at T0/T72.",
        "Unrelated/overlapping FEMA disasters are flagged using EVENT_ID reuse; other attribution ambiguity remains.",
        "No independent test: FY2021-2024 outcomes have been examined during earlier iterative experiments.",
        "The original accepted 912-case frozen Track A is not modified or directly compared with 59-case flood-only metrics."]}
    (OUT/"phase34_summary.json").write_text(json.dumps(sums,indent=2))
    report=("# Phase 34 — NOAA event universe for FEMA flood designated areas\n\n"
      "## Status\n\nOfficial FEMA designated areas + incident dates joined to all NOAA core flood/flash-flood/coastal/lakeshore event reports at county and date. The results are **candidate co-occurrences**, NOT certified FEMA flood footprints or total economic losses. Retrospective-only.\n\n"
      f"Original FEMA floods: {len(ids)}; FEMA API matches: {int(fema.disasterNumber.nunique())}; "
      f"cases with strict county+date+flood candidates: {int(cases.candidate_noaa_core_count.gt(0).sum())}; "
      f"cases with nonmissing unambiguous damage reports: {int(cases.unambiguous_nonmissing_damage_reports.gt(0).sum())}.\n\n"
      f"Candidate strict links: {int(m.strict_core_overlap.sum())}; shared-across-FEMA ambiguous links: "
      f"{int((m.strict_core_overlap&m.ambiguous_event_attribution).sum())}.\n\n"
      "## Exploratory same-cohort chronological model comparison\n\n"
      +metrics.to_markdown(index=False)+"\n\n"
      "## Caveats\n\nFinal NOAA estimates have no confirmed declaration-time version; FEMA source designations may be recorded after declaration; geographic and temporal co-occurrence is insufficient to establish causal attribution. These are source-integrity findings and a repeatedly inspected retrospective 59-case sample. Preserve frozen Track A.\n")
    (OUT/"REPORT.md").write_text(report)
    print("FINAL_PHASE34_SUMMARY")
    print(json.dumps({k:v for k,v in sums.items() if k!="source_manifest"},indent=2),flush=True)

if __name__=="__main__":
    main()
