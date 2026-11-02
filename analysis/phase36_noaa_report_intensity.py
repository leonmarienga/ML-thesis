#!/usr/bin/env python3
"""Phase36: diagnostic ablation of NOAA report intensity vs hazard footprint.

Retrospective 92 FEMA Flood declarations FY2019-2024, held-out FY2021-2024
(59 test cases). The original six-band Track A is NEVER modified.
All model variants reuse the Phase25 identical ExtraTrees parameterization and
strict earlier-fiscal-year target training. Note: NOAA and FEMA metadata may
have been revised AFTER an event; this is NOT T0/T+72 prediction.
"""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"analysis"))
import flood_forward_model_ablation as base
IN=ROOT/"audit_inputs"/"phase36"
OUT=ROOT/"audit_outputs"/"phase36"
OUT.mkdir(parents=True,exist_ok=True)
base.IN=IN/"phase25"/"model_inputs"
base.OUT=OUT/"base_comparison"
base.OUT.mkdir(parents=True,exist_ok=True)

def logplus(v):
    return np.log1p(pd.to_numeric(v,errors="coerce").clip(lower=0))

def historical_reporting(fema,noaa,df):
    # Year-specific prior 730-day flood-report frequency in FEMA-designated counties.
    # Reporting proxy ALSO reflects flood frequency; never call this an impartial
    # measure of reporting quality. No funding outcome enters this construction.
    county=fema[fema.is_strict_county].copy()
    county["county_fips"]=pd.to_numeric(county.county_fips,errors="coerce")
    core=noaa[(noaa.core_hazard==True)&(noaa.is_county==True)].copy()
    core["begin"]=pd.to_datetime(core.begin,errors="coerce")
    core["county_fips"]=pd.to_numeric(core.county_fips,errors="coerce")
    core=core.drop_duplicates("EVENT_ID")
    result=[]
    for _,r in df.iterrows():
        id=int(r.disasterNumber)
        sub=county[county.disasterNumber==id]
        counties=set(sub.county_fips.dropna().astype(int))
        start=pd.to_datetime(sub.incident_start,errors="coerce").min()
        prior=core[(core.county_fips.isin(counties))&(core.begin<start)&(core.begin>=start-pd.Timedelta(days=730))]
        result.append({"disasterNumber":id,
             "eligible_counties_seen":len(counties),
             "lookback_count":len(prior),
             "lookback_covered_counties":prior.county_fips.nunique(),
             "lookback_per_county":len(prior)/max(len(counties),1),
             "lookback_coverage_fraction":prior.county_fips.nunique()/max(len(counties),1),
             "lookback_start":(start-pd.Timedelta(days=730)).isoformat() if pd.notna(start) else "",
             "incident_start_date":start.isoformat() if pd.notna(start) else ""})
    return pd.DataFrame(result)

def read_arms():
    d,features,_=base.prepare()
    assert len(d)==92
    imp=pd.read_csv(IN/"phase34"/"fema_disaster_candidate_impact_features.csv")
    fema=pd.read_csv(IN/"phase34"/"fema_official_designated_areas.csv")
    noaa=pd.read_csv(IN/"phase34"/"all_hazard_scope_noaa_events.csv",low_memory=False)
    hist=historical_reporting(fema,noaa,d)
    assert len(hist)==92
    imp=imp.merge(hist,on="disasterNumber",how="left",validate="one_to_one")
    imp.to_csv(OUT/"phase36_intensity_features_92_cases.csv",index=False)
    # Reproduce exact raw geography/counts ordering in Phase35.
    imp["noaa_has_coverage"]=imp.unambiguous_noaa_core_count.gt(0).astype(int)
    imp["noaa_reported_damage_available"]=imp.unambiguous_nonmissing_damage_reports.gt(0).astype(int)
    names={
     "noaa_log1p_report_count":"unambiguous_noaa_core_count",
     "noaa_log1p_reported_damage_count":"unambiguous_nonmissing_damage_reports",
     "noaa_log1p_eligible_counties":"eligible_FEMA_county_areas",
     "noaa_log1p_counties_matched":"matched_FEMA_counties"}
    for k,v in names.items():imp[k]=logplus(imp[v])
    # Normalize to prevent model attributing bigger administrative scope to
    # more severe flooding without also seeing per-county report density.
    den=imp.eligible_FEMA_county_areas.clip(lower=1)
    matched=imp.matched_FEMA_counties.clip(lower=1)
    imp["report_count_per_eligible_county"]=logplus(imp.unambiguous_noaa_core_count/den)
    imp["report_count_per_matched_county"]=logplus(imp.unambiguous_noaa_core_count/matched)
    imp["reporting_county_fraction"]=(imp.matched_FEMA_counties/den).clip(0,1)
    imp["reported_damage_fraction"]=(imp.unambiguous_nonmissing_damage_reports/imp.unambiguous_noaa_core_count.replace(0,np.nan)).fillna(0).clip(0,1)
    imp["lookback_event_count"]=logplus(imp.lookback_count)
    imp["lookback_per_county_log"]=logplus(imp.lookback_per_county)
    imp["lookback_covered_counties_frac"]=imp.lookback_coverage_fraction.clip(0,1)
    imp["current_to_past_reporting_ratio"]=logplus((imp.unambiguous_noaa_core_count/den)/
                                          (1+(imp.lookback_count/den)))
    imp["report_excess_relative_prior"]=logplus((imp.unambiguous_noaa_core_count/den)-(imp.lookback_count/den).clip(lower=0))
    # small negative arithmetic artifact is explicitly clipped for logplus
    geo=["noaa_log1p_eligible_counties","noaa_log1p_counties_matched","noaa_has_coverage"]
    reports=["noaa_log1p_report_count","noaa_log1p_reported_damage_count","noaa_reported_damage_available"]
    density=["report_count_per_eligible_county","report_count_per_matched_county",
             "reporting_county_fraction","reported_damage_fraction"]
    history=["lookback_event_count","lookback_per_county_log","lookback_covered_counties_frac"]
    contrast=["current_to_past_reporting_ratio","report_excess_relative_prior"]
    needed=list(dict.fromkeys(geo+reports+density+history+contrast))
    d=d.merge(imp[["disasterNumber"]+needed],on="disasterNumber",validate="one_to_one")
    variants={
       "fema_only":features["fema_only"],
       "fema_multiloc_rain":features["fema_multi_rain"],
       "fema_geo_event_counts":features["fema_only"]+geo+reports,
       "fema_prior_reporting_only":features["fema_only"]+history,
       "fema_normalized_report_density":features["fema_only"]+geo[:1]+density,
       "fema_normalized_plus_prior_reporting":features["fema_only"]+geo[:1]+density+history+contrast,
       "fema_geo_event_counts_plus_rain":features["fema_multi_rain"]+geo+reports,
       "fema_normalized_plus_prior_and_rain":features["fema_multi_rain"]+geo[:1]+density+history+contrast}
    assert d.disasterNumber.nunique()==len(d)==92
    return d,variants,imp,geo,reports

def run():
    d,variants,imp,geo,reports=read_arms()
    preds=[]; folds=[]
    for year in base.YEARS:
        training=d[d.fyDeclared<year].copy()
        testing=d[d.fyDeclared==year].copy()
        assert len(training)>=33 and len(testing)>0 and training.fyDeclared.max()<year
        folds.append({"holdout_year":year,"training_n":len(training),"test_n":len(testing)})
        for name,cols in variants.items():
            p,b=base.fit_fold(training,testing,cols,year)
            for (_,case),amount,klass in zip(testing.iterrows(),p,b):
                preds.append({"disasterNumber":int(case.disasterNumber),"fyDeclared":int(year),
                   "arm":name,"target":float(case.target),"pred_amount":float(amount),
                   "true_band":str(case.band),"pred_band":str(klass)})
        print("FY",year,"trained",len(training),"tested",len(testing),flush=True)
    p=pd.DataFrame(preds);p.to_csv(OUT/"phase36_predictions_59x8.csv",index=False)
    pd.DataFrame(folds).to_csv(OUT/"chronological_fold_provenance.csv",index=False)
    # Fixed raw prior Phase35 must reproduce exactly before interpreting
    old=pd.read_csv(IN/"phase35"/"phase35_predictions_59x5.csv")
    for oldname,newname in [("fema_only","fema_only"),("fema_geo_counts","fema_geo_event_counts")]:
        a=old[old.arm==oldname].sort_values("disasterNumber")
        b=p[p.arm==newname].sort_values("disasterNumber")
        assert np.array_equal(a.disasterNumber.to_numpy(),b.disasterNumber.to_numpy())
        assert np.allclose(a.pred_amount,b.pred_amount,rtol=0,atol=0.001),(oldname,"amount mismatch")
        assert (a.pred_band.to_numpy()==b.pred_band.to_numpy()).all(),(oldname,"band mismatch")
    metrics=[]
    for name,g in p.groupby("arm"):
        err=(g.pred_amount-g.target)
        q=g[g.target>0]
        metrics.append({"arm":name,"n":len(g),"MAE_USD":float(err.abs().mean()),
         "RMSE_USD":float(np.sqrt((err**2).mean())),"R2":float(r2_score(g.target,g.pred_amount)),
         "six_band_accuracy":float(np.mean(g.true_band==g.pred_band)),
         "n_positive":len(q),"positive_target_MAE_USD":float(np.abs(q.pred_amount-q.target).mean())})
    M=pd.DataFrame(metrics).sort_values("MAE_USD")
    M.to_csv(OUT/"phase36_model_metrics.csv",index=False)
    wide=p.pivot(index="disasterNumber",columns="arm",values="pred_amount").reset_index()
    truth=p[p.arm=="fema_only"][["disasterNumber","fyDeclared","target","true_band"]]
    cases=wide.merge(truth,on="disasterNumber").merge(imp,on="disasterNumber",validate="one_to_one")
    for name in variants:cases["error_"+name]=abs(cases[name]-cases.target)
    cases.to_csv(OUT/"phase36_caselevel_analysis.csv",index=False)
    baseerr=cases.error_fema_only.to_numpy()
    cols=[]
    rng=np.random.default_rng(36261102)
    for name in variants:
        if name=="fema_only":continue
        delt=cases["error_"+name].to_numpy()-baseerr
        fy=cases.fyDeclared.to_numpy()
        years=np.unique(fy)
        arr=[]
        for _ in range(3000):
            yy=rng.choice(years,len(years),replace=True)
            ix=np.concatenate([np.where(fy==y)[0] for y in yy])
            arr.append(float(np.mean(delt[ix])))
        lo,hi=np.quantile(arr,[.025,.975])
        cols.append({"arm":name,"mean_MAE_change_new_minus_FEMA_USD":float(delt.mean()),
           "improved":int(sum(delt < -1e-5)),"worsened":int(sum(delt>1e-5)),
           "unchanged":int(sum(abs(delt)<=1e-5)),
           "FY_bootstrap_interval_low_USD":float(lo),"FY_bootstrap_interval_high_USD":float(hi)})
    paired=pd.DataFrame(cols).sort_values("mean_MAE_change_new_minus_FEMA_USD")
    paired.to_csv(OUT/"phase36_paired_exploratory_bootstrap.csv",index=False)
    bins=[-0.1,0,1,4,9,float("inf")]
    cases["NOAA_eventcount_bin"]=pd.cut(cases.unambiguous_noaa_core_count,bins=bins,
                 labels=["0","1","2-4","5-9","10+"])
    sub=cases.groupby("NOAA_eventcount_bin",observed=False).agg(n=("disasterNumber","count"),
      median_actual_funding=("target","median"),
      avg_abs_error_FEMA=("error_fema_only","mean"),
      avg_abs_error_raw_count=("error_fema_geo_event_counts","mean"),
      avg_abs_error_prior_intensity=("error_fema_prior_reporting_only","mean")).reset_index()
    sub.to_csv(OUT/"phase36_error_by_report_count_bin.csv",index=False)
    # Fiscal year and positive-only audit.
    yearstats=[]
    for year,g in cases.groupby("fyDeclared"):
        for name in variants:
            yearstats.append({"fy":int(year),"arm":name,"n":len(g),
                "MAE":float(g["error_"+name].mean()),
                "MAE_positive":float(g.loc[g.target>0,"error_"+name].mean()) if any(g.target>0) else np.nan})
    pd.DataFrame(yearstats).to_csv(OUT/"phase36_metrics_by_test_year.csv",index=False)
    # Context-enriched audits from outcome-independent raw NOAA events.
    controls=imp[["disasterNumber","eligible_FEMA_county_areas","matched_FEMA_counties",
       "unambiguous_noaa_core_count","lookback_count","lookback_per_county",
       "report_count_per_eligible_county","current_to_past_reporting_ratio"]].copy()
    controls.to_csv(OUT/"phase36_reporting_intensity_source_features.csv",index=False)
    findings={"dataset":92,"tested":59,"holdout_years":base.YEARS,
       "metric_results":M.to_dict(orient="records"),
       "paired_results":paired.to_dict(orient="records"),
       "raw_phase35_reproduced_exactly":True,
       "restrictions":[
       "NOAA event report volume reflects physical hazard frequency AND reporting and administrative practices.",
       "FEMA designated counties are current API values, not verified declaration-time footprint.",
       "Past 730-day NOAA frequency is based on today's historical NOAA archive, not verified archived as-of snapshots.",
       "Early response T0/T72 cannot use contemporaneous final NOAA event records or final FEMA operations.",
       "Retrospective observational correlations and repeatedly inspected FY2021-2024 test are exploratory.",
       "Track A accepted model remains frozen and its 912-case metrics cannot be directly compared."]}
    (OUT/"summary.json").write_text(json.dumps(findings,indent=2))
    (OUT/"REPORT.md").write_text("# Phase36 — Reporting intensity and normalized hazard-count sensitivity\n\n"
       "This is an explicitly exploratory feature-family audit, NOT independent validation or an accepted model upgrade.\n\n"
       "## 59-case fiscal-year forward-test metrics\n\n"+M.to_markdown(index=False)
       +"\n\n## Exploratory paired comparisons\n\n"+paired.to_markdown(index=False)
       +"\n\n## Scope\nPast NOAA county reporting frequency is a proxy for both weather frequency and reporting practices. It does not distinguish these causally. Current FEMA designated counties and finalized NOAA records are retrospective.\n")
    print("PHASE36_SUMMARY",json.dumps({k:v for k,v in findings.items() if k!="restrictions"},indent=2),flush=True)
if __name__=="__main__":run()
