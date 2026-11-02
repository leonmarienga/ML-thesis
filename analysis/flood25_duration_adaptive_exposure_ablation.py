#!/usr/bin/env python3
"""Phase25: duration-adaptive Daymet sensitivity & county population proxy ablation.

Change only rainfall exclusion for 1/2-day flood incidents: >=1 observed
event day, at least 80% coverage. Original >=3 rule retained elsewhere.
Recompute excluded cases from original NOAA event locations and dates.
For multi-location cases, recompute excluded point rows, then aggregate by
exact same feature names as fixed Phase23. Compare original FY2021-2024
forward-year FEMA baseline and feature challengers on identical 59 cases.

County population represents residents of NOAA-reporting counties, NOT harmed
people, structural exposure, impacted households or verified flood footprint.
FEMA full operational fields and post-event NOAA reports are retrospective.
No accepted 912-case model artifacts modified.
"""
from pathlib import Path
import json
import numpy as np,pandas as pd

import flood24_daymet_precipitation_enrichment as daymet
import flood_forward_model_ablation as ablation

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"audit_inputs"/"flood25_adaptive"
OUT=ROOT/"audit_outputs"/"flood25_adaptive"
OUT.mkdir(parents=True,exist_ok=True)

def recalc(master,p,loc):
    f=master.loc[master.disasterNumber==int(p.disasterNumber)].iloc[0].copy()
    f["normalized_discharge_available"]=False
    q=p.copy()
    q["nearest_noaa_EVENT_ID"]=int(loc.noaa_event_id)
    q["noaa_point_lat"]=float(loc.noaa_point_lat)
    q["noaa_point_lon"]=float(loc.noaa_point_lon)
    q["candidate_station_id"]="NO_GAUGE_USED"
    q["station_to_nearest_noaa_report_km"]=np.nan
    return daymet.evaluate_case(f,q,min_event_days=1)

def main():
    master=pd.read_excel(ROOT/"master_openfema_40plus.xlsx")
    assert len(master)==971
    s=pd.read_csv(SRC/"single"/"all_92_flood_daymet_rainfall.csv")
    pts=pd.read_csv(SRC/"multi"/"all_noaa_point_daymet_measurements.csv")
    m=pd.read_csv(SRC/"multi"/"all_92_multilocation_rainfall_profiles.csv")
    f=pd.read_csv(SRC/"flow"/"flood_stratified_24_normalized_discharge.csv")
    exposure=pd.read_csv(SRC/"exposure"/"exposure_features_971.csv")
    assert len(s)==len(m)==92 and len(pts)==248 and len(f)==24
    assert s.disasterNumber.nunique()==m.disasterNumber.nunique()==92
    assert exposure.disasterNumber.nunique()==971

    changed_single=[]
    for _,r in s[s.rainfall_status=="insufficient_event_daily_coverage"].iterrows():
        z=recalc(master,r,r)
        assert z["rainfall_available"] and z["available_event_days"] in (1,2),(r.disasterNumber,z)
        changed_single.append(z)
        for col,val in z.items():
            if col in s.columns:
                s.loc[s.disasterNumber==int(r.disasterNumber),col]=val
    assert len(changed_single)==18
    # Recompute every point originally excluded due to one/two-day duration
    # and explicitly verify status and observed record count.
    changed_points=0
    for idx,r in pts[pts.rainfall_status=="insufficient_event_daily_coverage"].iterrows():
        z=recalc(master,r,r)
        assert z["rainfall_available"] and z["available_event_days"] in (1,2),(r.disasterNumber,r.point_rank,z)
        changed_points+=1
        for col,val in z.items():
            if col in pts.columns:
                pts.loc[idx,col]=val
    assert changed_points==47
    assert int(s.rainfall_available.sum())==87
    assert int(pts.rainfall_available.sum())==248

    # Use same Phase23 aggregation, with only the eligibility change.
    for dn,g in pts.groupby("disasterNumber"):
        v=g[g.rainfall_available==True]
        i=m.index[m.disasterNumber==dn]
        assert len(i)==1
        m.loc[i,"n_usable_locations"]=len(v)
        m.loc[i,"any_rainfall"]=bool(len(v))
        m.loc[i,"at_least_two_locations"]=bool(len(v)>=2)
        m.loc[i,"at_least_three_locations"]=bool(len(v)>=3)
        m.loc[i,"available_point_ranks"]="|".join(map(str,v.point_rank.astype(int)))
        for key in ["event_precip_total_mm","event_max_1day_mm",
              "event_max_3day_mm","event_max_7day_mm",
              "antecedent_7day_total_mm",
              "event_peak_3day_prior_season_percentile"]:
            vnum=pd.to_numeric(v[key],errors="coerce").dropna().to_numpy(float)
            m.loc[i,"n_"+key]=len(vnum)
            if len(vnum):
                for suffix,value in [
                     ("median",np.median(vnum)),("max",np.max(vnum)),
                     ("min",np.min(vnum)),
                     ("range",np.max(vnum)-np.min(vnum) if len(vnum)>=2 else np.nan),
                     ("cv",np.std(vnum)/np.mean(vnum) if len(vnum)>=2 and np.mean(vnum)>0 else np.nan)]:
                    m.loc[i,"multi_"+key+"_"+suffix]=value
    assert int(m.any_rainfall.sum())==87
    assert int(m.at_least_two_locations.sum())==82
    assert int(m.at_least_three_locations.sum())==79

    # Save complete input files for exact replay.
    inp=OUT/"model_inputs"
    for sub in ("single","multi","flow"):
        (inp/sub).mkdir(parents=True,exist_ok=True)
    s.to_csv(inp/"single"/"all_92_flood_daymet_rainfall.csv",index=False)
    m.to_csv(inp/"multi"/"all_92_multilocation_rainfall_profiles.csv",index=False)
    f.to_csv(inp/"flow"/"flood_stratified_24_normalized_discharge.csv",index=False)
    pts.to_csv(OUT/"all_248_point_duration_adaptive.csv",index=False)

    ablation.IN=inp
    ablation.OUT=OUT/"model_ablation"
    ablation.OUT.mkdir(parents=True,exist_ok=True)
    original_prepare=ablation.prepare
    def prepare_with_exposure():
        d,features,available=original_prepare()
        extra=exposure[["disasterNumber","pop_reported_counties",
               "noaa_reported_counties_n","noaa_reported_counties_with_census"]]
        d=d.merge(extra,on="disasterNumber",how="left",validate="one_to_one")
        for col in ["pop_reported_counties","noaa_reported_counties_n",
               "noaa_reported_counties_with_census"]:
            d["county_"+col]=pd.to_numeric(d[col],errors="coerce")
        cols=["county_pop_reported_counties","county_noaa_reported_counties_n",
              "county_noaa_reported_counties_with_census"]
        features["fema_county_exposure_proxy"]=features["fema_only"]+cols
        features["fema_multi_rain_county_proxy"]=features["fema_multi_rain"]+cols
        return d,features,available
    ablation.prepare=prepare_with_exposure
    ablation.ARMS=["fema_only","fema_single_rain","fema_multi_rain",
                   "fema_multi_rain_usgs","fema_county_exposure_proxy",
                   "fema_multi_rain_county_proxy"]
    ablation.main()
    results=json.loads((ablation.OUT/"summary.json").read_text())
    assert results["predicted_fy2021_2024_cases"]==59
    answer={
      "duration_adaptive_coverage":{"fema_floods":92,
        "single_location_before":69,"single_location_after":int(s.rainfall_available.sum()),
        "multi_any_before":69,"multi_any_after":int(m.any_rainfall.sum()),
        "multi_two_locations_after":int(m.at_least_two_locations.sum()),
        "multi_three_locations_after":int(m.at_least_three_locations.sum()),
        "recomputed_single_cases":len(changed_single),
        "recomputed_point_records":changed_points,
        "short_events_same_80pct_completeness_threshold":True,
        "short_event_minimum_days":1,
        "long_event_minimum_days":3},
      "cohort_test_n":59,"paired_ablation":results["paired_vs_fema_only"],
      "overall":{k:{f:v.get(f) for f in ["MAE_USD","R2","funding_band_accuracy"]}
                 for k,v in results["results"].items()},
      "limitations":[
        "Retrospective FEMA full-operation covariates and NOAA finalized matches not decision-time valid.",
        "County population is not affected population or flood-damaged housing.",
        "This is a new exploratory test on already repeatedly inspected 2021-24 fiscal years; not independent verification.",
        "Earlier-year training only for model targets, but features use retrospective source versions.",
        "Only 59 flood-only held-out cases, 11 usable flow cases and potentially correlated storms.",
        "This duration-adaptive protocol corrects an artificial exclusion rule and is not a demonstrated predictive accuracy gain.",
        "Accepted six-band 912-row 830/912 router and $5.015M accepted stack unchanged."
      ]}
    (OUT/"summary.json").write_text(json.dumps(answer,indent=2))
    print("BEGIN_PHASE25_ADAPTIVE")
    print(json.dumps(answer,indent=2))
    print("END_PHASE25_ADAPTIVE")
if __name__=="__main__":main()
