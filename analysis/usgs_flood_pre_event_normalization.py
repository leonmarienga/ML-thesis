#!/usr/bin/env python3
"""Research-only USGS flow-normalization of twelve flood-gauge pilot rows.

USGS parameter 00060, statistic 00003: daily MEAN streamflow (cfs).
Reference uses up to 10 previous years, no event or later dates, seasonal
window +/-30 days around event peak calendar date. These are candidate station
metrics NOT verified flood footprint, return period, nor decision-time snapshots.
"""
import json,time,math
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"audit_inputs"/"flood_gauge_normalization"
OUT=ROOT/"audit_outputs"/"flood_gauge_normalization"
OUT.mkdir(parents=True,exist_ok=True)
S=requests.Session()
S.headers["User-Agent"]="Academic-FEMA-USGS-daily-streamflow-audit/1.0"
BASE="https://waterservices.usgs.gov/nwis/"

def fetch(endpoint,params):
    for a in range(3):
        try:
            r=S.get(BASE+endpoint,params=params,timeout=(12,85))
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            print("RETRY",endpoint,a,str(e)[:100],flush=True)
            time.sleep(2*(a+1))
    return None

def metadata(site):
    r=fetch("site/",{"format":"rdb","sites":site,"siteOutput":"expanded","siteStatus":"all"})
    if r is None:return None,"site_api_failed"
    try:
        lines="\n".join(x for x in r.text.splitlines() if not x.startswith("#"))
        d=pd.read_csv(StringIO(lines),sep="\t",dtype=str)
        q=d[d.site_no==site]
        if len(q)!=1:return None,"site_no_missing"
        return q.iloc[0].to_dict(),"ok"
    except Exception as e:return None,"parse_error_"+str(e)[:70]

def discharge(site,start,end):
    r=fetch("dv/",{"format":"json","sites":site,"startDT":start.strftime("%Y-%m-%d"),
       "endDT":end.strftime("%Y-%m-%d"),"parameterCd":"00060",
       "statCd":"00003","siteStatus":"all"})
    if r is None:return pd.DataFrame(),"daily_api_unavailable"
    try:
        observations=[]
        for series in r.json().get("value",{}).get("timeSeries",[]):
            p=series.get("variable",{}).get("variableCode",[{}])[0].get("value")
            if p!="00060":continue
            for block in series.get("values",[]):
                for item in block.get("value",[]):
                    v=float(item["value"])
                    t=pd.to_datetime(item["dateTime"],utc=True,errors="coerce")
                    if pd.notna(t) and np.isfinite(v) and v>=0:
                        observations.append((t.normalize(),v))
        d=pd.DataFrame(observations,columns=["date","flow_cfs"])
        if d.empty:return d,"missing_daily_mean"
        if (d.groupby("date").flow_cfs.nunique()>1).any():
            return pd.DataFrame(),"conflicting_daily_series"
        return d.drop_duplicates("date").sort_values("date"),"ok"
    except Exception as e:
        return pd.DataFrame(),"parse_error_"+str(e)[:65]

def seasonal_distance(anchor,dates):
    a=pd.Timestamp(year=2000,month=anchor.month,day=anchor.day).dayofyear
    v=np.array([pd.Timestamp(year=2000,month=d.month,day=d.day).dayofyear for d in dates])
    return np.minimum(abs(v-a),366-abs(v-a))

def scores(event,history):
    if len(event)<3:return {}, "fewer_than_3_event_days"
    if history.empty:return {}, "missing_preevent_history"
    peakrow=event.loc[event.flow_cfs.idxmax()]
    peak=float(peakrow.flow_cfs)
    seasonal=history.loc[seasonal_distance(peakrow.date,history.date)<=30]
    years=seasonal.date.dt.year.nunique()
    base={"peak_daily_mean_cfs":peak,"event_peak_date":str(peakrow.date.date()),
        "event_observation_days":len(event),"historical_observation_days":len(history),
        "seasonal_reference_days":len(seasonal),"seasonal_reference_years":int(years)}
    if len(seasonal)<60 or years<3:return base,"insufficient_seasonal_baseline"
    median=float(seasonal.flow_cfs.median())
    p90=float(seasonal.flow_cfs.quantile(.9))
    p95=float(seasonal.flow_cfs.quantile(.95))
    rank=(int((seasonal.flow_cfs<peak).sum())+.5*int((seasonal.flow_cfs==peak).sum()))/len(seasonal)
    base.update({"prior_seasonal_median_cfs":median,
      "prior_seasonal_p90_cfs":p90,"prior_seasonal_p95_cfs":p95,
      "peak_to_seasonal_median_ratio":peak/median if median>0 else None,
      "peak_prior_seasonal_percentile":float(rank),
      "event_days_above_prior_seasonal_p90":int((event.flow_cfs>p90).sum()),
      "event_peak_above_prior_seasonal_p95":bool(peak>p95)})
    return base,"normalized_candidate"

def main():
    pilot=pd.read_csv(SRC/"usgs_flood_county_gauge_pilot.csv",
      dtype={"selected_site":str,"county_fips":str})
    assert len(pilot)==12 and pilot.disasterNumber.nunique()==12
    rows=[]
    for _,r in pilot.iterrows():
        site=r.selected_site if pd.notna(r.selected_site) else None
        county=str(r.county_fips).zfill(5)
        output={"disasterNumber":int(r.disasterNumber),"county_fips":county,
           "selected_site":site,"feature_status":"no_gauge",
           "footprint_verified":False,"watershed_verified":False,
           "observation_version_asof_verified":False}
        if site is None:
            rows.append(output);continue
        site=str(site)
        info,stat=metadata(site)
        output["metadata_status"]=stat
        if info is None:
            output["feature_status"]="site_metadata_missing"
            rows.append(output);continue
        output.update({"site_name":info.get("station_nm"),
          "site_county_fips":str(info.get("county_cd","")).zfill(5),
          "latitude":info.get("dec_lat_va"),"longitude":info.get("dec_long_va"),
          "hydrologic_unit_code":info.get("huc_cd"),
          "drainage_area_sq_miles":info.get("drain_area_va"),
          "station_url":"https://waterdata.usgs.gov/monitoring-location/USGS-"+site})
        output["county_verified"]=output["site_county_fips"]==county
        if not output["county_verified"]:
            output["feature_status"]="county_mismatch"
            rows.append(output);continue
        start=pd.to_datetime(r.window_start,utc=True)
        end=pd.to_datetime(r.window_end,utc=True)
        prior_start=max(pd.Timestamp("2000-01-01",tz="UTC"),
            start-pd.DateOffset(years=10))
        prior_end=start-pd.Timedelta(days=1)
        if prior_start>=prior_end:
            output["feature_status"]="no_baseline_period"
            rows.append(output);continue
        event,es=discharge(site,start,end)
        history,hs=discharge(site,prior_start,prior_end)
        output.update({"event_api_status":es,"baseline_api_status":hs})
        if es=="ok" and hs=="ok":
            measurements,status=scores(event,history)
            output.update(measurements)
            output["feature_status"]=status
        else:
            output["feature_status"]="discharge_missing"
        rows.append(output)
        print("GAUGE",r.disasterNumber,site,output["feature_status"],
             output.get("peak_prior_seasonal_percentile"),flush=True)
    df=pd.DataFrame(rows)
    assert len(df)==12
    df.to_csv(OUT/"normalized_gauge_candidates.csv",index=False)
    status={str(k):int(v) for k,v in df.feature_status.value_counts().items()}
    report={"pilot_total":12,"candidate_gauges":int(df.selected_site.notna().sum()),
      "county_verified_gauges":int(df.county_verified.fillna(False).sum()),
      "normalized_candidates":int((df.feature_status=="normalized_candidate").sum()),
      "status":status,
      "method":"USGS daily MEAN streamflow cfs (00060, 00003); peak in event window compared to preceding up-to-10-year daily means +/-30 calendar days of event peak; minimum 60 seasonal days and 3 years",
      "not_a_return_period":True,
      "funding_outcome_used_for_selection":False,
      "disclaimer":"FIPS county match and station coordinates do not establish flood footprint, upstream basin, or site event attribution; revised USGS daily values do not verify historical decision-time availability.",
      "not_yet_done":["watershed/footprint validation","NASA IMERG rain extraction",
       "flood polygons and property exposure","full model ablation"]}
    (OUT/"summary.json").write_text(json.dumps(report,indent=2))
    print("BEGIN_NORMALIZED_GAUGES")
    print(json.dumps(report,indent=2))
    print("END_NORMALIZED_GAUGES")
if __name__=="__main__":main()
