#!/usr/bin/env python3
"""Outcome-blind stratified flood hydrology pilot, USGS drainage-network candidates.

Input: all 57 FEMA FY2019-2024 Flood disasters with NOAA report-point inside
NLDI gauge upstream basin, from immutable run 41404741366.

Select <=24 disasters using round-robin fiscal-year stratification by disaster
ID; NO funding outcomes or bands. Use nearest upstream-basin-matched gauge,
fall back to second such gauge only when discharge/history unavailable.
Validate county FIPS from station metadata in saved candidate list (from USGS
site response). Compare event-window DAILY MEAN discharge (USGS 00060/00003)
to strictly pre-incident historical daily means, up to 10 years and within 30
calendar days of event peak. Baseline >=60 days, >=3 distinct calendar years.

This is retrospective, not t0-safe; physical flood causation is NOT proved by
NOAA report point within basin. This script never trains funding models.
"""
from __future__ import annotations
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from functools import lru_cache
import json,math,time
import numpy as np,pandas as pd,requests

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/"audit_inputs"/"flood57_flow"
OUT=ROOT/"audit_outputs"/"flood57_flow"
OUT.mkdir(parents=True,exist_ok=True)
SITE="https://waterservices.usgs.gov/nwis/"
N_CASES=24
MAX_YEARS=10
WINDOW_DAYS=21
MIN_SEASON_DAYS=60
MIN_SEASON_YEARS=3
MIN_EVENT_DAYS=3
WORKERS=4

def request(endpoint,params):
    for attempt in range(3):
        try:
            resp=requests.get(SITE+endpoint,params=params,timeout=(15,80),
              headers={"User-Agent":"Master-thesis research USGS-FEMA hydrology audit"})
            if resp.status_code in (400,404):
                return None,"http_"+str(resp.status_code)
            resp.raise_for_status()
            return resp.json(),"ok"
        except Exception as ex:
            if attempt==2:return None,"request_failed_"+str(ex)[:100]
            time.sleep(1.25*(attempt+1))
    return None,"unknown_failure"

def read_daily(site,start,end):
    payload,status=request("dv/",{"format":"json","sites":str(site),
       "startDT":start.strftime("%Y-%m-%d"),"endDT":end.strftime("%Y-%m-%d"),
       "parameterCd":"00060","statCd":"00003","siteStatus":"all"})
    if payload is None:return pd.DataFrame(),status
    rows=[]
    try:
        for series in payload.get("value",{}).get("timeSeries",[]):
            info=series.get("variable",{})
            codes=[c.get("value") for c in info.get("variableCode",[])]
            if "00060" not in codes:continue
            for block in series.get("values",[]):
                for item in block.get("value",[]):
                    try:
                        val=float(item["value"])
                        date=pd.to_datetime(item["dateTime"],utc=True,errors="coerce")
                        if np.isfinite(val) and val>=0 and pd.notna(date):
                            rows.append((date.normalize(),val))
                    except (ValueError,KeyError,TypeError):pass
        if not rows:return pd.DataFrame(),"no_daily_mean"
        df=pd.DataFrame(rows,columns=["date","flow_cfs"])
        if (df.groupby("date").flow_cfs.nunique()>1).any():
            return pd.DataFrame(),"conflicting_series"
        return df.drop_duplicates("date").sort_values("date"),"ok"
    except Exception as ex:
        return pd.DataFrame(),"parse_failed_"+str(ex)[:90]

def day_distance(anchor,dates):
    anchor_doy=pd.Timestamp(year=2000,month=anchor.month,day=anchor.day).dayofyear
    dd=np.array([pd.Timestamp(year=2000,month=d.month,day=d.day).dayofyear for d in dates])
    return np.minimum(abs(dd-anchor_doy),366-abs(dd-anchor_doy))

def normalize(event,history,start,end):
    # Data completeness is on inclusive event-date interval.
    required=max(1,(end.normalize()-start.normalize()).days+1)
    features={"event_daily_n":int(len(event)),
      "event_daily_completeness":float(len(event)/required),
      "reference_daily_n":int(len(history)),
      "reference_calendar_years":int(history.date.dt.year.nunique()) if len(history) else 0}
    if len(event)<MIN_EVENT_DAYS:return features,"insufficient_event_days"
    if len(history)<MIN_SEASON_DAYS:return features,"insufficient_historical_days"
    peak=event.loc[event.flow_cfs.idxmax()]
    peak_day=peak.date
    seasonal=history.loc[day_distance(peak_day,history.date)<=30]
    n_years=seasonal.date.dt.year.nunique() if len(seasonal) else 0
    features.update({"event_peak_daily_mean_cfs":float(peak.flow_cfs),
       "event_peak_date":peak_day.strftime("%Y-%m-%d"),
       "prior_seasonal_n":int(len(seasonal)),
       "prior_seasonal_years":int(n_years)})
    if len(seasonal)<MIN_SEASON_DAYS or n_years<MIN_SEASON_YEARS:
        return features,"insufficient_seasonal_reference"
    ref=seasonal.flow_cfs.to_numpy(float)
    p50=float(np.median(ref));p90=float(np.percentile(ref,90));p95=float(np.percentile(ref,95))
    peakval=float(peak.flow_cfs)
    features.update({
      "prior_seasonal_median_cfs":p50,
      "prior_seasonal_p90_cfs":p90,
      "prior_seasonal_p95_cfs":p95,
      "peak_to_prior_seasonal_median_ratio":peakval/p50 if p50>0 else None,
      "peak_percentile_in_prior_seasonal_daily_means":float((np.count_nonzero(ref<peakval)+.5*np.count_nonzero(ref==peakval))/len(ref)),
      "event_peak_exceeds_p95":bool(peakval>p95),
      "event_days_above_prior_p90":int(np.count_nonzero(event.flow_cfs.to_numpy(float)>p90)),
      "event_fraction_observed_days_above_prior_p90":float(np.mean(event.flow_cfs.to_numpy(float)>p90)),
    })
    return features,"normalized_candidate"

def process_disaster(row,candidates):
    caseid=int(row.disasterNumber)
    start=pd.to_datetime(row.incidentBeginDate,utc=True,errors="coerce")
    end=pd.to_datetime(row.incidentEndDate,utc=True,errors="coerce")
    out={"disasterNumber":caseid,"fyDeclared":int(row.fyDeclared),
        "incidentBeginDate":str(start),"incidentEndDate":str(end),
        "normalized_discharge_available":False,
        "fema_causation_certified":False,"snapshot_asof_certified":False,
        "status":"not_attempted"}
    if pd.isna(start):
        out["status"]="missing_incident_start"
        return out,[]
    if pd.isna(end) or end<start:end=start+pd.Timedelta(days=7)
    end=min(end,start+pd.Timedelta(days=WINDOW_DAYS-1))
    historical_start=max(pd.Timestamp("2000-01-01",tz="UTC"),start-pd.DateOffset(years=MAX_YEARS))
    historical_end=start-pd.Timedelta(days=1)
    if historical_end<=historical_start:
        out["status"]="history_span_too_short";return out,[]
    attempt_rows=[]
    for rank,(_,station) in enumerate(candidates.iterrows(),1):
        site=str(station.candidate_station_id)
        strow={"disasterNumber":caseid,"station_id":site,"candidate_rank":rank,
          "station_name":station.station_name,
          "county_fips":str(station.county_fips).zfill(5),
          "station_huc":str(station.station_huc),
          "distance_noaa_report_km":float(station.station_to_nearest_noaa_report_km),
          "noaa_report_id":int(station.nearest_noaa_EVENT_ID) if pd.notna(station.nearest_noaa_EVENT_ID) else None,
          "nldi_upstream_intersection":True,
          "hydrological_causation_verified":False,
          "flow_status":"not_attempted"}
        event,es=read_daily(site,start,end)
        historic,hs=read_daily(site,historical_start,historical_end)
        strow.update({"event_api_status":es,"historical_api_status":hs,
           "event_daily_observations":len(event),"historic_daily_observations":len(historic)})
        if es=="ok" and hs=="ok":
            feature,status=normalize(event,historic,start,end)
            strow.update(feature);strow["flow_status"]=status
            attempt_rows.append(strow)
            if status=="normalized_candidate":
                out.update({
                  "status":"normalized_candidate","normalized_discharge_available":True,
                  "selected_station_id":site,"selected_station_rank":rank,
                  "station_huc":str(station.station_huc),
                  "distance_noaa_report_km":float(station.station_to_nearest_noaa_report_km),
                  "selected_station_county_fips":str(station.county_fips).zfill(5),
                  **feature})
                break
        else:
            strow["flow_status"]="api_missing"
            attempt_rows.append(strow)
        out["status"]=strow["flow_status"]
    out["candidate_attempts"]=len(attempt_rows)
    return out,attempt_rows

def choose_cases(rows):
    # One storm may have multiple FEMA declarations; this selects disaster IDs,
    # not independent meteorological storm clusters.
    peryear={}
    for fy,g in rows.groupby("fyDeclared"):
        peryear[int(fy)]=g.sort_values("disasterNumber").reset_index(drop=True)
    chosen=[]
    indexes={key:0 for key in peryear}
    while len(chosen)<N_CASES:
        progress=False
        for year in sorted(peryear):
            i=indexes[year]
            if i<len(peryear[year]):
                chosen.append(peryear[year].iloc[i])
                indexes[year]=i+1
                progress=True
                if len(chosen)>=N_CASES:break
        if not progress:break
    return pd.DataFrame(chosen)

def main():
    m=pd.read_excel(ROOT/"master_openfema_40plus.xlsx")
    assert len(m)==971
    m=m[(m.incidentType=="Flood")&m.fyDeclared.between(2019,2024)].copy()
    assert len(m)==92
    stations=pd.read_csv(INPUT/"top2_station_upstream_basin_screen.csv",
       dtype={"candidate_station_id":str,"county_fips":str,"station_huc":str})
    valid=stations[stations.upstream_match.eq(True)].copy()
    assert valid.disasterNumber.nunique()==57,(valid.disasterNumber.nunique(),len(valid))
    valid=valid.sort_values(["disasterNumber","station_to_nearest_noaa_report_km","candidate_station_id"])
    opts=m.merge(valid[["disasterNumber"]].drop_duplicates(),on="disasterNumber",validate="one_to_one")
    cases=choose_cases(opts)
    assert len(cases)==N_CASES and cases.disasterNumber.nunique()==N_CASES
    print("SELECTED_CASES",cases.groupby("fyDeclared").size().to_dict(),flush=True)
    tasks={int(row.disasterNumber):(row,valid[valid.disasterNumber==row.disasterNumber])
           for _,row in cases.iterrows()}
    case_records=[];station_attempts=[]
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        jobs={executor.submit(process_disaster,*item):dn for dn,item in tasks.items()}
        for fut in as_completed(jobs):
            dn=jobs[fut]
            try:
                case,tries=fut.result()
            except Exception as ex:
                case={"disasterNumber":dn,"status":"uncaught_error_"+str(ex)[:120],
                      "normalized_discharge_available":False}
                tries=[]
            case_records.append(case)
            station_attempts.extend(tries)
            print("FLOOD",dn,case["status"],
                  case.get("selected_station_id","none"),flush=True)
    cases_df=pd.DataFrame(case_records).sort_values("disasterNumber")
    tries_df=pd.DataFrame(station_attempts)
    assert len(cases_df)==N_CASES and cases_df.disasterNumber.nunique()==N_CASES
    assert cases_df.normalized_discharge_available.fillna(False).sum()<=N_CASES
    cases_df.to_csv(OUT/"flood_stratified_24_normalized_discharge.csv",index=False)
    tries_df.to_csv(OUT/"gauge_attempts_audit.csv",index=False)
    summary={
     "screened_flood_disasters":92,
     "disasters_with_upstream_basin_candidate":57,
     "sampled_disasters":len(cases_df),
     "sample_year_counts":{str(y):int(v) for y,v in cases.groupby("fyDeclared").size().items()},
     "usable_normalized_gauge_candidates":int(cases_df.normalized_discharge_available.sum()),
     "attempted_station_disaster_pairs":len(tries_df),
     "failures":{str(s):int(n) for s,n in cases_df.status.value_counts().items() if s!="normalized_candidate"},
     "method":"Historical USGS daily-mean discharge 00060/00003; closest screened gauge with NOAA report point inside upstream NLDI basin; preceding 10-year seasonal +/-30 days baseline, >=60 prior seasonal days >=3 years; 24 calendar-year-round-robin selected disasters, outcome-blind.",
     "feature_uses_true_funding":False,
     "held_out_model_comparison_performed":False,
     "caveats":[
       "USGS NLDI upstream basin contains the reported NOAA point, but this is not definitive hydrologic causation of FEMA flood losses.",
       "The event-time peak and reconstructed NOAA report matches are retrospective, not certified available at T0/T+72.",
       "Daily mean attenuates sudden flash-flood river peaks; station availability and hydrologic network coverage are selective.",
       "A small round-robin 24-case sample is not representative of all 92 floods, and disaster IDs can share larger weather events.",
       "A daily mean seasonal empirical percentile is not an annual flood return period.",
       "No NASA rainfall, flood polygons or measured property exposure included; accepted funding models unchanged."
     ]}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2))
    print("BEGIN_FLOOD57_FLOW")
    print(json.dumps(summary,indent=2))
    print("END_FLOOD57_FLOW")
if __name__=="__main__":
    main()
