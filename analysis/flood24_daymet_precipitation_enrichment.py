#!/usr/bin/env python3
"""NASA Daymet 1-km DAILY precipitation enrichment for 24 FEMA Flood cases.

Distinct from GPM IMERG! Daymet is station-informed gridded meteorology.
FEMA disaster locations are NOAA FLOOD report POINTS selected through NLDI
upstream-USGS screen; these are not verified inundation polygons.

Date selection and matching do NOT use FEMA funding. A 21-day disaster window,
a prior 7-day antecedent interval and up to 10 pre-incident calendar years
define a strictly pre-event seasonal comparator. Note: historical Daymet data
and finalized NOAA reports are retrospective, not certified as-of T0/T+72.

Outputs include complete 24-row case-level file, request diagnostics,
retrospective rainfall features, missingness and a paired flow availability
comparison. No target modelling in this experiment.
"""
from __future__ import annotations
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from io import StringIO
import json,time
import numpy as np,pandas as pd,requests

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/"audit_inputs"/"daymet_flood24"
OUT=ROOT/"audit_outputs"/"daymet_flood24"
CACHE=ROOT/".cache"/"daymet_flood24"
OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
SERVICE="https://daymet.ornl.gov/single-pixel/api/data"
MAX_CASES=24
WINDOW_DAYS=21
PAST_YEARS=10
SEASON_WINDOW=30
MIN_SEASON_DAYS=60
MIN_SEASON_YEARS=3

def download_daymet(lat,lon,years):
    params={"lat":f"{lat:.4f}","lon":f"{lon:.4f}",
            "vars":"prcp","years":",".join(map(str,years))}
    cache=CACHE/("%+.4f_%+.4f_%d_%d.csv"%(lat,lon,min(years),max(years)))
    if cache.exists() and cache.stat().st_size>150:
        return cache.read_text(encoding="utf-8"),"cached"
    last="missing"
    for k in range(3):
        try:
            r=requests.get(SERVICE,params=params,timeout=(15,95),
                 headers={"User-Agent":"University NASA Daymet disaster-precipitation audit"})
            r.raise_for_status()
            if len(r.content)<120:raise ValueError("Response too short")
            response=r.text
            if "year" not in response.lower() or "prcp" not in response.lower():
                raise ValueError("No year/prcp in response")
            cache.write_text(response,encoding="utf-8")
            return response,"ok"
        except Exception as e:
            last=str(e)[:180];print("DAYMET_RETRY",lat,lon,k,last,flush=True)
            time.sleep(1.8*(k+1))
    return None,"request_failed:"+last

def parse_daymet(txt):
    # Daymet CSV has metadata comment rows and a nonstandard unit-bearing
    # precipitation header, usually 'prcp (mm/day)'.
    d=pd.read_csv(StringIO(txt),comment="#")
    d.columns=[str(x).strip() for x in d.columns]
    y=next((x for x in d.columns if x.lower()=="year"),None)
    doy=next((x for x in d.columns if x.lower()=="yday"),None)
    rain=next((x for x in d.columns if x.lower().startswith("prcp")),None)
    if not all([y,doy,rain]):
        raise ValueError("Expected columns year,yday,prcp; got "+str(d.columns.tolist()))
    d["year"]=pd.to_numeric(d[y],errors="coerce")
    d["yday"]=pd.to_numeric(d[doy],errors="coerce")
    d["rain_mm"]=pd.to_numeric(d[rain],errors="coerce")
    d=d.dropna(subset=["year","yday","rain_mm"]).copy()
    d=d[d.year.between(1980,2030)&d.yday.between(1,365)&d.rain_mm.ge(0)].copy()
    d["year"]=d.year.astype(int);d["yday"]=d.yday.astype(int)
    d["date"]=pd.to_datetime(d.year.astype(str)+"-01-01",utc=True)+pd.to_timedelta(d.yday-1,unit="D")
    if d.date.duplicated().any():raise ValueError("Duplicated daily date in Daymet")
    d=d.sort_values("date").reset_index(drop=True)
    # Only compute 3/7-day totals for truly consecutive dates in same year.
    for window in [3,7]:
        rolling=d.rain_mm.rolling(window,min_periods=window).sum()
        consecutive=d.date.diff().dt.days.eq(1)
        contiguous=consecutive.rolling(window-1,min_periods=window-1).sum().eq(window-1).shift(1,fill_value=False)
        # Robust simpler check on date at beginning and end, all adjacent.
        complete=(d.date-d.date.shift(window-1)).dt.days.eq(window-1)
        same_year=d.year.eq(d.year.shift(window-1))
        d[f"rolling_{window}d_mm"]=rolling.where(complete&same_year)
    return d

def season_distance(anchor,series):
    a=pd.Timestamp(year=2000,month=anchor.month,day=anchor.day).dayofyear
    doy=np.array([pd.Timestamp(year=2000,month=t.month,day=t.day).dayofyear for t in series])
    return np.minimum(np.abs(doy-a),366-np.abs(doy-a))

def evaluate_case(info,geo):
    start=pd.to_datetime(info.incidentBeginDate,utc=True,errors="coerce")
    end=pd.to_datetime(info.incidentEndDate,utc=True,errors="coerce")
    dn=int(info.disasterNumber)
    rec={"disasterNumber":dn,"fyDeclared":int(info.fyDeclared),
         "original_flow_candidate":bool(info.normalized_discharge_available),
         "rainfall_available":False,"rainfall_status":"not_attempted",
         "NASA_product":"Daymet Single Pixel daily precipitation",
         "NASA_product_is_IMERG":False,
         "geographic_attribution_certified":False,"asof_version_certified":False}
    if pd.isna(start):
        rec["rainfall_status"]="missing_start_date";return rec
    if pd.isna(end) or end<start:end=start+pd.Timedelta(days=7)
    end=min(end,start+pd.Timedelta(days=WINDOW_DAYS-1))
    lat=float(geo.noaa_point_lat);lon=float(geo.noaa_point_lon)
    years=list(range(start.year-PAST_YEARS,end.year+1))
    rec.update({"noaa_point_lat":lat,"noaa_point_lon":lon,
       "noaa_event_id":int(geo.nearest_noaa_EVENT_ID),
       "reference_station_id":str(geo.candidate_station_id),
       "noaa_point_to_station_km":float(geo.station_to_nearest_noaa_report_km),
       "window_start":start.strftime("%Y-%m-%d"),"window_end":end.strftime("%Y-%m-%d"),
       "requested_years":",".join(str(y) for y in years),
       "daymet_rest_url":SERVICE})
    txt,status=download_daymet(lat,lon,years)
    rec["source_status"]=status
    if txt is None:
        rec["rainfall_status"]="download_failed";return rec
    try:
        d=parse_daymet(txt)
    except Exception as e:
        rec["rainfall_status"]="parse_failed:"+str(e)[:110]
        return rec
    event=d[d.date.between(start.normalize(),end.normalize())].copy()
    before=d[d.date.between(start.normalize()-pd.Timedelta(days=7),
                            start.normalize()-pd.Timedelta(days=1))].copy()
    hist=d[(d.date<start.normalize())&
           (d.year<start.year)].copy()
    expected=int((end.normalize()-start.normalize()).days)+1
    rec.update({"available_event_days":len(event),
      "expected_event_days":expected,
      "event_coverage":float(len(event)/expected),
      "antecedent_7day_obs_count":len(before),
      "reference_years_available":int(hist.year.nunique()),
      "reference_daily_n":len(hist)})
    if len(event)<3 or len(event)/max(expected,1)<.80:
        rec["rainfall_status"]="insufficient_event_daily_coverage"
        return rec
    rec.update({
       "event_precip_total_mm":float(event.rain_mm.sum()),
       "event_max_1day_mm":float(event.rain_mm.max()),
       "event_max_3day_mm":float(event.rolling_3d_mm.max()) if event.rolling_3d_mm.notna().any() else None,
       "event_max_7day_mm":float(event.rolling_7d_mm.max()) if event.rolling_7d_mm.notna().any() else None,
       "event_rainy_days_ge_1mm":int((event.rain_mm>=1).sum()),
       "antecedent_7day_total_mm":float(before.rain_mm.sum()) if len(before)>=5 else None,
    })
    # Seasonal percentile is the event-window peak 3-DAY rolling accumulation
    # vs historical 3-day daily sums in preceding years during same season.
    if event.rolling_3d_mm.notna().any():
        peakrow=event.loc[event.rolling_3d_mm.idxmax()]
        h=hist[hist.rolling_3d_mm.notna()].copy()
        season=h.loc[season_distance(peakrow.date,h.date)<=SEASON_WINDOW]
        rec["seasonal_reference_3day_days"]=len(season)
        rec["seasonal_reference_3day_years"]=int(season.year.nunique())
        if len(season)>=MIN_SEASON_DAYS and season.year.nunique()>=MIN_SEASON_YEARS:
            baseline=season.rolling_3d_mm.to_numpy(float)
            peak=float(peakrow.rolling_3d_mm)
            rec.update({
              "prior_seasonal_3day_median_mm":float(np.median(baseline)),
              "prior_seasonal_3day_p90_mm":float(np.percentile(baseline,90)),
              "prior_seasonal_3day_p95_mm":float(np.percentile(baseline,95)),
              "event_peak_3day_prior_season_percentile":float(
                  (np.count_nonzero(baseline<peak)+.5*np.count_nonzero(baseline==peak))/len(baseline)),
              "event_peak_3day_exceeds_prior_season_p95":bool(peak>np.percentile(baseline,95))
            })
    rec["rainfall_available"]=True
    rec["rainfall_status"]="computed_retro"
    return rec

def main():
    f=pd.read_csv(INPUT/"flood_stratified_24_normalized_discharge.csv")
    candidate=pd.read_csv(INPUT/"top2_station_upstream_basin_screen.csv",
         dtype={"candidate_station_id":str})
    assert len(f)==MAX_CASES and f.disasterNumber.nunique()==MAX_CASES
    c=candidate[candidate.upstream_match==True].sort_values(
          ["disasterNumber","station_to_nearest_noaa_report_km","candidate_station_id"])
    c=c.drop_duplicates("disasterNumber")
    case=f.merge(c,on="disasterNumber",how="left",validate="one_to_one",indicator=True)
    assert (case._merge=="both").all(),case._merge.value_counts().to_dict()
    responses=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks={pool.submit(evaluate_case,x,x):int(x.disasterNumber)
               for _,x in case.iterrows()}
        for future in as_completed(tasks):
            dn=tasks[future]
            try:rec=future.result()
            except Exception as e:
                rec={"disasterNumber":dn,"rainfall_available":False,
                   "rainfall_status":"uncaught:"+str(e)[:140]}
            responses.append(rec)
            print("DAYMET_CASE",dn,rec["rainfall_status"],
                  rec.get("event_precip_total_mm","NA"),flush=True)
    final=case[["disasterNumber","fyDeclared","normalized_discharge_available"]].merge(
         pd.DataFrame(responses).drop(columns=["fyDeclared","original_flow_candidate"],
                    errors="ignore"),
         on="disasterNumber",how="left",validate="one_to_one")
    assert len(final)==MAX_CASES and final.disasterNumber.nunique()==MAX_CASES
    final.to_csv(OUT/"flood24_daymet_rainfall_and_flow_availability.csv",index=False)
    rain=final.rainfall_available.fillna(False)
    flow=final.normalized_discharge_available.fillna(False)
    both=int((rain&flow).sum());rain_only=int((rain&~flow).sum())
    report={
       "total_flood_disasters_in_pilot":len(final),
       "rainfall_available":int(rain.sum()),
       "flow_available_from_prior_pilot":int(flow.sum()),
       "both_rain_and_flow":both,
       "rain_only":rain_only,
       "flow_only":int((~rain&flow).sum()),
       "neither":int((~rain&~flow).sum()),
       "rainfall_by_fiscal_year":{str(y):{"n":len(g),
          "rain_count":int(g.rainfall_available.fillna(False).sum())}
          for y,g in final.groupby("fyDeclared")},
       "failure_status_counts":{str(k):int(v) for k,v in
          final.rainfall_status.value_counts().items()},
       "source":"NASA ORNL DAAC Daymet Single Pixel daily precipitation, 1-km station-informed gridded estimates, not IMERG satellite retrievals.",
       "source_docs":"https://daymet.ornl.gov/web_services.html",
       "spatiotemporal_match":"NOAA reported Flood point linked to FEMA disaster by historical storm event IDs, selected from NLDI point-in-station-upstream-basin candidates; FEMA incident dates capped to 21 days.",
       "annual_reference":"up to 10 previous calendar years, +/-30 days around max 3-day rain accumulation, >=60 prior seasonal observations and >=3 years.",
       "funding_target_used_for_matching":False,
       "funding_model_trained":False,
       "limitations":[
         "Daymet is not satellite IMERG; satellite retrieval was not acquired or independently validated.",
         "NOAA report point is not inundation footprint or verified area affected by the FEMA disaster.",
         "Current Daymet and finalized NOAA records are retrospective and not certified as-of disaster declaration.",
         "Daymet omits Dec 31 for leap years; missing dates and incomplete roll windows are explicitly tracked.",
         "Single point rainfall cannot represent heterogeneous multi-county disaster rainfall.",
         "Sample of 24 is selected from 57 NLDI screened disasters, not random all-FEMA sample.",
         "No prediction improvement is claimed and accepted six-band router and specialists remain immutable."
       ]}
    (OUT/"summary.json").write_text(json.dumps(report,indent=2))
    print("BEGIN_DAYMET_FLOOD24\n"+json.dumps(report,indent=2)+
          "\nEND_DAYMET_FLOOD24",flush=True)
if __name__=="__main__":main()
