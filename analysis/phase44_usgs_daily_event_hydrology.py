#!/usr/bin/env python3
"""Phase44 independent river-stage and flood event evidence screen.

Exact frozen Phase42 radar polygons (FEMA DR4718/4719) compared against:
1) USGS historic 2023 daily-discharge gauge observations within 100km;
2) USGS Short-Term Network high water mark service when accessible;
3) source-dated official FEMA and state declaration context (citation manifest).
Observations at nearby watersheds are NOT proof of polygon flooding.
No funding targets used, no model fitted, and accepted Track A is untouched.
"""
from pathlib import Path
import json, math, time
from datetime import datetime
from urllib.parse import urlparse
import requests
import numpy as np
import pandas as pd
from shapely.geometry import shape
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"audit_outputs"/"phase44"
OUT.mkdir(parents=True,exist_ok=True)
IN=ROOT/"audit_inputs"/"phase44"
poly=json.loads((IN/"phase42"/"radar_water_change_cANDIDATES_unverified.geojson").read_text())
prior=json.loads((IN/"phase43"/"phase43_case_audits.json").read_text())
assert len(poly["features"])==2 and len(prior)==2
shape_map={int(i["properties"]["disasterNumber"]):shape(i["geometry"]) for i in poly["features"]}
incident={
 4718:{"start":"2023-04-09","end":"2023-05-05","pre_radar":"2023-03-17","dark_radar":"2023-04-10","recovery_radar":"2023-04-22","state":"SD"},
 4719:{"start":"2023-04-30","end":"2023-05-01","pre_radar":"2023-04-25","dark_radar":"2023-05-07","recovery_radar":"2023-05-19","state":"ME"}}
assert set(shape_map)==set(incident)=={4718,4719}
S=requests.Session()
S.headers.update({"User-Agent":"Academic research disaster flood physical evidence audit 2026/1.0"})
USGS="https://api.waterdata.usgs.gov/ogcapi/v1/collections/daily/items"
STN="https://stn.wim.usgs.gov/STNServices/HWMs/FilteredHWMs.geojson"
def get(url,params=None,timeout=80):
    error=None
    for n in range(3):
      try:
        x=S.get(url,params=params,timeout=timeout)
        x.raise_for_status()
        return x
      except Exception as e:
        error=e
        time.sleep((n+1)*3)
    raise error
def dist_km(lon1,lat1,lon2,lat2):
    earth=6371.0088
    rad=np.radians([lat1,lat2])
    dlat=rad[1]-rad[0]
    dlon=np.radians(lon2-lon1)
    a=np.sin(dlat/2)**2+np.cos(rad[0])*np.cos(rad[1])*np.sin(dlon/2)**2
    return float(2*earth*np.arcsin(np.sqrt(min(1.0,a))))
def daily_obs(loc_lon,loc_lat):
    # Geographic eligibility is a radius, not a hydrological-basin association.
    body={"bbox":f"{loc_lon-.7},{loc_lat-.5},{loc_lon+.7},{loc_lat+.5}",
      "datetime":"2023-03-10/2023-06-05","parameter_code":"00060",
      "statistic_id":"00003","limit":1000,"f":"json"}
    rows=[];pages=0;url=USGS;params=body
    while pages<8:
      x=get(url,params,timeout=100)
      data=x.json()
      if data.get("type")!="FeatureCollection":raise ValueError("Unexpected USGS daily response "+str(data)[:260])
      feats=data.get("features",[])
      for f in feats:
        p=f.get("properties",{})
        geom=f.get("geometry")
        xy=geom.get("coordinates",[]) if isinstance(geom,dict) else []
        if not isinstance(xy,list) or len(xy)<2:continue
        sid=p.get("monitoring_location_id")
        if not sid:continue
        value=pd.to_numeric(p.get("value"),errors="coerce")
        date=pd.to_datetime(p.get("time"),errors="coerce")
        if not np.isfinite(value) or pd.isna(date):continue
        if str(p.get("parameter_code"))!="00060":continue
        if str(p.get("statistic_id"))!="00003":continue
        rows.append({"monitoring_location_id":sid,"time":str(date.date()),
          "discharge_cfs_daily_mean":float(value),"site_lat":float(xy[1]),
          "site_lon":float(xy[0]),"unit":p.get("unit_of_measure"),
          "distance_km":dist_km(loc_lon,loc_lat,float(xy[0]),float(xy[1])),
          "revision_last_modified":p.get("last_modified"),
          "approval_status":str(p.get("approvals_status",p.get("approval_status","")))})
      pages+=1
      print("USGS_PAGE",pages,"features",len(feats),"total_valid",len(rows),flush=True)
      next_link=next((l.get("href") for l in data.get("links",[]) if l.get("rel")=="next"),None)
      if not next_link:break
      if urlparse(next_link).hostname!="api.waterdata.usgs.gov":raise ValueError("Unexpected pagination host")
      url=next_link;params=None
    return pd.DataFrame(rows),pages,bool(next_link)
def highwater():
    """Read state STN marks; may be unavailable or too large, never infer absence."""
    try:
      params={"Event":"","EventType":"","EventStatus":"","States":"SD,ME","County":"",
       "HWMType":"","HWMQuality":"","HWMEnvironment":"","SurveyComplete":"","StillWater":""}
      x=get(STN,params,timeout=90)
      if len(x.content)>20_000_000:raise RuntimeError("STN result too large")
      data=x.json()
      feats=data.get("features",[]) if isinstance(data,dict) else []
      return {"status":"retrieved","feature_count":len(feats),
              "source":x.url,"event_specific_2023_fema_match_verified":False}
    except Exception as e:
      return {"status":"unavailable","reason":type(e).__name__+": "+str(e)[:200],
        "event_specific_2023_fema_match_verified":False}
full=[];gauges=[];results=[]
for dn in sorted(incident):
    geom=shape_map[dn]
    center=geom.centroid
    source={"disasterNumber":dn,"lat":center.y,"lon":center.x,
      "frozen_radar_candidate_km2":geom.area,
      "FEMA_state":incident[dn]["state"],
      "FEMA_incident_window":incident[dn]["start"]+" to "+incident[dn]["end"],
      "nearby_gauges_are_same_basin_verified":False,
      "site_specific_flood_stage_verified":False,"fema_footprint_verified":False}
    try:
      daily,pages,truncated=daily_obs(center.x,center.y)
      source.update({"USGS_daily_request":"ok","USGS_pages":pages,
          "USGS_pagination_truncated":truncated,"USGS_values_returned":len(daily)})
      if len(daily):
        daily["disasterNumber"]=dn
        daily["time_dt"]=pd.to_datetime(daily.time)
        daily=daily.sort_values(["monitoring_location_id","time_dt"]).drop_duplicates(
          ["monitoring_location_id","time_dt"])
        full.append(daily.drop(columns=["time_dt"]))
        for site,g in daily.groupby("monitoring_location_id"):
          dist=float(g.distance_km.iloc[0])
          if dist>100:continue
          prev=g[g.time_dt.between("2023-03-10","2023-04-08")] if dn==4718 else g[g.time_dt.between("2023-04-01","2023-04-29")]
          wave=g[g.time_dt.between("2023-04-09","2023-04-21")] if dn==4718 else g[g.time_dt.between("2023-04-30","2023-05-08")]
          late=g[g.time_dt.between("2023-04-22","2023-05-28")] if dn==4718 else g[g.time_dt.between("2023-05-09","2023-06-01")]
          usable=(len(prev)>=5 and len(wave)>=3)
          initial=float(prev.discharge_cfs_daily_mean.median()) if len(prev) else np.nan
          peak=float(wave.discharge_cfs_daily_mean.max()) if len(wave) else np.nan
          rec={"disasterNumber":dn,"site_id":site,"distance_km":dist,
            "days_full":len(g),"pre_n":len(prev),"wave_n":len(wave),"late_n":len(late),
            "candidate_window_mean_discharge_cfs":float(wave.discharge_cfs_daily_mean.mean()) if len(wave) else None,
            "candidate_window_max_discharge_cfs":peak if np.isfinite(peak) else None,
            "local_pre_period_median_cfs":initial if np.isfinite(initial) else None,
            "wave_peak_to_pre_median_ratio":peak/initial if initial>0 and np.isfinite(peak) else None,
            "wave_peak_date":str(wave.loc[wave.discharge_cfs_daily_mean.idxmax(),"time"]) if len(wave) else None,
            "candidate_signal_date_same_as_radar_dark":str(incident[dn]["dark_radar"]),
            "sampling_eligible":bool(usable),
            "local_river_and_radar_polygon_connection_verified":False,
            "flood_stage_threshold_available":False}
          gauges.append(rec)
        sites=pd.DataFrame([a for a in gauges if a["disasterNumber"]==dn])
        good=sites[sites.sampling_eligible] if len(sites) else pd.DataFrame()
        source.update({"sites_with_observation_within_100km":len(sites),
          "sites_with_5d_pre_3d_event_coverage":len(good),
          "closest_usable_gauge_km":float(good.distance_km.min()) if len(good) else None,
          "nearest_usable_gauge_ID":str(good.sort_values("distance_km").iloc[0].site_id) if len(good) else None,
          "gauge_observations_independent_of_radar":True})
      else:source["sites_with_5d_pre_3d_event_coverage"]=0
    except Exception as e:
      source.update({"USGS_daily_request":"failed","USGS_error":type(e).__name__+": "+str(e)[:260]})
    results.append(source)
    print("CASE",dn,source,flush=True)
hwm=highwater()
r=pd.DataFrame(results)
r.to_csv(OUT/"phase44_radar_vs_hydrologic_evidence_2_cases.csv",index=False)
if full:pd.concat(full,ignore_index=True).to_csv(OUT/"phase44_USGS_all_daily_discharge_observations.csv",index=False)
if gauges:pd.DataFrame(gauges).sort_values(["disasterNumber","distance_km"]).to_csv(
    OUT/"phase44_USGS_nearby_gauge_peak_screen.csv",index=False)
(OUT/"USGS_STN_high_water_marks_access_audit.json").write_text(json.dumps(hwm,indent=2))
source_manifest={
  "Official_USGS_daily_API":"https://api.waterdata.usgs.gov/ogcapi/v1/collections/daily/items",
  "Official_USGS_STN_HWM":"https://stn.wim.usgs.gov/STNServices/HWMs/FilteredHWMs.geojson",
  "Official_South_Dakota_2023_spring_flooding":"https://news.sd.gov/news?id=news_kb_article_view&sys_id=197dd2761bb3ad5031b1ebdbac4bcb70",
  "Official_FEMA_4718_map":"https://gis.fema.gov/maps/dec_4718.pdf",
  "Current_FEMA_designated_counties":"Phase34 linked original FEMA rows",
  "Maine_2023_context":"https://www.maine.gov/governor/mills/news/governor-mills-requests-disaster-declaration-president-biden-damage-caused-severe-spring-storm",
  "radar_phase43_artifact":"GitHub Actions run 41498539358"}
(OUT/"source_manifest.json").write_text(json.dumps(source_manifest,indent=2))
summary={"cases":results,"STN_HWM":hwm,
   "primary_status":"Independent nearby-gauge screening, not polygon-specific flood verification",
   "new_verified_FEMA_flood_footprints":0,
   "new_affected_population_estimates":0,
   "model_retrained":False,"track_A_changed":False,
   "caveats":[
     "The South Dakota state source independently confirms substantial April 2023 snowmelt flooding, not flood depth at the 43-pixel radar polygon.",
     "Nearby USGS station discharge cannot be assumed hydrologically upstream/downstream of a satellite patch without a verified network basin match.",
     "USGS daily data can be revised retrospectively and do not establish observed FEMA-declaration-time data availability.",
     "The Maine radar darkness occurred after the short incident; earlier repeated dark pulses weaken flood attribution.",
     "Radar polygons were selected from the tested satellite image, making repeated radar checks subject to selection bias.",
     "A lack of nearby gauge observations or high water marks is missing evidence, not absence of flooding.",
     "Archived static FEMA flood hazard layers are probabilistic risk, not observed April 2023 inundation."]}
(OUT/"phase44_summary.json").write_text(json.dumps(summary,indent=2))
(OUT/"PHASE44_REPORT.md").write_text("# Phase 44 — Independent hydrologic evidence screen\n\n"
  "This audit attempts **actual historical USGS daily discharge** and a USGS Short-Term Network high-water mark lookup around two previously selected small radar polygons. "
  "The evidence is not yet a verified event flood extent.\n\n"
  "## Cases\n\n"+r.to_markdown(index=False)+"\n\n"
  "## High-water marks source\n\n"+json.dumps(hwm,indent=2)+"\n\n"
  "## Limitations\n\n"+"\n".join("- "+x for x in summary["caveats"]))
print("PHASE44_SUMMARY",json.dumps({k:v for k,v in summary.items() if k!="caveats"},indent=2),flush=True)
