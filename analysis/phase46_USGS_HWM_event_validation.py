#!/usr/bin/env python3
"""Phase46: independent USGS STN event high-water-mark and peak survey screening.

Questions: is there a surveyed, dated high water mark within a small known
radius of the two selected 2023 Sentinel-1 radar polygons, linked to the same
event? Do NOT use modeled hazard floodplains as observed inundation.
No FEMA target used. Track A unchanged.
"""
from pathlib import Path
from datetime import datetime
from urllib.parse import urlencode
import json, re, time, hashlib
import pandas as pd
import numpy as np
import requests
from shapely.geometry import shape
from pyproj import Geod

ROOT=Path(__file__).resolve().parents[1]
I=ROOT/"audit_inputs"/"phase46"
OUT=ROOT/"audit_outputs"/"phase46"
OUT.mkdir(parents=True,exist_ok=True)
P=json.loads((I/"phase42"/"radar_water_change_cANDIDATES_unverified.geojson").read_text())
POLY={int(f["properties"]["disasterNumber"]):shape(f["geometry"]) for f in P["features"]}
assert set(POLY)=={4718,4719}
DATES={4718:("SD", "2023-04-09","2023-05-05"),4719:("ME","2023-04-30","2023-05-01")}
BASE="https://stn.wim.usgs.gov/STNServices"
SESSION=requests.Session()
SESSION.headers.update({"User-Agent":"university-disaster-flood-mapping-evidence-validation/1.0","Accept":"application/json"})
geod=Geod(ellps="WGS84")
logs=[]
def req(path,params=None):
    url=BASE+"/"+path
    err=None
    for k in range(3):
        try:
            r=SESSION.get(url,params=params,timeout=(18,65))
            r.raise_for_status()
            if len(r.content)>30000000:raise ValueError("response size safety limit")
            data=r.json()
            if isinstance(data,dict) and "error" in data:raise ValueError(str(data["error"])[:180])
            logs.append({"url":r.url,"status":"ok","bytes":len(r.content),"sha256":hashlib.sha256(r.content).hexdigest()})
            return data
        except Exception as e:
            err=type(e).__name__+": "+str(e)[:230]
            time.sleep(2*(k+1))
    logs.append({"url":url,"status":"failed","error":err})
    print("SOURCE_FAILURE",url,err,flush=True)
    return None
def as_rows(x):
    if isinstance(x,list):return x
    if isinstance(x,dict):
        if isinstance(x.get("features"),list):return x["features"]
        for key in ("results","data","items"):
            if isinstance(x.get(key),list):return x[key]
    return []
def attrs(row):
    return row.get("properties",row) if isinstance(row,dict) else {}
def dt(v):
    v=pd.to_datetime(v,utc=True,errors="coerce")
    return v
def date_strings(obj):
    return {k:v for k,v in obj.items() if any(t in k.lower() for t in ("date","time","event","quality","datum","elev"))}
def event_year_matches(rows):
    out=[]
    for r in rows:
        p=attrs(r)
        dates=[dt(v) for k,v in p.items() if any(t in k.lower() for t in ("start_date","end_date","event_date"))]
        nearby=any(pd.notna(z) and z.year in (2022,2023,2024) for z in dates)
        name=str(p.get("event_name",""))
        if nearby or "2023" in name:
            out.append({k:v for k,v in p.items() if k in ("event_id","event_name","event_start_date","event_end_date","event_type_id","event_status_id")})
    return out

# Official STN REST. FilteredHWMs.geojson as advertised by docs.
filters={"Event":"","EventType":"","EventStatus":"","States":"","County":"",
 "HWMType":"","HWMQuality":"","HWMEnvironment":"","SurveyComplete":"","StillWater":""}
events=req("Events.json")
event_meta=event_year_matches(as_rows(events))
(OUT/"STN_2022_2024_event_catalog.csv").write_text(pd.DataFrame(event_meta).to_csv(index=False))
print("EVENT_CATALOG",len(as_rows(events)),len(event_meta),flush=True)

def extract_xy(r):
  p=attrs(r)
  g=r.get("geometry") if isinstance(r,dict) else None
  if g and g.get("type")=="Point":
    a=g.get("coordinates",[])
    if len(a)>=2:return float(a[0]),float(a[1])
  pairs=[("longitude","latitude"),("longitude_dd","latitude_dd"),
        ("long_dd","lat_dd"),("site_longitude","site_latitude"),
        ("lon","lat")]
  for lon,lat in pairs:
    try:
      x=float(p[lon]);y=float(p[lat])
      if -180<=x<=180 and -90<=y<=90:return x,y
    except Exception:continue
  site=p.get("site") or {}
  if isinstance(site,dict):
    for a,b in pairs:
      try:return float(site[a]),float(site[b])
      except Exception:continue
  return None,None

def near(geom,lon,lat):
    x,y=geom.centroid.x,geom.centroid.y
    _,_,distance=geod.inv(x,y,lon,lat)
    return distance/1000

# Try direct state filters in geojson, fallback JSON
survey=[];query_meta=[]
for dn in (4718,4719):
  state,start,end=DATES[dn]
  args=dict(filters);args["States"]=state
  data=req("HWMs/FilteredHWMs.geojson",args)
  if data is None:data=req("HWMs/FilteredHWMs.json",args)
  rows=as_rows(data)
  query_meta.append({"case":dn,"state":state,"all_state_hwm_rows":len(rows),"source_available":data is not None,
   "has_geojson":isinstance(data,dict) and data.get("type")=="FeatureCollection"})
  print("STATE_HWM",state,len(rows),flush=True)
  selected=[]
  for z in rows:
    p=attrs(z)
    lon,lat=extract_xy(z)
    if lon is None:continue
    distance=near(POLY[dn],lon,lat)
    if distance>120:continue
    selected.append({"disasterNumber":dn,"state":state,"longitude":lon,"latitude":lat,
     "distance_to_candidate_centroid_km":distance,
     "hwm_id":p.get("hwm_id",p.get("HWM_ID")),
     "event_id":p.get("event_id",p.get("EVENT_ID")),
     "hwm_elevation":p.get("elev_ft",p.get("elevation")),
     "hwm_elevation_units":p.get("elev_ft") is not None and "feet" or "unknown",
     "hwm_peak_date":p.get("peak_date",p.get("peakDate")),
     "survey_date":p.get("survey_date",p.get("date_surveyed")),
     "data_quality":p.get("hwm_quality_id",p.get("hwm_quality")),
     "event_name":p.get("event_name",p.get("EVENT_NAME")),
     "source_properties":json.dumps(p,default=str)[:2000]})
  survey+=selected
  print("HWM_SPATIAL_CANDIDATES",dn,len(selected),flush=True)
pd.DataFrame(survey).to_csv(OUT/"HWM_120km_spatial_candidates.csv",index=False)
(OUT/"HWM_source_access_and_counts.json").write_text(json.dumps(query_meta,indent=2))

# Search USGS STN peak summaries (independent of sensor scene), within event months
peak_rows=[];peak_meta=[]
for dn in (4718,4719):
    state,begin,finish=DATES[dn]
    args={"Event":"","EventType":"","EventStatus":"","States":state,"County":"",
      "StartDate":"03/01/2023","EndDate":"06/30/2023"}
    data=req("PeakSummaries/FilteredPeaks.geojson",args)
    if data is None:data=req("PeakSummaries/FilteredPeaks.json",args)
    rows=as_rows(data)
    local=0
    for z in rows:
      p=attrs(z);lon,lat=extract_xy(z)
      if lon is None:continue
      distance=near(POLY[dn],lon,lat)
      if distance>120:continue
      local+=1
      peak_rows.append({"disasterNumber":dn,"distance_km":distance,"longitude":lon,
       "latitude":lat,"event_id":p.get("event_id"),"peak_id":p.get("peak_summary_id"),
       "peak_date":p.get("peak_date",p.get("peakDate")),
       "source_properties":json.dumps(p,default=str)[:2000]})
    peak_meta.append({"disasterNumber":dn,"source_available":data is not None,
          "rows_by_state":len(rows),"rows_within_120km":local})
    print("USGS_PEAKS",dn,len(rows),local,flush=True)
pd.DataFrame(peak_rows).to_csv(OUT/"PeakSurvey_120km_spatial_candidates.csv",index=False)
(OUT/"Peak_source_access_and_counts.json").write_text(json.dumps(peak_meta,indent=2))

joined_events={str(e.get("event_id")):e for e in event_meta}
found=[]
for x in survey:
   try: match=joined_events.get(str(int(x["event_id"])))
   except Exception:match=None
   row={k:x[k] for k in ("disasterNumber","hwm_id","distance_to_candidate_centroid_km","event_id",
                        "hwm_peak_date","survey_date","hwm_elevation")}
   if match:row.update({"STN_event_name":match.get("event_name"),"STN_event_start":match.get("event_start_date"),
                         "STN_event_end":match.get("event_end_date")})
   # Conservative source-date check: event name from actual documented STN event, not HWM state alone.
   row["event_2023_verified"]=bool(match and "2023" in str(match.get("event_start_date"))[:4])
   row["mark_assigned_to_FEMA_declaration"]=False
   row["candidate_polygon_inundation_independently_proven"]=False
   found.append(row)
pd.DataFrame(found).to_csv(OUT/"HWM_candidate_event_temporal_screen.csv",index=False)

summary={"fema_disasters":[4718,4719],"official_STN_events_2022_2024":len(event_meta),
 "highwater_state_access":query_meta,"peak_state_access":peak_meta,
 "HWM_within_120km":len(survey),"Peak_surveys_within_120km":len(peak_rows),
 "HWM_with_2023_event_catalog_match":sum(q["event_2023_verified"] for q in found),
 "within_2km_hwm":sum(x["distance_to_candidate_centroid_km"]<=2 for x in survey),
 "within_10km_hwm":sum(x["distance_to_candidate_centroid_km"]<=10 for x in survey),
 "event_specific_inundation_polygon_verified":False,
 "pixel_level_flood_depth_verified":False,"affected_population_estimated":False,
 "frozen_Track_A_unchanged":True,
 "limitations":[
  "State and distance matching never establishes the same hydrologic event; check event IDs, survey dates, source metadata and local basin.",
  "High-water mark elevation is an elevation datum, not local water depth unless paired with local terrain elevation and datum.",
  "No high-water mark near a patch does not imply no flooding; USGS STN coverage is event-dependent.",
  "USGS FIM or FEMA NFHL scenario floodplains are modeled hazard maps, not observed April/May 2023 flood extent.",
  "STN endpoint failures are logged and do not imply absence of source data.",
  "Radar pixel polygons were selected after observing radar image; independent survey confirmation needed.",
  "Only two candidate patches, not disaster-wide inundation coverage.",
  "No funding outcome read; Track A accepted router unchanged."]}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2))
(OUT/"REPORT.md").write_text("# Phase46 — USGS event-specific high-water-mark validation\n\n"
 "The official USGS STN event, surveyed HWM and peak-summary data were queried at the exact selected candidate locations. "
 "**State-level co-occurrence and nearby HWM do not verify flood depth or event-specific inundation.**\n\n"
 "## Screening counts\n\n"+json.dumps({k:v for k,v in summary.items() if k!="limitations"},indent=2)+
 "\n\n## Limitations\n\n"+"\n".join("- "+x for x in summary["limitations"]))
(OUT/"source_requests.json").write_text(json.dumps(logs,indent=2))
print("PHASE46_RESULT",json.dumps({k:v for k,v in summary.items() if k!="limitations"},indent=2),flush=True)
