#!/usr/bin/env python3
"""Phase40: fixed satellite pilot SAR acquisition search and USGS/NWS event evidence provenance.

STAC discovery is public, repeatable, timestamp aware, no funding targets.
No observation is certified as event-specific flooded area solely from a
satellite scene footprint. Any radar pixel analysis must be on matched orbit
and a comparable acquisition pair with justified calibration and controls.
"""
from pathlib import Path
from datetime import timedelta
import json, time, hashlib, re
import pandas as pd
import requests
from urllib.parse import quote

ROOT=Path(__file__).resolve().parents[1]
IN=ROOT/"audit_inputs"/"phase40"
OUT=ROOT/"audit_outputs"/"phase40"
OUT.mkdir(parents=True,exist_ok=True)
cases=pd.read_csv(IN/"phase28"/"case_level_cmr_catalog_feasibility.csv")
space=pd.read_csv(IN/"phase38"/"all_92_flood_point_FEMA_designated_county_audit.csv")
q=pd.read_csv(IN/"phase38"/"12_case_satellite_spatiotemporal_and_quality_audit.csv")
assert len(cases)==len(q)==12
assert set(cases.disasterNumber)==set(q.disasterNumber)
assert len(space)==92
cases=cases.merge(q[["disasterNumber","quality_pass","noaa_point_inside_designated_county"]],
                  on="disasterNumber",validate="one_to_one")
API="https://planetarycomputer.microsoft.com/api/stac/v1/search"
COLLECTIONS="https://planetarycomputer.microsoft.com/api/stac/v1/collections"
SESSION=requests.Session()
SESSION.headers.update({"User-Agent":"University academic noncommercial NASA radar observational feasibility audit/1.0"})
def call(url,body=None):
    out=None
    for k in range(3):
        try:
            r=SESSION.post(url,json=body,timeout=(25,75)) if body is not None else SESSION.get(url,timeout=(25,75))
            r.raise_for_status()
            return r
        except Exception as exc:
            out=exc
            time.sleep(3*(k+1))
    raise out
collection_audit={}
for name in ["sentinel-1-rtc","jrc-gsw"]:
    try:
        raw=call(COLLECTIONS+"/"+name)
        a=raw.json()
        collection_audit[name]={"http_ok":True,"title":a.get("title"),
           "description_prefix":a.get("description","")[:450],
           "extent":a.get("extent"),"links":[x.get("href") for x in a.get("links",[]) if x.get("rel") in ["self","license"]]}
    except Exception as e:
        collection_audit[name]={"http_ok":False,"error":type(e).__name__+": "+str(e)[:250]}
(OUT/"stac_collection_verification.json").write_text(json.dumps(collection_audit,indent=2,default=str))
scene_records=[];summary=[];scene_assets=[]
for _,case in cases.sort_values("disasterNumber").iterrows():
    dn=int(case.disasterNumber);lon=float(case.NOAA_report_lon);lat=float(case.NOAA_report_lat)
    start=pd.to_datetime(case.incident_begin,utc=True)
    end=pd.to_datetime(case.incident_end_capped,utc=True)
    if pd.isna(end) or end<start:end=start+pd.Timedelta(days=10)
    # Fixed 2023/2024 observational window 45d before to 35d after event.
    first=start-pd.Timedelta(days=45)
    last=end+pd.Timedelta(days=35)
    if last-first>pd.Timedelta(days=150):last=first+pd.Timedelta(days=150)
    body={"collections":["sentinel-1-rtc"],"bbox":[lon-.08,lat-.08,lon+.08,lat+.08],
       "datetime":first.isoformat().replace("+00:00","Z")+"/"+last.isoformat().replace("+00:00","Z"),
       "limit":100}
    result={"disasterNumber":dn,"fyDeclared":int(case.fyDeclared),
      "point_lat":lat,"point_lon":lon,"FEMA_designated_county":bool(case.noaa_point_inside_designated_county),
      "original_optical_pair_quality_pass":bool(case.quality_pass),
      "incident_begin":str(case.incident_begin),"incident_end_capped":str(case.incident_end_capped),
      "SAR_search_window_begin":str(first),"SAR_search_window_end":str(last),
      "scene_count":0,"pre_scene_count":0,"event_or_post_scene_count":0,
      "matched_orbit_pre_post_pairs":0,"STAC_status":"unattempted"}
    try:
      raw=call(API,body)
      catalog=raw.json()
      items=catalog.get("features",[])
      if not isinstance(items,list):raise ValueError("Expected STAC features")
      scene_by_orbit={}
      for item in items:
        dt=pd.to_datetime(item["properties"].get("datetime"),utc=True)
        orbit=item["properties"].get("sat:relative_orbit",item["properties"].get("s1:relative_orbit"))
        direction=item["properties"].get("sat:orbit_state",item["properties"].get("s1:orbit_state","unknown"))
        orbit=f"{orbit}:{direction}"
        period="pre" if dt<start else ("post" if dt>=start and dt<=last else "other")
        scene_by_orbit.setdefault(orbit,set()).add(period)
        assets=item.get("assets",{})
        keys=list(assets.keys())
        row={"disasterNumber":dn,"item_id":item.get("id"),
         "datetime":dt.isoformat(),"phase":period,"orbit_key":orbit,
         "platform":item["properties"].get("platform"),
         "polarization":str(item["properties"].get("sar:polarizations")),
         "asset_keys":"|".join(keys),
         "stac_item_self":[a.get("href") for a in item.get("links",[]) if a.get("rel")=="self"][:1]}
        scene_records.append(row)
        if dn==4781:
          for k,v in assets.items():
            if k.lower() in ["vv","vh","hh","hv"]:
              scene_assets.append({"item_id":item.get("id"),"phase":period,"orbit_key":orbit,
                "asset":k,"href":v.get("href"),"type":v.get("type"),"raster_bands":v.get("raster:bands")})
      result.update({"STAC_status":"success","scene_count":len(items),
       "pre_scene_count":len([x for x in items if pd.to_datetime(x["properties"]["datetime"],utc=True)<start]),
       "event_or_post_scene_count":len([x for x in items if pd.to_datetime(x["properties"]["datetime"],utc=True)>=start]),
       "matched_orbit_pre_post_pairs":sum("pre" in a and "post" in a for a in scene_by_orbit.values()),
       "matched_orbit_keys":"|".join(k for k,v in scene_by_orbit.items() if "pre" in v and "post" in v),
       "stac_returned_matched":catalog.get("context",{}).get("matched"),
       "may_have_more_pages":any(x.get("rel")=="next" for x in catalog.get("links",[]))})
    except Exception as e:result.update({"STAC_status":"error","error":type(e).__name__+": "+str(e)[:240]})
    summary.append(result)
    print("SAR_CASE",dn,result["scene_count"],result["matched_orbit_pre_post_pairs"],result["STAC_status"],flush=True)
pd.DataFrame(summary).to_csv(OUT/"all_12_sentinel1_RTC_matched_orbit_availability.csv",index=False)
pd.DataFrame(scene_records).to_csv(OUT/"all_public_sentinel1_RTC_STAC_scene_metadata.csv",index=False)
(OUT/"4781_RTC_radar_assets_read_only.json").write_text(json.dumps(scene_assets,indent=2,default=str))
# JRC independent long-term permanence baseline: metadata only unless asset can be
# independently read and a hydrologically meaningful metric validated.
jrc={"point":[-95.20091816319608,30.60372608338876],"status":"not_run","items":[]}
try:
    rr=call(API,{"collections":["jrc-gsw"],"bbox":[-95.203,30.601,-95.198,30.607],
                 "limit":10})
    found=rr.json().get("features",[])
    jrc.update({"status":"success","n_items":len(found),
      "items":[{"id":x["id"],"keys":list(x.get("assets",{}).keys()),
                "asset_hrefs":{k:v.get("href") for k,v in x.get("assets",{}).items() if re.search("occurrence|seasonality|water",k,re.I)}} for x in found]})
except Exception as e:jrc["status"]="error_"+type(e).__name__+":"+str(e)[:250]
(OUT/"4781_JRC_global_surface_water_STAC_feasibility.json").write_text(json.dumps(jrc,indent=2))
df=pd.DataFrame(summary)
audit={
 "fixed_pilot_cases":12,
 "cases_STAC_search_successful":int(df.STAC_status.eq("success").sum()),
 "cases_with_pre_and_post_same_orbit":int(df.matched_orbit_pre_post_pairs.gt(0).sum()),
 "cases_optical_quality_failed":int((~df.original_optical_pair_quality_pass).sum()),
 "cases_optical_failed_with_SAR_orbit_pair":int(((~df.original_optical_pair_quality_pass)&df.matched_orbit_pre_post_pairs.gt(0)).sum()),
 "case4781_SAR_matched_orbit":bool(df.loc[df.disasterNumber==4781,"matched_orbit_pre_post_pairs"].iloc[0]>0),
 "case4781_jrc_STAC_status":jrc["status"],
 "event_specific_flood_raster_verified":False,
 "historical_2024_S1_RTC_metadata_only":True,
 "NOAA_FEMA_optical_same_instrument_confound_not_resolved":True,
 "track_A_frozen":True,
 "restrictions":[
 "Satellite acquisition metadata do not prove radar imagery is accessible, adequately calibrated or identifies floodwater.",
 "Same-relative-orbit alone does not guarantee comparable radar geometry, polarization, viewing or atmospheric/noise conditions.",
 "An optical historical water baseline dataset through 2021 is independent of 2024 event but is NOT an event inundation map.",
 "Quality screening must use paired observations; scenes obscured by SAR speckle/layover/shadow are not automatically dry.",
 "The pilot has 12 cases, some NOAA report points outside FEMA designated counties; not representative of all 92 floods.",
 "The six-band 912-case accepted Track A and its trained files were not modified."]}
(OUT/"phase40_summary.json").write_text(json.dumps(audit,indent=2))
(OUT/"PHASE40_REPORT.md").write_text(
 "# Phase 40 — Independent Sentinel-1 radar and historical water-baseline discovery\n\n"
 "**This study is acquisition screening, not a validated flood map or a new prediction model.**\n\n"
 "## Results\n\n"+df[["disasterNumber","original_optical_pair_quality_pass","FEMA_designated_county",
 "scene_count","pre_scene_count","event_or_post_scene_count","matched_orbit_pre_post_pairs","STAC_status"]].to_markdown(index=False)
 +"\n\n## Aggregate\n\n"+json.dumps({k:v for k,v in audit.items() if k!="restrictions"},indent=2)
 +"\n\n## Limitations\n\n"+"\n".join("- "+s for s in audit["restrictions"]))
print("PHASE40_RESULT",json.dumps({k:v for k,v in audit.items() if k!="restrictions"},indent=2),flush=True)
