#!/usr/bin/env python3
"""Phase28: NASA OPERA DSWx-HLS CMR granule retrieval/access feasibility.

Fixed outcome-independent sample of 6 FEMA Flood incidents per calendar year
2023 and 2024 (when a NOAA event-linked coordinate is present). Uses source
FEMA incidentBeginDate, incidentEndDate and NOAA reported event location,
never final funding amounts. Compare CMR granules 45d pre-event vs incident
to 14d after. CMR only confirms candidate surface-water observations, NOT
actual flooding; optical cloud/quality and permanent water need raster audit.

CMR metadata open; PO.DAAC WTR GeoTIFF protected Earthdata URLs may require
credentials. Probe ONE water classification TIFF via bounded HTTP Range and
report auth/access status, without saving/processing raster data. Access probe
must never be mistaken for a successful water-pixel read.

NASA source: OPERA_L3_DSWX-HLS_V1, collection C2617126679-POCLOUD.
"""
from __future__ import annotations
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlsplit,urlunsplit
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,time
import pandas as pd, numpy as np,requests

ROOT=Path(__file__).resolve().parents[1]
IN=ROOT/"audit_inputs"/"dswx28"
OUT=ROOT/"audit_outputs"/"dswx28"
OUT.mkdir(parents=True,exist_ok=True)
NASA="https://cmr.earthdata.nasa.gov/search"
COLLECTION_ID="C2617126679-POCLOUD"
COLLECTION_SHORT="OPERA_L3_DSWX-HLS_V1"
MAX_YEAR=6
PRE_DAYS=45
POST_DAYS=14
MAX_TILES_PER_QUERY=200
WORKERS=4
HEADERS={"User-Agent":"Academic-disaster-flood-research-CMR-DSWx feasibility/1.0"}

def get(url,params=None):
    err=None
    for k in range(3):
        try:
            r=requests.get(url,params=params,timeout=(12,70),headers=HEADERS)
            r.raise_for_status()
            return r,"ok"
        except Exception as ex:
            err=str(ex)[:180]
            time.sleep(.8*(k+1))
    return None,"request_failed:"+str(err)

def date_ts(x):
    return pd.to_datetime(x,errors="coerce",utc=True)

def clean_url(url):
    p=urlsplit(str(url))
    return urlunsplit((p.scheme,p.netloc,p.path,"",""))

def cmr_search(fema):
    dn=int(fema.disasterNumber)
    lat=float(fema.noaa_point_lat);lon=float(fema.noaa_point_lon)
    begin=date_ts(fema.incidentBeginDate)
    end=date_ts(fema.incidentEndDate)
    if pd.isna(end) or end<begin:end=begin+pd.Timedelta(days=7)
    # Bound a FEMA incident extending months: this is only a feasibility pilot.
    end=min(end,begin+pd.Timedelta(days=21))
    start_q=begin-pd.Timedelta(days=PRE_DAYS)
    end_q=end+pd.Timedelta(days=POST_DAYS)
    params={
       "collection_concept_id":COLLECTION_ID,
       "point":f"{lon:.6f},{lat:.6f}",
       "temporal":f"{start_q.strftime('%Y-%m-%dT00:00:00Z')},{end_q.strftime('%Y-%m-%dT23:59:59Z')}",
       "page_size":MAX_TILES_PER_QUERY}
    response,status=get(NASA+"/granules.json",params)
    audit={
      "disasterNumber":dn,"fyDeclared":int(fema.fyDeclared),
      "incident_begin":begin.strftime("%Y-%m-%d"),
      "incident_end_capped":end.strftime("%Y-%m-%d"),
      "NOAA_report_lat":lat,"NOAA_report_lon":lon,
      "CMR_query_start":start_q.strftime("%Y-%m-%d"),
      "CMR_query_end":end_q.strftime("%Y-%m-%d"),
      "CMR_status":status,
      "CMR_reported_hits":None,"CMR_returned_n":0,
      "CMR_metadata_truncated":False,"pre_event_n":0,"incident_or_post_n":0,
      "CMR_pre_event_WTR_links_n":0,"CMR_event_WTR_links_n":0}
    records=[]
    if response is None:return audit,records
    try:
        data=response.json()
        entries=data["feed"]["entry"]
        audit["CMR_reported_hits"]=int(response.headers.get("CMR-Hits",len(entries)))
        audit["CMR_returned_n"]=len(entries)
        audit["CMR_metadata_truncated"]=audit["CMR_reported_hits"]>len(entries)
        for item in entries:
            timestamp=date_ts(item.get("time_start"))
            if pd.isna(timestamp):continue
            links=item.get("links",[]) or []
            # Use *actual* water classification WTR links; not other imagery or
            # auxiliary metadata. A WTR URL is not evidence of accessible bytes.
            wtr=[]
            for link in links:
                u=str(link.get("href",""))
                if u.lower().endswith("_b01_wtr.tif") and u.startswith("https://"):
                    wtr.append(clean_url(u))
            phase="pre" if timestamp<begin else "event_or_post" if timestamp<=end_q else "outside"
            if phase=="pre":
                audit["pre_event_n"]+=1
                audit["CMR_pre_event_WTR_links_n"]+=int(bool(wtr))
            elif phase=="event_or_post":
                audit["incident_or_post_n"]+=1
                audit["CMR_event_WTR_links_n"]+=int(bool(wtr))
            records.append({
              "disasterNumber":dn,"time_start":timestamp.isoformat(),
              "scene_phase":phase,
              "granule_id":item.get("id"),
              "title":item.get("title"),
              "cloud_cover_metadata":item.get("cloud_cover"),
              "wtr_link":wtr[0] if wtr else None,
              "link_count":len(links),"wtr_layer_raster_downloaded":False,
              "wtr_water_pixels_read":False,
              "water_change_flood_area_verified":False
            })
        audit["CMR_status"]="ok"
    except Exception as ex:
        audit["CMR_status"]="parse_error:"+str(ex)[:160]
    return audit,records

def test_wtr_access(url):
    # Protected NASA Earthdata asset; do NOT transmit credentials.
    result={"wtr_link_host":urlsplit(url).netloc,"wtr_layer_path":urlsplit(url).path,
      "download_attempt":"byte_range_0_1023",
      "accessibility":"not_tested",
      "actual_raster_downloaded":False,"actual_pixel_values_read":False}
    try:
        with requests.get(url,headers={**HEADERS,"Range":"bytes=0-1023"},
             timeout=(12,30),allow_redirects=True,stream=True) as response:
            result["http_status"]=response.status_code
            result["final_host"]=urlsplit(response.url).netloc
            result["content_type"]=response.headers.get("Content-Type","")
            result["content_range"]=response.headers.get("Content-Range")
            if response.status_code in (401,403):
                result["accessibility"]="earthdata_credentials_required_or_forbidden"
            elif response.status_code==206:
                result["accessibility"]="range_read_response_success"
            elif response.status_code==200:
                if "text/html" in result["content_type"].lower() or (
                    result["final_host"].endswith("urs.earthdata.nasa.gov")):
                    result["accessibility"]="login_or_html_not_raster"
                else:
                    result["accessibility"]="http_200_without_bounded_range_unverified"
            else:
                result["accessibility"]="http_"+str(response.status_code)
            # DO NOT stream any full large tif accidentally; only read if 206
            # and content_range proves bounded retrieval, and only <=1024 B.
            if response.status_code==206 and response.headers.get("Content-Range"):
                blob=next(response.iter_content(chunk_size=1024),b"")
                result["sampled_bytes"]=len(blob[:1024])
                result["sample_magic_hex"]=blob[:4].hex()
                result["actual_raster_downloaded"]=False
        return result
    except Exception as ex:
        result["accessibility"]="probe_error"
        result["error"]=str(ex)[:160]
        return result

def main():
    master=pd.read_excel(ROOT/"master_openfema_40plus.xlsx")
    noaa=pd.read_csv(IN/"selected_noaa_flood_report_locations.csv")
    assert len(master)==971 and len(noaa)==87
    d=master[(master.incidentType=="Flood")&
             master.fyDeclared.between(2023,2024)].copy()
    d=d.merge(noaa[["disasterNumber","noaa_point_lat","noaa_point_lon"]],
              on="disasterNumber",how="inner",validate="one_to_one")
    d["incident_year"]=pd.to_datetime(d.incidentBeginDate,utc=True,errors="coerce").dt.year
    d=d[d.incident_year.isin([2023,2024])]
    fixed=d.sort_values(["incident_year","disasterNumber"]).groupby(
             "incident_year",sort=True).head(MAX_YEAR)
    assert fixed.disasterNumber.nunique()==len(fixed)
    assert (fixed.groupby("incident_year").size()>=3).all(),(
         fixed.groupby("incident_year").size().to_dict())
    meta=[];granules=[]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        tasks={pool.submit(cmr_search,row):int(row.disasterNumber)
               for _,row in fixed.iterrows()}
        for fut in as_completed(tasks):
            dn=tasks[fut]
            try:a,scenes=fut.result()
            except Exception as ex:
                a={"disasterNumber":dn,"CMR_status":"fatal:"+str(ex)[:130]}
                scenes=[]
            meta.append(a);granules.extend(scenes)
            print("CMR_CASE",dn,a.get("CMR_status"),
                  a.get("pre_event_n"),a.get("incident_or_post_n"),flush=True)
    cases=pd.DataFrame(meta).sort_values("disasterNumber")
    scenes=pd.DataFrame(granules)
    cases.to_csv(OUT/"case_level_cmr_catalog_feasibility.csv",index=False)
    scenes.to_csv(OUT/"scene_level_cmr_metadata.csv",index=False)
    sample=scenes[scenes.wtr_link.notna()].sort_values(
      ["scene_phase","disasterNumber","time_start"]).head(1) if len(scenes) else pd.DataFrame()
    if len(sample):
        access=test_wtr_access(str(sample.iloc[0].wtr_link))
    else:
        access={"accessibility":"no_WTR_link_found","actual_raster_downloaded":False,
                "actual_pixel_values_read":False}
    (OUT/"wtr_http_access_probe.json").write_text(json.dumps(access,indent=2))
    valid=cases[cases.CMR_status=="ok"]
    report={
      "cohort_original_floods_2019_2024":92,
      "pilot_2023_2024_selected_cases":len(fixed),
      "pilot_cases_by_incident_year":{str(y):int(n) for y,n in
                  fixed.groupby("incident_year").size().items()},
      "cmr_collection_short_name":COLLECTION_SHORT,
      "cmr_collection_concept_id":COLLECTION_ID,
      "case_queries_ok":len(valid),
      "cases_with_pre_event_granules":int((valid.pre_event_n>0).sum()),
      "cases_with_incident_or_post_granules":int((valid.incident_or_post_n>0).sum()),
      "cases_with_both_temporal_sides":int(((valid.pre_event_n>0)&
                        (valid.incident_or_post_n>0)).sum()),
      "cases_with_WTR_links_on_both_sides":int(((valid.CMR_pre_event_WTR_links_n>0)&
                        (valid.CMR_event_WTR_links_n>0)).sum()),
      "candidate_scene_metadata_records":len(scenes),
      "cmr_query_truncated_cases":int(valid.CMR_metadata_truncated.sum()),
      "catalog_errors":cases.loc[cases.CMR_status!="ok",
                ["disasterNumber","CMR_status"]].to_dict("records"),
      "wtr_access_probe":access,
      "verified_downloaded_raster_count":0,
      "verified_water_pixels_read":0,
      "derived_inundation_polygons":0,
      "verified_exposed_households":0,
      "model_training_performed":False,
      "methodology_notes":[
        "Catalog geolocation is one NOAA report point per FEMA Flood and not the whole flood footprint.",
        "Pre-event observations and during/after acquisitions are candidate surface-water scene metadata, not actual inundation evidence.",
        "Optical scenes may be cloud-obscured or shadow/ice masked; actual WTR classes have not been read.",
        "Flood change requires seasonal reference water masks, same-grid alignment, quality masking, and event-specific validation.",
        "A WTR URL and a 206 HTTP range probe are NOT successful raster decoding or water classification.",
        "Protected Earthdata HLS imagery may require authenticated access and authorization.",
        "The first fixed pilot uses up to six case IDs in each of 2023 and 2024, not a representative national sample.",
        "No disaster funding target or model used in sampling. Accepted 830/912 router and ~$5.015M stack unchanged."
      ]}
    (OUT/"summary.json").write_text(json.dumps(report,indent=2))
    print("BEGIN_PHASE28_OPERA_CMR\n"+json.dumps(report,indent=2)+
          "\nEND_PHASE28_OPERA_CMR",flush=True)
if __name__=="__main__":main()
