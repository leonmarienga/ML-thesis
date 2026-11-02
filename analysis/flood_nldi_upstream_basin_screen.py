#!/usr/bin/env python3
"""Hydrological network screening of NOAA-to-USGS flood candidate pairs.

Uses NLDI NHDPlusV2 upstream basin for NWIS gages and verifies whether the
associated NOAA flood report POINT lies inside the station's upstream basin.
Point-in-upstream-basin is stronger than distance alone but NOT proof the
flood-report point is a flooded reach, nor that the gage recorded FEMA damages.

All 87 2019-24 flood events with geolocated reports are considered. First two
closest candidate stations <=25 km per disaster. Never use funding outcomes.
No river discharge is modeled here. Do not treat unmatched as zero severity.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,time
import numpy as np,pandas as pd,requests
from shapely.geometry import shape,Point

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"audit_inputs"/"nldi_flood"
OUT=ROOT/"audit_outputs"/"nldi_flood"
CACHE=ROOT/".cache"/"nldi_flood"
OUT.mkdir(parents=True,exist_ok=True)
CACHE.mkdir(parents=True,exist_ok=True)
API="https://api.water.usgs.gov/nldi/linked-data/nwissite/"
MAX_PER_DISASTER=2
DISTANCE_LIMIT_KM=25
WORKERS=5

def basin(site):
    path=CACHE/(str(site)+".json")
    if path.exists():
        payload=json.loads(path.read_text())
    else:
        url=API+"USGS-"+str(site)+"/basin"
        payload=None
        for attempt in range(3):
            try:
                res=requests.get(url,params={"f":"json"},timeout=(12,55),
                  headers={"User-Agent":"University disaster hydrology network research"})
                if res.status_code in (404,400):
                    return None,"not_indexed_"+str(res.status_code)
                res.raise_for_status()
                payload=res.json()
                path.write_text(json.dumps(payload))
                break
            except Exception as ex:
                print("NLDI_RETRY",site,attempt,str(ex)[:110],flush=True)
                time.sleep(attempt+1)
        if payload is None:return None,"nldi_unavailable"
    try:
        if payload.get("type")=="FeatureCollection":
            parts=[shape(f["geometry"]) for f in payload.get("features",[])
                   if f.get("geometry")]
        elif payload.get("type")=="Feature":
            parts=[shape(payload["geometry"])]
        else:
            return None,"unexpected_json_shape"
        parts=[g for g in parts if not g.is_empty and g.is_valid]
        if not parts:return None,"missing_valid_basin"
        return parts,"ok"
    except Exception as ex:
        return None,"geometry_error_"+str(ex)[:80]

def examine(record,basin_result):
    parts,status=basin_result
    out=record.copy()
    out["basin_api_status"]=status
    out["inside_station_upstream_basin"]=None
    out["basin_area_degrees2"]=None
    if not parts:return out
    p=Point(float(record["noaa_point_lon"]),float(record["noaa_point_lat"]))
    out["inside_station_upstream_basin"]=bool(any(g.covers(p) for g in parts))
    out["basin_area_degrees2"]=float(sum(g.area for g in parts))
    return out

def main():
    candidates=pd.read_csv(SRC/"top3_candidate_stations_by_disaster.csv",
        dtype={"candidate_station_id":"string","county_fips":"string"})
    assert candidates.disasterNumber.nunique()==87
    candidates=candidates[candidates.station_to_nearest_noaa_report_km<=DISTANCE_LIMIT_KM].copy()
    candidates=candidates.sort_values(
       ["disasterNumber","station_to_nearest_noaa_report_km","candidate_station_id"])
    candidates=candidates.groupby("disasterNumber",sort=False).head(MAX_PER_DISASTER)
    candidates=candidates[candidates.candidate_station_id.notna()].copy()
    candidates["candidate_station_id"]=candidates.candidate_station_id.str.replace(r"\.0$","",regex=True)
    candidates["candidate_station_id"]=candidates.candidate_station_id.str.zfill(8)
    assert candidates.candidate_station_id.str.fullmatch(r"\d{8,15}").all()
    selected=candidates.to_dict("records")
    station_ids=sorted(set(x["candidate_station_id"] for x in selected))
    print("NLDI_BASIN_QUERY_COUNTS",len(selected),"pairs",len(station_ids),"unique stations",flush=True)
    result={}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        pending={pool.submit(basin,site):site for site in station_ids}
        for f in as_completed(pending):
            site=pending[f]
            try:result[site]=f.result()
            except Exception as e:result[site]=(None,"exception_"+str(e)[:90])
            if len(result)%15==0 or len(result)==len(station_ids):
                print("NLDI_PROGRESS",len(result),"/",len(station_ids),flush=True)
    rows=[examine(x,result[x["candidate_station_id"]]) for x in selected]
    d=pd.DataFrame(rows)
    d["basin_checked"]=d.basin_api_status.eq("ok")
    d["upstream_match"]=d.inside_station_upstream_basin.eq(True)
    d["is_nearby"]=d.station_to_nearest_noaa_report_km.le(DISTANCE_LIMIT_KM)
    # Only observational quality stage: NOT a verified FEMA flood attribution.
    d["fema_flood_attribution_certified"]=False
    d.to_csv(OUT/"top2_station_upstream_basin_screen.csv",index=False)
    winners=(d.sort_values(["disasterNumber","upstream_match",
       "station_to_nearest_noaa_report_km"],ascending=[True,False,True])
       .groupby("disasterNumber").head(1).copy())
    winners.to_csv(OUT/"best_hydrological_candidate_by_flood.csv",index=False)
    report={
       "population":{"geolocated_flood_disasters":87,
           "disasters_with_station_within_25km_before_nldi":int(candidates.disasterNumber.nunique()),
           "selected_station_disaster_pairs":len(d),
           "queried_unique_stations":len(station_ids)},
       "nldi_results":{
           "pairs_with_valid_upstream_basin":int(d.basin_checked.sum()),
           "disasters_with_valid_upstream_basin":int(d.loc[d.basin_checked,"disasterNumber"].nunique()),
           "pairs_with_report_point_in_station_upstream_basin":int(d.upstream_match.sum()),
           "disasters_with_report_point_upstream_of_candidate_station":int(d.loc[d.upstream_match,"disasterNumber"].nunique()),
           "status_counts":{str(k):int(v) for k,v in d.basin_api_status.value_counts().items()}},
       "hydrological_attribution_certified":False,
       "not_model_features_yet":True,
       "method":"Check NOAA point within NLDI NHDPlusV2 upstream basin polygon of USGS NWIS station. Compare at most two nearest stations <=25km per FEMA flood.",
       "cautions":[
          "NOAA event coordinate is a report point, not inundation polygon.",
          "An upstream catchment intersection is evidence of possible hydrologic relevance, not disaster attribution.",
          "Gauge upstream basin may not overlap flood damage sites, even when report point falls inside.",
          "NLDI NHDPlusV2 basin polygons can be incomplete, especially for coastal and Alaska watersheds.",
          "Only top two nearby gauges screened, other legitimate gauges might be omitted.",
          "Retrospective NOAA and USGS records are not version certified at disaster declaration.",
          "No funding amount or funding band used during screening; no models refitted."
       ]}
    (OUT/"summary.json").write_text(json.dumps(report,indent=2))
    print("BEGIN_FLOOD_NLDI_NETWORK_SCREEN\n"+json.dumps(report,indent=2)+"\nEND_FLOOD_NLDI_NETWORK_SCREEN",flush=True)
if __name__=="__main__":main()
