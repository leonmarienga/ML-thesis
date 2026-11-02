#!/usr/bin/env python3
"""Phase42: Actual Sentinel-1 RTC VV/VH measurements for five optical-failure flood cases.

Protocol frozen before pixel reading: Phase40 same-orbit acquisition candidates,
earliest post within FEMA incident-end+7d; nearest pre same orbit; up to one
additional pre and post for temporal robustness. 2km x 2km grid at the prior NOAA
report location; paired pixel validity and local controls; historic JRC water
exclusion where independently available. This is a candidate screen, NOT a
validated flood polygon, model training, or event attribution.
"""
from __future__ import annotations
from pathlib import Path
import json, math, time, re
import numpy as np
import pandas as pd
import requests
import planetary_computer as pc
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.features import shapes
from pyproj import Transformer
from shapely.geometry import shape, mapping
from shapely.ops import transform as geometry_transform, unary_union
from scipy.ndimage import label

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"audit_inputs"/"phase42"/"phase40"
OUT=ROOT/"audit_outputs"/"phase42"
OUT.mkdir(parents=True,exist_ok=True)
COHORT=[4708,4717,4718,4719,4793,4797]
TARGET=[4708,4717,4718,4719,4793]
CATALOG=pd.read_csv(SRC/"all_12_sentinel1_RTC_matched_orbit_availability.csv")
SCENES=pd.read_csv(SRC/"all_public_sentinel1_RTC_STAC_scene_metadata.csv")
assert len(CATALOG)==12
assert set(TARGET).issubset(set(CATALOG.disasterNumber))
assert int(CATALOG.loc[CATALOG.disasterNumber==4797,"matched_orbit_pre_post_pairs"].iloc[0])==0

PC="https://planetarycomputer.microsoft.com/api/stac/v1"
SESSION=requests.Session()
SESSION.headers["User-Agent"]="Academic-Disaster-Research-Phase42-Radar-Physical-QA/1.0"
N=200
SPACING=10
QUALITY_THRESHOLD=.85
MIN_CONTROL_PIXELS=2500
MAX_CONTROL_SHIFT_DB=2
MIN_COMPONENT_CELLS=9
PIX_KM2=(SPACING**2)/1e6
GRIDS={}

def get(url,body=None):
    for i in range(4):
        try:
            x=SESSION.get(url,timeout=(25,65)) if body is None else SESSION.post(url,json=body,timeout=(25,65))
            x.raise_for_status()
            return x.json()
        except Exception as e:
            if i==3:raise
            time.sleep(2*(i+1))

def stac_item(itemid):
    x=get(PC+"/collections/sentinel-1-rtc/items/"+itemid)
    if x.get("id")!=itemid:raise RuntimeError("STAC ID mismatch")
    if "vv" not in x.get("assets",{}) or "vh" not in x.get("assets",{}):
        raise RuntimeError("No VV and VH assets")
    return x

def select_acquisitions(case,scenes):
    start=pd.to_datetime(case.incident_begin,utc=True)
    end=pd.to_datetime(case.incident_end_capped,utc=True)
    subset=scenes[scenes.disasterNumber==int(case.disasterNumber)].copy()
    subset["dt"]=pd.to_datetime(subset.datetime,utc=True)
    choices=[]
    for orbit,g in subset.groupby("orbit_key"):
        pre=g[g.dt<start].sort_values("dt")
        valid_post=g[(g.dt>=start)&(g.dt<=end+pd.Timedelta(days=7))].sort_values("dt")
        if pre.empty or valid_post.empty:continue
        first_post=valid_post.iloc[0]
        last_pre=pre.iloc[-1]
        score=((first_post.dt-start).total_seconds()/86400,
               (start-last_pre.dt).total_seconds()/86400,
               str(orbit))
        choices.append((score,orbit,pre,valid_post,g))
    if not choices:return None,None
    _,orbit,pre,valid_post,g=sorted(choices,key=lambda z:z[0])[0]
    chosen=pd.concat([pre.tail(2),valid_post.head(2)]).drop_duplicates("item_id")
    chosen=chosen.sort_values("dt").copy()
    assert (chosen.dt<start).sum()>=1 and (chosen.dt>=start).sum()>=1
    return orbit,chosen

def signed(url):
    from urllib.parse import urlparse
    host=urlparse(str(url)).hostname
    if host not in ("sentinel1euwestrtc.blob.core.windows.net","ai4edataeuwest.blob.core.windows.net"):
        raise ValueError("unexpected raster host: "+str(host))
    return pc.sign(str(url))

def grid_for(radar_url,lon,lat):
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",GDAL_HTTP_TIMEOUT="45",GDAL_HTTP_MAX_RETRY="2"):
      with rasterio.open(signed(radar_url)) as ds:
        crs=ds.crs
        assert crs is not None and crs.is_projected
    x,y=Transformer.from_crs("EPSG:4326",crs,always_xy=True).transform(lon,lat)
    west=math.floor((x-N*SPACING/2)/SPACING)*SPACING
    north=math.ceil((y+N*SPACING/2)/SPACING)*SPACING
    tf=from_origin(west,north,SPACING,SPACING)
    cols,rows=np.meshgrid(np.arange(N),np.arange(N))
    X=tf.c+(cols+.5)*tf.a
    Y=tf.f+(rows+.5)*tf.e
    d=np.hypot(X-x,Y-y)
    inner=d<150
    control=(d>=300)&(d<=750)
    assert int(control.sum())>MIN_CONTROL_PIXELS
    return crs,tf,inner,control

def read_layer(url,crs,tf,kind="sar"):
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
      GDAL_HTTP_TIMEOUT="55",GDAL_HTTP_MAX_RETRY="3",
      GDAL_HTTP_RETRY_DELAY="2",CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tiff,.tif",
      GDAL_HTTP_MULTIRANGE="YES"):
      with rasterio.open(signed(url)) as src:
        with WarpedVRT(src,crs=crs,transform=tf,width=N,height=N,
          resampling=Resampling.bilinear if kind=="sar" else Resampling.nearest,
          nodata=src.nodata) as vrt:
          raw=vrt.read(1,masked=True)
    arr=np.asarray(raw.filled(np.nan if kind=="sar" else 255),dtype=float)
    if kind=="sar":
        arr[(arr<=0)|~np.isfinite(arr)]=np.nan
        with np.errstate(invalid="ignore",divide="ignore"):
            arr=10*np.log10(arr)
    return arr

def jrc_water(lon,lat,crs,tf):
    result={"STAC_status":"unattempted","JRC_occurrence_valid_cells":0,
            "JRC_referenced_dry_cells":0}
    dry=None
    try:
        catalog=get(PC+"/search",{"collections":["jrc-gsw"],
             "bbox":[lon-.008,lat-.008,lon+.008,lat+.008],"limit":6})
        feats=catalog.get("features",[])
        if not feats:raise RuntimeError("no JRC tile")
        refs=[]
        for f in feats:
            urls={k:v.get("href") for k,v in f.get("assets",{}).items()}
            if "occurrence" not in urls or "seasonality" not in urls:continue
            occ=read_layer(urls["occurrence"],crs,tf,"jrc")
            sea=read_layer(urls["seasonality"],crs,tf,"jrc")
            valid=(occ>=0)&(occ<=100)&(sea>=0)&(sea<=12)
            refs.append((valid,valid&(occ<=10)&(sea==0)))
        if not refs:raise RuntimeError("no readable JRC occurrence and seasonality")
        valid=np.zeros((N,N),bool)
        dry=np.zeros((N,N),bool)
        for v,d in refs:valid|=v;dry|=d
        result.update({"STAC_status":"read_ok","tile_count":len(refs),
          "JRC_occurrence_valid_cells":int(valid.sum()),
          "JRC_referenced_dry_cells":int(dry.sum()),
          "JRC_reference_coverage":float(valid.mean()),
          "JRC_source_years":"1984-2020 historic reference"})
    except Exception as ex:
        result.update({"STAC_status":"failed","error":type(ex).__name__+": "+str(ex)[:230]})
    return dry,result

scene_rows=[];event_rows=[];comparison_rows=[];map_features=[]
for disaster in COHORT:
    case=CATALOG.loc[CATALOG.disasterNumber==disaster].iloc[0]
    record={"disasterNumber":disaster,
      "optical_quality_failed":not bool(case.original_optical_pair_quality_pass),
      "NOAA_point_inside_current_FEMA_designated_county":bool(case.FEMA_designated_county),
      "incident_begin":str(case.incident_begin),"incident_end_capped":str(case.incident_end_capped),
      "point_lon":float(case.point_lon),"point_lat":float(case.point_lat),
      "radar_verified_flood_extent":False,"FEMA_event_attribution_verified":False,
      "affected_population_verified":False,"funding_predictions_changed":False}
    orbit,chosen=select_acquisitions(case,SCENES)
    if chosen is None:
        record.update({"status":"NO_MATCHED_ORBIT_BEFORE_AFTER_WITHIN_INCIDENT_END_PLUS_7_DAYS",
                       "radar_pair_evaluable":False,"candidate_pixels":None,"candidate_area_km2":None})
        event_rows.append(record)
        print("CASE",disaster,"NO_VALID_PAIR",flush=True)
        continue
    record["chosen_orbit"]=str(orbit)
    record["selected_scene_count"]=len(chosen)
    start=pd.to_datetime(case.incident_begin,utc=True)
    item_map={}
    for _,r in chosen.iterrows():
        try:item_map[r.item_id]=stac_item(str(r.item_id))
        except Exception as ex:
            item_map[r.item_id]={"error":type(ex).__name__+": "+str(ex)[:230]}
    first=next((item_map[x] for x in chosen.item_id if "assets" in item_map[x]),None)
    if first is None:
        record.update({"status":"all_item_metadata_failed","radar_pair_evaluable":False})
        event_rows.append(record);continue
    try:
        crs,tf,inner,control=grid_for(first["assets"]["vv"]["href"],
                                     float(case.point_lon),float(case.point_lat))
    except Exception as ex:
        record.update({"status":"grid_failure","error":type(ex).__name__+": "+str(ex)[:240],
                       "radar_pair_evaluable":False})
        event_rows.append(record);continue
    arrays={}
    for _,r in chosen.iterrows():
        item=item_map[r.item_id]
        for band in ("vv","vh"):
            row={"disasterNumber":disaster,"item_id":str(r.item_id),
                 "orbit":str(orbit),"date":str(r.datetime)[:10],
                 "phase":str(r.phase),"polarization":band}
            try:
                img=read_layer(item["assets"][band]["href"],crs,tf)
                arrays[(str(r.item_id),band)]=img
                valid=np.isfinite(img)
                row.update({"pixel_read":"success","grid_valid_n":int(valid.sum()),
                  "center_inner_valid_n":int((valid&inner).sum()),
                  "inner_median_db":float(np.median(img[valid&inner])) if (valid&inner).any() else None,
                  "control_median_db":float(np.median(img[valid&control])) if (valid&control).any() else None})
            except Exception as e:
                row.update({"pixel_read":"failed","error":type(e).__name__+": "+str(e)[:200]})
            scene_rows.append(row)
            print("PIXEL",disaster,str(r.datetime)[:10],band,row["pixel_read"],flush=True)
    before=chosen[chosen.dt<start].iloc[-1]
    after=chosen[chosen.dt>=start].iloc[0]
    record["selected_pre_date"]=str(before.dt)
    record["selected_post_date"]=str(after.dt)
    record["days_from_event_start_to_post"]=float((after.dt-start).total_seconds()/86400)
    # Critically: post can occur near incident close, not necessarily maximum flooding.
    if not all((str(x.item_id),band) in arrays for x in (before,after) for band in ("vv","vh")):
        record.update({"status":"paired_band_read_incomplete","radar_pair_evaluable":False})
        event_rows.append(record);continue
    diffs={}
    paired={}
    median_ctrl=[]
    for band in ("vv","vh"):
        a=arrays[(str(before.item_id),band)]
        b=arrays[(str(after.item_id),band)]
        v=np.isfinite(a)&np.isfinite(b)
        diff=b-a
        paired[band]=v
        diffs[band]=diff
        ctrl=diff[v&control]
        center=diff[v&inner]
        control_shift=float(np.median(ctrl)) if len(ctrl) else np.nan
        median_ctrl.append(control_shift)
        comparison_rows.append({"disasterNumber":disaster,"orbit":str(orbit),
           "pre_date":str(before.dt),"post_date":str(after.dt),"polarization":band,
           "paired_valid_fraction":float(v.mean()),
           "control_valid_cells":int(len(ctrl)),
           "control_median_change_db":control_shift,
           "inner_center_valid_cells":int(len(center)),
           "inner_center_median_change_db":float(np.median(center)) if len(center) else None,
           "inner_center_fraction_drop3db":float((center<=-3).mean()) if len(center) else None})
    vmask=paired["vv"]&paired["vh"]
    valid_frac=float(vmask.mean())
    control_stable=all(np.isfinite(x) and abs(x)<=MAX_CONTROL_SHIFT_DB for x in median_ctrl)
    enough_control=all(int((paired[band]&control).sum())>MIN_CONTROL_PIXELS for band in ("vv","vh"))
    record.update({"paired_both_polarization_valid_frac":valid_frac,
                   "local_control_stable":bool(control_stable),"enough_control":bool(enough_control)})
    dry,jrc_meta=jrc_water(float(case.point_lon),float(case.point_lat),crs,tf)
    record.update({("jrc_"+k):v for k,v in jrc_meta.items()})
    if dry is None:
        record.update({"status":"no_independent_historical_water_mask",
             "radar_pair_evaluable":True,"candidate_pixels":None,"candidate_area_km2":None})
        event_rows.append(record);continue
    water_signature=vmask&(diffs["vv"]<=-3)&(diffs["vh"]<=-3)&(
        arrays[(str(after.item_id),"vv")]<=-15)&(arrays[(str(after.item_id),"vh")]<=-20)
    admissible=water_signature&dry
    # Keep connected components of at least 9 radar cells, 4-neighbor
    comps,number=label(admissible,structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    retained=np.zeros(admissible.shape,dtype=bool)
    for id in range(1,number+1):
        component=(comps==id)
        if int(component.sum())>=MIN_COMPONENT_CELLS:
            retained|=component
    valid_quality=(valid_frac>=QUALITY_THRESHOLD and control_stable and enough_control)
    if not valid_quality:retained[:]=False
    record.update({"status":"QA_PASSED_CANDIDATE_SCREEN" if valid_quality else "quality_or_control_gate_failed",
      "radar_pair_evaluable":True,"radar_quality_accepted":bool(valid_quality),
      "strict_raw_signature_cells":int(water_signature.sum()),
      "historical_nonwater_signature_cells":int(admissible.sum()),
      "min9_component_cells":int(retained.sum()),
      "candidate_pixels":int(retained.sum()) if valid_quality else None,
      "candidate_area_km2":round(float(retained.sum()*PIX_KM2),7) if valid_quality else None,
      "candidate_area_is_NOT_certified_flood":True})
    if valid_quality and retained.any():
        convert=Transformer.from_crs(crs,"EPSG:4326",always_xy=True)
        polys=[]
        for geo,value in shapes(retained.astype(np.uint8),mask=retained,transform=tf):
            if int(value)==1:
                polys.append(geometry_transform(convert.transform,shape(geo)))
        if polys:
            geom=unary_union(polys)
            map_features.append({"type":"Feature","geometry":mapping(geom),"properties":{
                "disasterNumber":int(disaster),"source":"Sentinel-1 RTC dual-polarization + JRC historic screening",
                "candidate_km2":record["candidate_area_km2"],"event_flood_validated":False,
                "note":"Radar surface change candidate, not verified inundation or exposure"}})
    event_rows.append(record)
    print("CASE",disaster,record["status"],record.get("candidate_area_km2"),flush=True)

s=pd.DataFrame(scene_rows);s.to_csv(OUT/"radar_scene_band_qa.csv",index=False)
v=pd.DataFrame(comparison_rows);v.to_csv(OUT/"paired_radar_backscatter_by_disaster.csv",index=False)
e=pd.DataFrame(event_rows);e.to_csv(OUT/"six_optical_failed_flood_cases_radar_screen.csv",index=False)
(OUT/"radar_water_change_cANDIDATES_unverified.geojson").write_text(
    json.dumps({"type":"FeatureCollection","features":map_features}))
aggregate={
 "n_optically_unusable_cases":6,
 "matched_orbit_case_count_from_phase40":5,
 "radar_pair_with_both_polarizations_read":int(e.radar_pair_evaluable.fillna(False).sum()),
 "cases_passing_paired_validity_and_local_controls":int(e.radar_quality_accepted.fillna(False).sum()),
 "cases_with_nonzero_radar_screened_candidate":int(e.candidate_area_km2.fillna(0).gt(0).sum()),
 "sum_radar_candidate_km2_not_independent_ground_truth":float(e.candidate_area_km2.fillna(0).sum()),
 "cases_with_geographic_FEMA_county_containment":int(e.NOAA_point_inside_current_FEMA_designated_county.sum()),
 "cases_outside_FEMA_designated_counties":int((~e.NOAA_point_inside_current_FEMA_designated_county).sum()),
 "cases_with_new_verified_flood_extents":0,"affected_population_verified":0,
 "amount_regression_retrained":False,"frozen_TrackA_changed":False,
 "selection_rule":"Earliest same-orbit radar post-acquisition from incident onset to incident-end+7 days; closest preceding pre scene; optional extra dates are descriptive only.",
 "physical_signal_rule":"VV and VH backscatter both decrease >=3 dB; post VV<=-15 and VH<=-20; JRC historic occurrence<=10%, seasonality=0; four-connected radar component>=9 10m pixels; controls must be stable and >=85% paired-valid.",
 "limitations":[
   "Radar backscatter signatures do not uniquely imply flooding and thresholds are heuristics, not independently calibrated with flood masks.",
   "Snowmelt, soil moisture, seasonal conditions, and radar speckle can cause apparent backscatter changes.",
   "Historic JRC permanence is not 2023 or 2024 event-specific inundation truth.",
   "NOAA anchor locations are not randomly sampled; one disaster may have multiple flood footprints.",
   "One case has no matched pre-event radar; an unavailable observation is not absence of flooding.",
   "One report point is outside FEMA-designated counties, not valid for disaster-wide flood extent.",
   "Observed candidates are limited 2km by 2km local patches, not total state or disaster footprint.",
   "No fiscal funding outcomes read and accepted Track A remains untouched."]}
(OUT/"summary.json").write_text(json.dumps(aggregate,indent=2,default=str))
(OUT/"REPORT.md").write_text("# Phase 42 — Sentinel-1 radar pixel-screen for previously cloud-obscured FEMA flood cases\n\n"
 "**Actual RTC VV/VH raster measurements were attempted. No physical FEMA flood extent has been validated and no funding model updated.**\n\n"
 +e.to_markdown(index=False)+"\n\n## Overall\n\n"+json.dumps({k:v for k,v in aggregate.items() if k!="limitations"},indent=2)
 +"\n\n## Limitations\n\n"+"\n".join("- "+a for a in aggregate["limitations"]))
print("PHASE42_SUMMARY",json.dumps({k:v for k,v in aggregate.items() if k!="limitations"},indent=2),flush=True)
