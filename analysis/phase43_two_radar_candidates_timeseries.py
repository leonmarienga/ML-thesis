#!/usr/bin/env python3
"""Phase43: follow-up radar time-series and independent flood context for DR4718/4719.

Diagnostics at two POST-SELECTED polygons from Phase42. Selection therefore
NOT independent confirmation. Read actual public Sentinel-1 RTC VV/VH scenes
at same orbit and 10m coordinate grid. Preserve original 43/41 candidates.
Measure time-series, local controls, candidate recession; compare with USGS NHD
static mapped waterways. No new flood extent is certified. No funding model.
"""
from pathlib import Path
from urllib.parse import urlparse
import json,math,time
import requests
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.features import rasterize
from shapely.geometry import shape, mapping
from shapely.ops import transform as geom_transform, unary_union
from pyproj import Transformer
import planetary_computer as pc

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"audit_inputs"/"phase43"
OUT=ROOT/"audit_outputs"/"phase43"
OUT.mkdir(parents=True,exist_ok=True)
PH42=pd.read_csv(SOURCE/"phase42"/"six_optical_failed_flood_cases_radar_screen.csv")
SAT=pd.read_csv(SOURCE/"phase40"/"all_public_sentinel1_RTC_STAC_scene_metadata.csv")
G=json.loads((SOURCE/"phase42"/"radar_water_change_cANDIDATES_unverified.geojson").read_text())
assert len(G["features"])==2
assert {int(a["properties"]["disasterNumber"]) for a in G["features"]}=={4718,4719}
REGIONS={int(a["properties"]["disasterNumber"]):shape(a["geometry"]) for a in G["features"]}
assert all(g.is_valid for g in REGIONS.values())
MPC="https://planetarycomputer.microsoft.com/api/stac/v1"
N=200;STEP=10
S=requests.Session();S.headers.update({"User-Agent":"academic-disaster-physical-severity-phase43/1.0"})
SRC_URLS={
    4718:"https://news.sd.gov/news?id=news_kb_article_view&sysparm_article=KB0040537",
    4719:"https://www.maine.gov/governor/mills/news/governor-mills-requests-disaster-declaration-president-biden-damage-caused-severe-spring-storm"}
DETAIL={
  4718:{"fema_state":"SD","storm_context":"Rapid spring snowmelt and river/overland flooding","declaration_date":"2023-07-06",
        "incident_start":"2023-04-09","official_incident_end":"2023-05-05"},
  4719:{"fema_state":"ME","storm_context":"May Day storm, heavy rain/river flooding","declaration_date":"2023-07-06",
        "incident_start":"2023-04-30","official_incident_end":"2023-05-01"}}
def get(uri):
    for k in range(4):
        try:
            x=S.get(uri,timeout=(25,90));x.raise_for_status();return x.json()
        except Exception as ex:
            if k==3:raise
            time.sleep(2*(k+1))
def sign(uri):
    u=urlparse(uri)
    if u.scheme!="https" or u.hostname not in (
      "sentinel1euwestrtc.blob.core.windows.net",
      "ai4edataeuwest.blob.core.windows.net"):
        raise ValueError("Untrusted satellite host "+str(u.hostname))
    return pc.sign(uri)
def config(url,lon,lat):
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",GDAL_HTTP_TIMEOUT="50"):
      with rasterio.open(sign(url)) as src:
        crs=src.crs
        assert crs is not None and crs.is_projected
    xx,yy=Transformer.from_crs("EPSG:4326",crs,always_xy=True).transform(lon,lat)
    west=math.floor((xx-1000)/10)*10
    north=math.ceil((yy+1000)/10)*10
    tf=from_origin(west,north,STEP,STEP)
    cols,rows=np.meshgrid(np.arange(N),np.arange(N))
    x=tf.c+(cols+.5)*tf.a;y=tf.f+(rows+.5)*tf.e
    ring=(np.hypot(x-xx,y-yy)>=300)&(np.hypot(x-xx,y-yy)<=750)
    return crs,tf,ring
def read(uri,crs,tf):
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",GDAL_HTTP_TIMEOUT="60",
      GDAL_HTTP_MAX_RETRY="3",GDAL_HTTP_RETRY_DELAY="2",CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff"):
      with rasterio.open(sign(uri)) as src:
        with WarpedVRT(src,crs=crs,transform=tf,width=N,height=N,
           nodata=src.nodata,resampling=Resampling.bilinear) as vrt:
             z=vrt.read(1,masked=True).filled(np.nan).astype(float)
    z[~np.isfinite(z)|(z<=0)]=np.nan
    with np.errstate(invalid="ignore",divide="ignore"):
      return 10*np.log10(z)
def nhd(geom):
    """USGS historical mapped permanent water features (not a 2023 inundation map)."""
    base="https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/"
    b=geom.bounds
    envelope=(b[0]-.015,b[1]-.015,b[2]+.015,b[3]+.015)
    output=[]
    for layer,name in [(6,"flowline"),(9,"water_area"),(12,"waterbody")]:
      params={"f":"geojson","where":"1=1","geometryType":"esriGeometryEnvelope",
        "geometry":",".join(map(str,envelope)),"inSR":"4326","outSR":"4326",
        "spatialRel":"esriSpatialRelIntersects","returnGeometry":"true",
        "outFields":"*"}
      row={"layer":name,"layer_id":layer,"status":"unattempted"}
      try:
        for i in range(3):
          try:
            x=S.get(base+str(layer)+"/query",params=params,timeout=(25,70));x.raise_for_status();break
          except Exception:
            if i==2:raise
            time.sleep(2)
        data=x.json()
        if "error" in data:raise RuntimeError(str(data["error"])[:250])
        geometries=[shape(q["geometry"]) for q in data.get("features",[]) if q.get("geometry")]
        row.update({"status":"ok","returned_features":len(geometries)})
        if geometries:
          union=unary_union(geometries)
          local=Transformer.from_crs("EPSG:4326","EPSG:5070",always_xy=True).transform
          pg=geom_transform(local,geom)
          ug=geom_transform(local,union)
          row["distance_m"]=float(pg.distance(ug))
          row["overlap_km2"]=float(pg.intersection(ug).area/1e6)
          row["sample_names"]=[str(q.get("properties",{}).get("GNIS_NAME","")) for q in data.get("features",[])][:10]
      except Exception as ex:row.update({"status":"error","error":type(ex).__name__+": "+str(ex)[:240]})
      output.append(row)
    return output

all_scenes=[];all_pair_diffs=[];all_case=[]
for dn in [4718,4719]:
  prior=PH42[PH42.disasterNumber==dn].iloc[0]
  geom=REGIONS[dn]
  c=SAT[(SAT.disasterNumber==dn)&(SAT.orbit_key==prior.chosen_orbit)].copy()
  c["dt"]=pd.to_datetime(c.datetime,utc=True)
  c=c.sort_values("dt")
  print("CASE",dn,"sameorbit",len(c),flush=True)
  items={}
  for _,r in c.iterrows():
      items[str(r.item_id)]=get(MPC+"/collections/sentinel-1-rtc/items/"+str(r.item_id))
  ref=items[str(c.iloc[0].item_id)]
  crs,tf,ring=config(ref["assets"]["vv"]["href"],float(prior.point_lon),float(prior.point_lat))
  target=geom_transform(Transformer.from_crs("EPSG:4326",crs,always_xy=True).transform,geom)
  mask=rasterize([(mapping(target),1)],out_shape=(N,N),transform=tf,all_touched=False,
    dtype=np.uint8).astype(bool)
  orig=int(prior.candidate_pixels)
  if mask.sum()!=orig:raise RuntimeError(f"Candidate pixel grid mismatch FEMA {dn}: {mask.sum()} != {orig}")
  im={}
  dates={}
  for _,r in c.iterrows():
    id=str(r.item_id)
    t=pd.Timestamp(r.dt)
    date=str(t.date())
    dates[id]=date
    scene={"disasterNumber":dn,"item_id":id,"date":date,"relative_days_to_start":
       float((t-pd.Timestamp(DETAIL[dn]["incident_start"],tz="UTC")).total_seconds()/86400),
       "phase":"pre_event" if date<DETAIL[dn]["incident_start"] else
             ("incident" if date<=DETAIL[dn]["official_incident_end"] else "post_incident"),
       "polarization_valid":False}
    try:
      for pol in ("vv","vh"):
          arr=read(items[id]["assets"][pol]["href"],crs,tf)
          im[(id,pol)]=arr
          v=arr[mask];q=arr[ring]
          scene[f"{pol}_candidate_valid_n"]=int(np.isfinite(v).sum())
          scene[f"{pol}_candidate_median_db"]=float(np.nanmedian(v)) if np.isfinite(v).any() else None
          scene[f"{pol}_control_median_db"]=float(np.nanmedian(q)) if np.isfinite(q).any() else None
          scene[f"{pol}_candidate_dark_fraction"]=float(np.mean(v<=(-15 if pol=="vv" else -20)))
      scene["polarization_valid"]=True
      combo=(im[(id,"vv")]<=-15)&(im[(id,"vh")]<=-20)&mask
      scene["both_polarizations_dark_pixels"]=int(combo.sum())
      scene["both_dark_fraction_of_polygon"]=float(combo.sum()/mask.sum())
    except Exception as ex:scene["error"]=type(ex).__name__+": "+str(ex)[:220]
    all_scenes.append(scene)
    print("PIXEL",dn,date,scene.get("vv_candidate_median_db"),scene.get("vh_candidate_median_db"),flush=True)
  selected_pre=str(c[c.item_id==str(c.iloc[0].item_id)].iloc[0].item_id)
  # Use exactly Phase42 selected pre, never choose by signal.
  original_pre= c[pd.to_datetime(c.datetime,utc=True)==pd.to_datetime(prior.selected_pre_date,utc=True)]
  assert len(original_pre)==1
  preid=str(original_pre.iloc[0].item_id)
  for _,r in c.iterrows():
    id=str(r.item_id)
    if id==preid:continue
    pair={"disasterNumber":dn,"pre_id":preid,"pre_date":dates[preid],
      "comparison_id":id,"comparison_date":dates[id]}
    for pol in ("vv","vh"):
      try:
       a=im[(preid,pol)];b=im[(id,pol)]
       valid=np.isfinite(a)&np.isfinite(b)
       center=(b-a)[mask&valid];back=(b-a)[ring&valid]
       pair[f"{pol}_paired_n"]=int(len(center))
       pair[f"{pol}_polygon_median_delta_db"]=float(np.median(center)) if len(center) else None
       pair[f"{pol}_control_median_delta_db"]=float(np.median(back)) if len(back) else None
       pair[f"{pol}_polygon_fraction_drop3db"]=float((center<=-3).mean()) if len(center) else None
      except Exception as ex:pair[f"{pol}_error"]=type(ex).__name__
    all_pair_diffs.append(pair)
  rows=[r for r in all_scenes if r["disasterNumber"]==dn and r["polarization_valid"]]
  before=[r for r in rows if r["phase"]=="pre_event"]
  during=[r for r in rows if r["phase"]=="incident"]
  after=[r for r in rows if r["phase"]=="post_incident"]
  summary={"disasterNumber":dn,"state":DETAIL[dn]["fema_state"],
     "incident_start":DETAIL[dn]["incident_start"],"incident_end":DETAIL[dn]["official_incident_end"],
     "selected_phase42_pre_date":str(prior.selected_pre_date),
     "selected_phase42_post_date":str(prior.selected_post_date),
     "candidate_km2_phase42":float(prior.candidate_area_km2),
     "candidate_radar_cells":int(mask.sum()),
     "same_orbit_available_scenes":len(c),"raster_scenes_complete":len(rows),
     "pre_scenes":len(before),"during_scenes":len(during),"post_scenes":len(after),
     "pre_both_dark_median":float(np.median([x["both_dark_fraction_of_polygon"] for x in before])) if before else None,
     "during_both_dark_median":float(np.median([x["both_dark_fraction_of_polygon"] for x in during])) if during else None,
     "post_both_dark_median":float(np.median([x["both_dark_fraction_of_polygon"] for x in after])) if after else None,
     "post_event_dark_fraction_all_scenes":[{"date":r["date"],"fraction":r["both_dark_fraction_of_polygon"]} for r in after],
     "first_radar_after_incident_end_days":float((pd.Timestamp(prior.selected_post_date)-pd.Timestamp(DETAIL[dn]["official_incident_end"],tz="UTC")).total_seconds()/86400),
     "point_in_FEMA_designated_county":bool(prior.NOAA_point_inside_current_FEMA_designated_county),
     "NHD_static_hydrography":nhd(geom),
     "official_flood_declaration_source":SRC_URLS[dn],
     "FEMA_event_specific_inundation_polygon_verified":False,
     "candidate_selection_posthoc":True,
     "asof_declaration_safe":False,
     "independent_flood_map_overlap_verified":False,
     "exposed_population_or_buildings_validated":False}
  all_case.append(summary)
  print("SUMMARY",dn,{k:v for k,v in summary.items() if k in ("pre_scenes","during_scenes","post_scenes","pre_both_dark_median","during_both_dark_median","post_both_dark_median")},flush=True)
df=pd.DataFrame(all_scenes);df.to_csv(OUT/"phase43_every_scene_candidate_RTC_backscatter.csv",index=False)
pd.DataFrame(all_pair_diffs).to_csv(OUT/"phase43_pairwise_relative_backscatter_diagnostics.csv",index=False)
(OUT/"phase43_case_audits.json").write_text(json.dumps(all_case,indent=2,default=str))
summary={
 "analyzed_disasters":[4718,4719],
 "scene_counts":{str(q["disasterNumber"]):q["raster_scenes_complete"] for q in all_case},
 "observational_status":"post-selected time-series diagnostics; not independent flood truth",
 "case_summaries":all_case,
 "source_limitations":[
    "Selected polygons were discovered in the Phase42 radar image; follow-up time series of those polygons are diagnostic and subject to selection bias.",
    "Neither source is a ground-truth flooded building/flood extent map, and a static USGS waterway does not establish event-specific inundation.",
    "Maine first radar post scene May 7 occurs after May 1 incident end; attribution to the May Day storm is not proven.",
    "South Dakota cold-to-warm spring transition can produce soil-moisture and snowmelt radar signatures without standing floodwater.",
    "Radar contrast and dual-polarization absolute brightness are not independently trained flood-water detectors.",
    "Only local polygons from NOAA-based sampling, not total disaster-wide flood area.",
    "Post-disaster imagery/features are not FEMA declaration-time safe and do not support T0/T+72 funding decisions.",
    "Frozen accepted Track A model not modified."],
 "number_of_new_independently_verified_disaster_flood_footprints":0,
 "new_funding_models_trained":0,
 "track_A_unchanged":True
}
(OUT/"phase43_summary.json").write_text(json.dumps(summary,indent=2,default=str))
(OUT/"PHASE43_REPORT.md").write_text(
 "# Phase43: Repeat Sentinel-1 optical-failure candidates 4718 and 4719\n\n"
 "This study analyzes every available matched-orbit acquisition at the selected small candidate polygons. "
 "It does **not** assert that candidate radar signatures are verified FEMA flood footprints.\n\n"
 "## Per-date actual radar measurements\n\n"+df[["disasterNumber","date","phase","vv_candidate_median_db","vh_candidate_median_db","both_dark_fraction_of_polygon"]].to_markdown(index=False)
 +"\n\n## Event audit\n\n"+json.dumps(all_case,indent=2,default=str)
 +"\n\n## Limitations\n\n"+"\n".join("- "+x for x in summary["source_limitations"]))
print("PHASE43_RESULT",json.dumps({k:v for k,v in summary.items() if k not in ["case_summaries","source_limitations"]},indent=2),flush=True)
