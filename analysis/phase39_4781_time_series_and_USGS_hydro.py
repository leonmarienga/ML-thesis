#!/usr/bin/env python3
"""Phase39 reproducible independent evidence tests for FEMA DR4781.
Source artifacts: Phase38 tiny Sentinel-2 candidate geometry; Phase30 public
STAC scene metadata; official USGS NHD waterbody/flowline GIS service.
Neither source proves flood caused by DR4781. Track A untouched.
"""
from pathlib import Path
import json, math, time, hashlib
from urllib.parse import urlencode
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point, shape
from shapely.ops import transform, unary_union
from shapely import distance as shp_distance
from pyproj import Transformer
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.features import rasterize

ROOT=Path(__file__).resolve().parents[1]
IN=ROOT/"audit_inputs"/"phase39"
OUT=ROOT/"audit_outputs"/"phase39"
OUT.mkdir(parents=True,exist_ok=True)
S2=pd.read_csv(IN/"phase30"/"public_stac_scene_metadata.csv")
G=json.loads((IN/"phase38"/"optical_water_change_candidate_components_MIN4.geojson").read_text())
QA=pd.read_csv(IN/"phase38"/"12_case_satellite_spatiotemporal_and_quality_audit.csv")
D=pd.read_csv(IN/"phase38"/"all_92_flood_point_FEMA_designated_county_audit.csv")
p=G["features"][0]
assert p["properties"]["disasterNumber"]==4781
geom=shape(p["geometry"])
assert geom.is_valid and len(geom.geoms)==2
d=QA.loc[QA.disasterNumber==4781].iloc[0]
assert d.quality_pass and d.retained_4_pixel_component_pixels==30
assert d.noaa_point_inside_designated_county
lat=float(D.loc[D.disasterNumber==4781,"NOAA_lat"].iloc[0])
lon=float(D.loc[D.disasterNumber==4781,"NOAA_lon"].iloc[0])
s=S2[S2.disasterNumber==4781].copy()
assert len(s)>=20
s["time"]=pd.to_datetime(s.timestamp,utc=True)
assert s.stac_id.nunique()==len(s)
# Data quality stage: all 2024 available images, no outcome-based image choices.
# 2024 local square is matched to earlier Phase38 square through projected 20m grid.
selected=s.loc[s.stac_id==d.selected_pre_id].iloc[0]
from pyproj import Transformer
from rasterio.transform import from_origin
with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",GDAL_HTTP_TIMEOUT="35"):
    with rasterio.open(selected.scl_url) as src:
        crs=src.crs
        x,y=Transformer.from_crs("EPSG:4326",crs,always_xy=True).transform(lon,lat)
        transform_=from_origin(math.floor((x-2000)/20)*20,math.ceil((y+2000)/20)*20,20,20)
n=200
to_raster=Transformer.from_crs("EPSG:4326",crs,always_xy=True).transform
localgeom=transform(to_raster,geom)
roi=rasterize([(localgeom,1)],out_shape=(n,n),transform=transform_,fill=0,dtype="uint8").astype(bool)
assert int(roi.sum())==30,("candidate pixel count moved",int(roi.sum()))
envelope=(geom.bounds[0]-.01,geom.bounds[1]-.01,geom.bounds[2]+.01,geom.bounds[3]+.01)

def read_band(url,mode="scl"):
    from urllib.parse import urlparse
    parsed=urlparse(str(url))
    if parsed.scheme!="https" or parsed.hostname!="e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com":
        raise ValueError("Unapproved host "+str(parsed.hostname))
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",GDAL_HTTP_TIMEOUT="35",
                      GDAL_HTTP_MAX_RETRY="2",GDAL_HTTP_RETRY_DELAY="2"):
        with rasterio.open(url) as source:
            with WarpedVRT(source,crs=crs,transform=transform_,width=n,height=n,
                       resampling=Resampling.nearest if mode=="scl" else Resampling.bilinear) as vrt:
                return vrt.read(1)
records=[]
for _,item in s.sort_values("time").iterrows():
    rec={"disasterNumber":4781,"stac_id":str(item.stac_id),
         "time":str(item.timestamp),"days_relative":float(item.days_relative),
         "phase":str(item.phase),"catalog_cloud_pct":float(item.cloud_cover),
         "pixels_in_candidate_polygon":int(roi.sum())}
    try:
        arr=read_band(item.scl_url,"scl")
        clear=np.isin(arr,[4,5,6])
        roi_valid=clear&roi
        rec.update({"scl_read_ok":True,"scl_roi_clear_n":int(roi_valid.sum()),
              "scl_roi_clear_frac":float(roi_valid.sum()/roi.sum()),
              "scl_roi_water6_n":int(((arr==6)&roi).sum()),
              "scl_roi_water6_among_clear_frac":float(((arr==6)&roi_valid).sum()/roi_valid.sum()) if roi_valid.any() else None,
              "scl_local_clear_frac":float(clear.mean())})
        # Only read green/NIR for candidate pixels with 70%+ direct clear coverage.
        if roi_valid.sum()>=int(np.ceil(.7*roi.sum())):
            green=read_band(item.green_url,"spectrum").astype(np.float32)*float(item.green_scale)+float(item.green_offset)
            nir=read_band(item.nir_url,"spectrum").astype(np.float32)*float(item.nir_scale)+float(item.nir_offset)
            denom=green+nir
            ndwi=np.divide(green-nir,denom,out=np.full_like(denom,np.nan),where=(denom>0)&(green>=0)&(nir>=0))
            ndwi_valid=roi_valid&np.isfinite(ndwi)
            rec.update({"ndwi_read_ok":True,"ndwi_roi_valid_n":int(ndwi_valid.sum()),
             "ndwi_roi_median":float(np.nanmedian(ndwi[ndwi_valid])) if ndwi_valid.any() else None,
             "ndwi_roi_water_gt_0p2_n":int((ndwi_valid&(ndwi>=.2)).sum()),
             "ndwi_roi_water_frac":float((ndwi_valid&(ndwi>=.2)).sum()/ndwi_valid.sum()) if ndwi_valid.any() else None})
        else:rec["ndwi_read_ok"]=False
    except Exception as exc:
        rec.update({"scl_read_ok":False,"ndwi_read_ok":False,"error":type(exc).__name__+": "+str(exc)[:180]})
    records.append(rec)
    print("SCENE",rec["time"][:10],"clear",rec.get("scl_roi_clear_n","error"),
         "SCLw",rec.get("scl_roi_water6_n","na"),"NDWI",rec.get("ndwi_roi_water_frac"),flush=True)
ts=pd.DataFrame(records)
ts.to_csv(OUT/"4781_all_scene_candidate_pixel_time_series.csv",index=False)
assert len(ts)==len(s)
# Strictly independent static surface water comparator, not event-time data.
# USA USGS NHD service waterbody/flowline categories. Query against small envelope.
base="https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/"
session=requests.Session()
session.headers["User-Agent"]="university-flood-remote-sensing-methodology-audit"
nhd=[]
for layer,name in [(12,"waterbody"),(9,"water_area"),(6,"flowline")]:
    opts={"f":"geojson","where":"1=1","geometryType":"esriGeometryEnvelope",
      "geometry":",".join(str(x) for x in envelope),"inSR":"4326","outSR":"4326",
      "spatialRel":"esriSpatialRelIntersects","returnGeometry":"true","outFields":"*"}
    row={"layer_id":layer,"type":name,"service":base+str(layer)+"/query",
      "query_params":opts,"request_ok":False,"feature_count":0}
    try:
        rr=session.get(base+str(layer)+"/query",params=opts,timeout=75)
        rr.raise_for_status()
        content=rr.json()
        if "error" in content:raise ValueError(str(content["error"])[:250])
        if content.get("type")!="FeatureCollection":raise ValueError("GeoJSON unavailable "+str(content)[:100])
        features=content.get("features",[])
        row.update({"request_ok":True,"feature_count":len(features),
                "feature_ids":[str(x.get("id",x.get("properties",{}).get("NHDPlusID",""))) for x in features][:50]})
        geos=[shape(f["geometry"]) for f in features if f.get("geometry")]
        if geos:
            united=unary_union(geos)
            roi_projected=transform(Transformer.from_crs("EPSG:4326",crs,always_xy=True).transform,geom)
            out_projected=transform(Transformer.from_crs("EPSG:4326",crs,always_xy=True).transform,united)
            row["overlap_candidate_km2"]=float(roi_projected.intersection(out_projected).area/1e6)
            row["nearest_distance_m"]=float(roi_projected.distance(out_projected))
            row["feature_properties_sample"]=[str(f.get("properties",{}))[:450] for f in features[:5]]
        # Only derived intersections are stored; not redistributing raw NHD geometries.
    except Exception as exc:row["error"]=type(exc).__name__+": "+str(exc)[:220]
    nhd.append(row)
    print("NHD",name,row.get("feature_count"),row.get("overlap_candidate_km2"),row.get("error"),flush=True)
(OUT/"USGS_NHD_independent_static_waterway_intersection.json").write_text(json.dumps(nhd,indent=2))
# Time-series support is only for 2024 April 26 response; 2024 may have later floods.
q=ts[ts.scl_read_ok==True]
e=q[q.ndwi_read_ok==True]
pre=e[e.phase=="pre"];post=e[e.phase=="post"]
def median_safe(col,df):
 z=pd.to_numeric(df[col],errors="coerce").dropna()
 return float(z.median()) if len(z) else None
# Baseline BEFORE Apr 26 explicitly, including scene farther from May 2024.
baseline_counts=int((pre.ndwi_roi_water_frac>.5).sum())
post_counts=int((post.ndwi_roi_water_frac>.5).sum())
record={
  "case":"FEMA-4781-DR-TX",
  "fema_incident_start":"2024-04-26",
  "fema_declaration_date":"2024-05-17",
  "source_satellite_noaa_point":{"lat":lat,"lon":lon},
  "component_geometry_centroid_lon_lat":[geom.centroid.x,geom.centroid.y],
  "candidate_component_pixels":int(roi.sum()),
  "candidate_component_area_km2":.012,
  "candidate_component_units":"Optical water-change pixels; not flood inundation",
  "scenes_in_stac_2024":len(s),
  "scenes_scl_read_ok":len(q),
  "scenes_roi_sufficient_clear_ge70pct":len(e),
  "eligible_pre_event_scenes":len(pre),
  "eligible_post_event_scenes":len(post),
  "pre_ndwi_water_fraction_median":median_safe("ndwi_roi_water_frac",pre),
  "post_ndwi_water_fraction_median":median_safe("ndwi_roi_water_frac",post),
  "pre_scenes_majority_water":baseline_counts,
  "post_scenes_majority_water":post_counts,
  "USGS_NHD_layer_results":nhd,
  "event_specific_independent_inundation_verified":False,
  "satellite_source_quality_not_event_causality":True,
  "flooded_buildings_or_population_not_assessed":True,
  "model_trained":False,
  "frozen_track_A_modified":False,
  "source_PDA":"https://www.fema.gov/sites/default/files/documents/PDAReport_FEMA4781DR-TX.pdf",
  "source_PDA_scope_note":"1,840 impacted residences is statewide/requested-area disaster assessment, not this 0.012 km² patch",
  "limitations":[
   "USGS hydrography is a mapped static surface-water reference; NHD data are not 2024 water-level measurements.",
   "Sentinel-2 repeated optical samples test persistence but cannot alone attribute a change to a FEMA declaration.",
   "April-June 2024 disaster contains multiple storms; after Apr 26 does not uniquely identify a flood pulse.",
   "SCL/cloud exclusion makes missing pixels NOT observed as dry.",
   "FEMA statewide multi-hazard PDA estimates cannot be geocoded to the tiny candidate patch.",
   "Preliminary satellite search limited to available Phase30 March-May 2024 Sentinel-2 scenes.",
   "No independent flood polygon was retrieved; no claim of validated flood footprint or economic exposure."]}
(OUT/"phase39_summary.json").write_text(json.dumps(record,indent=2))
(OUT/"PHASE39_REPORT.md").write_text(
 "# Phase 39 — FEMA 4781 physical flood-candidate validation\n\n"
 "Independent checks: repeat Sentinel-2 acquisitions at the exact 30 water-change pixels and USGS static hydrography intersection. "
 "This is **not an event-specific inundation certification**.\n\n"
 "## Quantitative results\n\n"
 +json.dumps({k:v for k,v in record.items() if k not in ["USGS_NHD_layer_results","limitations"]},indent=2)
 +"\n\n## Hydrography\n\n"+json.dumps(nhd,indent=2)
 +"\n\n## Caveats\n\n"+"\n".join("- "+v for v in record["limitations"]))
print("PHASE39_SUMMARY",json.dumps({k:v for k,v in record.items() if k not in ["USGS_NHD_layer_results","limitations"]},indent=2),flush=True)
