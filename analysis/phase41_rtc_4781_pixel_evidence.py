#!/usr/bin/env python3
"""Phase41: real Sentinel-1 RTC backscatter at frozen FEMA 4781 optical-change polygon.

Use fixed same-orbit acquisitions chosen by date (not by observed backscatter).
Correct signed MPC assets; 10m stable UTM grid; inspect VV/VH in linear->dB
and an annular local control. Independent historic JRC occurrence is optional
and not a 2024 flood-map label. No funding model trained.
"""
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from rasterio.transform import from_origin
from rasterio.features import rasterize
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from shapely.geometry import shape, mapping
from shapely.ops import transform as geom_transform
from pyproj import Transformer
import rasterio, planetary_computer as pc
import numpy as np
import pandas as pd
import json, math, hashlib

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/"audit_inputs"/"phase41"
OUT=ROOT/"audit_outputs"/"phase41"
OUT.mkdir(parents=True,exist_ok=True)
G=json.loads((INPUT/"phase38"/"optical_water_change_candidate_components_MIN4.geojson").read_text())
assert len(G["features"])==1
geom=shape(G["features"][0]["geometry"])
assert G["features"][0]["properties"]["disasterNumber"]==4781
SOURCE=json.loads((INPUT/"phase40"/"4781_RTC_radar_assets_read_only.json").read_text())
JRC=json.loads((INPUT/"phase40"/"4781_JRC_global_surface_water_STAC_feasibility.json").read_text())
SCENES=pd.read_csv(INPUT/"phase40"/"all_public_sentinel1_RTC_STAC_scene_metadata.csv")
assert SCENES[SCENES.disasterNumber==4781].shape[0]>=20

# Orbit-34 series has observations around the Apr 26 event, same geometry/orbit
# and polarization. Predeclare dates before inspecting any raster pixel values.
DATES=["2024-04-06","2024-04-18","2024-04-30","2024-05-12","2024-05-24"]
ORBIT="34:ascending"
candidate=SCENES[(SCENES.disasterNumber==4781)&(SCENES.orbit_key==ORBIT)].copy()
candidate["day"]=candidate.datetime.astype(str).str[:10]
assert all(int((candidate.day==d).sum())==1 for d in DATES)
IDs={day:str(candidate.loc[candidate.day==day,"item_id"].iloc[0]) for day in DATES}
assets={(a["item_id"],a["asset"]):a for a in SOURCE}
assert all((IDs[day],pol) in assets for day in DATES for pol in ("vv","vh"))
def signed(url):
    u=urlparse(url)
    if u.scheme!="https" or u.hostname not in [
        "sentinel1euwestrtc.blob.core.windows.net",
        "ai4edataeuwest.blob.core.windows.net"]:
        raise ValueError("Unexpected imagery asset host")
    return pc.sign(url)

def config(href):
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",GDAL_HTTP_TIMEOUT="50",GDAL_HTTP_MAX_RETRY="2"):
      with rasterio.open(href) as src:
        if src.crs is None:raise RuntimeError("RTC asset has no projected CRS")
        if not src.crs.is_projected:raise RuntimeError("RTC asset not projected")
        crs=src.crs
    lng,lat=float(geom.centroid.x),float(geom.centroid.y)
    xx,yy=Transformer.from_crs("EPSG:4326",crs,always_xy=True).transform(lng,lat)
    # Fixed 2km square, 10m output cells; 200x200 raster
    pixel=10.0;width=height=200
    left=math.floor((xx-1000)/pixel)*pixel
    top=math.ceil((yy+1000)/pixel)*pixel
    return crs,from_origin(left,top,pixel,pixel),width,height
auth0=signed(assets[(IDs[DATES[0]],"vv")]["href"])
CRS,TRANSFORM,W,H=config(auth0)
polygon_native=geom_transform(Transformer.from_crs("EPSG:4326",CRS,always_xy=True).transform,geom)
patch=rasterize([(mapping(polygon_native),1)],out_shape=(H,W),transform=TRANSFORM,
   fill=0,dtype=np.uint8,all_touched=False).astype(bool)
assert 50<=int(patch.sum())<=180,("10m pixel patch mismatch",int(patch.sum()))
ix,iy=np.meshgrid(np.arange(W),np.arange(H))
cx,cy=~0,~0
longitude,latitude=geom.centroid.x,geom.centroid.y
mx,my=Transformer.from_crs("EPSG:4326",CRS,always_xy=True).transform(longitude,latitude)
xs=TRANSFORM.c+(ix+.5)*TRANSFORM.a
ys=TRANSFORM.f+(iy+.5)*TRANSFORM.e
dist=np.hypot(xs-mx,ys-my)
control=(dist>=250)&(dist<=700)
assert control.sum()>5000
def read_tif(url):
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",GDAL_HTTP_TIMEOUT="55",
      GDAL_HTTP_MAX_RETRY="3",GDAL_HTTP_RETRY_DELAY="2",CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tiff,.tif",
      GDAL_HTTP_MULTIRANGE="YES"):
      with rasterio.open(url) as src:
        if src.crs is None:raise RuntimeError("Not georeferenced")
        with WarpedVRT(src,crs=CRS,transform=TRANSFORM,width=W,height=H,
           resampling=Resampling.bilinear,nodata=src.nodata) as vrt:
            raw=vrt.read(1,masked=True).astype(np.float64)
    z=raw.filled(np.nan)
    z[(z<=0)|(~np.isfinite(z))]=np.nan
    db=np.full(z.shape,np.nan,dtype=float)
    np.log10(z,out=db,where=(z>0))
    db*=10
    db[~np.isfinite(z)]=np.nan
    return db
arr={}
details=[]
asset_access=[]
for day in DATES:
    for pol in ("vv","vh"):
        info=assets[(IDs[day],pol)]
        url=info["href"]
        entry={"day":day,"pol":pol,"item_id":IDs[day],
            "orbit":ORBIT,"dataset":"planetary_computer_sentinel_1_rtc",
            "asset_href_without_signed_token":url}
        try:
            img=read_tif(signed(url))
            v=img[patch];bg=img[control]
            valid=np.isfinite(v);control_valid=np.isfinite(bg)
            arr[(day,pol)]=img
            entry.update({"read_status":"pixel_data_read",
                "polygon_pixels":int(len(v)),"polygon_valid":int(valid.sum()),
                "polygon_valid_frac":float(valid.mean()),
                "polygon_db_median":float(np.median(v[valid])) if valid.any() else None,
                "polygon_db_p10":float(np.percentile(v[valid],10)) if valid.any() else None,
                "polygon_db_p90":float(np.percentile(v[valid],90)) if valid.any() else None,
                "local_control_valid":int(control_valid.sum()),
                "control_db_median":float(np.median(bg[control_valid])) if control_valid.any() else None})
            print("RASTER",day,pol,entry["polygon_valid"],entry["polygon_db_median"],flush=True)
        except Exception as exc:
            entry.update({"read_status":"read_failed","failure_type":type(exc).__name__,
                          "failure":str(exc)[:260]})
            print("RASTER_FAILED",day,pol,entry["failure"],flush=True)
        details.append(entry)
pd.DataFrame(details).to_csv(OUT/"4781_radar_10m_polygon_backscatter_by_date.csv",index=False)
pairs=[]
for pol in ("vv","vh"):
    prev=arr.get(("2024-04-18",pol))
    if prev is None:continue
    for day in DATES:
      if day=="2024-04-18" or (day,pol) not in arr:continue
      aft=arr[(day,pol)]
      valid=np.isfinite(prev)&np.isfinite(aft)
      vp=valid&patch
      bg=valid&control
      diff=aft-prev
      p=diff[vp];b=diff[bg]
      if not len(p):continue
      row={"before_day":"2024-04-18","after_day":day,"pol":pol,"orbit":ORBIT,
           "valid_polygon_pixels":int(len(p)),"valid_control_pixels":int(len(b)),
           "polygon_median_delta_db":float(np.median(p)),
           "polygon_mean_delta_db":float(np.mean(p)),
           "polygon_fraction_drop_at_least_3db":float((p<=-3).mean()),
           "polygon_fraction_increase_at_least_3db":float((p>=3).mean()),
           "control_median_delta_db":float(np.median(b)) if len(b) else None,
           "difference_from_local_control_median_db":float(np.median(p)-np.median(b)) if len(b) else None,
           "no_event_flood_label":True}
      pairs.append(row)
pd.DataFrame(pairs).to_csv(OUT/"4781_same_orbit_pairwise_radar_changes.csv",index=False)
# JRC "occurrence" covers 1984-2020 history; relevant to permanence but no 2024
# event attribution. Signed URLs replace prior Phase40 HTTP-409 unsigned failures.
jrc=[]
if JRC.get("status")=="success":
    for layer in ("occurrence","seasonality"):
      try:
        href=JRC["items"][0]["asset_hrefs"][layer]
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",GDAL_HTTP_TIMEOUT="55",GDAL_HTTP_MAX_RETRY="2"):
          with rasterio.open(signed(href)) as ds:
            with WarpedVRT(ds,crs=CRS,transform=TRANSFORM,width=W,height=H,
                resampling=Resampling.nearest) as vrt:
              im=vrt.read(1,masked=True)
        v=im.filled(255)[patch]
        v=v[(v>=0)&(v<250)]
        vals,counts=np.unique(v,return_counts=True)
        jrc.append({"layer":layer,"status":"read_ok","candidate_pixels_with_reference":int(len(v)),
         "histogram":{str(int(k)):int(count) for k,count in zip(vals,counts)},
         "median":float(np.median(v)) if len(v) else None,
         "source_url_unsigned":href})
      except Exception as e:
        jrc.append({"layer":layer,"status":"failed","error":type(e).__name__+": "+str(e)[:250]})
(OUT/"4781_jrc_historical_water_pixel_reference.json").write_text(json.dumps(jrc,indent=2))
successful=sum(z["read_status"]=="pixel_data_read" for z in details)
summary={"disasterNumber":4781,"frozen_optical_patch_km2":.012,
 "calibrated_rtc_real_pixel_read_count":successful,
 "attempted_polarization_scenes":len(DATES)*2,
 "dates_with_complete_two_polarizations":[d for d in DATES if
    (d,"vv") in arr and (d,"vh") in arr],
 "matched_orbit":ORBIT,"fixed_post_event_date":"2024-04-30",
 "raster_crs":str(CRS),"radar_pixel_size_m":10,"patch_radar_pixels":int(patch.sum()),
 "control_pixels":int(control.sum()),"jrc_reference":jrc,
 "pairwise":pairs,
 "SAR_backscatter_is_not_a_verified_flood_label":True,
 "independent_event_specific_inundation_verified":False,
 "asof_declaration_usable":False,
 "funding_model_trained":False,"track_A_changed":False,
 "restrictions":[
  "SAR brightness change can reflect ground moisture, roughness, urban double-bounce and vegetation, not just floodwater.",
  "Fixed 30-pixel 20m optical candidate is small; regridding to 10m radar does not increase actual independent ground truth.",
  "JRC historical water is not 2024 flood validation and does not measure disaster impacts.",
  "Same ascending orbit mitigates but does not eliminate incidence, geometry and calibration differences.",
  "This is a post hoc candidate selected via optical water change; it is not a blind test of radar flood identification.",
  "Do not label as FEMA event-specific inundation without independent hydrological or mapped footprint comparison.",
  "No original Track A training, prediction or acceptance criteria modified."]}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
(OUT/"REPORT.md").write_text("# Phase 41 — Sentinel-1 RTC real backscatter at FEMA DR4781\n\n"
 "This is a read of actual raster pixel backscatter, not just STAC metadata.\n\n"
 "## Fixed acquisition days and measurements\n\n"+pd.DataFrame(details).to_markdown(index=False)
 +"\n\n## Paired changes\n\n"+pd.DataFrame(pairs).to_markdown(index=False)
 +"\n\n## Interpretive limits\n\n"+"\n".join("- "+x for x in summary["restrictions"]))
print("PHASE41_OUTPUT",json.dumps({k:v for k,v in summary.items() if k not in ["pairwise","restrictions","jrc_reference"]},indent=2),flush=True)
