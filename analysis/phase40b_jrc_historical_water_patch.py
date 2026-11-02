#!/usr/bin/env python3
"""Phase40b JRC historical water reference at exact Phase38 polygon;
not an event-specific flood map. Reads real geospatial COG pixels remotely.
No funding model trained. Original 2024 Sentinel-2 candidate geometry is frozen.
"""
from pathlib import Path
import json
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from shapely.geometry import shape
from pyproj import Transformer
from shapely.ops import transform

ROOT=Path(__file__).resolve().parents[1]
IN=ROOT/"audit_inputs"/"phase40b"
OUT=ROOT/"audit_outputs"/"phase40b"
OUT.mkdir(parents=True,exist_ok=True)
geo=json.loads((IN/"phase38"/"optical_water_change_candidate_components_MIN4.geojson").read_text())
assert len(geo["features"])==1
polygon=shape(geo["features"][0]["geometry"])
data=json.loads((IN/"phase40"/"4781_JRC_global_surface_water_STAC_feasibility.json").read_text())
assert data["status"]=="success" and len(data["items"])==1
urls=data["items"][0]["asset_hrefs"]
results={"disasterNumber":4781,"JRC_period":"1984-2020","candidate_optical_footprint_km2":.012,
 "reference_pixel_values":[],"independent_2024_inundation_validated":False,
 "reported_exposure_certified":False,"track_A_modified":False}
for name in ["occurrence","seasonality"]:
 try:
  uri=urls[name]
  assert uri.startswith("https://ai4edataeuwest.blob.core.windows.net/jrcglobalwater/")
  with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",GDAL_HTTP_TIMEOUT="45",GDAL_HTTP_MAX_RETRY="2"):
   with rasterio.open(uri) as data:
    from rasterio.warp import transform_bounds
    bounds=transform_bounds("EPSG:4326",data.crs,*polygon.bounds,densify_pts=5)
    window=from_bounds(*bounds,transform=data.transform).round_offsets().round_lengths()
    window=window.round_lengths()
    from rasterio.windows import Window
    # Pad 5 pixels each direction; clamp to imagery extent.
    window=Window(max(0,int(window.col_off)-5),max(0,int(window.row_off)-5),
                  max(1,int(window.width)+10),max(1,int(window.height)+10))
    window=window.intersection(Window(0,0,data.width,data.height))
    values=data.read(1,window=window,masked=True)
    tf=data.window_transform(window)
    geom_raster=transform(Transformer.from_crs("EPSG:4326",data.crs,always_xy=True).transform,polygon)
    inside=geometry_mask([geom_raster],out_shape=values.shape,transform=tf,invert=True,all_touched=False)
    inside_values=values.data[inside&~np.ma.getmaskarray(values)]
    if not len(inside_values):raise RuntimeError("No JRC pixels within candidate polygon")
    vals,counts=np.unique(inside_values,return_counts=True)
    sample={str(int(v)):int(c) for v,c in zip(vals,counts)}
    item={"layer":name,"status":"read_ok","image_crs":str(data.crs),
      "candidate_pixel_n":int(len(inside_values)),"pixel_value_histogram":sample,
      "minimum":int(inside_values.min()),"maximum":int(inside_values.max()),
      "median":float(np.median(inside_values)),
      "source_uri":uri}
    results["reference_pixel_values"].append(item)
    print("JRC_LAYER",name,item,flush=True)
 except Exception as exc:
  msg=type(exc).__name__+": "+str(exc)[:350]
  results["reference_pixel_values"].append({"layer":name,"status":"unavailable","error":msg})
  print("JRC_FAILED",name,msg,flush=True)
(OUT/"phase40b_jrc_hist_water_at_4781_candidate.json").write_text(json.dumps(results,indent=2))
(OUT/"REPORT.md").write_text("# Phase40b — Historical water-occurrence reference at FEMA 4781 optical candidate\n\n"+
 json.dumps(results,indent=2)+"\n\nJRC historical surface water is not 2024 inundation ground truth. No complete flood footprint, buildings, or population estimated. No Track A changes.\n")
