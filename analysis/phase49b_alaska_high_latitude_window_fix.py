#!/usr/bin/env python3
"""Phase49 targeted source-window correction for Alaska DR4730.

Other Phase49 cases and sampled sites remain frozen. The original Phase49
window ceiling of 260 raster columns excluded valid 4km Alaska samples due
to longitude convergence at high latitude. Permit up to 500x500 cells;
do not change satellite or geometry. Neither water signal nor funding
determines correction. Outputs are supplementary raw audit for merging.
"""
from pathlib import Path
import math,json
import pandas as pd,numpy as np,rasterio
from shapely.geometry import shape,mapping
from rasterio.windows import from_bounds,Window
from rasterio.features import geometry_mask
from pyproj import Geod
ROOT=Path(__file__).resolve().parents[1]
I=ROOT/"audit_inputs"/"phase49b"
OUT=ROOT/"audit_outputs"/"phase49b";OUT.mkdir(parents=True,exist_ok=True)
RA=pd.read_csv(I/"phase49"/"phase49_predeclared_36_disaster_county_sampling_roster.csv")
G=json.loads((I/"phase49"/"phase49_county_windows_2023_2024.geojson").read_text())
polys={(int(g["properties"]["disasterNumber"]),int(g["properties"]["county_fips"]),int(g["properties"]["site_index"])):shape(g["geometry"]) for g in G["features"]}
r=RA[RA.disasterNumber==4730].copy()
assert len(r)==6 and len(set(r.county_fips))==2
SOURCE="https://data.source.coop/epoch/jrc-global-surface-water-monthly/v1_5"
GEOD=Geod(ellps="WGS84")
def tile(lon,lat):
 a=math.floor(lon/10)*10;b=math.ceil(lat/10)*10
 return f"{abs(a)}{'W' if a<0 else 'E'}_{abs(b)}{'S' if b<0 else 'N'}"
def uri(t,ym):
 y,m=ym.split("-")
 return f"{SOURCE}/{y}/{y}_{m}/monthlyhistory_{t}_v1_5_{y}_{m}.tif"
obs=[];paired=[]
for _,z in r.sort_values(["county_fips","site_index"]).iterrows():
 fips=int(z.county_fips);site=int(z.site_index);dn=int(z.disasterNumber)
 geom=polys[(dn,fips,site)]
 sample_id=f"{dn}_{fips}_{site}"
 t=tile(z.anchor_lon,z.anchor_lat)
 dates={"baseline":z.prior_year_same_month,"event":z.event_month,"following":z.following_month}
 arrays={};t_ref=None;mask_ref=None;pixarea=None
 for phase,ym in dates.items():
  row={"sample_id":sample_id,"disasterNumber":dn,"county_fips":fips,"site_index":site,
   "pilot_disaster":False,"source_url":uri(t,ym),"jrc_month":ym,"phase":phase,
   "status":"unattempted","site_area_km2":z.window_area_km2}
  try:
   with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",GDAL_HTTP_TIMEOUT="55",
    GDAL_HTTP_MAX_RETRY="2",GDAL_HTTP_MULTIRANGE="YES"):
    with rasterio.open(row["source_url"]) as src:
     assert str(src.crs)=="EPSG:4326"
     win=from_bounds(*geom.bounds,transform=src.transform).round_offsets().round_lengths()
     win=win.intersection(Window(0,0,src.width,src.height))
     if win.width>500 or win.height>500:raise RuntimeError("source_window_exceeds_500")
     mat=src.read(1,window=win)
     tf=src.window_transform(win)
     mask=geometry_mask([mapping(geom)],out_shape=mat.shape,transform=tf,invert=True,all_touched=False)
     if not mask.any():raise RuntimeError("empty_polygon_mask")
     if t_ref is not None and (tf!=t_ref or not np.array_equal(mask,mask_ref)):raise RuntimeError("unaligned_monthly_grid")
     t_ref=tf;mask_ref=mask
     values=mat[mask]
     assert np.isin(values,[0,1,2]).all()
     arrays[phase]=values
     dx,dy=map(abs,src.res)
     from shapely.geometry import box
     bx=box(z.anchor_lon-dx/2,z.anchor_lat-dy/2,z.anchor_lon+dx/2,z.anchor_lat+dy/2)
     pixarea=abs(GEOD.geometry_area_perimeter(bx)[0])/1e6
     present=(values>0)
     row.update({"status":"read_ok","sample_pixels":len(values),
       "observed_pixels":int(present.sum()),"unobserved_pixels":int((values==0).sum()),
       "water_pixels":int((values==2).sum()),"dry_pixels":int((values==1).sum()),
       "observation_fraction":float(present.mean()),
       "water_fraction_of_observed":float((values==2).sum()/present.sum()) if present.sum() else None,
       "pixel_area_km2":pixarea})
  except Exception as e:row.update({"status":"read_failed","error":type(e).__name__+": "+str(e)[:230]})
  obs.append(row)
  print("ALASKA_MONTH",sample_id,phase,row["status"],row.get("water_pixels"),flush=True)
 for comp in ("event","following"):
  item={"sample_id":sample_id,"disasterNumber":dn,"county_fips":fips,"site_index":site,
        "event_month":z.event_month,"comparison_phase":comp,"status":"missing_source_month",
        "site_area_km2":float(z.window_area_km2),
        "measured_independently_from_radar_NOAA_or_dollar_targets":True,
        "observed_water_change_is_not_verified_flood":True}
  if "baseline" in arrays and comp in arrays:
   a,b=arrays["baseline"],arrays[comp];v=(a>0)&(b>0)
   gain=((a==1)&(b==2));loss=((a==2)&(b==1))
   item.update({"status":"paired_read_ok","pixel_area_km2_approx":pixarea,
     "both_observed_n":int(v.sum()),"both_observed_frac_of_site":float(v.mean()),
     "new_water_cells_prior_dry_event_water":int(gain.sum()),
     "disappeared_water_cells_prior_water_event_dry":int(loss.sum()),
     "new_water_sampled_km2_approx":float(gain.sum()*pixarea),
     "water_loss_sampled_km2_approx":float(loss.sum()*pixarea),
     "only_unobserved_is_NOT_dry":True})
  paired.append(item)
pd.DataFrame(obs).to_csv(OUT/"phase49b_Alaska_monthly_source_window_correction.csv",index=False)
pd.DataFrame(paired).to_csv(OUT/"phase49b_Alaska_paired_source_window_correction.csv",index=False)
summary={"disasterNumber":4730,"corrected_sites":6,"monthly_reads_successful":sum(q["status"]=="read_ok" for q in obs),
 "event_paired_sites":sum(q["status"]=="paired_read_ok" and q["comparison_phase"]=="event" for q in paired),
 "source_bug":"Original >260 raster-column guard unsuitable for 4km high-latitude Alaska windows; corrected to 500.",
 "selection_is_identical_to_phase49":True,
 "retraining_performed":False,"track_A_changed":False}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2))
print("PHASE49B_RESULT",json.dumps(summary),flush=True)
