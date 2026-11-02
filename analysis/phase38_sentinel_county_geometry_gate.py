#!/usr/bin/env python3
"""Phase38: FEMA county containment audit + georeferenced, screened optical water-change candidates.
This is NOT verified flood inundation or verified damage; no model training.
"""
from pathlib import Path
import json, zipfile, hashlib
import numpy as np
import pandas as pd
from shapely.geometry import Point, mapping, shape
from shapely.ops import unary_union, transform
import geopandas as gpd
from pyproj import Transformer
from rasterio.features import shapes
from scipy.ndimage import label
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"audit_outputs"/"phase38"
OUT.mkdir(parents=True,exist_ok=True)
S=ROOT/"audit_inputs"/"phase38"
sys.path.insert(0,str(ROOT/"analysis"))
import phase31_sentinel2_observability_first_screen as p31

COUNTY_URL="https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip"
import requests
r=requests.get(COUNTY_URL,timeout=(30,120),headers={"User-Agent":"academic-source-validation/1.0"})
r.raise_for_status()
raw=OUT/"cb_2023_us_county_500k.zip"; raw.write_bytes(r.content)
county_sha=hashlib.sha256(raw.read_bytes()).hexdigest()
counties=gpd.read_file("zip://"+str(raw))
assert len(counties)>3000 and "GEOID" in counties.columns
counties=counties[["GEOID","NAME","geometry"]].copy()
counties["FIPS"]=pd.to_numeric(counties.GEOID,errors="coerce").astype("Int64")
assert counties.crs is not None
counties=counties.to_crs("EPSG:4326")
geoms={int(row.FIPS):row.geometry for _,row in counties.iterrows() if pd.notna(row.FIPS)}
assert len(geoms)>3000
fema=pd.read_csv(S/"phase34"/"fema_official_designated_areas.csv")
fema["county_fips"]=pd.to_numeric(fema.county_fips,errors="coerce")
fema=fema[(fema.is_strict_county==True)&fema.county_fips.notna()].copy()
design={int(dn):set(g.county_fips.astype(int)) for dn,g in fema.groupby("disasterNumber")}
case92=pd.read_csv(S/"phase26"/"all_92_fema_flood_footprint_source_audit.csv")
assert len(case92)==92
case92["disasterNumber"]=pd.to_numeric(case92.disasterNumber).astype(int)
rows=[]
for _,c in case92.iterrows():
 dn=int(c.disasterNumber);lat=c.noaa_point_lat;lon=c.noaa_point_lon
 fips=None
 if pd.notna(lat) and pd.notna(lon):
  p=Point(float(lon),float(lat))
  for county,geom in geoms.items():
   if geom.covers(p):
    fips=county;break
 assigned=design.get(dn,set())
 rows.append({"disasterNumber":dn,"fyDeclared":int(c.fyDeclared),"state":c.state,
   "NOAA_lat":lat,"NOAA_lon":lon,"point_geolocated":bool(pd.notna(lat) and pd.notna(lon)),
   "2023_census_polygon_point_county":fips,
   "fema_designated_county_count":len(assigned),
   "NOAA_point_in_FEMA_designated_county":bool(fips is not None and fips in assigned),
   "point_county_polygon_available":bool(fips is not None),
   "point_geo_method":"2023_Census_500k_generalized_boundary_point_containment",
   "point_is_flood_footprint":False,"spatial_validation_is_event_attribution":False})
space=pd.DataFrame(rows)
space.to_csv(OUT/"all_92_flood_point_FEMA_designated_county_audit.csv",index=False)
print("SPATIAL POINT STATS",space.groupby(["point_geolocated","NOAA_point_in_FEMA_designated_county"]).size().to_dict(),flush=True)
raw.unlink()

roster=pd.read_csv(S/"phase31"/"selected_scene_screening_roster.csv")
prior=pd.read_csv(S/"phase31"/"case_quality_gate_and_water_change.csv")
assert len(prior)==12 and prior.quality_pass.sum()==6
records=[]; features=[]
for _,rec in prior.sort_values("disasterNumber").iterrows():
 dn=int(rec.disasterNumber)
 row=space.loc[space.disasterNumber==dn].iloc[0]
 item={"disasterNumber":dn,"quality_pass":bool(rec.quality_pass),
       "noaa_point_inside_designated_county":bool(row.NOAA_point_in_FEMA_designated_county),
       "fema_designated_counties":int(row.fema_designated_county_count),
       "point_census_county":int(row["2023_census_polygon_point_county"]) if pd.notna(row["2023_census_polygon_point_county"]) else None,
       "selected_pre_id":rec.selected_pre_id if pd.notna(rec.selected_pre_id) else None,
       "selected_post_id":rec.selected_post_id if pd.notna(rec.selected_post_id) else None,
       "selected_tile":rec.selected_tile if pd.notna(rec.selected_tile) else None,
       "strict_optical_water_change_candidate_not_verified":True,
       "disaster_event_attribution_verified":False,
       "population_exposure_verified":False}
 if not rec.quality_pass:
  item.update({"status":"no_eligible_pair", "observed_pixels":0,"water_change_raw_pixels":None,
               "water_change_strict_pixels":None,"retained_4_pixel_component_pixels":None})
  records.append(item);continue
 g=roster[roster.disasterNumber==dn].set_index("stac_id")
 try:
  pre=g.loc[rec.selected_pre_id];post=g.loc[rec.selected_post_id]
  lat=float(pre.NOAA_report_lat);lon=float(pre.NOAA_report_lon)
  frame=p31.frame_config(pre.scl_url,lon,lat)
  a=p31.read_band(pre.scl_url,frame,"scl")
  b=p31.read_band(post.scl_url,frame,"scl")
  good=np.isin(a,[4,5,6])&np.isin(b,[4,5,6])
  raw_gain=good&(a!=6)&(b==6)
  def ndwi(row):
   green=p31.read_band(row.green_url,frame,"green").astype(np.float32)*float(row.green_scale)+float(row.green_offset)
   nir=p31.read_band(row.nir_url,frame,"nir").astype(np.float32)*float(row.nir_scale)+float(row.nir_offset)
   den=green+nir
   return np.divide(green-nir,den,out=np.full_like(den,np.nan),where=(den>0)&(green>=0)&(nir>=0))
  before=ndwi(pre);after=ndwi(post)
  strict=raw_gain&np.isfinite(before)&np.isfinite(after)&(before<.2)&(after>=.2)
  components,n=label(strict,structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
  keep=np.zeros(strict.shape,dtype=bool)
  for i in range(1,n+1):
   c=(components==i)
   if c.sum()>=4:keep|=c
  assigned=design.get(dn,set())
  polys=[geoms[f] for f in assigned if f in geoms]
  designated=unary_union(polys) if polys else None
  geo_tx=Transformer.from_crs(frame["crs"],"EPSG:4326",always_xy=True)
  projected_m=unary_union([shape(geo) for geo,value in shapes(keep.astype(np.uint8),mask=keep,transform=frame["transform"]) if int(value)==1]) if keep.any() else None
  wgs=transform(geo_tx.transform,projected_m) if projected_m is not None else None
  overlap=wgs.intersection(designated) if wgs is not None and designated is not None else None
  overlap_frac=(overlap.area/wgs.area) if overlap is not None and wgs.area>0 else None
  if wgs is not None and not wgs.is_empty:
   features.append({"type":"Feature","geometry":mapping(wgs),"properties":{
      "disasterNumber":dn,"pre_scene":str(pre.name),"post_scene":str(post.name),
      "candidate_area_km2":round(float(keep.sum())*.0004,7),
      "within_FEMA_designated_polygon_area_fraction":overlap_frac,
      "confidence_label":"FOUR_CONNECTED_PIXELS_MINIMUM; NOT VALIDATED FLOOD"}})
  original=int(rec.candidate_new_water_scl_pixels)
  assert int(raw_gain.sum())==original,(dn,int(raw_gain.sum()),original)
  item.update({"status":"real_rasters_decoded",
   "pre_acquired_utc":str(pre.timestamp),"post_acquired_utc":str(post.timestamp),
   "pre_days_relative_incident_start":float(pre.days_relative),
   "post_days_relative_incident_start":float(post.days_relative),
   "observed_pixels":int(good.sum()),"paired_observed_fraction":float(good.mean()),
   "water_change_raw_pixels":int(raw_gain.sum()),
   "water_change_strict_pixels":int(strict.sum()),
   "retained_4_pixel_component_pixels":int(keep.sum()),
   "four_connected_component_count":int(n),
   "candidate_raw_area_km2":float(raw_gain.sum()*.0004),
   "candidate_strict_area_km2":float(strict.sum()*.0004),
   "candidate_component_area_km2":float(keep.sum()*.0004),
   "component_area_within_designated_counties_fraction":overlap_frac})
  print("IMAGE_CASE",dn,"raw",int(raw_gain.sum()),"strict",int(strict.sum()),"4px",int(keep.sum()),"county",item["noaa_point_inside_designated_county"],flush=True)
 except Exception as ex:
  item["status"]="read_or_projection_error_"+type(ex).__name__
  item["error_details"]=str(ex)[:250]
  print("IMAGE_CASE_FAIL",dn,type(ex).__name__,str(ex)[:250],flush=True)
 records.append(item)

q=pd.DataFrame(records)
q.to_csv(OUT/"12_case_satellite_spatiotemporal_and_quality_audit.csv",index=False)
geojson={"type":"FeatureCollection","features":features}
(OUT/"optical_water_change_candidate_components_MIN4.geojson").write_text(json.dumps(geojson))
report={
 "fema_flood_declarations":92,"noaa_point_geolocated":int(space.point_geolocated.sum()),
 "noaa_points_in_designated_FEMA_counties":int(space.NOAA_point_in_FEMA_designated_county.sum()),
 "points_outside_or_no_admin_county":int((space.point_geolocated&~space.NOAA_point_in_FEMA_designated_county).sum()),
 "satellite_pilot_cases":12,"satellite_quality_pass_cases":6,
 "real_images_re_read":int((q.status=="real_rasters_decoded").sum()),
 "candidate_polygons_with_4_pixel_component_threshold":len(features),
 "cases_with_nonzero_min4_components":int(q.retained_4_pixel_component_pixels.fillna(0).gt(0).sum()),
 "original_candidate_raw_km2":float(q.candidate_raw_area_km2.sum(min_count=1)),
 "strict_NDWI_transition_km2":float(q.candidate_strict_area_km2.sum(min_count=1)),
 "retained_candidate_component_km2":float(q.candidate_component_area_km2.sum(min_count=1)),
 "validated_event_inundation_polygon_count":0,
 "measured_affected_population_count":0,
 "trained_model_count":0,
 "source_county_boundary":"US_Census_2023_1_500000","boundary_sha256":county_sha,
 "fema_current_admin_areas_not_event_footprint":True,
 "source_issues":["FEMA designated counties are current records, and 2023 Census boundaries are generalized; neither is a physical inundation boundary.",
 "All imagery is 4x4 km near NOAA sample points, not whole-event flood extents.",
 "SCL water/NDWI transitions could reflect seasonal, tidal, management, or unrelated surface-water changes.",
 "No permanent-water seasonal baseline, independent inundation truth, or FEMA event causality verified.",
 "No source certified T0/T72 or independent prediction test. Original Track A frozen."]}
(OUT/"summary.json").write_text(json.dumps(report,indent=2))
(OUT/"REPORT.md").write_text("# Phase 38 — Sentinel-2 candidate inundation geometry and FEMA county audit\n\n"
 "The study measures **candidate optical water transitions** and whether NOAA point locations lie within FEMA-designated counties. No validated inundation, damaged properties, affected population, or funding model improvement is claimed.\n\n"
 +pd.DataFrame([report]).drop(columns=["source_issues"]).to_markdown(index=False)
 +"\n\n## Pilot scene audit\n\n"+q[["disasterNumber","quality_pass","noaa_point_inside_designated_county","status","water_change_raw_pixels","water_change_strict_pixels","retained_4_pixel_component_pixels"]].to_markdown(index=False)
 +"\n\n## Limitations\n\n"+"\n".join("- "+v for v in report["source_issues"])+"\n")
print("PHASE38_RESULT",json.dumps(report,indent=2),flush=True)
