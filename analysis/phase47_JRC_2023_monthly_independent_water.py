#!/usr/bin/env python3
"""Phase47 JRC Global Surface Water v1.5 independent optical-water monthly test.

JRC/European Commission Landsat C2 2022-2024 monthly maps, redistributed
unchanged at source.coop. Read actual 30m geotiff pixels at fixed Phase42 radar
polygons for DR4718 and DR4719. Source class 0 UNOBSERVED; 1 NOT WATER;
2 WATER. Never treat 0 as dry. 2022 and 2024 same-season negative controls.
Monthly evidence cannot date an April 10 or May 7 event. Feature sites are
radar-selected post hoc. No disaster-level inundation or model changes.
"""
from pathlib import Path
from urllib.parse import urlparse
from collections import Counter
import json,math
import pandas as pd,numpy as np,rasterio
from shapely.geometry import shape,mapping
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds,Window
from shapely.ops import transform as shapely_transform
from pyproj import Transformer

ROOT=Path(__file__).resolve().parents[1]
IN=ROOT/"audit_inputs"/"phase47"/"phase42"
OUT=ROOT/"audit_outputs"/"phase47"
OUT.mkdir(parents=True,exist_ok=True)
dataset=json.loads((IN/"radar_water_change_cANDIDATES_unverified.geojson").read_text())
assert len(dataset["features"])==2
P={int(v["properties"]["disasterNumber"]):shape(v["geometry"]) for v in dataset["features"]}
assert set(P)=={4718,4719}
EVENT={
 4718:{"state":"SD","start":"2023-04-09","end":"2023-04-30","radar_dark":"2023-04-10",
       "radar_area_km2":.0043},
 4719:{"state":"ME","start":"2023-04-30","end":"2023-05-01","radar_dark":"2023-05-07",
       "radar_area_km2":.0041}}
DATES=["2022-04","2022-05","2023-03","2023-04","2023-05","2023-06","2024-04","2024-05"]
BASE="https://data.source.coop/epoch/jrc-global-surface-water-monthly/v1_5"
def coordtag(lon,lat):
   a=math.floor(lon/10)*10
   b=math.ceil(lat/10)*10
   lon_tag=str(abs(a))+("W" if a<0 else "E")
   lat_tag=str(abs(b))+("S" if b<0 else "N")
   return f"{lon_tag}_{lat_tag}"
def link(tag,yearmonth):
   year,month=yearmonth.split("-")
   return f"{BASE}/{year}/{year}_{month}/monthlyhistory_{tag}_v1_5_{year}_{month}.tif"

rows=[];request_audit=[]
for dn in [4718,4719]:
    poly=P[dn]
    tag=coordtag(poly.centroid.x,poly.centroid.y)
    assert tag==("100W_50N" if dn==4718 else "80W_50N"),(dn,tag)
    for ym in DATES:
        source=link(tag,ym)
        assert urlparse(source).hostname=="data.source.coop"
        row={"disasterNumber":dn,"state":EVENT[dn]["state"],"month":ym,
             "source_url":source,"tile":tag,"status":"unattempted",
             "sensor":"Landsat_Collection2_JRC_GSW_v1_5",
             "water_class_0_no_observation":True,
             "entire_radar_candidate_area_km2":EVENT[dn]["radar_area_km2"],
             "radar_selected_polygon":True,
             "monthly_date_not_daily_event":True}
        try:
            # 0.004 degrees lat/lon spans 200-450m, adequate local controls.
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                 CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
                 GDAL_HTTP_MAX_RETRY="2",GDAL_HTTP_TIMEOUT="40",
                 GDAL_HTTP_MULTIRANGE="YES"):
              with rasterio.open(source) as src:
                assert src.count==1 and src.crs is not None
                target=shapely_transform(
                   Transformer.from_crs("EPSG:4326",src.crs,always_xy=True).transform,poly)
                lon,lat=poly.centroid.x,poly.centroid.y
                # 250m surrounding rectangle (excludes target polygon).
                bbox=(lon-.004,lat-.003,lon+.004,lat+.003)
                tr=Transformer.from_crs("EPSG:4326",src.crs,always_xy=True)
                x1,y1=tr.transform(bbox[0],bbox[1])
                x2,y2=tr.transform(bbox[2],bbox[3])
                window=from_bounds(min(x1,x2),min(y1,y2),max(x1,x2),max(y1,y2),
                                   transform=src.transform)
                window=window.round_offsets().round_lengths()
                window=window.intersection(Window(0,0,src.width,src.height))
                if window.width>200 or window.height>200:raise RuntimeError("source window too large")
                values=src.read(1,window=window)
                geometry_tf=src.window_transform(window)
                center=geometry_mask([mapping(target)],out_shape=values.shape,
                                      transform=geometry_tf,invert=True,all_touched=False)
                touched=geometry_mask([mapping(target)],out_shape=values.shape,
                                      transform=geometry_tf,invert=True,all_touched=True)
                if center.sum()==0 and touched.sum()==0:raise RuntimeError("radar polygon missed JRC grid")
                surrounding=~touched
                if src.res[0]<0:raise RuntimeError("bad pixel scale")
                def dist(m):
                    vv=values[m]
                    return {"n":int(len(vv)),"missing":int((vv==0).sum()),
                       "dry":int((vv==1).sum()),"water":int((vv==2).sum()),
                       "observed":int(np.isin(vv,[1,2]).sum()),
                       "invalid":int((~np.isin(vv,[0,1,2])).sum())}
                c=dist(center);t=dist(touched);bg=dist(surrounding)
                assert c["invalid"]==t["invalid"]==bg["invalid"]==0
                row.update({"status":"read_ok",
                  "native_pixel_res_x":float(src.res[0]),
                  "native_pixel_res_y":float(src.res[1]),
                  "center_pixel_n":c["n"],"center_observed_n":c["observed"],
                  "center_water_n":c["water"],"center_dry_n":c["dry"],
                  "center_unobserved_n":c["missing"],
                  "center_fraction_water_of_OBSERVED":c["water"]/c["observed"] if c["observed"] else None,
                  "touched_pixel_n":t["n"],"touched_observed_n":t["observed"],
                  "touched_water_n":t["water"],"touched_dry_n":t["dry"],
                  "touched_unobserved_n":t["missing"],
                  "touched_fraction_water_of_OBSERVED":t["water"]/t["observed"] if t["observed"] else None,
                  "surrounding_pixel_n":bg["n"],"surrounding_observed_n":bg["observed"],
                  "surrounding_water_n":bg["water"],
                  "surrounding_fraction_water_of_OBSERVED":bg["water"]/bg["observed"] if bg["observed"] else None})
        except Exception as ex:
            row.update({"status":"read_failed","error":type(ex).__name__+": "+str(ex)[:260]})
        rows.append(row)
        print("MONTH",dn,ym,row["status"],"center",row.get("center_water_n"),
              "/",row.get("center_observed_n"),"touch",row.get("touched_water_n"),
              "/",row.get("touched_observed_n"),flush=True)
df=pd.DataFrame(rows)
df.to_csv(OUT/"phase47_jrc_monthly_pixel_audit.csv",index=False)
assess=[]
for dn in [4718,4719]:
    d=df[df.disasterNumber==dn].set_index("month")
    ok=int((d.status=="read_ok").sum())
    m2023={ym:{"observed":int(d.loc[ym].center_observed_n) if d.loc[ym].status=="read_ok" else None,
               "water":int(d.loc[ym].center_water_n) if d.loc[ym].status=="read_ok" else None}
               for ym in ("2023-03","2023-04","2023-05","2023-06")}
    apr=d.loc["2023-04"]
    may=d.loc["2023-05"]
    qa={"disasterNumber":dn,"reference_years":["2022","2024"],
         "year_2023_months":m2023,
         "months_read_successfully":ok,
         "April_2023_observed_water_in_polygon":bool(apr.center_water_n>0) if apr.status=="read_ok" and apr.center_observed_n>0 else None,
         "May_2023_observed_water_in_polygon":bool(may.center_water_n>0) if may.status=="read_ok" and may.center_observed_n>0 else None,
         "both_months_actual_observations":bool(apr.center_observed_n>0 and may.center_observed_n>0) if apr.status==may.status=="read_ok" else False,
         "SENTINEL1_independent_2023_flood_polygon_proven":False,
         "classification_rationale":"Independent Landsat monthly water detection only; monthly temporal aggregation does not establish precise radar event date"}
    assess.append(qa)
(OUT/"phase47_caselevel_evidence_gate.json").write_text(json.dumps(assess,indent=2))
summary={"fema_disasters":[4718,4719],"raster_months_attempted":len(df),
         "months_successful":int((df.status=="read_ok").sum()),
         "2023_April_May_months_with_usable_center_water_observations":
              int(df[(df.month.isin(["2023-04","2023-05"]))&
                     df.center_observed_n.fillna(0).gt(0)].shape[0]),
         "newly_validated_event_inundation_footprints":0,
         "population_and_building_exposure_computed":False,"model_retrained":False,
         "frozen_TrackA_unchanged":True,
         "source":"JRC Global Surface Water v1.5 Monthly History (Landsat Collection 2), 2022-2024, public source.coop mirror",
         "source_notes":[
             "This data product was published/reprocessed after 2023 and is NOT T0 or T+72 decision-time evidence.",
             "JRC Monthly History is a classified Landsat product independent of Sentinel-1 radar but not a field-survey flood truth mask.",
             "30 m source pixels are larger than the 10 m radar-grid candidate pixels; border disagreement is anticipated.",
             "0 is no observation, not dry. Quantities with no observed pixels are treated as missing.",
             "Monthly classification cannot attribute an April10 or May7 radar signal to a precise event day.",
             "The same-month 2022/2024 comparisons provide seasonality checks, not statistical validation.",
             "The small polygons were selected after radar screening; evaluation is exploratory and selection-biased.",
             "No disaster-wide inundation, population exposure, or accepted funding model update."]}
(OUT/"phase47_summary.json").write_text(json.dumps(summary,indent=2))
display=["disasterNumber","month","status","center_pixel_n","center_observed_n",
  "center_water_n","center_dry_n","center_unobserved_n","touched_water_n",
  "touched_observed_n","surrounding_fraction_water_of_OBSERVED"]
(OUT/"PHASE47_REPORT.md").write_text(
  "# Phase 47 — Independent JRC monthly Landsat water classification at two radar flood candidates\n\n"
  "Actual 2022–2024 raster pixels were requested from the JRC v1.5 monthly water history, "
  "with water=2, dry=1, unobserved=0, never recoding missing to dry.\n\n"
  "## Measurements\n\n"+df[display].to_markdown(index=False)+
  "\n\n## Summary\n\n"+json.dumps({k:v for k,v in summary.items() if k!="source_notes"},indent=2)+
  "\n\n## Scientific limits\n\n"+"\n".join("- "+x for x in summary["source_notes"]))
print("PHASE47_SUMMARY",json.dumps({k:v for k,v in summary.items() if k!="source_notes"},indent=2),flush=True)
