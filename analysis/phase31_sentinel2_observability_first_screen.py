#!/usr/bin/env python3
"""Phase31 public Sentinel-2 real pixel OBSERVABILITY first audit.

Reuses fixed 12 Phase28 FEMA 2023-24 event locations and already downloaded
Phase30 STAC scene metadata. For each FEMA case, consider <=2 MGRS tiles,
<=4 pre and <=4 post images PER tile, ranked by time separation and metadata
cloud cover without target or water class. Read actual local 20m SCL pixels
in 4x4km square at NOAA report point and reject low-observation scenes.

After measured quality only, rank SAME-TILE temporal pairs by >=30% strict
paired-clear pixels (SCL 4 vegetation,5 bare,6 open water; exclude SCL 7
unclassified, 8/9/10 cloud/shadow, 11 snow/ice, 0 nodata). Select
best quality-gated pair by paired valid fraction, then date proximity,
never by water-change outcome. Where qualifying, read green B03 and NIR B08
for ONE pair to calculate an exploratory NDWI-supported transition count.

Two-scene difference is NOT verified flood, not inundation polygon, not
disaster-specific impact, and not exposed homes. Permanent-water and seasonal
reference are still unavailable. No model training/funding variables used.
"""
from __future__ import annotations
import json,math,os
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from urllib.parse import urlparse
import numpy as np,pandas as pd

ROOT=Path(__file__).resolve().parents[1]
IN=ROOT/"audit_inputs"/"sentinel31"
OUT=ROOT/"audit_outputs"/"sentinel31"
OUT.mkdir(parents=True,exist_ok=True)
MAX_CASES=12
MAX_TILES_PER_CASE=2
MAX_PER_PHASE_PER_TILE=4
MAX_WORKERS=4
SCL_VALID={4,5,6}
SCL_VALID_SENSITIVITY={4,5,6,7}
MIN_PAIRED_VALID_FRAC=.30
MIN_SCENE_CLEAR_FRAC=.20
RADIUS_M=2000
TARGET_RES=20.0
NDWI_THRESHOLD=.20
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN","EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS",".tif")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY","2")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY","2")
os.environ.setdefault("GDAL_HTTP_TIMEOUT","35")
os.environ.setdefault("GDAL_HTTP_MULTIRANGE","YES")

def allowed_url(url):
    p=urlparse(str(url))
    if p.scheme!="https" or not (
       p.hostname=="e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com"):
        raise ValueError("untrusted_cog_asset_host")
    if not p.path.lower().endswith(".tif"):raise ValueError("asset_not_tif")
    return str(url)

def subset_candidates(cases,allmeta):
    rows=[]
    for _,case in cases.sort_values("disasterNumber").iterrows():
        dn=int(case.disasterNumber)
        g=allmeta[allmeta.disasterNumber==dn].copy()
        g["timestamp_dt"]=pd.to_datetime(g.timestamp,utc=True,format='mixed')
        g["meta_cloud"]=pd.to_numeric(g.cloud_cover,errors="coerce").fillna(100)
        g["phase"]=g.phase.astype(str)
        both_tiles=[]
        for tile,part in g.groupby("tile"):
            if part[part.phase=="pre"].empty or part[part.phase=="post"].empty:continue
            count=min(len(part[part.phase=="pre"]),len(part[part.phase=="post"]))
            both_tiles.append((count,tile))
        for _,tile in sorted(both_tiles,key=lambda x:(-x[0],str(x[1])))[:MAX_TILES_PER_CASE]:
            t=g[g.tile==tile]
            for phase in ("pre","post"):
                gg=t[t.phase==phase].copy()
                # Shortlist metadata, actual local clear SCL determines eligibility.
                gg["shortlist_rank"]=gg.days_relative.abs()+.075*gg.meta_cloud
                gg=gg.sort_values(["shortlist_rank","timestamp_dt","stac_id"])
                for _,r in gg.head(MAX_PER_PHASE_PER_TILE).iterrows():
                    row=r.to_dict()
                    row["NOAA_report_lat"]=float(case.NOAA_report_lat)
                    row["NOAA_report_lon"]=float(case.NOAA_report_lon)
                    rows.append(row)
    d=pd.DataFrame(rows)
    assert len(d)>0
    assert not d.duplicated(["disasterNumber","tile","stac_id"]).any()
    return d

def frame_config(pre_url,lon,lat):
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import from_origin
    with rasterio.open(allowed_url(pre_url)) as src:
        if src.crs is None or not src.crs.is_projected:
            raise RuntimeError("unprojected_source")
        x,y=Transformer.from_crs("EPSG:4326",src.crs,always_xy=True).transform(lon,lat)
        # Canonical stable local square grid; 200x200 pixels at 20m.
        xmin=math.floor((x-RADIUS_M)/TARGET_RES)*TARGET_RES
        ymax=math.ceil((y+RADIUS_M)/TARGET_RES)*TARGET_RES
        width=int(2*RADIUS_M/TARGET_RES)
        height=width
        transform=from_origin(xmin,ymax,TARGET_RES,TARGET_RES)
        return {"crs":src.crs,"transform":transform,"width":width,"height":height}

def read_band(url,config,band):
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling
    allowed_url(url)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                      GDAL_HTTP_MAX_RETRY="2",GDAL_HTTP_TIMEOUT="35"):
      with rasterio.open(url) as src:
        with WarpedVRT(src,crs=config["crs"],
           transform=config["transform"],width=config["width"],
           height=config["height"],
           resampling=Resampling.nearest if band=="scl" else Resampling.bilinear) as vrt:
            arr=vrt.read(1)
    assert arr.shape==(config["height"],config["width"])
    return arr

def read_local_scl(record,config):
    row={"disasterNumber":int(record["disasterNumber"]),
         "tile":str(record["tile"]),"stac_id":str(record["stac_id"]),
         "timestamp":str(record["timestamp"]),"phase":str(record["phase"]),
         "cloud_cover_catalog":float(record["meta_cloud"]),
         "days_relative":float(record["days_relative"])}
    try:
        data=read_band(record["scl_url"],config,"scl")
        values,counts=np.unique(data,return_counts=True)
        clean=np.isin(data,list(SCL_VALID))
        clean_lenient=np.isin(data,list(SCL_VALID_SENSITIVITY))
        row.update({"read_status":"ok",
          "n_pixels":int(data.size),
          "local_strict_clear_fraction":float(clean.mean()),
          "local_lenient_clear_fraction":float(clean_lenient.mean()),
          "local_water_class6_fraction":float((data==6).mean()),
          "local_cloud_snow_nodata_fraction":float((~clean_lenient).mean()),
          "scl_value_counts":json.dumps({str(k):int(v) for k,v in zip(values,counts)})})
        return row,data
    except Exception as ex:
        row.update({"read_status":"error_"+type(ex).__name__,
           "local_strict_clear_fraction":None,
           "local_lenient_clear_fraction":None,
           "local_water_class6_fraction":None,
           "scl_value_counts":None})
        return row,None

def one_disaster(case,scene):
    dn=int(case.disasterNumber)
    diagnostics=[]
    arrays={}
    metadata={str(row.stac_id):row for _,row in scene.iterrows()}
    tiles=list(scene.tile.dropna().unique())
    for tile in tiles:
        group=scene[scene.tile==tile]
        before=group[group.phase=="pre"]
        after=group[group.phase=="post"]
        if before.empty or after.empty:continue
        try:
            frame=frame_config(before.iloc[0].scl_url,
                  float(case.NOAA_report_lon),float(case.NOAA_report_lat))
        except Exception as ex:
            for _,row in group.iterrows():
                diagnostics.append({"disasterNumber":dn,"tile":tile,"stac_id":row.stac_id,
                     "read_status":"frame_error_"+type(ex).__name__})
            continue
        for _,r in group.iterrows():
            rec,arr=read_local_scl(r,frame)
            diagnostics.append(rec)
            if arr is not None:
                arrays[str(r.stac_id)]=(arr,frame)
    pairs=[]
    for tile,g in scene.groupby("tile"):
        a=g[g.phase=="pre"]
        b=g[g.phase=="post"]
        for _,pre in a.iterrows():
            for _,post in b.iterrows():
                if pre.stac_id not in arrays or post.stac_id not in arrays:continue
                arr1,config=arrays[str(pre.stac_id)]
                arr2,otherconfig=arrays[str(post.stac_id)]
                # Same tile frame, same actual grid.
                if config["crs"]!=otherconfig["crs"] or config["transform"]!=otherconfig["transform"]:continue
                p=np.isin(arr1,list(SCL_VALID))
                q=np.isin(arr2,list(SCL_VALID))
                quality=(p&q)
                ps=p.mean();qs=q.mean();paired=quality.mean()
                reject=[]
                if ps<MIN_SCENE_CLEAR_FRAC:reject.append("pre_strict_clear_below_20pct")
                if qs<MIN_SCENE_CLEAR_FRAC:reject.append("post_strict_clear_below_20pct")
                if paired<MIN_PAIRED_VALID_FRAC:reject.append("paired_strict_clear_below_30pct")
                if not reject:reject=["PASS"]
                pairs.append({"disasterNumber":dn,"tile":tile,
                    "pre_id":str(pre.stac_id),"post_id":str(post.stac_id),
                    "pre_time":str(pre.timestamp),"post_time":str(post.timestamp),
                    "pre_days_relative":float(pre.days_relative),
                    "post_days_relative":float(post.days_relative),
                    "pre_clear":float(ps),"post_clear":float(qs),
                    "paired_clear":float(paired),
                    "paired_valid_pixels":int(quality.sum()),
                    "total_pixels":int(quality.size),
                    "gate_reason":"|".join(reject),
                    "quality_gate_pass":len(reject)==1 and reject[0]=="PASS",
                    "paired_water_gain_pixels_uncomputed":True})
    pairs.sort(key=lambda r:(
         not r["quality_gate_pass"],-r["paired_clear"],
         abs(r["pre_days_relative"])+1.5*abs(r["post_days_relative"]),
         r["tile"],r["pre_id"],r["post_id"]))
    return diagnostics,pairs,arrays,metadata

def ndwi_real_pair(pair,arrays,metadata):
    pre=metadata[pair["pre_id"]]
    post=metadata[pair["post_id"]]
    left,frame=arrays[pair["pre_id"]]
    right,_=arrays[pair["post_id"]]
    mask=np.isin(left,list(SCL_VALID))&np.isin(right,list(SCL_VALID))
    gain=(left!=6)&(right==6)&mask
    try:
        for label,r in [("pre",pre),("post",post)]:
            green=read_band(r.green_url,frame,"green").astype(np.float32)
            nir=read_band(r.nir_url,frame,"nir").astype(np.float32)
            grefl=green*float(r.green_scale)+float(r.green_offset)
            nrefl=nir*float(r.nir_scale)+float(r.nir_offset)
            den=grefl+nrefl
            ndwi=np.divide(grefl-nrefl,den,out=np.full_like(den,np.nan),
                          where=(den>0)&(grefl>=0)&(nrefl>=0))
            if label=="post":
                supported=gain & np.isfinite(ndwi)&(ndwi>=NDWI_THRESHOLD)
                post_ndwi_valid=int(np.isfinite(ndwi).sum())
        return {"raster_read_status":"real_b03_b08_and_scl_decoded",
                "eligible_pair_pixels":int(mask.sum()),
                "eligible_pair_fraction":float(mask.mean()),
                "candidate_new_water_scl_pixels":int(gain.sum()),
                "candidate_new_water_scl_km2":float(gain.sum()*(TARGET_RES**2)/1e6),
                "candidate_new_water_ndwi_supported_pixels":int(supported.sum()),
                "candidate_new_water_ndwi_supported_km2":float(supported.sum()*(TARGET_RES**2)/1e6),
                "post_ndwi_finite_pixels":post_ndwi_valid,
                "label":"CANDIDATE OPTICAL WATER CHANGE ONLY; not validated inundation"}
    except Exception as e:
        return {"raster_read_status":"reflectance_error_"+type(e).__name__,
          "candidate_new_water_scl_pixels":None,
          "candidate_new_water_ndwi_supported_pixels":None}

def main():
    case=pd.read_csv(IN/"phase28"/"case_level_cmr_catalog_feasibility.csv")
    scenes=pd.read_csv(IN/"phase30"/"public_stac_scene_metadata.csv")
    assert len(case)==12 and case.disasterNumber.nunique()==12
    assert len(scenes)==283
    chosen=subset_candidates(case,scenes)
    chosen.drop(columns=["timestamp_dt"],errors="ignore").to_csv(
        OUT/"selected_scene_screening_roster.csv",index=False)
    results=[];all_scenes=[];all_pairs=[]
    # Real raster data live in memory only and are not redistributed.
    # Evaluate each case independently to ensure all 12 retained in audit.
    for _,r in case.sort_values("disasterNumber").iterrows():
        dn=int(r.disasterNumber)
        subset=chosen[chosen.disasterNumber==dn]
        diagnostics,pairs,arrays,metadata=one_disaster(r,subset)
        all_scenes.extend(diagnostics)
        all_pairs.extend(pairs)
        passing=[p for p in pairs if p["quality_gate_pass"]]
        chosen_pair=passing[0] if passing else None
        evaldata=ndwi_real_pair(chosen_pair,arrays,metadata) if chosen_pair else {}
        result={"disasterNumber":dn,
          "selected_scenes":int(len(subset)),
          "scenes_pixel_read_ok":int(sum(x.get("read_status")=="ok" for x in diagnostics)),
          "possible_pairs_measured":int(len(pairs)),
          "gate_passing_pairs":len(passing),
          "quality_pass":bool(chosen_pair),
          "selected_pre_id":chosen_pair["pre_id"] if chosen_pair else None,
          "selected_post_id":chosen_pair["post_id"] if chosen_pair else None,
          "selected_tile":chosen_pair["tile"] if chosen_pair else None,
          "selected_valid_pair_fraction":chosen_pair["paired_clear"] if chosen_pair else None,
          "selected_pre_clear_fraction":chosen_pair["pre_clear"] if chosen_pair else None,
          "selected_post_clear_fraction":chosen_pair["post_clear"] if chosen_pair else None,
          "selection_basis":"local SCL paired clear coverage then temporal distance, never water change",
          "fema_flood_attribution_verified":False,
          **evaldata}
        results.append(result)
        print("QUALITY_CASE",dn,"scenes",len(diagnostics),
              "pairs",len(pairs),"passing",len(passing),
              "raster",evaldata.get("raster_read_status","none"),flush=True)
    df=pd.DataFrame(results)
    assert len(df)==12 and df.disasterNumber.nunique()==12
    pd.DataFrame(all_scenes).to_csv(OUT/"real_scene_local_scl_quality.csv",index=False)
    pd.DataFrame(all_pairs).to_csv(OUT/"all_rejected_and_accepted_pair_quality.csv",index=False)
    df.to_csv(OUT/"case_quality_gate_and_water_change.csv",index=False)
    passing=df[df.quality_pass]
    summary={
      "original_fixed_fema_flood_cases":12,
      "phase30_stac_catalog_scenes":283,
      "local_scl_scenes_selected_for_screen":len(chosen),
      "local_scl_scenes_successfully_read":int(sum(x.get("read_status")=="ok" for x in all_scenes)),
      "quality_assessed_same_tile_pairs":len(all_pairs),
      "pair_eligibility_pre_and_post_strict_clear_20pct_paired_30pct":True,
      "disaster_cases_with_at_least_one_quality_gate_pass":len(passing),
      "cases_with_real_b03_b08_scl_water_change":int((df.get("raster_read_status")==
                                      "real_b03_b08_and_scl_decoded").sum()) if "raster_read_status" in df else 0,
      "disaster_cases_without_valid_pair":int((~df.quality_pass).sum()),
      "candidate_gain_km2_on_eligible_pairs_only":float(passing.candidate_new_water_scl_km2.sum(min_count=1)) if len(passing) and "candidate_new_water_scl_km2" in passing else None,
      "zero_quality_pairs_are_unobservable_not_no_flood":True,
      "results_by_case":df.to_dict("records"),
      "quality_definition":"SCL strict observed vegetation/bare/water 4/5/6, exclude unclassified 7 and snow/ice 11, cloud/shadow 8/9/10, other nonclear classes. Pre/post >=20% clear, overlap >=30%.",
      "no_emergency_decision_or_funding_model_changed":True,
      "flood_extent_footprint_verified":False,
      "affected_people_or_houses_estimated":False,
      "limitations":[
        "Same two-scene optical change is insufficient to attribute water to the FEMA event.",
        "Availability is for <=4 candidate pre and post per tile; failing does not imply no clear image exists in full archive.",
        "Only 4km by 4km local NOAA report-point square, not the disaster footprint.",
        "No seasonal or permanent-water baseline and no independent flood map overlap.",
        "Cloud/snow and STAC scene metadata effects can reduce measured sample coverage.",
        "The 12-case 2023-24 FEMA sample was examined previously; exploratory, not independent test.",
        "Physical source data are retrospective, not certified as available at FEMA declaration.",
        "Funding router accepted 830/912 and MAE about $5.015M remain frozen."
      ]}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2))
    print("BEGIN_PHASE31_CLEAR_PIXEL_SCREEN")
    print(json.dumps({k:v for k,v in summary.items() if k!="results_by_case"},indent=2))
    print("END_PHASE31_CLEAR_PIXEL_SCREEN")
if __name__=="__main__":main()
