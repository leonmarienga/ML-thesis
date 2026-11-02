#!/usr/bin/env python3
"""Phase 30: OPEN Sentinel-2 COG real pixel water-change feasibility.

This is NOT NASA OPERA DSWx and NOT a validated event flood footprint.
Source: Element84 Earth Search v1 sentinel-2-c1-l2a public STAC and openly
hosted Level-2A COG raster assets, no secrets/authentication.

Fixed sample: 12 Phase28 FEMA Flood 2023-24 NOAA reported point cases.
Use original incident begin date; search <=45 days before, <=21 days after
incident start and select a single same-MGRS-tile optical scene pair from
published STAC, independent of funding target or water outcome.

Read actual pre/post SCL (20m), green B03, NIR B08 COG windows in a
2-km radius around NOAA point; align 10m reflectance to 20m SCL template
using WarpedVRT; mask SCL invalid/cloud/shadow/snow (allowed 4,5,6,7);
water candidate SCL class6; NDWI-support where >=0.2 using calibrated
reflectance scale/offset. Report clear-pixel coverage and candidate changes.

NO claim of FEMA event causation, inundation area, affected housing, or
decision-time data availability; baseline is ONE preceding image, not an
adequate multiyear seasonal permanent-water baseline. Funding model frozen.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from urllib.parse import urlparse
import json,os,math,time
import numpy as np,pandas as pd,requests

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/"audit_inputs"/"sentinel30"
OUT=ROOT/"audit_outputs"/"sentinel30"
OUT.mkdir(parents=True,exist_ok=True)
STAC="https://earth-search.aws.element84.com/v1/search"
COLLECTION="sentinel-2-c1-l2a"
WINDOW_M=2000
MAX_STAC_ITEMS=180
MAX_CASES=12
PRE_DAYS=45
POST_DAYS=21
MAX_CLOUD=65.
QA_VALID={4,5,6,7}
WATER=6
NDWI_MIN=.20
HEADERS={"User-Agent":"Academic-flood-research-public-sentinel2-COG-phase30/1.0"}
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN","EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS",".tif")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY","2")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY","2")
os.environ.setdefault("GDAL_HTTP_TIMEOUT","30")

def http(url,params=None):
    for attempt in range(3):
        try:
            res=requests.get(url,params=params,headers=HEADERS,timeout=(12,65))
            res.raise_for_status()
            return res.json(),"ok"
        except Exception as e:
            if attempt==2:return None,"fetch_failure_"+type(e).__name__
            time.sleep(attempt+1)
    return None,"retry_exhausted"

def ts(v):
    return pd.to_datetime(v,utc=True,errors="coerce")

def item_tile(v):
    props=v.get("properties",{})
    tile=props.get("s2:mgrs_tile") or props.get("grid:code") or props.get("mgrs_tile")
    return str(tile or "NA")

def catalog(case):
    dn=int(case.disasterNumber)
    begin=ts(case.incident_begin)
    lat=float(case.NOAA_report_lat);lon=float(case.NOAA_report_lon)
    # Fixed spatial point for all 12 cases: no manual cherry-picking.
    pars={"collections":COLLECTION,
          "bbox":",".join(map(str,[lon-.008,lat-.008,lon+.008,lat+.008])),
          "datetime":(begin-pd.Timedelta(days=PRE_DAYS)).strftime("%Y-%m-%dT00:00:00Z")+
              "/"+(begin+pd.Timedelta(days=POST_DAYS)).strftime("%Y-%m-%dT23:59:59Z"),
          "limit":MAX_STAC_ITEMS}
    response,status=http(STAC,pars)
    rows=[]
    diag={"disasterNumber":dn,"status":status,"total_returned":0,
          "stac_truncated":False,"valid_assets":0,
          "pre_scenes":0,"post_scenes":0,"candidate_pairs":0}
    if response is None:return diag,rows
    try:
        items=response.get("features",[])
        diag["total_returned"]=len(items)
        matched=response.get("context",{}).get("matched")
        if matched is None:matched=response.get("numberMatched")
        diag["stac_truncated"]=bool(matched is not None and matched>len(items))
        for item in items:
            at=item.get("assets",{})
            if not all(k in at for k in ["scl","green","nir"]):continue
            names={k:str(at[k].get("href","")) for k in ["scl","green","nir"]}
            if not all(u.startswith("https://") and u.lower().split("?")[0].endswith(".tif") for u in names.values()):
                continue
            t=ts(item.get("properties",{}).get("datetime"))
            if pd.isna(t):continue
            delta=(t-begin).total_seconds()/86400
            phase="pre" if delta<0 else "post"
            props=item.get("properties",{})
            clouds=pd.to_numeric(props.get("eo:cloud_cover"),errors="coerce")
            rec={"disasterNumber":dn,"stac_id":item.get("id"),
                 "tile":item_tile(item),"timestamp":t.isoformat(),
                 "days_relative":float(delta),"phase":phase,
                 "cloud_cover":float(clouds) if pd.notna(clouds) else None,
                 "scl_url":names["scl"],"green_url":names["green"],"nir_url":names["nir"],
                 "green_scale":float(at["green"].get("raster:bands",[{}])[0].get("scale",.0001)),
                 "green_offset":float(at["green"].get("raster:bands",[{}])[0].get("offset",0.0)),
                 "nir_scale":float(at["nir"].get("raster:bands",[{}])[0].get("scale",.0001)),
                 "nir_offset":float(at["nir"].get("raster:bands",[{}])[0].get("offset",0.0)),
                 "reflectance_assets_accessed":False}
            rows.append(rec)
        diag["valid_assets"]=len(rows)
        diag["pre_scenes"]=sum(r["phase"]=="pre" for r in rows)
        diag["post_scenes"]=sum(r["phase"]=="post" for r in rows)
        diag["status"]="ok"
    except Exception as e:
        diag["status"]="parse_"+type(e).__name__
    return diag,rows

def select_pair(allrows):
    if not allrows:return None,0
    d=pd.DataFrame(allrows)
    candidates=[]
    for (dn,tile),part in d.groupby(["disasterNumber","tile"]):
        pre=part[(part.phase=="pre") & part.cloud_cover.le(MAX_CLOUD)]
        post=part[(part.phase=="post") & part.cloud_cover.le(MAX_CLOUD)]
        if pre.empty or post.empty:continue
        for _,a in pre.iterrows():
            for _,b in post.iterrows():
                if b.timestamp<=a.timestamp:continue
                score=(abs(a.days_relative)+1.5*abs(b.days_relative)
                       +.2*(a.cloud_cover+b.cloud_cover))
                candidates.append((float(score),int(dn),str(tile),a,b))
    if not candidates:return None,0
    candidates.sort(key=lambda x:(x[0],x[1],x[2],x[3].timestamp,x[4].timestamp))
    score,dn,tile,pre,post=candidates[0]
    return {"disasterNumber":dn,"tile":tile,"score":score,
             "pre":pre.to_dict(),"post":post.to_dict()},len(candidates)

def allowed_cog(url):
    host=urlparse(str(url)).hostname
    # Keep public HTTPS only; no credentials or unexpected hosts.
    if host is None or not host.endswith(".amazonaws.com"):
        # Some public Element 84 STAC COGs are distributed from open AWS storage
        # with s3.amazonaws.com hosts; record refusal rather than guessing.
        raise ValueError("unrecognized_public_cog_host")
    return str(url)

def synthetic_test():
    a=np.array([[4,6,5,8],[6,6,4,5]],dtype=np.uint8)
    b=np.array([[6,6,4,6],[4,5,8,6]],dtype=np.uint8)
    v=np.isin(a,list(QA_VALID))&np.isin(b,list(QA_VALID))
    gain=(a!=6)&(b==6)&v
    assert gain.sum()==1 and not gain[0,3] and not gain[1,3]
    return True

def read_pair(pair,lat,lon):
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling
    from pyproj import Transformer
    out={"raster_read_status":"unverified",
         "real_scl_pixels_decoded":False,
         "real_green_nir_pixels_decoded":False,
         "candidate_new_surface_water_pixels":None,
         "candidate_new_surface_water_km2":None}
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                      GDAL_HTTP_TIMEOUT="30",GDAL_HTTP_MAX_RETRY="2"):
      with rasterio.open(allowed_cog(pair["pre"]["scl_url"])) as template:
        if template.crs is None or not template.crs.is_projected:
            out["raster_read_status"]="invalid_scl_crs"
            return out
        x,y=Transformer.from_crs("EPSG:4326",template.crs,
                                 always_xy=True).transform(lon,lat)
        win=from_bounds(x-WINDOW_M,y-WINDOW_M,x+WINDOW_M,y+WINDOW_M,
                        transform=template.transform).round_offsets().round_lengths()
        # Safeguard point outside template tile.
        c0=max(0,int(win.col_off));r0=max(0,int(win.row_off))
        c1=min(template.width,int(win.col_off+win.width))
        r1=min(template.height,int(win.row_off+win.height))
        if c0>=c1 or r0>=r1:
            out["raster_read_status"]="reported_point_outside_tile";return out
        window=rasterio.windows.Window(c0,r0,c1-c0,r1-r0)
        ht,wd=int(window.height),int(window.width)
        if ht*wd>100000:
            out["raster_read_status"]="unexpectedly_large_window";return out
        arrays={}
        for phase in ["pre","post"]:
            for band in ["scl","green","nir"]:
                url=allowed_cog(pair[phase][band+"_url"])
                with rasterio.open(url) as src:
                    with WarpedVRT(src,crs=template.crs,
                         transform=template.transform,
                         width=template.width,height=template.height,
                         resampling=Resampling.nearest if band=="scl" else Resampling.bilinear) as vrt:
                        arr=vrt.read(1,window=window)
                        assert arr.shape==(ht,wd)
                        arrays[(phase,band)]=arr
        pre=arrays[("pre","scl")]
        post=arrays[("post","scl")]
        valid=np.isin(pre,list(QA_VALID)) & np.isin(post,list(QA_VALID))
        newly=(pre!=WATER)&(post==WATER)&valid
        gain_pixels=int(newly.sum())
        # Compute NDWI for each scene, using STAC scale and offset values.
        ndwi={}
        for phase in ["pre","post"]:
            g=arrays[(phase,"green")].astype(np.float32)
            n=arrays[(phase,"nir")].astype(np.float32)
            gg=g*float(pair[phase]["green_scale"])+float(pair[phase]["green_offset"])
            nn=n*float(pair[phase]["nir_scale"])+float(pair[phase]["nir_offset"])
            denom=gg+nn
            q=np.divide(gg-nn,denom,out=np.full_like(denom,np.nan),
                        where=(denom>0)&(g>0)&(n>0))
            ndwi[phase]=q
        confirm=newly & (ndwi["post"]>=NDWI_MIN) & np.isfinite(ndwi["post"])
        pixelarea=abs(template.transform.a*template.transform.e-
                      template.transform.b*template.transform.d)
        # Calculate only raw observed-pixel count, NOT definitive flood extent.
        out.update({
         "raster_read_status":"decoded_real_public_sentinel2_cogs",
         "real_scl_pixels_decoded":True,"real_green_nir_pixels_decoded":True,
         "image_pre_crs":template.crs.to_string(),"observed_pixel_size_m":list(template.res),
         "window_total_pixels":int(wd*ht),
         "paired_good_quality_pixels":int(valid.sum()),
         "paired_quality_coverage":float(valid.mean()),
         "pre_scl_water_pixels":int(((pre==WATER)&valid).sum()),
         "post_scl_water_pixels":int(((post==WATER)&valid).sum()),
         "candidate_new_surface_water_pixels":gain_pixels,
         "candidate_new_surface_water_km2":float(gain_pixels*pixelarea/1e6),
         "candidate_new_surface_water_ndwi_ge_02_pixels":int(confirm.sum()),
         "candidate_new_surface_water_ndwi_ge_02_km2":float(confirm.sum()*pixelarea/1e6),
         "ndwi_threshold":NDWI_MIN,
         "pre_quality_class_counts":{str(k):int(v) for k,v in zip(*np.unique(pre,return_counts=True))},
         "post_quality_class_counts":{str(k):int(v) for k,v in zip(*np.unique(post,return_counts=True))},
         "no_verified_inundation_polygon":True,
         "no_verified_disaster_event_causation":True,
         "no_verified_exposed_buildings":True})
        return out

def main():
    assert synthetic_test()
    case=pd.read_csv(INPUT/"case_level_cmr_catalog_feasibility.csv")
    assert len(case)==MAX_CASES
    statuses=[]
    items=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs={pool.submit(catalog,row):int(row.disasterNumber)
              for _,row in case.iterrows()}
        for f in as_completed(jobs):
            dn=jobs[f]
            try:a,b=f.result()
            except Exception as e:
                a={"disasterNumber":dn,"status":"unexpected_"+type(e).__name__};b=[]
            statuses.append(a);items.extend(b)
            print("SENTINEL_CATALOG_CASE",dn,a.get("status"),a.get("pre_scenes"),
                  a.get("post_scenes"),flush=True)
    d=pd.DataFrame(statuses).sort_values("disasterNumber")
    d.to_csv(OUT/"public_stac_flood12_catalog_status.csv",index=False)
    i=pd.DataFrame(items)
    i.to_csv(OUT/"public_stac_scene_metadata.csv",index=False)
    pair,n=select_pair(items)
    result={"source":"Element84 Earth Search public Sentinel-2 Collection 1 L2A COG",
      "not_opera_dswx":True,
      "study":"Public Sentinel-2 real raster positive control near 2023-24 FEMA Flood report points",
      "fixed_flood_cases":12,"catalog_queries_ok":int((d.status=="ok").sum()),
      "cases_with_pre_scenes":int((d.pre_scenes.fillna(0)>0).sum()),
      "cases_with_post_scenes":int((d.post_scenes.fillna(0)>0).sum()),
      "catalog_scene_count":len(i),
      "paired_same_tile_scene_candidates":int(n),
      "chosen_pair":{k:v for k,v in pair.items() if k not in ("pre","post")} if pair else None,
      "chosen_scene_names":{
          "pre":pair["pre"]["stac_id"],"post":pair["post"]["stac_id"],
          "pre_time":pair["pre"]["timestamp"],"post_time":pair["post"]["timestamp"],
          "pre_cloud_cover":pair["pre"]["cloud_cover"],"post_cloud_cover":pair["post"]["cloud_cover"]} if pair else None,
      "real_raster_read":None,
      "verified_water_pixels_decoded":False,
      "inundation_polygon_validated":False,
      "exposed_housing_measured":False,
      "model_training_performed":False}
    if pair:
        row=case[case.disasterNumber==pair["disasterNumber"]].iloc[0]
        try:
            read=read_pair(pair,float(row.NOAA_report_lat),float(row.NOAA_report_lon))
        except Exception as ex:
            read={"raster_read_status":"read_failure_"+type(ex).__name__,
                  "real_scl_pixels_decoded":False}
            print("REAL_RASTER_READ_FAILED",type(ex).__name__,str(ex)[:180],flush=True)
        result["real_raster_read"]=read
        result["verified_water_pixels_decoded"]=bool(read.get("real_scl_pixels_decoded"))
    else:
        result["real_raster_read"]={"raster_read_status":"no_temporal_pair_found"}
    result["scientific_limits"]=[
       "Public Sentinel-2 L2A SCL is not NASA OPERA DSWx WTR.",
       "A paired single pre/post optical scene is not a seasonal permanent-water baseline.",
       "SCL water class=6 and NDWI>=.20 are exploratory detections only, not independently validated flood extents.",
       "SCL clear classes 4,5,6,7 used; cloud/shadow/snow excluded, changes unobservable where masked.",
       "Only a 2-km square window near a NOAA report location, not the FEMA flood geographic footprint.",
       "Subsequent revisions to imagery may change retrospective reflectance and SCL outputs.",
       "This experiment has not measured flooded homes, losses, allocation needs or improved FEMA funding predictions.",
       "The accepted 830/912 six-band router and $5.015M MAE remain unchanged."
    ]
    (OUT/"summary.json").write_text(json.dumps(result,indent=2))
    print("BEGIN_PHASE30_PUBLIC_COG")
    print(json.dumps(result,indent=2))
    print("END_PHASE30_PUBLIC_COG")
if __name__=="__main__":main()
