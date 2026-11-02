#!/usr/bin/env python3
"""Phase29: OPERA DSWx-HLS WTR REAL GeoTIFF pixel-read gate.

Uses immutable Phase28 scene catalog (12 FEMA floods in 2023–24). Picks
a SAME-MGRS-TILE pre-event and during/post-event WTR pair by documented
date proximity and cloud-cover metadata, independent of FEMA funding.

NASA Earthdata bearer token is optionally provided through a GitHub
Actions SECRET named NASA_EARTHDATA_TOKEN. Never print, persist or
incorporate credentials in URL, errors or reports. No direct guessing,
bypass, or redistribution of protected archive files.

If token unavailable, clearly report blocked real raster retrieval
and provide HTTP metadata only. If available, stream <=90MB per WTR
to temporary disk, decode REAL class values with rasterio, reproject
post onto pre grid at 30m with nearest-neighbor, and inspect a 2-km
radius around original NOAA report point. Pre/post valid pixels only:
 WTR classes 0 not water; 1 open water; 2 partial surface water;
 classes >=252 and 255 are masked. Compatible class 1/2 water
 transitions from class 0 are CANDIDATE newly detected water, NOT
 verified event-caused flood, economic damages, or footprint exposure.

Limit to ONE pair, no more than 2 source files downloaded, bounded.
Do not claim validated impact/case availability merely from catalog.
"""
from __future__ import annotations
from pathlib import Path
from urllib.parse import urlparse
import json,os,re,tempfile,math
import numpy as np,pandas as pd,requests

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/"audit_inputs"/"dswx29"
OUT=ROOT/"audit_outputs"/"dswx29"
OUT.mkdir(parents=True,exist_ok=True)
MAX_BYTES=90_000_000
RADIUS_M=2000
SECRET=os.environ.get("NASA_EARTHDATA_TOKEN","").strip()
ARCHIVE_HOST="archive.podaac.earthdata.nasa.gov"
KNOWN_CLASS={0,1,2,252,253,254,255}
BASENAME_RE=re.compile(r"DSWx-HLS_(T\d{2}[A-Z]{3})_",re.I)

def safe_name(url):
    return urlparse(str(url)).path.rsplit("/",1)[-1]

def safe_probe(url,token):
    host=urlparse(str(url)).hostname
    if host!=ARCHIVE_HOST:raise ValueError("Disallowed URL host for protected fetch")
    headers={"User-Agent":"University-thesis-DSWx-read-only-audit/1.0",
             "Range":"bytes=0-1023"}
    if token:headers["Authorization"]="Bearer "+token
    result={"attempted":True,"token_configured":bool(token),
        "http_status":None,"content_type":None,
        "final_hostname":None,"response_status":"unverified",
        "first_bytes_are_tiff":False,
        "source_filename":safe_name(url)}
    try:
        with requests.get(url,headers=headers,timeout=(15,40),stream=True,
             allow_redirects=True) as response:
            result["http_status"]=int(response.status_code)
            result["content_type"]=response.headers.get("Content-Type")
            result["final_hostname"]=urlparse(response.url).hostname
            typ=(result["content_type"] or "").lower()
            if response.status_code==206 and response.headers.get("Content-Range"):
                b=next(response.iter_content(1024),b"")[:1024]
                result["sample_length"]=len(b)
                result["first_bytes_are_tiff"]=b[:4] in [
                    b"II\x2a\x00",b"MM\x00\x2a",b"II\x2b\x00",b"MM\x00\x2b"]
                result["response_status"]="bounded_tiff_bytes" if result[
                    "first_bytes_are_tiff"] else "range_not_tiff"
            elif response.status_code==200 and "tif" in typ:
                result["response_status"]="server_ignored_range_200_tiff_unread"
            elif response.status_code in (401,403):
                result["response_status"]="unauthorized_or_forbidden"
            elif response.status_code==400:
                result["response_status"]="http_400_protected_endpoint_unresolved"
            else:
                result["response_status"]="not_authenticated_raster"
    except Exception as e:
        # Never include exception strings, which might embed secure URL headers.
        result["response_status"]="request_exception_"+type(e).__name__
    return result

def choose_pair(scenes,cases):
    # Different water scenes may share a tile but be processed a later year.
    # Inspect acquisition time, not processing string timestamp.
    s=scenes[(scenes.wtr_link.notna())&
             (scenes.scene_phase.isin(["pre","event_or_post"]))].copy()
    s["tile"]=s.title.fillna("").astype(str).str.extract(BASENAME_RE,expand=False)
    s["time"]=pd.to_datetime(s.time_start,utc=True,errors="coerce")
    s=s.dropna(subset=["tile","time"])
    c=cases.set_index("disasterNumber")
    candidates=[]
    for (dn,tile),g in s.groupby(["disasterNumber","tile"]):
        pre=g[g.scene_phase=="pre"]
        post=g[g.scene_phase=="event_or_post"]
        if pre.empty or post.empty:continue
        if dn not in c.index:continue
        t=pd.to_datetime(c.loc[dn,"incident_begin"],utc=True)
        for _,a in pre.iterrows():
            for _,b in post.iterrows():
                # Avoid a no-change comparison between same timestamp.
                if b.time<=a.time:continue
                before_days=(t-a.time).total_seconds()/86400
                after_days=(b.time-t).total_seconds()/86400
                if before_days<0 or after_days <0:continue
                if after_days>35:continue
                ac=pd.to_numeric(a.cloud_cover_metadata,errors="coerce")
                bc=pd.to_numeric(b.cloud_cover_metadata,errors="coerce")
                # Cloud metadata can be absent; late optical may be unusable.
                cloud=float(np.nanmean([ac,bc])) if pd.notna(ac) or pd.notna(bc) else 100.
                candidates.append({
                    "disasterNumber":int(dn),"tile":tile,
                    "pre_time":a.time.isoformat(),"post_time":b.time.isoformat(),
                    "days_before_event":float(before_days),
                    "days_after_event":float(after_days),
                    "mean_catalog_cloud_cover":cloud,
                    "pre_url":str(a.wtr_link),"post_url":str(b.wtr_link),
                    "pre_granule_title":str(a.title),"post_granule_title":str(b.title),
                    "noaa_lat":float(c.loc[dn,"NOAA_report_lat"]),
                    "noaa_lon":float(c.loc[dn,"NOAA_report_lon"])})
    if not candidates:return None,0
    # Prefer time proximity, then lower cloud, then stable identifiers.
    d=pd.DataFrame(candidates)
    d["score"]=d.days_before_event + 1.5*d.days_after_event+.10*d.mean_catalog_cloud_cover
    d=d.sort_values(["score","disasterNumber","tile","pre_time","post_time"])
    return d.iloc[0].to_dict(),int(len(d))

def download(url,path,token):
    if urlparse(url).hostname!=ARCHIVE_HOST:raise ValueError("Disallowed source")
    headers={"User-Agent":"Academic-OPERA-DSWx-classification-study/1.0",
             "Authorization":"Bearer "+token}
    count=0
    with requests.get(url,headers=headers,timeout=(20,110),
                      stream=True,allow_redirects=True) as resp:
        if resp.status_code!=200:
            return {"status":"http_"+str(resp.status_code),"download_bytes":0}
        ct=(resp.headers.get("Content-Type") or "").lower()
        length=resp.headers.get("Content-Length")
        if length and int(length)>MAX_BYTES:
            return {"status":"large_object_refused","download_bytes":0}
        if any(z in ct for z in ["text/html","application/json"]):
            return {"status":"not_tiff_content_type","download_bytes":0}
        with open(path,"wb") as f:
            for buf in resp.iter_content(256*1024):
                count+=len(buf)
                if count>MAX_BYTES:
                    f.close()
                    path.unlink(missing_ok=True)
                    return {"status":"large_stream_refused","download_bytes":count}
                f.write(buf)
    with open(path,"rb") as f:signature=f.read(4)
    valid=signature in [b"II\x2a\x00",b"MM\x00\x2a",
                         b"II\x2b\x00",b"MM\x00\x2b"]
    if not valid:
        path.unlink(missing_ok=True)
        return {"status":"not_valid_tiff_magic","download_bytes":count}
    return {"status":"tiff_downloaded","download_bytes":count}

def classify_local(prepath,postpath,pair):
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import reproject,Resampling
    from pyproj import Transformer
    with rasterio.open(prepath) as left,rasterio.open(postpath) as right:
        assert left.crs is not None and right.crs is not None
        assert left.count>=1 and right.count>=1
        transformer=Transformer.from_crs("EPSG:4326",left.crs,always_xy=True)
        x,y=transformer.transform(pair["noaa_lon"],pair["noaa_lat"])
        # Confirm actual event coordinate falls in *both* real mosaics.
        win=from_bounds(x-RADIUS_M,y-RADIUS_M,x+RADIUS_M,
                        y+RADIUS_M,transform=left.transform)
        win=win.round_offsets().round_lengths()
        bounds=(max(0,int(win.col_off)),max(0,int(win.row_off)),
             min(left.width,int(win.col_off+win.width)),
             min(left.height,int(win.row_off+win.height)))
        c0,r0,c1,r1=bounds
        if c1<=c0 or r1<=r0:return {"status":"NOAA_point_outside_pre_tile"}
        window=rasterio.windows.Window(c0,r0,c1-c0,r1-r0)
        p=left.read(1,window=window)
        local_transform=left.window_transform(window)
        q=np.full(p.shape,255,dtype=np.uint8)
        reproject(source=rasterio.band(right,1),destination=q,
          src_transform=right.transform,src_crs=right.crs,
          dst_transform=local_transform,dst_crs=left.crs,
          src_nodata=right.nodata if right.nodata is not None else 255,
          dst_nodata=255,resampling=Resampling.nearest)
        allowed_before=np.isin(p,[0,1,2])
        allowed_after=np.isin(q,[0,1,2])
        common=allowed_before&allowed_after
        bwater=np.isin(p,[1,2]);awater=np.isin(q,[1,2])
        changed=(~bwater)&awater&common
        total=common.sum()
        if abs(left.transform.a*left.transform.e-left.transform.b*left.transform.d)<=0:
            return {"status":"invalid_affine_area"}
        pixel_area=abs(left.transform.a*left.transform.e-left.transform.b*left.transform.d)
        if not left.crs.is_projected:
            return {"status":"geographic_crs_not_meter_area"}
        pvals,pcnts=np.unique(p,return_counts=True)
        qvals,qcnts=np.unique(q,return_counts=True)
        return {"status":"real_wtr_pixels_decoded",
          "pre_crs":left.crs.to_string(),"post_crs":right.crs.to_string(),
          "pre_resolution_m":list(left.res),"window_pixel_count":int(p.size),
          "valid_paired_pixel_count":int(total),
          "valid_paired_pixel_fraction":float(total/p.size),
          "pre_class_counts":{str(v):int(n) for v,n in zip(pvals,pcnts)},
          "post_class_counts":{str(v):int(n) for v,n in zip(qvals,qcnts)},
          "candidate_new_surface_water_pixels":int(changed.sum()),
          "candidate_new_surface_water_area_km2":float(changed.sum()*pixel_area/1e6),
          "verified_fema_flood_extent":False,
          "verified_human_exposure":False,
          "note":"Raw paired WTR-pixel class changes near report point; not flood causation or validation. Pre-date is not seasonal baseline."}

def main():
    case=pd.read_csv(INPUT/"case_level_cmr_catalog_feasibility.csv")
    scenes=pd.read_csv(INPUT/"scene_level_cmr_metadata.csv")
    assert len(case)==12 and len(scenes)==387
    pair,n_candidates=choose_pair(scenes,case)
    result={
      "study":"Phase29 OPERA DSWx real WTR raster gate",
      "fema_flood_pilot_cases":12,
      "catalog_scene_count":387,
      "same_tile_temporal_pairs_considered":n_candidates,
      "pair_selected":{k:v for k,v in pair.items() if not k.endswith("_url")} if pair else None,
      "earthdata_token_configured":bool(SECRET),
      "bounded_http_probe":None,"real_wtr_files_downloaded":0,
      "real_raster_pixel_classes_decoded":False,
      "candidate_surface_water_gain_area_km2":None,
      "event_specific_inundation_validated":False,
      "fema_flood_affected_population_measured":False,
      "model_training_performed":False}
    if pair is None:
        result["status"]="no_same_tile_temporal_pair"
    else:
        probe=safe_probe(pair["post_url"],SECRET)
        result["bounded_http_probe"]=probe
        result["status"]="protected_archive_access_not_established"
        if not SECRET:
            result["reason"]="NASA_EARTHDATA_TOKEN GitHub secret is not configured; protected PO.DAAC GeoTIFF cannot be assumed publicly readable."
        else:
            # Download two WTR layers only from authenticated archive.
            with tempfile.TemporaryDirectory() as temp:
                first=Path(temp)/"pre_WTR.tif"
                second=Path(temp)/"post_WTR.tif"
                a=download(pair["pre_url"],first,SECRET)
                b=download(pair["post_url"],second,SECRET)
                result["downloads"]={"pre":a,"post":b}
                result["real_wtr_files_downloaded"]=int(a["status"]=="tiff_downloaded")+int(b["status"]=="tiff_downloaded")
                if result["real_wtr_files_downloaded"]==2:
                    try:
                        check=classify_local(first,second,pair)
                        result["raster_read"]=check
                        result["real_raster_pixel_classes_decoded"]=check.get("status")=="real_wtr_pixels_decoded"
                        if result["real_raster_pixel_classes_decoded"]:
                            result["candidate_surface_water_gain_area_km2"]=check["candidate_new_surface_water_area_km2"]
                            result["status"]="paired_wtr_pixels_read_not_validated_flood"
                        else:result["status"]="raster_read_unusable"
                    except Exception as e:
                        result["status"]="raster_decode_error_"+type(e).__name__
    result["caveats"]=[
       "NASA DSWx-HLS detects surface water; new water between two optical scenes is a candidate, not a validated FEMA flood extent.",
       "Only one selected MGRS tile and ~2km window around a NOAA report point; no FEMA disaster-wide inundation estimate.",
       "Class 252 snow/ice, 253 cloud/shadow, 254 ocean, 255 fill excluded from both dates.",
       "An optical two-scene comparison is not an adequate seasonal baseline or independent field validation.",
       "Original CMR WTR granules may be reprocessed in 2026; acquisition time versus processing time differs.",
       "No exposed households or hurricane/flood financial impacts estimated; accepted 830/912 router & $5.015M point stack unchanged.",
       "Secrets must be stored in GitHub Actions secret store only, never git/source/CSV/logs."
    ]
    (OUT/"summary.json").write_text(json.dumps(result,indent=2))
    print("BEGIN_PHASE29_RASTER_GATE")
    print(json.dumps(result,indent=2))
    print("END_PHASE29_RASTER_GATE")
if __name__=="__main__":main()
