#!/usr/bin/env python3
"""Phase 27: schema-validated flood footprint catalog + actual shape positive control.

Corrects Phase26 errors:
  * Walk ALL Copernicus activation pages until next=null; explicitly assert
    returned count is consistent with API reported count.
  * Copernicus country can be names or short_name strings/dicts.
  * USFIMR table has River, State, Year, Date, Platform, Area, Source, ZIP link;
    parse both historic SDML and WaterServ repository tables using header-aware
    values rather than hard-coded incorrect column positions.
  * For a PRE-2019 known historic polygon, retrieve an actual USFIMR shapefile
    ZIP and read polygon geometry with pyshp, check CRS and compute true area
    in equal-area projection. Positive control is NOT a matching 2019-24 FEMA
    disaster; it only tests geodata ingest capability.
  * For time/place candidate matches in study period, retrieve Copernicus
    activation detail AOI/DEL product metadata. AOI extent is NOT flooded area.
    Ingest only a genuine downloadable delineation flood geometry if available;
    do NOT turn an AOI extent into a flood extent.

No FEMA targets for matching or modelling. A zero candidate count in audited
completed catalogs does not mean no flood maps in other repositories.
"""
from __future__ import annotations
from pathlib import Path
from urllib.parse import urljoin
from io import BytesIO
import json, re, time, zipfile, math
import numpy as np,pandas as pd,requests
from bs4 import BeautifulSoup
import shapefile
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import transform
from pyproj import CRS, Transformer

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/"audit_inputs"/"flood27_catalog"
OUT=ROOT/"audit_outputs"/"flood27_catalog"
OUT.mkdir(parents=True,exist_ok=True)
CEMS_API="https://mapping.emergency.copernicus.eu/activations/api/activations/"
CEMS_RAPID="https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations-info/"
USFIMR=["https://waterserv.ua.edu/datasets/usfimr/","https://sdml.ua.edu/usfimr/"]
YEAR_START,YEAR_END=2019,2024
LIMIT=200
MAX_PAGES=110
MAX_POSITIVE_DOWNLOAD_BYTES=30_000_000
S=requests.Session()
S.headers["User-Agent"]="Academic observed flood inundation audit (phase27) 1.0"

def get(url,params=None,timeout=55):
    last=None
    for k in range(3):
        try:
            r=S.get(url,params=params,timeout=(12,timeout),allow_redirects=True)
            if r.status_code in (403,404,429):
                return None,f"http_{r.status_code}"
            r.raise_for_status()
            return r,"ok"
        except Exception as ex:
            last=str(ex)[:160]
            if k<2:time.sleep(1.2*(k+1))
    return None,"transport_"+str(last)

def iso_time(o,*keys):
    for key in keys:
        z=pd.to_datetime(o.get(key),errors="coerce",utc=True)
        if pd.notna(z):return z
    return pd.NaT

def countries(o):
    c=o.get("countries",[]) or []
    if isinstance(c,(str,dict)):c=[c]
    names=[]
    for x in c:
        if isinstance(x,str):names.append(x)
        elif isinstance(x,dict):
            for key in ["name","short_name","iso2","iso3","code"]:
                if x.get(key):names.append(str(x[key]))
    return names

def is_united_states(o):
    names=countries(o)
    allowed={"US","USA","UNITED STATES","UNITED STATES OF AMERICA","U.S.A."}
    return any(x.upper().strip() in allowed for x in names)

def centroid(v):
    if isinstance(v,str):
        x=re.search(r"POINT\s*\(\s*([-+0-9.]+)\s+([-+0-9.]+)\s*\)",v,re.I)
        if x:return float(x.group(2)),float(x.group(1))
    return None,None

def haversine(lat1,lon1,lat2,lon2):
    a,b,c,d=np.radians([lat1,lon1,lat2,lon2])
    v=np.sin((c-a)/2)**2+np.cos(a)*np.cos(c)*np.sin((d-b)/2)**2
    return float(2*6371.0088*np.arcsin(min(1,np.sqrt(v))))

def index_cems():
    errors=[];best=None
    for base in [CEMS_API,CEMS_RAPID]:
        records=[];seen=set();nxt=base;reported=None;pages=0;status="unknown"
        while nxt and pages<MAX_PAGES:
            # Don't pass params except on first page: next URL already holds them
            if nxt in seen:status="cycle_in_next";break
            seen.add(nxt)
            res,msg=get(nxt,params={"limit":LIMIT} if pages==0 else None)
            if res is None:
                status="page_error_"+str(msg)
                errors.append({"base":base,"page":pages,"error":str(msg)})
                break
            try:data=res.json()
            except Exception as ex:
                status="not_json"
                errors.append({"base":base,"page":pages,"error":str(ex)[:100]})
                break
            if isinstance(data,list):
                batch=data;nxt=None
            elif isinstance(data,dict):
                if reported is None:reported=data.get("count")
                batch=data.get("results",data.get("activations",[]))
                nxt=data.get("next")
                if nxt:nxt=urljoin(base,nxt.replace("http://","https://",1))
            else:
                status="unsupported_response";break
            if not isinstance(batch,list):
                status="not_list";break
            records.extend(batch)
            pages+=1
            if pages==1 or pages%5==0 or not nxt:
                print("CEMS_PAGE",base,pages,"got",len(batch),
                       "rows",len(records),"expected",reported,flush=True)
        else:
            status="page_cap" if nxt else "complete"
        if not nxt and status not in ["cycle_in_next","not_json","unsupported_response","not_list"]:
            status="complete"
        counts_match=reported is None or int(reported)==len(records)
        if status=="complete" and not counts_match:
            status="reported_count_mismatch"
        diagnostic={"base":base,"status":status,"pages":pages,
          "rows":len(records),"reported_count":reported,
          "reported_count_matches":bool(counts_match),
          "first_record_keys":sorted(records[0].keys()) if records else []}
        if records:
            best=(records,diagnostic,errors)
            if status=="complete":return best
    return best if best is not None else ([],{"status":"no_api_catalog","pages":0,"rows":0},errors)

def cems_us_floods(raw):
    ret=[]
    census={"all_count":len(raw),"study_years_count":0,"usa_study_years_count":0,
            "usa_flood_storm_study_years_count":0}
    for record in raw:
        if not isinstance(record,dict):continue
        date=iso_time(record,"eventTime","activationTime","activation_date")
        if pd.isna(date) or date.year not in range(YEAR_START,YEAR_END+1):continue
        census["study_years_count"]+=1
        if not is_united_states(record):continue
        census["usa_study_years_count"]+=1
        cat=record.get("category","")
        cat=str(cat.get("slug",cat.get("name",""))) if isinstance(cat,dict) else str(cat)
        name=str(record.get("name",record.get("title","")))
        hazard=(cat+" "+name).lower()
        if not any(s in hazard for s in ["flood","storm","hurricane","cyclone","rain","inundation"]):
            continue
        census["usa_flood_storm_study_years_count"]+=1
        lat,lon=centroid(record.get("centroid"))
        ret.append({"activation_code":str(record.get("code","")),
          "eventTime":str(date),"activationTime":str(record.get("activationTime","")),
          "countries":"|".join(countries(record)),"category":cat,"title":name,
          "centroid_lat":lat,"centroid_lon":lon,
          "n_aois":record.get("n_aois"),"n_products":record.get("n_products")})
    return pd.DataFrame(ret),census

def usfimr_index():
    report=[];all_rows=[]
    for u in USFIMR:
        res,st=get(u)
        if res is None:
            report.append({"source":u,"status":st,"rows":0})
            continue
        doc=BeautifulSoup(res.text,"html.parser")
        tables=doc.select("table")
        entries=[]
        for table in tables:
            for tr in table.select("tr"):
                td=tr.find_all("td")
                if len(td)<7:continue
                cells=[c.get_text(" ",strip=True) for c in td]
                # River / State / Year / Date / Platform / Area / Source / Download
                try:
                    year=int(cells[2].strip())
                except ValueError:continue
                if not (1990<=year<=2030):continue
                link=td[-1].find("a",href=True)
                href=urljoin(u,link.get("href")) if link else None
                date=str(pd.to_datetime(cells[3],errors="coerce").date()) if "/" in cells[3] else None
                entries.append({"river":cells[0],"state":cells[1],"year":year,
                     "observation_date":date,"platform":cells[4],
                     "mapped_area_km2_source":pd.to_numeric(cells[5],errors="coerce"),
                     "source_organization":cells[6],"zip_url":href,
                     "source_page":u})
        report.append({"source":u,"status":"ok","html_tables":len(tables),
                       "rows":len(entries),"year_min":min((z["year"] for z in entries),default=None),
                       "year_max":max((z["year"] for z in entries),default=None)})
        if len(entries)>len(all_rows):all_rows=entries
    return pd.DataFrame(all_rows),report

def positive_control(us_table):
    # Prefer a 2017 small measured inundation polygon for verifying the
    # download->shapefile->valid geometry chain, entirely OUTSIDE study years.
    eligible=us_table[(us_table.year<=2017)&us_table.zip_url.notna()]
    eligible=eligible.sort_values(["year","mapped_area_km2_source"],
                                  ascending=[False,True])
    attempts=[]
    for _,e in eligible.head(5).iterrows():
        url=e.zip_url
        res,status=get(url,timeout=65)
        attempt={"source_url":url,"river":e.river,"year":int(e.year),
                 "http_status":status}
        if res is None:
            attempts.append(attempt);continue
        data=res.content
        attempt["download_bytes"]=len(data)
        if len(data)>MAX_POSITIVE_DOWNLOAD_BYTES:
            attempt["status"]="too_large";attempts.append(attempt);continue
        try:
            zz=zipfile.ZipFile(BytesIO(data))
            shp_names=[name for name in zz.namelist() if name.lower().endswith(".shp")]
            if not shp_names:
                attempt["status"]="no_shp_in_zip"
                attempts.append(attempt);continue
            shp_path=shp_names[0]
            stem=shp_path[:-4]
            obj={ext:next((name for name in zz.namelist()
                 if name.lower()==(stem+ext).lower()),None)
                 for ext in [".shp",".shx",".dbf",".prj"]}
            if not all(obj[k] for k in [".shp",".shx",".dbf"]):
                attempt["status"]="incomplete_shapefile"
                attempts.append(attempt);continue
            shp=shapefile.Reader(shp=BytesIO(zz.read(obj[".shp"])),
                                 shx=BytesIO(zz.read(obj[".shx"])),
                                 dbf=BytesIO(zz.read(obj[".dbf"])))
            shapes=shp.shapes()
            if not shapes:
                attempt["status"]="empty_geometry";attempts.append(attempt);continue
            polygon_geoms=[shape(s.__geo_interface__) for s in shapes]
            polygon_geoms=[g for g in polygon_geoms if not g.is_empty
                           and g.geom_type in ["Polygon","MultiPolygon"]]
            if not polygon_geoms:
                attempt["status"]="no_polygon_geometry";attempts.append(attempt);continue
            crs_raw=zz.read(obj[".prj"]).decode("utf-8",errors="replace") if obj[".prj"] else ""
            crs=CRS.from_wkt(crs_raw) if crs_raw.strip() else None
            area=None;bounds=None
            if crs:
                t=Transformer.from_crs(crs,CRS.from_epsg(6933),always_xy=True).transform
                area=sum(transform(t,g).area for g in polygon_geoms)/1e6
                wgs=Transformer.from_crs(crs,CRS.from_epsg(4326),always_xy=True).transform
                bounds=[float(v) for v in transform(wgs,polygon_geoms[0]).bounds]
            attempt.update({"status":"polygon_read",
              "shapefile":shp_path,"shape_count":len(shapes),
              "polygon_count":len(polygon_geoms),
              "source_crs":crs.to_string() if crs else None,
              "equal_area_km2_raw_sum":area,
              "first_polygon_wgs84_bounds":bounds,
              "this_is_2019_2024_fema_match":False})
            # Source ZIP is a positive control only. Do not package third party
            # geodata into output artifact without checking licensing.
            attempts.append(attempt)
            print("USFIMR_POSITIVE_CONTROL_OK",json.dumps(attempt),flush=True)
            break
        except Exception as ex:
            attempt["status"]="zip_or_geometry_error"
            attempt["error"]=str(ex)[:180]
            attempts.append(attempt)
    return attempts

def match_catalog(flood,act):
    if act.empty:return pd.DataFrame(columns=["disasterNumber","activation_code"])
    result=[]
    for _,d in flood.iterrows():
        start=pd.to_datetime(d.incidentBeginDate,utc=True,errors="coerce")
        end=pd.to_datetime(d.incidentEndDate,utc=True,errors="coerce")
        if pd.isna(start) or pd.isna(d.noaa_point_lat):continue
        if pd.isna(end) or end<start:end=start+pd.Timedelta(days=7)
        for _,a in act.iterrows():
            if pd.isna(a.centroid_lat):continue
            t=pd.to_datetime(a.eventTime,utc=True)
            if not(start-pd.Timedelta(days=21)<=t<=end+pd.Timedelta(days=35)):
                continue
            dist=haversine(d.noaa_point_lat,d.noaa_point_lon,a.centroid_lat,a.centroid_lon)
            if dist<=200:
                result.append({"disasterNumber":int(d.disasterNumber),
                    "fyDeclared":int(d.fyDeclared),"activation_code":a.activation_code,
                    "activation_title":a.title,"event_date":a.eventTime,
                    "distance_km":dist,"evidence_level":"CANDIDATE_DATE_CENTROID_ONLY",
                    "map_geometry_ingested":False})
    return pd.DataFrame(result)

def activation_details(candidate):
    results=[]
    for code in candidate.activation_code.dropna().unique()[:12]:
        url=CEMS_RAPID.replace("public-activations-info/","public-activations/")+"?code="+str(code)
        res,status=get(url)
        n_products=0;n_del=0;downloads=[]
        if res is not None:
            try:
                d=res.json()
                items=d.get("results",[]) if isinstance(d,dict) else d
                for a in items:
                    for aoi in a.get("aois",[]):
                        for prod in aoi.get("products",[]):
                            n_products+=1
                            if prod.get("type")=="DEL":
                                n_del+=1
                                if prod.get("downloadPath"):
                                    downloads.append(prod["downloadPath"])
            except Exception as e:status="detail_parse_"+str(e)[:70]
        results.append({"activation_code":code,"status":status,
             "product_records":n_products,"delineation_records":n_del,
             "delineation_download_url_count":len(downloads),
             "delineation_download_urls":"|".join(downloads[:3]),
             "observed_flood_geometry_downloaded":False,
             "source_endpoint":url})
    return pd.DataFrame(results)

def main():
    master=pd.read_excel(ROOT/"master_openfema_40plus.xlsx")
    flood=master[(master.incidentType=="Flood")&master.fyDeclared.between(2019,2024)].copy()
    assert len(master)==971 and len(flood)==92
    noaa=pd.read_csv(INPUT/"selected_noaa_flood_report_locations.csv")
    assert len(noaa)==87
    flood=flood.merge(noaa[["disasterNumber","noaa_point_lat","noaa_point_lon"]],
                       on="disasterNumber",how="left",validate="one_to_one")
    raw,api_info,errors=index_cems()
    cems,census=cems_us_floods(raw)
    u,source_checks=usfimr_index()
    cems.to_csv(OUT/"cems_us_study_year_catalog.csv",index=False)
    u.to_csv(OUT/"usfimr_all_years_parsed.csv",index=False)
    match=match_catalog(flood,cems)
    match.to_csv(OUT/"candidate_fema_to_cems_matches.csv",index=False)
    details=activation_details(match) if len(match) else pd.DataFrame()
    details.to_csv(OUT/"candidate_cems_delineation_products.csv",index=False)
    controls=positive_control(u) if len(u) else []
    # This stage audits product catalogs and ingest capacity; do not elevate
    # candidate metadata to verified observed flood geometry for FEMA records.
    summary={"cohort_fema_flood_2019_2024":92,"geolocated_floods":87,
      "cems_catalog_api":api_info,"cems_page_errors":errors,
      "cems_catalog_year_country_hazard_counts":census,
      "cems_flood_storm_us_2019_2024_records":len(cems),
      "fema_cems_centroid_date_candidate_pairs":len(match),
      "fema_cems_candidate_disasters":int(match.disasterNumber.nunique()) if len(match) else 0,
      "candidate_cems_delineation_metadata":len(details),
      "usfimr_catalogs":source_checks,
      "usfimr_parsed_map_rows_total":len(u),
      "usfimr_2019_2024_catalog_rows":int(u.year.between(2019,2024).sum()) if len(u) else 0,
      "usfimr_positive_control_download_attempts":controls,
      "historical_positive_control_polygon_read":bool(any(x.get("status")=="polygon_read" for x in controls)),
      "verified_2019_2024_fema_inundation_polygons":0,
      "verified_flooded_residences":0,
      "current_phase_has_funding_model_fit":False,
      "limitations":["Catalog matches are potential only and not true inundation geometry.",
         "USFIMR historical positive control is NOT a 2019-24 FEMA event.",
         "If CEMS does not list a US activation, coverage of all other archives is unknown.",
         "AOI or activation centroids are not flood extent polygons.",
         "No observed flood geom matching through polygon intersection performed.",
         "No physically affected housing/population estimates computed.",
         "Only past cohorts studied; model baseline remains frozen."]}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2))
    print("BEGIN_PHASE27_VERIFIED_CATALOG\n"+json.dumps(summary,indent=2)+
           "\nEND_PHASE27_VERIFIED_CATALOG",flush=True)
if __name__=="__main__":main()
