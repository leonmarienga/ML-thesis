#!/usr/bin/env python3
"""Event-specific inundation footprint *catalog availability* audit.

This script does NOT claim observed inundation at any FEMA disaster yet.
Crosswalk three independent source catalogs:
  1. Copernicus EMS mapping historical activations (public REST JSON).
  2. Alabama USFIMR historically observed flood map catalog (HTML).
  3. NASA OPERA DSWx availability and temporal provenance (documented, not
     downloaded in this audit because Earthdata-protected granules may require
     login and a water-presence raster is NOT itself a flood change mask).

For Copernicus, only candidate activation matches based on US-country,
hazard flood/storm, activation date and distance to NOAA point. No assumption
an activation centroid equals AOI or inundation footprint. The returned
product_count is metadata, not an observed vector area or exposure estimate.

For USFIMR, match state code and observation date to FEMA incident dates.
A state+date match is unconfirmed; actual shapefile geometry must be read,
intersected, and temporally validated before event exposure calculations.

No FEMA funding target read for matching and no accepted model changed.
"""
from __future__ import annotations
from pathlib import Path
from datetime import timedelta
import json,re,time
from urllib.parse import urljoin
import numpy as np,pandas as pd,requests
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parents[1]
IN=ROOT/"audit_inputs"/"flood_footprint_catalog"
OUT=ROOT/"audit_outputs"/"flood_footprint_catalog"
OUT.mkdir(parents=True,exist_ok=True)
API="https://mapping.emergency.copernicus.eu/activations/api/activations/"
RAPID="https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations-info/"
USFIMR="https://sdml.ua.edu/usfimr/"
YEAR_START,YEAR_END=2019,2024
DIST_CUTOFF_KM=200.
MAX_CATALOG_PAGES=12
HEADERS={"User-Agent":"University research / Flood footprint retrospective availability study"}
US_CODES={"US","USA","UNITED STATES","UNITED STATES OF AMERICA",
          "U.S.A.","UNITED STATES (USA)"}
STATES={"AL":"ALABAMA","AK":"ALASKA","AZ":"ARIZONA","AR":"ARKANSAS",
"CA":"CALIFORNIA","CO":"COLORADO","CT":"CONNECTICUT","DE":"DELAWARE",
"FL":"FLORIDA","GA":"GEORGIA","HI":"HAWAII","ID":"IDAHO","IL":"ILLINOIS",
"IN":"INDIANA","IA":"IOWA","KS":"KANSAS","KY":"KENTUCKY","LA":"LOUISIANA",
"ME":"MAINE","MD":"MARYLAND","MA":"MASSACHUSETTS","MI":"MICHIGAN",
"MN":"MINNESOTA","MS":"MISSISSIPPI","MO":"MISSOURI","MT":"MONTANA",
"NE":"NEBRASKA","NV":"NEVADA","NH":"NEW HAMPSHIRE","NJ":"NEW JERSEY",
"NM":"NEW MEXICO","NY":"NEW YORK","NC":"NORTH CAROLINA","ND":"NORTH DAKOTA",
"OH":"OHIO","OK":"OKLAHOMA","OR":"OREGON","PA":"PENNSYLVANIA",
"RI":"RHODE ISLAND","SC":"SOUTH CAROLINA","SD":"SOUTH DAKOTA",
"TN":"TENNESSEE","TX":"TEXAS","UT":"UTAH","VT":"VERMONT","VA":"VIRGINIA",
"WA":"WASHINGTON","WV":"WEST VIRGINIA","WI":"WISCONSIN","WY":"WYOMING",
"PR":"PUERTO RICO","DC":"DISTRICT OF COLUMBIA"}
S=requests.Session();S.headers.update(HEADERS)

def req(url,params=None):
    error=None
    for a in range(3):
        try:
            r=S.get(url,params=params,timeout=(15,55))
            if r.status_code in [403,404,429]:
                return None,"HTTP_"+str(r.status_code)
            r.raise_for_status()
            return r,"ok"
        except Exception as e:
            error=str(e)[:160]
            time.sleep(1.2*(a+1))
    return None,error or "failure"

def parse_centroid(x):
    if isinstance(x,str):
        m=re.search(r"POINT\s*\(\s*([-0-9.]+)\s+([-0-9.]+)\s*\)",x,re.I)
        if m:return float(m.group(2)),float(m.group(1))
    return None,None

def haversine(lat1,lon1,lat2,lon2):
    R=6371.0088
    a,b,c,d=np.radians([lat1,lon1,lat2,lon2])
    q=np.sin((c-a)/2)**2+np.cos(a)*np.cos(c)*np.sin((d-b)/2)**2
    return float(2*R*np.arcsin(min(1,np.sqrt(q))))

def activation_date(x):
    for key in ["activationTime","start_date","activation_date","eventDate","date"]:
        z=pd.to_datetime(x.get(key),utc=True,errors="coerce")
        if pd.notna(z):return z
    return pd.NaT

def is_us(x):
    countries=x.get("countries",[]) or []
    if isinstance(countries,dict):countries=[countries]
    names=[]
    for entry in countries:
        if isinstance(entry,str):names.append(entry)
        elif isinstance(entry,dict):
            names.extend([str(entry.get(k,"")) for k in
              ["short_name","name","iso3","iso2","code"]])
    return any(z.strip().upper() in US_CODES for z in names),names

def source_activations():
    errors=[]
    for base in [API,RAPID]:
        rows=[]
        seen=set()
        nexturl=base
        for page in range(MAX_CATALOG_PAGES):
            if not nexturl or nexturl in seen:break
            seen.add(nexturl)
            params={"limit":200,"offset":0} if page==0 else None
            res,status=req(nexturl,params=params)
            if res is None:
                errors.append({"source":base,"page":page,"error":status})
                break
            try:data=res.json()
            except Exception as ex:
                errors.append({"source":base,"page":page,"error":"not_json:"+str(ex)[:120]})
                break
            if isinstance(data,dict):
                batch=data.get("results",data.get("activations",[]))
                nxt=data.get("next")
                if nxt and nxt.startswith("http://"):nxt="https://"+nxt[len("http://"):]
                nexturl=urljoin(base,nxt) if nxt else None
            elif isinstance(data,list):
                batch=data;nexturl=None
            else:
                errors.append({"source":base,"page":page,"error":"bad_result_shape"})
                break
            if not isinstance(batch,list):
                errors.append({"source":base,"page":page,"error":"missing_results"})
                break
            rows.extend(batch)
            print("COPERNICUS_CATALOG",base,page,len(batch),"total",len(rows),flush=True)
            if not batch:break
        if rows:
            return rows,base,errors
    return [],None,errors

def cems_index(raw):
    rows=[]
    for item in raw:
        if not isinstance(item,dict):continue
        ok,countries=is_us(item)
        date=activation_date(item)
        cat=item.get("category") or {}
        c=str(cat.get("slug",cat.get("name",""))) if isinstance(cat,dict) else str(cat)
        title=str(item.get("name",item.get("title","")))
        typ=(c+" "+title).lower()
        if not (pd.notna(date) and YEAR_START<=date.year<=YEAR_END):continue
        if not ok:continue
        if not any(k in typ for k in ["flood","storm","hurricane","cyclone",
                                      "inundation","heavy rain"]):continue
        lat,lon=parse_centroid(item.get("centroid"))
        rows.append({
         "activation_code":str(item.get("code",item.get("id",""))),
         "activation_utc":date.isoformat(),
         "hazard_category":c,
         "activation_name":title,
         "countries":"|".join(countries),
         "centroid_lat":lat,"centroid_lon":lon,
         "number_of_aois":item.get("n_aois"),
         "number_of_products":item.get("n_products"),
         "activation_footprint_polygon_retrieved":False})
    return pd.DataFrame(rows)

def match_cems(disasters,acts):
    if acts.empty:return pd.DataFrame()
    rows=[]
    for _,d in disasters.iterrows():
        if pd.isna(d.noaa_point_lat) or pd.isna(d.noaa_point_lon):continue
        begin=pd.to_datetime(d.incidentBeginDate,utc=True,errors="coerce")
        end=pd.to_datetime(d.incidentEndDate,utc=True,errors="coerce")
        if pd.isna(begin):continue
        if pd.isna(end) or end<begin:end=begin+pd.Timedelta(days=7)
        for _,a in acts.iterrows():
            if pd.isna(a.centroid_lat) or pd.isna(a.centroid_lon):continue
            day=pd.to_datetime(a.activation_utc,utc=True)
            # Broad window for SOURCE CANDIDATES, NOT evidence of same event.
            if day < begin-pd.Timedelta(days=21) or day > end+pd.Timedelta(days=35):
                continue
            distance=haversine(d.noaa_point_lat,d.noaa_point_lon,
                               a.centroid_lat,a.centroid_lon)
            if distance>DIST_CUTOFF_KM:continue
            rows.append({
              "disasterNumber":int(d.disasterNumber),"fyDeclared":int(d.fyDeclared),
              "fema_state":str(d.state),
              "fema_begin":begin.isoformat(),"fema_end":end.isoformat(),
              "noaa_lat":float(d.noaa_point_lat),"noaa_lon":float(d.noaa_point_lon),
              "activation_code":str(a.activation_code),
              "activation_utc":str(a.activation_utc),
              "activation_title":str(a.activation_name),
              "activation_category":str(a.hazard_category),
              "distance_noaa_point_to_activation_centroid_km":distance,
              "evidence_stage":"POSSIBLE_CATALOG_MATCH_ONLY",
              "inundation_polygon_intersected":False,
              "event_attribution_validated":False
            })
    return pd.DataFrame(rows)

def usfimr_catalog():
    r,status=req(USFIMR)
    if r is None:return [],"unavailable_"+status
    soup=BeautifulSoup(r.text,"html.parser")
    output=[]
    for table in soup.select("table"):
        for tr in table.select("tr"):
            tds=tr.find_all(["td"])
            if len(tds)<6:continue
            cells=[x.get_text(" ",strip=True) for x in tds]
            yearmatch=re.search(r"\b(20\d{2})\b",cells[2])
            if not yearmatch:continue
            year=int(yearmatch.group(1))
            if not (YEAR_START<=year<=YEAR_END):continue
            date=pd.to_datetime(cells[3],errors="coerce")
            link=tr.find("a",href=True)
            output.append({"river_name":cells[0],"state_text":cells[1],
               "year":year,"map_observation_date":str(date.date()) if pd.notna(date) else None,
               "platform":cells[4],
               "mapped_area_km2_reported":pd.to_numeric(cells[5],errors="coerce"),
               "download_url":urljoin(USFIMR,link.get("href")) if link else None,
               "footprint_geometry_retrieved":False})
    return output,"ok"

def usfimr_matches(d,rows):
    matched=[]
    for _,dis in d.iterrows():
        start=pd.to_datetime(dis.incidentBeginDate,utc=True,errors="coerce")
        end=pd.to_datetime(dis.incidentEndDate,utc=True,errors="coerce")
        if pd.isna(start):continue
        if pd.isna(end) or end<start:end=start+pd.Timedelta(days=7)
        state=str(dis.state).upper().strip()
        for r in rows:
            listed=str(r["state_text"]).upper().replace(" ","")
            states=[x.strip() for x in re.split(r"[,/;]|-",listed)]
            acceptable=(state in states or (state in STATES and STATES[state] in states))
            if not acceptable:continue
            date=pd.to_datetime(r["map_observation_date"],utc=True,errors="coerce")
            if pd.isna(date):continue
            if start-pd.Timedelta(days=7)<=date<=end+pd.Timedelta(days=21):
                matched.append({
                 "disasterNumber":int(dis.disasterNumber),
                 "fyDeclared":int(dis.fyDeclared),
                 "state":state,
                 "usfimr_river_name":r["river_name"],
                 "usfimr_obs_date":r["map_observation_date"],
                 "usfimr_reported_inundation_km2":r["mapped_area_km2_reported"],
                 "usfimr_download_url":r["download_url"],
                 "evidence_stage":"STATE_DATE_CANDIDATE_ONLY",
                 "disaster_footprint_intersection_verified":False})
    return pd.DataFrame(matched)

def main():
    m=pd.read_excel(ROOT/"master_openfema_40plus.xlsx")
    flood=m[(m.incidentType=="Flood")&m.fyDeclared.between(2019,2024)].copy()
    assert len(m)==971 and len(flood)==92
    noaa=pd.read_csv(IN/"selected_noaa_flood_report_locations.csv")
    assert len(noaa)==87 and noaa.disasterNumber.nunique()==87
    d=flood[["disasterNumber","fyDeclared","state","incidentBeginDate",
             "incidentEndDate"]].merge(noaa[["disasterNumber","noaa_point_lat",
                 "noaa_point_lon"]],on="disasterNumber",how="left",validate="one_to_one")
    raw,src,errs=source_activations()
    cems=cems_index(raw)
    if len(cems):cems.to_csv(OUT/"copernicus_us_flood_activation_catalog.csv",index=False)
    hit=match_cems(d,cems)
    if len(hit):hit.to_csv(OUT/"copernicus_fema_candidate_matches.csv",index=False)
    maps,usstatus=usfimr_catalog()
    pd.DataFrame(maps).to_csv(OUT/"usfimr_2019_2024_catalog.csv",index=False)
    us_matches=usfimr_matches(d,maps)
    if len(us_matches):us_matches.to_csv(OUT/"usfimr_fema_candidate_matches.csv",index=False)
    matched_ids=set(hit.disasterNumber) if len(hit) else set()
    matched_ids|=set(us_matches.disasterNumber) if len(us_matches) else set()
    audit=d.copy()
    audit["cems_candidate"]=audit.disasterNumber.isin(hit.disasterNumber) if len(hit) else False
    audit["usfimr_candidate"]=audit.disasterNumber.isin(us_matches.disasterNumber) if len(us_matches) else False
    audit["verified_observed_inundation_polygon"]=False
    audit["event_footprint_exposure_computed"]=False
    audit.to_csv(OUT/"all_92_fema_flood_footprint_source_audit.csv",index=False)
    report={
      "fema_flood_cohort":92,"noaa_geolocated":87,
      "copernicus_api_source":src,"copernicus_api_errors":errs,
      "copernicus_flood_storm_activations_in_usa_2019_24":int(len(cems)),
      "fema_floods_with_copernicus_catalog_candidate":len(set(hit.disasterNumber)) if len(hit) else 0,
      "copernicus_candidate_pairs":len(hit),
      "copernicus_radius_km":DIST_CUTOFF_KM,
      "usfimr_source_status":usstatus,
      "usfimr_2019_24_catalog_event_rows":len(maps),
      "fema_floods_with_usfimr_state_date_candidate":len(set(us_matches.disasterNumber)) if len(us_matches) else 0,
      "usfimr_candidate_pairs":len(us_matches),
      "fema_floods_with_candidate_from_either_source":len(matched_ids),
      "fema_floods_with_verified_observed_footprints":0,
      "fema_floods_with_verified_footprint_exposure":0,
      "model_training_performed":False,
      "critical_interpretations":[
        "A catalog activation centroid match, even in date window, is NOT proof of flood polygon intersection.",
        "USFIMR state/date match is NOT proof the observed mapped area belongs to FEMA declaration.",
        "All unmatched entries are no verified catalog candidate, not proof no flood occurred or no map exists.",
        "A FEMA hazard-zone floodplain map describes probabilistic risk, NOT observed event-time inundation.",
        "OPERA DSWx-HLS validated records since April 2023 (radar S1 from Sep 2024); observed surface water must be differenced against permanent/background water and quality-masked to approximate flooding.",
        "NASA OPERA observation granules may require Earthdata access; no raster ingested in this step.",
        "No exposure data intersected. County population cannot be labelled people physically flooded.",
        "The accepted 830/912 router and $5.015M amount stack remain unchanged."
      ]}
    (OUT/"summary.json").write_text(json.dumps(report,indent=2))
    print("BEGIN_FLOOD_FOOTPRINT_CATALOG")
    print(json.dumps(report,indent=2))
    print("END_FLOOD_FOOTPRINT_CATALOG")
if __name__=="__main__":main()
