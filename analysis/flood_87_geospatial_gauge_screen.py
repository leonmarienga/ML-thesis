#!/usr/bin/env python3
"""Screen all 2019-24 county-linked FEMA Flood cases for USGS station proximity.

Each station needs to meet:
  - NOAA Flood/Flash Flood EVENT_ID matched to this FEMA disaster (pre-existing
    target-blind matching); NOAA event STATE_FIPS + CZ_FIPS = USGS station FIPS.
  - Valid NOAA BEGIN/END lat/lon point AND USGS gauge coordinates.
  - Distance threshold chosen before outcome evaluation (<=50 km; <=25 km strong).
  - Station HUC recorded for future basin-intersection review; no claim of basin
    membership match or event attribution based only on location.
  - No funding outcome read at any stage.

Selecting maximum two event counties per disaster by geolocated NOAA event
count reduces API cost, not a representative national survey. No zeros imputed.
NOAA event reports may be finalized post-disaster; not a t0-safe source.
"""
import json, math, re, time, gzip
from pathlib import Path
from io import StringIO
from collections import defaultdict
import numpy as np, pandas as pd, requests

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/"audit_inputs"/"flood_geo_screen"
OUT=ROOT/"audit_outputs"/"flood_geo_screen"
CACHE=ROOT/".cache"/"flood_geo_screen"
OUT.mkdir(parents=True,exist_ok=True)
CACHE.mkdir(parents=True,exist_ok=True)
NOAA="https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
USGS="https://waterservices.usgs.gov/nwis/site/"
S=requests.Session()
S.headers["User-Agent"]="Academic-FEMA-NOAA-USGS-event-location-screening/1.0"
YEARS=range(2018,2025)
MAX_COUNTIES_PER_DISASTER=2
MAX_STATIONS_PER_COUNTY=80
SCREEN_KM=50.0
STRONG_KM=25.0
MAX_EVENT_COORDS_PER_COUNTY=150

def request(url,params=None,local=None):
    if local and local.exists() and local.stat().st_size>100:
        return local.read_bytes()
    for n in range(4):
        try:
            r=S.get(url,params=params,timeout=(15,140))
            r.raise_for_status()
            data=r.content
            if local:local.write_bytes(data)
            return data
        except Exception as e:
            print("RETRY",url,n,str(e)[:120],flush=True)
            time.sleep(2*(n+1))
    raise RuntimeError("Unavailable source: "+url)

def event_index(audit):
    ids=set()
    for text in audit.noaa_event_ids.fillna("").astype(str):
        for token in text.split("|"):
            if token.strip().isdigit():ids.add(int(token))
    html=request(NOAA,local=CACHE/"index.html").decode("utf8",errors="replace")
    results=[]
    for y in YEARS:
        p=rf"StormEvents_details-ftp_v1\.0_d{y}_c\d+\.csv\.gz"
        names=sorted(set(re.findall(p,html)))
        if not names:
            print("NOAA_NO_YEAR",y,flush=True)
            continue
        filename=names[-1]
        gz=request(NOAA+filename,local=CACHE/filename)
        cols={"EVENT_ID","STATE_FIPS","CZ_TYPE","CZ_FIPS","BEGIN_LAT","BEGIN_LON","END_LAT","END_LON","EVENT_TYPE"}
        try:
            frame=pd.read_csv(StringIO(gzip.decompress(gz).decode("utf8",errors="replace")),
                     usecols=lambda c:c in cols,low_memory=False)
        except Exception as e:
            print("NOAA_PARSE_ERROR",y,str(e)[:160],flush=True)
            continue
        missing=cols-set(frame.columns)
        if missing:
            print("NOAA_MISSING_COLUMNS",y,list(missing),flush=True)
            continue
        frame["EVENT_ID"]=pd.to_numeric(frame.EVENT_ID,errors="coerce")
        chosen=frame.loc[frame.EVENT_ID.isin(ids)].copy()
        chosen=chosen[chosen.CZ_TYPE.astype(str).str.upper()=="C"]
        chosen=chosen[chosen.EVENT_TYPE.astype(str).str.lower().isin(["flood","flash flood"])]
        chosen["st"]=pd.to_numeric(chosen.STATE_FIPS,errors="coerce")
        chosen["co"]=pd.to_numeric(chosen.CZ_FIPS,errors="coerce")
        chosen=chosen[chosen.st.between(1,79)&chosen.co.between(1,999)].copy()
        chosen["fips"]=chosen.st.astype(int).astype(str).str.zfill(2)+chosen.co.astype(int).astype(str).str.zfill(3)
        for prefix in ["BEGIN","END"]:
            chosen[prefix+"_LAT"]=pd.to_numeric(chosen[prefix+"_LAT"],errors="coerce")
            chosen[prefix+"_LON"]=pd.to_numeric(chosen[prefix+"_LON"],errors="coerce")
        # Country bounds, including AK, PR and HI. Extreme coordinates omitted.
        for pref in ["BEGIN","END"]:
            ok=chosen[pref+"_LAT"].between(14,72)&chosen[pref+"_LON"].between(-172,-64)
            chosen.loc[~ok,[pref+"_LAT",pref+"_LON"]]=np.nan
        results.append(chosen[["EVENT_ID","fips","BEGIN_LAT","BEGIN_LON","END_LAT","END_LON"]])
        print("NOAA_YEAR",y,"matched county event rows",len(chosen),flush=True)
    if not results:raise RuntimeError("No NOAA event location data available")
    return pd.concat(results,ignore_index=True).drop_duplicates(["EVENT_ID","fips"])

def station_list(fips):
    filename=CACHE/("stations_"+fips+".rdb")
    try:
        raw=request(USGS,params={"format":"rdb","countyCd":fips,"siteType":"ST",
             "parameterCd":"00060","siteStatus":"all","siteOutput":"expanded"},local=filename)
        text=raw.decode("utf8",errors="replace")
        table=pd.read_csv(StringIO("\n".join(x for x in text.splitlines() if not x.startswith("#"))),
                          sep="\t",dtype=str)
        if "site_no" not in table or "dec_lat_va" not in table or "dec_long_va" not in table:
            return pd.DataFrame(),"site_missing_coordinates"
        table=table[table.site_no.fillna("").str.fullmatch(r"\d{8,15}")].copy()
        table["lat"]=pd.to_numeric(table.dec_lat_va,errors="coerce")
        table["lon"]=pd.to_numeric(table.dec_long_va,errors="coerce")
        table=table[table.lat.between(14,72)&table.lon.between(-172,-64)].copy()
        table["meta_fips"]=table.state_cd.fillna("").str.zfill(2)+table.county_cd.fillna("").str.zfill(3)
        table=table[table.meta_fips==fips]
        table=table.drop_duplicates("site_no").sort_values("site_no").head(MAX_STATIONS_PER_COUNTY)
        return table,"ok"
    except Exception as e:
        return pd.DataFrame(), "site_query_error_"+str(e)[:80]

def distance_km(la,lo,lat_arr,lon_arr):
    p1=np.radians(float(la));p2=np.radians(lat_arr)
    dp=np.radians(lat_arr-float(la))
    dl=np.radians(lon_arr-float(lo))
    a=np.sin(dp/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 6371.0088*2*np.arcsin(np.minimum(1,np.sqrt(a)))

def main():
    m=pd.read_excel(ROOT/"master_openfema_40plus.xlsx")
    assert len(m)==971
    flood=m[(m.incidentType.astype(str)=="Flood")&m.fyDeclared.between(2019,2024)].copy()
    assert len(flood)==92
    county=pd.read_csv(INPUT/"event_county_census_matches.csv",dtype={"fips":str})
    audits=pd.read_csv(INPUT/"external_match_audit.csv")
    a=audits[audits.disasterNumber.isin(flood.disasterNumber)][["disasterNumber","noaa_event_ids"]].copy()
    a["EVENT_ID"]=a.noaa_event_ids.fillna("").astype(str).str.split("|")
    a=a.explode("EVENT_ID")
    a["EVENT_ID"]=pd.to_numeric(a.EVENT_ID,errors="coerce")
    a=a.dropna(subset=["EVENT_ID"])
    a["EVENT_ID"]=a.EVENT_ID.astype("int64")
    events=event_index(audits[audits.disasterNumber.isin(flood.disasterNumber)])
    merged=a[["disasterNumber","EVENT_ID"]].merge(events,on="EVENT_ID",how="inner")
    merged=merged.merge(county[["disasterNumber","fips"]].drop_duplicates(),
                        on=["disasterNumber","fips"],how="inner")
    assert merged.disasterNumber.nunique()<=len(flood)
    # Each NOAA coordinate is only a point report, not the inundation footprint.
    point_rows=[]
    for _,x in merged.iterrows():
        for pref in ["BEGIN","END"]:
            if pd.notna(x[pref+"_LAT"]) and pd.notna(x[pref+"_LON"]):
                point_rows.append({"disasterNumber":int(x.disasterNumber),
                    "EVENT_ID":int(x.EVENT_ID),"fips":x.fips,
                    "lat":float(x[pref+"_LAT"]),"lon":float(x[pref+"_LON"]),
                    "point_type":pref})
    locations=pd.DataFrame(point_rows)
    if locations.empty:raise RuntimeError("NOAA geolocated flood events not found")
    locations=locations.drop_duplicates()
    ranked=(locations.groupby(["disasterNumber","fips"]).size().reset_index(name="n_points")
       .sort_values(["disasterNumber","n_points","fips"],ascending=[True,False,True]))
    ranked["county_rank"]=ranked.groupby("disasterNumber").cumcount()+1
    picked=ranked[ranked.county_rank<=MAX_COUNTIES_PER_DISASTER]
    selected=locations.merge(picked[["disasterNumber","fips","n_points","county_rank"]],
       on=["disasterNumber","fips"],validate="many_to_one")
    site_cache={}
    pairs=[]
    for (dn,fips),part in selected.groupby(["disasterNumber","fips"]):
        if fips not in site_cache:site_cache[fips]=station_list(fips)
        stations,status=site_cache[fips]
        if stations.empty:
            pairs.append({"disasterNumber":int(dn),"county_fips":fips,
               "site_status":status,"candidate_station_id":None,
               "station_to_nearest_noaa_report_km":None,"proximity_50km":False,
               "hydrological_attribution_validated":False})
            continue
        la=part.lat.to_numpy(float);lo=part.lon.to_numpy(float)
        # Candidate station ranking within each matching NOAA county.
        for _,st in stations.iterrows():
            dist=distance_km(st.lat,st.lon,la,lo)
            k=int(np.argmin(dist))
            pairs.append({"disasterNumber":int(dn),"county_fips":fips,
              "site_status":status,"candidate_station_id":str(st.site_no),
              "station_name":st.get("station_nm"),
              "station_lat":float(st.lat),"station_lon":float(st.lon),
              "station_huc":st.get("huc_cd"),
              "drainage_area_sqmi":st.get("drain_area_va"),
              "station_to_nearest_noaa_report_km":float(dist[k]),
              "nearest_noaa_EVENT_ID":int(part.iloc[k].EVENT_ID),
              "nearest_noaa_point_type":str(part.iloc[k].point_type),
              "noaa_point_lat":float(la[k]),"noaa_point_lon":float(lo[k]),
              "noaa_geo_points_in_county":len(part),
              "proximity_25km":bool(dist[k]<=STRONG_KM),
              "proximity_50km":bool(dist[k]<=SCREEN_KM),
              "hydrological_attribution_validated":False})
        print("SCREENED",dn,fips,"stations",len(stations),"NOAA points",len(part),flush=True)
    allsites=pd.DataFrame(pairs)
    assert len(allsites)>0
    allsites.to_csv(OUT/"station_noaa_point_proximity_all_candidates.csv",index=False)
    relevant=allsites[allsites.candidate_station_id.notna()].sort_values(
      ["disasterNumber","station_to_nearest_noaa_report_km","candidate_station_id"])
    top=relevant.groupby("disasterNumber").head(3)
    top.to_csv(OUT/"top3_candidate_stations_by_disaster.csv",index=False)
    summary={
      "all_recent_fema_floods":len(flood),
      "floods_with_original_noaa_county_match":int(county[county.disasterNumber.isin(flood.disasterNumber)].disasterNumber.nunique()),
      "floods_with_geolocated_noaa_flood_report":int(locations.disasterNumber.nunique()),
      "floods_with_picked_geolocated_county":int(picked.disasterNumber.nunique()),
      "counties_screened":len(site_cache),
      "floods_with_station_metadata":int(relevant.disasterNumber.nunique()),
      "floods_with_gauge_within_25km_of_report":int(allsites.loc[allsites.proximity_25km.fillna(False),"disasterNumber"].nunique()),
      "floods_with_gauge_within_50km_of_report":int(allsites.loc[allsites.proximity_50km.fillna(False),"disasterNumber"].nunique()),
      "total_station_candidate_pairs":int(relevant.shape[0]),
      "recommended_gauge_status":"PROXIMITY SCREEN ONLY, not watershed or flood impact validation",
      "sample_policy":"Top 2 geolocated event counties by count for each FEMA Flood, up to 80 stations per county; no funding target used.",
      "source_limitations":["NOAA points may represent reported event locations and not actual flood inundation polygons.",
        "Same county and <=25/50 km proximity is not enough to demonstrate shared river basin, flood causality, or exposure.",
        "No cross-year gauge baseline/quantile test performed in this spatial screen.",
        "Station coordinates and NOAA final data may have been revised after FEMA decision cutoff.",
        "Missing event coordinates or stations are missing data, never zero severity.",
        "Next stage: select stations for hydrologic HUC and network topology inspection, then calculate anomalies for high-confidence candidates."],
      "accepted_model_changes":False}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2))
    print("BEGIN_FLOOD_GEOSPATIAL_SCREEN\n"+json.dumps(summary,indent=2)+"\nEND_FLOOD_GEOSPATIAL_SCREEN",flush=True)
if __name__=="__main__":main()
