#!/usr/bin/env python3
"""NASA Daymet precipitation audit for *all 92* FY2019–24 FEMA Flood records.

Geolocate NOAA Flood / Flash Flood event points from pre-existing
target-blind FEMA↔NOAA EVENT_ID matches. The rainfall location is a real
NOAA report coordinate closest to the median NOAA event coordinate for
each FEMA disaster (medoid approximation). This location is chosen without
using USGS gauge availability, which avoids the prior gauge-selection bias.
87 cases are expected to have NOAA geolocation; missing 5 remain missing.

Retrieve Daymet single-pixel 1 km station-informed daily precipitation
(NOT NASA GPM IMERG). Use capped 21-day FEMA incident window and ten
pre-incident calendar years for seasonal rainfall reference.

All data retrospective; no evidence of early-response as-of availability.
No FEMA dollar targets, model training or router changes.
"""
from __future__ import annotations
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import numpy as np,pandas as pd

import flood_87_geospatial_gauge_screen as geo
import flood24_daymet_precipitation_enrichment as rain

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"audit_inputs"/"daymet_flood87"
OUT=ROOT/"audit_outputs"/"daymet_flood87"
OUT.mkdir(parents=True,exist_ok=True)
MAX_WORKERS=5

def representative_points(flood,audit,county):
    one=audit[audit.disasterNumber.isin(flood.disasterNumber)].copy()
    a=one[["disasterNumber","noaa_event_ids"]].copy()
    a["EVENT_ID"]=a.noaa_event_ids.fillna("").astype(str).str.split("|")
    a=a.explode("EVENT_ID")
    a["EVENT_ID"]=pd.to_numeric(a.EVENT_ID,errors="coerce")
    a=a.dropna(subset=["EVENT_ID"]).copy()
    a["EVENT_ID"]=a.EVENT_ID.astype("int64")
    event=geo.event_index(one)
    valid=a[["disasterNumber","EVENT_ID"]].merge(event,on="EVENT_ID",how="inner")
    valid=valid.merge(county[["disasterNumber","fips"]].drop_duplicates(),
                      on=["disasterNumber","fips"],how="inner")
    valid=valid.drop_duplicates(["disasterNumber","EVENT_ID","fips"])
    valid["lat"]=valid.BEGIN_LAT.combine_first(valid.END_LAT)
    valid["lon"]=valid.BEGIN_LON.combine_first(valid.END_LON)
    valid=valid.dropna(subset=["lat","lon"])
    valid=valid[valid.lat.between(14,72)&valid.lon.between(-172,-64)]
    # Select an actual NOAA report location, not an artificial median point.
    selected=[]
    for disaster,g in valid.groupby("disasterNumber"):
        la=float(g.lat.median());lo=float(g.lon.median())
        gg=g.copy()
        gg["distance_to_median_km"]=geo.distance_km(
            la,lo,gg.lat.to_numpy(float),gg.lon.to_numpy(float))
        gg=gg.sort_values(["distance_to_median_km","EVENT_ID","fips"])
        best=gg.iloc[0]
        distances=geo.distance_km(float(best.lat),float(best.lon),
                                  gg.lat.to_numpy(float),gg.lon.to_numpy(float))
        selected.append({"disasterNumber":int(disaster),
           "noaa_point_lat":float(best.lat),"noaa_point_lon":float(best.lon),
           "noaa_event_id":int(best.EVENT_ID),
           "chosen_county_fips":str(best.fips).zfill(5),
           "noaa_event_count_geolocated":int(gg.EVENT_ID.nunique()),
           "noaa_report_location_count":int(len(gg)),
           "location_spread_max_km":float(np.max(distances)),
           "location_spread_p90_km":float(np.quantile(distances,.90)),
           "location_policy":"Actual NOAA Flood/Flash Flood report point nearest median coordinates among all event-linked locations; independent of gauge or FEMA funding."})
    return pd.DataFrame(selected),valid

def compute(fema,point):
    dummy=fema.copy()
    dummy["normalized_discharge_available"]=False
    proxy=point.copy()
    proxy["nearest_noaa_EVENT_ID"]=int(point.noaa_event_id)
    proxy["candidate_station_id"]="NOT_USED_FOR_RAIN"
    proxy["station_to_nearest_noaa_report_km"]=float("nan")
    # The original 24-case Daymet data algorithm is reused unchanged.
    r=rain.evaluate_case(dummy,proxy)
    r["rainfall_selection_does_not_use_gauge"]=True
    # The inherited pilot field was always a dummy, and must not be mistaken
    # for missing streamflow or an actual station.
    r.pop("original_flow_candidate",None)
    r.pop("reference_station_id",None)
    r.pop("noaa_point_to_station_km",None)
    return r

def main():
    master=pd.read_excel(ROOT/"master_openfema_40plus.xlsx")
    assert len(master)==971
    flood=master[(master.incidentType=="Flood")&
                 master.fyDeclared.between(2019,2024)].copy()
    assert len(flood)==92
    audit=pd.read_csv(SRC/"external_match_audit.csv")
    county=pd.read_csv(SRC/"event_county_census_matches.csv",dtype={"fips":str})
    loc,all_reports=representative_points(flood,audit,county)
    assert loc.disasterNumber.nunique()==87,loc.disasterNumber.nunique()
    print("GEOLOCATED",len(loc),"of",len(flood),
          "NOAA_REPORT_LOCATIONS",len(all_reports),flush=True)

    joined=flood[["disasterNumber","fyDeclared","incidentBeginDate","incidentEndDate"]].merge(
       loc,on="disasterNumber",how="left",validate="one_to_one",indicator=True)
    records=[]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        requests={}
        for _,row in joined.iterrows():
            if row._merge!="both":
                records.append({"disasterNumber":int(row.disasterNumber),
                 "fyDeclared":int(row.fyDeclared),
                 "rainfall_available":False,"rainfall_status":"no_noaa_location",
                 "rainfall_selection_does_not_use_gauge":True})
            else:
                requests[executor.submit(compute,row,row)]=int(row.disasterNumber)
        for task in as_completed(requests):
            dn=requests[task]
            try:r=task.result()
            except Exception as e:
                r={"disasterNumber":dn,"rainfall_available":False,
                   "rainfall_status":"unhandled_"+str(e)[:160]}
            records.append(r)
            print("RAIN",dn,r["rainfall_status"],
                  round(r.get("event_precip_total_mm",float("nan")),2),flush=True)

    observations=pd.DataFrame(records)
    assert len(observations)==92 and observations.disasterNumber.nunique()==92
    # Retain requested geographical provenance and original status.
    provenance=joined[["disasterNumber","fyDeclared","noaa_point_lat",
       "noaa_point_lon","noaa_event_id","chosen_county_fips",
       "noaa_event_count_geolocated","noaa_report_location_count",
       "location_spread_max_km","location_spread_p90_km","location_policy"]]
    measurements=provenance.merge(
       observations.drop(columns=["fyDeclared","noaa_point_lat","noaa_point_lon","noaa_event_id"],
                        errors="ignore"),
       on="disasterNumber",how="left",validate="one_to_one")
    measurements["rainfall_available"]=measurements.rainfall_available.fillna(False)
    measurements.sort_values("disasterNumber").to_csv(
       OUT/"all_92_flood_daymet_rainfall.csv",index=False)
    loc.to_csv(OUT/"selected_noaa_flood_report_locations.csv",index=False)
    byyear=measurements.groupby("fyDeclared").agg(
       cases=("disasterNumber","size"),with_noaa_point=("noaa_point_lat","count"),
       with_daymet=("rainfall_available","sum"))
    byyear.to_csv(OUT/"coverage_by_fiscal_year.csv")
    success=measurements[measurements.rainfall_available]
    report={
      "total_2019_2024_fema_floods":92,
      "geolocated_noaa_floods":len(loc),
      "rainfall_usable_cases":int(measurements.rainfall_available.sum()),
      "geolocated_but_rainfall_unusable":int(len(loc)-len(success)),
      "no_noaa_report_location":92-len(loc),
      "status_counts":{str(x):int(v) for x,v in
                       measurements.rainfall_status.value_counts().items()},
      "by_year":{str(y):{"n":int(v.cases),"geolocated":int(v.with_noaa_point),
          "with_daymet":int(v.with_daymet)} for y,v in byyear.iterrows()},
      "rain_total_mm_median":float(success.event_precip_total_mm.median()) if len(success) else None,
      "rain_total_mm_p90":float(success.event_precip_total_mm.quantile(.9)) if len(success) else None,
      "rain_max_3day_mm_median":float(success.event_max_3day_mm.median()) if len(success) else None,
      "noaa_selected_point_p90_disaster_location_spread_km":
           float(loc.location_spread_p90_km.quantile(.90)),
      "source":"NASA ORNL DAAC Daymet 1-km daily precipitation, NOT GPM IMERG",
      "fixed_feature_protocol":"Capped <=21-day incident window, max1-day, trailing3/7-day max, event total, pre-event 7-day precipitation, prior up-to-ten-year seasonal 3-day reference",
      "selection":"NOAA event-linked Flood/Flash Flood actual report point nearest medoid; independent of gauge availability and funding",
      "changed_funding_model":False,
      "model_comparison_performed":False,
      "interpretation_warnings":[
         "Daymet is a station-informed gridded estimate rather than an observed inundation footprint.",
         "One report point can be unrepresentative of a large FEMA disaster; inspect location_spread_p90_km.",
         "Matching NOAA event ID, report point and disaster is retrospective and not independently validated as-of declaration.",
         "Current Daymet data are retrospective/reprocessed and do not prove original response-time availability.",
         "Point-based rain cannot quantify affected area, flood depth, inundation damage or exposed homes.",
         "The original 24-case study selected rain points using gauge proximity; these 87-case results use gauge-independent report point selection, so individual case values may differ.",
         "Rainfall completeness rule min3 observed days and >=80pct event-window coverage excludes some legitimate short events. Missing is not zero rain.",
         "No change to frozen 830/912 router and ~$5.015M accepted point MAE."
      ]}
    (OUT/"summary.json").write_text(json.dumps(report,indent=2))
    print("BEGIN_FLOOD87_DAYMET")
    print(json.dumps(report,indent=2))
    print("END_FLOOD87_DAYMET")
if __name__=="__main__":
    main()
