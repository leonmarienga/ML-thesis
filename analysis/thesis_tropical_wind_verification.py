#!/usr/bin/env python3
"""Verify nominal operational NHC wind contours against later NOAA coastal observations."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Geod
import requests

from thesis_retrieve_sources import acquire

STORMS = {
    "laura_2020": (2020, "13", "2020-08-25", "2020-08-28"),
    "ida_2021": (2021, "09", "2021-08-28", "2021-08-31"),
    "ian_2022": (2022, "09", "2022-09-27", "2022-09-30"),
}
EXTENSION = {
    "harvey_2017": (2017, "09", "2017-08-24", "2017-08-29"),
    "irma_2017": (2017, "11", "2017-09-09", "2017-09-13"),
    "maria_2017": (2017, "15", "2017-09-18", "2017-09-23"),
    "michael_2018": (2018, "14", "2018-10-08", "2018-10-12"),
    "idalia_2023": (2023, "10", "2023-08-29", "2023-09-01"),
    "helene_2024": (2024, "09", "2024-09-25", "2024-09-30"),
}
STATION = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json?type=met"
MEASURE = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
GEODESY = Geod(ellps="WGS84")
KNOT_MS = 1852 / 3600


def coordinate(text):
    factor = -1 if text[-1] in "SW" else 1
    return factor * int(text[:-1]) / 10.


def forecast(raw, name, year, basin_id, start, end):
    file=f"aal{basin_id}{year}.dat.gz"
    link=f"https://ftp.nhc.noaa.gov/atcf/archive/{year}/{file}"
    detail=acquire("nhc_atcf_archive",link,file,raw)
    rows=[]
    with gzip.open(raw/file,"rt",errors="replace") as stream:
        for packed in csv.reader(stream,skipinitialspace=True):
            values=[item.strip() for item in packed]
            if len(values)<17 or values[4]!="OFCL" or values[5] not in ("24","48") or values[11]!="34":
                continue
            initial=pd.to_datetime(values[2],format="%Y%m%d%H",utc=True)
            lead=int(values[5])
            verified=initial+timedelta(hours=lead)
            if not pd.Timestamp(start,tz="UTC")<=verified<pd.Timestamp(end,tz="UTC")+pd.DateOffset(days=1):
                continue
            method=values[12]
            if method=="AAA" and values[13].isdigit():
                radius=np.repeat(int(values[13]),4)
            elif method in ("NEQ","SEQ","SWQ","NWQ") and all(v.isdigit() for v in values[13:17]):
                rotation=("NEQ","SEQ","SWQ","NWQ").index(method)
                radius=np.roll(np.array([int(v) for v in values[13:17]]),rotation)
            else:
                continue
            rows.append({"storm":name,"nominal_initial_utc":initial,"nominal_lead_hours":lead,
                         "valid_utc":verified,"lat":coordinate(values[6]),"lon":coordinate(values[7]),
                         "threshold_knots":34,**dict(zip(("ne","se","sw","nw"),radius,strict=True))})
    result=pd.DataFrame(rows).drop_duplicates().sort_values(["valid_utc","nominal_lead_hours"])
    assert not result.duplicated(["valid_utc","nominal_lead_hours"]).any()
    return result,detail


def station_candidates(stations, predictions):
    result=[]
    for station in stations.itertuples(index=False):
        n=len(predictions)
        _, _, distance=GEODESY.inv(predictions.lon.to_numpy(), predictions.lat.to_numpy(),
                                   np.full(n,station.lng),np.full(n,station.lat))
        best=min(distance)/1000
        if best<=350:
            result.append({"station":str(station.id),"name":station.name,"lat":station.lat,
                           "lon":station.lng,"state":station.state,"min_forecast_center_distance_km":best})
    return pd.DataFrame(result).sort_values(["min_forecast_center_distance_km","station"]).head(12)


def station_winds(storm,start,end,station,raw):
    key=str(station.station)
    q={"begin_date":start.replace("-","")+" 00:00","end_date":end.replace("-","")+" 23:59",
       "station":key,"product":"wind","units":"metric","time_zone":"gmt","interval":"6","format":"json"}
    url=requests.Request("GET",MEASURE,params=q).prepare().url
    name=f"{storm}_{key}_wind.json"
    try:
        detail=acquire("noaa_coops_wind",url,name,raw)
    except requests.RequestException as e:
        return None,None,f"http_{getattr(e.response,'status_code','unknown')}"
    source=json.loads((raw/name).read_text())
    if not source.get("data"):
        return None,detail,"no_source_records"
    assert source["metadata"]["id"]==key
    approx_km=GEODESY.inv(float(source["metadata"]["lon"]),float(source["metadata"]["lat"]),
                          station.lon,station.lat)[2]/1000
    if approx_km>1:
        return None,detail,"station_location_inconsistent"
    wind=pd.DataFrame(source["data"])
    wind["observed_utc"]=pd.to_datetime(wind.t,utc=True)
    wind["wind_ms"]=pd.to_numeric(wind.s,errors="coerce")
    ok=wind.f.str.fullmatch(r"0(?:,0)*") & wind.wind_ms.between(0,85)
    selected=wind.loc[ok,["observed_utc","wind_ms"]].sort_values("observed_utc")
    assert selected.observed_utc.is_unique
    return selected,detail,"read"


def join(predictions,station,wind):
    p=predictions.copy()
    if wind is not None and not wind.empty:
        p=pd.merge_asof(p,wind,left_on="valid_utc",right_on="observed_utc",direction="nearest",
                        tolerance=pd.Timedelta(timedelta(minutes=15)))
    else:
        p=p.assign(observed_utc=pd.NaT,wind_ms=np.nan)
    n=len(p)
    bearings,_,distance=GEODESY.inv(p.lon.to_numpy(),p.lat.to_numpy(),np.full(n,station.lon),np.full(n,station.lat))
    quadrants=(np.mod(bearings,360)//90).astype(int)
    radii=p[["ne","se","sw","nw"]].to_numpy()
    selected_radii=radii[np.arange(n),quadrants]
    area_radius=np.sqrt((radii**2).mean(axis=1))
    distance_nm=distance/1852
    obs_knots=p.wind_ms/KNOT_MS
    return p.assign(station=station.station,station_lat=station.lat,station_lon=station.lon,
        forecast_center_distance_nm=distance_nm,forecast_bearing_degrees=np.mod(bearings,360),
        observed_knots=obs_knots,observed_34kt=obs_knots.ge(34).where(obs_knots.notna()),
        predicted_quadrant_34kt=(distance_nm<=selected_radii)&(selected_radii>0),
        predicted_equal_area_circle_34kt=(distance_nm<=area_radius)&(area_radius>0))


def evaluate(table):
    subsets=[("all",0,table),*((storm,lead,g) for (storm,lead),g in
                                               table.groupby(["storm","nominal_lead_hours"]))]
    output=[]
    for storm,lead,rows in subsets:
        real=rows[rows.observed_knots.notna()]
        actual=real.observed_knots.ge(34)
        for variant in ("quadrant","equal_area_circle"):
            flag=real[f"predicted_{variant}_34kt"]
            tp,fp,fn,tn=(int((a & b).sum()) for a,b in [(actual,flag),(~actual,flag),(actual,~flag),(~actual,~flag)])
            output.append({"storm":storm,"nominal_lead_hours":lead,"extent":variant,"station_time_pairs":len(rows),
                          "observed_pairs":len(real),"observed_stations":real.station.nunique(),
                          "tp":tp,"fp":fp,"fn":fn,"tn":tn,"sensitivity":tp/(tp+fn) if tp+fn else np.nan,
                          "precision":tp/(tp+fp) if tp+fp else np.nan})
    return pd.DataFrame(output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--independent-extension", action="store_true")
    args=p.parse_args()
    args.raw.mkdir(parents=True,exist_ok=True)
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=[acquire("noaa_coops_meteorological_sites",STATION,"coops_sites.json",args.raw)]
    stations=pd.DataFrame(json.loads((args.raw/"coops_sites.json").read_text())["stations"])
    station_log, matched=[],[]
    cohort = EXTENSION if args.independent_extension else STORMS
    for storm, (year,basin_id,start,end) in cohort.items():
        rows,detail=forecast(args.raw,storm,year,basin_id,start,end)
        manifest.append(detail)
        selected=station_candidates(stations,rows)
        roster=list(selected.itertuples(index=False))
        with ThreadPoolExecutor(5) as pool:
            winds=pool.map(lambda station:station_winds(storm,start,end,station,args.raw),roster)
            for ref,(obs,source,state) in zip(roster,winds,strict=True):
                if source:
                    manifest.append(source)
                station_log.append({"storm":storm,**ref._asdict(),"status":state,
                                    "good_station_samples":len(obs) if obs is not None else 0})
                matched.append(join(rows,ref,obs))
    paired=pd.concat(matched,ignore_index=True)
    metric=evaluate(paired)
    for name, value in [("station_audit",pd.DataFrame(station_log)),("wind_match_cases",paired),
                        ("station_contingencies",metric),("source_manifest",pd.DataFrame(manifest))]:
        value.to_csv(args.output/f"{name}.csv",index=False)
    print(f"Compared {paired.observed_knots.notna().sum()} observed station-time pairs across {paired.storm.nunique()} storms.")


if __name__=="__main__":
    main()
