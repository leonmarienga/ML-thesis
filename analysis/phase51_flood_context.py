#!/usr/bin/env python3
"""Compare DR4722 window context with independently recorded daily river levels."""
import argparse
import json
from pathlib import Path

import pandas as pd
from pyproj import Geod
import requests

from thesis_retrieve_sources import acquire

API = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/daily/items"
STATIONS = {"USGS-05227500": "Mississippi River at Aitkin",
            "USGS-05385000": "Root River near Houston"}
GEOD = Geod(ellps="WGS84")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase50", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.output / "raw"
    raw.mkdir(exist_ok=True)
    daily, manifest = [], []
    for station, parameter, year in ((s, v, y) for s in STATIONS for v in ("00060", "00065") for y in (2022, 2023, 2024)):
        query = {"f": "json", "monitoring_location_id": station, "parameter_code": parameter,
                 "statistic_id": "00003", "time": f"{year}-03-01/{year}-04-30", "limit": 100}
        url = requests.Request("GET", API, params=query).prepare().url
        filename = f"{station}_{parameter}_{year}.geojson"
        record = acquire("usgs_daily_ogc", url, filename, raw)
        manifest.append(record)
        data = json.loads((raw / filename).read_text())
        assert not any(l["rel"] == "next" for l in data.get("links", []))
        for feature in data["features"]:
            props = feature["properties"]
            daily.append({"id": feature["id"], "station_name": STATIONS[station],
                          "lon": feature["geometry"]["coordinates"][0],
                          "lat": feature["geometry"]["coordinates"][1], **props})
    pd.DataFrame(manifest).to_csv(args.output/"dr4722_gauge_source_manifest.csv", index=False)
    d = pd.DataFrame(daily).drop_duplicates("id")
    assert not d.duplicated(["monitoring_location_id", "parameter_code", "time"]).any()
    d["value"] = pd.to_numeric(d.value, errors="coerce")
    d["year"] = pd.to_datetime(d.time).dt.year
    d["month"] = pd.to_datetime(d.time).dt.month
    d.to_csv(args.output/"dr4722_daily_gauges.csv", index=False)
    valid = d.query("approval_status == 'Approved'").dropna(subset=["value"])
    peak = valid.loc[valid.groupby(["monitoring_location_id", "parameter_code", "year", "month"]).value.idxmax()]
    peak[["monitoring_location_id", "station_name", "parameter_code", "unit_of_measure", "year", "month", "time", "value"]].to_csv(
        args.output/"dr4722_gauge_monthly_peaks.csv", index=False)
    sites = pd.read_csv(args.phase50/"phase50_site_comparisons.csv").query(
        "disasterNumber == 4722 and design == 'seasonal' and denominator == 'common'").copy()
    for station, rows in d.groupby("monitoring_location_id"):
        lon, lat = rows[["lon", "lat"]].iloc[0]
        sites[f"km_to_{station}"] = [GEOD.inv(x, y, lon, lat)[2]/1000 for x, y in zip(sites.anchor_lon, sites.anchor_lat)]
    sites.to_csv(args.output/"dr4722_window_context.csv", index=False)
    print(peak[["station_name", "parameter_code", "time", "value"]].to_string(index=False))


if __name__ == "__main__":
    main()
