#!/usr/bin/env python3
"""Assess independent earthquake impact support before attempting physical-to-impact transfer."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Geod
import requests

from thesis_retrieve_sources import acquire

API = "https://www.ngdc.noaa.gov/hazel/hazard-service/api/v1/earthquakes"
WGS84 = Geod(ellps="WGS84")
IMPACT = ("damageMillionsDollars", "damageMillionsDollarsTotal", "housesDestroyed", "housesDestroyedTotal",
          "deaths", "deathsTotal")
ORDERS = ("damageAmountOrder", "damageAmountOrderTotal", "housesDestroyedAmountOrder",
          "housesDestroyedAmountOrderTotal", "deathsAmountOrder", "deathsAmountOrderTotal")


def collect(raw):
    pages, rows, initial_size = [], [], None
    for page in range(1, 21):
        query = {"minYear": 1998, "maxYear": 2024, "page": page, "itemsPerPage": 100}
        url = requests.Request("GET", API, params=query).prepare().url
        name = f"noaa_impact_catalog_{page}.json"
        pages.append(acquire("noaa_ncei_significant_earthquakes", url, name, raw))
        response = json.loads((raw / name).read_text())
        assert int(response["page"]) == page
        rows.extend(response["items"])
        if initial_size is None:
            initial_size = response["totalItems"]
        else:
            assert response["totalItems"] == initial_size
        if page == response["totalPages"]:
            break
    assert len(rows) == initial_size
    result = pd.DataFrame(rows)
    assert result.year.between(1998, 2024).all() and result.id.is_unique
    return result, pd.DataFrame(pages)


def date_fields(catalog):
    dates = pd.to_datetime(catalog[["year", "month", "day"]], errors="coerce", utc=True)
    clock = pd.to_numeric(catalog.hour, errors="coerce").between(0, 23)
    clock &= pd.to_numeric(catalog.minute, errors="coerce").between(0, 59)
    seconds = pd.to_numeric(catalog.second, errors="coerce").fillna(0)
    time = (dates + pd.to_timedelta(catalog.hour.fillna(0), unit="h")
            + pd.to_timedelta(catalog.minute.fillna(0), unit="m")
            + pd.to_timedelta(seconds, unit="s"))
    return dates, time.where(clock)


def overlaps(catalog, earthquakes):
    actual = catalog.copy()
    actual["day"], actual["utc"] = date_fields(actual)
    actual["magnitude"] = actual.eqMagMw.fillna(actual.eqMagnitude)
    geocoded = actual[actual.latitude.notna() & actual.longitude.notna()]
    result = []
    for e in earthquakes.itertuples():
        when = pd.Timestamp(e.origin_utc)
        same_day = geocoded[geocoded.day.eq(when.normalize())]
        within = []
        for candidate in same_day.itertuples():
            km = WGS84.inv(e.lon, e.lat, candidate.longitude, candidate.latitude)[2] / 1000
            mag_gap = abs(e.mag-candidate.magnitude) if np.isfinite(candidate.magnitude) else None
            delay = abs((when-candidate.utc).total_seconds()/60) if pd.notna(candidate.utc) else None
            if km <= 50 and (mag_gap is None or mag_gap <= .5) and (delay is None or delay <= 30):
                within.append((candidate, km, mag_gap, delay))
        summary = {"event_id": e.event_id, "episode": e.episode, "origin_utc": e.origin_utc,
                   "scored_instruments": e.qualified_stations, "candidate_count": len(within),
                   "status": "unique_match" if len(within)==1 else "ambiguous" if within else "no_match"}
        if len(within) == 1:
            a, km, gap, minutes = within[0]
            summary.update({"noaa_event_id": a.id, "distance_km": km, "magnitude_gap": gap,
                            "minutes_apart": minutes, "noaa_location": a.locationName,
                            **{k: getattr(a, k) for k in (*IMPACT, *ORDERS)}})
        result.append(summary)
    table = pd.DataFrame(result)
    links = table[table.status.eq("unique_match")]
    repeated = set(links[links.noaa_event_id.duplicated(keep=False)].noaa_event_id)
    if repeated:
        table.loc[table.noaa_event_id.isin(repeated), "status"] = "non_unique_source_event"
    return table


def coverage(events):
    summarized = []
    for place, sample in [("global", events), *events.groupby("country")]:
        for label in IMPACT:
            cases = sample[label]
            summarized.append({"population": place, "field": label, "records": len(cases),
                               "cadence": "all_years_1998_to_2024", "numeric_present": int(cases.notna().sum()),
                               "strictly_positive": int(cases.gt(0).sum()), "documented_zero": int(cases.eq(0).sum())})
    return pd.DataFrame(summarized)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.raw.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    events, source = collect(args.raw)
    earthquake = pd.read_csv("research/continuous/earthquake/inputs/earthquake_sources.csv")
    result = overlaps(events, earthquake)
    assert earthquake.origin_utc.max() < "2024-10-01"
    col = ["id", "year", "month", "day", "hour", "minute", "second", "latitude", "longitude", "country",
           "eqDepth", "eqMagnitude", "eqMagMw", "intensity", *IMPACT, *ORDERS]
    data = {"impact_catalog_fields.csv": events[col], "impact_coverage.csv": coverage(events),
            "comcat_impact_links.csv": result, "source_manifest.csv": source}
    for name, frame in data.items():
        frame.to_csv(args.output / name, index=False)
    print(f"Retained {len(events):,} published source events and screened {len(earthquake)} USGS earthquake candidates.")


if __name__ == "__main__":
    main()
