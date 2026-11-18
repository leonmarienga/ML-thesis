#!/usr/bin/env python3
"""Match complete NOAA nClimGrid-Daily final county precipitation to incidents."""
import argparse
import calendar
from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path
import re

import numpy as np
import pandas as pd
import requests

from thesis_retrieve_sources import acquire

BASE = "https://www.ncei.noaa.gov/data/nclimgrid-daily/access/averages"
HAZARDS = {"Hurricane", "Tropical Storm", "Typhoon", "Coastal Storm", "Flood", "Severe Storm", "Tornado"}


def load_month(period, cache):
    tag = period.strftime("%Y%m")
    root = f"{BASE}/{period.year}"
    source = []
    for filename in (f"prcp-{tag}-cty-scaled.csv", f"ncdd-{tag}-version.txt"):
        try:
            row = acquire("ncei_nclimgrid", f"{root}/{filename}", filename, cache)
            source.append({**row, "month": str(period), "status": "retrieved"})
        except requests.RequestException as exc:
            source.append({"month": str(period), "filename": filename, "status": type(exc).__name__,
                           "download_url": f"{root}/{filename}"})
    if any(x["status"] != "retrieved" for x in source):
        return period, None, source
    note = (cache/f"ncdd-{tag}-version.txt").read_text()
    production = re.search(r"(?:created|last modified) on (\d{4}-\d{2}-\d{2})", note)
    upstream = re.search(r"(?:downloaded|provided) on ([A-Z][a-z]{2} [A-Z][a-z]{2} +\d{1,2} \d{2}:\d{2}:\d{2} \d{4})", note)
    documented_dates = [pd.Timestamp(x) for x in (production[1] if production else None, upstream[1] if upstream else None) if x]
    latest_documented = max(documented_dates) if documented_dates else pd.NaT
    software = re.search(r"software (\S+)", note)
    archive_statement = " ".join(note.splitlines()[0].split())
    for row in source:
        row.update(final_creation_or_revision_date=production[1] if production else None, upstream_data_date=upstream[1] if upstream else None,
                   processing_software=software[1] if software else None, archive_statement=archive_statement)
    if not (" complete for " in note.lower() or " final for " in note.lower()):
        return period, None, source
    n = calendar.monthrange(period.year, period.month)[1]
    all_counties = {}
    with (cache/f"prcp-{tag}-cty-scaled.csv").open(encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) != 37 or row[0] != "cty" or row[5] != "PRCP":
                raise ValueError(f"Wrong nClimGrid format for {period}: {row[:6]}")
            assert (int(row[3]), int(row[4])) == (period.year, period.month)
            county = int(row[1])
            assert county not in all_counties
            values = np.array(row[6:6+n], dtype=float)
            all_counties[county] = np.where((values >= 0) & (values <= 1200), values, np.nan)
    assert len(all_counties) > 3000
    return period, (all_counties, latest_documented), source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cohort", type=Path)
    p.add_argument("counties", type=Path)
    p.add_argument("cache", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.cohort, parse_dates=["start", "end", "declaration"])
    assert data.fyDeclared.max() == 2024
    data = data[data.incidentType.isin(HAZARDS)]
    counties = pd.read_csv(args.counties).groupby("disasterNumber").county_fips.agg(list)
    valid = data[data.end.notna() & data.end.ge(data.start) & data.disasterNumber.isin(counties.index)]
    dates = sorted({month for z in valid.itertuples(index=False) for month in pd.period_range(z.start, z.end, freq="M")})
    with ThreadPoolExecutor(max_workers=10) as pool:
        files = list(pool.map(lambda m: load_month(m, args.cache), dates))
    sources = {month: payload for month, payload, _ in files}
    manifesto = pd.DataFrame(e for _, _, rows in files for e in rows)
    manifesto.to_csv(args.output/"nclimgrid_sources.csv", index=False)
    records, coverage = [], []
    for z in data.itertuples(index=False):
        declared = counties.get(z.disasterNumber, [])
        complete = []
        if pd.notna(z.start) and pd.notna(z.end) and z.start <= z.end:
            period = pd.date_range(z.start, z.end)
            groups = pd.Series(period).groupby(period.to_period("M")).agg(list)
            for county in sorted(set(declared)):
                obs, release = [], []
                for month, days in groups.items():
                    source = sources.get(month)
                    if source is None or county not in source[0]:
                        break
                    mm, published = source
                    values = mm[county][[day.day-1 for day in days]]
                    obs.extend(values)
                    release.append(published)
                if len(obs) != len(period) or not np.isfinite(obs).all():
                    continue
                complete.append({"disasterNumber": z.disasterNumber, "county_fips": county,
                                 "observation_start": str(z.start.date()), "observation_end": str(z.end.date()),
                                 "days": len(obs), "max_1day_mm": max(obs), "total_mm": sum(obs),
                                 "last_documented_source_date": max((r for r in release if pd.notna(r)), default=pd.NaT), "documented_date_before_t0": max((r for r in release if pd.notna(r)), default=pd.NaT) < z.declaration,
                                 "documented_date_before_t72": max((r for r in release if pd.notna(r)), default=pd.NaT) < z.declaration+pd.Timedelta(days=3)})
        records.extend(complete)
        coverage.append({"disasterNumber": z.disasterNumber, "incidentType": z.incidentType,
                         "clim_declared_counties": len(declared), "clim_complete_counties": len(complete),
                         "clim_climate_release_date": max((c["last_documented_source_date"] for c in complete), default=None)})
    raw = pd.DataFrame(records)
    raw.to_csv(args.output/"nclimgrid_county_incidents.csv", index=False)
    output = pd.DataFrame(coverage)
    values = raw.groupby("disasterNumber").agg(clim_max_1day_mm=("max_1day_mm", "max"),
                                              clim_max_county_total_mm=("total_mm", "max"))
    output = output.join(values, on="disasterNumber")
    output["clim_all_declared_counties"] = output.clim_complete_counties.gt(0) & output.clim_complete_counties.eq(output.clim_declared_counties)
    output.to_csv(args.output/"precipitation_features.csv", index=False)
    print(output.groupby("incidentType").agg(cases=("disasterNumber","size"),
          sourced=("clim_max_1day_mm","count"), all_counties=("clim_all_declared_counties","sum")).to_string(), flush=True)


if __name__ == "__main__":
    main()
