#!/usr/bin/env python3
"""Audit storm-linked availability of the separate sustained county outage endpoint."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from thesis_outage_observability import COUNTIES
from thesis_retrieve_sources import acquire
from thesis_tropical_wind_verification import EXTENSION, STORMS, forecast, join

CENSUS_FILE = "2017_Gaz_counties_national.zip"
CENSUS = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2017_Gazetteer/" + CENSUS_FILE
EPISODES = {key: (STORMS | EXTENSION)[key] for key in ("michael_2018", "laura_2020", "ian_2022", "idalia_2023")}
LAG = pd.Timedelta(6, unit="h")
REGIONAL_KM = 400


def county_points(raw):
    details = acquire("census_county_points_2017", CENSUS, CENSUS_FILE, raw)
    table = pd.read_csv(raw / CENSUS_FILE, sep="\t", encoding="latin-1", dtype={"GEOID": str})
    table.columns = table.columns.str.strip()
    chosen = table[table.GEOID.isin(COUNTIES)].rename(columns={
        "GEOID": "fips_code", "USPS": "state", "NAME": "name", "INTPTLAT": "latitude", "INTPTLONG": "longitude",
    })[["fips_code", "state", "name", "latitude", "longitude"]]
    assert len(chosen) == len(COUNTIES) and chosen.fips_code.is_unique
    return chosen.sort_values("fips_code"), details


def match(windows, points, raw):
    coverage, matches, manifest = [], [], []
    for storm, (year, basin, start, end) in EPISODES.items():
        pred, source = forecast(raw, storm, year, basin, start, end)
        pred = pred[pred.nominal_lead_hours.eq(24)].sort_values("valid_utc")
        manifest.append(source)
        left = pd.Timestamp(start, tz="UTC")
        right = pd.Timestamp(end, tz="UTC") + pd.DateOffset(days=1)
        event = windows[windows.onset_utc.between(left, right, inclusive="left")]
        matched = []
        for point in points.itertuples(index=False):
            m = event[event.fips_code.eq(point.fips_code)].sort_values("onset_utc")
            proxy = SimpleNamespace(station=point.fips_code, lat=point.latitude, lon=point.longitude)
            extent = join(pred, proxy, None)[[
                "valid_utc", "nominal_initial_utc", "forecast_center_distance_nm",
                "predicted_quadrant_34kt", "predicted_equal_area_circle_34kt",
            ]]
            joined = pd.merge_asof(m, extent, left_on="onset_utc", right_on="valid_utc",
                                  direction="backward", tolerance=LAG - pd.Timedelta(1, unit="ns"))
            joined["state"] = point.state
            joined["storm"] = storm
            joined["valid_to_entry_hours"] = (joined.onset_utc - joined.valid_utc).dt.total_seconds() / 3600
            joined["forecast_center_distance_km"] = joined.forecast_center_distance_nm * 1.852
            assert joined.loc[joined.valid_utc.notna(), "nominal_initial_utc"].le(joined.onset_utc).all()
            matched.append(joined)
        rows = pd.concat(matched, ignore_index=True)
        assert rows.window_id.is_unique
        matches.append(rows)
        for label, a in [
            ("no_recent_forecast", rows[rows.valid_utc.isna()]),
            ("within_400km", rows[rows.forecast_center_distance_km.le(REGIONAL_KM)]),
            ("beyond_400km", rows[rows.forecast_center_distance_km.gt(REGIONAL_KM)]),
        ]:
            for state, subset in [("all", a), *list(a.groupby("state", sort=True))]:
                complete = subset[subset.observed_complete]
                have = subset.valid_utc.notna()
                quadrant = subset.predicted_quadrant_34kt.eq(True)
                circle = subset.predicted_equal_area_circle_34kt.eq(True)
                positive = int(complete.last_3h_below.eq(True).sum())
                coverage.append({"storm": storm, "state": state, "stratum": label, "rows": len(subset),
                                 "counties": subset.fips_code.nunique(), "complete": len(complete),
                                 "sustained_below": positive, "not_sustained_below": len(complete) - positive,
                                 "quadrant_inside": int((have & quadrant).sum()), "circle_inside": int((have & circle).sum()),
                                 "geometry_disagrees": int((have & quadrant.ne(circle)).sum()),
                                 "complete_and_quadrant_inside": int((subset.observed_complete & have & quadrant).sum()),
                                 "complete_and_circle_inside": int((subset.observed_complete & have & circle).sum())})
    return pd.concat(matches, ignore_index=True), pd.DataFrame(coverage), manifest


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("raw", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    windows = pd.read_csv("research/continuous/sustained_outage/results/window_outcomes.csv", dtype={"fips_code": str},
                          parse_dates=["onset_utc"])
    assert windows.year.max() == 2023
    counties, census = county_points(args.raw)
    linked, counts, docs = match(windows, counties, args.raw)
    for name, table in [("matched_entries", linked), ("storm_support", counts),
                        ("county_points", counties), ("source_manifest", pd.DataFrame([census, *docs]))]:
        table.to_csv(args.output / f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
