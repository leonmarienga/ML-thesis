#!/usr/bin/env python3
"""Freeze a forecast-selected county panel and screen sustained outage support."""
import argparse
from contextlib import ExitStack
import gzip
from hashlib import md5, sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from thesis_county_wind_exposure import CENSUS, CENSUS_FILE, LAG, REGIONAL_KM
from thesis_outage_observability import valid_county_rows
from thesis_retrieve_sources import acquire
from thesis_sustained_outage_transfer import county_spells
from thesis_tropical_wind_verification import GEODESY, forecast

FRAME = (
    ("Florence", "06", "2018-09-14"), ("Michael", "14", "2018-10-10"),
    ("Hanna", "08", "2020-07-25"), ("Isaias", "09", "2020-08-04"),
    ("Laura", "13", "2020-08-27"), ("Sally", "19", "2020-09-16"),
    ("Delta", "26", "2020-10-09"), ("Zeta", "28", "2020-10-28"),
    ("Ian", "09", "2022-09-28"), ("Nicole", "17", "2022-11-10"),
    ("Idalia", "10", "2023-08-30"),
)
PREVIOUSLY_STUDIED = {"Michael", "Laura", "Ian", "Idalia"}
DAY = pd.Timedelta(1, unit="D")
TCR_PREFIX = "https://www.nhc.noaa.gov/data/tcr/"
MAX_COUNTIES = 60


def bounds(date):
    landfall = pd.Timestamp(date, tz="UTC")
    return landfall, landfall - 2 * DAY, landfall + 3 * DAY


def county_points(raw):
    detail = acquire("census_county_points_2017", CENSUS, CENSUS_FILE, raw)
    data = pd.read_csv(raw / CENSUS_FILE, sep="\t", encoding="latin-1", dtype={"GEOID": str})
    data.columns = data.columns.str.strip()
    selected = data[data.INTPTLAT.between(24, 50) & data.INTPTLONG.between(-125, -65)].rename(
        columns={"GEOID": "fips_code", "USPS": "state", "NAME": "county_name",
                 "INTPTLAT": "point_latitude", "INTPTLONG": "point_longitude"}
    )[["fips_code", "state", "county_name", "point_latitude", "point_longitude"]]
    assert selected.fips_code.is_unique
    return selected, detail


def storms(raw):
    for name, number, date in FRAME:
        landfall, start, end = bounds(date)
        year = landfall.year
        archive = f"AL{number}{year}_{name}.pdf"
        tcr = acquire("nhc_cyclone_report", TCR_PREFIX + archive, archive, raw)
        assert (raw / archive).read_bytes().startswith(b"%PDF-")
        track, info = forecast(raw, name, year, number, str(start.date()), str((end - DAY).date()))
        selected = track[track.nominal_lead_hours.eq(24)]
        assert len(selected) and selected.valid_utc.is_unique
        yield name, year, landfall, start, end, selected, info, tcr


def geography(track, points):
    pairs = points.merge(track, how="cross")
    bearing, _, distance = GEODESY.inv(pairs.lon.to_numpy(), pairs.lat.to_numpy(),
                                       pairs.point_longitude.to_numpy(), pairs.point_latitude.to_numpy())
    radii = pairs[["ne", "se", "sw", "nw"]].to_numpy()
    radius = radii[np.arange(len(radii)), (np.mod(bearing, 360) // 90).astype(int)]
    round_radius = np.sqrt((radii**2).mean(axis=1))
    pairs["distance_km"] = distance / 1000
    pairs["quadrant"] = (radius > 0) & (distance <= radius * 1852)
    pairs["circle"] = (round_radius > 0) & (distance <= round_radius * 1852)
    return pairs


def build_panel(raw):
    points, detail = county_points(raw)
    panels, evidence = [], [detail]
    for name, year, landfall, start, end, track, info, tcr in storms(raw):
        earlier = track[track.nominal_initial_utc.le(landfall)]
        pairs = geography(earlier, points)
        selected = pairs.groupby(["fips_code", "state", "county_name", "point_latitude", "point_longitude"], as_index=False).agg(
            minimum_design_distance_km=("distance_km", "min"), ever_inside_design_quadrant=("quadrant", "any"),
            ever_inside_design_circle=("circle", "any"),
        )
        selected["priority_any_design_contour"] = selected.ever_inside_design_quadrant | selected.ever_inside_design_circle
        selected = selected[selected.minimum_design_distance_km.le(REGIONAL_KM)].sort_values(
            ["priority_any_design_contour", "minimum_design_distance_km", "fips_code"], ascending=[False, True, True]
        ).head(MAX_COUNTIES).copy()
        selected["selection_rank"] = np.arange(1, len(selected) + 1)
        panels.append(selected.assign(storm=name, year=year, landfall_utc=landfall, start_utc=start, stop_utc=end,
                                      design_forecasts=len(earlier), previously_studied=name in PREVIOUSLY_STUDIED))
        evidence.extend([info, tcr])
    result = pd.concat(panels, ignore_index=True)
    assert result.storm.nunique() == len(FRAME) and not result.duplicated(["storm", "fips_code"]).any()
    return result, pd.DataFrame(evidence)


def extract(national, file_card, panel, slices):
    provenance = json.loads(file_card.read_text())
    for year, group in panel.groupby("year"):
        national_file = national / f"eaglei_outages_{year}.csv"
        card = provenance[str(year)]
        assert national_file.stat().st_size == card["annual_bytes"]
        by_date, paths, counts = {}, {}, {}
        for name, subset in group.groupby("storm"):
            county_keys = set(subset.fips_code.str.encode("ascii"))
            start = subset.start_utc.iloc[0] - 2 * DAY
            end = subset.stop_utc.iloc[0] + DAY
            for date in pd.date_range(start, end, freq="D", inclusive="left"):
                by_date.setdefault(date.strftime("%Y-%m-%d").encode("ascii"), []).append((name, county_keys))
            paths[name] = slices / f"{name}_{year}.csv.gz"
            counts[name] = 0
        checksum, sha = md5(usedforsecurity=False), sha256()
        with national_file.open("rb") as stream, ExitStack() as stack:
            sinks = {name: stack.enter_context(gzip.GzipFile(filename="", fileobj=stack.enter_context(path.open("wb")),
                                                               mode="wb", mtime=0)) for name, path in paths.items()}
            header = stream.readline()
            expected = b"sum" if year == 2023 else b"customers_out"
            assert header.strip().split(b",") == [b"fips_code", b"county", b"state", expected, b"run_start_time"]
            checksum.update(header)
            sha.update(header)
            for w in sinks.values():
                w.write(header)
            for line in stream:
                checksum.update(line)
                sha.update(line)
                items = by_date.get(line.rsplit(b",", 1)[-1].strip()[:10])
                if items:
                    county = line.split(b",", 1)[0].zfill(5)
                    for name, keys in items:
                        if county in keys:
                            sinks[name].write(line)
                            counts[name] += 1
        assert checksum.hexdigest() == card["annual_md5"] and sha.hexdigest() == card["annual_sha256"]
        for name, path in paths.items():
            record = {"storm": name, "year": int(year), "source": path.name, "rows": counts[name],
                      "source_sha256": sha.hexdigest(), "slice_sha256": sha256(path.read_bytes()).hexdigest()}
            (slices / f"{name}_{year}.json").write_text(json.dumps(record, indent=2) + "\n")


def evaluate(raw, slices, panel, exclude_zeros):
    match_rows, summaries, clock_rows = [], [], []
    for name, year, landfall, start, end, track, _, _ in storms(raw):
        counties = panel[panel.storm.eq(name)]
        path = slices / f"{name}_{year}.csv.gz"
        source = json.loads((slices / f"{name}_{year}.json").read_text())
        assert sha256(path.read_bytes()).hexdigest() == source["slice_sha256"]
        audits, observed = valid_county_rows(slices, year=year, source=path.name, counties=tuple(counties.fips_code))
        clock_rows.append(audits.drop(columns=f"usable_fraction_of_{year}_quarters").assign(storm=name, year=year))
        if exclude_zeros:
            observed = [(fips, r[r.customers_out.ne(0)]) for fips, r in observed]
        windows, _, _ = county_spells(year, observed)
        windows = windows[windows.onset_utc.between(start, end, inclusive="left")].drop(columns="sequence_row", errors="ignore")
        features = geography(track, counties[["fips_code", "state", "point_latitude", "point_longitude"]])
        results = []
        for fips, rows in windows.groupby("fips_code"):
            local = features[features.fips_code.eq(fips)].sort_values("valid_utc")
            matched = pd.merge_asof(rows.sort_values("onset_utc"), local[[
                "valid_utc", "nominal_initial_utc", "state", "distance_km", "quadrant", "circle",
            ]], left_on="onset_utc", right_on="valid_utc", direction="backward", tolerance=LAG - pd.Timedelta(1, unit="ns"))
            results.append(matched)
        assert results, name
        m = pd.concat(results, ignore_index=True)
        assert m.loc[m.valid_utc.notna(), "nominal_initial_utc"].lt(m.onset_utc).all()
        m["stratum"] = np.select([m.valid_utc.isna(), m.distance_km.le(REGIONAL_KM)], ["no_recent_forecast", "regional"],
                                 default="remote")
        m["storm"] = name
        match_rows.append(m)
        complete = m[m.observed_complete & m.stratum.eq("regional")]
        total = len(complete)
        positive = int(complete.last_3h_below.eq(True).sum())
        quadrant = int(complete.quadrant.eq(True).sum())
        gate = (total >= 12 and complete.fips_code.nunique() >= 4 and min(positive, total - positive) >= 3
                and min(quadrant, total - quadrant) >= 2)
        summaries.append({"storm": name, "year": year, "period": "prior" if year <= 2020 else "held",
                          "previously_studied": name in PREVIOUSLY_STUDIED, "selected_counties": len(counties),
                          "verified_entries": len(m), "no_recent_forecast": int(m.valid_utc.isna().sum()),
                          "remote_entries": int(m.stratum.eq("remote").sum()), "regional_entries": int(m.stratum.eq("regional").sum()),
                          "complete_regional": total, "complete_regional_counties": complete.fips_code.nunique(),
                          "sustained_positive": positive, "sustained_negative": total - positive,
                          "complete_in_quadrant": quadrant, "complete_out_quadrant": total - quadrant,
                          "complete_in_circle": int(complete.circle.eq(True).sum()),
                          "complete_geometry_disagreements": int(complete.quadrant.ne(complete.circle).sum()),
                          "storm_gate_pass": gate})
    support = pd.DataFrame(summaries)
    gates = []
    for population, selected in [("all_storms", support), ("new_storms", support[~support.previously_studied])]:
        earlier = int((selected.period.eq("prior") & selected.storm_gate_pass).sum())
        held = int((selected.period.eq("held") & selected.storm_gate_pass).sum())
        gates.append({"population": population, "qualifying_prior_storms": earlier, "qualifying_held_storms": held,
                      "gate_pass": earlier >= 4 and held >= 2})
    return pd.concat(match_rows, ignore_index=True), support, pd.DataFrame(gates), pd.concat(clock_rows, ignore_index=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=["panel", "evaluate", "extract"])
    cli.add_argument("raw", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--panel", type=Path)
    cli.add_argument("--slices", type=Path)
    cli.add_argument("--national", type=Path)
    cli.add_argument("--file-card", type=Path)
    cli.add_argument("--exclude-zeros", action="store_true")
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "panel":
        panel, manifest = build_panel(args.raw)
        panel.to_csv(args.output / "storm_county_panel.csv", index=False)
        manifest.to_csv(args.output / "source_manifest.csv", index=False)
        return
    panel = pd.read_csv(args.panel, dtype={"fips_code": str}, parse_dates=["start_utc", "stop_utc"])
    if args.mode == "extract":
        args.slices.mkdir(parents=True, exist_ok=True)
        extract(args.national, args.file_card, panel, args.slices)
        return
    matches, support, gate, clocks = evaluate(args.raw, args.slices, panel, args.exclude_zeros)
    for name, table in [("matched_windows", matches), ("storm_support", support), ("gate_summary", gate), ("clock_audit", clocks)]:
        table.to_csv(args.output / f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
