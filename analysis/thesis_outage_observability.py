#!/usr/bin/env python3
"""Audit readout support for EAGLE-I county windows and Mesa utility events."""
import argparse
from hashlib import md5, sha256
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from thesis_retrieve_sources import acquire

NATIONAL_DOCUMENT = "eaglei_outages_2018.csv"
FIGSHARE = "https://api.figshare.com/v2/articles/24237376/versions/1"
COUNTY_SOURCE = "eaglei_2018_24_counties.csv.gz"
COUNTIES = tuple("01073 01097 01101 06037 06073 06075 12011 12086 12095 22033 22051 22071 27053 27123 27137 "
                 "34003 34013 34017 37119 37129 37183 48113 48201 48453".split())
COUNTY_START = pd.Timestamp("2018-01-01", tz="UTC")
COUNTY_END = pd.Timestamp("2019-01-01", tz="UTC")
MESA_START = pd.Timestamp("2014-01-01")
MESA_END = pd.Timestamp("2024-10-01")
PERIOD = pd.Timedelta(15, unit="min")
FUTURE_STEPS = 96
THRESHOLD = 1000
MESA_FIELDS = ("row_id,event_id,total_customers_impacted,start_date_time,end_date_time,"
               "major_event_flag,duration,partial_restoration_flag,cause_name,total_customers")


def retain_county_rows(national, raw):
    release = json.loads((raw / "figshare_release.json").read_text())
    assert release["doi"] == "10.6084/m9.figshare.24237376.v1"
    metadata = next(row for row in release["files"] if row["name"] == NATIONAL_DOCUMENT)
    assert national.stat().st_size == metadata["size"] == 999_210_542
    assert metadata["id"] == 42547879
    source_digest = sha256()
    original_digest = md5(usedforsecurity=False)
    fips_values = {value.encode() for value in COUNTIES}
    out = raw / COUNTY_SOURCE
    rows = 0
    with national.open("rb") as source, out.open("wb") as compressed:
        with gzip.GzipFile(fileobj=compressed, filename="", mode="wb", mtime=0) as selected:
            header = source.readline()
            assert header.strip().lower() == b"fips_code,county,state,customers_out,run_start_time"
            source_digest.update(header)
            original_digest.update(header)
            selected.write(header)
            for line in source:
                source_digest.update(line)
                original_digest.update(line)
                if line.split(b",", 1)[0].zfill(5) in fips_values:
                    selected.write(line)
                    rows += 1
    assert original_digest.hexdigest() == metadata["supplied_md5"]
    return {"annual_doi": release["doi"], "annual_file_id": metadata["id"], "annual_bytes": metadata["size"],
            "annual_md5": original_digest.hexdigest(), "annual_sha256": source_digest.hexdigest(),
            "annual_url": metadata["download_url"], "license": release["license"]["name"],
            "slice": COUNTY_SOURCE, "slice_rows": rows, "slice_sha256": sha256(out.read_bytes()).hexdigest()}


def mesa_url():
    base = "https://citydata.mesaaz.gov/resource/iyqi-xjyw.csv"
    where = f"start_date_time >= '{MESA_START.isoformat()}' and start_date_time < '{MESA_END.isoformat()}'"
    args = {"$select": MESA_FIELDS, "$where": where, "$limit": "50000", "$order": "start_date_time,row_id"}
    return requests.Request("GET", base, params=args).prepare().url


def valid_county_rows(raw):
    data = pd.read_csv(raw / COUNTY_SOURCE, dtype={"fips_code": "string"})
    data["fips_code"] = data.fips_code.str.zfill(5)
    data["utc"] = pd.to_datetime(data.run_start_time, format="mixed", errors="coerce", utc=True)
    data["customers_out"] = pd.to_numeric(data.customers_out, errors="coerce")
    assert set(data.fips_code) <= set(COUNTIES)
    audit, observations = [], []
    for fips in COUNTIES:
        subset = data[data.fips_code.eq(fips)]
        in_year = subset.utc.between(COUNTY_START, COUNTY_END, inclusive="left")
        on_grid = in_year & subset.utc.eq(subset.utc.dt.floor("15min"))
        selected = subset[on_grid]
        exact = selected[["utc", "customers_out"]].drop_duplicates()
        conflict = exact.duplicated("utc", keep=False)
        remaining = exact[~conflict]
        usable = remaining[remaining.customers_out.ge(0) & remaining.customers_out.mod(1).eq(0)]
        missing_count = remaining.customers_out.isna()
        malformed_count = ~missing_count & (remaining.customers_out.lt(0) | remaining.customers_out.mod(1).ne(0))
        audit.append({"fips_code": fips, "raw_rows": len(subset), "outside_or_invalid_clock_rows": int((~in_year).sum()),
                      "off_quarter_clock_rows": int((in_year & ~on_grid).sum()),
                      "exact_duplicate_count_rows": len(selected) - len(exact), "conflicting_snapshot_count": exact.loc[conflict, "utc"].nunique(),
                      "missing_or_nonnumeric_count_rows": int(missing_count.sum()),
                      "negative_or_fractional_count_rows": int(malformed_count.sum()), "usable_snapshot_count": len(usable),
                      "explicit_zero_count": int(usable.customers_out.eq(0).sum()),
                      "usable_fraction_of_2018_quarters": len(usable) / int((COUNTY_END - COUNTY_START) / PERIOD)})
        observations.append((fips, usable.sort_values("utc")))
    return pd.DataFrame(audit), observations


def threshold_windows(observations):
    entries = []
    for fips, rows in observations:
        dates = list(rows.utc)
        counts = rows.customers_out.to_numpy()
        for i in np.flatnonzero(counts >= THRESHOLD):
            known_prior = i > 0 and dates[i] - dates[i - 1] == PERIOD
            if known_prior and counts[i - 1] >= THRESHOLD:
                continue
            limit = dates[i] + FUTURE_STEPS * PERIOD
            if not known_prior:
                entries.append({"fips_code": fips, "onset_utc": dates[i], "onset_customers": counts[i],
                                "status": "unknown_predecessor", "observed_quarters_after_entry": None,
                                "observed_below_within_24h": None, "below_by_24h_complete_window": None,
                                "below_at_24h_complete_window": None, "reentered_after_observed_drop": None})
                continue
            j = rows.utc.searchsorted(limit, side="right")
            within = counts[i + 1:j]
            below = bool((within < THRESHOLD).any())
            complete = len(within) == FUTURE_STEPS and limit < COUNTY_END
            status = "complete_window" if complete else "incomplete_or_year_boundary"
            first_below = np.flatnonzero(within < THRESHOLD)
            reentry = bool((within[first_below[0] + 1:] >= THRESHOLD).any()) if len(first_below) else False
            entries.append({"fips_code": fips, "onset_utc": dates[i], "onset_customers": counts[i],
                            "status": status, "observed_quarters_after_entry": len(within),
                            "observed_below_within_24h": below, "below_by_24h_complete_window": below if complete else None,
                            "below_at_24h_complete_window": bool(within[-1] < THRESHOLD) if complete else None,
                            "reentered_after_observed_drop": reentry})
    return pd.DataFrame(entries)


def county_window_summary(windows):
    summary = []
    for name, sample in [("all_counties", windows), *list(windows.groupby("fips_code", sort=True))]:
        entries = sample[sample.status.ne("unknown_predecessor")]
        complete = entries[entries.status.eq("complete_window")]
        elapsed = (entries.onset_utc.dt.normalize() - COUNTY_START).dt.days
        known_below = int(entries.observed_below_within_24h.eq(True).sum())
        known_still_high = int(complete.below_by_24h_complete_window.eq(False).sum())
        row = {"geography": name, "verified_threshold_entries": len(entries),
               "high_snapshots_with_unknown_predecessor": int(sample.status.eq("unknown_predecessor").sum()),
               "complete_24h_windows": len(complete), "confirmed_below_by_24h_in_complete_windows": int(
                   complete.below_by_24h_complete_window.eq(True).sum()),
               "confirmed_below_at_24h_in_complete_windows": int(complete.below_at_24h_complete_window.eq(True).sum()),
               "complete_windows_with_reentry_after_drop": int(complete.reentered_after_observed_drop.eq(True).sum()),
               "affirmative_below_with_partial_or_complete_history": known_below,
               "unknown_below_status": len(entries) - known_below - known_still_high,
               "all_entry_14_day_blocks": (elapsed // 14).nunique(),
               "complete_window_14_day_blocks": (elapsed.loc[complete.index] // 14).nunique(),
               "complete_window_28_day_blocks": (elapsed.loc[complete.index] // 28).nunique(),
               "below_rate_lower_bound": known_below / len(entries) if len(entries) else np.nan,
               "below_rate_upper_bound": 1 - known_still_high / len(entries) if len(entries) else np.nan}
        summary.append(row)
    return pd.DataFrame(summary)


def mesa_summary(path):
    data = pd.read_csv(path, dtype={"row_id": "string", "event_id": "string"})
    data["start"] = pd.to_datetime(data.start_date_time, format="mixed", errors="coerce")
    data["end"] = pd.to_datetime(data.end_date_time, format="mixed", errors="coerce")
    assert data.start.between(MESA_START, MESA_END, inclusive="left").all()
    assert len(data) < 50000
    data["event_hours"] = (data.end - data.start).dt.total_seconds() / 3600
    data["duration"] = pd.to_numeric(data.duration, errors="coerce")
    data["weather_cause_explicit"] = data.cause_name.str.contains(
        r"\b(?:weather|lightning|wind|rain|storm|flood|ice|snow)\b", case=False, regex=True, na=False)
    data["end_endpoint_status"] = np.select(
        [data.end.isna(), data.end.ge(MESA_END), data.event_hours.le(0)],
        ["missing_end", "end_after_cutoff", "nonpositive_elapsed"],
        default="valid",
    )
    data["duration_matches_minutes"] = (data.duration - data.event_hours * 60).abs().le(1)
    metrics = []
    for year, group in data.groupby(data.start.dt.year):
        for name, chosen in [("all_causes", group), ("explicit_weather_cause", group[group.weather_cause_explicit])]:
            valid = chosen[chosen.end_endpoint_status.eq("valid")]
            metrics.append({"calendar_year": year, "population": name, "source_rows": len(chosen),
                            "distinct_event_ids": chosen.event_id.nunique(), "rows_without_event_id": chosen.event_id.isna().sum(),
                            "rows_with_valid_end": len(valid), "rows_missing_end": chosen.end_endpoint_status.eq("missing_end").sum(),
                            "valid_elapsed_over_24h": valid.event_hours.gt(24).sum(),
                            "valid_gte_1000_customers_reported": valid.total_customers_impacted.ge(THRESHOLD).sum(),
                            "valid_with_vendor_duration": valid.duration.notna().sum(),
                            "vendor_duration_matches_clock_minutes_within_one": valid.duration_matches_minutes.sum(),
                            "partial_restoration_report_rows": chosen.partial_restoration_flag.eq(True).sum(),
                            "valid_14_day_blocks": ((valid.start.dt.normalize() - MESA_START).dt.days // 14).nunique()})
    causes = data.cause_name.fillna("not_reported").value_counts().rename_axis("cause").rename("rows").reset_index()
    return data, pd.DataFrame(metrics), causes


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("raw", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--national", type=Path, help="Verify and re-extract the fixed county slice from the national file")
    args = cli.parse_args()
    args.raw.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.national:
        acquire("author_figshare_eaglei_2018", FIGSHARE, "figshare_release.json", args.raw)
        source = retain_county_rows(args.national, args.raw)
        (args.raw / "national_slice_manifest.json").write_text(json.dumps(source, indent=2) + "\n")
    manifest = json.loads((args.raw / "national_slice_manifest.json").read_text())
    assert sha256((args.raw / COUNTY_SOURCE).read_bytes()).hexdigest() == manifest["slice_sha256"]
    acquire("mesa_utility_e_reliability", mesa_url(), "mesa_historical.csv", args.raw)
    county, snapshots = valid_county_rows(args.raw)
    windows = threshold_windows(snapshots)
    municipal, utility_summary, utility_causes = mesa_summary(args.raw / "mesa_historical.csv")
    tables = {"county_clock_audit": county, "county_threshold_windows": windows,
              "county_window_summary": county_window_summary(windows), "municipal_event_audit": municipal,
              "municipal_summary": utility_summary, "municipal_cause_counts": utility_causes}
    for name, table in tables.items():
        table.to_csv(args.output / (name + ".csv"), index=False)


if __name__ == "__main__":
    main()
