#!/usr/bin/env python3
"""Archive fixed NOAA tide requests and screen matched verified station-hours."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
PRODUCTS = {"hourly_height": "data", "predictions": "predictions"}
CAP = 2_000_000


def url_for(row, product):
    center = date.fromisoformat(row.center_date)
    queries = dict(product=product, station=row.station_id, application="ThesisResearch",
                   begin_date=(center - timedelta(days=1)).strftime("%Y%m%d"),
                   end_date=(center + timedelta(days=1)).strftime("%Y%m%d"),
                   datum="MSL", units="metric", time_zone="gmt", format="json")
    if product == "predictions":
        queries["interval"] = "h"
    return API + "?" + urlencode(queries)


def acquire(job, make_url=url_for):
    row, product, root = job
    name = f"{row.storm}_{row.station_id}_{product}"
    url = make_url(row, product)
    raw = b""
    for attempt in (1, 2):
        try:
            with urlopen(url, timeout=12) as reply:
                status = reply.status
                raw = reply.read(CAP + 1)
            assert len(raw) <= CAP
            break
        except HTTPError as exc:
            status = exc.code
            if exc.code < 500:
                raw = exc.read(CAP + 1)
                assert len(raw) <= CAP
                break
        except (URLError, TimeoutError, OSError):
            status = None
    (root / f"{name}.json").write_bytes(raw)
    manifest = {"url": url, "http_status": status, "attempts": attempt,
                "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                "retrieval_utc": datetime.now(timezone.utc).isoformat()}
    (root / f"{name}.manifest").write_text(json.dumps(manifest, indent=2) + "\n")


def verified_response(row, product, raw_root, make_url=url_for):
    name = f"{row.storm}_{row.station_id}_{product}"
    raw = (raw_root / f"{name}.json").read_bytes()
    manifest = json.loads((raw_root / f"{name}.manifest").read_text())
    assert manifest["url"] == make_url(row, product)
    assert manifest["sha256"] == hashlib.sha256(raw).hexdigest()
    if manifest["http_status"] != 200:
        return {}, "transport_unavailable"
    response = json.loads(raw)
    if "error" in response:
        return {}, "source_no_product"
    if product != "predictions":
        assert str(response["metadata"]["id"]) == row.station_id
    return response, "available"


def parse(row, product, raw_root, unflagged_only=False):
    response, status = verified_response(row, product, raw_root)
    if not response:
        return pd.Series(dtype=float), status
    table = pd.DataFrame(response[PRODUCTS[product]])
    if table.empty:
        return pd.Series(dtype=float), "source_no_product"
    if unflagged_only:
        assert product == "hourly_height" and table.f.str.fullmatch("[01],[01]").all()
        table = table[table.f.eq("0,0")]
    stamps = pd.to_datetime(table.t, utc=True)
    assert stamps.is_unique and stamps.dt.minute.eq(0).all()
    return pd.Series(pd.to_numeric(table.v, errors="coerce").to_numpy(), index=stamps), "available"


def replay(panel, raw_root, output, unflagged_only):
    output.mkdir(parents=True, exist_ok=True)
    details, coverage, errors = [], [], []
    for row in panel.itertuples(index=False):
        obs, obs_status = parse(row, "hourly_height", raw_root, unflagged_only)
        tide, tide_status = parse(row, "predictions", raw_root)
        start = pd.Timestamp(date.fromisoformat(row.center_date) - timedelta(days=1), tz="UTC")
        index = pd.date_range(start, freq="h", periods=72)
        series = pd.DataFrame({"observed_m": obs, "astronomical_m": tide}, index=index)
        series["residual_m"] = series.observed_m - series.astronomical_m
        valid = series.residual_m.notna()
        longest = (~valid).groupby(valid.cumsum()).sum().max()
        key = dict(storm=row.storm, station_id=row.station_id)
        coverage.append({**key, "observed_hours": int(series.observed_m.notna().sum()),
                         "tide_hours": int(series.astronomical_m.notna().sum()), "matched_hours": int(valid.sum()),
                         "longest_missing_hours": int(longest), "observation_status": obs_status,
                         "prediction_status": tide_status, "passes": bool(valid.sum() >= 54),
                         "peak_residual_m": series.residual_m.max()})
        details.append(series.assign(**key, timestamp_utc=index).reset_index(drop=True))
        for lead in (1, 6):
            matched = valid & valid.shift(lead, fill_value=False)
            residual = series.residual_m
            pairs = zip(index[matched], residual[matched], residual.shift(lead)[matched], strict=True)
            for at, actual, former in pairs:
                errors.append({**key, "hours": lead, "timestamp_utc": at,
                               "tide_only_abs_error_m": abs(actual), "persistence_abs_error_m": abs(actual - former)})
    sources = pd.DataFrame(coverage)
    scores = pd.DataFrame(errors)
    storm_gate = sources.groupby("storm", as_index=False).agg(
        passing_sites=("passes", "sum"), matched_hours=("matched_hours", "sum"))
    storm_gate["qualifies"] = storm_gate.passing_sites.eq(2)
    accepted = sources[sources.passes][["storm", "station_id"]]
    if not scores.empty:
        eligible = scores.merge(accepted, on=["storm", "station_id"], validate="many_to_one")
        metrics = eligible.groupby(["storm", "hours"], as_index=False).agg(
            timestamp_pairs=("timestamp_utc", "size"), tide_only_mae_m=("tide_only_abs_error_m", "mean"),
            persistence_mae_m=("persistence_abs_error_m", "mean"))
    else:
        metrics = pd.DataFrame(columns=["storm", "hours", "timestamp_pairs", "tide_only_mae_m", "persistence_mae_m"])
    for name, table in [("station_coverage", sources), ("hourly_measurements", pd.concat(details, ignore_index=True)),
                        ("storm_gate", storm_gate), ("storm_diagnostics", metrics)]:
        table.to_csv(output / f"{name}.csv", index=False)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("panel", type=Path)
    cli.add_argument("archive", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--acquire", action="store_true")
    cli.add_argument("--unflagged-only", action="store_true")
    args = cli.parse_args()
    panel = pd.read_csv(args.panel, dtype={"station_id": str})
    assert len(panel) == 14 and panel.storm.value_counts().eq(2).all()
    if args.acquire:
        args.archive.mkdir(parents=True, exist_ok=True)
        jobs = [(row, product, args.archive) for row in panel.itertuples(index=False) for product in PRODUCTS]
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(acquire, jobs))
    replay(panel, args.archive, args.output, args.unflagged_only)
