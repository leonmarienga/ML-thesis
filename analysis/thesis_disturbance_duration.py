#!/usr/bin/env python3
"""Replay a historical-origin synthetic prior comparison for DOE weather reports."""
import argparse
from datetime import datetime, time
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

from thesis_retrieve_sources import acquire

YEARS = range(2018, 2025)
ORIGINS = range(2021, 2025)
END = pd.Timestamp("2024-10-01")
SAFETY_GAP = pd.Timedelta(14, unit="D")
PARAMETRIC_WEIGHT = 20
BASE = "empirical_seasonal"
HYBRID = "hybrid_half_synthetic"
SEASONS = ("DJF", "MAM", "JJA", "SON")
DATE_COLUMNS = {
    "start_date": ("Date Event Began", "Date Began"),
    "start_time": ("Time Event Began", "Time Began"),
    "end_date": ("Date of Restoration", "Date Restored"),
    "end_time": ("Time of Restoration", "Time Restored"),
}


def parse_clock(date_value, time_value):
    if pd.isna(date_value) or pd.isna(time_value):
        return pd.NaT
    day = pd.to_datetime(date_value, format="mixed", errors="coerce")
    if pd.isna(day):
        return pd.NaT
    if isinstance(time_value, datetime):
        hour = time_value.time()
    elif isinstance(time_value, time):
        hour = time_value
    else:
        match = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(AM|PM)?",
                             str(time_value).strip(), re.I)
        if not match:
            return pd.NaT
        h, m, s, ampm = match.groups()
        h = int(h)
        if ampm:
            if not 1 <= h <= 12:
                return pd.NaT
            h = h % 12 + (12 if ampm.upper() == "PM" else 0)
        try:
            hour = time(h, int(m), int(s or 0))
        except ValueError:
            return pd.NaT
    return pd.Timestamp.combine(day.date(), hour)


def load(path, year):
    workbook = pd.read_excel(path, header=None)
    header = next(i for i, row in workbook.iterrows() if "Alert Criteria" in row.values)
    columns = {str(value).strip(): i for i, value in enumerate(workbook.iloc[header])}
    selectors = {key: next(columns[name] for name in names if name in columns)
                 for key, names in DATE_COLUMNS.items()}
    data = workbook.iloc[header + 1:]
    annotated = data[columns["Event Type"]].notna() | data[columns["Alert Criteria"]].notna()
    weather = data[columns["Event Type"]].astype(str).str.contains(
        r"severe weather|weather or natural disaster", case=False
    )
    criterion = data[columns["Alert Criteria"]].astype(str).str.replace(
        r"\s+", " ", regex=True
    ).str.contains(r"loss of electric service.{0,30}50[, ]?000 customers", case=False)
    selected = data.loc[weather & criterion]
    rows = []
    for i, row in selected.iterrows():
        clock_data = {key: row.iloc[j] for key, j in selectors.items()}
        start = parse_clock(clock_data["start_date"], clock_data["start_time"])
        end = parse_clock(clock_data["end_date"], clock_data["end_time"])
        if pd.isna(start):
            status = "missing_onset"
        elif start.year != year or start >= END:
            status = "outside_calendar_scope"
        elif pd.isna(end):
            status = "missing_endpoint"
        elif end <= start:
            status = "nonpositive_clock_duration"
        elif end >= END:
            status = "endpoint_after_scope_cutoff"
        else:
            status = "scored"
        rows.append({
            "archive_year": year, "spreadsheet_row": i + 1, "onset": start, "reported_end": end,
            "duration_hours": (end - start).total_seconds() / 3600 if status == "scored" else np.nan,
            "status": status,
        })
    selection = {"archive_year": year, "annotated_rows": int(annotated.sum()),
                 "weather_rows": int(weather.sum()), "criterion_selected": len(rows)}
    return rows, selection


def duration_summary(evidence):
    records = []
    for year, rows in evidence.groupby("archive_year"):
        in_scope = rows[rows.status != "outside_calendar_scope"]
        valid = in_scope[in_scope.status == "scored"]
        records.append({"archive_year": year, "criteria_selected_in_scope": len(in_scope),
                        "usable_duration": len(valid), "missing_endpoint": int((in_scope.status == "missing_endpoint").sum()),
                        "invalid_or_unavailable_clock": len(in_scope) - len(valid) - int((in_scope.status == "missing_endpoint").sum()),
                        "under_one_hour": int((valid.duration_hours < 1).sum()),
                        "rows_sharing_both_clocks": int(valid.duplicated(["onset", "reported_end"], keep=False).sum())})
    return pd.DataFrame(records)


def describe(values, weights, observed):
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cdf = np.cumsum(weights[order])
    q50, q90 = sorted_values[np.searchsorted(cdf, [.5, .9])]
    risk = float(weights[values > np.log1p(24)].sum())
    log_y = np.log1p(observed)
    return {
        "p_over_24h": risk,
        "median_hours": np.expm1(q50),
        "upper_90_hours": np.expm1(q90),
        "brier_24h": (risk - (observed > 24)) ** 2,
        "log_pinball_50": .5 * abs(log_y - q50),
        "log_pinball_90": max(.9 * (log_y - q90), -.1 * (log_y - q90)),
        "covered_90": int(log_y <= q90),
    }


def distributions(pool, local):
    n = len(local)
    mean = (local.sum() + PARAMETRIC_WEIGHT * pool.mean()) / (n + PARAMETRIC_WEIGHT)
    second = ((local ** 2).sum() + PARAMETRIC_WEIGHT * (pool ** 2).mean()) / (n + PARAMETRIC_WEIGHT)
    scale = np.sqrt(max(second - mean ** 2, 0))
    # Truncation at zero prevents negative log1p durations.
    lower = ndtr(-mean / scale) if scale else 0
    grid = (np.arange(1024) + .5) / 1024
    sim = mean + scale * ndtri(lower + (1 - lower) * grid)
    assert np.isfinite(sim).all() and np.all(sim >= -1e-12)
    return {
        "empirical_pooled": (pool, np.ones(len(pool)) / len(pool)),
        BASE: (local, np.ones(n) / n),
        "seasonal_synthetic_prior": (sim, np.ones(len(sim)) / len(sim)),
        HYBRID: (np.r_[local, sim], np.r_[np.ones(n) * .5 / n, np.ones(len(sim)) * .5 / len(sim)]),
    }


def compare(data):
    predictions, support, membership = [], [], []
    valid = data[data.status == "scored"].copy()
    valid["season"] = valid.onset.dt.month.map(lambda month: SEASONS[(month % 12) // 3])
    difference = (valid.onset.dt.normalize() - pd.Timestamp("2018-01-01")).dt.days
    valid["block14"] = difference // 14
    valid["block28"] = difference // 28
    for year in ORIGINS:
        cutoff = pd.Timestamp(year - 1, 1, 1) - SAFETY_GAP
        train = valid[(valid.onset < cutoff) & (valid.reported_end < cutoff)]
        test = valid[valid.archive_year == year]
        assert len(train) >= 30 and train.block14.nunique() >= 12
        pool = np.log1p(train.duration_hours.to_numpy())
        variant = {}
        for season in SEASONS:
            selected = train[train.season == season]
            if len(selected) < 8:
                selected = train
            variant[season] = distributions(pool, np.log1p(selected.duration_hours.to_numpy()))
            season_rows = set(zip(selected.archive_year, selected.spreadsheet_row))
            for source in train.itertuples():
                membership.append({"origin": year, "generation_season": season, "source_archive_year": source.archive_year,
                                   "source_spreadsheet_row": source.spreadsheet_row,
                                   "in_local_or_fallback_moments": (source.archive_year, source.spreadsheet_row) in season_rows})
        for row in test.itertuples():
            for model, (values, weights) in variant[row.season].items():
                predictions.append({"origin": year, "spreadsheet_row": row.spreadsheet_row,
                                    "onset": row.onset, "block14": row.block14, "block28": row.block28,
                                    "season": row.season, "duration_hours": row.duration_hours,
                                    "model": model, **describe(values, weights, row.duration_hours)})
        support.append({"origin": year, "training_cutoff": cutoff.date(), "training_reports": len(train),
                        "training_14day_blocks": train.block14.nunique(), "held_out_reports": len(test),
                        "held_out_over_24h": int((test.duration_hours > 24).sum())})
    return pd.DataFrame(predictions), pd.DataFrame(support), pd.DataFrame(membership)


def evaluate(predictions):
    metrics = ["brier_24h", "log_pinball_50", "log_pinball_90", "covered_90"]
    results = []
    populations = [("all", predictions),
                   ("outside_23_to_25h", predictions[~predictions.duration_hours.between(23, 25)])]
    populations.extend((str(year), cases) for year, cases in predictions.groupby("origin"))
    for population, cases in populations:
        for key, frame in cases.groupby("model", sort=True):
            results.append({"population": population, "model": key, "n": len(frame),
                            "over_24h": int((frame.duration_hours > 24).sum()),
                            "median_observed_hours": frame.duration_hours.median(),
                            "median_predicted_hours": frame.median_hours.median(),
                            "median_upper_90_hours": frame.upper_90_hours.median(),
                            **frame[metrics].mean().to_dict()})
    paired = predictions.pivot(index=["origin", "spreadsheet_row", "block14", "block28"],
                               columns="model", values="brier_24h").reset_index()
    paired["difference"] = paired[HYBRID] - paired[BASE]
    intervals = []
    for block in ("block14", "block28"):
        blocks = paired.groupby(block).difference.agg(["sum", "size"]).to_numpy()
        rng = np.random.default_rng(78262)
        samples = blocks[rng.integers(0, len(blocks), size=(3000, len(blocks)))].sum(axis=1)
        lo, hi = np.quantile(samples[:, 0] / samples[:, 1], [.025, .975])
        intervals.append({"block": block, "clusters": len(blocks), "hybrid_minus_empirical_brier":
                          paired.difference.mean(), "lower_95": lo, "upper_95": hi})
    return pd.DataFrame(results), pd.DataFrame(intervals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.raw.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    source, observations, counts = [], [], []
    for year in YEARS:
        filename = f"{year}_Annual_Summary." + ("xlsx" if year == 2024 else "xls")
        source.append(acquire("doe_417_annual_summary", "https://doe417.energy.gov/summaries/" + filename,
                              filename, args.raw))
        rows, selection = load(args.raw / filename, year)
        observations.extend(rows)
        counts.append(selection)
    evidence = pd.DataFrame(observations)
    predictions, support, membership = compare(evidence)
    scores, intervals = evaluate(predictions)
    for name, table in [("source_manifest", pd.DataFrame(source)), ("source_counts", pd.DataFrame(counts)),
                        ("duration_audit", evidence), ("duration_summary", duration_summary(evidence)),
                        ("origin_support", support), ("prior_training_membership", membership), ("predictions", predictions),
                        ("scores", scores), ("paired_brier_intervals", intervals)]:
        table.to_csv(args.output / f"{name}.csv", index=False)
    print(f"Scored {len(predictions) // 4} held-out weather disturbance reports.")


if __name__ == "__main__":
    main()
