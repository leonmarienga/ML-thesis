#!/usr/bin/env python3
"""Compare matched real and physics-prior coastal nowcasts on separate storms."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from functools import partial
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from thesis_tidal_residual import acquire, parse, url_for, verified_response

PRODUCTS = ("hourly_height", "predictions", "air_pressure", "wind")
LEADS = (1, 6)
VARIABLES = ("level0", "level_change", "pressure0", "pressure_change", "stress0", "stress_change")
SCALES = np.array([1., .3, 25., 8., 130., 75.])
REFERENCE_PRESSURE = 1013.25
SYNTHETIC_N = 12_000


def request(row, product):
    if product in PRODUCTS[:2]:
        return url_for(row, product)
    tide_request = urlsplit(url_for(row, "predictions"))
    params = dict(parse_qsl(tide_request.query))
    params.pop("datum")
    params["product"] = product
    return tide_request._replace(query=urlencode(params)).geturl()


def meteorology(row, product, raw_root):
    response, status = verified_response(row, product, raw_root, request)
    if not response or not response["data"]:
        return pd.Series(dtype=float), status
    data = pd.DataFrame(response["data"])
    t = pd.to_datetime(data.t, utc=True)
    assert t.is_unique and t.dt.minute.eq(0).all()
    if product == "air_pressure":
        v = pd.to_numeric(data.v, errors="coerce")
        value = v.where(v.between(870, 1060)) - REFERENCE_PRESSURE
    else:
        v = pd.to_numeric(data.s, errors="coerce")
        angle = pd.to_numeric(data.d, errors="coerce")
        radians = np.deg2rad(angle.where(angle.between(0, 360)) + 180 - row.landward_degrees)
        value = v.where(v.between(0, 75))**2 * np.cos(radians)
    return pd.Series(value.to_numpy(), index=t), status


def load(panel, raw, unflagged):
    pairs, statuses, measurements = [], [], []
    for row in panel.itertuples(index=False):
        height, h = parse(row, "hourly_height", raw, unflagged)
        tide, t = parse(row, "predictions", raw)
        pressure, p = meteorology(row, "air_pressure", raw)
        stress, w = meteorology(row, "wind", raw)
        start = pd.Timestamp(date.fromisoformat(row.center_date) - timedelta(days=1), tz="UTC")
        clocks = pd.date_range(start, periods=72, freq="h")
        series = pd.DataFrame({"height": height, "tide": tide, "pressure": pressure, "stress": stress}, index=clocks)
        for source in ("height", "pressure", "stress"):
            assert len(series[source]) == 72
        level = series.height - series.tide
        measurements.append(series.assign(storm=row.storm, station_id=row.station_id,
                                           partition=row.partition, clock=clocks).reset_index(drop=True))
        x = pd.DataFrame(dict(level0=level, level_change=level - level.shift(3),
                              pressure0=series.pressure, pressure_change=series.pressure - series.pressure.shift(3),
                              stress0=series.stress, stress_change=series.stress - series.stress.shift(3)))
        hourly, n = [], {}
        for lead in LEADS:
            eligible = x.assign(future=level.shift(-lead)).dropna()
            n[f"complete_{lead}h"] = len(eligible)
            hourly.append(eligible.assign(storm=row.storm, station_id=row.station_id, partition=row.partition,
                                          lead=lead, issue_utc=eligible.index).reset_index(drop=True))
        statuses.append(dict(storm=row.storm, station_id=row.station_id, partition=row.partition,
                             matched_level=int(level.notna().sum()), wind_hours=int(series.stress.notna().sum()),
                             pressure_hours=int(series.pressure.notna().sum()), height_status=h, tide_status=t,
                             pressure_status=p, wind_status=w, **n))
        pairs.extend(hourly)
    support = pd.DataFrame(statuses)
    level_pass = support.matched_level.ge(54)
    forcing_pass = support[["complete_1h", "complete_6h"]].min(axis=1).ge(24)
    gate = support.assign(level_site=level_pass, forcing_site=forcing_pass).groupby(
        ["partition", "storm"], as_index=False).agg(level_sites=("level_site", "sum"),
                                                    forcing_sites=("forcing_site", "sum"))
    gate["qualifies"] = gate.level_sites.eq(2) & gate.forcing_sites.ge(1)
    matched_sites = support[level_pass & forcing_pass][["storm", "station_id"]]
    examples = pd.concat(pairs, ignore_index=True).merge(matched_sites, on=["storm", "station_id"],
                                                       validate="many_to_one")
    return support, gate, examples, pd.concat(measurements, ignore_index=True)


def design(frame):
    return frame[list(VARIABLES)].to_numpy() / SCALES


def process_prior(lead):
    rng = np.random.default_rng(83711 + lead)
    n = SYNTHETIC_N
    r0 = rng.uniform(-1.4, 3.6, n)
    rate = rng.normal(0, .28, n)
    p = rng.uniform(-65, 15, n)
    p_change = rng.uniform(-14, 14, n)
    stress = rng.uniform(-625, 625, n)
    stress_change = rng.uniform(-230, 230, n)
    # Slab balance: air density, drag, fetch, water density, gravity and depth.
    factor = 1.225 * rng.uniform(.001, .002, n) * rng.uniform(5000, 30000, n)
    factor /= 1025 * 9.81 * rng.uniform(8, 40, n)
    tau = rng.uniform(3, 18, n)
    p_future = (p + p_change * lead / 6).clip(-65, 15)
    stress_future = (stress + stress_change * lead / 6).clip(-625, 625)
    eq = -.01 * p_future + factor * stress_future
    y = (eq - r0) * (1 - np.exp(-lead / tau)) + rng.normal(0, .10, n)
    x = np.column_stack((r0, rate, p, p_change, stress, stress_change)) / SCALES
    assert np.isfinite(x).all() and np.isfinite(y).all()
    return Ridge(alpha=2).fit(x, y)


def fit_and_evaluate(examples, gate):
    earlier = set(gate.loc[gate.partition.eq("earlier") & gate.qualifies, "storm"])
    held = set(gate.loc[gate.partition.eq("held") & gate.qualifies, "storm"])
    assert earlier.isdisjoint(held)
    eligible = len(earlier) >= 4 and len(held) >= 2
    rows = []
    for lead in LEADS:
        past = examples[examples.storm.isin(earlier) & examples.lead.eq(lead)]
        observed = examples[examples.storm.isin(held) & examples.lead.eq(lead)].copy()
        if observed.empty:
            continue
        y0 = observed.level0.to_numpy()
        arms = {"tide_only": np.zeros(len(observed)), "persistence": y0,
                "hydrostatic": y0 + (-.01 * observed.pressure0.to_numpy() - y0) * (1 - np.exp(-lead / 9))}
        if eligible:
            y = (past.future - past.level0).to_numpy()
            x, z = design(past), design(observed)
            weights = 1 / past.storm.map(past.storm.value_counts()).to_numpy()
            weights = weights / weights.mean()
            synthetic = process_prior(lead)
            statistical = Ridge(alpha=8).fit(x, y, sample_weight=weights)
            correction = Ridge(alpha=8).fit(x, y - synthetic.predict(x), sample_weight=weights)
            arms.update(statistical=y0+statistical.predict(z), synthetic_only=y0+synthetic.predict(z),
                        hybrid=y0+synthetic.predict(z)+correction.predict(z))
        for arm, pred in arms.items():
            out = observed[["storm", "station_id", "issue_utc", "lead", "future"]].copy()
            rows.append(out.assign(arm=arm, forecast=pred, abs_error=abs(pred - out.future)))
    columns = ["storm", "station_id", "issue_utc", "lead", "future", "arm", "forecast", "abs_error"]
    predictions = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)
    storm_scores = predictions.groupby(["lead", "arm", "storm"], as_index=False).agg(
        station_origins=("abs_error", "size"), mae_m=("abs_error", "mean"))
    return predictions, storm_scores


def contrasts(storm_scores):
    rows = []
    rng = np.random.default_rng(57418)
    for lead, group in storm_scores.groupby("lead"):
        wide = group.pivot(index="storm", columns="arm", values="mae_m")
        indices = rng.integers(0, len(wide), size=(2000, len(wide)))
        for ref in ("persistence", "statistical"):
            if ref not in wide:
                continue
            for target in wide.drop(columns=ref):
                delta = (wide[target] - wide[ref]).to_numpy()
                low, high = np.quantile(delta[indices].mean(axis=1), [.025, .975])
                rows.append((lead, target, ref, len(delta), wide[target].mean(),
                             delta.mean(), low, high, delta.min(), delta.max()))
    return pd.DataFrame(rows, columns=["lead", "arm", "reference", "held_storms", "storm_mean_mae_m", "delta_m",
                                       "lower_95", "upper_95", "best_storm_change_m", "worst_storm_change_m"])


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("panel", type=Path)
    cli.add_argument("raw", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--acquire", action="store_true")
    args = cli.parse_args()
    panel = pd.read_csv(args.panel, dtype={"station_id": str})
    assert panel.storm.nunique() == 12 and panel.storm.value_counts().eq(2).all()
    assert max(panel[panel.partition.eq("earlier")].center_date) < min(panel[panel.partition.eq("held")].center_date)
    if args.acquire:
        args.raw.mkdir(parents=True, exist_ok=True)
        jobs = [(row, product, args.raw) for row in panel.itertuples(index=False) for product in PRODUCTS]
        with ThreadPoolExecutor(max_workers=6) as executor:
            list(executor.map(partial(acquire, make_url=request), jobs))
    for name in ("verified", "unflagged"):
        sources, gate, origins, measurements = load(panel, args.raw, name == "unflagged")
        pred, scores = fit_and_evaluate(origins, gate)
        dest = args.output / name
        dest.mkdir(parents=True, exist_ok=True)
        tables = [("station_support", sources), ("storm_gate", gate), ("predictions", pred),
                  ("storm_scores", scores), ("contrasts", contrasts(scores))]
        if name == "verified":
            tables.append(("aligned_source_hours", measurements))
        for key, frame in tables:
            frame.to_csv(dest / (key + ".csv"), index=False)


if __name__ == "__main__":
    main()
