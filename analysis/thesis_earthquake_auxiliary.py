#!/usr/bin/env python3
"""Measure physics-informed synthetic transfer against independent instrumented earthquakes."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np
import pandas as pd
from pyproj import Geod
import requests
from scipy.stats import qmc
from sklearn.linear_model import Ridge

from thesis_retrieve_sources import acquire

API = "https://earthquake.usgs.gov/fdsnws/event/1/query"
QUERY = {"format": "geojson", "starttime": "1998-01-01", "endtime": "2024-10-01",
         "minlatitude": 32, "maxlatitude": 42.1, "minlongitude": -125, "maxlongitude": -114,
         "minmagnitude": 5.5, "maxmagnitude": 7.3, "maxdepth": 20,
         "eventtype": "earthquake", "producttype": "shakemap", "orderby": "time-asc"}
DESIGNS = (100, 500, 1000, 5000)
SEED = 8102026
WGS84 = Geod(ellps="WGS84")
GATE = 5


def url(params):
    return requests.Request("GET", API, params=params).prepare().url


def utc(milliseconds):
    return datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc).isoformat()


def earthquake_catalog(raw):
    details = acquire("comcat_event_search", url(QUERY), "catalog_1998_2024.json", raw)
    events = []
    for item in json.loads((raw / "catalog_1998_2024.json").read_text())["features"]:
        p = item["properties"]
        lon, lat, depth = item["geometry"]["coordinates"]
        events.append({"event_id": item["id"], "origin_utc": utc(p["time"]), "epoch_ms": p["time"],
                       "mag": p["mag"], "mag_type": p["magType"].lower(), "place": p["place"],
                       "lon": lon, "lat": lat, "depth_km": depth})
    events = pd.DataFrame(events)
    assert events.event_id.is_unique and events.origin_utc.max() < "2024-10-01"
    assert events.depth_km.le(20).all()
    # Connected components over a target-free temporal and geographic screen.
    edges = {row.event_id: {row.event_id} for row in events.itertuples()}
    for i, earlier in enumerate(events.itertuples()):
        for later in events.iloc[i+1:].itertuples():
            if later.epoch_ms - earlier.epoch_ms > 14 * 86_400_000:
                break
            distance = WGS84.inv(earlier.lon, earlier.lat, later.lon, later.lat)[2]
            if distance <= 100_000:
                edges[earlier.event_id].add(later.event_id)
                edges[later.event_id].add(earlier.event_id)
    remaining = set(edges)
    representatives = {}
    while remaining:
        component = set()
        queue = [min(remaining)]
        while queue:
            key = queue.pop()
            if key not in component:
                component.add(key)
                queue.extend(edges[key] - component)
        remaining -= component
        pooled = events[events.event_id.isin(component)]
        picked = pooled.sort_values(["mag", "epoch_ms", "event_id"], ascending=[False, True, True]).iloc[0].event_id
        representatives.update({key: picked for key in component})
    events["episode"] = events.event_id.map(representatives)
    return events, details


def station_pga(properties):
    pairs = defaultdict(dict)
    supported = ("EN", "HN", "HH", "BH")
    for channel in properties.get("channels", []):
        location, _, key = channel["name"].upper().rpartition(".")
        if len(key) != 3 or not key[:2].isalpha() or key[-1] not in "EN12":
            continue
        for a in channel.get("amplitudes", []):
            flag = str(a.get("flag", "missing"))
            if a["name"] == "pga" and a["units"] == "%g" and all(bit in ("", "0") for bit in flag.split(",")):
                val = a.get("value")
                if isinstance(val, (int, float)) and 0 < val < 300:
                    pairs[(location, key[:2])][key[-1]] = val / 100
    order = sorted(pairs, key=lambda k: (supported.index(k[1]) if k[1] in supported else len(supported), k))
    for location, family in order:
        values = pairs[(location, family)]
        for axes in ("EN", "12"):
            if set(axes) <= values.keys():
                label = f"{location}.{family}{axes}" if location else family+axes
                return float(np.sqrt(np.prod([values[axis] for axis in axes]))), label
    return None


def read_event(row, source):
    key = row.event_id
    try:
        d = acquire("comcat_event_detail", url({"eventid": key, "format": "geojson"}), key+".json", source)
        item = json.loads((source / d["filename"]).read_text())
        maps = item["properties"]["products"].get("shakemap", [])
        m = next((map_ for map_ in maps if "download/stationlist.json" in map_["contents"]), None)
        if m is None:
            return [], [d], "no_json_station_list", 0
        product = m["contents"]["download/stationlist.json"]
        s = acquire("shakemap_instrument_stations", product["url"], key+"_stations.json", source)
        payload = json.loads((source / s["filename"]).read_text())
    except requests.RequestException as error:
        return [], [], f"retrieval_{getattr(error.response, 'status_code', None)}", 0
    read = []
    seen = set()
    for station in payload["features"]:
        p = station["properties"]
        if p.get("station_type") != "seismic":
            continue
        distance = p.get("distances", {}).get("rjb")
        site = p.get("vs30")
        if not isinstance(distance, (int, float)) or not 1 <= distance <= 80:
            continue
        if not isinstance(site, (int, float)) or not 180 <= site <= 1300:
            continue
        pga = station_pga(p)
        if pga is None:
            continue
        station_key = station.get("id")
        if not station_key or station_key in seen:
            continue
        seen.add(station_key)
        lon, lat = station["geometry"]["coordinates"][:2]
        read.append({"event_id": key, "episode": row.episode, "origin_utc": row.origin_utc,
                     "magnitude": row.mag, "mag_type": row.mag_type, "station": station_key,
                     "station_lon": lon, "station_lat": lat, "rjb_km": distance, "vs30_mps": site,
                     "measured_pga_g": pga[0], "horizontal_pair": pga[1], "station_network": p.get("network"),
                     "product_source": m["source"], "product_update_utc": utc(m["updateTime"])})
    return read, [d, s], "read", len(payload["features"])


def fetch(raw, output):
    raw.mkdir(parents=True, exist_ok=True)
    events, page = earthquake_catalog(raw)
    stations, sources, status = [], [page], []
    selected = events[events.event_id.eq(events.episode) & events.mag_type.str.startswith("mw")]
    with ThreadPoolExecutor(max_workers=5) as pool:
        for row, result in zip(selected.itertuples(), pool.map(lambda row: read_event(row, raw),
                               selected.itertuples()), strict=True):
            data, details, disposition, product_n = result
            stations.extend(data)
            sources.extend(details)
            status.append({"event_id": row.event_id, "product_rows": product_n, "qualified_stations": len(data),
                           "retrieval": disposition})
    events = events.merge(pd.DataFrame(status), on="event_id", how="left", validate="one_to_one")
    events.loc[events.retrieval.isna() & events.event_id.ne(events.episode), "retrieval"] = "linked_episode"
    events.loc[events.retrieval.isna(), "retrieval"] = "non_moment_magnitude"
    events.to_csv(output / "earthquake_sources.csv", index=False)
    pd.DataFrame(sources).to_csv(output / "earthquake_source_manifest.csv", index=False)
    pd.DataFrame(stations).to_csv(output / "instrumental_pga.csv", index=False)
    print(f"Retrieved {len(stations)} eligible station measurements across {len(selected)} screened earthquake episodes.")


def basis(d):
    return np.column_stack([d.magnitude - 6, np.log(np.hypot(d.rjb_km, 5.57)), np.log(d.vs30_mps / 1396)])


def prediction(d):
    return -.242 + basis(d) @ np.array([.527, -.778, -.371])


def simulate():
    spaced = qmc.LatinHypercube(3, seed=np.random.default_rng(SEED)).random(max(DESIGNS))
    lower = [5.5, 0, np.log(180)]
    upper = [7.3, np.log(80), np.log(1300)]
    draws = qmc.scale(spaced, lower, upper)
    d = pd.DataFrame({"scenario": np.arange(1, max(DESIGNS)+1), "magnitude": draws[:, 0],
                      "rjb_km": np.exp(draws[:, 1]), "vs30_mps": np.exp(draws[:, 2])})
    d["simulated_median_pga_g"] = np.exp(prediction(d))
    assert d.rjb_km.between(1, 80).all() and d.vs30_mps.between(180, 1300).all()
    assert d.simulated_median_pga_g.gt(0).all()
    return d


def fit(events, simulated):
    records, membership = [], []
    counts = events.groupby("episode").station.transform("size")
    eligible = events[counts >= GATE].copy()
    eligible["within_episode_weight"] = 1 / counts[counts >= GATE]
    check = eligible.groupby("episode").origin_utc.first().sort_values()
    support = []
    for i, (episode, origin) in enumerate(check.items()):
        held = eligible[eligible.episode.eq(episode)]
        prior = eligible[eligible.episode.isin(check.index[:i])]
        if i < GATE:
            continue
        assert prior.origin_utc.max() < held.origin_utc.min()
        assert not set(prior.episode) & set(held.episode)
        hold = basis(held)
        predictor = basis(prior)
        observed = np.log(prior.measured_pga_g.to_numpy())
        weights = prior.within_episode_weight.to_numpy()
        residual = np.average(observed - prediction(prior), weights=weights)
        preds = {"physics_only": prediction(held), "physics_intercept_calibrated": prediction(held)+residual,
                 "real_only": Ridge(alpha=.03).fit(predictor, observed, sample_weight=weights).predict(hold)}
        for n in DESIGNS:
            synthetic = simulated.iloc[:n]
            x = np.vstack([predictor, basis(synthetic)])
            y = np.r_[observed, np.log(synthetic.simulated_median_pga_g.to_numpy())]
            w = np.r_[weights, np.full(n, 1/n)]
            preds[f"hybrid_{n}"] = Ridge(alpha=.03).fit(x, y, sample_weight=w).predict(hold)
        for name, pred in preds.items():
            records.append(held.assign(model=name, estimate_pga_g=np.exp(pred)))
        for item in prior.episode.unique():
            membership.append({"held_episode": episode, "held_origin_utc": origin, "prior_episode": item})
        support.append({"episode": episode, "origin_utc": origin, "training_episodes": i,
                        "training_stations": len(prior), "held_stations": len(held)})
    if not records:
        raise ValueError("Fewer than six eligible episodes had qualified instruments")
    return pd.concat(records, ignore_index=True), pd.DataFrame(membership), pd.DataFrame(support)


def evaluate(cases):
    d = cases.copy()
    d["log_error"] = np.log(d.estimate_pga_g) - np.log(d.measured_pga_g)
    d["abs_log_error"] = d.log_error.abs()
    d["reference_band_hit90"] = d.abs_log_error.le(1.6448536269514722*.520)
    d["recorded_ge_point10g"] = d.measured_pga_g.ge(.10)
    per = d.groupby(["model", "episode"]).agg(stations=("station", "size"), mae=("abs_log_error", "mean"),
          observed_strong_motion=("recorded_ge_point10g", "sum"),
          mean_log_error=("log_error", "mean"), reference_band_hit90=("reference_band_hit90", "mean")).reset_index()
    summary = []
    for name, group in d.groupby("model"):
        episode = per[per.model.eq(name)]
        strong = group[group.recorded_ge_point10g]
        summary.append({"model": name, "held_episodes": len(episode), "held_stations": len(group),
             "equal_episode_log_mae": episode.mae.mean(), "pooled_log_mae": group.abs_log_error.mean(),
             "pooled_log_rmse": np.sqrt((group.log_error**2).mean()), "strong_station_n": len(strong),
             "strong_station_log_mae": strong.abs_log_error.mean(),
             "reference_stress_band_coverage": group.reference_band_hit90.mean()})
    matrix = per.pivot(index="episode", columns="model", values="mae")
    old = matrix.real_only
    w = np.random.default_rng(SEED + 1).integers(0, len(old), size=(2000, len(old)))
    paired = []
    for name in matrix:
        delta = (matrix[name] - old).to_numpy()
        lo, hi = np.quantile(delta[w].mean(axis=1), [.025, .975])
        paired.append({"model": name, "difference_vs_real_only": delta.mean(), "p025": lo, "p975": hi,
                       "episodes": len(old)})
    return d, per, pd.DataFrame(summary), pd.DataFrame(paired)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--fetch", type=Path, help="Retrieve a new version and record the source SHA-256")
    args = p.parse_args()
    args.data.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.fetch:
        fetch(args.fetch, args.data)
    measurements = pd.read_csv(args.data / "instrumental_pga.csv", float_precision="round_trip")
    assert not measurements.duplicated(["event_id", "station"]).any()
    assert measurements.origin_utc.max() < "2024-10-01"
    synthetic = simulate()
    cases, membership, support = fit(measurements, synthetic)
    cases, episodes, metrics, paired = evaluate(cases)
    tables = {"pga_scenarios.csv.gz": synthetic, "pga_predictions.csv.gz": cases,
              "pga_training_origins.csv": membership, "pga_fold_support.csv": support,
              "pga_per_episode.csv": episodes, "pga_metrics.csv": metrics, "pga_paired_bootstrap.csv": paired}
    for name, frame in tables.items():
        compression = {"method": "gzip", "mtime": 0} if name.endswith(".gz") else None
        frame.to_csv(args.output / name, index=False, compression=compression)
    print(f"Evaluated {len(metrics)} fixed candidate policies across {len(support)} independent held earthquake episodes.")


if __name__ == "__main__":
    main()
