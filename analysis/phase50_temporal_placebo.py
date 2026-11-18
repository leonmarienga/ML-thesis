#!/usr/bin/env python3
"""Re-read fixed Phase 49 windows for same-edition seasonal water comparisons."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path

import aiohttp
import fsspec
import numpy as np
import pandas as pd
from pyproj import Geod
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from shapely.geometry import box, mapping, shape

BASE = "https://data.source.coop/epoch/jrc-global-surface-water-monthly/v1_5"
KEY = ["disasterNumber", "county_fips", "site_index"]
GEOD = Geod(ellps="WGS84")


def load_roster(root):
    a48, a49, a49b = [root / f"phase{x}" for x in ("48", "49", "49b")]
    roster = pd.read_csv(a49 / "phase49_predeclared_36_disaster_county_sampling_roster.csv")
    assert len(roster) == 187 and not roster.duplicated(KEY).any()
    assert roster.disasterNumber.nunique() == 36
    roster["sample_id"] = roster[KEY].astype(str).agg("_".join, axis=1)
    features = json.loads((a49 / "phase49_county_windows_2023_2024.geojson").read_text())["features"]
    polygons = {tuple(f["properties"][k] for k in KEY): shape(f["geometry"]) for f in features}
    assert len(polygons) == 187 and set(polygons) == set(roster[KEY].itertuples(index=False, name=None))
    anchors = pd.read_csv(a48 / "phase48_predeclared_36_disaster_county_sampling_roster.csv")
    joined = anchors.merge(roster.query("site_index == 1"), on=KEY[:2], validate="one_to_one")
    assert len(joined) == len(anchors) == 63
    for coord in ("anchor_lon", "anchor_lat"):
        assert np.allclose(joined[f"{coord}_x"], joined[f"{coord}_y"], rtol=0, atol=1e-9)
    assert joined.event_month_x.equals(joined.event_month_y)

    archives = [
        ("phase49_paired_prior_year_surface_water_transitions.csv",
         "phase49b_Alaska_paired_source_window_correction.csv", "comparison_phase", 374, "paired_read_ok"),
        ("phase49_real_JRC_monthly_2022_24_county_samples.csv",
         "phase49b_Alaska_monthly_source_window_correction.csv", "phase", 561, "read_ok"),
    ]
    corrected = {}
    for original, correction, phase, n, ok in archives:
        past, fix = pd.read_csv(a49 / original), pd.read_csv(a49b / correction)
        alaska = past.query("disasterNumber == 4730")
        assert fix.disasterNumber.eq(4730).all() and len(alaska) == len(fix)
        assert set(map(tuple, alaska[KEY + [phase]].to_numpy())) == set(map(tuple, fix[KEY + [phase]].to_numpy()))
        data = pd.concat([past.query("disasterNumber != 4730"), fix], ignore_index=True)
        assert len(data) == n and not data.duplicated(KEY + [phase]).any()
        assert set(data.sample_id) == set(roster.sample_id)
        assert data.status.eq(ok).all()
        corrected[phase] = data
    sampled = roster.query("site_index == 1 or disasterNumber == 4722")
    assert len(sampled) == 67 and sampled.disasterNumber.eq(4722).sum() == 6
    return sampled, polygons, corrected


def months(z):
    event = pd.Period(z.event_month, freq="M")
    result = {"event": event, "prior": event - 12, "preceding": event - 1}
    if (event - 13).year >= 2022:
        result["prior_preceding"] = event - 13
    if event.year == 2024:
        result["two_years_prior"] = event - 24
    assert all(2022 <= month.year <= 2024 for month in result.values())
    return {k: str(v) for k, v in result.items()}


def url_for(z, month):
    lon = math.floor(z.anchor_lon / 10) * 10
    lat = math.ceil(z.anchor_lat / 10) * 10
    tile = f"{abs(lon)}{'W' if lon < 0 else 'E'}_{abs(lat)}{'S' if lat < 0 else 'N'}"
    y, m = month.split("-")
    return f"{BASE}/{y}/{y}_{m}/monthlyhistory_{tile}_v1_5_{y}_{m}.tif"


def read_windows(url, entries, cache):
    pending = [(z, month, geom) for z, month, geom in entries if not (cache / f"{z.sample_id}_{month}.npz").exists()]
    if not pending:
        return []
    failed = []
    try:
        fs = fsspec.filesystem("https", client_kwargs={"trust_env": True},
                               timeout=aiohttp.ClientTimeout(total=35),
                               block_size=131072, cache_type="blockcache")
        etag = fs.info(url).get("ETag", "")
        with rasterio.open(url, opener=fs, sharing=False) as dataset:
            if dataset.crs.to_epsg() != 4326 or dataset.count != 1:
                raise ValueError("Unexpected CRS or band count")
            for z, month, geom in pending:
                try:
                    native = from_bounds(*geom.bounds, transform=dataset.transform).round_offsets().round_lengths()
                    window = native.intersection(Window(0, 0, dataset.width, dataset.height))
                    if window.width > 500 or window.height > 500:
                        raise ValueError("Raster window exceeds 500-cell guard")
                    matrix = dataset.read(1, window=window)
                    transform = dataset.window_transform(window)
                    mask = geometry_mask([mapping(geom)], matrix.shape, transform, invert=True)
                    values = matrix[mask]
                    if not len(values) or not np.isin(values, [0, 1, 2]).all():
                        raise ValueError("Empty mask or unexpected water codes")
                    dx, dy = [abs(i) / 2 for i in dataset.res]
                    x, y = z.anchor_lon, z.anchor_lat
                    area = abs(GEOD.geometry_area_perimeter(box(x-dx, y-dy, x+dx, y+dy))[0]) / 1e6
                    path = cache / f"{z.sample_id}_{month}.npz"
                    temporary = path.with_suffix(".tmp.npz")
                    np.savez_compressed(
                        temporary, values=values, transform=np.array(transform), mask=mask,
                        area=area, clipped=native != window, source_url=url,
                        etag=etag, retrieved=datetime.now(timezone.utc).isoformat(),
                    )
                    temporary.replace(path)
                except Exception as exc:
                    failed.append((z.sample_id, month, type(exc).__name__, str(exc)[:300]))
    except Exception as exc:
        failed.extend((z.sample_id, month, type(exc).__name__, str(exc)[:300])
                      for z, month, _ in pending if not (cache / f"{z.sample_id}_{month}.npz").exists())
    return failed


def transition(a, b, eligible, area):
    return {
        "observed_n": int(eligible.sum()),
        "observed_fraction": float(eligible.mean()),
        "observed_km2": float(eligible.sum() * area),
        "gain_n": int(((a == 1) & (b == 2) & eligible).sum()),
        "loss_n": int(((a == 2) & (b == 1) & eligible).sum()),
    }


def compare(site, arrays, design, arms, aligned, area):
    if not all(k in arrays for arm in arms for k in arm):
        return []
    valid = [(arrays[a] > 0) & (arrays[b] > 0) for a, b in arms]
    common = valid[0] & valid[1]
    results = []
    for method, masks in (("common", [common, common]), ("pair_specific", valid)):
        result = {**site, "design": design, "denominator": method, "aligned_grid": aligned}
        for label, (a, b), mask in zip(("event", "placebo"), arms, masks, strict=True):
            result.update({f"{label}_{k}": v for k, v in transition(arrays[a], arrays[b], mask, area).items()})
        result["pixel_area_km2"] = area
        for threshold in (30, 60, 80):
            result[f"passes_{threshold}"] = min(result["event_observed_fraction"], result["placebo_observed_fraction"]) >= threshold / 100
        results.append(result)
    return results


def summarize(data, output):
    def rates(rows):
        result = {}
        for arm in ("event", "placebo"):
            denominator = rows[f"{arm}_observed_km2"].sum()
            result[f"{arm}_observed_km2"] = denominator
            for direction in ("gain", "loss"):
                amount = (rows[f"{arm}_{direction}_n"] * rows.pixel_area_km2).sum()
                result[f"{arm}_{direction}_km2"] = amount
                result[f"{arm}_{direction}_rate"] = 100 * amount / denominator
            result[f"{arm}_net_rate"] = result[f"{arm}_gain_rate"] - result[f"{arm}_loss_rate"]
        result["delta_net_rate"] = result["event_net_rate"] - result["placebo_net_rate"]
        return result

    overall, cases = [], []
    for (design, denominator), rows in data.groupby(["design", "denominator"]):
        for threshold in (30, 60, 80):
            available = rows[rows[f"passes_{threshold}"]]
            for population, selected in (
                ("anchors", available.query("site_index == 1")),
                ("anchors_excluding_4722", available.query("site_index == 1 and disasterNumber != 4722")),
                ("six_4722_sites", available.query("disasterNumber == 4722")),
            ):
                metadata = {"design": design, "denominator": denominator, "threshold_percent": threshold,
                            "population": population}
                group_rates = []
                for number, group in selected.groupby("disasterNumber"):
                    group_rates.append(rates(group))
                    cases.append({**metadata, "disasterNumber": number, "sites": len(group), **group_rates[-1]})
                deltas = np.array([g["delta_net_rate"] for g in group_rates])
                overall.append({**metadata, "sites": len(selected), "disasters": len(deltas),
                                "positive_case_deltas": int((deltas > 0).sum()),
                                "median_case_delta": np.median(deltas) if len(deltas) else None,
                                **(rates(selected) if len(deltas) else {})})
    pd.DataFrame(overall).to_csv(output / "phase50_summary.csv", index=False)
    pd.DataFrame(cases).to_csv(output / "phase50_case_differences.csv", index=False)


def run(root, output, workers):
    roster, polygons, history = load_roster(root)
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "pixels"
    cache.mkdir(exist_ok=True)
    tasks = defaultdict(list)
    for z in roster.itertuples(index=False):
        geom = polygons[tuple(getattr(z, k) for k in KEY)]
        for month in months(z).values():
            tasks[url_for(z, month)].append((z, month, geom))
    failures = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        completed = pool.map(lambda task: read_windows(*task, cache), tasks.items())
        for index, errors in enumerate(completed, 1):
            failures.update({(site, month): f"{kind}: {detail}" for site, month, kind, detail in errors})
            if index % 10 == 0 or index == len(tasks):
                print(f"Tiles processed {index}/{len(tasks)}; failed reads {len(failures)}", flush=True)

    past = history["comparison_phase"].query("comparison_phase == 'event'").set_index("sample_id")
    monthly, paired, replay = [], [], []
    for z in roster.itertuples(index=False):
        fields = {key: getattr(z, key) for key in [*KEY, "sample_id", "state", "county_label", "anchor_lon", "anchor_lat", "incident_start"]}
        arrays, grids, areas = {}, [], []
        for role, month in months(z).items():
            path = cache / f"{z.sample_id}_{month}.npz"
            row = {**fields, "month": month, "role": role, "source_url": url_for(z, month),
                   "status": "read_failed", "error": failures.get((z.sample_id, month))}
            if path.exists():
                with np.load(path) as raw:
                    assert str(raw["source_url"]) == row["source_url"]
                    values = raw["values"]
                    arrays[role] = values
                    grids.append((raw["transform"], raw["mask"]))
                    areas.append(float(raw["area"]))
                    row.update(status="read_ok", error=None, pixel_count=len(values),
                               observed_n=int((values > 0).sum()), water_n=int((values == 2).sum()),
                               pixel_classes_sha256=sha256(values.tobytes()).hexdigest(),
                               etag=str(raw["etag"]), retrieved_utc=str(raw["retrieved"]),
                               source_tile_clipped=bool(raw["clipped"]))
            monthly.append(row)
        if "prior_preceding" not in months(z):
            monthly.append({**fields, "month": "2021-12", "role": "prior_preceding",
                            "status": "excluded_mixed_edition"})
        aligned = bool(grids) and all(np.array_equal(t, grids[0][0]) and np.array_equal(m, grids[0][1])
                                      for t, m in grids)
        audited = {**fields, "status": "unaligned_grid" if grids and not aligned else "missing_read"}
        if aligned and {"event", "prior"} <= arrays.keys():
            p, e = arrays["prior"], arrays["event"]
            result = transition(p, e, (p > 0) & (e > 0), areas[0])
            archived = past.loc[z.sample_id]
            expected = {"observed_n": int(archived.both_observed_n),
                        "gain_n": int(archived.new_water_cells_prior_dry_event_water),
                        "loss_n": int(archived.disappeared_water_cells_prior_water_event_dry)}
            matches = all(result[k] == v for k, v in expected.items())
            audited.update(result, **{f"archived_{k}": v for k, v in expected.items()},
                           status="exact_match" if matches else "mismatch")
        replay.append(audited)
        if aligned:
            paired.extend(compare(fields, arrays, "seasonal", (("preceding", "event"), ("prior_preceding", "prior")), aligned, areas[0]))
            paired.extend(compare(fields, arrays, "interannual_2024", (("prior", "event"), ("two_years_prior", "prior")), aligned, areas[0]))
    pd.DataFrame(monthly).to_csv(output / "phase50_monthly_source_log.csv", index=False)
    pd.DataFrame(replay).to_csv(output / "phase50_phase49_replay.csv", index=False)
    comparisons = pd.DataFrame(paired)
    comparisons.to_csv(output / "phase50_site_comparisons.csv", index=False)
    if len(comparisons):
        summarize(comparisons, output)
    print(pd.Series([r["status"] for r in monthly]).value_counts().to_string(), flush=True)
    print(pd.Series([r["status"] for r in replay]).value_counts().to_string(), flush=True)
    if failures or not len(comparisons):
        raise SystemExit("Coverage incomplete. Inspect the source log and rerun to retry failed reads.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run(args.artifacts, args.output, args.workers)
