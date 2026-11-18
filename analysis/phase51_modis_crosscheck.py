#!/usr/bin/env python3
"""Cross-check six DR4722 windows using NASA's public rendered MODIS flood layer."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
import requests
from shapely.geometry import mapping, shape

from thesis_retrieve_sources import public_get

ROOT = "https://gibs.earthdata.nasa.gov"
COLOR_MAP = ROOT + "/colormaps/v1.3/MODIS_Flood.xml"
LAYER = ROOT + "/wmts/epsg4326/best/MODIS_Combined_Flood_3-Day/default"
LAADS = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MCDWD_L3/2023/115/"
# Highest advertised zoom in 250m GIBS EPSG:4326 matrix. Each 512-cell tile spans 1.125 degrees.
ZOOM, SIZE, SPAN = 8, 512, 1.125


def tiles_for(geometry):
    xmin, ymin, xmax, ymax = geometry.bounds
    cols = range(math.floor((xmin+180)/SPAN), math.floor((xmax+180)/SPAN)+1)
    rows = range(math.floor((90-ymax)/SPAN), math.floor((90-ymin)/SPAN)+1)
    return [(row, col) for row in rows for col in cols]


def collect(url, path):
    resp = public_get(url)
    path.write_bytes(resp.content)
    return {"url": url, "sha256": sha256(resp.content).hexdigest(), "bytes": len(resp.content),
            "retrieved_utc": datetime.now(timezone.utc).isoformat(), "status": resp.status_code, "filename": path.name}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geometry", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--phase50-pixels", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output/"raw_modis"
    cache.mkdir(exist_ok=True)
    audit = [collect(COLOR_MAP, cache/"modis_palette.xml")]
    mapping_xml = ET.parse(cache/"modis_palette.xml")
    palette = {tuple(map(int, entry.attrib["rgb"].split(","))): entry.attrib["tooltip"]
               for entry in mapping_xml.findall(".//LegendEntry")}
    assert set(palette.values()) == {"No Water", "Surface Water", "Recurring Flood", "Flood", "Insufficient Data"}
    data = json.loads(args.geometry.read_text())["features"]
    sites = {(z["properties"]["county_fips"], z["properties"]["site_index"]): shape(z["geometry"])
             for z in data if z["properties"]["disasterNumber"] == 4722}
    assert len(sites) == 6
    tiles = sorted({tile for geom in sites.values() for tile in tiles_for(geom)})
    days = [f"{year}-04-{day:02d}" for year in (2022, 2023, 2024) for day in (10, 15, 20, 25, 30)]
    def fetch(item):
        date, (row, col) = item
        url = f"{LAYER}/{date}/250m/{ZOOM}/{row}/{col}.png"
        filename = f"{date}_{row}_{col}.png"
        try:
            return collect(url, cache/filename)
        except requests.RequestException as exc:
            return {"url": url, "filename": filename, "status": "retrieval_failed", "error": str(exc)[:180]}
    with ThreadPoolExecutor(max_workers=6) as pool:
        audit.extend(pool.map(fetch, [(day, tile) for day in days for tile in tiles]))
    pd.DataFrame(audit).to_csv(args.output/"dr4722_modis_source_manifest.csv", index=False)

    rows = []
    for date in days:
        for (county, site), geom in sites.items():
            row = {"date": date, "county_fips": county, "site_index": site, "status": "read_ok"}
            pixels = []
            for y, x in tiles_for(geom):
                file = cache/f"{date}_{y}_{x}.png"
                if not file.exists():
                    row["status"] = "missing_tile"
                    continue
                values = np.asarray(Image.open(file).convert("RGBA"))
                assert values.shape == (SIZE, SIZE, 4)
                tf = from_origin(-180+x*SPAN, 90-y*SPAN, SPAN/SIZE, SPAN/SIZE)
                mask = geometry_mask([mapping(geom)], (SIZE, SIZE), tf, invert=True)
                pixels.extend(values[mask])
            if row["status"] == "read_ok":
                p = np.array(pixels)
                visible = p[:, -1] > 0
                counts = {label.lower().replace(" ", "_"): int(((p[:, :3] == rgb).all(axis=1) & visible).sum())
                          for rgb, label in palette.items()}
                assert sum(counts.values()) == visible.sum()
                row.update(counts, total_cells=len(p), transparent_unresolved=int((~visible).sum()))
            rows.append(row)
    windows = pd.DataFrame(rows)
    windows.to_csv(args.output/"dr4722_modis_windows.csv", index=False)
    if args.phase50_pixels:
        overlay = []
        for county, site in sites:
            sample = f"4722_{county}_{site}"
            classes = []
            for month in ("2022-03", "2022-04", "2023-03", "2023-04"):
                with np.load(args.phase50_pixels/f"{sample}_{month}.npz") as data:
                    classes.append(data["values"])
                    mask, affine = data["mask"], data["transform"]
            a, b, c, d = classes
            gain = (a > 0) & (b > 0) & (c == 1) & (d == 2)
            iy, ix = np.argwhere(mask)[gain].T
            x = affine[0]*(ix + .5) + affine[1]*(iy + .5) + affine[2]
            y = affine[3]*(ix + .5) + affine[4]*(iy + .5) + affine[5]
            gx = np.floor((x+180)*SIZE/SPAN).astype(int)
            gy = np.floor((90-y)*SIZE/SPAN).astype(int)
            for date in days:
                counts, checked = 0, 0
                for tr, tc in set(zip(gy//SIZE, gx//SIZE)):
                    file = cache/f"{date}_{tr}_{tc}.png"
                    local = (gy//SIZE == tr) & (gx//SIZE == tc)
                    if file.exists():
                        data = np.array(Image.open(file).convert("RGBA"))
                        pic = data[gy[local]%SIZE, gx[local]%SIZE]
                        counts += int(((pic[:, :3] == [250, 30, 36]).all(axis=1) & (pic[:, 3] > 0)).sum())
                        checked += int(local.sum())
                overlay.append({"date": date, "sample_id": sample, "jrc_gain_centers": int(gain.sum()),
                                "gaining_centers_with_tile": checked, "gaining_centers_in_rendered_flood_cell": counts})
        pd.DataFrame(overlay).to_csv(args.output/"dr4722_modis_jrc_overlay.csv", index=False)
    print(windows.groupby(windows.date.str[:4])[["flood", "recurring_flood", "surface_water", "insufficient_data", "transparent_unresolved"]].sum().to_string())
    listing = public_get(LAADS)
    found = sorted(set(re.findall(r"MCDWD_L3\.A2023115\.h1[12]v04\.061\.\d+\.hdf", listing.text)))
    assert found
    outcome = []
    for name in found:
        response = requests.get(LAADS+name, timeout=20, allow_redirects=False, stream=True)
        outcome.append({"url": LAADS+name, "status": response.status_code,
                        "redirect_path": (response.headers.get("Location") or "").split("?")[0],
                        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                        "used_for_pixels": False})
        response.close()
    pd.DataFrame(outcome).to_csv(args.output/"laads_archive_access_check.csv", index=False)


if __name__ == "__main__":
    main()
