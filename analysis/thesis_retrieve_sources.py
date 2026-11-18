#!/usr/bin/env python3
"""Retrieve bounded, versioned public source inputs for the retrospective audit."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import urljoin

import pandas as pd
import requests

NOAA = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
FEMA = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"
MTBS = "https://apps.fs.usda.gov/arcx/rest/services/EDW/EDW_MTBS_01/MapServer/63/query"
CENSUS = "https://www2.census.gov/programs-surveys/popest/datasets/2020-2025/counties/totals/co-est2025-alldata.csv"
COUNTIES = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip"
USGS = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def public_get(url):
    for attempt in range(2):
        try:
            response = requests.get(url, timeout=(15, 75))
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 1:
                raise


def acquire(source, url, name, output):
    path = output / name
    metadata = output / f"{name}.json"
    if path.exists() and metadata.exists():
        row = json.loads(metadata.read_text())
        assert sha256(path.read_bytes()).hexdigest() == row["sha256"]
        assert url == row["download_url"]
        return row
    resp = public_get(url)
    if resp.content.lstrip()[:20].lower().startswith((b"<html", b"<!doctype")):
        raise ValueError("HTML returned for data download")
    if len(resp.content) > 160_000_000:
        raise ValueError("Individual source exceeds retrieval budget")
    row = {"source_id": source, "filename": name, "download_url": url,
           "sha256": sha256(resp.content).hexdigest(), "bytes": len(resp.content),
           "retrieved_utc": datetime.now(timezone.utc).isoformat(),
           "last_modified": resp.headers.get("Last-Modified"),
           "etag": resp.headers.get("ETag")}
    path.write_bytes(resp.content)
    metadata.write_text(json.dumps(row, indent=2))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workbook", type=Path, default=Path("master_openfema_40plus.xlsx"))
    parser.add_argument("--manifest", type=Path, help="Replay exact URLs and verify SHA-256")
    parser.add_argument("--hurdat-atlantic")
    parser.add_argument("--hurdat-pacific")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.manifest:
        manifest = pd.read_csv(args.manifest).query("status == 'retrieved'")
        def replay(row):
            existing = args.output / row.filename
            if existing.exists():
                return {"filename": row.filename, "sha256": sha256(existing.read_bytes()).hexdigest()}
            return acquire(row.source_id, row.download_url, row.filename, args.output)
        with ThreadPoolExecutor(max_workers=8) as pool:
            rows = pool.map(replay, manifest.itertuples(index=False))
            for expected, actual in zip(manifest.sha256, rows, strict=True):
                if expected != actual["sha256"]:
                    raise ValueError(f"Source changed: {actual['filename']}")
        return
    index = public_get(NOAA).text
    tasks = [("census_popest", CENSUS, "census_county_population.csv"),
             ("census_geometry", COUNTIES, "census_county_boundaries.zip")]
    for year in range(2009, 2025):
        names = sorted(set(re.findall(rf"StormEvents_details-ftp_v1\.0_d{year}_c\d{{8}}\.csv\.gz", index)))
        if not names:
            raise ValueError(f"Missing archive release for {year}")
        tasks.append(("noaa_stormevents", urljoin(NOAA, names[-1]), names[-1]))
    for yr in range(2009, 2025):
        # A coarse U.S. and territory screen; exact county geometry is checked downstream.
        query = (
            f"?format=geojson&starttime={yr}-01-01&endtime={yr+1}-01-01"
            "&minmagnitude=4&minlatitude=10&maxlatitude=74"
            "&minlongitude=-180&maxlongitude=-60&eventtype=earthquake"
        )
        tasks.append(("usgs_comcat", USGS + query, f"comcat_{yr}.geojson"))
    for source, url, name in [
        ("nhc_hurdat", args.hurdat_atlantic, "hurdat_atlantic.txt"),
        ("nhc_hurdat", args.hurdat_pacific, "hurdat_pacific.txt"),
    ]:
        if url:
            tasks.append((source, url, name))
    master = pd.read_excel(args.workbook)
    columns = ("id,disasterNumber,state,fyDeclared,declarationTitle,incidentBeginDate,"
               "incidentEndDate,fipsStateCode,fipsCountyCode,designatedArea,placeCode")
    def fema_pages(year):
        count, offset = None, 0
        while count is None or offset < count:
            params = {"$filter": f"fyDeclared eq {year}", "$top": 1000, "$skip": offset,
                      "$inlinecount": "allpages", "$orderby": "id", "$select": columns}
            url = requests.Request("GET", FEMA, params=params).prepare().url
            name = f"fema_{year}_{offset}.json"
            result = execute(("openfema", url, name))
            yield result
            if result["status"] != "retrieved":
                break
            raw = json.loads((args.output / name).read_text())
            page = raw["DisasterDeclarationsSummaries"]
            if not page or offset != raw["metadata"]["skip"]:
                raise ValueError(f"Invalid OpenFEMA pagination for FY{year}")
            count = raw["metadata"]["count"] if count is None else count
            if raw["metadata"]["count"] != count or count > 15000:
                raise ValueError(f"Unstable OpenFEMA count for FY{year}")
            offset += len(page)

    for fire in master.query("incidentType == 'Fire'").itertuples(index=False):
        start = pd.to_datetime(fire.incidentBeginDate) - pd.Timedelta(days=10)
        end = pd.to_datetime(fire.incidentEndDate, errors="coerce")
        if pd.isna(end):
            end = start + pd.Timedelta(days=10)
        params = {
            "where": f"ig_date BETWEEN {start:%Y%m%d} AND {end:%Y%m%d} AND fire_type = 'Wildfire' AND fire_id LIKE '{fire.state}%'",
            "outFields": "fire_id,fire_name,ig_date,acres,fire_type,irwinid,map_id",
            "returnGeometry": "true", "outSR": "4326", "f": "geojson", "resultRecordCount": 500,
        }
        url = requests.Request("GET", MTBS, params=params).prepare().url
        tasks.append(("mtbs", url, f"mtbs_{fire.disasterNumber}.geojson"))

    def execute(task):
        try:
            row = acquire(*task, args.output)
            print("OK", task[2], row["bytes"], flush=True)
            return {**row, "status": "retrieved"}
        except (ValueError, requests.RequestException) as exc:
            print("FAILED", task[2], type(exc).__name__, flush=True)
            return {"source_id": task[0], "filename": task[2], "download_url": task[1],
                    "status": type(exc).__name__, "error": str(exc)[:250]}
    with ThreadPoolExecutor(max_workers=8) as pool:
        fema_futures = [pool.submit(lambda y: list(fema_pages(y)), int(y)) for y in sorted(master.fyDeclared.unique())]
        rows = list(pool.map(execute, tasks))
        rows.extend(row for future in fema_futures for row in future.result())
    pd.DataFrame(rows).to_csv(args.output / "retrieval_manifest.csv", index=False)


if __name__ == "__main__":
    main()
