#!/usr/bin/env python3
"""Build traceable retrospective hazard evidence and a matched FEMA analysis table."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Geod
from shapely import make_valid
from shapely.geometry import Point, shape
from shapely.ops import nearest_points, unary_union

GEOD = Geod(ellps="WGS84")
FLOOD = {"Flood", "Flash Flood", "Coastal Flood", "Lakeshore Flood", "Heavy Rain"}
CONVECTIVE = {"Thunderstorm Wind", "Hail", "Tornado", "Strong Wind", "Heavy Rain"}
WINTER = {"Winter Storm", "Winter Weather", "Ice Storm", "Heavy Snow", "Blizzard", "Extreme Cold/Wind Chill"}
TROPICAL = {"Hurricane", "Tropical Storm", "Typhoon", "Coastal Storm"}
NOAA_SCOPE = {
    "Flood": FLOOD, "Severe Storm": CONVECTIVE | FLOOD,
    "Tornado": {"Tornado"}, "Straight-Line Winds": {"Thunderstorm Wind", "Strong Wind"},
    "Snowstorm": WINTER, "Severe Ice Storm": WINTER, "Winter Storm": WINTER,
    "Freezing": WINTER | {"Frost/Freeze"}, "Fire": {"Wildfire"},
    "Mud/Landslide": {"Debris Flow", "Heavy Rain", "Flash Flood"},
    "Dam/Levee Break": FLOOD,
    **{h: FLOOD | {"Hurricane", "Hurricane (Typhoon)", "Tropical Storm", "Storm Surge/Tide"} for h in TROPICAL},
}
CUTS = [100000, 1000000, 50000000, 200000000, 500000000]
BANDS = ["0-100K", "100K-1M", "1M-50M", "50-200M", "200-500M", "500M+"]
PHYSICAL = [
    "measured_gust_kt", "estimated_gust_kt", "hail_in", "tornado_ef", "tornado_length_mi",
    "tornado_width_yd", "cyclone_max_wind_kt", "cyclone_min_pressure_mb",
    "cyclone_max_34kt_radius_nm", "cyclone_landfall_records", "mtbs_burned_acres",
    "quake_magnitude", "quake_depth_km", "quake_shakemap_max_mmi",
    "daymet_point_max_24h_mm", "daymet_point_incident_total_mm", "usgs_discharge_anomaly",
]
EXPOSURE = ["designated_county_count", "designated_2020_population"]
PROXY = ["noaa_event_count", "noaa_episode_count", "noaa_county_count"]


def days(values):
    return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()


def read_master(path):
    d = pd.read_excel(path)
    assert len(d) == 971 and d.disasterNumber.nunique() == 971
    fields = ["disasterNumber", "fyDeclared", "state", "incidentType", "declarationType",
              "declarationDate", "incidentBeginDate", "incidentEndDate", "ihProgramDeclared",
              "iaProgramDeclared", "paProgramDeclared", "hmProgramDeclared", "totalObligatedFunding"]
    d = d[fields].copy()
    d["target"] = d.pop("totalObligatedFunding").clip(lower=0)
    d["actual_band"] = np.searchsorted(CUTS, d.target, side="right")
    d["start"], d["end"], d["declaration"] = [days(d.pop(c)) for c in
        ("incidentBeginDate", "incidentEndDate", "declarationDate")]
    d["incident_duration_days"] = (d.end - d.start).dt.days.add(1).where(d.end >= d.start)
    d["declaration_delay_days"] = (d.declaration - d.start).dt.days
    d["incident_month"] = d.start.dt.month
    assert d.target.notna().all() and d.start.notna().all() and d.declaration.notna().all()
    return d


def fema_geography(raw, master):
    files = sorted(raw.glob("fema_*_[0-9].json"))
    assert files, "OpenFEMA designated-area pages missing"
    rows = [row for p in files for row in json.loads(p.read_text())["DisasterDeclarationsSummaries"]]
    raw_fema = pd.DataFrame(rows)
    assert not raw_fema.id.duplicated().any()
    fema = raw_fema[raw_fema.disasterNumber.isin(master.disasterNumber)].copy()
    fema["new_start"] = days(fema.incidentBeginDate)
    fema["new_end"] = days(fema.incidentEndDate)
    fema = fema.merge(master[["disasterNumber", "state", "start", "end"]], on="disasterNumber",
                      suffixes=("", "_workbook"), validate="many_to_one")
    fema["dates_agree"] = fema.new_start.eq(fema.start) & (fema.new_end.eq(fema.end) | (fema.new_end.isna() & fema.end.isna()))
    fema["state_agrees"] = fema.state.eq(fema.state_workbook)
    s = pd.to_numeric(fema.fipsStateCode, errors="coerce")
    c = pd.to_numeric(fema.fipsCountyCode, errors="coerce")
    fema["county_fips"] = (s * 1000 + c).where(s.between(1, 99) & c.between(1, 998)).astype("Int64")
    selected = fema[fema.dates_agree & fema.state_agrees]
    counties = selected.dropna(subset=["county_fips"]).drop_duplicates(["disasterNumber", "county_fips"]).copy()
    titles = selected.groupby("disasterNumber").declarationTitle.agg(lambda x: " | ".join(sorted(set(x.dropna()))))
    return fema, counties, titles


def noaa_reports(raw, county, master):
    scope = set().union(*NOAA_SCOPE.values())
    take = ["EVENT_ID", "EPISODE_ID", "STATE_FIPS", "CZ_TYPE", "CZ_FIPS", "EVENT_TYPE",
            "BEGIN_DATE_TIME", "END_DATE_TIME", "MAGNITUDE", "MAGNITUDE_TYPE", "TOR_F_SCALE",
            "TOR_LENGTH", "TOR_WIDTH"]
    tables = []
    files = sorted(raw.glob("StormEvents_details*.csv.gz"))
    assert len(files) == 16
    for file in files:
        d = pd.read_csv(file, usecols=take, low_memory=False, encoding="latin-1")
        d = d[d.CZ_TYPE.eq("C") & d.EVENT_TYPE.isin(scope)].copy()
        d["archive_file"] = file.name
        tables.append(d)
    n = pd.concat(tables, ignore_index=True)
    n["county_fips"] = pd.to_numeric(n.STATE_FIPS, errors="coerce") * 1000 + pd.to_numeric(n.CZ_FIPS, errors="coerce")
    n["noaa_begin"] = pd.to_datetime(n.BEGIN_DATE_TIME, errors="coerce", format="%d-%b-%y %H:%M:%S").dt.normalize()
    n["noaa_end"] = pd.to_datetime(n.END_DATE_TIME, errors="coerce", format="%d-%b-%y %H:%M:%S").dt.normalize()
    assert n.noaa_begin.notna().mean() > .99
    n = n.dropna(subset=["county_fips", "noaa_begin", "noaa_end"]).drop_duplicates("EVENT_ID")
    n = n[n.noaa_begin <= n.noaa_end]
    c = county[["disasterNumber", "county_fips", "start", "end"]].merge(master[["disasterNumber", "incidentType"]], on="disasterNumber")
    c = c[c.end.notna() & c.incidentType.isin(NOAA_SCOPE)]
    m = c.merge(n, on="county_fips", how="inner")
    m = m[m.noaa_begin.le(m.end) & m.noaa_end.ge(m.start)].copy()
    m = m[[kind in NOAA_SCOPE[hazard] for hazard, kind in zip(m.incidentType, m.EVENT_TYPE)]].copy()
    m = m.drop_duplicates(["disasterNumber", "EVENT_ID"])
    attribution = m.groupby("EVENT_ID").disasterNumber.transform("nunique")
    m["evidence_class"] = np.where(attribution.gt(1), "REJECTED", "UNVERIFIED CANDIDATE")
    m["shared_declaration_count"] = attribution
    m["measurement_mag"] = pd.to_numeric(m.MAGNITUDE, errors="coerce")
    m["measurement_ef"] = pd.to_numeric(m.TOR_F_SCALE.astype(str).str.extract(r"^EF([0-5])$")[0], errors="coerce")
    for new, raw_col, upper in [("measurement_length", "TOR_LENGTH", 500), ("measurement_width", "TOR_WIDTH", 10000)]:
        values = pd.to_numeric(m[raw_col], errors="coerce")
        m[new] = values.where(values.gt(0) & values.le(upper) & m.EVENT_TYPE.eq("Tornado"))
    rows = []
    for number, group in m[m.evidence_class.ne("REJECTED")].groupby("disasterNumber"):
        mag = group.measurement_mag
        rows.append({
            "disasterNumber": number, "noaa_event_count": group.EVENT_ID.nunique(),
            "noaa_episode_count": group.EPISODE_ID.nunique(), "noaa_county_count": group.county_fips.nunique(),
            "measured_gust_kt": mag[group.MAGNITUDE_TYPE.eq("MG") & mag.between(1, 250)].max(),
            "estimated_gust_kt": mag[group.MAGNITUDE_TYPE.eq("EG") & mag.between(1, 250)].max(),
            "hail_in": mag[group.EVENT_TYPE.eq("Hail") & mag.between(.1, 12)].max(),
            "tornado_ef": group.measurement_ef.max(),
            "tornado_length_mi": group.measurement_length.max(), "tornado_width_yd": group.measurement_width.max(),
        })
    return m, pd.DataFrame(rows)


def coordinate(value):
    return float(value[:-1]) * (-1 if value[-1] in "SW" else 1)


def best_track(raw, master, areas, artifacts):
    tracks = defaultdict(list)
    headers = {}
    for path in sorted(raw.glob("hurdat_*.txt")):
        for text in path.read_text().splitlines():
            row = [s.strip() for s in text.split(",")]
            if re.fullmatch(r"(AL|EP|CP)\d{6}", row[0]):
                key = row[0]
                headers[key] = row[1]
            elif row[0].isdigit() and 2009 <= int(row[0][:4]) <= 2024:
                tracks[key].append((pd.Timestamp(row[0]), row[2], row[3], coordinate(row[4]), coordinate(row[5]),
                                    int(row[6]), int(row[7]), max(map(int, row[8:12]))))
    names = pd.read_csv(artifacts / "nhc" / "storm_matching_and_timestamp_audit.csv")
    names = names.set_index("disasterNumber")
    matches = []
    for z in master[master.incidentType.isin(TROPICAL)].itertuples(index=False):
        if z.disasterNumber not in names.index or z.disasterNumber not in areas or pd.isna(z.end):
            continue
        storm = names.loc[z.disasterNumber, "named_storm"]
        if pd.isna(storm):
            continue
        candidates = [key for key in tracks if headers[key] == storm and abs(int(key[-4:]) - z.start.year) <= 1]
        for key in candidates:
            local = []
            for date, marker, status, lat, lon, wind, pressure, radius in tracks[key]:
                if not z.start <= date <= z.end:
                    continue
                p = Point(lon, lat)
                close = nearest_points(p, areas[z.disasterNumber])[1]
                distance = abs(GEOD.inv(lon, lat, close.x, close.y)[2]) / 1000
                if distance <= 250:
                    local.append((date, marker, status, wind, pressure, radius, distance))
            if not local:
                continue
            samples = pd.DataFrame(local, columns=["date", "marker", "phase", "wind", "pressure", "radius", "distance"])
            matches.append({"disasterNumber": z.disasterNumber, "source_event_id": key, "storm_name": storm,
                "source_first_date": samples.date.min().date(), "source_last_date": samples.date.max().date(),
                "minimum_track_distance_km": samples.distance.min(), "nearby_best_track_fixes": len(samples),
                "cyclone_max_wind_kt": samples.wind.where(samples.wind.gt(0)).max(),
                "cyclone_min_pressure_mb": samples.pressure.where(samples.pressure.gt(0)).min(),
                "cyclone_max_34kt_radius_nm": samples.radius.where(samples.radius.ge(0)).max(),
                "cyclone_landfall_records": int(samples.marker.eq("L").sum()),
            })
    result = pd.DataFrame(matches)
    if len(result):
        result["evidence_class"] = np.where(result.disasterNumber.duplicated(keep=False), "REJECTED", "VERIFIED DERIVED MEASUREMENT")
    return result


def fire_and_quake(raw, master, areas, titles):
    fires, quakes = [], []
    quake_catalog = {}
    for file in sorted(raw.glob("comcat_*.geojson")):
        quake_catalog.update({x["id"]: x for x in json.loads(file.read_text())["features"]})
    for z in master[master.incidentType.isin(["Fire", "Earthquake"])].itertuples(index=False):
        if z.disasterNumber not in areas or pd.isna(z.end):
            continue
        area = areas[z.disasterNumber]
        if z.incidentType == "Fire":
            path = raw / f"mtbs_{z.disasterNumber}.geojson"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            assert data["type"] == "FeatureCollection" and len(data["features"]) < 500
            for hit in data["features"]:
                if not hit["geometry"]:
                    continue
                footprint = make_valid(shape(hit["geometry"]))
                if not area.intersects(footprint):
                    continue
                p = {k.lower(): v for k, v in hit["properties"].items()}
                text = str(titles.get(z.disasterNumber, "")).upper()
                normalized = " ".join(re.findall(r"[A-Z0-9]+", text))
                token = " ".join(re.findall(r"[A-Z0-9]+", p["fire_name"].upper()))
                name_agrees = len(token) >= 4 and f" {token} " in f" {normalized} " and token not in {"FIRE", "WILDFIRE", "COMPLEX", "UNNAMED"}
                if pd.notna(p["acres"]) and p["acres"] > 0:
                    fires.append({"disasterNumber": z.disasterNumber, "source_event_id": p["fire_id"],
                        "source_name": p["fire_name"], "ignition_date": pd.to_datetime(str(p["ig_date"]), format="%Y%m%d").date(),
                        "irwinid": p["irwinid"], "map_id": p["map_id"], "mtbs_burned_acres": p["acres"],
                        "geometry_intersects_designated_county": True, "name_agrees": name_agrees,
                        "evidence_class": "VERIFIED DERIVED MEASUREMENT" if name_agrees else "UNVERIFIED CANDIDATE"})
        else:
            for identifier, hit in quake_catalog.items():
                p = hit["properties"]
                date = pd.to_datetime(p["time"], unit="ms").normalize()
                if not z.start <= date <= z.end:
                    continue
                lon, lat, depth = hit["geometry"]["coordinates"]
                if not area.covers(Point(lon, lat)):
                    continue
                quakes.append({"disasterNumber": z.disasterNumber, "source_event_id": identifier,
                    "source_date": date.date(), "source_url": p["url"], "quake_magnitude": p["mag"],
                    "magnitude_type": p["magType"], "quake_depth_km": depth,
                    "quake_shakemap_max_mmi": p["mmi"], "point_inside_designated_county": True,
                    "event_begins_incident": date == z.start,
                    "evidence_class": "UNVERIFIED CANDIDATE"})
    return pd.DataFrame(fires), pd.DataFrame(quakes)


def attach_groups(data, noaa, cyclone, fire, quakes):
    parent = {i: i for i in data.disasterNumber}
    def find(node):
        while node != parent[node]:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    def connect(groups):
        for members in groups:
            keys = list(dict.fromkeys(members))
            for other in keys[1:]:
                a, b = find(keys[0]), find(other)
                parent[max(a, b)] = min(a, b)
    connect(noaa.groupby("EPISODE_ID").disasterNumber.agg(list))
    for table in (cyclone, fire, quakes):
        if len(table):
            connect(table.groupby("source_event_id").disasterNumber.agg(list))
    for _, cohort in data.groupby(["state", "incidentType"]):
        records = list(cohort[["disasterNumber", "start", "end"]].itertuples(index=False, name=None))
        for i, (number, start, end) in enumerate(records):
            if pd.isna(end):
                end = start
            connect([[number, other] for other, a, b in records[i+1:] if a <= end and (b if pd.notna(b) else a) >= start])
    data["event_group"] = data.disasterNumber.map(find)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workbook", type=Path, default=Path("master_openfema_40plus.xlsx"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    d = read_master(args.workbook)
    fema, county, titles = fema_geography(args.raw, d)
    fema.to_csv(args.output / "openfema_declaration_row_audit.csv", index=False)
    county[["disasterNumber", "county_fips", "designatedArea", "id"]].to_csv(args.output / "county_keys.csv", index=False)
    d = d.join(titles.rename("declaration_title"), on="disasterNumber")
    geo = gpd.read_file(args.raw / "census_county_boundaries.zip")[["GEOID", "geometry"]].to_crs(4326)
    geo = geo.assign(county_fips=geo.GEOID.astype(int))
    pop = pd.read_csv(args.raw / "census_county_population.csv", encoding="latin-1")
    pop = pop.query("SUMLEV == 50").assign(county_fips=lambda x: 1000*x.STATE + x.COUNTY)
    pop = county.merge(pop[["county_fips", "POPESTIMATE2020"]], on="county_fips", how="left", validate="many_to_one")
    aggregate = pop.groupby("disasterNumber").agg(
        designated_county_count=("county_fips", "nunique"),
        population_counties_observed=("POPESTIMATE2020", "count"),
        designated_2020_population=("POPESTIMATE2020", "sum"),
    )
    aggregate.loc[aggregate.designated_county_count.ne(aggregate.population_counties_observed), "designated_2020_population"] = np.nan
    d = d.join(aggregate, on="disasterNumber")
    area_rows = county[["disasterNumber", "county_fips"]].merge(geo, on="county_fips", validate="many_to_one")
    areas = {number: unary_union([make_valid(g) for g in group.geometry]) for number, group in area_rows.groupby("disasterNumber")}

    reports, report_features = noaa_reports(args.raw, county, d)
    hurricanes = best_track(args.raw, d, areas, args.artifacts)
    fire, quakes = fire_and_quake(args.raw, d, areas, titles)
    for table, file in [(reports, "noaa_candidate_event_links"), (hurricanes, "cyclone_best_track_links"),
                        (fire, "mtbs_perimeter_links"), (quakes, "comcat_candidate_links")]:
        table.to_csv(args.output / f"{file}.csv", index=False)
    d = d.merge(report_features, on="disasterNumber", how="left", validate="one_to_one")
    if len(hurricanes):
        d = d.merge(hurricanes.query("evidence_class != 'REJECTED'")[["disasterNumber", *[c for c in PHYSICAL if c.startswith("cyclone_")]]],
                    on="disasterNumber", how="left", validate="one_to_one")
    if len(fire):
        accepted = fire.query("name_agrees").drop_duplicates(["disasterNumber", "source_event_id"])
        single = accepted[~accepted.disasterNumber.duplicated(keep=False)]
        d = d.merge(single[["disasterNumber", "mtbs_burned_acres"]], on="disasterNumber", how="left", validate="one_to_one")
    if len(quakes):
        started = quakes.query("event_begins_incident")
        single = started[~started.disasterNumber.duplicated(keep=False)]
        d = d.merge(single[["disasterNumber", *[c for c in PHYSICAL if c.startswith("quake_")]]],
                    on="disasterNumber", how="left", validate="one_to_one")
    flood = args.artifacts / "flood25" / "model_inputs"
    rain = pd.read_csv(flood / "multi" / "all_92_multilocation_rainfall_profiles.csv")
    rain_columns = {"multi_event_max_1day_mm_max": "daymet_point_max_24h_mm",
                    "multi_event_precip_total_mm_max": "daymet_point_incident_total_mm"}
    rain = rain[["disasterNumber", *rain_columns]].rename(columns=rain_columns)
    flow = pd.read_csv(flood / "flow" / "flood_stratified_24_normalized_discharge.csv")
    flow_columns = {"peak_to_prior_seasonal_median_ratio": "usgs_discharge_anomaly"}
    flow = flow[["disasterNumber", *flow_columns]].rename(columns=flow_columns)
    d = d.merge(rain, on="disasterNumber", how="left", validate="one_to_one")
    d = d.merge(flow, on="disasterNumber", how="left", validate="one_to_one")
    # Keep independently published source measurements distinct from uncertain event links.
    d = d.reindex(columns=list(dict.fromkeys([*d.columns, *PHYSICAL, *EXPOSURE, *PROXY])))
    attach_groups(d, reports, hurricanes, fire, quakes)
    assert len(d) == 971 and d.disasterNumber.is_unique
    d.to_csv(args.output / "cohort_features.csv", index=False)
    observed = d[[*PHYSICAL, *EXPOSURE, *PROXY]].notna().astype(int).assign(incidentType=d.incidentType)
    coverage = observed.groupby("incidentType").sum()
    coverage.insert(0, "total", d.incidentType.value_counts())
    coverage.to_csv(args.output / "hazard_coverage.csv")
    print(coverage.to_string(), flush=True)


if __name__ == "__main__":
    main()
