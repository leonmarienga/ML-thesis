#!/usr/bin/env python3
"""Build explicit wind and wildfire exposure proxies using public NOAA, NIFC and Census data."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import re

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Geod, Transformer
import requests
import shapely
from shapely import make_valid
from shapely.geometry import GeometryCollection, Point, Polygon, mapping, shape
from shapely.ops import nearest_points, transform, unary_union

from thesis_retrieve_sources import acquire
from thesis_hazard_features import coordinate

GEOD = Geod(ellps="WGS84")
TO_AREA = Transformer.from_crs(4326, 6933, always_xy=True).transform
TIGER = "https://tigerweb.geo.census.gov/arcgis/rest/services/Census2020/Tracts_Blocks/MapServer"
FIRE_SERVICES = {
    "NIFC_history": "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/InterAgencyFirePerimeterHistory_All_Years_View/FeatureServer/0/query",
    "NIFC_WFIGS": "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters/FeatureServer/0/query",
}
CYCLONES = {"Hurricane", "Tropical Storm", "Typhoon", "Coastal Storm"}
GENERIC = {"FIRE", "FIRES", "WILDFIRE", "WILDFIRES", "UNNAMED", "UNKNOWN"}


def canonical(name):
    value = " ".join(re.findall(r"[A-Z0-9]+", str(name).upper()))
    return re.sub(r" (FIRE|WILDFIRE)$", "", value)


def shape_cells(geometry):
    if geometry.geom_type == "GeometryCollection":
        return unary_union([part for part in geometry.geoms if part.geom_type in ("Polygon", "MultiPolygon")])
    return geometry


def nearby_km(geometry, point):
    close = nearest_points(point, geometry)[1]
    return abs(GEOD.inv(point.x, point.y, close.x, close.y)[2])/1000


def quadrant_footprint(lon, lat, radii):
    wedges = []
    for q, distance in enumerate(radii):
        if distance <= 0:
            continue
        angles = np.linspace(q*90, (q+1)*90, 19)
        x, y, _ = GEOD.fwd(np.full(19, lon), np.full(19, lat), angles, np.full(19, distance*1852))
        if np.ptp(x) > 180:
            return GeometryCollection()
        wedges.append(Polygon([(lon, lat), *zip(x, y)]))
    return unary_union(wedges)


def load_tracks(raw, ibtracs):
    tracks = defaultdict(list)
    names = {}
    for p in sorted(raw.glob("hurdat_*.txt")):
        for line in p.read_text().splitlines():
            fields = [s.strip() for s in line.split(",")]
            if re.fullmatch(r"(AL|EP|CP)\d{6}", fields[0]):
                sid = fields[0]
                names[sid] = (canonical(fields[1]), "NHC_HURDAT2")
            elif fields[0].isdigit() and "20090101" <= fields[0] <= "20241231":
                tracks[sid].append((pd.Timestamp(fields[0] + " " + fields[1]), coordinate(fields[5]),
                                    coordinate(fields[4]), int(fields[6]), tuple(map(int, fields[8:12]))))
    extras = []
    columns = ["SID", "SEASON", "NAME", "ISO_TIME", "TRACK_TYPE", "USA_LAT", "USA_LON", "USA_WIND",
               "USA_R34_NE", "USA_R34_SE", "USA_R34_SW", "USA_R34_NW", "USA_ATCF_ID"]
    for p in sorted(ibtracs.glob("ibtracs.*.csv")):
        d = pd.read_csv(p, usecols=columns, skiprows=[1], skipinitialspace=True, low_memory=False)
        d["time"] = pd.to_datetime(d.ISO_TIME, errors="coerce")
        d = d[d.time.dt.year.between(2009, 2024) & d.TRACK_TYPE.str.strip().eq("main")].copy()
        d = d[d.time.dt.minute.eq(0) & d.time.dt.hour.mod(6).eq(0)]
        for field in columns[5:-1]:
            d[field] = pd.to_numeric(d[field], errors="coerce")
        d = d.dropna(subset=["SID", "time", "USA_LAT", "USA_LON", "USA_WIND"])
        extras.append(d)
        for row in d.itertuples(index=False):
            names[row.SID] = (canonical(row.NAME), f"NOAA_IBTrACS_{p.name.split('.')[1]}")
            radii = (row.USA_R34_NE, row.USA_R34_SE, row.USA_R34_SW, row.USA_R34_NW)
            tracks[row.SID].append((row.time, row.USA_LON, row.USA_LAT, row.USA_WIND, radii))
    if extras:
        snapshots = pd.concat(extras).drop_duplicates(["SID","time"])
    else:
        snapshots = pd.DataFrame()
    return names, tracks, snapshots


def cyclone_links(cohort, geometry, raw, ibtracs, advisory_names, titles, known):
    names, tracks, extract = load_tracks(raw, ibtracs)
    by_id = advisory_names.set_index("disasterNumber").named_storm
    official = titles[titles.case_hazard.eq("cyclone")].groupby("disasterNumber").matched_name.agg(list)
    audits, footprints = [], {}
    for z in cohort[cohort.incidentType.isin(CYCLONES)].itertuples(index=False):
        case_title = canonical(z.declaration_title)
        options = set(map(canonical, official.get(z.disasterNumber, [])))
        if z.disasterNumber in by_id.index and pd.notna(by_id[z.disasterNumber]):
            options.add(canonical(by_id[z.disasterNumber]))
        eligible_names = [(sid, name, agency) for sid, (name, agency) in names.items()
                          if name not in ("UNNAMED", "NOT NAMED", "NONAME") and len(name)>2 and
                          (name in options or f" {name} " in f" {case_title} ")]
        fields = {"disasterNumber": z.disasterNumber, "incidentType": z.incidentType,
                  "old_link": z.disasterNumber in known, "title": z.declaration_title}
        if z.disasterNumber not in geometry or pd.isna(z.end):
            audits.append({**fields, "status": "missing_matched_county_or_incident_end"})
            continue
        area = geometry[z.disasterNumber]
        available, in_window = [], 0
        for sid, name, agency in eligible_names:
            points = [fix for fix in tracks.get(sid, []) if z.start <= fix[0] < z.end+pd.Timedelta(days=1)]
            in_window += bool(points)
            nearby = [fix for fix in points if nearby_km(area, Point(fix[1], fix[2])) <= 250]
            if nearby:
                available.append((sid, agency, nearby))
        # Original HURDAT and IBTrACS basins are disjoint in this audit.
        if len(available) != 1:
            reason = ("ambiguous_named_track" if len(available)>1 else
                      "track_outside_250km" if in_window else "no_named_track_in_window")
            audits.append({**fields, "status": reason})
            continue
        sid, agency, fixes = available[0]
        valid = [fix for fix in fixes if np.isfinite(fix[4]).all() and min(fix[4])>=0 and
                 0<max(fix[4])<=500 and fix[3]>=34]
        parts = [quadrant_footprint(x,y,r) for _,x,y,_,r in valid]
        footprint = shape_cells(make_valid(unary_union(parts).intersection(area))) if valid else GeometryCollection()
        status = ("modeled_wind_overlap" if not footprint.is_empty and footprint.area>0 else
                  "no_overlap_of_modeled_radii" if valid else "missing_threshold_radii")
        audits.append({**fields, "source_event_id": sid, "archive": agency, "status": status,
                       "eligible_track_fixes": len(fixes), "complete_positive_radii_fixes": len(valid),
                       "max_nearby_wind_kt": max(f[3] for f in fixes),
                       "wind_county_overlap_km2": transform(TO_AREA, footprint).area/1e6 if valid else np.nan})
        if status == "modeled_wind_overlap":
            footprints[(z.disasterNumber, "wind")] = footprint
    return pd.DataFrame(audits), footprints, extract


def arc_query(base, query, cache, name):
    url = requests.Request("GET", base, params=query).prepare().url
    metadata = cache/f"{name}.json"
    if metadata.exists() and json.loads(metadata.read_text())["download_url"] != url:
        name=f"{Path(name).stem}_{sha256(url.encode()).hexdigest()[:12]}.geojson"
    try:
        meta = acquire("public_arcgis_spatial", url, name, cache)
        doc = json.loads((cache/name).read_text())
    except (ValueError, requests.RequestException) as exc:
        return [], {"status": type(exc).__name__, "filename": name, "download_url": url, "error": str(exc)[:200]}
    if doc.get("error") or doc.get("exceededTransferLimit") or doc.get("properties", {}).get("exceededTransferLimit"):
        return [], {**meta, "status": "api_error_or_truncated", "error": str(doc.get("error", ""))[:250]}
    return doc["features"], {**meta, "status": "retrieved"}


def fire_tasks(row, area, reviewed_names, cache):
    dn = row.disasterNumber
    hits, manifests = [], []
    names = reviewed_names.get(dn, [])
    if not names or area is None or pd.isna(row.end):
        return hits, manifests
    quoted = ",".join("'" + x.replace("'", "''") + "'" for x in sorted(set(names+ [n+" FIRE" for n in names])))
    earliest = row.start-pd.Timedelta(days=10)
    latest = row.end+pd.Timedelta(days=31)
    sources = [("NIFC_history",
        f"FIRE_YEAR_INT IN ({earliest.year},{row.end.year}) AND UPPER(INCIDENT) IN ({quoted})",
        "OBJECTID,INCIDENT,DATE_CUR,IRWINID,FORID,UNQE_FIRE_ID,SOURCE,GIS_ACRES")]
    if row.end.year >= 2020:
        w = f"attr_IncidentTypeCategory = 'WF' AND attr_FireDiscoveryDateTime >= TIMESTAMP '{earliest:%Y-%m-%d} 00:00:00' AND attr_FireDiscoveryDateTime < TIMESTAMP '{latest:%Y-%m-%d} 00:00:00'"
        w += f" AND (UPPER(attr_IncidentName) IN ({quoted}) OR UPPER(poly_IncidentName) IN ({quoted}))"
        sources.append(("NIFC_WFIGS", w,
                        "OBJECTID,attr_IncidentName,poly_IncidentName,attr_FireDiscoveryDateTime,poly_DateCurrent,poly_PolygonDateTime,attr_IrwinID,attr_UniqueFireIdentifier,poly_Source,poly_GISAcres"))
    for source, where, columns in sources:
        xmin,ymin,xmax,ymax = area.bounds
        params = {"where": where, "outFields": columns, "f": "geojson", "outSR": 4326,
                  "returnGeometry": "true", "geometryType": "esriGeometryEnvelope", "inSR": 4326,
                  "geometry": f"{xmin},{ymin},{xmax},{ymax}", "spatialRel": "esriSpatialRelIntersects",
                  "resultRecordCount": 1000, "geometryPrecision": 6, "orderByFields": "OBJECTID"}
        source_rows, log = arc_query(FIRE_SERVICES[source], params, cache, f"{source}_{dn}{'_scope' if source == 'NIFC_WFIGS' else ''}.geojson")
        manifests.append({**log, "disasterNumber":dn, "archive":source})
        for entry in source_rows:
            if not entry["geometry"]:
                continue
            q=entry["properties"]
            if source == "NIFC_history":
                name = canonical(q["INCIDENT"])
                date = pd.to_datetime(str(q["DATE_CUR"])[:8], format="%Y%m%d", errors="coerce")
                discovery = pd.NaT
                identifier = q["IRWINID"] or q["UNQE_FIRE_ID"] or q["FORID"]
            else:
                name = canonical(q["attr_IncidentName"] or q["poly_IncidentName"])
                date = pd.to_datetime(q["poly_PolygonDateTime"] or q["poly_DateCurrent"],unit="ms",errors="coerce")
                discovery = pd.to_datetime(q["attr_FireDiscoveryDateTime"],unit="ms",errors="coerce")
                identifier = q["attr_IrwinID"] or q["attr_UniqueFireIdentifier"]
            perimeter = shape_cells(make_valid(shape(entry["geometry"])))
            overlapping = shape_cells(make_valid(perimeter.intersection(area)))
            accepted = name in names and not overlapping.is_empty and overlapping.area > 0
            accepted = accepted and ((pd.notna(discovery) and earliest <= discovery < latest) or
                                     (pd.isna(discovery) and pd.notna(date) and earliest <= date < latest))
            hits.append({"disasterNumber":dn,"fire_name":name,"archive":source,"geometry":overlapping,
                         "source_object_id":q["OBJECTID"], "incident_id":identifier,
                         "perimeter_date":date, "discovery_date":discovery, "admissible":accepted,
                         "area_m2":transform(TO_AREA,overlapping).area})
    return hits, manifests


def wildfire_links(cohort, geo, government_names, mtbs_links, raw, cache):
    official = government_names[government_names.case_hazard.eq("fire")].groupby("disasterNumber").matched_name.agg(list)
    recorded = mtbs_links[mtbs_links.name_agrees].groupby("disasterNumber").source_name.agg(list)
    names = {int(k): sorted({canonical(n) for n in list(official.get(k,[]))+list(recorded.get(k,[]))} - GENERIC)
             for k in set(official.index)|set(recorded.index)}
    rows = list(cohort.query("incidentType == 'Fire'").itertuples(index=False))
    with ThreadPoolExecutor(max_workers=7) as pool:
        batches = list(pool.map(lambda z: fire_tasks(z,geo.get(z.disasterNumber),names,cache), rows))
    observations = [x for group,_ in batches for x in group]
    audits, manifests, footprints = [], [x for _,log in batches for x in log], {}
    for z in rows:
        dn = z.disasterNumber
        source_parts = [o for o in observations if o["disasterNumber"] == dn and o["admissible"]]
        selected = {}
        for name in names.get(dn,[]):
            hits = [a for a in source_parts if a["fire_name"] == name]
            if hits:
                first = max(hits,key=lambda a:(a["archive"]=="NIFC_WFIGS",a["area_m2"]))
                same_site = all(a["geometry"].intersects(first["geometry"]) for a in hits)
                if same_site:
                    selected[name] = first["geometry"]
            # Supplemental names require the same independent date/state/geometry gates.
            if name not in selected and dn in geo:
                file = raw/f"mtbs_{dn}.geojson"
                if file.exists():
                    matches = []
                    for feature in json.loads(file.read_text())["features"]:
                        prop = feature["properties"]
                        if canonical(prop["fire_name"]) != name or not str(prop["fire_id"]).startswith(z.state):
                            continue
                        discovery = pd.to_datetime(str(prop["ig_date"]), format="%Y%m%d", errors="coerce")
                        if pd.isna(discovery) or not z.start-pd.Timedelta(days=10) <= discovery <= z.end:
                            continue
                        shapely_geom = shape_cells(make_valid(shape(feature["geometry"])).intersection(geo[dn]))
                        if not shapely_geom.is_empty and shapely_geom.area > 0:
                            matches.append((prop["fire_id"], discovery, shapely_geom))
                    if len(matches)==1:
                        identifier, discovery, geom = matches[0]
                        selected[name] = geom
                        observations.append({"disasterNumber":dn,"fire_name":name,"archive":"USGS_USFS_MTBS",
                            "geometry":geom,"source_object_id":identifier,"incident_id":identifier,
                            "perimeter_date":pd.NaT,"discovery_date":discovery,"admissible":True,
                            "area_m2":transform(TO_AREA,geom).area})
        status = "no_named_evidence" if not names.get(dn) else "no_matched_perimeter"
        if selected:
            union = shape_cells(make_valid(unary_union(list(selected.values()))))
            if not union.is_empty and union.area>0:
                footprints[(dn,"fire")] = union
                status = "all_reviewed_names_mapped" if len(selected)==len(names[dn]) else "partial_reviewed_names"
        audits.append({"disasterNumber":dn,"status":status,"expected_reviewed_names":"|".join(names.get(dn,[])),
                       "mapped_reviewed_names":"|".join(sorted(selected)),
                       "nifc_observations":len(source_parts), "names_from_supplemental_documents":dn in official.index,
                       "fire_mapped_union_acres":transform(TO_AREA,footprints[(dn,"fire")]).area/4046.8564224 if (dn,"fire") in footprints else np.nan})
    for x in observations:
        x.pop("geometry")
    return pd.DataFrame(audits), pd.DataFrame(observations), footprints, manifests


def census_exposure(key, geom, census_year, cache):
    dn, hazard = key
    layer=1 if census_year==2020 else 5
    xmin,ymin,xmax,ymax=geom.bounds
    params={"f":"geojson","where":"1=1","outFields":"GEOID,POP100,HU100,AREALAND",
            "returnGeometry":"true","outSR":4326,"inSR":4326,"geometryType":"esriGeometryEnvelope",
            "geometry":f"{xmin:.6f},{ymin:.6f},{xmax:.6f},{ymax:.6f}",
            "spatialRel":"esriSpatialRelIntersects","orderByFields":"GEOID",
            "geometryPrecision":5,"resultRecordCount":30000}
    name=f"census_{census_year}_{hazard}_{dn}.geojson"
    cells, manifest=arc_query(f"{TIGER}/{layer}/query",params,cache,name)
    info={"disasterNumber":dn,"hazard":hazard,"census_vintage":census_year}
    if not cells or len(cells)>=30000:
        return {**info,"exposure_status":"no_complete_census_response"}, [], {**manifest,**info}
    footprint=transform(TO_AREA,geom)
    shapely.prepare(footprint)
    # Clip the bounding-box response before overlay. Vectorized area operations
    # keep large multistate storm footprints tractable.
    usable = [f for f in cells if f["geometry"] and f["properties"]["POP100"] is not None
              and f["properties"]["HU100"] is not None]
    if not usable:
        return {**info,"exposure_status":"missing_census_attributes"},[],{**manifest,**info}
    groups=gpd.GeoDataFrame.from_features(usable, crs="EPSG:4326").to_crs(6933)
    groups.geometry=groups.geometry.make_valid()
    touching=shapely.intersects(footprint, groups.geometry.to_numpy())
    groups=groups[touching].copy()
    local=groups.geometry.to_numpy()
    overlaps=shapely.area(shapely.intersection(local,footprint))
    extent=shapely.area(local)
    positive=(overlaps>1)&(extent>0)
    fraction=np.minimum(overlaps[positive]/extent[positive],1)
    audit=groups.loc[positive,["GEOID","POP100","HU100"]].rename(columns={
        "GEOID":"geoid","POP100":"population","HU100":"housing_units"})
    audit=audit.assign(**info,total_polygon_m2=extent[positive],overlap_m2=overlaps[positive],
                       overlap_fraction=fraction,fully_covered=fraction>=1-1e-6)
    blocks=audit.to_dict("records")
    if not len(audit):
        return {**info,"exposure_status":"no_census_group_with_positive_area"},blocks,{**manifest,**info}
    assert audit.geoid.is_unique and audit.overlap_fraction.between(0,1).all()
    out={**info,"exposure_status":"area_weighted_model_estimate","n_blockgroups":len(audit)}
    for name in ("population","housing_units"):
        out[f"{hazard}_blockgroup_{'housing' if name=='housing_units' else name}_estimate"]=(audit[name]*audit.overlap_fraction).sum()
        out[f"{name}_whole_group_lower_bound"]=audit.loc[audit.fully_covered,name].sum()
        out[f"{name}_all_touched_upper_bound"]=audit[name].sum()
    return out,blocks,{**manifest,**info}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--research",type=Path,default=Path("research"))
    p.add_argument("--raw",type=Path,required=True)
    p.add_argument("--ibtracs",type=Path,required=True)
    p.add_argument("--advisories",type=Path,required=True)
    p.add_argument("--event-names",type=Path,required=True)
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    args.cache.mkdir(parents=True,exist_ok=True)
    args.output.mkdir(parents=True,exist_ok=True)
    root=args.research/"results"/"hazards"
    cohort=pd.read_csv(root/"cohort_features.csv",parse_dates=["start","end","declaration"])
    assert cohort.fyDeclared.max()==2024
    keys=pd.read_csv(root/"county_keys.csv")[["disasterNumber","county_fips"]]
    counties=gpd.read_file(args.raw/"census_county_boundaries.zip")[["GEOID","geometry"]].to_crs(4326)
    source=keys.merge(counties.assign(county_fips=counties.GEOID.astype(int)),on="county_fips",how="inner")
    scope=set(cohort.loc[cohort.incidentType.isin(CYCLONES|{"Fire"}),"disasterNumber"])
    geo={dn:unary_union([make_valid(p) for p in group.geometry]) for dn,group in source[source.disasterNumber.isin(scope)].groupby("disasterNumber")}
    evidence=pd.read_csv(args.event_names)
    # The original source did not contain a harmonized FIPS row for this territory.
    # FEMA explicitly scoped the expedited Mawar declaration to Guam.
    if 4715 in scope and evidence.query("disasterNumber == 4715").scope.eq("territory of Guam").any():
        guam = counties[counties.GEOID.eq("66010")]
        assert len(guam) == 1
        geo[4715] = guam.geometry.iloc[0]
    archive=pd.read_csv(root/"cyclone_best_track_links.csv")
    known=set(archive.loc[archive.evidence_class.ne("REJECTED"),"disasterNumber"])
    w, wind, ib_view=cyclone_links(cohort,geo,args.raw,args.ibtracs,pd.read_csv(args.advisories),evidence,known)
    w.to_csv(args.output/"cyclone_link_and_radii_audit.csv",index=False)
    if len(ib_view):
        ib_view=ib_view[ib_view.SID.isin(w.source_event_id.dropna())]
        ib_view.to_csv(args.output/"ibtracs_matched_track_fixes.csv",index=False)
    f, fire_links, fire, fire_sources=wildfire_links(cohort,geo,evidence,pd.read_csv(root/"mtbs_perimeter_links.csv"),args.raw,args.cache)
    f.to_csv(args.output/"wildfire_perimeter_coverage.csv",index=False)
    fire_links.to_csv(args.output/"nifc_perimeter_link_audit.csv",index=False)
    footprints=wind|fire
    dates=cohort.set_index("disasterNumber").start
    work=[(key,foot,2020 if dates[key[0]]>=pd.Timestamp("2021-09-16") else 2010,args.cache) for key,foot in sorted(footprints.items())]
    with ThreadPoolExecutor(max_workers=8) as pool:
        exposures=list(pool.map(lambda args:census_exposure(*args),work))
    e=pd.DataFrame([r for r,_,_ in exposures])
    e.to_csv(args.output/"census_exposure_proxy.csv",index=False)
    pd.DataFrame([r for _,rows,_ in exposures for r in rows]).to_csv(args.output/"census_exposure_blocks.csv",index=False)
    pd.DataFrame(fire_sources+[s for _,_,s in exposures]).to_csv(args.output/"spatial_source_manifest.csv",index=False)
    geojson={"type":"FeatureCollection","features":[{"type":"Feature","geometry":mapping(geom),
             "properties":{"disasterNumber":key[0],"hazard":key[1]}} for key,geom in sorted(footprints.items())]}
    (args.output/"retrospective_footprints.geojson").write_text(json.dumps(geojson,separators=(",",":")))
    data=cohort[["disasterNumber"]].copy()
    for frame, columns in [(w,["wind_county_overlap_km2"]),(f,["fire_mapped_union_acres"]),
                           (e[e.hazard.eq("wind")],["wind_blockgroup_population_estimate"]),
                           (e[e.hazard.eq("fire")],["fire_blockgroup_housing_estimate"])]:
        data=data.merge(frame[["disasterNumber",*columns]],on="disasterNumber",how="left",validate="one_to_one")
    data.to_csv(args.output/"spatial_features.csv",index=False)
    print(w.groupby(["incidentType","status"]).size(), f.status.value_counts(), e.exposure_status.value_counts(),sep="\n")


if __name__=="__main__":
    main()
