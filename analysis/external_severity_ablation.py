#!/usr/bin/env python3
"""
Retrospective external-severity ablation for the disaster-relief thesis.

This experiment enriches ALL 971 master disasters without using funding or
funding band during matching, then evaluates only after enrichment.

Sources
-------
* NOAA/NCEI Storm Events bulk CSV: state/date/hazard impact features.
* NHC HURDAT2: tropical cyclone track/intensity features.
* CAL FIRE FRAP fire perimeters: California wildfire acreage.
* CAL FIRE DINS: California structure damage.

Important
---------
This is a RETROSPECTIVE DIAGNOSTIC. HURDAT2 is a best-track product and NOAA /
CAL FIRE final damage records can contain information finalized after the
prediction moment. These features are therefore used first to test whether
physical severity closes the information gap. A later t0 experiment must
replace them with time-valid operational observations.
"""

from __future__ import annotations

import gzip
import io
import json
import math
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
    recall_score,
)

from mission_semantic_audit import (
    CURRENT_19,
    build_semantic_rollup,
    fetch_all_mission_assignments,
    normalize_master,
    normalize_model_frame,
    prep_pipeline,
)

ROOT = Path(__file__).resolve().parents[1]
MASTER_PATH = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "external_severity"
CACHE = ROOT / ".cache" / "external_severity"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "thesis-external-severity-audit/1.0"})

NOAA_INDEX = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
NHC_HURDAT_DIR = "https://www.nhc.noaa.gov/data/hurdat/"
FEMA_DECL_URL = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"
CALFIRE_PERIMETER_QUERY = (
    "https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/"
    "California_Historic_Fire_Perimeters/FeatureServer/0/query"
)
CALFIRE_DINS_QUERY = (
    "https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/ArcGIS/rest/services/"
    "POSTFIRE_MASTER_DATA_SHARE/FeatureServer/0/query"
)

# Approximate geographic centers. These are used only for target-blind
# cyclone-to-jurisdiction exposure features, not for matching funding.
STATE_CENTROIDS = {
    "AL": (32.806671, -86.791130), "AK": (61.370716, -152.404419),
    "AZ": (33.729759, -111.431221), "AR": (34.969704, -92.373123),
    "CA": (36.116203, -119.681564), "CO": (39.059811, -105.311104),
    "CT": (41.597782, -72.755371), "DE": (39.318523, -75.507141),
    "FL": (27.766279, -81.686783), "GA": (33.040619, -83.643074),
    "HI": (21.094318, -157.498337), "ID": (44.240459, -114.478828),
    "IL": (40.349457, -88.986137), "IN": (39.849426, -86.258278),
    "IA": (42.011539, -93.210526), "KS": (38.526600, -96.726486),
    "KY": (37.668140, -84.670067), "LA": (31.169546, -91.867805),
    "ME": (44.693947, -69.381927), "MD": (39.063946, -76.802101),
    "MA": (42.230171, -71.530106), "MI": (43.326618, -84.536095),
    "MN": (45.694454, -93.900192), "MS": (32.741646, -89.678696),
    "MO": (38.456085, -92.288368), "MT": (46.921925, -110.454353),
    "NE": (41.125370, -98.268082), "NV": (38.313515, -117.055374),
    "NH": (43.452492, -71.563896), "NJ": (40.298904, -74.521011),
    "NM": (34.840515, -106.248482), "NY": (42.165726, -74.948051),
    "NC": (35.630066, -79.806419), "ND": (47.528912, -99.784012),
    "OH": (40.388783, -82.764915), "OK": (35.565342, -96.928917),
    "OR": (44.572021, -122.070938), "PA": (40.590752, -77.209755),
    "RI": (41.680893, -71.511780), "SC": (33.856892, -80.945007),
    "SD": (44.299782, -99.438828), "TN": (35.747845, -86.692345),
    "TX": (31.054487, -97.563461), "UT": (40.150032, -111.862434),
    "VT": (44.045876, -72.710686), "VA": (37.769337, -78.169968),
    "WA": (47.400902, -121.490494), "WV": (38.491226, -80.954453),
    "WI": (44.268543, -89.616508), "WY": (42.755966, -107.302490),
    "DC": (38.9072, -77.0369),
    "PR": (18.2208, -66.5901), "VI": (18.3358, -64.8963),
    "GU": (13.4443, 144.7937), "MP": (15.0979, 145.6739),
    "AS": (-14.2710, -170.1322),
}

STATE_NAMES = {
    "AL":"ALABAMA","AK":"ALASKA","AZ":"ARIZONA","AR":"ARKANSAS","CA":"CALIFORNIA",
    "CO":"COLORADO","CT":"CONNECTICUT","DE":"DELAWARE","FL":"FLORIDA","GA":"GEORGIA",
    "HI":"HAWAII","ID":"IDAHO","IL":"ILLINOIS","IN":"INDIANA","IA":"IOWA","KS":"KANSAS",
    "KY":"KENTUCKY","LA":"LOUISIANA","ME":"MAINE","MD":"MARYLAND","MA":"MASSACHUSETTS",
    "MI":"MICHIGAN","MN":"MINNESOTA","MS":"MISSISSIPPI","MO":"MISSOURI","MT":"MONTANA",
    "NE":"NEBRASKA","NV":"NEVADA","NH":"NEW HAMPSHIRE","NJ":"NEW JERSEY","NM":"NEW MEXICO",
    "NY":"NEW YORK","NC":"NORTH CAROLINA","ND":"NORTH DAKOTA","OH":"OHIO","OK":"OKLAHOMA",
    "OR":"OREGON","PA":"PENNSYLVANIA","RI":"RHODE ISLAND","SC":"SOUTH CAROLINA",
    "SD":"SOUTH DAKOTA","TN":"TENNESSEE","TX":"TEXAS","UT":"UTAH","VT":"VERMONT",
    "VA":"VIRGINIA","WA":"WASHINGTON","WV":"WEST VIRGINIA","WI":"WISCONSIN","WY":"WYOMING",
    "DC":"DISTRICT OF COLUMBIA","PR":"PUERTO RICO","VI":"VIRGIN ISLANDS",
    "GU":"GUAM","MP":"NORTHERN MARIANA ISLANDS","AS":"AMERICAN SAMOA",
}

TROPICAL_TYPES = {"Hurricane", "Tropical Storm", "Typhoon"}

NOAA_COMPATIBLE = {
    "Hurricane": {
        "Hurricane (Typhoon)", "Tropical Storm", "Storm Surge/Tide", "Coastal Flood",
        "Flash Flood", "Flood", "Heavy Rain", "High Wind", "Strong Wind",
        "Thunderstorm Wind", "Tornado",
    },
    "Tropical Storm": {
        "Hurricane (Typhoon)", "Tropical Storm", "Storm Surge/Tide", "Coastal Flood",
        "Flash Flood", "Flood", "Heavy Rain", "High Wind", "Strong Wind",
        "Thunderstorm Wind", "Tornado",
    },
    "Typhoon": {
        "Hurricane (Typhoon)", "Tropical Storm", "Storm Surge/Tide", "Coastal Flood",
        "Flash Flood", "Flood", "Heavy Rain", "High Wind", "Strong Wind",
        "Thunderstorm Wind",
    },
    "Fire": {"Wildfire"},
    "Flood": {"Flood", "Flash Flood", "Heavy Rain", "Coastal Flood", "Debris Flow"},
    "Severe Storm": {
        "Thunderstorm Wind", "High Wind", "Strong Wind", "Hail", "Tornado",
        "Heavy Rain", "Flash Flood", "Flood", "Lightning",
    },
    "Tornado": {"Tornado"},
    "Snowstorm": {"Winter Storm", "Heavy Snow", "Blizzard", "Lake-Effect Snow"},
    "Severe Ice Storm": {"Ice Storm", "Freezing Rain", "Sleet", "Winter Weather"},
    "Coastal Storm": {"Coastal Flood", "High Surf", "Storm Surge/Tide", "High Wind"},
    "Freezing": {"Cold/Wind Chill", "Extreme Cold/Wind Chill", "Frost/Freeze"},
    "Drought": {"Drought", "Heat", "Excessive Heat"},
}

def get(url: str, **kwargs) -> requests.Response:
    for attempt in range(6):
        try:
            r = SESSION.get(url, timeout=120, **kwargs)
            r.raise_for_status()
            return r
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def parse_damage(v) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, (int, float, np.number)):
        return float(v)
    s = str(v).strip().upper().replace(",", "")
    if not s:
        return np.nan
    m = re.match(r"^([-+]?\d+(?:\.\d+)?)\s*([KMBT]?)$", s)
    if not m:
        try:
            return float(s)
        except Exception:
            return np.nan
    n = float(m.group(1))
    mult = {"":1.0, "K":1e3, "M":1e6, "B":1e9, "T":1e12}[m.group(2)]
    return n * mult


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2-lat1)
    dl = math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*r*math.asin(math.sqrt(a))



def load_fema_declaration_titles(master: pd.DataFrame) -> Dict[int, str]:
    """Fetch official FEMA declaration titles for tropical-cyclone rows only."""
    cache = CACHE / "fema_tropical_declaration_titles.json"
    if cache.exists():
        raw = json.loads(cache.read_text(encoding="utf-8"))
        return {int(k): str(v) for k, v in raw.items()}

    ids = sorted(
        master.loc[master["incidentType"].isin(TROPICAL_TYPES), "disasterNumber"]
        .dropna().astype(int).unique().tolist()
    )
    out: Dict[int, str] = {}
    for i, dn in enumerate(ids, 1):
        params = {"$filter": f"disasterNumber eq {dn}", "$top": 1}
        js = get(FEMA_DECL_URL, params=params).json()
        rows = js.get("DisasterDeclarationsSummaries", [])
        title = str(rows[0].get("declarationTitle", "")) if rows else ""
        out[int(dn)] = title
        if i % 25 == 0:
            print(f"FEMA declaration titles {i}/{len(ids)}")
    cache.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# -------------------------- NHC HURDAT2 --------------------------

def discover_hurdat_urls() -> List[str]:
    html = get(NHC_HURDAT_DIR).text
    atl = re.findall(r'href="([^"]*hurdat2-1851-2025[^"]*\.txt)"', html, flags=re.I)
    pac = re.findall(r'href="([^"]*hurdat2-nepac-1949-2025[^"]*\.txt)"', html, flags=re.I)
    # Fall back to latest available 2024 files if directory naming changes.
    if not atl:
        atl = re.findall(r'href="([^"]*hurdat2-1851-2024[^"]*\.txt)"', html, flags=re.I)
    if not pac:
        pac = re.findall(r'href="([^"]*hurdat2-nepac-1949-2024[^"]*\.txt)"', html, flags=re.I)
    urls = []
    for seq in (atl, pac):
        if seq:
            urls.append(NHC_HURDAT_DIR + sorted(seq)[-1])
    if not urls:
        raise RuntimeError("Could not discover HURDAT2 text files")
    return urls


def parse_coord(s: str) -> float:
    s = s.strip().upper()
    val = float(s[:-1])
    if s.endswith(("S","W")):
        val = -val
    return val


def parse_hurdat(text: str) -> Dict[str, dict]:
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    storms: Dict[str, dict] = {}
    i = 0
    while i < len(lines):
        header = [x.strip() for x in lines[i].split(",")]
        sid, name, n = header[0], header[1], int(header[2])
        pts = []
        for j in range(i+1, i+1+n):
            p = [x.strip() for x in lines[j].split(",")]
            date = pd.to_datetime(p[0] + p[1].zfill(4), format="%Y%m%d%H%M", errors="coerce", utc=True)
            try:
                lat, lon = parse_coord(p[4]), parse_coord(p[5])
                wind = float(p[6]) if p[6] not in {"", "-999"} else np.nan
                pressure = float(p[7]) if p[7] not in {"", "-999"} else np.nan
            except Exception:
                continue
            pts.append({
                "date": date, "status": p[3], "lat": lat, "lon": lon,
                "wind": wind, "pressure": pressure,
            })
        if pts:
            df = pd.DataFrame(pts)
            valid_ace = df["status"].isin(["TS","HU","SS"]) & df["wind"].notna() & (df["wind"] >= 34)
            ace = float((df.loc[valid_ace, "wind"] ** 2).sum() / 10000.0)
            storms[sid] = {
                "id": sid, "name": name.strip().upper(),
                "year": int(sid[-4:]), "track": df,
                "start": df["date"].min(), "end": df["date"].max(),
                "max_wind": float(df["wind"].max()) if df["wind"].notna().any() else np.nan,
                "min_pressure": float(df["pressure"].min()) if df["pressure"].notna().any() else np.nan,
                "ace": ace,
            }
        i += n + 1
    return storms


def load_hurdat() -> Dict[str, dict]:
    storms = {}
    for idx, url in enumerate(discover_hurdat_urls()):
        cache = CACHE / f"hurdat_{idx}.txt"
        if not cache.exists():
            cache.write_text(get(url).text, encoding="utf-8")
        storms.update(parse_hurdat(cache.read_text(encoding="utf-8")))
    return storms


def cyclone_features_for_row(row: pd.Series, storms: Dict[str, dict], declaration_title: str = "") -> Tuple[dict, dict]:
    if row["incidentType"] not in TROPICAL_TYPES or row["state"] not in STATE_CENTROIDS:
        return {}, {}
    begin = pd.to_datetime(row["incidentBeginDate"], errors="coerce", utc=True)
    end = pd.to_datetime(row["incidentEndDate"], errors="coerce", utc=True)
    if pd.isna(begin):
        return {}, {}
    if pd.isna(end):
        end = begin + pd.Timedelta(days=7)
    lat0, lon0 = STATE_CENTROIDS[row["state"]]

    candidates = []
    for s in storms.values():
        if s["year"] not in {begin.year-1, begin.year, end.year}:
            continue
        overlap_start = max(begin - pd.Timedelta(days=2), s["start"])
        overlap_end = min(end + pd.Timedelta(days=2), s["end"])
        if overlap_end < overlap_start:
            continue
        tr = s["track"].copy()
        d = np.array([haversine_km(lat0, lon0, la, lo) for la, lo in zip(tr["lat"], tr["lon"])])
        min_d = float(np.nanmin(d))
        # Require plausible jurisdiction proximity. For island territories use tighter radius.
        max_match_d = 900 if row["state"] in {"AK","TX","CA","FL"} else 700
        if min_d > max_match_d:
            continue
        overlap_hours = max(0.0, (overlap_end - overlap_start).total_seconds()/3600)
        temporal_gap_days = float((tr["date"] - begin).abs().dt.total_seconds().min() / 86400.0)
        title_upper = (declaration_title or "").upper()
        name_match = bool(s["name"] and re.search(rf"\b{re.escape(s['name'])}\b", title_upper))
        # Name identity dominates. If a title has no usable storm name, prefer
        # storms closest to the FEMA incident-begin date, then geography.
        score = (10000.0 if name_match else 0.0) + overlap_hours - min_d/20.0 - temporal_gap_days*100.0
        candidates.append((score, min_d, s, d, name_match, temporal_gap_days))
    if not candidates:
        return {}, {"nhc_match": "none"}

    # If the official FEMA title identifies one of the candidate storm names,
    # reject differently named candidates entirely.
    if declaration_title:
        named = [x for x in candidates if x[4]]
        if named:
            candidates = named
    candidates.sort(key=lambda x: x[0], reverse=True)
    score, min_d, s, d, name_match, temporal_gap_days = candidates[0]
    tr = s["track"].copy()
    closest_i = int(np.nanargmin(d))
    closest = tr.iloc[closest_i]

    feat = {
        "nhc_min_distance_km": min_d,
        "nhc_storm_max_wind_kt": s["max_wind"],
        "nhc_storm_min_pressure_mb": s["min_pressure"],
        "nhc_ace": s["ace"],
        "nhc_closest_wind_kt": float(closest["wind"]) if pd.notna(closest["wind"]) else np.nan,
        "nhc_closest_pressure_mb": float(closest["pressure"]) if pd.notna(closest["pressure"]) else np.nan,
    }
    for radius in (200, 300, 500):
        mask = d <= radius
        feat[f"nhc_track_points_{radius}km"] = int(mask.sum())
        feat[f"nhc_local_max_wind_{radius}km"] = (
            float(tr.loc[mask, "wind"].max()) if mask.any() and tr.loc[mask, "wind"].notna().any() else np.nan
        )
    wind = tr["wind"].fillna(0).to_numpy(float)
    feat["nhc_wind_distance_index"] = float(np.sum(np.maximum(wind-34.0, 0.0)**2 / (1.0 + d/100.0)))
    audit = {
        "nhc_match": "matched",
        "nhc_storm_id": s["id"],
        "nhc_storm_name": s["name"],
        "nhc_match_score": score,
        "nhc_name_match": bool(name_match),
        "nhc_temporal_gap_days": float(temporal_gap_days),
        "fema_declaration_title": declaration_title,
        "nhc_min_distance_km": min_d,
    }
    return feat, audit


# -------------------------- NOAA Storm Events --------------------------

def discover_noaa_file(year: int) -> str:
    html = get(NOAA_INDEX).text
    pat = rf'(StormEvents_details-ftp_v1\.0_d{year}_c\d+\.csv\.gz)'
    files = sorted(set(re.findall(pat, html)))
    if not files:
        raise RuntimeError(f"No NOAA Storm Events details file found for {year}")
    return NOAA_INDEX + files[-1]


def load_noaa_year(year: int) -> pd.DataFrame:
    cache = CACHE / f"noaa_{year}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    url = discover_noaa_file(year)
    raw = get(url).content
    df = pd.read_csv(io.BytesIO(gzip.decompress(raw)), low_memory=False)
    keep = [
        "EVENT_ID","EPISODE_ID","STATE","STATE_FIPS","YEAR","MONTH_NAME","EVENT_TYPE",
        "CZ_TYPE","CZ_FIPS","CZ_NAME","BEGIN_DATE_TIME","END_DATE_TIME",
        "INJURIES_DIRECT","INJURIES_INDIRECT","DEATHS_DIRECT","DEATHS_INDIRECT",
        "DAMAGE_PROPERTY","DAMAGE_CROPS","SOURCE","MAGNITUDE","MAGNITUDE_TYPE",
        "EPISODE_NARRATIVE","EVENT_NARRATIVE","BEGIN_LAT","BEGIN_LON","END_LAT","END_LON",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["begin_dt"] = pd.to_datetime(df["BEGIN_DATE_TIME"], errors="coerce", utc=True)
    df["end_dt"] = pd.to_datetime(df["END_DATE_TIME"], errors="coerce", utc=True)
    df["end_dt"] = df["end_dt"].fillna(df["begin_dt"])
    for c in ["DAMAGE_PROPERTY","DAMAGE_CROPS"]:
        if c in df:
            df[c] = df[c].map(parse_damage)
    for c in ["INJURIES_DIRECT","INJURIES_INDIRECT","DEATHS_DIRECT","DEATHS_INDIRECT","MAGNITUDE"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Parquet cache reduces repeat-run cost.
    df.to_parquet(cache, index=False)
    return df


def load_noaa(master: pd.DataFrame) -> pd.DataFrame:
    years = sorted(set(pd.to_datetime(master["incidentBeginDate"], errors="coerce").dt.year.dropna().astype(int)))
    frames = []
    for year in years:
        if 1950 <= year <= 2025:
            print(f"NOAA {year}...")
            frames.append(load_noaa_year(year))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def noaa_features_for_row(row: pd.Series, noaa: pd.DataFrame, storm_name: Optional[str]) -> Tuple[dict, dict]:
    incident_type = row["incidentType"]
    compat = NOAA_COMPATIBLE.get(incident_type)
    state_name = STATE_NAMES.get(row["state"])
    if not compat or not state_name or noaa.empty:
        return {}, {"noaa_match": "unsupported"}
    begin = pd.to_datetime(row["incidentBeginDate"], errors="coerce", utc=True)
    end = pd.to_datetime(row["incidentEndDate"], errors="coerce", utc=True)
    if pd.isna(begin):
        return {}, {"noaa_match": "no_date"}
    if pd.isna(end):
        end = begin + pd.Timedelta(days=7)

    c = noaa[
        (noaa["STATE"].astype(str).str.upper() == state_name)
        & (noaa["begin_dt"] <= end + pd.Timedelta(days=1))
        & (noaa["end_dt"] >= begin - pd.Timedelta(days=1))
        & (noaa["EVENT_TYPE"].isin(compat))
    ].copy()

    # For tropical systems, derived hazards (flood/tornado/etc.) should either
    # explicitly mention the matched storm name or be a core tropical event.
    if incident_type in TROPICAL_TYPES and storm_name and not c.empty:
        narrative = (
            c.get("EPISODE_NARRATIVE", pd.Series("", index=c.index)).fillna("").astype(str)
            + " "
            + c.get("EVENT_NARRATIVE", pd.Series("", index=c.index)).fillna("").astype(str)
        ).str.upper()
        named = narrative.str.contains(re.escape(storm_name.upper()), regex=True, na=False)
        core = c["EVENT_TYPE"].isin({"Hurricane (Typhoon)", "Tropical Storm", "Storm Surge/Tide"})
        # Prefer explicit storm-name evidence whenever the Storm Events
        # narratives provide it. Only fall back to generic core tropical rows
        # when no named rows exist for that FEMA state/window.
        c = c[named].copy() if named.any() else c[core].copy()

    if c.empty:
        return {}, {"noaa_match": "none", "noaa_event_count": 0}

    f = {
        "noaa_event_count": float(len(c)),
        "noaa_episode_count": float(c["EPISODE_ID"].nunique()) if "EPISODE_ID" in c else np.nan,
        "noaa_unique_event_types": float(c["EVENT_TYPE"].nunique()),
        "noaa_unique_counties": float(c["CZ_NAME"].nunique()) if "CZ_NAME" in c else np.nan,
        "noaa_property_damage_total": float(c["DAMAGE_PROPERTY"].fillna(0).sum()),
        "noaa_property_damage_max": float(c["DAMAGE_PROPERTY"].max()) if c["DAMAGE_PROPERTY"].notna().any() else np.nan,
        "noaa_crop_damage_total": float(c["DAMAGE_CROPS"].fillna(0).sum()),
        "noaa_deaths_direct": float(c["DEATHS_DIRECT"].fillna(0).sum()),
        "noaa_deaths_indirect": float(c["DEATHS_INDIRECT"].fillna(0).sum()),
        "noaa_injuries_direct": float(c["INJURIES_DIRECT"].fillna(0).sum()),
        "noaa_injuries_indirect": float(c["INJURIES_INDIRECT"].fillna(0).sum()),
    }
    f["noaa_property_damage_log1p"] = math.log1p(max(f["noaa_property_damage_total"], 0.0))
    f["noaa_crop_damage_log1p"] = math.log1p(max(f["noaa_crop_damage_total"], 0.0))
    return f, {
        "noaa_match": "matched",
        "noaa_event_count": int(len(c)),
        "noaa_event_types": "|".join(sorted(map(str, c["EVENT_TYPE"].dropna().unique()))),
        "noaa_event_ids": "|".join(map(str, c["EVENT_ID"].dropna().astype(int).head(50).tolist())),
    }


# -------------------------- CAL FIRE --------------------------

def arcgis_all(query_url: str, out_fields: str, where: str = "1=1") -> pd.DataFrame:
    rows = []
    offset = 0
    while True:
        params = {
            "where": where,
            "outFields": out_fields,
            "returnGeometry": "false",
            "f": "json",
            "resultOffset": offset,
            "resultRecordCount": 2000,
            "orderByFields": "OBJECTID ASC",
        }
        js = get(query_url, params=params).json()
        feats = js.get("features", [])
        rows.extend([x.get("attributes", {}) for x in feats])
        if len(feats) < 2000:
            break
        offset += 2000
        if offset > 1_000_000:
            raise RuntimeError("ArcGIS pagination safety limit")
    return pd.DataFrame(rows)


def load_calfire() -> Tuple[pd.DataFrame, pd.DataFrame]:
    per_cache = CACHE / "calfire_perimeters.parquet"
    dins_cache = CACHE / "calfire_dins.parquet"
    if per_cache.exists():
        per = pd.read_parquet(per_cache)
    else:
        per = arcgis_all(
            CALFIRE_PERIMETER_QUERY,
            "OBJECTID,YEAR_,STATE,AGENCY,UNIT_ID,FIRE_NAME,INC_NUM,IRWINID,"
            "ALARM_DATE,CONT_DATE,GIS_ACRES,COMPLEX_NAME,COMPLEX_ID",
            "YEAR_ >= 2010 AND YEAR_ <= 2024",
        )
        for c in ["ALARM_DATE","CONT_DATE"]:
            per[c] = pd.to_datetime(per[c], unit="ms", errors="coerce", utc=True)
        per.to_parquet(per_cache, index=False)

    if dins_cache.exists():
        dins = pd.read_parquet(dins_cache)
    else:
        dins = arcgis_all(
            CALFIRE_DINS_QUERY,
            "OBJECTID,DAMAGE,STRUCTURETYPE,STRUCTURECATEGORY,COUNTY,INCIDENTNAME,"
            "INCIDENTNUM,INCIDENTSTARTDATE,HAZARDTYPE",
            "INCIDENTSTARTDATE IS NOT NULL",
        )
        dins["INCIDENTSTARTDATE"] = pd.to_datetime(dins["INCIDENTSTARTDATE"], unit="ms", errors="coerce", utc=True)
        dins.to_parquet(dins_cache, index=False)
    return per, dins


def calfire_features_for_row(row: pd.Series, per: pd.DataFrame, dins: pd.DataFrame) -> Tuple[dict, dict]:
    if row["incidentType"] != "Fire" or row["state"] != "CA":
        return {}, {"calfire_match": "unsupported"}
    begin = pd.to_datetime(row["incidentBeginDate"], errors="coerce", utc=True)
    end = pd.to_datetime(row["incidentEndDate"], errors="coerce", utc=True)
    if pd.isna(begin):
        return {}, {"calfire_match": "no_date"}
    if pd.isna(end):
        end = begin + pd.Timedelta(days=30)

    p = per[
        (per["ALARM_DATE"] >= begin - pd.Timedelta(days=2))
        & (per["ALARM_DATE"] <= end + pd.Timedelta(days=2))
    ].copy()

    # Collapse obvious complex/incident duplicates conservatively.
    if not p.empty:
        p["canonical_fire_key"] = (
            p["COMPLEX_ID"].replace("", np.nan)
            .fillna(p["IRWINID"].replace("", np.nan))
            .fillna(p["INC_NUM"].replace("", np.nan))
            .fillna(p["FIRE_NAME"].fillna("UNKNOWN") + "|" + p["ALARM_DATE"].astype(str))
        )
        acreage = pd.to_numeric(p["GIS_ACRES"], errors="coerce")
        p["_acres"] = acreage
        canon = p.groupby("canonical_fire_key", dropna=False)["_acres"].max()
        acres_total = float(canon.sum(min_count=1)) if canon.notna().any() else np.nan
        acres_max = float(canon.max()) if canon.notna().any() else np.nan
        fire_count = int(canon.shape[0])
    else:
        acres_total = acres_max = np.nan
        fire_count = 0

    d = dins[
        (dins["INCIDENTSTARTDATE"] >= begin - pd.Timedelta(days=2))
        & (dins["INCIDENTSTARTDATE"] <= end + pd.Timedelta(days=2))
        & (dins["HAZARDTYPE"].fillna("Fire").astype(str).str.contains("Fire", case=False, na=False))
    ].copy()
    damage = d["DAMAGE"].fillna("").astype(str) if not d.empty else pd.Series(dtype=str)

    f = {
        "calfire_fire_count": float(fire_count) if fire_count or not p.empty else np.nan,
        "calfire_acres_total": acres_total,
        "calfire_acres_max": acres_max,
        "calfire_dins_structure_count": float(len(d)) if not d.empty else np.nan,
        "calfire_dins_incident_count": float(d["INCIDENTNAME"].nunique()) if not d.empty else np.nan,
        "calfire_destroyed_structures": float(damage.str.contains("Destroyed", case=False, na=False).sum()) if not d.empty else np.nan,
        "calfire_major_structures": float(damage.str.contains("Major", case=False, na=False).sum()) if not d.empty else np.nan,
        "calfire_minor_structures": float(damage.str.contains("Minor", case=False, na=False).sum()) if not d.empty else np.nan,
        "calfire_affected_structures": float(damage.str.contains("Affected", case=False, na=False).sum()) if not d.empty else np.nan,
    }
    if pd.notna(acres_total):
        f["calfire_acres_log1p"] = math.log1p(max(acres_total, 0.0))
    audit = {
        "calfire_match": "matched" if (not p.empty or not d.empty) else "none",
        "calfire_fire_names": "|".join(sorted(map(str, p["FIRE_NAME"].dropna().unique()))) if not p.empty else "",
        "calfire_dins_incidents": "|".join(sorted(map(str, d["INCIDENTNAME"].dropna().unique()))) if not d.empty else "",
    }
    return f, audit


# -------------------------- Enrichment --------------------------

def build_external(master: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("Loading official FEMA declaration titles...")
    declaration_titles = load_fema_declaration_titles(master)

    print("Loading HURDAT2...")
    storms = load_hurdat()
    print(f"HURDAT2 storms: {len(storms):,}")

    print("Loading NOAA Storm Events...")
    noaa = load_noaa(master)
    print(f"NOAA rows loaded: {len(noaa):,}")

    print("Loading CAL FIRE...")
    per, dins = load_calfire()
    print(f"CAL FIRE perimeters: {len(per):,}; DINS structures: {len(dins):,}")

    # Index NOAA once by state + year. This preserves the exact matching rules
    # but avoids scanning the complete 2010-2024 table for every FEMA row.
    noaa = noaa.copy()
    noaa["_event_year"] = noaa["begin_dt"].dt.year
    noaa_groups = {
        (str(state).upper(), int(year)): g.copy()
        for (state, year), g in noaa.dropna(subset=["_event_year"]).groupby(["STATE", "_event_year"])
    }

    features = []
    audits = []
    for i, row in master.iterrows():
        declaration_title = declaration_titles.get(int(row["disasterNumber"]), "")
        nhc_f, nhc_a = cyclone_features_for_row(row, storms, declaration_title)
        storm_name = nhc_a.get("nhc_storm_name")

        state_name = STATE_NAMES.get(row["state"])
        begin_dt = pd.to_datetime(row["incidentBeginDate"], errors="coerce", utc=True)
        end_dt = pd.to_datetime(row["incidentEndDate"], errors="coerce", utc=True)
        years = set()
        if pd.notna(begin_dt):
            years.add(int(begin_dt.year))
        if pd.notna(end_dt):
            years.add(int(end_dt.year))
        local_parts = [
            noaa_groups[(state_name, y)]
            for y in years
            if state_name is not None and (state_name, y) in noaa_groups
        ]
        local_noaa = pd.concat(local_parts, ignore_index=True) if local_parts else noaa.iloc[0:0]
        noaa_f, noaa_a = noaa_features_for_row(row, local_noaa, storm_name)
        fire_f, fire_a = calfire_features_for_row(row, per, dins)
        rec = {"disasterNumber": int(row["disasterNumber"])}
        rec.update(nhc_f); rec.update(noaa_f); rec.update(fire_f)
        features.append(rec)
        aud = {
            "disasterNumber": int(row["disasterNumber"]),
            "state": row["state"], "incidentType": row["incidentType"],
            "incidentBeginDate": str(row["incidentBeginDate"]),
            "incidentEndDate": str(row["incidentEndDate"]),
        }
        aud.update(nhc_a); aud.update(noaa_a); aud.update(fire_a)
        audits.append(aud)
        if (i+1) % 100 == 0:
            print(f"Enriched {i+1}/{len(master)}")
    return pd.DataFrame(features), pd.DataFrame(audits)


# -------------------------- Evaluation --------------------------

def funding_band(v: float) -> str:
    if v < 200_000_000: return "50-200M"
    if v < 500_000_000: return "200-500M"
    return "500M+"


def lfyo_multiclass(df: pd.DataFrame, features: List[str]) -> Dict:
    y = df["funding_band"].astype(str).to_numpy()
    pred = np.empty(len(df), dtype=object)
    years = sorted(df["fyDeclared"].astype(int).unique())
    for year in years:
        te = df["fyDeclared"].astype(int).to_numpy() == year
        tr = ~te
        Xtr = normalize_model_frame(df.loc[tr, features])
        Xte = normalize_model_frame(df.loc[te, features])
        model = RandomForestClassifier(
            n_estimators=500, random_state=1000+year,
            class_weight="balanced_subsample", max_features="sqrt",
            min_samples_leaf=1,
        )
        pipe = prep_pipeline(Xtr, model)
        pipe.fit(Xtr, y[tr])
        pred[te] = pipe.predict(Xte)

    labels = ["50-200M","200-500M","500M+"]
    counts = {}
    for lab in labels:
        m = y == lab
        counts[lab] = {
            "correct": int((pred[m] == y[m]).sum()),
            "total": int(m.sum()),
            "recall": float((pred[m] == y[m]).mean()) if m.any() else None,
        }
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "per_band": counts,
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "predictions": pred.tolist(),
    }


def lfyo_binary(df: pd.DataFrame, features: List[str], kind: str) -> Dict:
    y = (df["totalObligatedFunding"].to_numpy(float) >= 500_000_000).astype(int)
    pred = np.zeros(len(df), dtype=int)
    years = sorted(df["fyDeclared"].astype(int).unique())
    for year in years:
        te = df["fyDeclared"].astype(int).to_numpy() == year
        tr = ~te
        Xtr = normalize_model_frame(df.loc[tr, features])
        Xte = normalize_model_frame(df.loc[te, features])
        if kind == "logistic":
            model = LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5)
        else:
            model = RandomForestClassifier(
                n_estimators=500, random_state=2000+year,
                class_weight="balanced_subsample", max_features="sqrt",
            )
        pipe = prep_pipeline(Xtr, model)
        pipe.fit(Xtr, y[tr])
        pred[te] = pipe.predict(Xte)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "extreme_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0,1]).tolist(),
        "predictions": pred.tolist(),
    }


def lfyo_regression(df: pd.DataFrame, features: List[str]) -> Dict:
    y_dollar = df["totalObligatedFunding"].to_numpy(float)
    y = np.log1p(np.clip(y_dollar, 0, None))
    pred_log = np.zeros(len(df), dtype=float)
    years = sorted(df["fyDeclared"].astype(int).unique())
    for year in years:
        te = df["fyDeclared"].astype(int).to_numpy() == year
        tr = ~te
        Xtr = normalize_model_frame(df.loc[tr, features])
        Xte = normalize_model_frame(df.loc[te, features])
        model = RandomForestRegressor(
            n_estimators=600, random_state=3000+year,
            min_samples_leaf=1, max_features=0.7,
        )
        pipe = prep_pipeline(Xtr, model)
        pipe.fit(Xtr, y[tr])
        pred_log[te] = pipe.predict(Xte)
    pred_dollar = np.expm1(pred_log)
    return {
        "log_mae": float(mean_absolute_error(y, pred_log)),
        "log_r2": float(r2_score(y, pred_log)),
        "dollar_mae": float(mean_absolute_error(y_dollar, pred_dollar)),
        "dollar_r2": float(r2_score(y_dollar, pred_dollar)),
        "predictions": pred_dollar.tolist(),
    }


def main():
    master = normalize_master(pd.read_excel(MASTER_PATH))
    assert len(master) == 971 and master["disasterNumber"].nunique() == 971

    # Build semantics for ALL rows from non-financial MissionAssignment fields.
    print("Loading MissionAssignments for semantic layer...")
    ma = fetch_all_mission_assignments()
    sem, _latest = build_semantic_rollup(master, ma)

    # Build external features for ALL rows before target/band inspection.
    ext, audit = build_external(master)
    enriched = master.merge(sem, on="disasterNumber", how="left").merge(ext, on="disasterNumber", how="left")

    semantic_cols = [c for c in enriched.columns if c.startswith("sem_") or c.startswith("ma_")]
    external_cols = [
        c for c in enriched.columns
        if c.startswith("nhc_") or c.startswith("noaa_") or c.startswith("calfire_")
    ]

    # Only now inspect the target and evaluation subset.
    high = enriched[
        (enriched["incidentType"] != "Biological")
        & (enriched["totalObligatedFunding"] >= 50_000_000)
    ].copy()
    high["funding_band"] = high["totalObligatedFunding"].map(funding_band)
    assert len(high) == 23, len(high)

    current19 = [c for c in CURRENT_19 if c in high.columns]
    semantic_use = [
        c for c in semantic_cols
        if high[c].notna().sum() >= 2 and high[c].nunique(dropna=True) > 1
    ]
    external_use = [
        c for c in external_cols
        if high[c].notna().sum() >= 2 and high[c].nunique(dropna=True) > 1
    ]

    sets = {
        "A_current19": current19,
        "B_current19_semantics": current19 + semantic_use,
        "C_current19_external": current19 + external_use,
        "D_current19_semantics_external": current19 + semantic_use + external_use,
    }

    results = {}
    case_predictions = high[
        ["disasterNumber","state","incidentType","fyDeclared","totalObligatedFunding","funding_band"]
    ].reset_index(drop=True).copy()

    for name, feats in sets.items():
        print(f"Evaluating {name}: {len(feats)} features")
        multi = lfyo_multiclass(high.reset_index(drop=True), feats)
        binary_log = lfyo_binary(high.reset_index(drop=True), feats, "logistic")
        binary_rf = lfyo_binary(high.reset_index(drop=True), feats, "rf")
        reg = lfyo_regression(high.reset_index(drop=True), feats)
        results[name] = {
            "feature_count": len(feats),
            "multiclass_rf": multi,
            "binary_logistic": binary_log,
            "binary_rf": binary_rf,
            "regression_rf": reg,
        }
        case_predictions[f"{name}_band_pred"] = multi["predictions"]
        case_predictions[f"{name}_extreme_log_pred"] = binary_log["predictions"]
        case_predictions[f"{name}_reg_pred"] = reg["predictions"]

    coverage = {
        "all_971": {
            "nhc_rows": int(enriched[[c for c in external_cols if c.startswith("nhc_")]].notna().any(axis=1).sum()) if any(c.startswith("nhc_") for c in external_cols) else 0,
            "noaa_rows": int(enriched[[c for c in external_cols if c.startswith("noaa_")]].notna().any(axis=1).sum()) if any(c.startswith("noaa_") for c in external_cols) else 0,
            "calfire_rows": int(enriched[[c for c in external_cols if c.startswith("calfire_")]].notna().any(axis=1).sum()) if any(c.startswith("calfire_") for c in external_cols) else 0,
        },
        "high_23": {
            "nhc_rows": int(high[[c for c in external_cols if c.startswith("nhc_")]].notna().any(axis=1).sum()) if any(c.startswith("nhc_") for c in external_cols) else 0,
            "noaa_rows": int(high[[c for c in external_cols if c.startswith("noaa_")]].notna().any(axis=1).sum()) if any(c.startswith("noaa_") for c in external_cols) else 0,
            "calfire_rows": int(high[[c for c in external_cols if c.startswith("calfire_")]].notna().any(axis=1).sum()) if any(c.startswith("calfire_") for c in external_cols) else 0,
        },
    }

    audit.to_csv(OUT / "external_match_audit.csv", index=False)
    ext.to_csv(OUT / "external_features_971.csv", index=False)
    high[
        ["disasterNumber","state","incidentType","fyDeclared","totalObligatedFunding","funding_band"]
        + current19 + semantic_use + external_use
    ].to_csv(OUT / "high23_enriched.csv", index=False)
    case_predictions.to_csv(OUT / "high23_ablation_predictions.csv", index=False)

    summary = {
        "experiment_type": "retrospective target-blind external-severity diagnostic",
        "master_rows_enriched_before_target_inspection": len(master),
        "high_nonbiological_rows": len(high),
        "funding_band_counts": high["funding_band"].value_counts().to_dict(),
        "semantic_features_used": len(semantic_use),
        "external_features_used": external_use,
        "coverage": coverage,
        "results": results,
        "sources": {
            "nhc": "NHC HURDAT2 current best-track archive",
            "noaa": "NCEI Storm Events current bulk details files",
            "calfire_perimeters": "CAL FIRE FRAP California Historic Fire Perimeters",
            "calfire_dins": "CAL FIRE Damage Inspection Program POSTFIRE dataset",
        },
        "leakage_controls": [
            "Funding and funding band are never used in external matching.",
            "All 971 disasters are enriched before the high-band subset is selected.",
            "MissionAssignment financial fields are excluded from semantic predictors.",
            "No explicit source-availability or match-confidence fields enter models.",
        ],
        "timing_warning": (
            "This run uses final/best-track and final damage products. It tests whether "
            "physical severity contains missing information; it is not a declaration-time model."
        ),
    }
    (OUT / "ablation_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    md = [
        "# External severity ablation",
        "",
        "**Retrospective diagnostic — not yet t0/declaration-time safe.**",
        "",
        f"- Enriched master rows before target inspection: **{len(master)}**",
        f"- High non-Biological evaluation rows: **{len(high)}**",
        f"- Bands: **{summary['funding_band_counts']}**",
        f"- External features used: **{len(external_use)}**",
        f"- Coverage all 971: **{coverage['all_971']}**",
        f"- Coverage high 23: **{coverage['high_23']}**",
        "",
        "## Strict leave-fiscal-year-out results",
        "",
        "| Feature set | 50-200M | 200-500M | 500M+ | Balanced acc. | Extreme recall (logistic) | Log MAE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in sets:
        rr = results[name]
        pb = rr["multiclass_rf"]["per_band"]
        md.append(
            f"| {name} | {pb['50-200M']['correct']}/{pb['50-200M']['total']} "
            f"({pb['50-200M']['recall']:.1%}) | "
            f"{pb['200-500M']['correct']}/{pb['200-500M']['total']} "
            f"({pb['200-500M']['recall']:.1%}) | "
            f"{pb['500M+']['correct']}/{pb['500M+']['total']} "
            f"({pb['500M+']['recall']:.1%}) | "
            f"{rr['multiclass_rf']['balanced_accuracy']:.3f} | "
            f"{rr['binary_logistic']['extreme_recall']:.1%} | "
            f"{rr['regression_rf']['log_mae']:.3f} |"
        )
    md += [
        "",
        "## Timing warning",
        "",
        summary["timing_warning"],
    ]
    (OUT / "ablation_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
