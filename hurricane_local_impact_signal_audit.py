from __future__ import annotations

import io
import json
import math
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import roc_auc_score, roc_curve

import external_physical_severity_high_router_experiment as base

OUT = Path("hurricane_local_impact_signal_audit_results")
OUT.mkdir(exist_ok=True)

HURDAT_ATL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2024-040425.txt"
CENSUS_GAZ = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/2024_Gaz_state_national.zip"
FEMA_URL = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"

RADII = [100, 200, 300, 500]


def num(x):
    return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp/2)**2 + np.cos(p1) * np.cos(p2) * np.sin(dl/2)**2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def coord(v):
    try:
        sign = -1 if v[-1] in "SW" else 1
        return sign * float(v[:-1])
    except Exception:
        return np.nan


def fetch_titles(dns):
    rows = []
    for i in range(0, len(dns), 40):
        chunk = dns[i:i+40]
        filt = " or ".join([f"disasterNumber eq {int(x)}" for x in chunk])
        params = {
            "$filter": filt,
            "$select": "disasterNumber,state,declarationTitle,declarationDate,incidentBeginDate,incidentEndDate",
            "$top": "1000",
            "$metadata": "off",
        }
        r = requests.get(FEMA_URL, params=params, timeout=180,
                         headers={"User-Agent":"ML-thesis-research/1.0"})
        r.raise_for_status()
        p = r.json()
        recs = p.get("DisasterDeclarationsSummaries", p.get("items", [])) if isinstance(p, dict) else p
        rows.extend(recs)
    d = pd.DataFrame(rows)
    d["disasterNumber"] = num(d["disasterNumber"]).astype("Int64")
    d["declarationDate"] = pd.to_datetime(d["declarationDate"], errors="coerce", utc=True).dt.tz_convert(None)
    d = d.sort_values(["disasterNumber","declarationDate"]).drop_duplicates("disasterNumber", keep="last")
    d.to_csv(OUT/"fema_hurricane_titles_snapshot.csv", index=False)
    return d


def fetch_state_centroids():
    r = requests.get(CENSUS_GAZ, timeout=180, headers={"User-Agent":"ML-thesis-research/1.0"})
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".txt")][0]
        raw = z.read(name)
    d = pd.read_csv(io.BytesIO(raw), sep="\t")
    d.columns = [str(c).strip() for c in d.columns]
    # Census state Gazetteer includes USPS, INTPTLAT, INTPTLONG.
    d["USPS"] = d["USPS"].astype(str).str.upper().str.strip()
    d["INTPTLAT"] = num(d["INTPTLAT"])
    d["INTPTLONG"] = num(d["INTPTLONG"])
    out = d[["USPS","NAME","INTPTLAT","INTPTLONG"]].rename(
        columns={"USPS":"state","INTPTLAT":"centroid_lat","INTPTLONG":"centroid_lon"}
    )
    out.to_csv(OUT/"census_state_centroids_snapshot.csv", index=False)
    return out


def parse_hurdat_points():
    r = requests.get(HURDAT_ATL, timeout=180, headers={"User-Agent":"ML-thesis-research/1.0"})
    r.raise_for_status()
    lines = [x.strip() for x in r.text.splitlines() if x.strip()]
    storms = []
    pts = []
    i = 0
    while i < len(lines):
        head = [x.strip() for x in lines[i].split(",")]
        if len(head) < 3 or not re.match(r"^[A-Z]{2}\d{6}$", head[0]):
            i += 1
            continue
        sid = head[0]
        name = head[1].upper().strip()
        n = int(head[2])
        storm_pts = []
        for j in range(i+1, min(i+1+n, len(lines))):
            p = [x.strip() for x in lines[j].split(",")]
            if len(p) < 8:
                continue
            dt = pd.to_datetime(p[0]+p[1], format="%Y%m%d%H%M", errors="coerce")
            wind = pd.to_numeric(pd.Series([p[6]]), errors="coerce").iloc[0]
            pressure = pd.to_numeric(pd.Series([p[7]]), errors="coerce").iloc[0]
            rec = {
                "storm_id":sid, "storm_name":name, "dt":dt, "status":p[3],
                "lat":coord(p[4]), "lon":coord(p[5]),
                "wind_kt":float(wind) if pd.notna(wind) else np.nan,
                "pressure_mb":float(pressure) if pd.notna(pressure) and float(pressure)>0 else np.nan,
            }
            pts.append(rec)
            storm_pts.append(rec)
        if storm_pts:
            z = pd.DataFrame(storm_pts)
            storms.append({
                "storm_id":sid, "storm_name":name, "year":int(sid[-4:]),
                "start":z.dt.min(), "end":z.dt.max(),
                "global_max_wind_kt":float(num(z.wind_kt).max()),
                "global_min_pressure_mb":float(num(z.pressure_mb).min()),
            })
        i += n + 1
    sdf = pd.DataFrame(storms)
    pdf = pd.DataFrame(pts)
    sdf.to_csv(OUT/"hurdat_atlantic_storms_snapshot.csv", index=False)
    pdf.to_csv(OUT/"hurdat_atlantic_points_snapshot.csv", index=False)
    return sdf, pdf


def exact_named_match(row, storms):
    title = str(row.get("declarationTitle","")).upper()
    dt = pd.to_datetime(row.get("effectiveDeclarationDate"), errors="coerce")
    if pd.isna(dt):
        return None
    yr = int(dt.year)
    cand = storms[(storms.year >= yr-1) & (storms.year <= yr+1)].copy()
    if cand.empty:
        return None
    exact = cand[cand.storm_name.map(lambda s: str(s).upper().strip() in title)]
    if exact.empty:
        return None
    exact = exact.assign(dd=(pd.to_datetime(exact.end)-dt).abs().dt.days)
    return exact.sort_values(["dd","global_max_wind_kt"], ascending=[True,False]).iloc[0]


def local_features(storm_id, points, clat, clon, pop):
    z = points[points.storm_id == storm_id].copy()
    z = z[z.lat.notna() & z.lon.notna() & z.wind_kt.notna()].copy()
    if z.empty or not np.isfinite(clat) or not np.isfinite(clon):
        return None
    z["distance_km"] = haversine_km(z.lat.to_numpy(float), z.lon.to_numpy(float), clat, clon)
    closest = z.loc[z.distance_km.idxmin()]
    f = {
        "min_track_distance_km":float(z.distance_km.min()),
        "wind_at_closest_kt":float(closest.wind_kt),
        "pressure_at_closest_mb":float(closest.pressure_mb) if pd.notna(closest.pressure_mb) else np.nan,
    }
    for rad in RADII:
        q = z[z.distance_km <= rad]
        f[f"hours_within_{rad}km"] = float(len(q)*6)
        f[f"max_wind_within_{rad}km"] = float(num(q.wind_kt).max()) if len(q) else 0.0
        f[f"min_pressure_within_{rad}km"] = float(num(q.pressure_mb).min()) if len(q) and num(q.pressure_mb).notna().any() else np.nan
    # Smooth local-impact metrics use every track point, downweighted by distance.
    w = np.exp(-z.distance_km.to_numpy(float)/200.0)
    wind = z.wind_kt.to_numpy(float)
    f["distance_weighted_wind"] = float(np.sum(w*wind)/max(np.sum(w),1e-12))
    f["distance_weighted_wind2"] = float(np.sum(w*(wind**2))/max(np.sum(w),1e-12))
    f["local_impact_integral"] = float(np.sum(w*(wind**2))*6/10000.0)
    f["local_peak_proxy"] = float(np.max(w*wind))
    f["local_major_hours_300km"] = float(((z.distance_km<=300)&(z.wind_kt>=96)).sum()*6)
    f["local_hurricane_hours_300km"] = float(((z.distance_km<=300)&(z.wind_kt>=64)).sum()*6)
    pop_m = float(pop)/1e6 if np.isfinite(pop) and pop>0 else np.nan
    f["population_millions"] = pop_m
    f["population_x_local_impact"] = float(pop_m*f["local_impact_integral"]) if np.isfinite(pop_m) else np.nan
    f["population_x_local_peak2"] = float(pop_m*(f["local_peak_proxy"]**2)/10000.0) if np.isfinite(pop_m) else np.nan
    return f


def threshold_screen(y, x):
    y=np.asarray(y,int); x=np.asarray(x,float)
    ok=np.isfinite(x)
    y=y[ok]; x=x[ok]
    if len(np.unique(y))<2 or len(np.unique(x))<2:
        return None
    best=None
    for sign in [1,-1]:
        s=sign*x
        vals=np.unique(s)
        thrs=np.r_[vals.min()-1e-12,(vals[:-1]+vals[1:])/2,vals.max()+1e-12]
        for t in thrs:
            pred=(s>=t).astype(int)
            r0=((pred[y==0]==0).sum()/max((y==0).sum(),1))
            r1=((pred[y==1]==1).sum()/max((y==1).sum(),1))
            key=(min(r0,r1),(r0+r1)/2)
            if best is None or key>best[0]:
                best=(key,sign,t,r0,r1)
    return {
        "orientation":"higher_extreme" if best[1]==1 else "lower_extreme",
        "threshold_oriented":float(best[2]),
        "lower_recall":float(best[3]),
        "upper_recall":float(best[4]),
        "min_recall":float(min(best[3],best[4])),
        "pass80":bool(best[3]>=.8 and best[4]>=.8),
    }


def main():
    d = base.load_master()
    hh = d[
        d.incidentType.astype(str).str.upper().isin(["HURRICANE","TROPICAL STORM"])
        & d.band.isin([3,4,5])
    ].copy()
    titles = fetch_titles(hh.disasterNumber.astype(int).tolist())
    cents = fetch_state_centroids()
    storms, points = parse_hurdat_points()

    hh = hh.merge(titles[["disasterNumber","declarationTitle","declarationDate"]],
                  on="disasterNumber", how="left")
    hh = hh.merge(cents[["state","centroid_lat","centroid_lon"]], on="state", how="left")

    rows=[]
    for _,r in hh.iterrows():
        q = {
            "disasterNumber":int(r.disasterNumber),"state":str(r.state),
            "fyDeclared":int(r.fyDeclared),"incidentType":str(r.incidentType),
            "target":float(r.target),"band":int(r.band),
            "declarationTitle":str(r.get("declarationTitle","")),
        }
        s = exact_named_match(r,storms)
        if s is None:
            q.update({"matched":0,"storm_id":"","storm_name":""})
        else:
            f = local_features(
                str(s.storm_id), points,
                float(r.centroid_lat) if pd.notna(r.centroid_lat) else np.nan,
                float(r.centroid_lon) if pd.notna(r.centroid_lon) else np.nan,
                float(base.POP.get(str(r.state),np.nan)),
            )
            q.update({"matched":1,"storm_id":str(s.storm_id),"storm_name":str(s.storm_name),
                      "global_max_wind_kt":float(s.global_max_wind_kt),
                      "global_min_pressure_mb":float(s.global_min_pressure_mb)})
            if f: q.update(f)
        rows.append(q)

    out=pd.DataFrame(rows)
    out.to_csv(OUT/"high_hurricane_local_impact_features.csv",index=False)

    pair=out[out.band.isin([4,5]) & out.matched.eq(1)].copy()
    y=(pair.band==5).astype(int).to_numpy()

    candidates = [
        "min_track_distance_km","wind_at_closest_kt","pressure_at_closest_mb",
        *[f"hours_within_{r}km" for r in RADII],
        *[f"max_wind_within_{r}km" for r in RADII],
        *[f"min_pressure_within_{r}km" for r in RADII],
        "distance_weighted_wind","distance_weighted_wind2","local_impact_integral",
        "local_peak_proxy","local_major_hours_300km","local_hurricane_hours_300km",
        "population_x_local_impact","population_x_local_peak2",
        "global_max_wind_kt","global_min_pressure_mb",
    ]

    audits=[]
    for c in candidates:
        if c not in pair: continue
        x=num(pair[c])
        ok=x.notna()
        if ok.sum()>=4 and len(np.unique(y[ok]))==2 and x[ok].nunique()>1:
            auc=float(roc_auc_score(y[ok],x[ok]))
            op=threshold_screen(y[ok],x[ok])
            audits.append({
                "feature":c,"n":int(ok.sum()),"auc":auc,
                "orientation_free_auc":float(max(auc,1-auc)),**(op or {})
            })
    audits=sorted(audits,key=lambda z:(z.get("pass80",False),z.get("min_recall",0),z["orientation_free_auc"]),reverse=True)
    pd.DataFrame(audits).to_csv(OUT/"top_boundary_local_impact_signal_screen.csv",index=False)

    focus=out[out.disasterNumber.isin([4339,4671,4086,4399,4673])].sort_values("disasterNumber")
    focus.to_csv(OUT/"focus_hurricane_local_impact.csv",index=False)

    summary={
        "purpose":"Test whether physical storm severity near the affected state/territory, rather than global storm maximum intensity, separates hurricane funding regimes.",
        "high_hurricane_count":int(len(out)),
        "matched_named_atlantic_storms":int(out.matched.sum()),
        "top_boundary_pair_count":int(len(pair)),
        "best_local_impact_features":audits[:15],
        "focus_cases":focus.to_dict("records"),
        "guardrails":[
            "Only named Atlantic hurricanes/tropical storms explicitly present in FEMA declaration titles are matched; no nearest-storm fallback is used.",
            "State representative coordinates come from the U.S. Census Gazetteer and contain no funding information.",
            "HURDAT2 track, wind, and pressure data contain no FEMA funding information.",
            "This is a development signal audit only. Any useful feature must next be tested under leave-fiscal-year-out and fully nested selection before a thesis claim."
        ]
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str),flush=True)


if __name__=="__main__":
    main()
