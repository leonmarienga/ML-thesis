#!/usr/bin/env python3
"""
Target-blind 2017 EAGLE-I outage diagnostic.

Purpose
-------
Test whether directly observed electricity outage scale provides the missing
infrastructure-severity signal for FEMA 4671 (Hurricane Fiona) without
hand-entering Fiona-specific facts.

All FEMA disasters beginning in calendar 2017 are enriched BEFORE funding is
inspected. Matching uses only state + incident time window.

Official source
---------------
ORNL EAGLE-I recorded electricity outages, Figshare article 24237376.
2017 CSV file id: 42547828
Modeled county customer counts (MCC.csv) file id: 42547708

The outage file is ~1.2 GB, so it is streamed in pandas chunks.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "eaglei_2017"
CACHE = ROOT / ".cache" / "eaglei_2017"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

EAGLE_URL = "https://ndownloader.figshare.com/files/42547828"
MCC_URL = "https://ndownloader.figshare.com/files/42547708"

EAGLE_PATH = CACHE / "eaglei_outages_2017.csv"
MCC_PATH = CACHE / "MCC.csv"

TROPICAL = {"Hurricane", "Tropical Storm", "Typhoon"}

STATE_FIPS = {
    "AL":"01","AK":"02","AZ":"04","AR":"05","CA":"06","CO":"08","CT":"09","DE":"10",
    "DC":"11","FL":"12","GA":"13","HI":"15","ID":"16","IL":"17","IN":"18","IA":"19",
    "KS":"20","KY":"21","LA":"22","ME":"23","MD":"24","MA":"25","MI":"26","MN":"27",
    "MS":"28","MO":"29","MT":"30","NE":"31","NV":"32","NH":"33","NJ":"34","NM":"35",
    "NY":"36","NC":"37","ND":"38","OH":"39","OK":"40","OR":"41","PA":"42","RI":"44",
    "SC":"45","SD":"46","TN":"47","TX":"48","UT":"49","VT":"50","VA":"51","WA":"53",
    "WV":"54","WI":"55","WY":"56","AS":"60","GU":"66","MP":"69","PR":"72","VI":"78",
}

def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        print(f"Using cached {path.name}: {path.stat().st_size/1e6:.1f} MB")
        return
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        t0 = time.time()
        with open(tmp, "wb") as f:
            for block in r.iter_content(chunk_size=8 * 1024 * 1024):
                if not block:
                    continue
                f.write(block)
                done += len(block)
                if total:
                    pct = 100.0 * done / total
                    print(f"\r  {done/1e6:,.0f}/{total/1e6:,.0f} MB ({pct:5.1f}%)", end="", flush=True)
                elif done % (128*1024*1024) < 8*1024*1024:
                    print(f"\r  {done/1e6:,.0f} MB", end="", flush=True)
        print(f"\nDownloaded in {time.time()-t0:.1f}s")
    tmp.replace(path)

def normalize_state(s: object) -> str:
    x = str(s).strip().upper()
    aliases = {
        "PUERTO RICO":"PR","PR":"PR",
        "FLORIDA":"FL","GEORGIA":"GA","TEXAS":"TX","LOUISIANA":"LA",
        "NORTH CAROLINA":"NC","SOUTH CAROLINA":"SC","CALIFORNIA":"CA",
        "NEW MEXICO":"NM","NEW YORK":"NY","HAWAII":"HI",
        "VIRGIN ISLANDS":"VI","U.S. VIRGIN ISLANDS":"VI",
    }
    if x in aliases:
        return aliases[x]
    if len(x) == 2:
        return x
    # reverse from known names where possible
    names = {
      "ALABAMA":"AL","ALASKA":"AK","ARIZONA":"AZ","ARKANSAS":"AR","COLORADO":"CO",
      "CONNECTICUT":"CT","DELAWARE":"DE","DISTRICT OF COLUMBIA":"DC","IDAHO":"ID",
      "ILLINOIS":"IL","INDIANA":"IN","IOWA":"IA","KANSAS":"KS","KENTUCKY":"KY",
      "MAINE":"ME","MARYLAND":"MD","MASSACHUSETTS":"MA","MICHIGAN":"MI",
      "MINNESOTA":"MN","MISSISSIPPI":"MS","MISSOURI":"MO","MONTANA":"MT",
      "NEBRASKA":"NE","NEVADA":"NV","NEW HAMPSHIRE":"NH","NEW JERSEY":"NJ",
      "NORTH DAKOTA":"ND","OHIO":"OH","OKLAHOMA":"OK","OREGON":"OR",
      "PENNSYLVANIA":"PA","RHODE ISLAND":"RI","SOUTH DAKOTA":"SD","TENNESSEE":"TN",
      "UTAH":"UT","VERMONT":"VT","VIRGINIA":"VA","WASHINGTON":"WA",
      "WEST VIRGINIA":"WV","WISCONSIN":"WI","WYOMING":"WY","AMERICAN SAMOA":"AS",
      "GUAM":"GU","NORTHERN MARIANA ISLANDS":"MP"
    }
    return names.get(x, x)

def load_customer_denominators() -> tuple[pd.DataFrame, Dict[str, float]]:
    download(MCC_URL, MCC_PATH)
    mcc = pd.read_csv(MCC_PATH, encoding="utf-8-sig")
    # expected columns County_FIPS, Customers
    fcol = next(c for c in mcc.columns if "fips" in c.lower())
    ccol = next(c for c in mcc.columns if "customer" in c.lower())
    mcc[fcol] = (
        pd.to_numeric(mcc[fcol], errors="coerce")
        .astype("Int64")
        .astype(str)
        .str.replace("<NA>", "", regex=False)
        .str.zfill(5)
    )
    mcc[ccol] = pd.to_numeric(mcc[ccol], errors="coerce")
    prefix_to_state = {v:k for k,v in STATE_FIPS.items()}
    mcc["state"] = mcc[fcol].str[:2].map(prefix_to_state)
    state_totals = mcc.groupby("state")[ccol].sum(min_count=1).to_dict()
    return mcc, {str(k): float(v) for k,v in state_totals.items() if pd.notna(v)}

def build_candidate_windows(master: pd.DataFrame) -> pd.DataFrame:
    x = master.copy()
    x["disasterNumber"] = pd.to_numeric(x["disasterNumber"], errors="coerce").astype("Int64")
    x["incidentBegin_dt"] = pd.to_datetime(x["incidentBeginDate"], errors="coerce", utc=True)
    x["incidentEnd_dt"] = pd.to_datetime(x["incidentEndDate"], errors="coerce", utc=True)
    x["declaration_dt"] = pd.to_datetime(x["declarationDate"], errors="coerce", utc=True)
    x["state"] = x["state"].astype(str).str.upper().str.strip()

    x = x[x["incidentBegin_dt"].dt.year == 2017].copy()

    # Target-blind windows. FEMA incidentEnd can span months; outage attribution
    # must not inherit an arbitrarily long declaration window.
    caps = np.where(x["incidentType"].isin(TROPICAL), 10, 14)
    cap_end = x["incidentBegin_dt"] + pd.to_timedelta(caps, unit="D")
    natural_end = x["incidentEnd_dt"].fillna(cap_end)
    x["outage_window_start"] = x["incidentBegin_dt"] - pd.Timedelta(hours=12)
    x["outage_window_end"] = pd.concat([natural_end, cap_end], axis=1).min(axis=1) + pd.Timedelta(hours=24)

    # Keep metadata only; funding is intentionally not needed in the scanning step.
    cols = [
        "disasterNumber","state","incidentType","incidentBegin_dt","incidentEnd_dt",
        "declaration_dt","outage_window_start","outage_window_end",
    ]
    return x[cols].sort_values(["state","outage_window_start"]).reset_index(drop=True)

def find_eagle_columns(path: Path) -> Dict[str, str]:
    head = pd.read_csv(path, nrows=5)
    cols = {c.lower().strip():c for c in head.columns}
    def pick(names):
        for n in names:
            if n in cols:
                return cols[n]
        for lc, orig in cols.items():
            if any(n in lc for n in names):
                return orig
        raise KeyError(f"Could not find one of {names}; columns={list(head.columns)}")
    return {
        "fips": pick(["fips_code","fips"]),
        "county": pick(["county"]),
        "state": pick(["state"]),
        "out": pick(["customers_out","sum"]),
        "time": pick(["run_start_time","timestamp","time"]),
    }

def scan_eaglei(candidates: pd.DataFrame, state_customers: Dict[str,float]) -> tuple[pd.DataFrame,pd.DataFrame]:
    download(EAGLE_URL, EAGLE_PATH)
    cm = find_eagle_columns(EAGLE_PATH)
    print("EAGLE-I columns:", cm)

    # Build state -> candidate list for fast chunk matching.
    by_state: Dict[str, List[dict]] = {}
    for _, r in candidates.iterrows():
        by_state.setdefault(r["state"], []).append(r.to_dict())

    # We aggregate state-level outage total at each timestamp for each candidate.
    # Store sparse timestamp totals and county sets.
    timelines: Dict[int, Dict[pd.Timestamp, float]] = {
        int(dn):{} for dn in candidates["disasterNumber"]
    }
    counties: Dict[int, set] = {int(dn):set() for dn in candidates["disasterNumber"]}
    matched_rows: Dict[int, int] = {int(dn):0 for dn in candidates["disasterNumber"]}

    usecols = [cm["fips"],cm["county"],cm["state"],cm["out"],cm["time"]]
    t0 = time.time()
    total_rows = 0
    chunks = 0
    for chunk in pd.read_csv(EAGLE_PATH, usecols=usecols, chunksize=500_000, low_memory=False):
        chunks += 1
        total_rows += len(chunk)
        chunk["_state"] = chunk[cm["state"]].map(normalize_state)
        chunk["_time"] = pd.to_datetime(chunk[cm["time"]], errors="coerce", utc=True)
        chunk["_out"] = pd.to_numeric(chunk[cm["out"]], errors="coerce").fillna(0.0)
        chunk["_fips"] = (
            pd.to_numeric(chunk[cm["fips"]], errors="coerce").astype("Int64").astype(str)
            .str.replace("<NA>","",regex=False).str.zfill(5)
        )

        present_states = set(chunk["_state"].dropna().unique()) & set(by_state)
        for st in present_states:
            sub = chunk[chunk["_state"] == st]
            if sub.empty:
                continue
            for cand in by_state[st]:
                dn = int(cand["disasterNumber"])
                m = (
                    (sub["_time"] >= cand["outage_window_start"])
                    & (sub["_time"] <= cand["outage_window_end"])
                )
                q = sub.loc[m]
                if q.empty:
                    continue
                matched_rows[dn] += len(q)
                # State outage total at each 15-min timestamp.
                ts = q.groupby("_time")["_out"].sum()
                td = timelines[dn]
                for t, v in ts.items():
                    # Chunk boundaries should not split identical timestamps often,
                    # but summing makes the aggregation correct if they do.
                    td[t] = td.get(t, 0.0) + float(v)
                counties[dn].update(q.loc[q["_out"] > 0, "_fips"].dropna().tolist())

        if chunks % 10 == 0:
            print(f"Scanned {total_rows:,} rows ({time.time()-t0:.1f}s)")

    feature_rows = []
    timeline_rows = []
    for _, cand in candidates.iterrows():
        dn = int(cand["disasterNumber"])
        td = timelines[dn]
        denom = state_customers.get(cand["state"], np.nan)
        if not td:
            feature_rows.append({
                "disasterNumber":dn,
                "eaglei_match":False,
                "eaglei_rows":matched_rows[dn],
                "eaglei_state_customers":denom,
            })
            continue

        ser = pd.Series(td).sort_index()
        # Reindex to complete 15-minute grid: omitted rows represent zero outages.
        idx = pd.date_range(
            cand["outage_window_start"].floor("15min"),
            cand["outage_window_end"].ceil("15min"),
            freq="15min", tz="UTC"
        )
        ser = ser.reindex(idx, fill_value=0.0)
        frac = ser / denom if pd.notna(denom) and denom > 0 else pd.Series(np.nan,index=ser.index)

        peak = float(ser.max())
        peak_time = ser.idxmax()
        cust_hours = float(ser.sum() * 0.25)
        positive = ser > 0
        feature_rows.append({
            "disasterNumber":dn,
            "eaglei_match":True,
            "eaglei_rows":matched_rows[dn],
            "eaglei_state_customers":denom,
            "eaglei_peak_customers_out":peak,
            "eaglei_peak_outage_fraction":float(frac.max()) if frac.notna().any() else np.nan,
            "eaglei_customer_hours":cust_hours,
            "eaglei_customer_hours_log1p":math.log1p(max(cust_hours,0.0)),
            "eaglei_peak_customers_log1p":math.log1p(max(peak,0.0)),
            "eaglei_affected_counties":len(counties[dn]),
            "eaglei_positive_outage_hours":float(positive.sum()*0.25),
            "eaglei_hours_ge_10pct":float((frac>=0.10).sum()*0.25) if frac.notna().any() else np.nan,
            "eaglei_hours_ge_25pct":float((frac>=0.25).sum()*0.25) if frac.notna().any() else np.nan,
            "eaglei_hours_ge_50pct":float((frac>=0.50).sum()*0.25) if frac.notna().any() else np.nan,
            "eaglei_hours_ge_75pct":float((frac>=0.75).sum()*0.25) if frac.notna().any() else np.nan,
            "eaglei_peak_time":str(peak_time),
        })
        for t,v in ser.items():
            timeline_rows.append({
                "disasterNumber":dn,
                "timestamp":t,
                "customers_out":float(v),
                "outage_fraction":float(v/denom) if pd.notna(denom) and denom>0 else np.nan,
            })

    return pd.DataFrame(feature_rows), pd.DataFrame(timeline_rows)

def main():
    print("Reading master...")
    master = pd.read_excel(MASTER)
    assert len(master) == 971
    candidates = build_candidate_windows(master)
    print(f"Calendar-2022 FEMA candidates: {len(candidates)}")
    print(candidates.groupby(["state","incidentType"]).size().to_string())

    mcc, state_customers = load_customer_denominators()
    print("States with customer denominators:", len(state_customers))
    print("Puerto Rico modeled customers:", state_customers.get("PR"))

    features, timeline = scan_eaglei(candidates, state_customers)

    # Target inspection happens only AFTER all 2022 candidate rows were enriched.
    target_cols = ["disasterNumber","totalObligatedFunding","missionAssignmentCount"]
    view = candidates.merge(features,on="disasterNumber",how="left").merge(
        master[target_cols],on="disasterNumber",how="left"
    )
    view["funding_band"] = pd.cut(
        pd.to_numeric(view["totalObligatedFunding"],errors="coerce"),
        bins=[-np.inf,50e6,200e6,500e6,np.inf],
        labels=["<50M","50-200M","200-500M","500M+"],
        right=False
    ).astype(str)

    features.to_csv(OUT/"eaglei_features_all_2022_fema.csv",index=False)
    timeline.to_csv(OUT/"eaglei_timelines_all_2022_fema.csv",index=False)
    view.to_csv(OUT/"eaglei_2017_fema_diagnostic.csv",index=False)

    key = view[view["disasterNumber"].isin([4671,4673,4652,4663,4670,4672])].copy()
    key.to_csv(OUT/"key_2022_cases.csv",index=False)

    summary = {
        "candidate_rows_enriched_before_target_inspection":int(len(candidates)),
        "candidate_states":sorted(candidates["state"].unique().tolist()),
        "matched_any_eaglei":int(features["eaglei_match"].fillna(False).sum()),
        "state_customer_denominator_count":len(state_customers),
        "puerto_rico_modeled_customers":state_customers.get("PR"),
        "key_cases":key.replace({np.nan:None}).to_dict(orient="records"),
        "source":{
            "article":"ORNL EAGLE-I / Figshare 24237376",
            "2017_file_id":42547828,
            "mcc_file_id":42547708
        },
        "matching":{
            "uses_target":False,
            "keys":["state","incident time window"],
            "tropical_window_cap_days":10,
            "other_window_cap_days":14,
            "pre_window_hours":12,
            "post_window_hours":24
        },
        "caution":"Retrospective diagnostic. Raw EAGLE-I observations are contemporaneous telemetry, but this run uses the complete post-event window; t0 features must later truncate to the selected decision time."
    }
    (OUT/"eaglei_2017_summary.json").write_text(json.dumps(summary,indent=2,default=str),encoding="utf-8")

    print("\nKEY CASES")
    show_cols=[
        "disasterNumber","state","incidentType","totalObligatedFunding",
        "eaglei_peak_customers_out","eaglei_state_customers","eaglei_peak_outage_fraction",
        "eaglei_customer_hours","eaglei_affected_counties",
        "eaglei_hours_ge_50pct","eaglei_hours_ge_75pct"
    ]
    print(key[[c for c in show_cols if c in key.columns]].to_string(index=False))

if __name__=="__main__":
    main()
