#!/usr/bin/env python3
"""Audit NHC advisory freshness and independently check bulletin UTC issuance.

Research-only. Uses the immutable 123-row result from run 41334493058.
No funding prediction retraining; funding excluded from source audits.
UTC index timestamps are NOT claimed as bulletin issue times until checked.
"""
import os, zipfile, json, re, time
from pathlib import Path
from urllib.parse import urlsplit
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parents[1]
ZIP=ROOT/"audit_inputs"/"tropical_recency"/"tropical-incident-year-matched-advisories.zip"
OUT=ROOT/"audit_outputs"/"tropical_decision_time_recency"
OUT.mkdir(parents=True,exist_ok=True)
S=requests.Session()
S.headers.update({"User-Agent":"Academic-disaster-event-historical-timing-audit/1.0"})
def get(url):
    for attempt in range(3):
        try:
            r=S.get(url,timeout=20);r.raise_for_status();return r.text
        except requests.RequestException:
            time.sleep(attempt+1)
    return None

def bulletin_timing(url, archived_time):
    if not isinstance(url,str) or not url.startswith("https://www.nhc.noaa.gov/archive/"):
        return {"header_status":"no_url"}
    page=get(url)
    if page is None:return {"header_status":"fetch_failed"}
    raw=BeautifulSoup(page,"html.parser").get_text("\n",strip=True)
    # Look at the actual bulletin header, not a later page footer.
    preview=raw[:2400]
    # Header formats: "1100 AM EDT Mon Aug 30 2021"; "0900 UTC ...".
    zone_regex=r"\b(UTC|GMT|AST|ADT|EST|EDT|CST|CDT|MST|MDT|PST|PDT|HST|CHST)\b"
    m=re.search(r"(?im)^\s*(\d{3,4})\s*(AM|PM)?\s*"+zone_regex+r"\b[^\n]*$",preview)
    if not m:return {"header_status":"header_time_not_parsed","snippet":preview[:200]}
    clock=m.group(1);ap=m.group(2);zone=m.group(3)
    h=int(clock[:-2]);minute=int(clock[-2:])
    if ap:
        h=(h%12)+(12 if ap=="PM" else 0)
    offsets={"UTC":0,"GMT":0,"AST":4,"ADT":3,"EST":5,"EDT":4,"CST":6,"CDT":5,"MST":7,"MDT":6,"PST":8,"PDT":7,"HST":10,"CHST":-10}
    expected=pd.to_datetime(archived_time,utc=True)
    local=expected-pd.Timedelta(hours=offsets[zone])
    same_clock=(local.hour==h and local.minute==minute)
    return {"header_status":"clock_matches_index" if same_clock else "clock_mismatch",
            "header_clock":f"{h:02}:{minute:02}","header_zone":zone,
            "index_utc":str(expected),"header_utc_clock_expected":str(local),
            "header_line":m.group(0).strip()[:150]}

def run():
    with zipfile.ZipFile(ZIP) as z:
        d=pd.read_csv(z.open("all_tropical_coverage_by_case.csv"))
        a=pd.read_csv(z.open("storm_matching_and_timestamp_audit.csv"))
    assert len(d)==123 and len(a)==123
    for key in ["nhc_t0_hours_before_cutoff","nhc_t72_hours_before_cutoff"]:
        d[key]=pd.to_numeric(d[key],errors="coerce")
    d["t0_recent24"]=d.nhc_t0_hours_before_cutoff.le(24)&d.nhc_t0_hours_before_cutoff.ge(0)
    d["t0_recent72"]=d.nhc_t0_hours_before_cutoff.le(72)&d.nhc_t0_hours_before_cutoff.ge(0)
    d["t0_recent168"]=d.nhc_t0_hours_before_cutoff.le(168)&d.nhc_t0_hours_before_cutoff.ge(0)
    d["t72_recent72"]=d.nhc_t72_hours_before_cutoff.le(72)&d.nhc_t72_hours_before_cutoff.ge(0)
    # Audit one case per selected source year, stratified by date and UTC zone.
    source=d.loc[d.nhc_t0_source_url.notna(),["disasterNumber","fyDeclared","nhc_t0_source_url","nhc_t0_issued_utc","nhc_t0_archive_local_time"]].copy()
    source["source_year"]=pd.to_datetime(source.nhc_t0_issued_utc,utc=True).dt.year
    for sample_year in sorted(source.source_year.unique()):
        for _,r in source[source.source_year==sample_year].head(2).iterrows():
            parsed=bulletin_timing(r.nhc_t0_source_url,r.nhc_t0_issued_utc)
            parsed.update({"disasterNumber":int(r.disasterNumber),"source_year":int(sample_year),
               "url":r.nhc_t0_source_url,"archive_comment":r.nhc_t0_archive_local_time})
            yield parsed
    d.drop(columns=["funding_band"]).to_csv(OUT/"target_blind_advisory_freshness.csv",index=False)
    summary={"records":len(d),"matched":int(d.nhc_t0_issued_utc.notna().sum()),
       "t0_recent24":int(d.t0_recent24.sum()),"t0_recent72":int(d.t0_recent72.sum()),
       "t0_recent168":int(d.t0_recent168.sum()),"t72_recent72":int(d.t72_recent72.sum()),
       "median_age_hours":float(d.nhc_t0_hours_before_cutoff.median()),
       "t0_age_by_year":d.assign(matched=d.nhc_t0_issued_utc.notna()).groupby("fyDeclared").agg(
          n=("disasterNumber","count"),matched=("matched","sum"),recent72=("t0_recent72","sum")
       ).reset_index().to_dict("records"),
       "source_audit":"NHC original bulletin header clock vs archive index timestamp; report mismatches, no assumptions",
       "cutoff_caveat":"FEMA declarationDate midnight UTC, not precise issuance; storm name/title not yet contemporaneously certified",
       "interpretation":"Latest advisory may be months old; do not treat as contemporary meteorological intensity."}
    (OUT/"freshness_summary.json").write_text(json.dumps(summary,indent=2))
    print("BEGIN_RECENCY_SUMMARY\n"+json.dumps(summary,indent=2)+"\nEND_RECENCY_SUMMARY",flush=True)

if __name__=="__main__":
    checks=list(run())
    pd.DataFrame(checks).to_csv(OUT/"bulletin_header_timestamp_checks.csv",index=False)
    print("BEGIN_BULLETIN_HEADER_CHECKS\n"+json.dumps(checks,indent=2)+"\nEND_BULLETIN_HEADER_CHECKS",flush=True)
