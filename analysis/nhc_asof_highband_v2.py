#!/usr/bin/env python3
"""High-value tropical cases: timestamped NHC public advisory pilot.

Only archival issue-time <= FEMA declaration 00:00 UTC (conservative date-only)
or declaration+72h. Storm name obtained from FEMA declaration title; ambiguous
name/year matches omitted. No outcome variables enter feature construction.
No final HURDAT2 or disaster retrospective intensity enters any as-of column.
"""
from __future__ import annotations
from pathlib import Path
import re,json,time
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"audit_outputs"/"nhc_asof_highband"
OUT.mkdir(parents=True,exist_ok=True)
S=requests.Session()
S.headers["User-Agent"]="Academic-disaster-research/1.0"
NHC="https://www.nhc.noaa.gov/archive/"
FEMA="https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"
def fetch(url,params=None):
    for k in range(3):
        try:
            z=S.get(url,params=params,timeout=25)
            z.raise_for_status()
            return z.text
        except Exception as e:
            print("GET failed",url,str(e)[:90],flush=True);time.sleep(1+k)
    raise RuntimeError("source unavailable")
def official_title(disaster_no):
    raw=json.loads(fetch(FEMA,{"$filter":f"disasterNumber eq {int(disaster_no)}","$top":1}))
    return str((raw.get("DisasterDeclarationsSummaries") or [{}])[0].get("declarationTitle",""))
def parse_name(title):
    m=re.search(r"\b(?:HURRICANE|TROPICAL STORM|TYPHOON)\s+([A-Z][A-Z-]{2,})\b",title.upper())
    if not m:return None
    if m.group(1) in {"DISASTER","WINDS","RAINS","FLOODING","STORM"}:return None
    return m.group(1)
def archive_storm(year,name):
    index=fetch(f"{NHC}{year}/")
    soup=BeautifulSoup(index,"html.parser")
    names={}
    for a in soup.select("a[href]"):
        label=a.get_text(" ",strip=True).upper()
        href=a["href"]
        if not href.lower().endswith(".shtml"):continue
        if not re.search(r"\b"+re.escape(name)+r"\b",label):continue
        url=urljoin(f"{NHC}{year}/",href)
        names[url]=label
    # Do not assume first record correct if the archive index is ambiguous.
    # Direct archive page is a stable public NHC path for named Atlantic systems.
    direct=f"{NHC}{year}/{name.upper()}.shtml"
    try:
        check=S.get(direct,timeout=12)
        if check.status_code==200 and "public." in check.text.lower():
            return direct,[("direct",direct)]
    except requests.RequestException:
        pass
    return list(names)[0] if len(names)==1 else None, list(names.items())[:5]
def archive_public(storm_url):
    text=fetch(storm_url)
    # Dates are present as HTML comments immediately before each bulletin link.
    pattern=re.compile(r'<!--\s*(20\d{10})\s*-->\s*<a\s+href="([^"]*\.public(?:_a)?\.[^"]+\.shtml)"',re.I)
    hits=[]
    for stamp,href in pattern.findall(text):
        t=pd.to_datetime(stamp,format="%Y%m%d%H%M",errors="coerce",utc=True)
        if pd.notna(t):
            hits.append({"issued_utc":t,"url":urljoin(storm_url,href)})
    return sorted(hits,key=lambda x:x["issued_utc"])
def intensity(url):
    text=BeautifulSoup(fetch(url),"html.parser").get_text("\n",strip=True)
    wind=re.search(r"(?i)MAXIMUM SUSTAINED WINDS\s*\.{0,5}\s*(\d{2,3})\s+MPH",text)
    if not wind:wind=re.search(r"(?i)MAXIMUM SUSTAINED WINDS[^0-9]{0,30}(\d{2,3})\s+MPH",text)
    pressure=re.search(r"(?i)MINIMUM CENTRAL PRESSURE[^0-9]{0,35}(\d{3,4})\s+MB",text)
    return {"wind_mph":int(wind.group(1)) if wind else None,"pressure_mb":int(pressure.group(1)) if pressure else None}
def run():
    d=pd.read_excel(ROOT/"master_openfema_40plus.xlsx")
    d=d[(d.incidentType.astype(str).str.contains("Hurricane|Tropical",case=False,regex=True)) &
        (pd.to_numeric(d.totalObligatedFunding,errors="coerce")>=50_000_000)]
    print("High tropical sample",len(d),flush=True)
    rows=[];audit=[]
    for _,r in d.iterrows():
        dn=int(r.disasterNumber)
        date=pd.to_datetime(r.declarationDate,errors="coerce",utc=True)
        diag={"disasterNumber":dn,"year":int(date.year) if pd.notna(date) else None,"state":r.state}
        result={"disasterNumber":dn}
        try:
            title=official_title(dn)
            name=parse_name(title)
            diag.update({"fema_declaration_title":title,"storm_name":name})
            if name is None or pd.isna(date):
                diag["status"]="missing_storm_identity";audit.append(diag);rows.append(result);continue
            storm,ambig=archive_storm(int(date.year),name)
            diag["candidate_storms"]=str(ambig)
            if not storm:
                diag["status"]="ambiguous_or_unmatched_archive";audit.append(diag);rows.append(result);continue
            advisories=archive_public(storm)
            diag.update({"archive":storm,"advisory_links":len(advisories)})
            # FEMA date-only timestamp is conservatively midnight UTC.
            t0=date.normalize()
            for name_,hours in [("t0",0),("t72",72)]:
                eligible=[a for a in advisories if a["issued_utc"]<=t0+pd.Timedelta(hours=hours)]
                if not eligible:continue
                last=eligible[-1]
                values=intensity(last["url"])
                for field,value in values.items():
                    result[f"nhc_asof_{name_}_{field}"]=value
                result[f"nhc_asof_{name_}_issued_utc"]=str(last["issued_utc"])
                result[f"nhc_asof_{name_}_source_url"]=last["url"]
            diag["status"]="matched_advisory" if any("issued_utc" in x for x in result) else "none_before_cutoff"
        except Exception as ex:
            diag.update({"status":"source_error","error":repr(ex)})
        rows.append(result);audit.append(diag)
        print("case",dn,diag["status"],flush=True)
    p=pd.DataFrame(rows)
    p.to_csv(OUT/"nhc_predeclaration_highband_features.csv",index=False)
    pd.DataFrame(audit).to_csv(OUT/"identity_source_audit.csv",index=False)
    summary={"high_tropical_cases":len(d),
      "T0_has_valid_issue_time":int(p.get("nhc_asof_t0_issued_utc",pd.Series(dtype=object)).notna().sum()),
      "T72_has_valid_issue_time":int(p.get("nhc_asof_t72_issued_utc",pd.Series(dtype=object)).notna().sum()),
      "status_counts":pd.Series([a["status"] for a in audit]).value_counts().to_dict(),
      "caveats":["FEMA date-only declarations treated as midnight UTC",
                 "Storm identity from FEMA declaration title may be published after initial event",
                 "Missing or ambiguous storms are not imputed using final best track",
                 "No model retraining until source coverage established"]}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2))
    print("BEGIN_ASOF_HIGHBAND_SUMMARY\n"+json.dumps(summary,indent=2)+"\nEND_ASOF_HIGHBAND_SUMMARY",flush=True)
if __name__=="__main__":run()
