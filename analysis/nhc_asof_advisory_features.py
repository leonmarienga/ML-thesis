#!/usr/bin/env python3
"""As-of archived NHC advisory feature extraction (research-only).

Only timestamped NHC advisories issued no later than FEMA declaration UTC or
72h after it are eligible. Never substitute HURDAT2 final best track.

IMPORTANT: storm identity currently uses the archived retrospective FEMA
declaration-title match. This creates an identity-matching/provenance caveat
for a truly prospective system. Records without a confident unique title match
remain missing. Date-only matching is deliberately forbidden.

Use with prior external_severity_ablation FEMA title function and NOAA master;
the funding target is never read when constructing features.
"""
from __future__ import annotations
import json,re,time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from external_severity_ablation import load_fema_declaration_titles, TROPICAL_TYPES

ROOT=Path(__file__).resolve().parents[1]
MASTER=ROOT/"master_openfema_40plus.xlsx"
OUT=ROOT/"audit_outputs"/"nhc_asof_advisories"
CACHE=ROOT/".cache"/"nhc_asof_advisories"
OUT.mkdir(parents=True,exist_ok=True)
CACHE.mkdir(parents=True,exist_ok=True)
S=requests.Session()
S.headers["User-Agent"]="Disaster relief research as-of NHC archive validation"
# NHC storm seasonal archive has public advisory links and history.
BASE="https://www.nhc.noaa.gov/archive/"
def get(url):
    key=re.sub(r"[^a-zA-Z0-9_-]","_",url)[-160:]
    cache=CACHE/(key+".txt")
    if cache.exists():return cache.read_text(encoding="utf-8")
    for n in range(4):
        try:
            r=S.get(url,timeout=45);r.raise_for_status()
            cache.write_text(r.text,encoding="utf-8")
            return r.text
        except Exception as e:
            print("fetch error",url,repr(e),flush=True);time.sleep(2**n)
    raise RuntimeError("unavailable "+url)

def unique_name(title):
    # A title without a specific hurricane name is not an identifier.
    title=str(title).upper()
    m=re.search(r"\b(?:HURRICANE|TROPICAL STORM|TYPHOON)\s+([A-Z][A-Z-]{2,})\b",title)
    if not m:return None
    value=m.group(1).strip("-")
    if value in {"DISASTER","FLOODING","STORM","WINDS","DAMAGE","AND"}:return None
    return value

def archive_index(year):
    url=f"{BASE}{year}.shtml"
    html=get(url)
    soup=BeautifulSoup(html,"html.parser")
    matches={}
    for a in soup.find_all("a",href=True):
        name=a.get_text(" ",strip=True).upper()
        href=a["href"]
        if "TROPICAL CYCLONE" in name:continue
        if not name or len(name)>100:continue
        matches.setdefault(name,[]).append(requests.compat.urljoin(url,href))
    return matches

def find_storm_url(year,name):
    idx=archive_index(year)
    candidates=[]
    for label,urls in idx.items():
        if re.search(r"\b"+re.escape(name)+r"\b",label) and ("HURRICANE" in label or "STORM" in label):
            candidates.extend(urls)
    candidates=list(dict.fromkeys(candidates))
    return candidates[0] if len(candidates)==1 else None

def advisory_links(storm_url):
    html=get(storm_url);soup=BeautifulSoup(html,"html.parser")
    result=[]
    for a in soup.find_all("a",href=True):
        text=a.get_text(" ",strip=True).upper()
        href=a["href"]
        if "PUBLIC ADVISORY" not in text and ".public." not in href.lower():continue
        u=requests.compat.urljoin(storm_url,href)
        if u.endswith(".shtml") or ".public." in u:result.append(u)
    return list(dict.fromkeys(result))

def parse_advisory(body):
    soup=BeautifulSoup(body,"html.parser")
    raw=soup.get_text("\n",strip=True)
    # Public advisories contain a time line in local clock time, but the
    # correct UTC issue date must be proved: do NOT guess timezone from local.
    z=re.search(r"(?im)^(\d{3,4})\s+UTC\s+(?:MON|TUE|WED|THU|FRI|SAT|SUN)\s+([A-Z]{3})\s+(\d{1,2})\s+(\d{4})\s*$",raw)
    if not z:
        # Some archives print UTC issuance with a date using dd/mm/yyyy.
        z2=re.search(r"(?im)^(\d{3,4})\s+UTC\s+(\d{1,2})/(\d{1,2})/(\d{4})",raw)
        if not z2:return None
        date=pd.Timestamp(year=int(z2.group(4)),month=int(z2.group(3)),day=int(z2.group(2)),tz="UTC")
        hour=int(z2.group(1).zfill(4)[:2]);minute=int(z2.group(1).zfill(4)[2:])
    else:
        date=pd.to_datetime(f"{z.group(2)} {z.group(3)} {z.group(4)}",utc=True,errors="coerce")
        hour=int(z.group(1).zfill(4)[:2]);minute=int(z.group(1).zfill(4)[2:])
    if pd.isna(date) or hour>23 or minute>59:return None
    issued=date+pd.Timedelta(hours=hour,minutes=minute)
    wind=re.search(r"(?i)MAXIMUM SUSTAINED WINDS[^0-9\n]*(\d+)\s+MPH",raw)
    pressure=re.search(r"(?i)MINIMUM CENTRAL PRESSURE[^0-9\n]*(\d+)\s+MB",raw)
    return {"issued_utc":issued,"wind_mph":float(wind.group(1)) if wind else np.nan,
            "pressure_mb":float(pressure.group(1)) if pressure else np.nan}

def main():
    df=pd.read_excel(MASTER)
    titles=load_fema_declaration_titles(df)
    rows=[];audit=[]
    for _,row in df[df.incidentType.isin(TROPICAL_TYPES)].iterrows():
        dn=int(row.disasterNumber)
        date=pd.to_datetime(row.declarationDate,utc=True,errors="coerce")
        name=unique_name(titles.get(dn,""))
        o={"disasterNumber":dn,"storm_name":name,"declaration_utc":str(date)}
        if pd.isna(date) or name is None:
            o["status"]="unidentified";audit.append(o);rows.append({"disasterNumber":dn});continue
        try:
            storm=find_storm_url(int(date.year),name)
            if not storm:
                o["status"]="ambiguous_storm_archive";audit.append(o);rows.append({"disasterNumber":dn});continue
            links=advisory_links(storm)
            observations=[]
            for link in links:
                found=parse_advisory(get(link))
                if found:
                    found["archive_url"]=link
                    observations.append(found)
            fields={"disasterNumber":dn}
            for suffix,hours in [("t0",0),("t72",72)]:
                cutoff=date+pd.Timedelta(hours=hours)
                eligible=[x for x in observations if x["issued_utc"]<=cutoff]
                fields[f"nhc_asof_{suffix}_advisory_count"]=len(eligible)
                if eligible:
                    latest=max(eligible,key=lambda x:x["issued_utc"])
                    fields[f"nhc_asof_{suffix}_wind_mph"]=latest["wind_mph"]
                    fields[f"nhc_asof_{suffix}_pressure_mb"]=latest["pressure_mb"]
                    fields[f"nhc_asof_{suffix}_latest_issued_utc"]=str(latest["issued_utc"])
                    fields[f"nhc_asof_{suffix}_source_url"]=latest["archive_url"]
            o.update({"status":"found","storm_archive":storm,"total_parsed_advisories":len(observations)})
            rows.append(fields);audit.append(o)
        except Exception as e:
            o["status"]="source_error";o["error"]=repr(e)
            audit.append(o);rows.append({"disasterNumber":dn})
        print("NHC",dn,o["status"],flush=True)
    pd.DataFrame(rows).to_csv(OUT/"nhc_asof_features.csv",index=False)
    pd.DataFrame(audit).to_csv(OUT/"nhc_asof_match_audit.csv",index=False)
    result={"all_master":len(df),"tropical_records":len(rows),
            "asof_t0_eligible":int(pd.DataFrame(rows).get("nhc_asof_t0_latest_issued_utc",pd.Series(dtype=object)).notna().sum()),
            "asof_t72_eligible":int(pd.DataFrame(rows).get("nhc_asof_t72_latest_issued_utc",pd.Series(dtype=object)).notna().sum()),
            "status_counts":pd.Series([a["status"] for a in audit]).value_counts().to_dict(),
            "limitation":"Date-level FEMA declaration treated as 00:00 UTC; advisory timestamp must be proven in source; missing remains missing"}
    (OUT/"summary.json").write_text(json.dumps(result,indent=2))
    print("BEGIN_NHC_ASOF_SUMMARY\n"+json.dumps(result,indent=2)+"\nEND_NHC_ASOF_SUMMARY",flush=True)
if __name__=="__main__":main()
