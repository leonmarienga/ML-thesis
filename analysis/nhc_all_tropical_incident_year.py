#!/usr/bin/env python3
"""All tropical FEMA disasters: target-blind timestamped NHC public advisory extraction.

Only archival issue-time <= FEMA declaration 00:00 UTC (conservative date-only)
or declaration+72h. Storm name obtained from FEMA declaration title; ambiguous
name/year matches omitted. No outcome variables enter feature construction.
No final HURDAT2 or disaster retrospective intensity enters any as-of column.
"""
from __future__ import annotations
from pathlib import Path
import re,json,time
from functools import lru_cache
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
@lru_cache(maxsize=2048)
def official_title(disaster_no):
    raw=json.loads(fetch(FEMA,{"$filter":f"disasterNumber eq {int(disaster_no)}","$top":1}))
    return str((raw.get("DisasterDeclarationsSummaries") or [{}])[0].get("declarationTitle",""))
def parse_name(title):
    m=re.search(r"\b(?:HURRICANE|TROPICAL STORM|TYPHOON)\s+([A-Z][A-Z-]{2,})\b",title.upper())
    if not m:return None
    if m.group(1) in {"DISASTER","WINDS","RAINS","FLOODING","STORM"}:return None
    return m.group(1)
@lru_cache(maxsize=512)
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
@lru_cache(maxsize=512)
def archive_public(storm_url):
    html=fetch(storm_url)
    # Adjacent HTML comment is a local archive timestamp in legacy years.
    # The label supplies a daylight/standard timezone; never infer UTC.
    pattern=re.compile(
        r'<!--\s*(20\d{6})\s+(\d{4})\s*-->\s*<a\s+href="([^"]*\.public(?:_a)?\.[^"]+\.shtml\??)"[^>]*>(.*?)</a>',
        re.I|re.S)
    offsets={"UTC":0,"GMT":0,"AST":4,"ADT":3,"EST":5,"EDT":4,"CST":6,"CDT":5,"MST":7,"MDT":6,"PST":8,"PDT":7,"HST":10,"CHST":-10}
    hits=[]
    for day,clock,href,label_html in pattern.findall(html):
        label=BeautifulSoup(label_html,"html.parser").get_text(" ",strip=True).upper()
        zone=re.search(r"\b(UTC|GMT|AST|ADT|EST|EDT|CST|CDT|MST|MDT|PST|PDT|HST|CHST)\b",label)
        if not zone:continue
        naive=pd.to_datetime(day+clock,format="%Y%m%d%H%M",errors="coerce",utc=True)
        if pd.isna(naive):continue
        stamp=naive+pd.Timedelta(hours=offsets[zone.group(1)])
        hits.append({"issued_utc":stamp,"url":urljoin(storm_url,href),
            "archive_local_time":day+" "+clock+" "+zone.group(1),
            "source":"NHC original index HTML timestamp + local zone label"})
    unique={h["url"]:h for h in hits}
    return sorted(unique.values(),key=lambda x:x["issued_utc"])
@lru_cache(maxsize=1024)
def intensity(url):
    text=BeautifulSoup(fetch(url),"html.parser").get_text("\n",strip=True)
    wind=re.search(r"(?i)MAXIMUM SUSTAINED WINDS\s*\.{0,5}\s*(\d{2,3})\s+MPH",text)
    if not wind:wind=re.search(r"(?i)MAXIMUM SUSTAINED WINDS[^0-9]{0,30}(\d{2,3})\s+MPH",text)
    pressure=re.search(r"(?i)MINIMUM CENTRAL PRESSURE[^0-9]{0,35}(\d{3,4})\s+MB",text)
    return {"wind_mph":int(wind.group(1)) if wind else None,"pressure_mb":int(pressure.group(1)) if pressure else None}

def funding_band_from_amount(v):
    # ONLY for POST-hoc coverage audit. Never used during any source matching.
    if pd.isna(v):return "missing_target"
    x=max(float(v),0)
    if x<1e5:return "0-100K"
    if x<1e6:return "100K-1M"
    if x<5e7:return "1M-50M"
    if x<2e8:return "50M-200M"
    if x<5e8:return "200M-500M"
    return "500M+"

def run():
    master=pd.read_excel(ROOT/"master_openfema_40plus.xlsx")
    assert len(master)==971 and master.disasterNumber.nunique()==971
    tropical=master[master.incidentType.astype(str).str.strip().isin(
        ["Hurricane","Tropical Storm","Typhoon"])].copy()
    assert tropical.disasterNumber.nunique()==len(tropical)
    print("ALL_TROPICAL_RECORDS",len(tropical),flush=True)
    feature_rows=[];audit=[]
    for number,(_,rec) in enumerate(tropical.sort_values(["fyDeclared","disasterNumber"]).iterrows(),1):
        dn=int(rec.disasterNumber)
        raw_date=str(rec.declarationDate)
        date=pd.to_datetime(rec.declarationDate,errors="coerce",utc=True)
        t0=date.normalize() if pd.notna(date) else pd.NaT
        details={
            "disasterNumber":dn,"state":rec.state,"incidentType":rec.incidentType,
            "declaration_field_raw":raw_date,"incident_begin_raw":str(rec.incidentBeginDate),
            "cutoff_t0_utc":str(t0),"cutoff_t72_utc":str(t0+pd.Timedelta(hours=72)) if pd.notna(t0) else None,
            "year":int(t0.year) if pd.notna(t0) else None,
            "fema_title_contemporaneously_verified":False,
            "official_declaration_exact_issuance_verified":False,
        }
        values={"disasterNumber":dn}
        try:
            title=official_title(dn)
            name=parse_name(title)
            details["fema_title_current"]=title
            details["named_storm"]=name
            if name is None or pd.isna(t0):
                details["status"]="missing_named_storm_or_date"
            elif str(rec.incidentType)=="Typhoon":
                details["status"]="western_pacific_requires_separate_source"
            else:
                # No funding target, agency count, or mission data used in matching.
                url,possible=archive_storm(int(pd.to_datetime(rec.incidentBeginDate,errors="coerce",utc=True).year) if pd.notna(pd.to_datetime(rec.incidentBeginDate,errors="coerce",utc=True)) else int(t0.year),name)
                details["candidate_archives"]=str(possible)
                if not url:
                    details["status"]="ambiguous_or_missing_nhc_archive"
                else:
                    events=archive_public(url)
                    details.update({"storm_archive_url":url,"advisory_links":len(events),
                                    "storm_id_source":"current FEMA declaration title + NHC incident-begin-year archive index"})
                    for tag,limit in [("t0",t0),("t72",t0+pd.Timedelta(hours=72))]:
                        valid=[a for a in events if a["issued_utc"]<=limit]
                        if not valid:
                            continue
                        latest=valid[-1]
                        parsed=intensity(latest["url"])
                        values.update({
                            f"nhc_{tag}_wind_mph":parsed.get("wind_mph"),
                            f"nhc_{tag}_pressure_mb":parsed.get("pressure_mb"),
                            f"nhc_{tag}_issued_utc":str(latest["issued_utc"]),
                            f"nhc_{tag}_source_url":latest["url"],
                            f"nhc_{tag}_archive_local_time":latest["archive_local_time"],
                            f"nhc_{tag}_hours_before_cutoff":round(float((limit-latest["issued_utc"]).total_seconds()/3600),2),
                        })
                    details["status"]="matched_advisory" if "nhc_t0_issued_utc" in values else (
                        "no_advisory_at_t0" if "nhc_t72_issued_utc" in values else "no_advisory_by_t72")
        except Exception as e:
            details["status"]="source_error"
            details["error"]=str(e)[:350]
        audit.append(details);feature_rows.append(values)
        if number%10==0 or number==len(tropical):
            print("PROGRESS",number,"/",len(tropical),"matched",sum(x["status"]=="matched_advisory" for x in audit),flush=True)
    features=pd.DataFrame(feature_rows)
    coverage=master[["disasterNumber","fyDeclared","incidentType","state","totalObligatedFunding"]].merge(
        features,on="disasterNumber",how="left",validate="one_to_one")
    assert len(coverage)==971
    coverage["funding_band"]=coverage.totalObligatedFunding.map(funding_band_from_amount)
    tropical_coverage=coverage[coverage.disasterNumber.isin(tropical.disasterNumber)].copy()
    case_audit=pd.DataFrame(audit)
    joined=tropical_coverage.merge(case_audit[["disasterNumber","status","named_storm","storm_archive_url"]]
       if "storm_archive_url" in case_audit.columns else case_audit[["disasterNumber","status","named_storm"]],
       on="disasterNumber",validate="one_to_one")
    out=ROOT/"audit_outputs"/"nhc_asof_incident_year"
    out.mkdir(parents=True,exist_ok=True)
    features.to_csv(out/"nhc_tropical_features_target_blind.csv",index=False)
    coverage.drop(columns="totalObligatedFunding").to_csv(out/"all_971_nhc_features_with_bands.csv",index=False)
    joined.drop(columns="totalObligatedFunding").to_csv(out/"all_tropical_coverage_by_case.csv",index=False)
    case_audit.to_csv(out/"storm_matching_and_timestamp_audit.csv",index=False)
    counts=case_audit.status.value_counts().to_dict()
    t0=int(tropical_coverage.nhc_t0_issued_utc.notna().sum()) if "nhc_t0_issued_utc" in tropical_coverage else 0
    t72=int(tropical_coverage.nhc_t72_issued_utc.notna().sum()) if "nhc_t72_issued_utc" in tropical_coverage else 0
    rows=[]
    for band,group in joined.groupby("funding_band"):
        rows.append({"funding_band":band,"n":int(len(group)),
            "t0":int(group.nhc_t0_issued_utc.notna().sum()),
            "t72":int(group.nhc_t72_issued_utc.notna().sum())})
    summary={
        "population_total":len(master),
        "all_tropical_target_blind_records":len(tropical),
        "t0_source_timestamp_coverage":t0,
        "t72_source_timestamp_coverage":t72,
        "t0_numeric_wind_coverage":int(tropical_coverage.nhc_t0_wind_mph.notna().sum()) if "nhc_t0_wind_mph" in tropical_coverage else 0,
        "unique_named_storm_year_pairs_with_t0":int(joined.loc[joined.nhc_t0_issued_utc.notna()].assign(
           key=lambda x:x.named_storm.astype(str)+"|"+x.fyDeclared.astype(str)).key.nunique()),
        "status_counts":{str(k):int(v) for k,v in counts.items()},
        "coverage_by_band":rows,
        "coverage_by_fiscal_year":joined.assign(
           t0_found=joined.nhc_t0_issued_utc.notna(),t72_found=joined.nhc_t72_issued_utc.notna()
        ).groupby("fyDeclared").agg(n=("disasterNumber","count"),t0=("t0_found","sum"),t72=("t72_found","sum")).reset_index().to_dict("records"),
        "critical_caveats":[
            "No funding amount or band is used during storm matching; amounts inspected only afterward for diagnostics.",
            "FEMA declarationDate appears date-based; t0 is 00:00 UTC on that date, NOT a verified precise declaration issue time.",
            "Current FEMA declarationTitle may have been updated after original declaration; storm identity is not certified as-of.",
            "NHC seasonal archive index HTML comment clock is converted using zone in bulletin link; validate original bulletin timestamps.",
            "Latest advisory wind may describe storm weakening and is not the event's maximum hazard or local exposure.",
            "Same named storm can have multiple disaster records: use storm-grouped tests.",
            "Typhoon and name-unmatched records have no NHC imputations.",
            "No accepted router or amount specialist is modified.","Archived original advisory issue-time should be independently spot-checked against the bulletin header."
        ]
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2,default=str),encoding="utf-8")
    print("BEGIN_ALL_TROPICAL_ASOF_SUMMARY",flush=True)
    print(json.dumps(summary,indent=2,default=str),flush=True)
    print("END_ALL_TROPICAL_ASOF_SUMMARY",flush=True)
if __name__=="__main__":run()
