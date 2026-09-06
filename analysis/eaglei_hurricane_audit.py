#!/usr/bin/env python3
"""
EAGLE-I hurricane outage audit.

Uses DOE/ORNL OpenEnergyDataPortal EAGLE-I historic outage data (2014+)
to measure grid-disruption persistence for high-value hurricane cases.

Target is NEVER used for source matching. Matching uses state + FEMA incident
dates only. This is a retrospective/final diagnostic until t0 is defined.
"""

from __future__ import annotations
import json, math, re, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from mission_semantic_audit import normalize_master

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "eaglei_hurricane"
OUT.mkdir(parents=True, exist_ok=True)

BASE = "https://openenergyhub.ornl.gov/api/explore/v2.1/catalog/datasets/eaglei_outages_2014"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent":"thesis-eaglei-hurricane-audit/1.0"})

FULL_STATE = {
    "AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California",
    "CO":"Colorado","CT":"Connecticut","DE":"Delaware","FL":"Florida","GA":"Georgia",
    "HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana","IA":"Iowa","KS":"Kansas",
    "KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland","MA":"Massachusetts",
    "MI":"Michigan","MN":"Minnesota","MS":"Mississippi","MO":"Missouri","MT":"Montana",
    "NE":"Nebraska","NV":"Nevada","NH":"New Hampshire","NJ":"New Jersey","NM":"New Mexico",
    "NY":"New York","NC":"North Carolina","ND":"North Dakota","OH":"Ohio","OK":"Oklahoma",
    "OR":"Oregon","PA":"Pennsylvania","RI":"Rhode Island","SC":"South Carolina",
    "SD":"South Dakota","TN":"Tennessee","TX":"Texas","UT":"Utah","VT":"Vermont",
    "VA":"Virginia","WA":"Washington","WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming",
    "DC":"District of Columbia","PR":"Puerto Rico","VI":"Virgin Islands","GU":"Guam",
    "MP":"Northern Mariana Islands","AS":"American Samoa",
}

def get(url, **kwargs):
    for a in range(6):
        try:
            r=SESSION.get(url, timeout=120, **kwargs)
            r.raise_for_status()
            return r
        except Exception:
            if a==5: raise
            time.sleep(2**a)

def norm(s): return re.sub(r"[^a-z0-9]","",str(s).lower())

def discover_fields():
    meta=get(BASE).json()
    fields=meta.get("fields",[])
    probe=get(BASE+"/records", params={"limit":1}).json()
    rec=(probe.get("results") or [{}])[0]

    catalog=[]
    for f in fields:
        fid=f.get("name") or f.get("id")
        lab=f.get("label") or fid
        catalog.append((fid,lab,f.get("type")))
    if not catalog and rec:
        catalog=[(k,k,type(v).__name__) for k,v in rec.items()]

    def pick(preds):
        for fid,lab,typ in catalog:
            text=norm(fid)+" "+norm(lab)
            if all(p(text) for p in preds):
                return fid
        return None

    state=pick([lambda x:"state" in x])
    timestamp=pick([lambda x:any(k in x for k in ["timestamp","datetime","runstarttime","time"])])
    outage=pick([
        lambda x:"customer" in x,
        lambda x:any(k in x for k in ["out","withoutpower","outage"]),
        lambda x:"totalcustomer" not in x
    ])
    total=pick([lambda x:"total" in x, lambda x:"customer" in x])

    info={"fields":catalog,"probe":rec,"selected":{
        "state":state,"timestamp":timestamp,"outage":outage,"total":total
    }}
    (OUT/"schema_probe.json").write_text(json.dumps(info,indent=2,default=str))
    if not all([state,timestamp,outage]):
        raise RuntimeError(f"Could not identify required fields: {info['selected']}")
    return state,timestamp,outage,total,info

def quote(v): return '"' + str(v).replace('"','\\"') + '"'

def query_series(state_field, ts_field, out_field, state_value, start, end):
    where=(
        f"{state_field}={quote(state_value)} AND "
        f"{ts_field} >= {quote(start.strftime('%Y-%m-%dT%H:%M:%SZ'))} AND "
        f"{ts_field} <= {quote(end.strftime('%Y-%m-%dT%H:%M:%SZ'))}"
    )
    rows=[]
    offset=0
    while True:
        params={
            "select":f"{ts_field}, sum({out_field}) as state_out",
            "where":where,
            "group_by":ts_field,
            "order_by":f"{ts_field} asc",
            "limit":100,
            "offset":offset,
        }
        js=get(BASE+"/records",params=params).json()
        batch=js.get("results",[])
        rows.extend(batch)
        if len(batch)<100: break
        offset += 100
        if offset>20000: raise RuntimeError("pagination safety")
    return pd.DataFrame(rows)

def outage_features(ts: pd.DataFrame, ts_field: str) -> Dict[str,float]:
    if ts.empty: return {}
    x=ts.copy()
    x["t"]=pd.to_datetime(x[ts_field],errors="coerce",utc=True)
    x["out"]=pd.to_numeric(x["state_out"],errors="coerce")
    x=x.dropna(subset=["t","out"]).sort_values("t")
    if x.empty: return {}
    # collapse any duplicates
    x=x.groupby("t",as_index=False)["out"].max()
    peak=float(x["out"].max())
    peak_idx=int(x["out"].idxmax())
    peak_t=x.loc[peak_idx,"t"]
    # integrate with actual timestamp deltas, cap huge gaps at 1 hour
    dt=x["t"].diff().dt.total_seconds().div(3600).fillna(0).clip(0,1)
    customer_hours=float((x["out"]*dt).sum())

    def residual(days):
        target=peak_t+pd.Timedelta(days=days)
        after=x[x["t"]>=target]
        if after.empty or peak<=0: return np.nan
        return float(after.iloc[0]["out"]/peak)

    half=x[(x["t"]>=peak_t)&(x["out"]<=0.5*peak)]
    half_hours=float((half.iloc[0]["t"]-peak_t).total_seconds()/3600) if not half.empty else np.nan
    q10=x[(x["t"]>=peak_t)&(x["out"]<=0.1*peak)]
    restore90=float((q10.iloc[0]["t"]-peak_t).total_seconds()/3600) if not q10.empty else np.nan

    return {
        "eaglei_peak_customers_out":peak,
        "eaglei_customer_hours":customer_hours,
        "eaglei_log_customer_hours":math.log1p(max(customer_hours,0)),
        "eaglei_half_life_hours":half_hours,
        "eaglei_restore90_hours":restore90,
        "eaglei_residual_3d":residual(3),
        "eaglei_residual_7d":residual(7),
        "eaglei_residual_14d":residual(14),
        "eaglei_samples":int(len(x)),
    }

def main():
    master=normalize_master(pd.read_excel(MASTER))
    h=master[(master["incidentType"]=="Hurricane") & (master["totalObligatedFunding"]>=50_000_000)].copy()
    state_field,ts_field,out_field,total_field,info=discover_fields()
    print("Selected fields:",info["selected"])

    rows=[]; audit=[]
    for _,r in h.iterrows():
        dn=int(r["disasterNumber"]); st=str(r["state"])
        begin=pd.to_datetime(r["incidentBeginDate"],errors="coerce",utc=True)
        end=pd.to_datetime(r["incidentEndDate"],errors="coerce",utc=True)
        if pd.isna(begin):
            continue
        # EAGLE-I begins in 2014. Don't turn absence into a zero.
        if begin.year < 2014:
            rows.append({"disasterNumber":dn})
            audit.append({"disasterNumber":dn,"state":st,"status":"pre_2014_no_coverage"})
            continue
        # Focus on outage/restoration around onset; cap FEMA-wide long windows.
        start=begin-pd.Timedelta(days=2)
        stop=min(end if pd.notna(end) else begin+pd.Timedelta(days=7), begin+pd.Timedelta(days=14))
        stop=stop+pd.Timedelta(days=14)

        candidates=[st,FULL_STATE.get(st,st)]
        data=pd.DataFrame(); used=None
        err=None
        for sv in candidates:
            try:
                q=query_series(state_field,ts_field,out_field,sv,start,stop)
                if not q.empty:
                    data=q; used=sv; break
            except Exception as e:
                err=str(e)
        feat={"disasterNumber":dn}
        feat.update(outage_features(data,ts_field))
        rows.append(feat)
        audit.append({
            "disasterNumber":dn,"state":st,"state_value_used":used,
            "start":str(start),"stop":str(stop),"rows":len(data),
            "status":"matched" if len(data) else "none","error":err
        })
        print(dn,st,used,len(data),feat.get("eaglei_peak_customers_out"))

    pd.DataFrame(rows).to_csv(OUT/"hurricane_eaglei_features.csv",index=False)
    pd.DataFrame(audit).to_csv(OUT/"hurricane_eaglei_match_audit.csv",index=False)

if __name__=="__main__":
    main()
