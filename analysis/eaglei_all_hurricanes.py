#!/usr/bin/env python3
"""
Build normalized EAGLE-I outage features for ALL high-value Hurricane declarations
with source coverage (2014+), using one national-file scan per event year.

No target values are used for matching. Target threshold is used only after the
features have been generated to define the evaluation subset.

Output rows with no source coverage retain NaN outage features.
"""

from __future__ import annotations
import json, re
from pathlib import Path
from typing import Dict

import duckdb
import numpy as np
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[1]
MASTER=ROOT/"master_openfema_40plus.xlsx"
OUT=ROOT/"audit_outputs"/"eaglei_high_hurricanes"
OUT.mkdir(parents=True,exist_ok=True)

ARTICLE="https://api.figshare.com/v2/articles/24237376"
MCC_FALLBACK="https://ndownloader.figshare.com/files/42547708"

STATE_NAMES={
"AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California","CO":"Colorado",
"CT":"Connecticut","DE":"Delaware","FL":"Florida","GA":"Georgia","HI":"Hawaii","ID":"Idaho",
"IL":"Illinois","IN":"Indiana","IA":"Iowa","KS":"Kansas","KY":"Kentucky","LA":"Louisiana",
"ME":"Maine","MD":"Maryland","MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MS":"Mississippi",
"MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada","NH":"New Hampshire","NJ":"New Jersey",
"NM":"New Mexico","NY":"New York","NC":"North Carolina","ND":"North Dakota","OH":"Ohio","OK":"Oklahoma",
"OR":"Oregon","PA":"Pennsylvania","RI":"Rhode Island","SC":"South Carolina","SD":"South Dakota",
"TN":"Tennessee","TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington",
"WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming","DC":"District of Columbia",
"PR":"Puerto Rico","VI":"Virgin Islands","GU":"Guam","MP":"Northern Mariana Islands","AS":"American Samoa"}

STATE_FIPS={
"AL":"01","AK":"02","AZ":"04","AR":"05","CA":"06","CO":"08","CT":"09","DE":"10","DC":"11",
"FL":"12","GA":"13","HI":"15","ID":"16","IL":"17","IN":"18","IA":"19","KS":"20","KY":"21",
"LA":"22","ME":"23","MD":"24","MA":"25","MI":"26","MN":"27","MS":"28","MO":"29","MT":"30",
"NE":"31","NV":"32","NH":"33","NJ":"34","NM":"35","NY":"36","NC":"37","ND":"38","OH":"39",
"OK":"40","OR":"41","PA":"42","RI":"44","SC":"45","SD":"46","TN":"47","TX":"48","UT":"49",
"VT":"50","VA":"51","WA":"53","WV":"54","WI":"55","WY":"56","AS":"60","GU":"66","MP":"69",
"PR":"72","VI":"78"}

METRIC_COLS=[
"eaglei_peak_customers_out","eaglei_total_customers_mcc","eaglei_peak_outage_share",
"eaglei_customer_hours_out","eaglei_customer_hours_per_customer",
"eaglei_residual_share_3d","eaglei_residual_share_7d","eaglei_residual_share_14d",
"eaglei_half_life_hours","eaglei_restore90_hours","eaglei_observations"
]

def figshare_files():
    js=requests.get(ARTICLE,timeout=120).json()
    out={}
    mcc=None
    for f in js.get("files",[]):
        name=f.get("name","")
        m=re.fullmatch(r"eaglei_outages_(20\\d{2})\\.csv",name)
        if m:
            out[int(m.group(1))]=f.get("download_url") or f"https://ndownloader.figshare.com/files/{f['id']}"
        if name=="MCC.csv":
            mcc=f.get("download_url") or f"https://ndownloader.figshare.com/files/{f['id']}"
    return out,mcc or MCC_FALLBACK,js

def load_mcc(url):
    p=OUT/"MCC.csv"
    if not p.exists():
        r=requests.get(url,timeout=120); r.raise_for_status(); p.write_bytes(r.content)
    m=pd.read_csv(p,encoding="utf-8-sig")
    m.columns=[c.strip() for c in m.columns]
    m["County_FIPS"]=pd.to_numeric(m["County_FIPS"],errors="coerce")
    m=m[m["County_FIPS"].notna()].copy()
    m["fips"]=m["County_FIPS"].astype(int).astype(str).str.zfill(5)
    m["Customers"]=pd.to_numeric(m["Customers"],errors="coerce")
    return m

def scan_year(url, states, start, end):
    con=duckdb.connect()
    placeholders=",".join(["?"]*len(states))
    q=f"""
    SELECT CAST(run_start_time AS VARCHAR) run_start_time,
           CAST(state AS VARCHAR) state,
           SUM(TRY_CAST(customers_out AS DOUBLE)) state_out
    FROM read_csv_auto(?,header=true,all_varchar=true,sample_size=200000)
    WHERE state IN ({placeholders})
      AND TRY_CAST(run_start_time AS TIMESTAMP) >= ?
      AND TRY_CAST(run_start_time AS TIMESTAMP) <= ?
    GROUP BY state,run_start_time
    ORDER BY state,TRY_CAST(run_start_time AS TIMESTAMP)
    """
    params=[url,*states,start.tz_localize(None).to_pydatetime(),end.tz_localize(None).to_pydatetime()]
    return con.execute(q,params).df()

def compute(ts,total_customers):
    if ts.empty:return {}
    x=ts.copy()
    x["t"]=pd.to_datetime(x.run_start_time,errors="coerce",utc=True)
    x["out"]=pd.to_numeric(x.state_out,errors="coerce")
    x=x.dropna(subset=["t","out"]).sort_values("t")
    if x.empty:return {}
    i=x["out"].idxmax(); peak=float(x.loc[i,"out"]); pt=x.loc[i,"t"]
    dt=x.t.diff().dt.total_seconds().div(3600).fillna(0).clip(0,1)
    ch=float((x.out*dt).sum())
    def residual(d):
        z=x[x.t>=pt+pd.Timedelta(days=d)]
        return float(z.iloc[0].out/peak) if len(z) and peak>0 else np.nan
    after=x[x.t>=pt]
    h50=after[after.out<=.5*peak]; h10=after[after.out<=.1*peak]
    return {
        "eaglei_peak_customers_out":peak,
        "eaglei_total_customers_mcc":float(total_customers),
        "eaglei_peak_outage_share":peak/total_customers if total_customers>0 else np.nan,
        "eaglei_customer_hours_out":ch,
        "eaglei_customer_hours_per_customer":ch/total_customers if total_customers>0 else np.nan,
        "eaglei_residual_share_3d":residual(3),
        "eaglei_residual_share_7d":residual(7),
        "eaglei_residual_share_14d":residual(14),
        "eaglei_half_life_hours":float((h50.iloc[0].t-pt).total_seconds()/3600) if len(h50) else np.nan,
        "eaglei_restore90_hours":float((h10.iloc[0].t-pt).total_seconds()/3600) if len(h10) else np.nan,
        "eaglei_observations":int(len(x)),
    }

def main():
    master=pd.read_excel(MASTER)
    # Features built for ALL Hurricane declarations, not only high target rows.
    h=master[master.incidentType=="Hurricane"].copy()
    h["begin"]=pd.to_datetime(h.incidentBeginDate,errors="coerce",utc=True)
    h["event_year"]=h["begin"].dt.year
    urls,mcc_url,meta=figshare_files()
    mcc=load_mcc(mcc_url)
    (OUT/"figshare_metadata.json").write_text(json.dumps({
        "article_id":24237376,
        "years_available":sorted(urls),
        "file_count":len(meta.get("files",[]))
    },indent=2),encoding="utf-8")

    rows=[]
    for _,r in h.iterrows():
        base={"disasterNumber":int(r.disasterNumber)}
        base.update({c:np.nan for c in METRIC_COLS})
        base["eaglei_coverage"]=0
        rows.append(base)
    feat=pd.DataFrame(rows).set_index("disasterNumber")

    eligible=h[h.event_year.isin(urls.keys()) & h.begin.notna()].copy()
    for year,g in eligible.groupby("event_year"):
        year=int(year)
        states=sorted({STATE_NAMES.get(s) for s in g.state if STATE_NAMES.get(s)})
        if not states:continue
        start=g.begin.min()-pd.Timedelta(days=2)
        end=g.begin.max()+pd.Timedelta(days=30)
        print(f"Scanning EAGLE-I {year}: {states} {start}..{end}",flush=True)
        yr=scan_year(urls[year],states,start,end)
        yr["t"]=pd.to_datetime(yr.run_start_time,errors="coerce",utc=True)

        for _,r in g.iterrows():
            state_name=STATE_NAMES.get(r.state)
            fips=STATE_FIPS.get(r.state)
            if not state_name or not fips:continue
            s=r.begin-pd.Timedelta(days=2); e=r.begin+pd.Timedelta(days=30)
            z=yr[(yr.state==state_name)&(yr.t>=s)&(yr.t<=e)].copy()
            total=float(mcc.loc[mcc.fips.str.startswith(fips),"Customers"].sum())
            m=compute(z,total)
            dn=int(r.disasterNumber)
            if m:
                for k,v in m.items(): feat.loc[dn,k]=v
                feat.loc[dn,"eaglei_coverage"]=1
            print(dn,r.state,year,len(z),m.get("eaglei_peak_outage_share"),m.get("eaglei_customer_hours_per_customer"),flush=True)

    feat=feat.reset_index()
    feat.to_csv(OUT/"all_hurricane_eaglei_features.csv",index=False)

    # Evaluation view selected only AFTER feature generation.
    ev=master[(master.incidentType=="Hurricane")&(master.totalObligatedFunding>=50_000_000)][
        ["disasterNumber","state","fyDeclared","totalObligatedFunding"]
    ].merge(feat,on="disasterNumber",how="left")
    ev.to_csv(OUT/"high_value_hurricane_eaglei_features.csv",index=False)

    summary={
        "all_hurricane_rows":int(len(h)),
        "feature_coverage_rows":int(feat.eaglei_coverage.sum()),
        "high_value_hurricanes":int(len(ev)),
        "high_value_with_coverage":int(ev.eaglei_coverage.fillna(0).sum()),
        "years_available":sorted(urls),
        "source":"ORNL EAGLE-I via Figshare",
        "timing_note":"Retrospective external_final diagnostic. Source absence remains missing, never zero."
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(summary)

if __name__=="__main__":
    main()
