from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import roc_auc_score

import external_physical_severity_high_router_experiment as base

OUT = Path("hurricane_physical_severity_signal_audit_results")
OUT.mkdir(exist_ok=True)

HURDAT_URLS = [
    "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2024-040425.txt",
    "https://www.nhc.noaa.gov/data/hurdat/hurdat2-nepac-1949-2024-040425.txt",
]
FEMA_URL = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"

def num(x):
    return pd.to_numeric(x, errors="coerce").replace([np.inf,-np.inf],np.nan)

def fetch_titles(dns):
    rows=[]
    for i in range(0,len(dns),40):
        chunk=dns[i:i+40]
        filt=" or ".join([f"disasterNumber eq {int(x)}" for x in chunk])
        params={"$filter":filt,"$select":"disasterNumber,state,declarationTitle,declarationDate,incidentBeginDate,incidentEndDate","$top":"1000","$metadata":"off"}
        r=requests.get(FEMA_URL,params=params,timeout=180,headers={"User-Agent":"ML-thesis-research/1.0"})
        r.raise_for_status()
        p=r.json()
        recs=p.get("DisasterDeclarationsSummaries", p.get("items", [])) if isinstance(p,dict) else p
        rows.extend(recs)
    d=pd.DataFrame(rows)
    if d.empty: raise RuntimeError("No FEMA declaration titles returned")
    d["disasterNumber"]=num(d["disasterNumber"]).astype("Int64")
    # one title/date per disaster
    d=d.sort_values(["disasterNumber","declarationDate"]).drop_duplicates("disasterNumber",keep="last")
    return d

def norm_name(s):
    s=str(s).upper()
    s=re.sub(r"[^A-Z0-9 ]+"," ",s)
    for w in ["HURRICANE","TROPICAL","STORM","TYPHOON","SEVERE","WINDS","WIND"]:
        s=s.replace(w," ")
    s=re.sub(r"\s+"," ",s).strip()
    return s

def parse_hurdat():
    storms=[]
    for url in HURDAT_URLS:
        r=requests.get(url,timeout=180,headers={"User-Agent":"ML-thesis-research/1.0"})
        if r.status_code == 404:
            continue
        r.raise_for_status()
        lines=[x.strip() for x in r.text.splitlines() if x.strip()]
        i=0
        while i<len(lines):
            head=[x.strip() for x in lines[i].split(",")]
            if len(head)<3 or not re.match(r"^[A-Z]{2}\d{6}$",head[0]):
                i+=1; continue
            sid=head[0]; name=head[1].strip(); n=int(head[2]); pts=[]
            for j in range(i+1,min(i+1+n,len(lines))):
                p=[x.strip() for x in lines[j].split(",")]
                if len(p)<8: continue
                dt=pd.to_datetime(p[0]+p[1],format="%Y%m%d%H%M",errors="coerce")
                status=p[3]
                wind=pd.to_numeric(pd.Series([p[6]]),errors="coerce").iloc[0]
                pressure=pd.to_numeric(pd.Series([p[7]]),errors="coerce").iloc[0]
                lat_s=p[4]; lon_s=p[5]
                def coord(v):
                    try:
                        sign=-1 if v[-1] in "SW" else 1
                        return sign*float(v[:-1])
                    except: return np.nan
                pts.append((dt,status,wind,pressure,coord(lat_s),coord(lon_s)))
            if pts:
                z=pd.DataFrame(pts,columns=["dt","status","wind","pressure","lat","lon"])
                storms.append({
                    "storm_id":sid,"storm_name":name.upper(),"year":int(sid[-4:]),
                    "start":z.dt.min(),"end":z.dt.max(),
                    "max_wind_kt":float(num(z.wind).max()),
                    "min_pressure_mb":float(num(z.pressure).replace(-999,np.nan).min()),
                    "major_hours":float(((num(z.wind)>=96).sum())*6),
                    "hurricane_hours":float(((num(z.wind)>=64).sum())*6),
                    "ace_proxy":float(((num(z.wind).fillna(0).clip(lower=34)**2).sum())/10000),
                    "track_span_deg":float((num(z.lat).max()-num(z.lat).min())+(num(z.lon).max()-num(z.lon).min())),
                })
            i+=n+1
    s=pd.DataFrame(storms)
    s.to_csv(OUT/"hurdat_storm_summary.csv",index=False)
    return s

def match_storm(row,storms):
    title=str(row.get("declarationTitle",""))
    nm=norm_name(title)
    dt=pd.to_datetime(row.get("declarationDate"),errors="coerce")
    yr=int(dt.year) if pd.notna(dt) else int(row.get("fyDeclared",0))
    cand=storms[(storms.year>=yr-1)&(storms.year<=yr+1)].copy()
    if cand.empty:return None
    # Prefer a storm name explicitly present in FEMA title.
    exact=cand[cand.storm_name.map(lambda s: s in title.upper() or norm_name(s)==nm)]
    if len(exact):
        exact=exact.assign(dd=(pd.to_datetime(exact.end)-dt).abs().dt.days)
        return exact.sort_values(["dd","max_wind_kt"],ascending=[True,False]).iloc[0]
    # Otherwise use nearest storm ending to declaration, but require <=30 days.
    cand=cand.assign(dd=(pd.to_datetime(cand.end)-dt).abs().dt.days)
    cand=cand[cand.dd<=30]
    if cand.empty:return None
    return cand.sort_values(["dd","max_wind_kt"],ascending=[True,False]).iloc[0]

def main():
    d=base.load_master()
    h=d[d.incidentType.astype(str).str.upper().isin(["HURRICANE","TROPICAL STORM","TYPHOON"])].copy()
    high=h[h.band.isin([3,4,5])].copy()
    titles=fetch_titles(high.disasterNumber.astype(int).tolist())
    high=high.merge(titles[["disasterNumber","declarationTitle","declarationDate","incidentBeginDate","incidentEndDate"]],on="disasterNumber",how="left",suffixes=("","_api"))
    storms=parse_hurdat()
    rows=[]
    for _,r in high.iterrows():
        s=match_storm(r,storms)
        q={
            "disasterNumber":int(r.disasterNumber),"state":str(r.state),"fyDeclared":int(r.fyDeclared),
            "incidentType":str(r.incidentType),"target":float(r.target),"band":int(r.band),
            "declarationTitle":str(r.get("declarationTitle","")),
        }
        if s is None:
            q.update({"matched":0,"storm_id":"","storm_name":"","max_wind_kt":np.nan,"min_pressure_mb":np.nan,"major_hours":np.nan,"hurricane_hours":np.nan,"ace_proxy":np.nan,"track_span_deg":np.nan})
        else:
            q.update({"matched":1,**{c:s[c] for c in ["storm_id","storm_name","max_wind_kt","min_pressure_mb","major_hours","hurricane_hours","ace_proxy","track_span_deg"]}})
        rows.append(q)
    out=pd.DataFrame(rows)
    out.to_csv(OUT/"high_hurricane_physical_features.csv",index=False)

    pair=out[out.band.isin([4,5])].copy()
    y=(pair.band==5).astype(int).to_numpy()
    aucs=[]
    for c in ["max_wind_kt","min_pressure_mb","major_hours","hurricane_hours","ace_proxy","track_span_deg"]:
        x=num(pair[c]); ok=x.notna()
        if ok.sum()>=4 and len(np.unique(y[ok]))==2 and x[ok].nunique()>1:
            auc=float(roc_auc_score(y[ok],x[ok]))
            aucs.append({"feature":c,"n":int(ok.sum()),"auc":auc,"orientation_free_auc":max(auc,1-auc)})
    aucs=sorted(aucs,key=lambda x:x["orientation_free_auc"],reverse=True)

    focus=out[out.disasterNumber.isin([4339,4671,4086,4399,4673,4715])].sort_values("disasterNumber")
    summary={
        "purpose":"Target-free hurricane physical-intensity signal audit for the non-Biological high-router failures.",
        "high_hurricane_like_count":int(len(out)),
        "matched_hurdat_count":int(out.matched.sum()),
        "top_boundary_auc":aucs,
        "focus_cases":focus.to_dict("records"),
        "guardrails":[
            "HURDAT2 storm intensity/track fields contain no FEMA funding information.",
            "FEMA declaration title is used only to match a disaster to a named storm.",
            "This is a signal audit before any router fitting.",
            "Any useful hurricane feature must subsequently be evaluated under leave-fiscal-year-out and then nested selection before final use."
        ]
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str),flush=True)

if __name__=="__main__":main()
