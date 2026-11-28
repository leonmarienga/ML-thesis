#!/usr/bin/env python3
"""Exact-time, single-event verification of two public MHBC1 stage forecasts."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import re

UTC=timezone.utc
SPECS=[("20221228","281713","202212281645","202212281645"),("20221231","311700","202212311657","202212311657")]
SIGNED_SOURCES={
    "iem-december.txt":"21e1ffb39f8a9f704407216703354174578cdd78fbe0f49781fbb2b7d8e9160e",
    "usgs-stage-modern.json":"5b4b099609bcd66193bff71890936d245759ef55a46c6a91f5f6e5237e1491b7",
    "MHBC1-28.png":"ccfff874a6507679b78a5c97146bf8f798e678e3a87221dd6da6f5a96b9629cd",
    "MHBC1-31.png":"f40f9e4a5fa31ef6b01dfac182e7a9c22f7acbf18f8b5e2b75d1b626238258ce",
}


def need(ok,msg):
    if not ok: raise ValueError(msg)


def dt(s):
    return datetime.strptime(s,"%Y%m%d%H%M").replace(tzinfo=UTC)


def rows_for(text,date,headline,creation,graphic):
    bulletins=[b for b in text.split("\x01") if re.search(rf"(?m)^FGUS56 KRSA {headline}\s*$",b)]
    need(len(bulletins)==1,"Missing or ambiguous WMO product "+headline)
    bulletin=bulletins[0]
    need(not re.search(rf"(?m)^FGUS56 KRSA {headline}[ \t]+[A-Z]{{3}}",bulletin),"Unexpected correction indicator")
    need("ALL TIMES IN SHEF ENCODED MESSAGES BELOW ARE UTC" in bulletin,"Missing UTC header")
    version=re.search(rf"(?m)^\.ER MHBC1 {date} Z DH(\d{{2}})/DC({creation})/DUE/HGIFE/DIH01\s*$",bulletin)
    need(version is not None and len(re.findall(r"(?m)^\.ER MHBC1 ",bulletin))==1,"Channel or format mismatch")
    need(creation==graphic and headline[:2]==date[-2:],"Graphics/WMO date metadata mismatch")
    issued=dt(date+headline[2:])
    start=dt(date+version[1]+"00")
    cont=[]
    for ln in bulletin[version.end():].strip().splitlines():
        if not ln.startswith(".ER"): break
        m=re.fullmatch(r"\.ER(\d+)\s+([\d./ ]+)",ln)
        need(m is not None and int(m[1])==len(cont)+1,"Unexpected .ER continuation")
        vals=[Decimal(v.strip()) for v in m[2].strip().rstrip("/").split("/")]
        cont.append(vals)
    raw=[v for r in cont for v in r]
    need(len(cont)==12 and all(len(a)==10 for a in cont[:-1]) and len(cont[-1])==5 and len(raw)==115,"Unexpected forecast count")
    timestamps=[start+timedelta(hours=i) for i in range(len(raw))]
    need(dt(creation)<issued<start,"A valid forecast must be future relative to the bulletin")
    return issued,list(zip(timestamps,raw))


def run(src,out,refetched_usgs=False):
    fil={}
    for n,expected in SIGNED_SOURCES.items():
        fil[n]=(src/n).read_bytes()
        if n!="usgs-stage-modern.json" or not refetched_usgs:
            need(sha256(fil[n]).hexdigest()==expected,"Changed source "+n)
    archive=fil["iem-december.txt"].decode("ascii")
    obs=json.loads(fil["usgs-stage-modern.json"])
    need(obs["numberReturned"]==len(obs["features"])==757 and not any(r["rel"]=="next" for r in obs["links"]),"Unexpected pagination")
    pin_fields=("monitoring_location_id","parameter_code","statistic_id","unit_of_measure",
                "approval_status","qualifier","time_series_id","time","value")
    rows=sorted(tuple(f["properties"][k] for k in pin_fields) for f in obs["features"])
    stable=json.dumps(rows,separators=(",",":")).encode("ascii")
    need(sha256(stable).hexdigest()=="3edb660b7f2260fd8427904235a1d47981cc5d2139d274cb02222b93c78901e0",
         "Refetch did not match the original USGS observation facts")
    measured={}
    for x in obs["features"]:
        p=x["properties"]
        asked=("USGS-11335000","00065","00011","ft","Approved",None,"130b9af0b77c45e68b19e403b3833149")
        keys=("monitoring_location_id","parameter_code","statistic_id","unit_of_measure","approval_status","qualifier","time_series_id")
        need(tuple(p[k] for k in keys)==asked,"USGS series identity mismatch")
        t=datetime.fromisoformat(p["time"]);need(t not in measured,"Duplicate USGS observation")
        measured[t]=Decimal(p["value"])
    need(all(v-u==timedelta(minutes=15) for u,v in zip(sorted(measured),sorted(measured)[1:])),"USGS 15-minute grid not complete")
    flood=Decimal("12.0")
    first_obs=min(t for t,y in measured.items() if y>=flood)
    need(measured[first_obs-timedelta(minutes=15)]<flood,"Observed crossing is left-censored")
    results=[]
    out.mkdir(exist_ok=True,parents=True)
    with (out/"stage_pairs.csv").open("w",newline="") as f:
        w=csv.writer(f,lineterminator="\n")
        w.writerow(["wmo_issue_utc","valid_utc","lead_minutes","forecast_stage_ft","approved_gage_height_ft"])
        for date,headline,creation,graphic in SPECS:
            issue,trace=rows_for(archive,date,headline,creation,graphic)
            need(all(t in measured for t,y in trace),"Missing exact-time match")
            times=[t for t,y in trace]
            residual=[y-measured[t] for t,y in trace]
            first_fc=min(t for t,y in trace if y>=flood)
            local_obs=[t for t in times if measured[t]>=flood]
            of=[(t,measured[t]) for t in times]
            for t,y in trace: w.writerow([issue.isoformat(),t.isoformat(),int((t-issue).total_seconds()/60),y,measured[t]])
            results.append({
                "bulletin_wmo_issue_utc":issue.isoformat(),"graphic_posted_and_shef_created_utc":dt(graphic).isoformat(),
                "matched_future_hours":len(times),"first_valid_utc":times[0].isoformat(),"last_valid_utc":times[-1].isoformat(),
                "first_forecast_hour_at_or_above_12_ft_utc":first_fc.isoformat(),
                "first_observed_at_or_above_12_ft_in_forecast_grid_utc":local_obs[0].isoformat(),
                "issue_preceded_observed_crossing":issue<first_obs,
                "actual_crossing_minutes_after_wmo_issue":int((first_obs-issue).total_seconds()/60),
                "forecast_crossing_minutes_after_actual_15m_crossing":int((first_fc-first_obs).total_seconds()/60) if issue<first_obs else None,
                "mae_ft":float(sum(abs(y) for y in residual)/len(residual)),
                "bias_ft":float(sum(residual)/len(residual)),
                "forecast_peak_ft":float(max(y for t,y in trace)),
                "observed_peak_at_forecast_hours_ft":float(max(y for t,y in of)),
            })
    payload={"observation_count":len(measured),"first_actual_15m_crossing_12ft_utc":first_obs.isoformat(),"episodes":1,"cases":results}
    (out/"stage_summary.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--refetched-usgs",action="store_true",help="allow different response envelope only; data digest remains enforced")
    a=p.parse_args();run(a.source,a.output,a.refetched_usgs)
