#!/usr/bin/env python3
"""Flood USGS gauge feasibility with geographically matched NOAA counties.

Strictly observational research pilot: county-level gauge is NOT necessarily
within inundation footprint or causally linked to a FEMA declaration. Never
impute unmatched gauges. No disaster target used in matching. No funding
predictions are modified.

Input: original county-event matches from run 41326254945 and 971 FEMA master.
For feasibility, choose <=12 FLOOD disasters from 2019-2024, first county per
disaster; rank gauges only by available evidence, not funding amount.
Production scaling requires high-confidence county-to-basin matching and
station-location validation.
"""
from pathlib import Path
from io import StringIO
import json,time
import numpy as np,pandas as pd,requests
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"audit_inputs"/"flood_physical"
OUT=ROOT/"audit_outputs"/"flood_physical_feasibility"
OUT.mkdir(parents=True,exist_ok=True)
API="https://waterservices.usgs.gov/nwis/"
S=requests.Session();S.headers["User-Agent"]="Academic-FEMA-USGS-county-streamflow-audit/1.0"
def query(endpoint,params):
    for i in range(3):
        try:
            r=S.get(API+endpoint,params=params,timeout=35);r.raise_for_status()
            return r
        except requests.RequestException as e:
            print("RETRY",endpoint,i,str(e)[:100],flush=True);time.sleep(2*(i+1))
    return None
def sites(fips):
    r=query("site/",{"format":"rdb","countyCd":str(fips),"siteType":"ST",
                     "parameterCd":"00060","siteStatus":"all"})
    if r is None:return [],"site_api_unavailable"
    body="\n".join(line for line in r.text.splitlines() if not line.startswith("#"))
    try:
        t=pd.read_csv(StringIO(body),sep="\t",dtype=str)
        t=t[t.site_no.notna()&t.site_no.str.fullmatch(r"\d{8,15}")].copy()
        return t.site_no.drop_duplicates().tolist()[:10],"ok"
    except Exception as ex:return [],"site_parse_"+str(ex)[:60]
def daily(site,start,end):
    r=query("dv/",{"format":"json","sites":site,"startDT":str(start.date()),
                    "endDT":str(end.date()),"parameterCd":"00060",
                    "siteStatus":"all"})
    if r is None:return [],"daily_api_unavailable"
    try:
        data=r.json()["value"]["timeSeries"]
        v=[]
        for series in data:
            for block in series.get("values",[]):
                for item in block.get("value",[]):
                    try:
                        z=float(item["value"])
                        if z>=0:v.append((item["dateTime"],z))
                    except (KeyError,TypeError,ValueError):pass
        return v,"ok"
    except Exception as ex:return [],"daily_parse_"+str(ex)[:60]
def main():
    master=pd.read_excel(ROOT/"master_openfema_40plus.xlsx")
    assert len(master)==971
    county=pd.read_csv(SRC/"event_county_census_matches.csv",dtype={"fips":str})
    assert "disasterNumber" in county and "fips" in county
    flood=master[(master.incidentType=="Flood")&
                 (master.fyDeclared>=2019)&(master.fyDeclared<=2024)].copy()
    area=county[county.disasterNumber.isin(flood.disasterNumber)]
    # Predefined deterministic selection, independent of funding.
    chosen=area.sort_values(["disasterNumber","fips"]).drop_duplicates("disasterNumber").head(12)
    tests=flood.merge(chosen[["disasterNumber","fips"]],on="disasterNumber",how="inner",validate="one_to_one")
    rows=[];station_cache={}
    for _,case in tests.iterrows():
        start=pd.to_datetime(case.incidentBeginDate,errors="coerce",utc=True)
        end=pd.to_datetime(case.incidentEndDate,errors="coerce",utc=True)
        if pd.isna(start):continue
        if pd.isna(end):end=start+pd.Timedelta(days=7)
        end=min(end,start+pd.Timedelta(days=21))
        if end<start:end=start+pd.Timedelta(days=7)
        fips=case.fips
        if fips not in station_cache:station_cache[fips]=sites(fips)
        station_ids,site_status=station_cache[fips]
        item={"disasterNumber":int(case.disasterNumber),"county_fips":fips,
          "window_start":str(start),"window_end":str(end),
          "n_candidate_gauges":len(station_ids),"site_api_status":site_status,
          "selected_site":None,"daily_observations":0,
          "peak_flow_cfs":None,"median_flow_cfs":None,
          "discharge_data_status":"no_selected_site"}
        for sid in station_ids[:4]:
            vals,status=daily(sid,start,end)
            if vals:
                arr=np.array([v for _,v in vals],float)
                item.update({"selected_site":sid,"daily_observations":len(arr),
                    "peak_flow_cfs":float(arr.max()),"median_flow_cfs":float(np.median(arr)),
                    "discharge_data_status":status})
                break
            item["discharge_data_status"]=status
        rows.append(item)
        print("CASE",item["disasterNumber"],item["county_fips"],
              item["selected_site"],item["daily_observations"],flush=True)
    result=pd.DataFrame(rows)
    assert len(result)>0
    result.to_csv(OUT/"usgs_flood_county_gauge_pilot.csv",index=False)
    summary={"flood_records_2019_2024":len(flood),
        "with_county_matches":int(area.disasterNumber.nunique()),
        "selected_pilot_cases":len(result),
        "with_streamflow":int(result.peak_flow_cfs.notna().sum()),
        "study_status":"feasibility only; no funding-model performance results",
        "limitations":[
         "NOAA reported event county is not an inundation footprint.",
         "Selected USGS stream gauge may not be hydrographically connected to impacted areas.",
         "Peak discharge alone does not establish flood stage, relative anomaly or building exposure.",
         "Historical gauge data may be revised; no time-of-publication snapshots verified.",
         "Dataset selection is outcome-blind but geographically and temporally constrained.",
         "NASA rainfall/IMERG is not yet extracted; satellite flood masks not yet matched."
        ]}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2))
    print("BEGIN_USGS_FEASIBILITY\n"+json.dumps(summary,indent=2)+"\nEND_USGS_FEASIBILITY",flush=True)
if __name__=="__main__":main()
