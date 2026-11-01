#!/usr/bin/env python3
"""Action-time-screened FEMA MissionAssignments for ALL 971 thesis disasters.

dateReceived is creation time of a mission-assignment action, NOT proof that
today's API row was available in identical form on that date.
Outputs are RECONSTRUCTED AS-OF PROXIES, not versioned historical snapshots.

No financial fields or funding targets are inspected in this script.
Strictly exclude all action rows after the indicated cutoff; never use latest
mission assignment state based on eventual amendment/finalized records.
"""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
import requests
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"audit_outputs"/"mission_asof_971"
OUT.mkdir(parents=True,exist_ok=True)
API="https://www.fema.gov/api/open/v2/MissionAssignments"
COLS=["disasterNumber","maId","maAmendNumber","actionId","dateReceived",
      "agencyId","supportFunction","priority","maType","lastRefresh"]
PAGE=4000
def fetch_actions():
    session=requests.Session()
    session.headers["User-Agent"]="Disaster-relief-thesis-MissionAssignments-asof-audit/1.0"
    all_rows=[]
    skip=0
    while True:
        params={"$top":PAGE,"$skip":skip,"$orderby":"actionId asc",
                "$select":",".join(COLS)}
        for k in range(5):
            try:
                r=session.get(API,params=params,timeout=80)
                r.raise_for_status()
                payload=r.json()
                batch=payload["MissionAssignments"]
                break
            except Exception as e:
                print("RETRY",skip,k,str(e)[:160],flush=True)
                if k==4:raise
                time.sleep(2**k)
        all_rows.extend(batch)
        skip+=len(batch)
        print("FETCHED_MA",len(all_rows),flush=True)
        if len(batch)<PAGE:break
        if skip>250000:raise ValueError("FEMA API pagination safety limit")
    return pd.DataFrame(all_rows)
def rollup(master,events):
    events=events.copy()
    events["disasterNumber"]=pd.to_numeric(events.disasterNumber,errors="coerce").astype("Int64")
    events=events[events.disasterNumber.isin(master.disasterNumber)].copy()
    events["date_received"]=pd.to_datetime(events.dateReceived,errors="coerce",utc=True)
    events["last_refresh"]=pd.to_datetime(events.lastRefresh,errors="coerce",utc=True)
    events["amend_no"]=pd.to_numeric(events.maAmendNumber,errors="coerce")
    events["action_id"]=pd.to_numeric(events.actionId,errors="coerce")
    events["maId"]=events.maId.astype(str)
    # Parse cutoff from date-only declarationDate; no extra time precision invented.
    dates=pd.to_datetime(master.declarationDate,utc=True,errors="coerce").dt.normalize()
    result=master[["disasterNumber"]].copy()
    result["t0_utc"]=dates
    result["t72_utc"]=dates+pd.Timedelta(hours=72)
    events=events.merge(result,on="disasterNumber",how="left",validate="many_to_one")
    frames=[]
    for label in ["t0","t72"]:
        boundary=events[label+"_utc"]
        valid=events.date_received.notna()&boundary.notna()&(events.date_received<=boundary)
        active=events.loc[valid].copy()
        # Keep amendment/action revision that had the most recent dateReceived
        # per MA ID *within cutoff*, not eventual latest amendment.
        active=active.sort_values(["disasterNumber","maId","date_received","amend_no","action_id"])
        latest=active.groupby(["disasterNumber","maId"],dropna=False).tail(1)
        sums=latest.groupby("disasterNumber").agg(
            mission_count=("maId","nunique"),
            agency_count=("agencyId","nunique"),
            esf_count=("supportFunction","nunique"),
            ma_type_count=("maType","nunique"),
            priority_count=("priority","nunique")).reset_index()
        sumactions=active.groupby("disasterNumber").agg(
            observed_action_count=("actionId","size"),
            updated_after_cutoff=("last_refresh",lambda s:int((s>active.loc[s.index,label+"_utc"]).sum())),
            amendment_actions=("amend_no",lambda s:int((s>0).sum())),
        ).reset_index()
        sums=sums.merge(sumactions,on="disasterNumber",how="outer")
        sums=sums.rename(columns={c:f"ma_{label}_{c}" for c in sums.columns if c!="disasterNumber"})
        result=result.merge(sums,on="disasterNumber",how="left",validate="one_to_one")
    numeric=[c for c in result.columns if c.startswith("ma_")]
    result[numeric]=result[numeric].fillna(0).astype("int64")
    totals=events.groupby("disasterNumber").agg(
        ma_total_action_rows=("maId","size"),
        ma_missing_date_received=("date_received",lambda s:int(s.isna().sum())),
        ma_post_t0_actions=("date_received",lambda s:int((s>events.loc[s.index,"t0_utc"]).sum())),
        ma_post_t72_actions=("date_received",lambda s:int((s>events.loc[s.index,"t72_utc"]).sum())),
        ma_refresh_after_t0=("last_refresh",lambda s:int((s>events.loc[s.index,"t0_utc"]).sum())),
        ma_original_actions=("amend_no",lambda s:int((s==0).sum())),
        ma_amend_actions=("amend_no",lambda s:int((s>0).sum())),
    ).reset_index()
    result=result.merge(totals,on="disasterNumber",how="left",validate="one_to_one")
    audit={"fema_api_action_rows":len(events),"fema_disasters_with_maa":events.disasterNumber.nunique(),
        "date_received_present":int(events.date_received.notna().sum()),
        "actions_before_or_at_t0":int((events.date_received<=events.t0_utc).sum()),
        "actions_before_or_at_t72":int((events.date_received<=events.t72_utc).sum()),
        "actions_after_t0":int((events.date_received>events.t0_utc).sum()),
        "action_lastRefresh_after_t0":int((events.last_refresh>events.t0_utc).sum()),
        "disasters_with_t0_missions":int((result.ma_t0_mission_count>0).sum()),
        "disasters_with_t72_missions":int((result.ma_t72_mission_count>0).sum()),
        "disasters_without_t0_missions":int((result.ma_t0_mission_count==0).sum()),
        "timing_only_caution":"dateReceived is created-action timestamp, and current lastRefresh may postdate historical cutoff; this is an action-time proxy, not an independently versioned as-of snapshot."}
    return result,events,audit

def main():
    master=pd.read_excel(ROOT/"master_openfema_40plus.xlsx")
    assert len(master)==971 and master.disasterNumber.nunique()==971
    master=master[["disasterNumber","declarationDate","incidentType","fyDeclared"]].copy()
    ma=fetch_actions()
    print("MISSION_COLS",ma.columns.tolist(),flush=True)
    required=set(COLS)
    assert required<=set(ma.columns),f"Missing FEMA columns: {required-set(ma.columns)}"
    features,actions,stats=rollup(master,ma)
    features.drop(columns=["t0_utc","t72_utc"]).to_csv(OUT/"mission_actions_t0_t72_all971.csv",index=False)
    features[["disasterNumber","t0_utc","t72_utc","ma_t0_mission_count",
        "ma_t72_mission_count","ma_total_action_rows","ma_post_t0_actions",
        "ma_refresh_after_t0","ma_missing_date_received"]].to_csv(
        OUT/"mission_temporal_validity_by_disaster.csv",index=False)
    # Disaggregate by *incident type*, not funding (source extraction fully target-blind).
    diag=features.merge(master[["disasterNumber","incidentType","fyDeclared"]],
      on="disasterNumber",validate="one_to_one")
    by_hazard=diag.assign(has_t0=diag.ma_t0_mission_count>0,has_t72=diag.ma_t72_mission_count>0).groupby("incidentType").agg(
        disaster_count=("disasterNumber","count"),t0_any=("has_t0","sum"),t72_any=("has_t72","sum"),
        mean_t0_missions=("ma_t0_mission_count","mean"),mean_t72_missions=("ma_t72_mission_count","mean")
    ).reset_index().sort_values("disaster_count",ascending=False)
    by_hazard.to_csv(OUT/"mission_coverage_by_incident_type.csv",index=False)
    stats.update({"total_master":len(master),"by_hazard":by_hazard.to_dict("records"),
       "retrieved_at_utc":datetime.now(timezone.utc).isoformat(),
       "financial_fields_used":False,
       "funding_amount_used_for_source_matching":False,
       "decision_cutoffs":"declarationDate normalized 00:00 UTC; +72 hours",
       "not_versioned":True,
       "field_provenance":"FEMA OpenFEMA v2 MissionAssignments dateReceived and maAmendNumber",
       "no_post_t0_future_mission_action_rows_in_t0_aggregates":True})
    (OUT/"summary.json").write_text(json.dumps(stats,indent=2,default=str),encoding="utf-8")
    print("BEGIN_MISSION_ASOF_SUMMARY",flush=True)
    print(json.dumps(stats,indent=2,default=str),flush=True)
    print("END_MISSION_ASOF_SUMMARY",flush=True)
if __name__=="__main__":main()
