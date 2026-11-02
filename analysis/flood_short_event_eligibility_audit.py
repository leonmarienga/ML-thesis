#!/usr/bin/env python3
"""Audit fixed 3-day Daymet eligibility without changing model experiment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
IN=ROOT/"audit_inputs"/"flood_short_event"
OUT=ROOT/"audit_outputs"/"flood_short_event"
OUT.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(IN/"all_92_flood_daymet_rainfall.csv")
assert len(d)==92 and d.disasterNumber.nunique()==92
failed=d[d.rainfall_status=="insufficient_event_daily_coverage"].copy()
assert len(failed)==18
assert failed.expected_event_days.between(1,2).all()
assert (failed.available_event_days==failed.expected_event_days).all()
assert failed.event_coverage.eq(1.0).all()
audit=failed[["disasterNumber","fyDeclared","expected_event_days",
 "available_event_days","event_coverage","rainfall_status","noaa_point_lat",
 "noaa_point_lon","noaa_event_id"]].copy()
audit["could_pass_duration_adaptive_completeness"]=True
audit["complete_rainfall_features_recovered"]=False
audit.to_csv(OUT/"short_duration_exclusions.csv",index=False)
n_geolocated=int(d.noaa_point_lat.notna().sum())
observed=int(d.rainfall_available.sum())
report={
 "n_fema_floods":len(d),"n_geolocated":n_geolocated,
 "original_usable_rainfall_n":observed,
 "excluded_due_to_fixed_min_3_days":18,
 "of_excluded_complete_daily_coverage":18,
 "excluded_1_day":int((failed.expected_event_days==1).sum()),
 "excluded_2_days":int((failed.expected_event_days==2).sum()),
 "potential_coverage_if_eligibility_revised_and_features_recomputed":87,
 "actual_feature_coverage_under_existing_protocol":observed,
 "any_new_features_computed":False,
 "any_model_refitted":False,
 "interpretation":"18 geolocated short-event exclusions have 100% observed event-window coverage yet fail fixed >=3 day threshold; do not count as newly measured valid features until recomputation and same-year model re-evaluation."
}
(OUT/"summary.json").write_text(json.dumps(report,indent=2))
print("BEGIN_SHORT_FLOOD_AUDIT")
print(json.dumps(report,indent=2))
print("END_SHORT_FLOOD_AUDIT")
