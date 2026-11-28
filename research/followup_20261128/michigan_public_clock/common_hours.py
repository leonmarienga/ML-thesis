#!/usr/bin/env python3
"""Descriptive same-valid-hour comparison from the immutable Michigan Bar pairs."""
import csv
import hashlib
import io
import json
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
import sys

EXPECTED="188c8dee74c31de9ce9cdd4807f7bef2870debf7fe5e3faf2c17591134e19435"


def verify(source):
    raw=source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=EXPECTED:
        raise ValueError("Archived stage pairs digest differs")
    grouped=defaultdict(dict)
    for row in csv.DictReader(io.StringIO(raw.decode("ascii"))):
        t=datetime.fromisoformat(row["valid_utc"])
        issue=datetime.fromisoformat(row["wmo_issue_utc"])
        if t in grouped[issue] or (t-issue).total_seconds()/60!=int(row["lead_minutes"]) or not t>issue:
            raise ValueError("Invalid lead or duplicate stage time")
        grouped[issue][t]=(Decimal(row["approved_gage_height_ft"]),Decimal(row["forecast_stage_ft"]))
    early,late=sorted(grouped)
    hours=sorted(grouped[early].keys() & grouped[late].keys())
    if len(hours)!=43 or hours[-1]-hours[0]!=timedelta(hours=42):
        raise ValueError("Unexpected common grid")
    if any(grouped[early][t][0]!=grouped[late][t][0] for t in hours):
        raise ValueError("Observed series disagree")
    threshold=Decimal("12.0")
    rows=[]
    for issue in (early,late):
        points=[grouped[issue][t] for t in hours]
        diff=[pred-actual for actual,pred in points]
        first_ob,first_pred=points[0]
        rows.append(dict(issue_utc=issue.isoformat(),
                         mae_ft=str((sum(abs(d) for d in diff)/len(hours)).quantize(Decimal(".001"))),
                         bias_ft=str((sum(diff)/len(hours)).quantize(Decimal(".001"))),
                         stage_at_first_common_hour_ft=str(first_pred),
                         minutes_lead_at_first_common_hour=int((hours[0]-issue).total_seconds()/60),
                         threshold_hits=sum(x>=threshold and y>=threshold for x,y in points),
                         threshold_false_positives=sum(x<threshold<=y for x,y in points)))
    return {"episode_count":1,"common_hours":len(hours),"first_common_utc":hours[0].isoformat(),
            "last_common_utc":hours[-1].isoformat(),"first_common_hour_observed_ft":str(first_ob),
            "threshold_ft":str(threshold),"observed_hours_at_or_above_threshold":sum(grouped[early][t][0]>=threshold for t in hours),
            "bulletins":rows}


if __name__=="__main__":
    print(json.dumps(verify(Path(sys.argv[1])),indent=2,sort_keys=True))
