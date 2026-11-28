"""Dry-run future exact-USGS requests; never opens the network or outcomes."""
import csv,json
from pathlib import Path
from urllib.parse import urlencode
from collections import Counter
root=Path(__file__).resolve().parent
records=list(csv.DictReader((root/'fixed-lead-dispositions.csv').open()))
eligible=[x for x in records if x['strict_status']=='PUBLIC_STAGE_MATCH_VERIFIED']
group=Counter(x['episode'] for x in eligible)
unlocked=len(group)>=2 and all(x['wmo'] and x['series'] for x in eligible)
base='https://api.waterdata.usgs.gov/ogcapi/v0/collections/continuous/items'
qs=[]
if unlocked:
    for r in eligible:
        dt=r['target_utc'].replace('+00:00','Z')
        qs.append({'episode':r['episode'],'expected_series':r['series'],'exact_time':dt,'url':base+'?'+urlencode(
            dict(f='json',monitoring_location_id=r['usgs_id'],parameter_code='00065',datetime=dt,limit=10))})
print(json.dumps({'action':'query_plan_only_no_network','status':'ready' if unlocked else 'blocked_public_clock_gate',
    'eligible_events':len(group),'eligible_rows':len(eligible),'requests':qs},indent=2)+'\n',end='')
