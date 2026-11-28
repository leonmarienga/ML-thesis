"""Offline-only lock of the 11 previously identified RFC-guidance pairs."""
import csv,json,re,hashlib
from collections import Counter
from datetime import datetime,timezone
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlencode

ROOT=Path(__file__).resolve().parent
UTC=timezone.utc
def check(ok,message):
    if not ok:
        raise ValueError(message)
def t(x):
    d=datetime.fromisoformat(x)
    check(d.tzinfo is not None and d.utcoffset().total_seconds()==0,'Clock must be explicitly UTC')
    return d
def z(d):
    return d.isoformat(timespec='seconds').replace('+00:00','Z')
def read_json(p):
    return json.loads(p.read_text())
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def plan(root=ROOT):
    d=read_json(root/'design.json')
    src=root/'source'
    rfiles={n.split('  ',1)[1]: n.split('  ',1)[0] for n in (root/'SOURCE_SHA256SUMS').read_text().splitlines()}
    check(all(sha(root/f)==h for f,h in rfiles.items()),'Source hash mismatch')
    check(rfiles.get('source/fixed-lead-dispositions.csv')=='b703dbe3364f4e0352ed6ef9f47825c23daa2ba184c525d5050f90568d3dcc4a','Unapproved disposition corpus')
    intended={k:set(v) for k,v in d['cohort'].items()}
    check(set(intended)=={'KY2022','OH2024','HELENE2024'},'Episode list changed')
    check(d['no_outcomes_accessed'] and d['acquisition_status']=='blocked_pending_root_freeze_verification_and_separate_authorization','Stay locked')
    pt=read_json(src/'protocol.json')
    check(d['lead_band_minutes_half_open']==pt['lead_minutes_half_open']==[1440,1800],'Lead band changed')
    strict=read_json(src/'phase2-plan.json')
    check(strict['status']=='blocked_public_clock_gate' and strict['eligible_rows']==0 and strict['requests']==[],'Original public gate changed')
    prior=read_json(src/'audit-summary.json')
    check(prior['strict_eligible_rows']==0 and prior['held_observations_accessed'] is False,'Original archive assumptions changed')
    md=read_json(src/'metadata-digest.json')['episodes']
    want={(x['id'],z(t(day+'T00:00:00Z').replace(hour=hour)))
           for x in pt['episodes'] for day in x['target_dates'] for hour in pt['valid_synoptic_utc_hours']}
    pairs=[]; seen=set(); selected=set()
    with (src/'fixed-lead-dispositions.csv').open(newline='') as f:
        for r in csv.DictReader(f):
            event=r['episode']; ts=z(t(r['target_utc']))
            key=(event,ts)
            check(key in want and key not in seen,'Planned timestamp absent, extra or duplicate')
            seen.add(key)
            check(r['strict_status']=='BLOCKED_NO_PUBLIC_HGIFE','Contrary strict disposition')
            if r['candidate_status']!='ARCHIVE_COMPATIBLE_RFC_GUIDANCE_ONLY':
                check(r['candidate_status']=='BLOCKED_NO_CORROBORATED_FIXED_BAND_CANDIDATE' and r['later_candidate_count']=='0',
                      'Conflicting or unexpected disposition')
                continue
            check(ts in intended[event] and int(r['later_candidate_count'])>=1,'Selection differs from frozen eligible tuple')
            selected.add(key)
            s=d['site_lock'][event]
            check([r['nws_lid'],r['usgs_id'],r['series'],r['channel']]==s,'Forecast/site/series lock violated')
            u=md[event]['usgs_series']
            check([u[k] for k in ('monitoring_location_id','id','parameter_code','statistic_id','unit_of_measure')]==[s[1],s[2],'00065','00011','ft'],
                  'USGS series metadata drift')
            check(t(u['begin_utc'])<=t(ts)<=t(u['end_utc']),'Target outside declared time series')
            check(md[event]['noaa_site']['usgsId']=='%s'%s[1][5:] and md[event]['noaa_site']['lid']==s[0],'NOAA exact site mismatch')
            w,m,h=map(t,(r['wmo'],r['dc'],r['hml_issued']))
            issue=max(w,m,h)
            check(issue<t(ts),'Nonfuture forecast')
            lead=(t(ts)-issue).total_seconds()/60
            check(1440<=lead<1800 and lead==int(lead)==int(r['lead_minutes']),'Invalid or nonfrozen lead')
            check(Decimal(r['stage_ft']).is_finite(),'Invalid stage')
            check(r['product_link'].startswith('https://mesonet.agron.iastate.edu/p.php?pid='+w.strftime('%Y%m%d%H%M')+'-'),'WMO link mismatch')
            pairs.append(dict(pair_id=f'{event}-{ts.replace(":","").replace("-","")}',episode=event,target_utc=ts,
                nws_lid=s[0],usgs_id=s[1],series=s[2],channel=s[3],wmo=z(w),shef_dc=z(m),hml_issue=z(h),
                analysis_vintage_clock=z(issue),lead_minutes=int(lead),stage_guidance_ft=r['stage_ft'],
                candidates_at_target=int(r['later_candidate_count']),product_url=r['product_link']))
    check(seen==want and len(want)==24,'Missing parent planned rows')
    check(selected=={(k,t0) for k,ss in intended.items() for t0 in ss},'Missing eligible row')
    counts=dict(Counter(x['episode'] for x in pairs))
    check(counts=={'KY2022':4,'OH2024':2,'HELENE2024':5},'Event count drift')
    queries=[]
    for p in pairs:
        check(int(p['target_utc'][:4])<2025 and re.fullmatch(r'USGS-\d{8}',p['usgs_id']),'Site/time boundary')
        fields={'monitoring_location_id':p['usgs_id'],'parameter_code':'00065',
                'datetime':p['target_utc'],'limit':10,'f':'json'}
        queries.append({'pair_id':p['pair_id'],'would_get':d['future_request_base']+'?'+urlencode(fields),
                        'expected_time_series_id':p['series']})
    return pairs,{'query_status':'not_executed_pending_root_review','prospective_only':True,
        'truth_requests_executed':0,'strict_public_eligible':0,'guidance_pair_count':len(pairs),
        'independent_event_units':3,'guidance_pairs_by_event':counts,'future_requests':queries}
def main():
    pairs,qs=plan()
    with (ROOT/'frozen_pairs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,pairs[0].keys(),lineterminator='\n')
        w.writeheader();w.writerows(pairs)
    (ROOT/'prospective_query_plan.json').write_text(json.dumps(qs,indent=2,sort_keys=True)+'\n')
if __name__=='__main__': main()
