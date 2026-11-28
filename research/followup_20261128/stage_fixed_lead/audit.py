"""Offline replay of the outcome-blind lead-time and source gate."""
import csv,json,re
from datetime import datetime,timedelta,timezone
from decimal import Decimal, InvalidOperation
from collections import defaultdict,Counter
from forecast_probe import D,E
UTC=timezone.utc
def time(s):
    a=datetime.fromisoformat(s)
    return a if a.tzinfo else a.replace(tzinfo=UTC)
def lead(t,a): return int((t-a).total_seconds()/60)
def trace(b, office):
    h=re.fullmatch(r'\.ER (\w+) (\d{4}|\d{8}) Z (\S+)',b['forecast_header'])
    assert h
    codes=h[3].strip('/').split('/')
    channel=next(c for c in codes if c.startswith('HGI'))
    stride=next(c[3:] for c in codes if c.startswith('DIH'))
    assert int(stride)==6
    dh=next(int(c[2:]) for c in codes if re.fullmatch(r'DH\d{2}',c))
    start=time(b['dc'])
    if len(h[2])==8:
        start=datetime.strptime(h[2],'%Y%m%d').replace(tzinfo=UTC)
    else:
        vals=[(start+timedelta(days=k)).replace(hour=0,minute=0) for k in (-1,0,1)
              if (start+timedelta(days=k)).strftime('%m%d')==h[2]]
        assert len(vals)==1
        start=vals[0]
    start=start.replace(hour=dh,minute=0)
    result={}
    for i,ln in enumerate(b['forecast_continuations']):
        m=re.fullmatch(r'\.E(?:R)?(\d+)\s+:\s*(\w+(?: \d{2})?)\s*:\s*(.*)',ln)
        assert m and int(m[1])==i+1
        body=m[3].strip()
        raw=body.split('/')
        filled=[(i,x.strip()) for i,x in enumerate(raw) if x.strip()]
        assert filled and len(filled)<=4 and [i for i,x in filled]==list(range(filled[0][0],filled[-1][0]+1))
        assert i==0 or i==len(b['forecast_continuations'])-1 or len(filled)==4
        for _,v in filled:
            t=start+timedelta(hours=6*len(result))
            # Ohio table begins 00Z; KORN table starts at 12 UTC, with the
            # 00Z/06Z columns belonging to the next UTC date.
            labelt=t-timedelta(hours=12 if office=='KORN' else 0)
            assert m[2]==labelt.strftime('%m%d' if office=='KORN' else '%b %d')
            result[t]=Decimal(v)
    return channel,result
def match_hml(tr, h, d, w):
    found=[]
    for issued,vector in h.items():
        # Corroboration only: require substantial shared trace and exact values.
        shared=set(tr)&set(vector)
        if len(shared)>=8 and max(abs(lead(issued,d)),abs(lead(issued,w)))<=15 and all(tr[t]==vector[t] for t in shared):
            found.append((issued,shared))
    return found

def run():
    protocol=json.loads((D/'protocol.json').read_text())
    records=[]; out={}
    for ev in protocol['episodes']:
        k=ev['id'];lid,office,pil,_,_=E[k]
        j=json.loads((D/f'forecast-{k}.json').read_text())
        meta=json.loads((D/f'metadata-{lid}.json').read_text())
        site=meta['noaa']['metadata']
        assert site['usgsId']==ev['usgs']
        expected='USGS-'+ev['usgs']
        obs=[p for f in meta['usgs']['metadata']['features'] if (p:=f['properties'])['statistic_id']=='00011'
             and p['monitoring_location_id']==expected and p['parameter_code']=='00065' and p['unit_of_measure']=='ft']
        assert len(obs)==1
        obs=obs[0]
        h=defaultdict(dict)
        for a in j['hml_forecast_rows']:
            assert (a['primaryname'],a['primaryunits'],a['station'])==('Stage','ft',lid)
            issued,t,v=time(a['issued[UTC]']),time(a['forecast_valid[UTC]']),Decimal(a['primary_value'])
            assert t not in h[issued] or v==h[issued][t]
            assert t>issued
            h[issued][t]=v
        decoded=[]
        for b in j['wmo_forecast_only']:
            try:
                c,zz=trace(b,office)
            except (AssertionError,InvalidOperation,AttributeError,StopIteration) as e:
                decoded.append((b,'parse_blocked',None,{},[])); continue
            w,d=time(b['wmo']),time(b['dc'])
            assert all(t>max(w,d) for t in zz)
            decoded.append((b,'decoded',c,zz,match_hml(zz,h,d,w)))
        # This feasibility replay does not contain a positive public-stage decoder.
        # Changed/ambiguous corpora fail closed instead of inheriting this disposition.
        if not decoded or any(status!='decoded' or c not in {'HGIFF','HGIFFZZ'}
                           or not b['guidance_disclaimer'] for b,status,c,_,_ in decoded):
            raise ValueError('Corpus may contain a contrary public-stage path or undecoded release; review before proceeding')
        matched=sum(len(x[4])==1 for x in decoded)
        for day in ev['target_dates']:
            for hour in protocol['valid_synoptic_utc_hours']:
                t=time(day).replace(hour=hour)
                assert time(obs['begin_utc'])<=t<=time(obs['end_utc'])
                candidates=[]
                for b,status,c,zz,a in decoded:
                    if status!='decoded' or len(a)!=1 or t not in a[0][1] or b['correction']:
                        continue
                    dd,ww,hh=time(b['dc']),time(b['wmo']),a[0][0]
                    clock=max(dd,ww,hh)
                    if 1440<=lead(t,clock)<1800:
                        candidates.append((clock,b,c,zz[t],dd,ww,hh))
                candidates.sort(key=lambda x: x[0],reverse=True)
                item={'episode':k,'nws_lid':lid,'usgs_id':expected,'series':obs['id'],'target_utc':t.isoformat(),
                    'strict_status':'BLOCKED_NO_PUBLIC_HGIFE',
                    'candidate_status':'BLOCKED_NO_CORROBORATED_FIXED_BAND_CANDIDATE',
                    'later_candidate_count':len(candidates)}
                if candidates:
                    a,b,c,v,dd,ww,hh=candidates[0]
                    if len({x[3] for x in candidates if x[0]==a})!=1:
                        item['candidate_status']='BLOCKED_CONFLICT'
                    else:
                        item.update(candidate_status='ARCHIVE_COMPATIBLE_RFC_GUIDANCE_ONLY',lead_minutes=lead(t,a),wmo=ww.isoformat(),
                            dc=dd.isoformat(),hml_issued=hh.isoformat(),stage_ft=str(v),channel=c,product_link=b['public_archive_link'])
                records.append(item)
        out[k]={'forecasts_with_encoded_site_stage':len(decoded),'shef_channel_counts':dict(Counter(x[2] or x[1] for x in decoded)),
            'wmo_hml_uniquely_corroborated':matched,'guidance_disclaimers':sum(x[0]['guidance_disclaimer'] for x in decoded),
            'hml_issue_vintages':len(h),'hml_forecast_rows':len(j['hml_forecast_rows']),
            'metadata_series_id':obs['id']}
    for k in E:
        out[k]['target_dispositions']=dict(Counter(x['candidate_status'] for x in records if x['episode']==k))
    out['strict_eligible_rows']=sum(not r['strict_status'].startswith('BLOCKED') for r in records)
    out['held_observations_accessed']=False
    (D/'audit-summary.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    fields=['episode','nws_lid','usgs_id','series','target_utc','strict_status','candidate_status','later_candidate_count',
           'wmo','dc','hml_issued','lead_minutes','stage_ft','channel','product_link']
    with (D/'fixed-lead-dispositions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader();w.writerows(records)
if __name__=='__main__': run()
