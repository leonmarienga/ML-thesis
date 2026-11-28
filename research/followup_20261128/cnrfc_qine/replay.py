"""Offline test of same-site, same-variable observation matching. No network."""
import csv, datetime as dt, email.utils, hashlib, io, json, pathlib, statistics, zipfile
from decimal import Decimal as D

R=pathlib.Path(__file__).resolve().parent
A=R/'raw'
UTC=dt.timezone.utc
LO=dt.datetime(2022,12,27,tzinfo=UTC)
HI=dt.datetime(2023,1,3,tzinfo=UTC)
CREST=dt.datetime(2022,12,31,18,tzinfo=UTC)
SELECTED='MHBC1'
GATE={
 'MHBC1':{'status':'comparable_documented_same_site_discharge_target','target':'USGS-11335000/00060','caveat':'CNRFC QINE is adjusted and may use gauge past, not an independent observation; future approved stage-derived discharge is target only.'},
 'CMPC1':{'status':'blocked_naturalized_inflow_vs_raw_gauge','target':None,'caveat':'Pardee Dam natural inflow adjusted to daily full natural flow; no raw discharge substitute.'},
 'MCNC1':{'status':'blocked_no_empirical_rating_curve','target':None,'caveat':'McConnell stage is not flow; CNRFC documents no empirical discharge rating.'},
 'MCNC1L':{'status':'blocked_local_component_and_not_in_returned_zip','target':None,'caveat':'Excludes upstream contributions from MHBC1 and OTTC1; no total gauge-flow substitute.'},
}
def stamp(x): return x.isoformat().replace('+00:00','Z')
def parse(s):
 x=dt.datetime.fromisoformat(s.replace('Z','+00:00'))
 return x.replace(tzinfo=UTC) if x.tzinfo is None else x.astimezone(UTC)
def num(s):
 x=D(s)
 assert x.is_finite()
 return x
def calc(pairs):
 if not pairs: return {'count':0}
 return {
  'count':len(pairs),
  'mae_cfs':str(round(statistics.mean(num(p['abs_diff_cfs']) for p in pairs),3)),
  'max_abs_diff_cfs':str(max(num(p['abs_diff_cfs']) for p in pairs)),
  'first_valid_utc':pairs[0]['valid_utc'],'last_valid_utc':pairs[-1]['valid_utc'],
 }
def main():
 manifest={x['asset']:x for x in json.loads((A/'MANIFEST.json').read_bytes())}
 for name, row in manifest.items():
  raw=(A/name).read_bytes()
  assert row['status']==200 and row['bytes']==len(raw) and row['sha256']==hashlib.sha256(raw).hexdigest()
 raw=json.loads((A/'usgs_11335000.geojson').read_bytes())
 assert raw['type']=='FeatureCollection' and raw['numberReturned']==len(raw['features'])<1600 and not any(x['rel']=='next' for x in raw['links'])
 gauge={}; series=set(); rejected=0
 for f in raw['features']:
  p=f['properties']
  assert (p['monitoring_location_id'],p['parameter_code'],p['statistic_id'],p['unit_of_measure'])==('USGS-11335000','00060','00011','ft^3/s')
  t=parse(p['time'])
  assert LO<=t<=HI
  if t==HI or p['approval_status']!='Approved' or p.get('qualifier') is not None:
   rejected+=1
   continue
  assert t not in gauge
  series.add(p['time_series_id'])
  gauge[t]=num(p['value'])
 assert len(series)==1
 all_pairs=[]; by_cycle={}
 for asset in ('cnrfc_2022123112.zip','cnrfc_2022123118.zip'):
  cycle=dt.datetime.strptime(asset[6:16],'%Y%m%d%H').replace(tzinfo=UTC)
  modified=email.utils.parsedate_to_datetime(manifest[asset]['last_modified'])
  assert modified>cycle
  with zipfile.ZipFile(A/asset) as z:
   member,=z.infolist()
   assert member.filename==f"{cycle.strftime('%Y%m%d%H')}_N_SanJoaquin_csv_export.csv" and member.file_size<100_000
   it=iter(csv.reader(io.StringIO(z.read(member).decode('utf-8-sig'))))
   head=next(it); kind=next(it); assert head[0]=='GMT' and len(head)==len(set(head))==len(kind)
   ix=head.index(SELECTED)
   assert kind[ix]=='QINE'
   assert 'CMPC1' in head and kind[head.index('CMPC1')]=='QINE'
   assert 'MCNC1L' not in head
   time_seen=set(); cycle_pairs=[]; exclusions={'outside_gauge_window':0,'no_exact_gauge':0,'missing_qine':0}
   for row in it:
    assert len(row)==len(head)
    t=parse(row[0]); assert t not in time_seen and t.minute==t.second==0
    time_seen.add(t)
    if not LO<=t<HI: exclusions['outside_gauge_window']+=1; continue
    if t not in gauge: exclusions['no_exact_gauge']+=1; continue
    if not row[ix]: exclusions['missing_qine']+=1; continue
    value=num(row[ix])*1000
    diff=value-gauge[t]
    band='pre_cycle' if t<cycle else ('cycle_to_archive_clock' if t<=modified else 'after_archive_clock')
    cycle_pairs.append(dict(cycle_utc=stamp(cycle),archive_last_modified_utc=stamp(modified),valid_utc=stamp(t),clock_band=band,qine_cfs=str(value),approved_usgs_cfs=str(gauge[t]),difference_cfs=str(diff),abs_diff_cfs=str(abs(diff))))
   assert len(cycle_pairs)+sum(exclusions.values())==len(time_seen)
  by_cycle[stamp(cycle)]={'archive_clock_utc':stamp(modified),'row_count':len(time_seen),'exclusions':exclusions,
   'pre_cycle_past_alignment':calc([p for p in cycle_pairs if p['clock_band']=='pre_cycle']),
   'cycle_to_archive_clock_ambiguous':calc([p for p in cycle_pairs if p['clock_band']=='cycle_to_archive_clock']),
   'after_archive_clock_descriptive':calc([p for p in cycle_pairs if p['clock_band']=='after_archive_clock']),
   'official_crest_time_pair':next((p for p in cycle_pairs if p['valid_utc']==stamp(CREST)),None)}
  all_pairs.extend(cycle_pairs)
 out=R/'results';out.mkdir(exist_ok=True)
 with (out/'exact_hour_matches.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(all_pairs[0]),lineterminator='\n')
  w.writeheader();w.writerows(all_pairs)
 summary={'specification':'One event. Exact USD-free physical discharge match. CNRFC archive time is an exclusion screen, not public dissemination evidence.',
  'observation_window':[stamp(LO),stamp(HI)],'raw_usgs_rows':raw['numberReturned'],'eligible_usgs_rows':len(gauge),'excluded_usgs_rows':rejected,
  'usgs_series':next(iter(series)),'max_approved_usgs_cfs':str(max(gauge.values())),
  'max_approved_times_utc':[stamp(t) for t in sorted(gauge) if gauge[t]==max(gauge.values())],'site_gate':GATE,'cycles':by_cycle}
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
