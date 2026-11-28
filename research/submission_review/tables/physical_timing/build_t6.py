"""Build Chapter 4 T6 after checking only already-pinned evidence artifacts."""
import argparse
import csv
from hashlib import sha256
from pathlib import Path
import subprocess
import json
from io import StringIO
from decimal import Decimal

cli=argparse.ArgumentParser(description=__doc__)
cli.add_argument("--repo",type=Path,required=True,help="local Git object cache containing the pinned revisions")
cli.add_argument("--output",type=Path,required=True,help="directory for the generated table")
a=cli.parse_args()
REPO=a.repo.resolve(strict=True)
ROOT=a.output.resolve()
ROOT.mkdir(parents=True,exist_ok=True)
F='research/followup_20261128/'
STAGE='2038178997ca47ef0ea5fb591c4519a1f288409c'
SAME='fe9896d62075fae423043a104a62ef1c07ee7a1b'
QINE='6bd74e7dceaf339df2b9f911141a8a7281428d5f'
PINS={
'M':('fb314e0edb71faccf9b7cbb196b0a245459aa696','research/submission_review/THESIS.md'),
'S':(STAGE,F+'michigan_public_clock/REPORT.md'),
'S2':(STAGE,F+'michigan_public_clock/stage_summary.json'),
'SH':(SAME,F+'michigan_public_clock/SAME_HOUR_NOTE.md'),
'Q':(QINE,F+'cnrfc_qine/REPORT.md'),
'Q2':(QINE,F+'cnrfc_qine/results/summary.json'),
'P':('e40aa61d005df96db353c99bcd297edf7d490425',F+'stage_fixed_lead/REPORT.md'),
'P2':('e40aa61d005df96db353c99bcd297edf7d490425',F+'stage_fixed_lead/audit-summary.json'),
'R':('780999b3409a96190d2d281862d2d7eb23783dc4',F+'rfc_guidance_stage_prereg/PREREGISTRATION.md'),
'R2':('780999b3409a96190d2d281862d2d7eb23783dc4',F+'rfc_guidance_stage_prereg/frozen_pairs.csv'),
'D':('7cc417b7febfb4f996162d0886df3dfc9cb0efbb',F+'rfc_guidance_datum_audit/REPORT.md'),
'D2':('7cc417b7febfb4f996162d0886df3dfc9cb0efbb',F+'rfc_guidance_datum_audit/EVIDENCE_LEDGER.csv'),
'A':(QINE,F+'adcirc_collocation/REPORT.md'),
'A2':(QINE,F+'adcirc_collocation/enc_screen.json'),
}
def get(c,p):
    return subprocess.check_output(['git','-C',str(REPO),'show',f'{c}:{p}'])
def url(k):
    c,p=PINS[k]
    return f'https://github.com/leonmarienga/ML-thesis/blob/{c}/{p}'
def write_csv(path,cols,rows):
    with (ROOT/path).open('w',encoding='utf-8',newline='') as h:
        w=csv.writer(h,lineterminator='\n')
        w.writerow(cols)
        w.writerows(rows)

raw={k:get(c,p) for k,(c,p) in PINS.items()}
s={k:v.decode('utf-8') for k,v in raw.items()}
def covers(k,*strings):
    assert all(x in s[k] for x in strings),k
covers('M','Montpelier','04286000','110 forecast rows from ten issues','Fifty-seven future rows','remaining 53 valid times',
       '15:08 UTC issue forecast 19.8 ft at July 11 06:00 UTC','approved observation was 20.33 ft','384 timestamp matches','exactly in 233 cases')
stage=json.loads(s['S2'])
c1,c2=stage['cases']
assert stage['episodes']==1 and stage['first_actual_15m_crossing_12ft_utc']=='2022-12-31T09:00:00+00:00'
assert [x['matched_future_hours'] for x in stage['cases']]==[115,115]
assert [x['bulletin_wmo_issue_utc'] for x in stage['cases']]==['2022-12-28T17:13:00+00:00','2022-12-31T17:00:00+00:00']
assert [x['graphic_posted_and_shef_created_utc'] for x in stage['cases']]==['2022-12-28T16:45:00+00:00','2022-12-31T16:57:00+00:00']
assert [round(x['mae_ft'],3) for x in stage['cases']]==[1.258,0.639]
assert [c1['actual_crossing_minutes_after_wmo_issue'],c2['actual_crossing_minutes_after_wmo_issue']]==[3827,-480]
assert c1['forecast_crossing_minutes_after_actual_15m_crossing']==840
covers('SH','43 valid hours','1.637','0.646','11/21','21/21','0/22','72 hours 47 minutes','one hour beforehand')
q=json.loads(s['Q2'])
assert q['site_gate']['MHBC1']['target']=='USGS-11335000/00060'
assert [(r['archive_clock_utc'],r['pre_cycle_past_alignment']['count'],r['cycle_to_archive_clock_ambiguous']['count'],
        r['after_archive_clock_descriptive']['count']) for r in q['cycles'].values()]==[
            ('2022-12-31T16:56:47Z',108,5,55),('2022-12-31T20:58:13Z',114,3,51)]
covers('Q','only 55 distinct valid hours','63,627.48 versus 63,600 cfs','not** a verified first public release time')
p=json.loads(s['P2'])
assert p['strict_eligible_rows']==0 and p['held_observations_accessed'] is False
assert [p[e]['guidance_disclaimers'] for e in ('KY2022','OH2024','HELENE2024')]==[10,7,12]
covers('P','All 24 planned targets','retained only WMO headers','does **not** establish same-issue')
rows=list(csv.DictReader(StringIO(s['R2'])))
from collections import Counter
assert Counter(x['episode'] for x in rows)=={'KY2022':4,'OH2024':2,'HELENE2024':5}
assert all(1440<=int(x['lead_minutes'])<1800 for x in rows)
covers('R','only if all 11 pairs','all three event-level reference checks pass','No real observations')
d={x['event']:x['disposition'] for x in csv.DictReader(StringIO(s['D2']))}
assert d=={'KY2022':'STRONG_DATED_LOCAL_REFERENCE','OH2024':'UNRESOLVED_DATED_USGS_LOCAL_REFERENCE','HELENE2024':'STRONG_DATED_LOCAL_REFERENCE'}
covers('D','697.674','697.67','1950.28','608.1','607.57','607.3','0.27 ft')
covers('A','1,639.25 m','4,028.78 m','Model Zero','native','No held-event hindcast water-level values')
enc=json.loads(s['A2'])
assert len(enc)==2 and all(v['land_features_intersecting_middle_90pct']==1 for st in enc.values() for v in st.values())

table=[
('Montpelier, July 2023: river stage (ft)',
 'NWS HML issue and valid times; exact future match to approved USGS 04286000 stage.',
 '57 of 110 rows across 10 issues matched; 53 outside the fixed window. July 10 15:08 UTC issue: 19.8 ft at July 11 06:00 UTC versus 20.33 ft observed.',
 'One flood. Public dissemination unverified for the full series; no operational benchmark.',
 ('M',)),
('Michigan Bar, Dec. 2022: river stage (ft)',
 'CNRFC stage chart posted times agree with SHEF creation; use later WMO header. Approved USGS 11335000/00065.',
 '115 future matches per issue. The earlier issue preceded the first measured 12-ft crossing by 63 h 47 min, but forecast the crossing 14 h late; the later issue followed it by 8 h. On the same 43 valid hours, MAE was 1.637 vs 0.646 ft.',
 '43 hours are a subset, not additional pairs. Later issue has shorter leads; no lead-adjusted or across-storm skill claim.',
 ('S','SH')),
('Michigan Bar, Dec. 2022: QINE discharge (cfs)',
 'Adjusted CNRFC QINE vs approved same-site USGS 11335000/00060. Nominal cycles differ from HTTP Last-Modified.',
 '222 pre-cycle comparison instances; 8 withheld between nominal cycle and file clock; 106 after file clock (55 distinct valid hours).',
 'Pre-cycle values can incorporate gauge data. File timestamps do not certify public issuance; stage records do not date QINE.',
 ('Q',)),
('Three-event public-stage screen',
 'Jackson 2022; Wheeling and Asheville 2024. Check numeric stage, exact target and contemporaneous public issue identity.',
 '0/24 eligible. All 29 stage bulletins were RFC guidance; 12 site-matched WFO warning/statement sections did not establish trace identity.',
 'No strict public forecast scores. Not a claim that no public forecasts were issued.',
 ('P',)),
('Exploratory RFC stage freeze and datum audit',
 'Exact USGS 00065/00011 stage targets; later of WMO, SHEF creation and corroborating HML clock; 24 to <30 h.',
 '11 locked targets (4 Jackson, 2 Wheeling, 5 Asheville). Strong dated local stage reference at Jackson and Asheville; Wheeling remains unresolved.',
 'Guidance archive only, not public release. No held gauge values accessed or scored; the preregistered three-event summary cannot yet be computed.',
 ('R','D')),
('Irma 2017: ADCIRC+SWAN suitability only',
 'Retrospective modeled water level vs candidate NOAA gauges. Test distance, shoreline barriers and local reference, before response comparison.',
 'Nearest extracted locations: 1.64 km (Vaca Key) and 4.03 km (Key West); modern chart land polygons intersect both straight paths.',
 'No match based on distance alone; native wet connection and model-to-gauge datum unverified. No hindcast response scored.',
 ('A',)),
]
cols=('Evidence and measurement','Measurement and clock check','Verified result','Interpretation limit','Sources')
write_csv('T6_AUTHENTIC_PHYSICAL_TIMING.csv',cols, [(*x[:4],'; '.join(url(k) for k in x[4])) for x in table])
write_csv('T6_SOURCE_PINS.csv',('key','commit','path','sha256','url'),[(k,c,p,sha256(raw[k]).hexdigest(),url(k)) for k,(c,p) in PINS.items()])
out=['**Table T6. Authentic physical timing and compatibility evidence.** Stage, discharge and simulated coastal water levels are distinct endpoints. The final row is a modeled coastal suitability screen, included for contrast.',
     '', '| '+' | '.join(cols)+' |', '| '+' | '.join('---' for _ in cols)+' |']
for *items,sources in table:
    out.append('| '+' | '.join([*items,', '.join(f'[{k}]({url(k)})' for k in sources)])+' |')
out+=['','The checks separate measurement compatibility from the forecast clock. The Michigan Bar stage archive provides the strongest issue-time evidence in this set, but both versions describe one flood and the later version was issued after the first measured crossing. The discharge and wider guidance studies do not inherit that clock. These studies do not estimate funding, humanitarian need, or allocation effectiveness.','']
(ROOT/'T6_AUTHENTIC_PHYSICAL_TIMING.md').write_text('\n'.join(out),encoding='utf-8')
