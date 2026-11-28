import csv,json
from collections import Counter,defaultdict
from datetime import datetime
from decimal import Decimal as D
from pathlib import Path
from hashlib import sha256
r=Path(__file__).resolve().parent
a=json.loads((r/'dated_rfc_metadata.json').read_text())
assert Counter(x['event'] for x in a)=={'KY2022':3,'OH2024':2,'HELENE2024':4}
for x in a:
    assert x['status']==200 and datetime.fromisoformat(x['issue'])<datetime.fromisoformat(x['first_target'])
    assert str(x['archived_product']).split('pid=')[-1]==str(x['retrieval']).split('/')[-1]
    mm=x['site_metadata_only']
    if x['event']=='KY2022': assert mm==[': Jackson - North Fork Kentucky River - JKL',': Zero Datum 697.674 ft (NGVD 1929)']
    if x['event']=='OH2024': assert mm==[': Wheeling - Ohio River - PBZ',': Zero Datum 608.1 ft (NGVD 1929)']
    if x['event']=='HELENE2024': assert mm==[':ASHEVILLE, NC - FRENCH BROAD RIVER']
m=json.loads((r/'metadata_probe.json').read_text())
assert len(m)==6 and all(x['status']==200 for x in m)
y=defaultdict(dict)
for x in m: y[x['event']][x['source']]=x
for k,(lid,s) in {'KY2022':('JKNK2','03280000'),'OH2024':('WLGW2','03112500'),'HELENE2024':('AVLN7','03451500')}.items():
    assert y[k]['hads']['metadata_rows'][0].replace(' ','').startswith(f'{lid}|{s}|')
    assert y[k]['usgs-location']['metadata']['id']=='USGS-'+s
ky=y['KY2022']['usgs-location']['metadata']
assert '697.67 ft above NGVD of 1929' in ky['revision_note'] and ky['revision_modified'].startswith('2022-11-29')
z={x['event']:x for x in json.loads((r/'ncat-gauge-zero.json').read_text())}
o=z['OH2024']['result']
assert o['vertconVersion']=='3.0' and o['destVertDatum']=='NAVD88'
diff=D(o['destOrthoht'])*3937/1200-D(str(y['OH2024']['usgs-location']['metadata']['altitude']))
assert diff.quantize(D('.01'))==D('.27') and o['sigOrthoht']=='0.004'
with (r/'EVIDENCE_LEDGER.csv').open() as h:
    status={x['event']:x['disposition'] for x in csv.DictReader(h)}
assert status=={'KY2022':'STRONG_DATED_LOCAL_REFERENCE','OH2024':'UNRESOLVED_DATED_USGS_LOCAL_REFERENCE','HELENE2024':'STRONG_DATED_LOCAL_REFERENCE'}
print('Validated the metadata ledger and cutoff checks. No truth opened.')
