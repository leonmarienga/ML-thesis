"""Read only allowlisted RFC guidance issues whose WMO precedes ALL held target instants.
Persist per-station metadata, never forecast/observation lines or bulletin bodies.
"""
from datetime import datetime
from urllib.request import urlopen
from urllib.parse import urlparse,parse_qs
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from hashlib import sha256
import csv,json,re

D=Path(__file__).resolve().parent
prereg=D.parent/'rfc-guidance-stage-prereg'/'frozen_pairs.csv'
expected='d4552c20e0dbf08358d4d6281e3680983412bb232c07cf32f80a6eaf96d56d2b'
if sha256(prereg.read_bytes()).hexdigest()!=expected: raise ValueError('wrong frozen pair registry')
pairs=list(csv.DictReader(prereg.open()))
out={}
for x in pairs:
    g=out.setdefault(x['episode'],{'first_target':x['target_utc'],'products':{}})
    g['first_target']=min(g['first_target'],x['target_utc'])
    g['products'][x['product_url']]=(x['nws_lid'],x['wmo'])

def read(job):
    event,url,lid,issue,first=job
    assert issue<first
    pid=parse_qs(urlparse(url).query)['pid'][0]
    dt,office,heading,afos=pid.split('-')
    assert datetime.fromisoformat(issue).strftime('%Y%m%d%H%M')==dt
    req='https://mesonet.agron.iastate.edu/api/1/nwstext/'+pid
    with urlopen(req,timeout=18) as r:
        raw=r.read(256*1024+1)
        assert len(raw)<=256*1024 and r.geturl()==req
        x=dict(event=event,lid=lid,issue=issue,first_target=first,archived_product=url,retrieval=req,
               bytes=len(raw),sha256=sha256(raw).hexdigest(),status=r.status)
    body=raw.decode('ascii','replace').replace('\r','')
    full_h=re.compile(r'(?m)^'+heading+' '+office+' '+dt[-6:]+r'[ \t]*$')
    assert len(full_h.findall(body))==1 and re.search(r'(?m)^'+afos+r'[ \t]*$',body)
    valid=re.compile(r"^\.ER (\w{5}) .*?/HGI(?:FF(?:ZZ)?|FE)/")
    lines=body.splitlines()
    site_markers=re.compile(r"^:[ \t]*(?:\*{5,}|.+ - .+ - [A-Z]{3})[ \t]*$")
    found=[i for i,line in enumerate(lines) if (m:=valid.match(line)) and m[1]==lid]
    assert len(found)==1,(event,found)
    (row,)=found
    start=max(i for i,line in enumerate(lines[:row]) if site_markers.fullmatch(line))
    stop=next((i for i in range(row+1,len(lines)) if site_markers.fullmatch(lines[i])),len(lines))
    block=lines[start:stop]
    assert {m[1] for l in block if (m:=valid.match(l))}=={lid}
    x['site_metadata_only']=[l.strip() for l in block if re.match(
        r"^:[ \t]*(?:Zero\s+Datum|Datum|.+(?: - PBZ| - JKL|FRENCH BROAD RIVER))\b",l,re.IGNORECASE)]
    return x
if __name__=='__main__':
    jobs=[]
    for e,g in out.items():
        for url,(lid,issue) in g['products'].items():
            if issue<g['first_target']:
                jobs.append((e,url,lid,issue,g['first_target']))
    assert len(jobs)==9, len(jobs)
    with ThreadPoolExecutor(3) as p:
        results=list(p.map(read,jobs))
    (D/'dated_rfc_metadata.json').write_text(json.dumps(results,sort_keys=True,indent=2)+'\n')
