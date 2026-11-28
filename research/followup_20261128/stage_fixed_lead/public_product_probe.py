"""Check WFO public bulletin candidates, retaining only forecast-paragraph evidence."""
from urllib.request import urlopen
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,re,hashlib,time
from urllib.error import HTTPError
ROOT=Path(__file__).resolve().parent
work=[('KY2022','JKNK2','JKL','2022-07-26','2022-07-29'),
      ('OH2024','WLGW2','PBZ','2024-04-01','2024-04-04'),
      ('HELENE2024','AVLN7','GSP','2024-09-25','2024-09-28')]
def load(job):
    event,lid,office,start,end,kind=job
    pil=kind+office
    q='https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py?'+urlencode(dict(pil=pil,sdate=start,edate=end,limit=200,fmt='text'))
    for attempt in (0,1):
        try:
            with urlopen(q,timeout=20) as f:
                raw=f.read(2_000_001); assert len(raw)<=2_000_000
                status=f.status
            break
        except HTTPError as e:
            if attempt==1 or e.code not in (429,500,502,503,504):
                return {'event':event,'pil':pil,'status':e.code,'source':q,'site_matched_warning_headers':[]}
            time.sleep(2)
    matches=[]
    for record in raw.decode('ascii','replace').replace('\r','').split('\x01'):
        head=re.search(r'(?m)^(WGUS\d{2} K'+office+r' \d{6}(?: [A-Z]{3})?)\s*$',record)
        # NOAA hydrologic VTEC station line; no observed-stage data displayed or retained.
        if not head: continue
        for p in record.split('$$'):
            if not re.search(r'(?m)^/'+lid+r'\.',p): continue
            paras=[]
            for m in re.finditer(r'(?m)^\s*[\-*] Forecast\.\.\.[^\n]*(?:\n[ \t]+[^\n*][^\n]*)*',p):
                paras.append(' '.join(m[0].split()))
            matches.append({'header':head[1], 'has_labelled_forecast_paragraph': bool(paras)})
    return {'event':event,'pil':pil,'status':status,'source':q,'source_sha256':hashlib.sha256(raw).hexdigest(),
      'source_bytes':len(raw),'site_matched_warning_headers':matches}
if __name__=='__main__':
    with ThreadPoolExecutor(3) as p:
        got=list(p.map(load,[(*spec,kind) for spec in work for kind in ('FLW','FLS')]))
    (ROOT/'public-wfo-forecast-paragraphs.json').write_text(json.dumps(got,indent=2,sort_keys=True)+'\n')
    for x in got: print(x['event'],x['pil'],x.get('source_bytes'),len(x['site_matched_warning_headers']))
