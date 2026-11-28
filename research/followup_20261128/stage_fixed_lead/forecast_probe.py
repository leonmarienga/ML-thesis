"""Acquire/sanitize issue-time evidence, never acquire held observations."""
import argparse,csv,io,json,re,hashlib
from pathlib import Path
from datetime import datetime,timedelta,timezone
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen
from urllib.parse import urlencode
D=Path(__file__).resolve().parent
E={'KY2022':('JKNK2','KTIR','RVFKTY','2022-07-26','2022-07-29'),
   'OH2024':('WLGW2','KTIR','RVFOHW','2024-04-01','2024-04-04'),
   'HELENE2024':('AVLN7','KORN','RVFGSP','2024-09-25','2024-09-28')}
UTC=timezone.utc
def dt(s): return datetime.fromisoformat(s).replace(tzinfo=UTC)
def stamp(s): return datetime.strptime(s,'%Y%m%d%H%M').replace(tzinfo=UTC)
def request(key):
    lid,office,pil,start,end=E[key]
    base='https://mesonet.agron.iastate.edu/cgi-bin/'
    ua=base+'afos/retrieve.py?'+urlencode(dict(pil=pil,sdate=start,edate=end,limit=200,fmt='text'))
    uh=base+'request/hml.py?'+urlencode(dict(kind='forecasts',station=lid,sts=start+'T00:00Z',ets=end+'T00:00Z',fmt='csv',tz='UTC'))
    r={'event':key,'sources':{}}
    def get(url, name):
        with urlopen(url,timeout=25) as f:
            raw=f.read(3_000_001)
            assert f.status==200 and len(raw)<=3_000_000
            r['sources'][name]={'url':f.geturl(),'status':f.status,'sha256':hashlib.sha256(raw).hexdigest(),'byte_count':len(raw)}
            return raw.decode('ascii')
    data=get(ua,'wmo')
    bulletins=[]
    for part in data.replace('\r','').split('\x01'):
        hdr=re.search(r'(?m)^((?:FGUS\d{2}) '+office+r' (\d{6})(?: ([A-Z]{3}))?)\s*$', part)
        is_guidance=bool(re.search(r'(?i)guidance product|public forecasts and warnings are issued',part))
        lines=part.splitlines()
        for i,line in enumerate(lines):
            match=re.match(r'^\.ER '+lid+r' (\d{4}|\d{8}) Z (\S+)',line)
            if not hdr or not match: continue
            tokens=match[2].strip('/').split('/')
            if not any(re.fullmatch(r'HGI(?:FF(?:ZZ)?|FE)',xx) for xx in tokens): continue
            dc=next((x[2:] for x in tokens if re.fullmatch(r'DC\d{12}',x)),None)
            if not dc: continue
            create=stamp(dc)
            possible=[(create+timedelta(days=j)).replace(hour=int(hdr[2][2:4]),minute=int(hdr[2][4:6]))
                    for j in (-1,0,1) if (create+timedelta(days=j)).day==int(hdr[2][:2])]
            assert len(possible)==1
            wmo=possible[0]
            label=re.search(r'(?m)^'+pil+r'\s*$',part)
            assert label
            pid=f'{wmo:%Y%m%d%H%M}-{office}-{hdr[1].split()[0]}-{pil}'+(f'-{hdr[3]}' if hdr[3] else '')
            enc=[]
            for jj in range(i+1,len(lines)):
                ln=lines[jj].strip()
                if not ln or ln.startswith(':'): continue
                if not re.match(r'^\.E(?:R)?\d+\s+',ln): break
                enc.append(ln)
            bulletins.append({'wmo_header':hdr[1],'wmo':wmo.isoformat(),'dc':create.isoformat(),
                'correction':hdr[3],'guidance_disclaimer':is_guidance,'forecast_header':line.strip(),'forecast_continuations':enc,
                'public_archive_link':'https://mesonet.agron.iastate.edu/p.php?pid='+pid})
    assert bulletins
    r['wmo_forecast_only']=bulletins
    raw=get(uh,'hml_forecasts_only')
    parsed=csv.DictReader(io.StringIO(raw))
    fields=['station','issued[UTC]','primaryname','primaryunits','forecast_valid[UTC]','primary_value']
    assert set(fields)<=set(parsed.fieldnames or [])
    r['hml_forecast_rows']=[{f:x[f] for f in fields} for x in parsed]
    assert all(x['station']==lid for x in r['hml_forecast_rows'])
    (D/f'forecast-{key}.json').write_text(json.dumps(r,sort_keys=True,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fetch',action='store_true');a=p.parse_args()
    if a.fetch:
        with ThreadPoolExecutor(3) as ex: list(ex.map(request,E))
