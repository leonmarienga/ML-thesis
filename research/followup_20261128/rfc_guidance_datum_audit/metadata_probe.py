"""Bounded metadata-only HTTP reads. No continuous/peaks/forecast measurements."""
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError,URLError
from urllib.request import urlopen
from pathlib import Path
from hashlib import sha256
import json

D=Path(__file__).resolve().parent
SITES={'KY2022':('KY','03280000','JKNK2'),'OH2024':('WV','03112500','WLGW2'),'HELENE2024':('NC','03451500','AVLN7')}
def capture(record):
    event,source,url=record
    result={'event':event,'source':source,'url':url}
    allowed={'https://hads.ncep.noaa.gov/USGS/','https://api.waterdata.usgs.gov/ogcapi/v1/collections/monitoring-locations/items/'}
    assert any(url.startswith(a) for a in allowed)
    try:
        with urlopen(url,timeout=20) as r:
            assert any(r.geturl().startswith(a) for a in allowed), 'Unexpected redirect'
            raw=r.read(256*1024+1)
            assert len(raw)<=256*1024
            result.update(status=r.status,date=r.headers.get('Date'),last_modified=r.headers.get('Last-Modified'),
                         bytes=len(raw),sha256=sha256(raw).hexdigest())
    except (HTTPError,URLError,TimeoutError) as e:
        result.update(status=getattr(e,'code',None),error=type(e).__name__)
        return result
    state,usgs,lid=SITES[event]
    if source=='hads':
        rs=[ln for ln in raw.decode('ascii','replace').splitlines() if usgs in ln or lid in ln]
        assert rs
        result['metadata_rows']=rs
    else:
        data=json.loads(raw)
        assert data['type']=='Feature' and data['id']=='USGS-'+usgs
        keys=('id','monitoring_location_name','vertical_datum','altitude','altitude_accuracy','altitude_method_name','horizontal_positional_accuracy_code','revision_note','revision_created','revision_modified')
        result['metadata']={k:data['properties'][k] for k in keys}
        result['coordinates_lon_lat']=data['geometry']['coordinates']
    return result
def main():
    requests=[]
    for event,(state,site,*_) in SITES.items():
        requests.extend([(event,'hads',f'https://hads.ncep.noaa.gov/USGS/{state}_USGS-HADS_SITES.txt'),
                         (event,'usgs-location',f'https://api.waterdata.usgs.gov/ogcapi/v1/collections/monitoring-locations/items/USGS-{site}?f=json')])
    with ThreadPoolExecutor(3) as pool:
        output=list(pool.map(capture,requests))
    (D/'metadata_probe.json').write_text(json.dumps(output,sort_keys=True,indent=2)+'\n')
if __name__=='__main__': main()
