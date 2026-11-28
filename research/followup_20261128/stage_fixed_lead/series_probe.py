import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parent
base='https://api.waterdata.usgs.gov/ogcapi/v0/collections/time-series-metadata/items'
def get(item):
    lid,site=item
    noaa='https://api.water.noaa.gov/nwps/v1/gauges/'+lid
    with urlopen(noaa,timeout=12) as f:
        data=json.load(f)
        meta={k:data[k] for k in ('lid','usgsId','name','reachId','rfc','latitude','longitude','pedts') if k in data}
    usgs=base+'?'+urlencode(dict(monitoring_location_id='USGS-'+site,parameter_code='00065',f='json',limit=100))
    with urlopen(usgs,timeout=12) as f:
        y=json.load(f)
        assert not any(l['rel']=='next' for l in y.get('links',[]))
        print(lid,meta, [z.get('properties') for z in y.get('features',[])])
    (ROOT/f'metadata-{lid}.json').write_text(json.dumps({'noaa':{'url':noaa,'metadata':meta},'usgs':{'url':usgs,'metadata':y}},indent=2)+'\n')
if __name__=='__main__':
    with ThreadPoolExecutor(4) as e: list(e.map(get,[('JKNK2','03280000'),('WLGW2','03112500'),('AVLN7','03451500')]))
