"""Run a conservative official-chart land-intersection screen; not a wet-mask test."""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode
from urllib.request import urlopen
import json
root=Path(__file__).resolve().parent
layers={'harbour':233,'approach':238}
def req(t):
    site,scale,label,typ,geo=t
    url=f'https://maritimeboundaries.noaa.gov/arcgis/rest/services/encdirect/enc_{scale}/MapServer/{layers[scale]}/query'
    u=url+'?'+urlencode({'geometry':json.dumps(geo,separators=(',',':')),'geometryType':typ,'inSR':4326,
       'spatialRel':'esriSpatialRelIntersects','returnIdsOnly':'true','f':'json','where':'1=1'})
    with urlopen(u,timeout=20) as r:
        raw=r.read(128*1024+1); assert r.status==200 and len(raw)<=128*1024
        obj=json.loads(raw); assert 'objectIds' in obj and not obj.get('exceededTransferLimit'),obj
        return f'{site}-{scale}-{label}',{'url':r.geturl(),'http':r.status,'date':r.headers.get('Date'),'result':obj}
def run(fetch):
    if fetch:
        tasks=[]
        for site,meta in json.loads((root/'coordinates.json').read_text())['stations'].items():
            n=meta['nearest_8'][0]
            pair=[[meta['longitude'],meta['latitude']],[n['longitude'],n['latitude']]]
            lowx,lowy=map(min,zip(*pair)); highx,highy=map(max,zip(*pair))
            box={'xmin':lowx-.01,'xmax':highx+.01,'ymin':lowy-.01,'ymax':highy+.01}
            trimmed=[[a+frac*(b-a) for a,b in zip(*pair)] for frac in (.05,.95)]
            for scale in layers:
                tasks.extend([(site,scale,'full','esriGeometryPolyline',{'paths':[pair]}),
                    (site,scale,'trimmed','esriGeometryPolyline',{'paths':[trimmed]}),
                    (site,scale,'box','esriGeometryEnvelope',box)])
        with ThreadPoolExecutor(max_workers=4) as ex:
            output=dict(ex.map(req,tasks))
        (root/'enc_requests.json').write_text(json.dumps(output,sort_keys=True,indent=2)+'\n')
    raw=json.loads((root/'enc_requests.json').read_text())
    report={}
    for site in ('8723970','8724580'):
        report[site]={}
        for scale in layers:
            box,segments,trimmed=[set(raw[f'{site}-{scale}-{geom}']['result']['objectIds'] or []) for geom in
                          ('box','full','trimmed')]
            assert trimmed<=segments<=box
            report[site][scale]={'land_features_in_nearby_window':len(box),
               'land_features_intersecting_straight_station_to_node_segment':len(segments),
               'land_features_intersecting_middle_90pct':len(trimmed),
               'interpretation':'land intersection flagged' if segments else 'not proof of a wet path'}
    (root/'enc_screen.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fetch',action='store_true')
    run(p.parse_args().fetch)
