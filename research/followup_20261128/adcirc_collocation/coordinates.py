"""Screen archive extraction nodes against fixed stations; no storm responses."""
import argparse
import json
from layout import ROOT, Range, SIZE
import h5py
import numpy as np

def great_circle_m(lon,lat,lon0,lat0):
    lat,lon,lat0,lon0=map(np.radians,(lat,lon,lat0,lon0))
    a=np.sin((lat-lat0)/2)**2+np.cos(lat)*np.cos(lat0)*np.sin((lon-lon0)/2)**2
    return 2*6371008.8*np.arcsin(np.sqrt(np.clip(a,0,1)))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fetch',action='store_true')
    a=p.parse_args()
    rf=Range(a.fetch)
    with h5py.File(rf,'r',driver='fileobj') as f:
        assert set(f) == {'node','longitude','latitude','time_strlen','time','subdir','time_str',
                           'name_strlen','subdir_names','swan_HS','swan_TPS','zeta'}
        lon,lat=[f[k][:].astype('float64') for k in ('longitude','latitude')]
        zz=f['zeta'] # Inspect metadata only.
        assert zz.chunks == (20,100,2000) and float(np.ravel(zz.attrs['_FillValue'])[0])==-999.
        zmeta={'shape':list(zz.shape),'chunks':list(zz.chunks),'dtype':str(zz.dtype),
               'compression':zz.compression,'units':zz.attrs['units'].decode(),
               'standard_name':zz.attrs['standard_name'].decode(),'fill_value':-999.0}
    assert lat.shape==lon.shape and len(lat)==zmeta['shape'][-1]
    assert np.all(np.isfinite(lon)) and np.all(np.isfinite(lat))
    assert np.all((-180<=lon)&(lon<=180)) and np.all((-90<=lat)&(lat<=90))
    out={'scope':'coordinate and metadata only; no hindcast response read', 'size_bytes':SIZE,
         'zeta_metadata':zmeta,'node_count':len(lon),
         'unique_coordinate_count':len(np.unique(np.column_stack([lon,lat]),axis=0)),
         'range_starts':sorted(rf.blocks),'range_bytes':sum(map(len,rf.blocks.values())), 'stations':{}}
    for site in ('8723970','8724580'):
        ss=json.loads((ROOT/f'coops-site-{site}.json').read_text())['stations']
        assert len(ss)==1 and ss[0]['id']==site
        sta=ss[0]; lat0,lon0=sta['lat'],sta['lng']
        dist=great_circle_m(lon,lat,lon0,lat0)
        i=np.lexsort((np.arange(len(dist)), dist))[:8]
        man=np.argmin(abs(lon-lon0)+abs(lat-lat0))
        out['stations'][site]={'name':sta['name'],'latitude':lat0,'longitude':lon0,
           'nearest_8':[{'index':int(j),'latitude':round(lat[j],7),'longitude':round(lon[j],7),
                'distance_m':round(dist[j],2)} for j in i],
           'publisher_manhattan_node':int(man),'nearest_same_as_manhattan':int(i[0])==int(man),
           'screen_radius_counts':{str(r):int((dist<=r).sum()) for r in [100,200,500,1000,2000]}}
    (ROOT/'coordinates.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
if __name__=='__main__': main()
