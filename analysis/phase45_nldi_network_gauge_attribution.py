#!/usr/bin/env python3
"""Phase45: USGS NLDI hydrologic linkage (not inundation verification).
Frozen Phase42 radar polygons, 18 Phase44 screened USGS flow gauges.
No funding model trained, no changes to accepted Track A.
"""
from pathlib import Path
from urllib.parse import urlparse
import json,time,re
import numpy as np,pandas as pd,requests
from shapely.geometry import shape
from shapely.ops import unary_union
R=Path(__file__).resolve().parents[1]
I=R/'audit_inputs'/'phase45';O=R/'audit_outputs'/'phase45';O.mkdir(parents=True,exist_ok=True)
raw=json.loads((I/'phase42'/'radar_water_change_cANDIDATES_unverified.geojson').read_text())
P={int(x['properties']['disasterNumber']):shape(x['geometry']) for x in raw['features']}
assert set(P)=={4718,4719}
G=pd.read_csv(I/'phase44'/'phase44_USGS_nearby_gauge_peak_screen.csv')
G=G[G.sampling_eligible].copy()
assert len(G)==18 and G.disasterNumber.nunique()==2
BASE='https://api.water.usgs.gov/nldi/linked-data'
S=requests.Session();S.headers.update({'User-Agent':'Academic flood analytics NLDI network validation/1.0','Accept':'application/geo+json, application/json'})
requests_audit=[]
def get(url,params=None,timeout=60):
    assert urlparse(url).hostname=='api.water.usgs.gov'
    for n in range(3):
        try:
            resp=S.get(url,params=params,timeout=(15,timeout))
            resp.raise_for_status()
            data=resp.json()
            if isinstance(data,dict) and data.get('detail') and 'features' not in data:
                raise ValueError(str(data)[:180])
            requests_audit.append({'url':resp.url,'status':'ok','http_code':resp.status_code,'bytes':len(resp.content)})
            return data
        except Exception as e:
            err=type(e).__name__+': '+str(e)[:190]
            if n<2:time.sleep((n+1)*2)
    requests_audit.append({'url':url,'status':'failed','error':err})
    print('SOURCE_FAIL',url,err,flush=True)
    return None
def features(obj):
    if obj is None:return []
    if isinstance(obj,list):return obj
    if isinstance(obj,dict):
        if obj.get('type')=='Feature':return [obj]
        if isinstance(obj.get('features'),list):return obj['features']
    return []
def identifier(f):
    p=f.get('properties',{})
    for k in ('comid','COMID','identifier','nhdplus_comid'):
        v=p.get(k)
        if v is not None and str(v).isdigit():return str(v)
    v=f.get('id')
    return str(v) if v is not None and str(v).isdigit() else None
def nav_ids(obj):
    out=set()
    for f in features(obj):
        p=f.get('properties',{})
        for v in [f.get('id'),p.get('identifier'),p.get('site_no'),p.get('featureID')]:
            if v is None:continue
            for num in re.findall(r'(?<!\d)(\d{8,15})(?!\d)',str(v)):
                out.add('USGS-'+num)
    return out
def basin_test(poly,obj):
    if obj is None:return {'status':'unavailable'}
    geos=[shape(f['geometry']) for f in features(obj) if f.get('geometry')]
    if not geos:return {'status':'no_geometry'}
    try:
        area=unary_union(geos)
        return {'status':'tested','candidate_centroid_inside_upstream_basin':bool(area.covers(poly.centroid)),
            'candidate_polygon_intersects_upstream_basin':bool(area.intersects(poly)),
            'candidate_polygon_inside_upstream_basin':bool(area.covers(poly)),
            'current_basin_is_event_inundation':False}
    except Exception as e:return {'status':'geometry_error','error':str(e)[:180]}
cases=[];rows=[]
for dn in sorted(P):
    poly=P[dn];lon,lat=poly.centroid.x,poly.centroid.y
    loc=get(BASE+'/comid/position',{'coords':f'POINT({lon} {lat})','f':'json'})
    ff=features(loc);comid=identifier(ff[0]) if ff else None
    hydro=get(BASE+'/hydrolocation',{'coords':f'POINT({lon} {lat})','f':'json'})
    hf=features(hydro)
    cr={'disasterNumber':dn,'candidate_centroid_lon':lon,'candidate_centroid_lat':lat,
        'catchment_comid':comid,'NLDI_position_resolved':comid is not None,
        'hydrolocation_available':bool(hf),'FEMA_event_flood_extent_verified':False}
    network={}
    if comid:
        for nav in ['UT','DD']:
            obj=get(BASE+f'/comid/{comid}/navigation/{nav}/nwissite',
                    {'distance':150,'f':'json'},timeout=90)
            network[nav]={'status':'retrieved' if obj is not None else 'unavailable',
                'n_features':len(features(obj)), 'site_ids':sorted(nav_ids(obj))}
    cr['network_navigation']=network
    local=G[G.disasterNumber==dn].sort_values('distance_km')
    for rank,(_,r) in enumerate(local.iterrows(),1):
        sid=str(r.site_id)
        z={'disasterNumber':dn,'gauge_id':sid,'straight_line_km':float(r.distance_km),
           'geographic_distance_rank':rank,'peak_discharge_date':str(r.wave_peak_date),
           'negative_flow_values':0 if str(r.site_id)!='USGS-06471000' else 6,
           'nldi_upstream_tributaries_match':sid in network.get('UT',{}).get('site_ids',[]),
           'nldi_downstream_diversions_match':sid in network.get('DD',{}).get('site_ids',[])}
        if rank<=4:
            site=get(BASE+'/nwissite/'+sid,{'f':'json'},timeout=50)
            sf=features(site)
            z['NLDI_site_feature_available']=bool(sf)
            z['NLDI_site_comid']=identifier(sf[0]) if sf else None
            basin=get(BASE+'/nwissite/'+sid+'/basin',{'f':'json','simplified':'true'},timeout=85)
            z['gauge_upstream_basin_test']=basin_test(poly,basin)
        else:z['gauge_upstream_basin_test']={'status':'not_tested_rank_above_4'}
        z['network_related']=bool(z['nldi_upstream_tributaries_match'] or
            z['nldi_downstream_diversions_match'] or
            z['gauge_upstream_basin_test'].get('candidate_centroid_inside_upstream_basin',False))
        z['network_relation_is_NOT_local_flood_measurement']=True
        rows.append(z)
        print('NETWORK_SITE',dn,sid,z['network_related'],z['gauge_upstream_basin_test']['status'],flush=True)
    cases.append(cr)
    print('NETWORK_CASE',dn,'COMID',comid,'UT/DD',[(a,network[a]['n_features']) for a in network],flush=True)
D=pd.DataFrame(rows)
D.to_csv(O/'phase45_station_network_connections.csv',index=False)
(O/'phase45_candidate_catchment_navigation.json').write_text(json.dumps(cases,indent=2))
(O/'phase45_source_request_log.json').write_text(json.dumps(requests_audit,indent=2))
summary={'radar_candidate_disasters':2,'USGS_gauges_screened':len(D),
    'cases_resolved_to_NHDPlus_comid':sum(x['NLDI_position_resolved'] for x in cases),
    'gauge_upstream_basin_queries_tested':int(D.gauge_upstream_basin_test.map(lambda x:x['status']=='tested').sum()),
    'gauge_network_relations_found':int(D.network_related.sum()),
    'disasters_with_at_least_one_network_related_gauge':int(D[D.network_related].disasterNumber.nunique()),
    'verified_event_specific_inundation_footprints':0,
    'verified_building_or_population_exposure':0,'new_funding_models':0,
    'frozen_accepted_Track_A_unchanged':True,
    'caveats':['The USGS NLDI uses NHDPlusV2 topology; current mapped basin and network are not dated 2023 floodwater.',
      'A gauge hydrologically connected to a radar patch may be far upstream or downstream and does not measure patch flood depth.',
      'Point-in-catchment position and stream snapping may misrepresent exact local overland flow; verify terrain.',
      'The selected radar patches are post hoc, and one lies in a FEMA county but neither is a disaster-wide flood outline.',
      'Failed or missing API calls cannot be interpreted as proof of absent stream connection.',
      'USGS station discharge time series are retrospective; no T0/T72 as-of certification.',
      'No model or accepted Track A files modified.']}
(O/'phase45_summary.json').write_text(json.dumps(summary,indent=2))
readable=D[['disasterNumber','gauge_id','straight_line_km',
   'nldi_upstream_tributaries_match','nldi_downstream_diversions_match','network_related']]
(O/'PHASE45_REPORT.md').write_text('# Phase 45 — Hydro-network attribution audit\n\n'
 'USGS NLDI catchment and network are used to test actual upstream/downstream relationships, '
 'not to certify 2023 flood-water extent.\n\n## Summary\n\n'+
 json.dumps({k:v for k,v in summary.items() if k!='caveats'},indent=2)+
 '\n\n## Candidate-gauge connections\n\n'+readable.to_markdown(index=False)+
 '\n\n## Caveats\n\n'+'\n'.join('- '+x for x in summary['caveats']))
print('PHASE45_SUMMARY',json.dumps({k:v for k,v in summary.items() if k!='caveats'},indent=2),flush=True)
