#!/usr/bin/env python3
"""Bounded, resumable JRC same-edition temporal placebo audit.

Reconstructs the fixed geographic design from original source code plus recovered
Phase49 log keys. Original ZIPs were subsequently recovered; the separate
archive-recovery audit verifies exact geometry and integer gain counts.
No funding targets are used. No observed surface-water pixel is a flood label.
"""
from __future__ import annotations
import argparse, hashlib, json, math, re, sys
from datetime import datetime, timezone
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import rasterio
from pyproj import Geod, Transformer
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds, Window
from shapely.geometry import Point, box, mapping, shape
from shapely.ops import transform as geo_transform

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'https://data.source.coop/epoch/jrc-global-surface-water-monthly/v1_5'
CENSUS = 'https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip'
GEOD = Geod(ellps='WGS84')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def shift(ym, months=0, years=0):
    return (pd.Timestamp(ym + '-01') + pd.DateOffset(months=months, years=years)).strftime('%Y-%m')

def comparison_dates(ym):
    seasonal = [shift(ym, months=-1), ym,
                shift(shift(ym, years=-1), months=-1), shift(ym, years=-1)]
    out = {}
    if all('2022-01' <= x <= '2024-12' for x in seasonal):
        out['seasonal'] = seasonal
    if ym.startswith('2024'):
        out['interannual'] = [shift(ym, years=-1), ym,
                              shift(ym, years=-2), shift(ym, years=-1)]
    return out

def reconstruct(out):
    text = (ROOT/'research/phase49_recovered_log_evidence.txt').read_text()
    pattern = r'COUNTY_SAMPLE \d+ of 187 (\d+)_(\d+)_(\d+) event_observed \[([^]]*)\] new_water \[([^]]*)\]'
    records=[]
    for dn, fips, site, obs, gains in re.findall(pattern, text):
        vals = [float(x) for x in gains.split(',') if x.strip()]
        records.append(dict(disasterNumber=int(dn), county_fips=int(fips), site_index=int(site),
                            historical_event_observed=float(obs) if obs not in ('None','') else np.nan,
                            historical_event_gain_km2_rounded=vals[0] if vals else np.nan))
    logged = pd.DataFrame(records)
    assert len(logged)==187 and not logged.duplicated(['disasterNumber','county_fips','site_index']).any()
    assert (logged.site_index==1).sum()==63 and logged.disasterNumber.nunique()==36
    raw = out/'raw'; raw.mkdir(exist_ok=True)
    census = raw/'cb_2023_us_county_500k.zip'
    if not census.exists():
        r=requests.get(CENSUS,timeout=(20,90));r.raise_for_status();census.write_bytes(r.content)
    county=gpd.read_file('zip://'+str(census)).to_crs('EPSG:4326')
    shapes={int(r.GEOID):r.geometry for _,r in county.iterrows()}
    # Select metadata only; never load funding columns into the geographic frame.
    master=pd.read_excel(ROOT/'master_openfema_40plus.xlsx',
                         usecols=['disasterNumber','incidentBeginDate','fyDeclared','state','incidentType'])
    assert master.disasterNumber.is_unique
    master=master.set_index('disasterNumber')
    official_path=raw/'fema36_current_designations.json'
    official_url='https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries'
    if not official_path.exists():
        ids=sorted(logged.disasterNumber.unique());official_rows=[];queries=[]
        for offset in range(0,len(ids),12):
            query=' or '.join(f'disasterNumber eq {int(x)}' for x in ids[offset:offset+12])
            for skip in range(0,6000,1000):
                params={'$filter':'('+query+')',
                    '$select':'disasterNumber,incidentBeginDate,incidentEndDate,fyDeclared,state,fipsStateCode,fipsCountyCode,designatedArea,id,lastRefresh',
                    '$top':1000,'$skip':skip}
                response=requests.get(official_url,params=params,timeout=(20,45));response.raise_for_status()
                batch=response.json()['DisasterDeclarationsSummaries'];official_rows+=batch
                queries.append(response.url)
                if len(batch)<1000:break
            else:raise RuntimeError('FEMA pagination limit reached')
        official_path.write_text(json.dumps(dict(records=official_rows,query_urls=queries,
            retrieved_utc=datetime.now(timezone.utc).isoformat()),indent=2))
    official_doc=json.loads(official_path.read_text())
    official=pd.DataFrame(official_doc['records'])
    assert set(official.disasterNumber)==set(logged.disasterNumber)
    assert official.groupby('disasterNumber').incidentBeginDate.nunique().max()==1
    official_meta=official.drop_duplicates('disasterNumber').set_index('disasterNumber')
    discrepancies=[]
    for dn in sorted(logged.disasterNumber.unique()):
        old=str(pd.Timestamp(master.loc[dn].incidentBeginDate).date())
        new=str(pd.Timestamp(official_meta.loc[dn].incidentBeginDate).date())
        if old!=new:discrepancies.append(dict(disasterNumber=int(dn),master_incident_start=old,
            official_current_incident_start=new,rule='official date used; source revision not authenticated as-of; log replay checked'))
    pd.DataFrame(discrepancies,columns=['disasterNumber','master_incident_start','official_current_incident_start','rule']).to_csv(out/'incident_date_discrepancies.csv',index=False)
    roster=[];features=[]
    for (dn,fips),group in logged.groupby(['disasterNumber','county_fips'],sort=True):
        c=shapes[fips];origin=c.representative_point()
        local=f'+proj=aeqd +lat_0={origin.y} +lon_0={origin.x} +datum=WGS84 +units=m +no_defs'
        forward=Transformer.from_crs('EPSG:4326',local,always_xy=True).transform
        back=Transformer.from_crs(local,'EPSG:4326',always_xy=True).transform
        xy=geo_transform(forward,c);ox,oy=forward(origin.x,origin.y)
        selected=[(ox,oy)]
        xmin,ymin,xmax,ymax=xy.bounds
        candidates=[]
        for xx in np.linspace(xmin,xmax,11):
            for yy in np.linspace(ymin,ymax,11):
                if not xy.covers(Point(xx,yy)):continue
                area=xy.intersection(box(xx-2000,yy-2000,xx+2000,yy+2000)).area
                if area>=8e6:candidates.append((float(xx),float(yy),float(area/16e6)))
        for slot in (2,3):
            options=[]
            for xx,yy,cover in candidates:
                dist=min(math.hypot(xx-sx,yy-sy) for sx,sy in selected)
                if dist>=6500:options.append((-dist,-cover,xx,yy))
            if not options:break
            _,_,xx,yy=min(options);selected.append((xx,yy))
        assert set(group.site_index)==set(range(1,len(selected)+1))
        meta=master.loc[dn]
        assert meta.incidentType=='Flood'
        incident=official_meta.loc[dn].incidentBeginDate
        ym=pd.Timestamp(incident).strftime('%Y-%m')
        assert ym[:4] in ('2023','2024')
        for site,(xx,yy) in enumerate(selected,1):
            anchor=geo_transform(back,Point(xx,yy))
            geom=geo_transform(back,box(xx-2000,yy-2000,xx+2000,yy+2000)).intersection(c)
            r=dict(disasterNumber=int(dn),county_fips=int(fips),site_index=site,
                   sample_id=f'{dn}_{fips}_{site}',event_month=ym,
                   incident_start=str(pd.Timestamp(incident).date()),
                   state=meta.state,fyDeclared=int(meta.fyDeclared),
                   anchor_lon=anchor.x,anchor_lat=anchor.y,
                   site_area_km2=abs(GEOD.geometry_area_perimeter(geom)[0])/1e6)
            roster.append(r);features.append(dict(type='Feature',properties=r,geometry=mapping(geom)))
    frame=pd.DataFrame(roster).merge(logged,on=['disasterNumber','county_fips','site_index'],validate='one_to_one')
    frame.to_csv(out/'reconstructed_all187_roster.csv',index=False)
    (out/'reconstructed_all187_windows.geojson').write_text(json.dumps(dict(type='FeatureCollection',features=features)))
    picked=frame[(frame.site_index==1)|(frame.disasterNumber==4722)].copy()
    assert len(picked)==67 and (picked.disasterNumber==4722).sum()==6
    picked.to_csv(out/'frozen_67_site_roster.csv',index=False)
    archive_recovered=(ROOT/'research/recovered_artifacts/RECOVERY_AUDIT.json').exists()
    return picked,{f['properties']['sample_id']:shape(f['geometry']) for f in features},dict(
        census_url=CENSUS,census_sha256=sha(census),
        official_fema_metadata_sha256=sha(official_path),official_fema_query_urls=official_doc['query_urls'],
        official_vs_master_date_discrepancies=discrepancies,
        original_archive_recovered=archive_recovered,log_run=41581240598,log_job=125736337527,
        log_sha256=sha(ROOT/'research/phase49_recovered_log_evidence.txt'),
        master_sha256=sha(ROOT/'master_openfema_40plus.xlsx'),
        lineage='reconstructed logged keys and committed geometry algorithm; not original CSV bytes')

def source_url(lon,lat,ym):
    a=math.floor(lon/10)*10;b=math.ceil(lat/10)*10
    tile=f'{abs(a)}'+('W' if a<0 else 'E')+f'_{abs(b)}'+('S' if b<0 else 'N')
    y,m=ym.split('-')
    return f'{SOURCE}/{y}/{y}_{m}/monthlyhistory_{tile}_v1_5_{y}_{m}.tif'

def read_month(row,geom,ym,out,cache_root=None,cache_metadata=None):
    url=source_url(row.anchor_lon,row.anchor_lat,ym)
    file=out/'pixels'/f'{row.sample_id}_{ym}.npz';file.parent.mkdir(exist_ok=True)
    if not file.exists() and cache_root is not None:
        alternate=cache_root/'pixels'/file.name
        if alternate.exists():file=alternate
    retrieval=datetime.now(timezone.utc).isoformat()
    reused=file.exists()
    if file.exists():
        with np.load(file,allow_pickle=False) as z:
            matrix=z['matrix'];mask=z['mask'];tf=rasterio.Affine(*z['transform']);weights=z['weights']
            retrieval=str(z['retrieval_utc']) if 'retrieval_utc' in z.files else (cache_metadata or {}).get((row.sample_id,ym),'unknown; archived cache')
    else:
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_TIMEOUT='35',
                          GDAL_HTTP_MAX_RETRY='1',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif,.tiff',
                          GDAL_HTTP_MULTIRANGE='YES'):
            with rasterio.open(url) as src:
                assert str(src.crs)=='EPSG:4326'
                win=from_bounds(*geom.bounds,transform=src.transform).round_offsets().round_lengths()
                clipped=win.intersection(Window(0,0,src.width,src.height))
                if clipped!=win:raise ValueError('tile_seam_requires_mosaic; excluded rather than silently clipped')
                assert win.width<=500 and win.height<=500
                matrix=src.read(1,window=win);tf=src.window_transform(win)
                mask=geometry_mask([mapping(geom)],out_shape=matrix.shape,transform=tf,invert=True)
                assert mask.any() and np.isin(matrix[mask],[0,1,2]).all()
                areas=[]
                # Exact ellipsoidal row cell areas, not a constant 30m assumption.
                for iy in range(matrix.shape[0]):
                    x,y=tf*(0,iy);xx,yy=tf*(1,iy+1)
                    areas.append(abs(GEOD.geometry_area_perimeter(box(x,yy,xx,y))[0])/1e6)
                weights=np.repeat(np.array(areas)[:,None],matrix.shape[1],axis=1)
        np.savez_compressed(file,matrix=matrix,mask=mask,weights=weights,transform=np.array(tuple(tf)[:6]),retrieval_utc=retrieval)
    values=matrix[mask]
    audit=dict(sample_id=row.sample_id,disasterNumber=int(row.disasterNumber),month=ym,
               source_url=url,status='read_ok',pixels=int(mask.sum()),
               observed_pixels=int((values>0).sum()),water_pixels=int((values==2).sum()),
               observed_fraction=float((values>0).mean()),cached_pixel_sha256=sha(file),
               retrieval_date_utc=retrieval,cache_reused=reused,
               audit_read_date_utc=datetime.now(timezone.utc).isoformat(),source_edition='JRC_v1_5_2022_2024',
               observation_period=ym,publication_date='unknown; retrospective',
               classification='VERIFIED DERIVED MEASUREMENT',
               license='JRC original terms apply; redistribution mirror license not independently established')
    return dict(values=values,weights=weights[mask],tf=tf,mask=mask),audit

def transition(before,after,valid,weights):
    gain=(before==1)&(after==2)&valid;loss=(before==2)&(after==1)&valid
    area=float(weights[valid].sum())
    return dict(observed_km2=area,gain_km2=float(weights[gain].sum()),
                loss_km2=float(weights[loss].sum()),gain_cells=int(gain.sum()),loss_cells=int(loss.sum()),
                net_km2=float(weights[gain].sum()-weights[loss].sum()))

def compare(values,dates,threshold,denominator):
    a,b,c,d=[values[x]['values'] for x in dates]
    weights=values[dates[0]]['weights']
    ev=(a>0)&(b>0);pl=(c>0)&(d>0);common=ev&pl
    if denominator=='common_pixels':ev=pl=common
    if ev.mean()<threshold or pl.mean()<threshold or not ev.any() or not pl.any():return None
    event=transition(a,b,ev,weights);placebo=transition(c,d,pl,weights)
    result=dict(event_observable_fraction=float(ev.mean()),placebo_observable_fraction=float(pl.mean()),
                common_observable_fraction=float(common.mean()))
    for arm,stats in [('event',event),('placebo',placebo)]:
        for k,v in stats.items():result[f'{arm}_{k}']=v
        for name in ('gain','loss','net'):
            result[f'{arm}_{name}_per100']=100*stats[f'{name}_km2']/stats['observed_km2']
    for name in ('gain','loss','net'):result[f'delta_{name}_per100']=result[f'event_{name}_per100']-result[f'placebo_{name}_per100']
    return result

def summarize(rows,out):
    sites=pd.DataFrame(rows)
    sites.to_csv(out/'site_temporal_comparisons.csv',index=False)
    if sites.empty:return []
    totals=[];cases=[]
    for (scheme,denom,threshold),g in sites.groupby(['scheme','denominator','threshold']):
        groups={'primary_63_anchors':g[g.site_index==1],
                'anchors_excluding_MN4722':g[(g.site_index==1)&(g.disasterNumber!=4722)],
                'sensitivity_67_with_MN_extras':g,
                'MN4722_six_sites_only':g[g.disasterNumber==4722]}
        for label,s in groups.items():
            r=dict(scheme=scheme,denominator=denom,threshold=threshold,group=label,
                   sites=len(s),cases=int(s.disasterNumber.nunique()))
            for arm in ('event','placebo'):
                area=float(s[f'{arm}_observed_km2'].sum());r[f'{arm}_observed_km2']=area
                for m in ('gain','loss','net'):
                    value=float(s[f'{arm}_{m}_km2'].sum());r[f'{arm}_{m}_km2']=value
                    r[f'{arm}_{m}_per100']=100*value/area if area else None
            for m in ('gain','loss','net'):
                r[f'delta_{m}_per100']=(r[f'event_{m}_per100']-r[f'placebo_{m}_per100']) if len(s) else None
            totals.append(r)
            for dn,z in s.groupby('disasterNumber'):
                q=dict(scheme=scheme,denominator=denom,threshold=threshold,group=label,disasterNumber=int(dn),sites=len(z))
                for arm in ('event','placebo'):
                    area=z[f'{arm}_observed_km2'].sum()
                    for m in ('gain','loss','net'):q[f'{arm}_{m}_per100']=100*z[f'{arm}_{m}_km2'].sum()/area
                for m in ('gain','loss','net'):q[f'delta_{m}_per100']=q[f'event_{m}_per100']-q[f'placebo_{m}_per100']
                cases.append(q)
    pd.DataFrame(totals).to_csv(out/'aggregate_temporal_comparisons.csv',index=False)
    pd.DataFrame(cases).to_csv(out/'case_temporal_comparisons.csv',index=False)
    return totals

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'work/phase50')
    ap.add_argument('--offline-inputs',type=Path,help='saved research/phase50 directory; no network; missing arrays fail explicitly')
    ap.add_argument('--max-sites',type=int);args=ap.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    cache_root=args.offline_inputs.resolve() if args.offline_inputs else None
    cache_metadata={}
    for source_log in [ROOT/'research/phase50/source_reads.csv',out/'source_reads.csv']:
        if source_log.exists():
            for r in pd.read_csv(source_log).itertuples():
                if hasattr(r,'retrieval_date_utc'):cache_metadata[(r.sample_id,r.month)]=r.retrieval_date_utc
    if cache_root:
        picked=pd.read_csv(cache_root/'frozen_67_site_roster.csv')
        doc=json.loads((cache_root/'reconstructed_all187_windows.geojson').read_text())
        geoms={f['properties']['sample_id']:shape(f['geometry']) for f in doc['features']}
        provenance=json.loads((cache_root/'summary.json').read_text())['provenance']
    else:picked,geoms,provenance=reconstruct(out)
    if args.max_sites:picked=picked.head(args.max_sites)
    observations=[];comparisons=[];exclusions=[];replication=[]
    for row in picked.itertuples(index=False):
        schemes=comparison_dates(row.event_month)
        if 'seasonal' not in schemes:exclusions.append(dict(sample_id=row.sample_id,scheme='seasonal',reason='prior_preceding_month_before_2022_same_edition_boundary'))
        months={shift(row.event_month,years=-1),row.event_month}
        months.update(x for dates in schemes.values() for x in dates)
        values={};reference=None
        for ym in sorted(months):
            try:
                if cache_root and not (cache_root/'pixels'/f'{row.sample_id}_{ym}.npz').exists():
                    raise FileNotFoundError('offline pixel array missing; network prohibited')
                v,audit=read_month(row,geoms[row.sample_id],ym,out,cache_root,cache_metadata)
                if reference is not None:
                    assert v['tf']==reference['tf'] and np.array_equal(v['mask'],reference['mask']),'unaligned_grid'
                reference=v;values[ym]=v
            except Exception as e:
                audit=dict(sample_id=row.sample_id,disasterNumber=int(row.disasterNumber),month=ym,
                           source_url=source_url(row.anchor_lon,row.anchor_lat,ym),status='read_failed',
                           error=type(e).__name__+': '+str(e)[:180])
            observations.append(audit)
        before=shift(row.event_month,years=-1)
        if before in values and row.event_month in values:
            a=values[before]['values'];b=values[row.event_month]['values']
            dx,dy=abs(reference['tf'].a),abs(reference['tf'].e)
            pixarea=abs(GEOD.geometry_area_perimeter(box(row.anchor_lon-dx/2,row.anchor_lat-dy/2,row.anchor_lon+dx/2,row.anchor_lat+dy/2))[0])/1e6
            gain=int(((a==1)&(b==2)).sum());obs=float((b>0).mean())
            logged_gain=row.historical_event_gain_km2_rounded
            logged_obs=row.historical_event_observed
            replication.append(dict(sample_id=row.sample_id,event_gain_cells=gain,event_gain_km2_original_approx=gain*pixarea,
                logged_event_gain_rounded=logged_gain,logged_event_observable_fraction=logged_obs,
                new_event_observable_fraction=obs,
                logged_gain_matches_round4=bool(round(gain*pixarea,4)==logged_gain) if np.isfinite(logged_gain) else None,
                logged_observability_matches=bool(abs(obs-logged_obs)<1e-12) if np.isfinite(logged_obs) else None))
        for scheme,dates in schemes.items():
            if any(x not in values for x in dates):
                exclusions.append(dict(sample_id=row.sample_id,scheme=scheme,reason='missing_source_month'));continue
            for denom in ('common_pixels','pair_specific'):
                for threshold in (.3,.6,.8):
                    result=compare(values,dates,threshold,denom)
                    if result:
                        comparisons.append(dict(sample_id=row.sample_id,disasterNumber=int(row.disasterNumber),
                            county_fips=int(row.county_fips),site_index=int(row.site_index),scheme=scheme,
                            denominator=denom,threshold=threshold,event_before=dates[0],event_after=dates[1],
                            placebo_before=dates[2],placebo_after=dates[3],**result))
        # Persist after every site; abrupt interruptions lose at most one site.
        pd.DataFrame(observations).to_csv(out/'source_reads.csv',index=False)
        pd.DataFrame(exclusions).to_csv(out/'exclusions.csv',index=False)
        pd.DataFrame(replication).to_csv(out/'historical_log_replication.csv',index=False)
        pd.DataFrame(comparisons).to_csv(out/'site_temporal_comparisons.csv',index=False)
        print('PHASE50_SITE',row.sample_id,'months_ok',len(values),'of',len(months),flush=True)
    totals=summarize(comparisons,out)
    summary=dict(status='completed' if len(picked)==67 else 'partial_smoke_run',sites_attempted=len(picked),
        reads_attempted=len(observations),reads_successful=sum(x['status']=='read_ok' for x in observations),
        comparisons=len(comparisons),provenance=provenance,
        inference='exploratory seasonal water variation; placebo is not known flood-free; no FPR or flood extent',
        primary_metric='event-minus-placebo gain km2 per100 common observed km2 at >=30%, 63 anchors',
        original_csv_reproduction=('exact originals recovered; 187 geometries and 67 integer gains verified by archive audit'
            if (ROOT/'research/recovered_artifacts/RECOVERY_AUDIT.json').exists() else 'original archive not available; log checks only'),
        scientific_stop='no funding prediction tuning; no certified inundation/exposure',
        aggregates=[r for r in totals if r['threshold']==.3 and r['denominator']=='common_pixels'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
