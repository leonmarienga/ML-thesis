#!/usr/bin/env python3
"""Bounded actual-source retrieval and all-hazard support audit; no model fitting.

Regional/time candidate overlaps are explicitly unverified, never accepted
disaster-linked severity. One-year NOAA pilot is not full-cohort coverage.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, io, json, re
import numpy as np
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/source_audit';OUT.mkdir(parents=True,exist_ok=True)
RAW=OUT/'raw';RAW.mkdir(exist_ok=True)
LOG=[]

def retrieve(name,url,params=None,limit=25_000_000):
    path=RAW/name
    stamp=datetime.now(timezone.utc).isoformat()
    item=dict(name=name,url=url,params=json.dumps(params or {},sort_keys=True),retrieved_utc=stamp)
    try:
        with requests.get(url,params=params,timeout=(20,50),stream=True) as r:
            item.update(actual_download_url=r.url,http_status=r.status_code)
            r.raise_for_status();parts=[];n=0
            for chunk in r.iter_content(256*1024):
                n+=len(chunk)
                if n>limit:raise ValueError('bounded_download_size_exceeded')
                parts.append(chunk)
            content=b''.join(parts);path.write_bytes(content)
            item.update(status='retrieved',bytes=len(content),sha256=hashlib.sha256(content).hexdigest(),local_path=str(path.relative_to(ROOT)))
    except Exception as e:
        item.update(status='failed',error=type(e).__name__)
        content=None
    LOG.append(item)
    pd.DataFrame(LOG).to_csv(OUT/'source_retrieval_log.csv',index=False)
    print('SOURCE',name,item['status'],item.get('http_status'),flush=True)
    return content

def band(values):
    return pd.cut(values.clip(lower=0),[0,1e5,1e6,50e6,200e6,500e6,np.inf],right=False,
                  labels=['0-100K','100K-1M','1M-50M','50-200M','200-500M','500M+'])

def main():
    master=pd.read_excel(ROOT/'master_openfema_40plus.xlsx')
    assert len(master)==971 and master.disasterNumber.nunique()==971
    master['funding_band']=band(pd.to_numeric(master.totalObligatedFunding))
    inventory=master.groupby('incidentType').agg(records=('disasterNumber','size'),
        fiscal_years=('fyDeclared','nunique'),states=('state','nunique'),
        earliest_fy=('fyDeclared','min'),latest_fy=('fyDeclared','max'))
    inventory.to_csv(OUT/'hazard_inventory.csv')
    pd.crosstab(master.incidentType,master.funding_band).to_csv(OUT/'hazard_band_support.csv')
    pd.crosstab(master.incidentType,master.fyDeclared).to_csv(OUT/'hazard_year_support.csv')
    print('MASTER_COUNTS',len(master),int((master.incidentType!='Biological').sum()),flush=True)
    # A bounded current official archive; physical fields inspected separately
    # from reporting counts, damage estimates and future publication dates.
    directory='https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/'
    listing=retrieve('noaa_directory.html',directory)
    if listing:
        names=sorted(set(re.findall(r'StormEvents_details-ftp_v1\.0_d2024_c\d{8}\.csv\.gz',listing.decode())))
        if names:
            content=retrieve(names[-1],directory+names[-1])
            if content:
                noaa=pd.read_csv(io.BytesIO(content),compression='gzip',low_memory=False)
                assert noaa.EVENT_ID.notna().all() and noaa.EVENT_ID.is_unique
                fields=['MAGNITUDE','MAGNITUDE_TYPE','TOR_F_SCALE','TOR_LENGTH','TOR_WIDTH','FLOOD_CAUSE']
                rows=[]
                for hazard,g in noaa.groupby('EVENT_TYPE'):
                    row=dict(source_event_type=hazard,records=len(g),classification='UNVERIFIED CANDIDATE for FEMA linkage')
                    for field in fields:
                        if field in g:row[field+'_nonmissing']=int(g[field].notna().sum())
                    rows.append(row)
                pd.DataFrame(rows).to_csv(OUT/'noaa2024_physical_field_availability.csv',index=False)
                cols=[c for c in ['EVENT_ID','EPISODE_ID','YEAR','STATE','STATE_FIPS','CZ_TYPE','CZ_FIPS','EVENT_TYPE','BEGIN_DATE_TIME','END_DATE_TIME']+fields if c in noaa]
                noaa[cols].to_csv(OUT/'noaa2024_physical_records.csv',index=False)
    nhc='https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2024-040425.txt'
    content=retrieve('hurdat2-1851-2024-040425.txt',nhc)
    if content:
        storms=[];fixes=[];current=None
        for line in content.decode().splitlines():
            p=[z.strip() for z in line.split(',')]
            if re.fullmatch(r'[A-Z]{2}\d{6}',p[0]):
                current=dict(storm_id=p[0],name=p[1],year=int(p[0][-4:]),declared_fixes=int(p[2]))
                storms.append(current)
            elif p[0].isdigit() and current:
                fixes.append(dict(storm_id=current['storm_id'],name=current['name'],date=p[0],time=p[1],
                    status=p[3],latitude=p[4],longitude=p[5],max_wind_kt=int(p[6]),pressure_hpa=int(p[7]),
                    classification='VERIFIED DERIVED MEASUREMENT; final best track; FEMA match unverified'))
        assert len(fixes)==sum(s['declared_fixes'] for s in storms)
        pd.DataFrame(fixes).query('date >= "20100101" and date <= "20241231"').to_csv(OUT/'nhc_atlantic2010_24_fixes.csv',index=False)
        pd.DataFrame(storms).to_csv(OUT/'nhc_atlantic_storm_index.csv',index=False)
    # Regional catalog feasibility for ALL eight earthquake declarations.
    # A regional earthquake can produce shaking beyond a state boundary. These
    # broad boxes generate candidates only and do not establish event linkage.
    boxes={'CA':(-126,31,-113,43),'VA':(-85,35,-74,41),'DC':(-85,35,-74,41),
           'AK':(-180,49,-129,73),'PR':(-70,16,-63,21),'UT':(-115,36,-108,43)}
    earthquake_rows=[];coverage=[]
    for r in master[master.incidentType=='Earthquake'].itertuples():
        start=pd.Timestamp(r.incidentBeginDate).date()
        end=(pd.Timestamp(r.incidentBeginDate)+pd.Timedelta(days=2)).date()
        xmin,ymin,xmax,ymax=boxes[r.state]
        params=dict(format='geojson',starttime=str(start),endtime=str(end),minmagnitude=4.5,
                    minlongitude=xmin,maxlongitude=xmax,minlatitude=ymin,maxlatitude=ymax,limit=200,orderby='time')
        content=retrieve(f'usgs_earthquake_{r.disasterNumber}.json','https://earthquake.usgs.gov/fdsnws/event/1/query',params)
        candidates=[]
        if content:
            doc=json.loads(content);candidates=doc.get('features',[])
            if len(candidates)>=200:raise RuntimeError('earthquake candidate limit reached; pagination needed')
            for item in candidates:
                p=item['properties'];lon,lat,depth=item['geometry']['coordinates']
                earthquake_rows.append(dict(disasterNumber=r.disasterNumber,state=r.state,usgs_id=item['id'],
                    magnitude=p['mag'],magnitude_type=p['magType'],depth_km=depth,latitude=lat,longitude=lon,
                    observation_epoch_ms=p['time'],last_updated_epoch_ms=p['updated'],
                    source_event_url=p['url'],matching_status='UNVERIFIED CANDIDATE',
                    rule='regional box and first48h; no certified earthquake-disaster linkage'))
        coverage.append(dict(disasterNumber=r.disasterNumber,candidate_records=len(candidates),retrieval_ok=content is not None))
    pd.DataFrame(earthquake_rows).to_csv(OUT/'earthquake_regional_candidates.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'earthquake_query_coverage.csv',index=False)
    # Authoritative wildfire metadata is accepted as documentation only. No
    # perimeter or burned-area value is fabricated when services fail.
    retrieve('mtbs_direct_download.html','https://burnseverity.cr.usgs.gov/direct-download?product=MTBS')
    retrieve('mtbs_program.html','https://burnseverity.cr.usgs.gov/products/mtbs')
    retrieve('mtbs_mapserver_metadata.json','https://apps.fs.usda.gov/arcx/rest/services/EDW/EDW_MTBS_01/MapServer/',{'f':'pjson'})
    retrieve('usgs_flood_events.json','https://stn.wim.usgs.gov/STNServices/Events.json')
    retrieve('nws_mississippi2023.html','https://www.weather.gov/arx/flood2023')
    retrieve('jrc_official_download.html','https://global-surface-water.appspot.com/download')
    retrieve('jrc_mirror_description.html','https://source.coop/epoch/jrc-global-surface-water-monthly')
    retrieve('noaa_physical_units.pdf',directory+'Storm-Data-Bulk-csv-Format.pdf')
    (OUT/'summary.json').write_text(json.dumps(dict(
        master_rows=len(master),nonbiological_rows=int((master.incidentType!='Biological').sum()),
        hazards=len(inventory),drought_rows=int((master.incidentType=='Drought').sum()),
        negative_funding_rows=int((master.totalObligatedFunding<0).sum()),
        completed='hazard inventory and bounded real-source availability retrieval',
        uncompleted='disaster-specific physical feature certification, full multi-hazard enrichment and independent validation',
        trained_models=0,accepted_unverified_features=0,
        limits='NOAA2024 only; Atlantic HURDAT only; regional earthquake candidates not independently linked; MTBS catalog not measured fire extent'),indent=2))

if __name__=='__main__':main()
