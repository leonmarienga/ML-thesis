from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr

MASTER=Path('master_openfema_40plus.xlsx')
OUT=Path('noaa_physical_severity_signal_audit_results')
OUT.mkdir(exist_ok=True)
BASE='https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/'

STATE_NAME={
'AL':'ALABAMA','AK':'ALASKA','AZ':'ARIZONA','AR':'ARKANSAS','CA':'CALIFORNIA','CO':'COLORADO','CT':'CONNECTICUT','DE':'DELAWARE','DC':'DISTRICT OF COLUMBIA',
'FL':'FLORIDA','GA':'GEORGIA','HI':'HAWAII','ID':'IDAHO','IL':'ILLINOIS','IN':'INDIANA','IA':'IOWA','KS':'KANSAS','KY':'KENTUCKY','LA':'LOUISIANA',
'ME':'MAINE','MD':'MARYLAND','MA':'MASSACHUSETTS','MI':'MICHIGAN','MN':'MINNESOTA','MS':'MISSISSIPPI','MO':'MISSOURI','MT':'MONTANA','NE':'NEBRASKA',
'NV':'NEVADA','NH':'NEW HAMPSHIRE','NJ':'NEW JERSEY','NM':'NEW MEXICO','NY':'NEW YORK','NC':'NORTH CAROLINA','ND':'NORTH DAKOTA','OH':'OHIO',
'OK':'OKLAHOMA','OR':'OREGON','PA':'PENNSYLVANIA','RI':'RHODE ISLAND','SC':'SOUTH CAROLINA','SD':'SOUTH DAKOTA','TN':'TENNESSEE','TX':'TEXAS',
'UT':'UTAH','VT':'VERMONT','VA':'VIRGINIA','WA':'WASHINGTON','WV':'WEST VIRGINIA','WI':'WISCONSIN','WY':'WYOMING','PR':'PUERTO RICO',
'VI':'VIRGIN ISLANDS','GU':'GUAM','MP':'NORTHERN MARIANA ISLANDS','AS':'AMERICAN SAMOA'
}

TYPE_PATTERNS={
'Hurricane':['HURRICANE','HURRICANE (TYPHOON)'],
'Tropical Storm':['TROPICAL STORM'],
'Typhoon':['HURRICANE (TYPHOON)','TYPHOON'],
'Fire':['WILDFIRE'],
'Flood':['FLOOD','FLASH FLOOD']
}

def band(x):
    if x<=50e6:return 'below50M'
    if x<=200e6:return '50M-200M'
    if x<=500e6:return '200M-500M'
    return '500M+'

def discover_files(years):
    r=requests.get(BASE,timeout=120,headers={'User-Agent':'ML-thesis-research/1.0'})
    r.raise_for_status()
    hrefs=re.findall(r'href=["\']([^"\']+)["\']',r.text)
    found={}
    for y in years:
        pat=re.compile(rf'^StormEvents_details-ftp_v1\.0_d{y}_c(\d{{8}})\.csv\.gz$')
        cand=[]
        for h in hrefs:
            m=pat.match(h)
            if m:cand.append((m.group(1),h))
        if cand:
            cand.sort()
            found[y]=cand[-1][1]
    return found

def load_noaa(years):
    files=discover_files(years)
    missing=[y for y in years if y not in files]
    if missing:
        raise RuntimeError(f'NOAA details files not found for years {missing}')
    frames=[]
    keep=['BEGIN_YEARMONTH','BEGIN_DAY','BEGIN_TIME','END_YEARMONTH','END_DAY','END_TIME','STATE','EVENT_TYPE','CZ_TYPE','CZ_FIPS','CZ_NAME',
          'INJURIES_DIRECT','INJURIES_INDIRECT','DEATHS_DIRECT','DEATHS_INDIRECT','MAGNITUDE','MAGNITUDE_TYPE','TOR_F_SCALE',
          'BEGIN_LAT','BEGIN_LON','END_LAT','END_LON','EPISODE_ID','EVENT_ID']
    meta=[]
    for y in years:
        url=BASE+files[y]
        rr=requests.get(url,timeout=180,headers={'User-Agent':'ML-thesis-research/1.0'})
        rr.raise_for_status()
        df=pd.read_csv(io.BytesIO(rr.content),compression='gzip',low_memory=False)
        cols=[c for c in keep if c in df.columns]
        df=df[cols].copy()
        df['sourceYear']=y
        frames.append(df)
        meta.append({'year':y,'file':files[y],'rows':int(len(df))})
    (OUT/'noaa_files.json').write_text(json.dumps(meta,indent=2))
    d=pd.concat(frames,ignore_index=True)

    def dt_from_parts(prefix):
        ym=pd.to_numeric(d.get(prefix+'_YEARMONTH'),errors='coerce')
        day=pd.to_numeric(d.get(prefix+'_DAY'),errors='coerce')
        tim=pd.to_numeric(d.get(prefix+'_TIME'),errors='coerce').fillna(0)
        yy=(ym//100).astype('Int64'); mm=(ym%100).astype('Int64')
        hh=(tim//100).astype('Int64'); mi=(tim%100).astype('Int64')
        txt=(yy.astype(str)+'-'+mm.astype(str).str.zfill(2)+'-'+day.astype('Int64').astype(str).str.zfill(2)+' '+hh.astype(str).str.zfill(2)+':'+mi.astype(str).str.zfill(2))
        return pd.to_datetime(txt,errors='coerce')
    d['begin_dt']=dt_from_parts('BEGIN')
    d['end_dt']=dt_from_parts('END')
    d['STATE']=d['STATE'].astype(str).str.upper().str.strip()
    d['EVENT_TYPE']=d['EVENT_TYPE'].astype(str).str.upper().str.strip()
    for c in ['INJURIES_DIRECT','INJURIES_INDIRECT','DEATHS_DIRECT','DEATHS_INDIRECT','MAGNITUDE','BEGIN_LAT','BEGIN_LON','END_LAT','END_LON']:
        if c not in d.columns:d[c]=np.nan
        d[c]=pd.to_numeric(d[c],errors='coerce')
    return d

def type_mask(event_type, incident_type):
    pats=TYPE_PATTERNS.get(incident_type,[])
    if not pats:return pd.Series(False,index=event_type.index)
    m=pd.Series(False,index=event_type.index)
    for p in pats:
        m |= event_type.str.contains(p,case=False,regex=False,na=False)
    return m

def aggregate_match(noaa,row):
    st=STATE_NAME.get(str(row.state))
    if not st:return {}
    b=pd.to_datetime(row.incidentBeginDate,errors='coerce')
    e=pd.to_datetime(row.incidentEndDate,errors='coerce')
    dec=pd.to_datetime(row.declarationDate,errors='coerce')
    if pd.isna(b):b=dec
    if pd.isna(e):e=dec
    if pd.isna(b) or pd.isna(e):return {}
    lo=b-pd.Timedelta(days=3); hi=e+pd.Timedelta(days=3)
    g=noaa[(noaa.STATE==st)&type_mask(noaa.EVENT_TYPE,str(row.incidentType))].copy()
    # Any NOAA event overlapping the FEMA incident window.
    g=g[(g.begin_dt<=hi)&(g.end_dt.fillna(g.begin_dt)>=lo)]
    if g.empty:
        # fallback: state + hazard within a tighter declaration window
        lo2=dec-pd.Timedelta(days=10); hi2=dec+pd.Timedelta(days=10)
        g=noaa[(noaa.STATE==st)&type_mask(noaa.EVENT_TYPE,str(row.incidentType))&(noaa.begin_dt<=hi2)&(noaa.end_dt.fillna(noaa.begin_dt)>=lo2)].copy()
    if g.empty:return {'noaaMatchRows':0}
    injuries=(g.INJURIES_DIRECT.fillna(0)+g.INJURIES_INDIRECT.fillna(0))
    deaths=(g.DEATHS_DIRECT.fillna(0)+g.DEATHS_INDIRECT.fillna(0))
    mag=g.MAGNITUDE.dropna()
    lat=pd.concat([g.BEGIN_LAT,g.END_LAT]).dropna(); lon=pd.concat([g.BEGIN_LON,g.END_LON]).dropna()
    county_keys=(g.CZ_TYPE.astype(str)+'|'+g.CZ_FIPS.astype(str)+'|'+g.CZ_NAME.astype(str)) if 'CZ_NAME' in g else pd.Series(dtype=str)
    event_durations=(g.end_dt.fillna(g.begin_dt)-g.begin_dt).dt.total_seconds()/86400
    return {
        'noaaMatchRows':int(len(g)),
        'noaaUniqueEvents':int(g.EVENT_ID.nunique()) if 'EVENT_ID' in g else int(len(g)),
        'noaaUniqueEpisodes':int(g.EPISODE_ID.nunique()) if 'EPISODE_ID' in g else 0,
        'noaaUniqueAreas':int(county_keys.nunique()) if len(county_keys) else 0,
        'noaaDeaths':float(deaths.sum()),
        'noaaInjuries':float(injuries.sum()),
        'noaaMaxMagnitude':float(mag.max()) if len(mag) else np.nan,
        'noaaMeanMagnitude':float(mag.mean()) if len(mag) else np.nan,
        'noaaMagnitudeObservedShare':float(g.MAGNITUDE.notna().mean()),
        'noaaLatSpan':float(lat.max()-lat.min()) if len(lat)>1 else 0.0,
        'noaaLonSpan':float(lon.max()-lon.min()) if len(lon)>1 else 0.0,
        'noaaMedianEventDurationDays':float(event_durations.median()) if event_durations.notna().any() else np.nan,
    }

def main():
    master=pd.read_excel(MASTER)
    master['target']=pd.to_numeric(master.totalObligatedFunding,errors='coerce').fillna(0).clip(lower=0)
    relevant=master[master.incidentType.isin(TYPE_PATTERNS)].copy()
    for c in ['incidentBeginDate','incidentEndDate','declarationDate']:
        relevant[c]=pd.to_datetime(relevant[c],errors='coerce').dt.tz_localize(None)
    years=sorted(set(relevant.incidentBeginDate.dropna().dt.year.astype(int))|set(relevant.declarationDate.dropna().dt.year.astype(int)))
    years=[y for y in years if 2010<=y<=2024]
    noaa=load_noaa(years)
    rows=[]
    for _,r in relevant.iterrows():
        z={'disasterNumber':int(r.disasterNumber),'state':str(r.state),'fyDeclared':int(r.fyDeclared),'incidentType':str(r.incidentType),
           'incidentBeginDate':r.incidentBeginDate,'incidentEndDate':r.incidentEndDate,'declarationDate':r.declarationDate,
           'target':float(r.target),'band':band(float(r.target))}
        z.update(aggregate_match(noaa,r))
        rows.append(z)
    out=pd.DataFrame(rows)
    metric_cols=['noaaMatchRows','noaaUniqueEvents','noaaUniqueEpisodes','noaaUniqueAreas','noaaDeaths','noaaInjuries','noaaMaxMagnitude','noaaMeanMagnitude',
                 'noaaMagnitudeObservedShare','noaaLatSpan','noaaLonSpan','noaaMedianEventDurationDays']
    for c in metric_cols:
        if c not in out.columns:out[c]=np.nan

    # Development-only target-free within-hazard percentile layer.
    # If promising, these transforms must be rebuilt fold-safely in final nested validation.
    pct_cols=[]
    for typ,gidx in out.groupby('incidentType').groups.items():
        idx=list(gidx)
        for c in metric_cols:
            x=pd.to_numeric(out.loc[idx,c],errors='coerce')
            if x.notna().sum()>=4 and x.nunique(dropna=True)>1:
                pc=c+'_hazard_pct'
                out.loc[idx,pc]=x.rank(pct=True,method='average')
                if pc not in pct_cols:pct_cols.append(pc)
    out['noaaSeverityComposite']=out[pct_cols].mean(axis=1,skipna=True) if pct_cols else np.nan
    out.to_csv(OUT/'noaa_nonfinancial_severity_features.csv',index=False)

    high=out[out.target>50e6].copy()
    high.to_csv(OUT/'noaa_high_value_cases.csv',index=False)
    assoc=[]
    y=np.log1p(high.target.to_numpy(float))
    test_cols=metric_cols+pct_cols+['noaaSeverityComposite']
    for c in test_cols:
        x=pd.to_numeric(high[c],errors='coerce')
        m=x.notna() & np.isfinite(y)
        if m.sum()>=4 and x[m].nunique()>1:
            rho,p=spearmanr(x[m],y[m])
            assoc.append({'feature':c,'n':int(m.sum()),'spearman_rho_log_funding':float(rho),'p_value':float(p)})
    a=pd.DataFrame(assoc)
    if len(a):a=a.sort_values('spearman_rho_log_funding',key=lambda s:s.abs(),ascending=False)
    a.to_csv(OUT/'severity_association_screen.csv',index=False)

    byband=high.groupby(['band','incidentType']).agg(n=('disasterNumber','size'),meanComposite=('noaaSeverityComposite','mean'),
        medianComposite=('noaaSeverityComposite','median'),meanRows=('noaaMatchRows','mean'),meanDeaths=('noaaDeaths','mean'),meanInjuries=('noaaInjuries','mean')).reset_index()
    byband.to_csv(OUT/'severity_by_band_type.csv',index=False)

    summary={
      'relevant_master_count':int(len(out)),
      'high_over_50m_count':int(len(high)),
      'high_type_counts':high.incidentType.value_counts().to_dict(),
      'match_coverage':{t:{'n':int(len(g)),'matched':int((pd.to_numeric(g.noaaMatchRows,errors='coerce').fillna(0)>0).sum())} for t,g in out.groupby('incidentType')},
      'best_absolute_spearman':a.head(20).to_dict('records') if len(a) else [],
      'by_band_type':byband.to_dict('records'),
      'guardrails':[
        'No NOAA property-damage or crop-damage dollar fields are loaded or used.',
        'NOAA records are matched only by target-free state, hazard type, and incident/declaration time window.',
        'Hazard percentile/composite values are a development signal audit. Any model using them must recompute normalization inside each outer training fold before a final claim.'
      ]
    }
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str))

if __name__=='__main__':main()
