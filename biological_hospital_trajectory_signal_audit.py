from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import cdc_biological_severity_signal_audit as cdc_audit

MASTER = Path('master_openfema_40plus.xlsx')
OUT = Path('biological_hospital_trajectory_signal_audit_results')
OUT.mkdir(exist_ok=True)
HHS_URL = 'https://healthdata.gov/resource/g62h-syeh.json'
POP = cdc_audit.POP

FOCUS = [4480, 4482, 4485, 4486, 4489, 4515]


def num(x):
    return pd.to_numeric(x, errors='coerce').replace([np.inf, -np.inf], np.nan)


def band6(x):
    if x <= 1e5: return '0-100K'
    if x <= 1e6: return '100K-1M'
    if x <= 50e6: return '1M-50M'
    if x <= 200e6: return '50M-200M'
    if x <= 500e6: return '200M-500M'
    return '500M+'


def fetch_hhs():
    headers={'User-Agent':'ML-thesis-research/1.0'}
    rows=[]; offset=0; limit=50000
    while True:
        params={
            '$limit':limit,
            '$offset':offset,
            '$where':"date >= '2020-01-01T00:00:00.000' and date <= '2021-12-31T23:59:59.999'",
        }
        r=requests.get(HHS_URL,params=params,timeout=180,headers=headers)
        if r.status_code >= 400:
            raise RuntimeError(f'HHS resource HTTP {r.status_code}: {r.text[:500]}')
        batch=r.json()
        if not batch: break
        rows.extend(batch)
        if len(batch)<limit: break
        offset += limit
    if not rows:
        raise RuntimeError('HHS hospital timeseries returned no rows')
    d=pd.DataFrame(rows)
    d.columns=[str(c).strip().lower().replace(' ','_') for c in d.columns]
    date_col=next((c for c in ['date','collection_week','reporting_cutoff_start','week_ending_date'] if c in d.columns),None)
    if date_col is None:
        raise RuntimeError(f'No HHS date field found. Columns={d.columns.tolist()}')
    state_col=next((c for c in ['state','state_abbr','jurisdiction'] if c in d.columns),None)
    if state_col is None:
        raise RuntimeError(f'No HHS state field found. Columns={d.columns.tolist()}')
    d['date']=pd.to_datetime(d[date_col],errors='coerce').dt.tz_localize(None)
    d['state_abbr']=d[state_col].astype(str).str.strip()
    d=d.dropna(subset=['date','state_abbr']).sort_values(['state_abbr','date']).copy()
    (OUT/'hhs_schema.json').write_text(json.dumps({'date_col':date_col,'state_col':state_col,'columns':d.columns.tolist(),'rows':int(len(d))},indent=2))
    return d


def window_sum(g, col, end, days):
    if col not in g.columns: return np.nan
    z=g[(g.date<=end)&(g.date>end-pd.Timedelta(days=days))]
    vals=num(z[col]).clip(lower=0)
    return float(vals.sum()) if vals.notna().any() else np.nan


def last_value(g, col, end):
    if col not in g.columns: return np.nan
    z=g[g.date<=end].sort_values('date')
    if z.empty: return np.nan
    v=num(pd.Series([z.iloc[-1].get(col)])).iloc[0]
    return float(v) if pd.notna(v) else np.nan


def cdc_features(cdc_state, dt, pop):
    g=cdc_state[cdc_state.date<=dt].sort_values('date').copy()
    if g.empty: return {}
    last=g.iloc[-1]
    first_pos=g[num(g['tot_cases']).fillna(0)>0]
    first_date=first_pos.date.min() if len(first_pos) else pd.NaT

    def w(col,days): return window_sum(g,col,dt,days)
    def cum(col):
        v=num(pd.Series([last.get(col)])).iloc[0]
        return float(v) if pd.notna(v) else np.nan

    n7=w('new_case',7); n14=w('new_case',14); n28=w('new_case',28)
    d7=w('new_death',7); d14=w('new_death',14); d28=w('new_death',28)
    prev7=max((n14-n7),0) if pd.notna(n14) and pd.notna(n7) else np.nan
    prev14=max((n28-n14),0) if pd.notna(n28) and pd.notna(n14) else np.nan

    out={
        'cumCases':cum('tot_cases'),'cumDeaths':cum('tot_death'),
        'newCases7d':n7,'newCases14d':n14,'newCases28d':n28,
        'newDeaths7d':d7,'newDeaths14d':d14,'newDeaths28d':d28,
        'caseGrowth7vsPrev7':(n7+1)/(prev7+1) if pd.notna(n7) and pd.notna(prev7) else np.nan,
        'caseGrowth14vsPrev14':(n14+1)/(prev14+1) if pd.notna(n14) and pd.notna(prev14) else np.nan,
        'deathGrowth7vsPrev7':(d7+1)/(max(d14-d7,0)+1) if pd.notna(d14) and pd.notna(d7) else np.nan,
        'daysSinceFirstCase':float((dt-first_date).days) if pd.notna(first_date) else np.nan,
    }
    if pop and pop>0:
        for c in ['cumCases','cumDeaths','newCases7d','newCases14d','newCases28d','newDeaths7d','newDeaths14d','newDeaths28d']:
            out[c+'Per100k']=out[c]/pop*100000 if pd.notna(out[c]) else np.nan
    return out


def hhs_features(hhs_state, dt, pop):
    g=hhs_state[hhs_state.date<=dt].sort_values('date').copy()
    if g.empty: return {'hhsAvailableAtOrBeforeDeclaration':0}
    last=g.iloc[-1]
    fields=[
        'inpatient_beds_used_covid','inpatient_bed_covid_utilization',
        'adult_icu_bed_covid_utilization','inpatient_beds_utilization',
        'critical_staffing_shortage_today_yes',
        'previous_day_admission_adult_covid_confirmed',
        'staffed_icu_adult_patients_confirmed_and_suspected_covid',
        'total_staffed_adult_icu_beds','inpatient_beds','inpatient_beds_used',
    ]
    out={'hhsAvailableAtOrBeforeDeclaration':1,'hhsObservationDate':last.date,'hhsLagDays':int((dt-last.date).days)}
    for c in fields:
        out[c]=last_value(g,c,dt)
    if pop and pop>0:
        for c in ['inpatient_beds_used_covid','previous_day_admission_adult_covid_confirmed','staffed_icu_adult_patients_confirmed_and_suspected_covid']:
            v=out.get(c,np.nan); out[c+'Per100k']=v/pop*100000 if pd.notna(v) else np.nan
    if pd.notna(out.get('inpatient_beds_used_covid')) and pd.notna(out.get('inpatient_beds')) and out['inpatient_beds']>0:
        out['covidShareTotalBeds']=out['inpatient_beds_used_covid']/out['inpatient_beds']
    else:
        out['covidShareTotalBeds']=np.nan
    if pd.notna(out.get('staffed_icu_adult_patients_confirmed_and_suspected_covid')) and pd.notna(out.get('total_staffed_adult_icu_beds')) and out['total_staffed_adult_icu_beds']>0:
        out['covidShareAdultICU']=out['staffed_icu_adult_patients_confirmed_and_suspected_covid']/out['total_staffed_adult_icu_beds']
    else:
        out['covidShareAdultICU']=np.nan
    return out


def state_snapshot_percentiles(cdc, dt, feature_builder):
    rows=[]
    for st,pop in POP.items():
        feat=feature_builder(cdc[cdc.state_abbr==st],dt,float(pop))
        if feat:
            rows.append({'state':st,**feat})
    if not rows: return pd.DataFrame()
    d=pd.DataFrame(rows)
    numeric=[c for c in d.columns if c!='state']
    for c in numeric:
        x=num(d[c])
        if x.notna().sum()>=3:
            d[c+'_statePct']=x.rank(pct=True,method='average')
    return d


def add_state_percentiles(rows, cdc, hhs):
    out=[]
    for row in rows:
        dt=row['declarationDate']; st=row['state']
        c_snap=state_snapshot_percentiles(cdc,dt,cdc_features)
        if len(c_snap) and st in set(c_snap.state):
            rr=c_snap[c_snap.state==st].iloc[-1]
            for c in c_snap.columns:
                if c.endswith('_statePct'):
                    row['cdc_'+c]=rr[c]
        hrows=[]
        for hs,pop in POP.items():
            feat=hhs_features(hhs[hhs.state_abbr==hs],dt,float(pop))
            if feat.get('hhsAvailableAtOrBeforeDeclaration')==1:
                hrows.append({'state':hs,**feat})
        if hrows:
            hd=pd.DataFrame(hrows)
            for c in [x for x in hd.columns if x not in ['state','hhsObservationDate']]:
                x=num(hd[c])
                if x.notna().sum()>=3:
                    hd[c+'_statePct']=x.rank(pct=True,method='average')
            if st in set(hd.state):
                rr=hd[hd.state==st].iloc[-1]
                for c in hd.columns:
                    if c.endswith('_statePct'):
                        row['hhs_'+c]=rr[c]
        out.append(row)
    return pd.DataFrame(out)


def signal_screen(high):
    ylog=np.log1p(num(high.target).to_numpy(float))
    rows=[]
    exclude={'disasterNumber','target','fyDeclared'}
    for c in high.columns:
        if c in exclude or c in ['state','band6','declarationDate','hhsObservationDate']:
            continue
        x=num(high[c])
        m=x.notna() & np.isfinite(ylog)
        if m.sum()>=4 and x[m].nunique()>1:
            rho,p=spearmanr(x[m],ylog[m])
            rows.append({'feature':c,'n':int(m.sum()),'spearman_rho_log_funding':float(rho),'p_value':float(p)})
    return pd.DataFrame(rows).sort_values('spearman_rho_log_funding',key=lambda s:s.abs(),ascending=False) if rows else pd.DataFrame()


def top_boundary_auc(high):
    z=high[high.band6.isin(['200M-500M','500M+'])].copy()
    y=(z.band6=='500M+').astype(int).to_numpy()
    rows=[]
    if len(np.unique(y))<2: return pd.DataFrame()
    for c in z.columns:
        if c in ['disasterNumber','state','band6','declarationDate','hhsObservationDate','target','fyDeclared']:
            continue
        x=num(z[c]); m=x.notna()
        if m.sum()>=4 and len(np.unique(y[m]))==2 and x[m].nunique()>1:
            auc=float(roc_auc_score(y[m],x[m]))
            rows.append({'feature':c,'n':int(m.sum()),'auc_500Mplus':auc,'orientation_free_auc':max(auc,1-auc)})
    return pd.DataFrame(rows).sort_values('orientation_free_auc',ascending=False) if rows else pd.DataFrame()


def main():
    master=pd.read_excel(MASTER)
    master['target']=num(master['totalObligatedFunding']).fillna(0).clip(lower=0)
    master['band6']=master.target.map(band6)
    master['declarationDateParsed']=pd.to_datetime(master['declarationDate'],errors='coerce').dt.tz_localize(None)
    bio=master[master.incidentType.astype(str).str.lower().eq('biological')].copy()
    cdc=cdc_audit.fetch_cdc()
    hhs=fetch_hhs()

    rows=[]
    for _,r in bio.iterrows():
        st=str(r.state); dt=r.declarationDateParsed; pop=float(POP.get(st,np.nan))
        row={'disasterNumber':int(r.disasterNumber),'state':st,'fyDeclared':int(r.fyDeclared),'declarationDate':dt,'target':float(r.target),'band6':r.band6,'population2010':pop}
        if pd.notna(dt):
            row.update(cdc_features(cdc[cdc.state_abbr==st],dt,pop))
            row.update(hhs_features(hhs[hhs.state_abbr==st],dt,pop))
        rows.append(row)

    feat=add_state_percentiles(rows,cdc,hhs)
    feat.to_csv(OUT/'biological_event_specific_features.csv',index=False)
    high=feat[feat.target>50e6].copy()
    high.to_csv(OUT/'biological_high_value_event_specific_features.csv',index=False)

    assoc=signal_screen(high)
    assoc.to_csv(OUT/'funding_association_screen.csv',index=False)
    auc=top_boundary_auc(high)
    auc.to_csv(OUT/'top_boundary_univariate_auc_screen.csv',index=False)

    focus=feat[feat.disasterNumber.isin(FOCUS)].sort_values('disasterNumber')
    focus.to_csv(OUT/'focus_2020_biological_cases.csv',index=False)

    hhs_cov=int(feat.hhsAvailableAtOrBeforeDeclaration.fillna(0).sum()) if 'hhsAvailableAtOrBeforeDeclaration' in feat else 0
    focus_cols=['disasterNumber','state','declarationDate','target','band6','cumCasesPer100k','cumDeathsPer100k','newCases14dPer100k','caseGrowth7vsPrev7','caseGrowth14vsPrev14','daysSinceFirstCase','hhsAvailableAtOrBeforeDeclaration','inpatient_beds_used_covidPer100k','adult_icu_bed_covid_utilization','covidShareAdultICU']
    focus_cols=[c for c in focus_cols if c in focus.columns]
    summary={
        'purpose':'Target-free Biological event-specific signal audit using declaration-time epidemic trajectory and retrospective HHS hospital/capacity observations.',
        'biological_count':int(len(feat)),
        'biological_high_over_50m_count':int(len(high)),
        'hhs_coverage_at_or_before_declaration_count':hhs_cov,
        'best_absolute_spearman':assoc.head(20).to_dict('records') if len(assoc) else [],
        'best_top_boundary_univariate_auc':auc.head(20).to_dict('records') if len(auc) else [],
        'focus_cases':focus[focus_cols].to_dict('records'),
        'guardrails':[
            'CDC observations are restricted to dates on or before each FEMA declaration.',
            'HHS hospital observations are also restricted to dates on or before each FEMA declaration, but the public HHS timeseries dataset was created later in 2020; therefore hospital variables are retrospective signal-audit features until contemporaneous operational availability is separately verified.',
            'No FEMA funding/obligation amount or funding-derived feature is used to construct the external features.',
            'This is a signal audit, not a leave-fiscal-year-out predictive claim, because nearly all informative Biological high-value cases are concentrated in fiscal year 2020.',
            'A feature family must be translated into a cross-hazard representation and then tested with strict outer leave-fiscal-year-out validation before it can support the global router.'
        ]
    }
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str),flush=True)


if __name__=='__main__':
    main()
