from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr

MASTER = Path('master_openfema_40plus.xlsx')
OUT = Path('cdc_biological_severity_signal_audit_results')
OUT.mkdir(exist_ok=True)
CDC_URL = 'https://data.cdc.gov/resource/pwn4-m3yp.json'

POP = {'AL':4779736,'AK':710231,'AZ':6392017,'AR':2915918,'CA':37253956,'CO':5029196,'CT':3574097,'DE':897934,'DC':601723,'FL':18801310,'GA':9687653,'HI':1360301,'ID':1567582,'IL':12830632,'IN':6483802,'IA':3046355,'KS':2853118,'KY':4339367,'LA':4533372,'ME':1328361,'MD':5773552,'MA':6547629,'MI':9883640,'MN':5303925,'MS':2967297,'MO':5988927,'MT':989415,'NE':1826341,'NV':2700551,'NH':1316470,'NJ':8791894,'NM':2059179,'NY':19378102,'NC':9535483,'ND':672591,'OH':11536504,'OK':3751351,'OR':3831074,'PA':12702379,'RI':1052567,'SC':4625364,'SD':814180,'TN':6346105,'TX':25145561,'UT':2763885,'VT':625741,'VA':8001024,'WA':6724540,'WV':1852994,'WI':5686986,'WY':563626,'PR':3725789,'AS':55519,'GU':159358,'MP':53883,'VI':106405}

def fetch_cdc():
    rows=[]; offset=0; limit=50000
    while True:
        params={'$limit':limit,'$offset':offset,'$order':'submission_date asc'}
        r=requests.get(CDC_URL,params=params,timeout=120,headers={'User-Agent':'ML-thesis-research/1.0'})
        r.raise_for_status(); batch=r.json()
        if not batch: break
        rows.extend(batch)
        if len(batch)<limit: break
        offset += limit
    d=pd.DataFrame(rows)
    if d.empty:
        raise RuntimeError('CDC API returned no rows')
    date_col = next((c for c in ['submission_date','date_updated','week_ending_date'] if c in d.columns), None)
    if date_col is None:
        raise RuntimeError(f'No CDC date column found. Columns={d.columns.tolist()}')
    state_col = next((c for c in ['state','jurisdiction'] if c in d.columns), None)
    if state_col is None:
        raise RuntimeError(f'No CDC state column found. Columns={d.columns.tolist()}')
    d['date']=pd.to_datetime(d[date_col],errors='coerce').dt.tz_localize(None)
    d['state_abbr']=d[state_col].astype(str).str.strip()
    aliases={
        'tot_cases':['tot_cases','total_cases'], 'new_case':['new_case','new_cases'],
        'tot_death':['tot_death','total_deaths'], 'new_death':['new_death','new_deaths']
    }
    for out,opts in aliases.items():
        src=next((c for c in opts if c in d.columns),None)
        d[out]=pd.to_numeric(d[src],errors='coerce') if src else np.nan
    d=d.dropna(subset=['date','state_abbr']).copy()
    d.to_csv(OUT/'cdc_raw_selected.csv',index=False)
    return d

def band6(x):
    if x <= 1e5: return '0-100K'
    if x <= 1e6: return '100K-1M'
    if x <= 50e6: return '1M-50M'
    if x <= 200e6: return '50M-200M'
    if x <= 500e6: return '200M-500M'
    return '500M+'

def features_at_date(cdc_state, dt):
    g=cdc_state[cdc_state.date<=dt].sort_values('date').copy()
    if g.empty: return None
    last=g.iloc[-1]
    def cumulative(col):
        v=pd.to_numeric(last.get(col),errors='coerce')
        return float(v) if pd.notna(v) else np.nan
    def window_sum(col,days):
        gg=g[g.date>dt-pd.Timedelta(days=days)]
        vals=pd.to_numeric(gg[col],errors='coerce') if col in gg else pd.Series(dtype=float)
        return float(vals.clip(lower=0).sum()) if len(vals) else np.nan
    return {
        'cdcObservationDate':last.date,
        'daysLagCDC':int((dt-last.date).days),
        'cumCases':cumulative('tot_cases'), 'cumDeaths':cumulative('tot_death'),
        'newCases7d':window_sum('new_case',7), 'newCases14d':window_sum('new_case',14), 'newCases28d':window_sum('new_case',28),
        'newDeaths7d':window_sum('new_death',7), 'newDeaths14d':window_sum('new_death',14), 'newDeaths28d':window_sum('new_death',28),
    }

def add_event_ranks(df, cols):
    out=df.copy()
    for c in cols:
        x=pd.to_numeric(out[c],errors='coerce')
        out[c+'_pct_rank']=x.rank(pct=True,method='average')
        total=x.sum(skipna=True)
        out[c+'_event_share']=x/total if total>0 else np.nan
        out[c+'_relative_event_avg']=out[c+'_event_share']*len(out) if total>0 else np.nan
    return out

def main():
    master=pd.read_excel(MASTER)
    master['target']=pd.to_numeric(master['totalObligatedFunding'],errors='coerce').fillna(0).clip(lower=0)
    master['band6']=master.target.map(band6)
    master['declarationDateParsed']=pd.to_datetime(master['declarationDate'],errors='coerce').dt.tz_localize(None)
    bio=master[master['incidentType'].astype(str).str.lower().eq('biological')].copy()
    if bio.empty: raise RuntimeError('No Biological declarations found in master')
    cdc=fetch_cdc()
    rows=[]
    for _,r in bio.iterrows():
        st=str(r['state']); dt=r['declarationDateParsed']; pop=float(POP.get(st,np.nan))
        feat=features_at_date(cdc[cdc.state_abbr==st],dt) if pd.notna(dt) else None
        row={
            'disasterNumber':int(r['disasterNumber']), 'state':st, 'fyDeclared':int(r['fyDeclared']),
            'declarationDate':dt, 'target':float(r['target']), 'band6':r['band6'],
            'population2010':pop,
        }
        if feat: row.update(feat)
        rows.append(row)
    out=pd.DataFrame(rows)
    burden=['cumCases','cumDeaths','newCases7d','newCases14d','newCases28d','newDeaths7d','newDeaths14d','newDeaths28d']
    for c in burden:
        out[c+'Per100k']=pd.to_numeric(out[c],errors='coerce')/pd.to_numeric(out.population2010,errors='coerce')*100000
    rank_cols=burden+[c+'Per100k' for c in burden]
    ranked=add_event_ranks(out,rank_cols)
    ranked.to_csv(OUT/'biological_cdc_severity_features.csv',index=False)

    high=ranked[ranked.target>50e6].copy()
    high.to_csv(OUT/'biological_high_value_cases.csv',index=False)
    assoc=[]
    y=np.log1p(high.target.to_numpy(float))
    for c in rank_cols+[c+'_pct_rank' for c in rank_cols]+[c+'_relative_event_avg' for c in rank_cols]:
        x=pd.to_numeric(high[c],errors='coerce')
        m=x.notna() & np.isfinite(y)
        if m.sum()>=4 and x[m].nunique()>1:
            rho,p=spearmanr(x[m],y[m])
            assoc.append({'feature':c,'n':int(m.sum()),'spearman_rho_log_funding':float(rho),'p_value':float(p)})
    assoc_df=pd.DataFrame(assoc)
    if len(assoc_df):
        assoc_df=assoc_df.sort_values('spearman_rho_log_funding',key=lambda s:s.abs(),ascending=False)
    assoc_df.to_csv(OUT/'severity_association_screen.csv',index=False)

    focus=ranked[ranked.disasterNumber.isin([4486,4515,4485,4480])].copy()
    focus.to_csv(OUT/'focus_2020_cases.csv',index=False)
    focus_cols=['disasterNumber','state','declarationDate','target','band6','cumCasesPer100k','cumDeathsPer100k','newCases14dPer100k','newDeaths14dPer100k','cumCasesPer100k_pct_rank','cumDeathsPer100k_pct_rank']
    summary={
        'cdc_dataset':'Weekly United States COVID-19 Cases and Deaths by State - ARCHIVED (pwn4-m3yp)',
        'prospective_rule':'Only CDC observations on or before each FEMA declaration date are used.',
        'biological_count':int(len(ranked)),
        'biological_high_over_50m_count':int(len(high)),
        'coverage_count':int(ranked.cumCases.notna().sum()),
        'best_absolute_spearman':assoc_df.head(15).to_dict('records') if len(assoc_df) else [],
        'focus_cases':focus[focus_cols].to_dict('records') if len(focus) else [],
        'interpretation_guardrail':'Signal audit only, not predictive validation. COVID-specific features cannot support a leave-2020-out predictive claim until translated into a cross-hazard severity representation learned from non-held years.'
    }
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str))

if __name__=='__main__':
    main()
