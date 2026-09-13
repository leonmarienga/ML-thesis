#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'audit_outputs'/'biological_baseline_audit'
OUT.mkdir(parents=True,exist_ok=True)
MASTER=ROOT/'master_openfema_40plus.xlsx'

def band(v):
    if v<100_000:return '0-100K'
    if v<1_000_000:return '100K-1M'
    if v<50_000_000:return '1M-50M'
    if v<200_000_000:return '50-200M'
    if v<500_000_000:return '200-500M'
    return '500M+'

def main():
    df=pd.read_excel(MASTER)
    b=df[df['incidentType'].astype(str).str.strip()=='Biological'].copy()
    b['target_clean']=pd.to_numeric(b['totalObligatedFunding'],errors='coerce').fillna(0).clip(lower=0)
    b['actual_band']=b['target_clean'].map(band)
    date_cols=[c for c in ['incidentBeginDate','incidentEndDate','declarationDate'] if c in b.columns]
    for c in date_cols:b[c]=pd.to_datetime(b[c],errors='coerce')
    keep=[c for c in ['disasterNumber','state','incidentType','fyDeclared','incidentBeginDate','incidentEndDate','declarationDate','durationDays','declarationDelayDays','missionAssignmentCount','uniqueAgencyCount','uniqueMaTypeCount','uniquePriorityCount','responseComplexityScore','missionDensity','agencyDensity','target_clean','actual_band'] if c in b.columns]
    b[keep].sort_values(['fyDeclared','disasterNumber']).to_csv(OUT/'biological_rows.csv',index=False)
    band_counts=b['actual_band'].value_counts().reindex(['0-100K','100K-1M','1M-50M','50-200M','200-500M','500M+'],fill_value=0).to_dict()
    fy_counts=b['fyDeclared'].value_counts().sort_index().to_dict()
    states=sorted(b['state'].astype(str).unique().tolist())
    summary={
        'rows':int(len(b)),
        'unique_disaster_numbers':int(b['disasterNumber'].nunique()),
        'unique_states':int(b['state'].nunique()),
        'states':states,
        'fiscal_year_counts':{str(int(k)):int(v) for k,v in fy_counts.items()},
        'band_counts':{k:int(v) for k,v in band_counts.items()},
        'target_min':float(b['target_clean'].min()) if len(b) else None,
        'target_median':float(b['target_clean'].median()) if len(b) else None,
        'target_max':float(b['target_clean'].max()) if len(b) else None,
        'unique_begin_dates':int(b['incidentBeginDate'].nunique()) if 'incidentBeginDate' in b else None,
        'unique_end_dates':int(b['incidentEndDate'].nunique()) if 'incidentEndDate' in b else None,
        'unique_declaration_dates':int(b['declarationDate'].nunique()) if 'declarationDate' in b else None,
    }
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Biological baseline audit','',f"- Rows: **{summary['rows']}**",f"- Unique disaster numbers: **{summary['unique_disaster_numbers']}**",f"- Unique states/territories: **{summary['unique_states']}**",f"- Fiscal years: **{summary['fiscal_year_counts']}**",f"- Funding bands: **{summary['band_counts']}**",f"- Target median: **${summary['target_median']:,.2f}**",f"- Target max: **${summary['target_max']:,.2f}**",f"- Unique incident begin dates: **{summary['unique_begin_dates']}**",f"- Unique incident end dates: **{summary['unique_end_dates']}**",f"- Unique declaration dates: **{summary['unique_declaration_dates']}**"]
    (OUT/'summary.md').write_text('\n'.join(lines));print('\n'.join(lines))
if __name__=='__main__':main()
