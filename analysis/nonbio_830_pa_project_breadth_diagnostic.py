#!/usr/bin/env python3
from __future__ import annotations
import json, time, re
from collections import Counter
from pathlib import Path
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[1]
ACCEPTED=ROOT/'audit_inputs'/'post830_pa_breadth'/'accepted'/'candidate_predictions.csv'
OUT=ROOT/'audit_outputs'/'nonbio_830_pa_project_breadth_diagnostic'
OUT.mkdir(parents=True,exist_ok=True)
API='https://www.fema.gov/api/open/v1/PublicAssistanceFundedProjectsDetails'
HIGH={'50-200M','200-500M','500M+'}


def norm(s): return re.sub(r'[^a-z0-9]','',str(s).lower())
def pick_key(keys,cands):
    m={norm(k):k for k in keys}
    for c in cands:
        if norm(c) in m:return m[norm(c)]
    return None

def fetch_all(dn):
    out=[]; skip=0; top=1000
    while True:
        p={'$filter':f'disasterNumber eq {int(dn)}','$top':top,'$skip':skip}
        r=requests.get(API,params=p,timeout=90); r.raise_for_status(); j=r.json()
        rows=j.get('PublicAssistanceFundedProjectsDetails',[])
        out.extend(rows)
        if len(rows)<top:break
        skip+=top
        if skip>50000:raise RuntimeError(f'pagination runaway {dn}')
    return out

def aggregate(dn,rows):
    if not rows:return {'disasterNumber':dn,'pa_project_count':0,'pa_unique_applicants':0,'pa_unique_categories':0,'pa_unique_counties':0,'pa_emergency_ab_count':0,'pa_permanent_cg_count':0}
    keys=list(rows[0].keys())
    app=pick_key(keys,['applicantId','applicantID'])
    cat=pick_key(keys,['damageCategoryCode','damageCategory','categoryCode','projectCategory'])
    county=pick_key(keys,['county','countyName'])
    pw=pick_key(keys,['pwNumber','projectWorksheetNumber'])
    def vals(k):return [str(x.get(k)).strip() for x in rows if k and x.get(k) not in (None,'')]
    cats=vals(cat); apps=vals(app); counties=vals(county); pws=vals(pw)
    cc=Counter(cats)
    d={'disasterNumber':dn,'pa_project_count':len(rows),'pa_unique_pws':len(set(pws)) if pw else len(rows),'pa_unique_applicants':len(set(apps)),'pa_unique_categories':len(set(cats)),'pa_unique_counties':len(set(counties)),'pa_emergency_ab_count':sum(v for k,v in cc.items() if str(k).upper().strip() in {'A','B'}),'pa_permanent_cg_count':sum(v for k,v in cc.items() if str(k).upper().strip() in {'C','D','E','F','G'}),'detected_applicant_key':app or '','detected_category_key':cat or '','detected_county_key':county or '','detected_pw_key':pw or ''}
    for k,v in sorted(cc.items()):
        safe=re.sub(r'[^A-Za-z0-9]+','_',str(k)).strip('_')[:40]
        if safe:d[f'pa_category_{safe}_count']=v
    return d

def main():
    acc=pd.read_csv(ACCEPTED);acc['disasterNumber']=acc.disasterNumber.astype(int)
    pred='candidate_pred_new' if 'candidate_pred_new' in acc.columns else 'candidate_pred'
    base=int((acc[pred]==acc.actual_band).sum())
    if base!=830:raise AssertionError(f'Expected 830 got {base}')
    focus=acc[(acc.actual_band.isin(HIGH)) | (acc[pred].isin(HIGH))].copy()
    rows=[]; samples=[]
    for i,dn in enumerate(sorted(focus.disasterNumber.unique()),1):
        print(f'PA {i}/{focus.disasterNumber.nunique()} FEMA {dn}',flush=True)
        rr=fetch_all(dn); rows.append(aggregate(int(dn),rr))
        if rr and len(samples)<3:samples.append({'disasterNumber':int(dn),'keys':sorted(rr[0].keys())})
        time.sleep(.05)
    pa=pd.DataFrame(rows)
    out=focus.merge(pa,on='disasterNumber',how='left')
    out.to_csv(OUT/'highband_pa_breadth.csv',index=False)
    (OUT/'sample_fields.json').write_text(json.dumps(samples,indent=2))
    sandy=out[out.disasterNumber.isin([4085,4086])].copy(); sandy.to_csv(OUT/'sandy_pa_comparison.csv',index=False)
    boundary=out[out.actual_band.isin(['50-200M','200-500M'])].copy()
    metrics=[]
    for c in [x for x in out.columns if x.startswith('pa_') and x not in {'pa_category_'} and pd.api.types.is_numeric_dtype(out[x])]:
        a=pd.to_numeric(boundary.loc[boundary.actual_band=='50-200M',c],errors='coerce').fillna(0); b=pd.to_numeric(boundary.loc[boundary.actual_band=='200-500M',c],errors='coerce').fillna(0); e=pd.to_numeric(boundary.loc[boundary.disasterNumber==4086,c],errors='coerce')
        metrics.append({'feature':c,'fema4086':float(e.iloc[0]) if len(e) else None,'band50_200_min':float(a.min()) if len(a) else None,'band50_200_median':float(a.median()) if len(a) else None,'band50_200_max':float(a.max()) if len(a) else None,'band200_500_min':float(b.min()) if len(b) else None,'band200_500_median':float(b.median()) if len(b) else None,'band200_500_max':float(b.max()) if len(b) else None})
    pd.DataFrame(metrics).to_csv(OUT/'pa_boundary_summary.csv',index=False)
    summary={'baseline_correct':base,'focus_rows':len(focus),'rows_with_pa_projects':int((out.pa_project_count.fillna(0)>0).sum()),'fema4086_project_count':int(out.loc[out.disasterNumber==4086,'pa_project_count'].fillna(0).iloc[0]) if (out.disasterNumber==4086).any() else 0,'prediction_changes':0}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Post-830 PA project breadth diagnostic','',f'- Accepted baseline: **{base}/912**',f'- High/root focus rows: **{len(focus)}**',f"- Rows with PA projects: **{summary['rows_with_pa_projects']}**",'- Prediction changes: **0** (diagnostic only)','','## Sandy NY/NJ']
    for _,r in sandy.sort_values('disasterNumber').iterrows():lines.append(f"- FEMA {int(r.disasterNumber)} {r.state} | actual {r.actual_band} | pred {r[pred]} | projects={int(r.get('pa_project_count',0))} | applicants={int(r.get('pa_unique_applicants',0))} | categories={int(r.get('pa_unique_categories',0))} | counties={int(r.get('pa_unique_counties',0))} | A/B={int(r.get('pa_emergency_ab_count',0))} | C-G={int(r.get('pa_permanent_cg_count',0))}")
    (OUT/'summary.md').write_text('\n'.join(lines));print('\n'.join(lines))
if __name__=='__main__':main()
