#!/usr/bin/env python3
from __future__ import annotations
import json, time, re
from collections import Counter
from pathlib import Path
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[1]
ACCEPTED=ROOT/'audit_inputs'/'post830_pa_applicants'/'accepted'/'candidate_predictions.csv'
OUT=ROOT/'audit_outputs'/'nonbio_830_pa_applicant_structure_diagnostic'
OUT.mkdir(parents=True,exist_ok=True)
API='https://www.fema.gov/api/open/v1/PublicAssistanceApplicants'
HIGH={'50-200M','200-500M','500M+'}

def norm(s): return re.sub(r'[^a-z0-9]','',str(s).lower())
def pick(keys,cands):
    m={norm(k):k for k in keys}
    for c in cands:
        if norm(c) in m:return m[norm(c)]
    return None

def fetch_all(dn):
    out=[];skip=0;top=1000
    while True:
        p={'$filter':f'disasterNumber eq {int(dn)}','$top':top,'$skip':skip}
        r=requests.get(API,params=p,timeout=90);r.raise_for_status();j=r.json()
        rows=j.get('PublicAssistanceApplicants',[]);out.extend(rows)
        if len(rows)<top:break
        skip+=top
        if skip>50000:raise RuntimeError(f'pagination runaway {dn}')
    return out

def aggregate(dn,rows):
    if not rows:return {'disasterNumber':dn,'paapp_rows':0,'paapp_unique_applicants':0,'paapp_unique_counties':0,'paapp_unique_cities':0}
    keys=list(rows[0].keys())
    app=pick(keys,['applicantId','applicantID'])
    county=pick(keys,['county','countyName'])
    city=pick(keys,['city'])
    atype=pick(keys,['applicantType'])
    gtype=pick(keys,['granteeType'])
    ptype=pick(keys,['privateNonProfitType','pnpType','privateNonprofitType'])
    def vals(k):return [str(x.get(k)).strip() for x in rows if k and x.get(k) not in (None,'')]
    apps=vals(app);counties=vals(county);cities=vals(city)
    d={'disasterNumber':dn,'paapp_rows':len(rows),'paapp_unique_applicants':len(set(apps)) if app else len(rows),'paapp_unique_counties':len(set(counties)),'paapp_unique_cities':len(set(cities)),'detected_applicant_key':app or '','detected_county_key':county or '','detected_city_key':city or '','detected_applicant_type_key':atype or '','detected_grantee_type_key':gtype or '','detected_pnp_type_key':ptype or ''}
    for prefix,k in [('applicant_type',atype),('grantee_type',gtype),('pnp_type',ptype)]:
        for name,n in Counter(vals(k)).items():
            safe=re.sub(r'[^A-Za-z0-9]+','_',name).strip('_')[:50]
            if safe:d[f'paapp_{prefix}_{safe}_count']=n
    return d

def main():
    acc=pd.read_csv(ACCEPTED);acc['disasterNumber']=acc.disasterNumber.astype(int)
    pred='candidate_pred_new' if 'candidate_pred_new' in acc.columns else 'candidate_pred'
    base=int((acc[pred]==acc.actual_band).sum())
    if base!=830:raise AssertionError(f'Expected 830 got {base}')
    focus=acc[(acc.actual_band.isin(HIGH)) | (acc[pred].isin(HIGH))].copy()
    rows=[];samples=[]
    for i,dn in enumerate(sorted(focus.disasterNumber.unique()),1):
        print(f'PA applicants {i}/{focus.disasterNumber.nunique()} FEMA {dn}',flush=True)
        rr=fetch_all(dn);rows.append(aggregate(int(dn),rr))
        if rr and len(samples)<3:samples.append({'disasterNumber':int(dn),'keys':sorted(rr[0].keys()),'example':rr[0]})
        time.sleep(.05)
    pa=pd.DataFrame(rows)
    out=focus.merge(pa,on='disasterNumber',how='left')
    out.to_csv(OUT/'highband_pa_applicants.csv',index=False)
    (OUT/'sample_fields.json').write_text(json.dumps(samples,indent=2,default=str))
    sandy=out[out.disasterNumber.isin([4085,4086])].copy();sandy.to_csv(OUT/'sandy_pa_applicant_comparison.csv',index=False)
    summary={'baseline_correct':base,'focus_rows':len(focus),'rows_with_applicants':int((out.paapp_rows.fillna(0)>0).sum()),'fema4086_applicants':int(out.loc[out.disasterNumber==4086,'paapp_unique_applicants'].fillna(0).iloc[0]) if (out.disasterNumber==4086).any() else 0,'prediction_changes':0}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Post-830 PA applicant-structure diagnostic','',f'- Accepted baseline: **{base}/912**',f'- High/root focus rows: **{len(focus)}**',f"- Rows with PA applicant data: **{summary['rows_with_applicants']}**",'- Prediction changes: **0** (diagnostic only)','','## Sandy NY/NJ']
    for _,r in sandy.sort_values('disasterNumber').iterrows():
        lines.append(f"- FEMA {int(r.disasterNumber)} {r.state} | actual {r.actual_band} | pred {r[pred]} | applicants={int(r.get('paapp_unique_applicants',0))} | counties={int(r.get('paapp_unique_counties',0))} | cities={int(r.get('paapp_unique_cities',0))}")
    (OUT/'summary.md').write_text('\n'.join(lines));print('\n'.join(lines))
if __name__=='__main__':main()
