#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from mission_semantic_audit import build_semantic_rollup, fetch_all_mission_assignments, normalize_master
from nonbio_all_ranges import funding_band, six_band_metrics

ROOT=Path(__file__).resolve().parents[1]
MASTER=ROOT/'master_openfema_40plus.xlsx'
ACCEPTED=ROOT/'audit_inputs'/'post826_fire_duration'/'accepted'/'candidate_predictions.csv'
OUT=ROOT/'audit_outputs'/'nonbio_826_fire_duration_floor'
OUT.mkdir(parents=True,exist_ok=True)
GRID=[0.5,0.6,0.7,0.8,0.9,1.0]

def main():
    master=normalize_master(pd.read_excel(MASTER))
    master['disasterNumber']=master['disasterNumber'].astype(int)
    master['target_clean']=pd.to_numeric(master['totalObligatedFunding'],errors='coerce').fillna(0).clip(lower=0)
    master['actual_band']=master['target_clean'].map(funding_band)
    ma=fetch_all_mission_assignments(); sem,_=build_semantic_rollup(master,ma)
    df=master.merge(sem,on='disasterNumber',how='left')
    df=df[df['incidentType']!='Biological'].copy()
    accepted=pd.read_csv(ACCEPTED); accepted['disasterNumber']=accepted['disasterNumber'].astype(int)
    if len(accepted)!=912: raise AssertionError(len(accepted))
    base=int((accepted['candidate_pred']==accepted['actual_band']).sum())
    if base!=826: raise AssertionError(f'Expected 826, got {base}')
    df=df.merge(accepted[['disasterNumber','candidate_pred']],on='disasterNumber',how='inner',validate='one_to_one')
    df['sem_duration_mean']=pd.to_numeric(df['sem_duration_mean'],errors='coerce')
    pred=dict(zip(df.disasterNumber.astype(int),df.candidate_pred.astype(str)))
    fire=df[(df.incidentType=='Fire')&(df.target_clean<50_000_000)].copy()
    changed=[]; diagnostics=[]
    for ofy in sorted(fire.fyDeclared.astype(int).unique()):
        train=fire[fire.fyDeclared.astype(int)!=ofy].copy()
        test=fire[(fire.fyDeclared.astype(int)==ofy)&(fire.candidate_pred=='1M-50M')].copy()
        train_hi=train[(train.target_clean>=1_000_000)&(train.target_clean<50_000_000)&train.sem_duration_mean.notna()]
        valid=[]
        for m in GRID:
            ok=True; held=0
            for ify in sorted(train.fyDeclared.astype(int).unique()):
                itr=train[train.fyDeclared.astype(int)!=ify]
                ite=train[(train.fyDeclared.astype(int)==ify)&(train.target_clean>=1_000_000)&(train.target_clean<50_000_000)&train.sem_duration_mean.notna()]
                ih=itr[(itr.target_clean>=1_000_000)&(itr.target_clean<50_000_000)&itr.sem_duration_mean.notna()]
                if ite.empty or ih.empty: continue
                held+=len(ite); floor=m*float(ih.sem_duration_mean.min())
                if (ite.sem_duration_mean<floor).any(): ok=False; break
            if ok and held>0: valid.append(m)
        if train_hi.empty or not valid:
            diagnostics.append({'outer_fy':ofy,'enabled':False,'promotions':0}); continue
        m=max(valid); floor=m*float(train_hi.sem_duration_mean.min())
        hit=test[test.sem_duration_mean<floor]
        for _,r in hit.iterrows():
            dn=int(r.disasterNumber); old=pred[dn]; pred[dn]='100K-1M'
            changed.append({'disasterNumber':dn,'state':r.state,'fyDeclared':int(r.fyDeclared),'target_clean':float(r.target_clean),'actual_band':r.actual_band,'old_pred':old,'new_pred':'100K-1M','sem_duration_mean':float(r.sem_duration_mean),'multiplier':m,'floor':floor})
        diagnostics.append({'outer_fy':ofy,'enabled':True,'multiplier':m,'floor':floor,'eligible_test_n':len(test),'promotions':len(hit)})
    out=df[['disasterNumber','state','incidentType','fyDeclared','target_clean','actual_band','candidate_pred']].copy()
    out['candidate_pred_new']=out.disasterNumber.map(pred)
    out.to_csv(OUT/'candidate_predictions.csv',index=False)
    pd.DataFrame(changed).to_csv(OUT/'changed_rows.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'fold_diagnostics.csv',index=False)
    cand=int((out.candidate_pred_new==out.actual_band).sum())
    metric_input=out.rename(columns={'candidate_pred_new':'new_pred'})
    metrics=six_band_metrics(metric_input,'new_pred')
    protected=out[out.target_clean>=50_000_000]
    pchg=int((protected.candidate_pred!=protected.candidate_pred_new).sum())
    correct=sum(r['actual_band']=='100K-1M' for r in changed); wrong=len(changed)-correct
    summary={'baseline_correct':base,'candidate_correct':cand,'total':len(out),'changed_count':len(changed),'correct_changes':correct,'wrong_changes':wrong,'protected_ge_50m_changes':pchg,'metrics':metrics}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Post-826 Fire duration floor','',f'- Baseline: **{base}/912**',f'- Candidate: **{cand}/912**',f'- Changed rows: **{len(changed)}** ({correct} correct, {wrong} wrong)',f'- >=$50M prediction changes: **{pchg}**','','## Changed rows']
    for r in changed:
        lines.append(f"- FEMA {r['disasterNumber']} {r['state']} FY{r['fyDeclared']}: ${r['target_clean']:,.2f} | actual {r['actual_band']} | {r['old_pred']} -> {r['new_pred']} | duration={r['sem_duration_mean']:.2f}, floor={r['floor']:.2f}, multiplier={r['multiplier']:.1f}")
    lines+=['','## Six-band recall']
    for b,v in metrics['per_band'].items(): lines.append(f"- {b}: **{v['correct']}/{v['total']} = {v['recall']:.1%}**")
    lines.append(f"- Macro recall: **{metrics['macro_recall']:.3f}**")
    (OUT/'summary.md').write_text('\n'.join(lines)); print('\n'.join(lines))
    if pchg: raise AssertionError('Changed protected >=50M rows')

if __name__=='__main__': main()
