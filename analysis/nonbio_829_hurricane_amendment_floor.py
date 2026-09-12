#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from mission_semantic_audit import build_semantic_rollup, fetch_all_mission_assignments, normalize_master
from nonbio_all_ranges import funding_band, six_band_metrics

ROOT=Path(__file__).resolve().parents[1]
MASTER=ROOT/'master_openfema_40plus.xlsx'
ACCEPTED=ROOT/'audit_inputs'/'post829_hurricane_amendment'/'accepted'/'candidate_predictions.csv'
OUT=ROOT/'audit_outputs'/'nonbio_829_hurricane_amendment_floor'
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
    acc=pd.read_csv(ACCEPTED); acc['disasterNumber']=acc['disasterNumber'].astype(int)
    base=int((acc['candidate_pred']==acc['actual_band']).sum())
    if base!=829: raise AssertionError(f'Expected 829, got {base}')
    df=df.merge(acc[['disasterNumber','candidate_pred']],on='disasterNumber',how='inner',validate='one_to_one')
    df['ma_mean_amendment']=pd.to_numeric(df['ma_mean_amendment'],errors='coerce')
    pred=dict(zip(df.disasterNumber.astype(int),df.candidate_pred.astype(str)))
    hur=df[(df.incidentType=='Hurricane')&(df.target_clean<50_000_000)].copy()
    changed=[]; diag=[]
    for ofy in sorted(hur.fyDeclared.astype(int).unique()):
        train=hur[hur.fyDeclared.astype(int)!=ofy].copy()
        test=hur[(hur.fyDeclared.astype(int)==ofy)&(hur.candidate_pred=='1M-50M')].copy()
        train_hi=train[(train.target_clean>=1_000_000)&(train.target_clean<50_000_000)&train.ma_mean_amendment.notna()]
        valid=[]
        for mult in GRID:
            ok=True; held=0
            for ify in sorted(train.fyDeclared.astype(int).unique()):
                itr=train[train.fyDeclared.astype(int)!=ify]
                ite=train[(train.fyDeclared.astype(int)==ify)&(train.target_clean>=1_000_000)&(train.target_clean<50_000_000)&train.ma_mean_amendment.notna()]
                ih=itr[(itr.target_clean>=1_000_000)&(itr.target_clean<50_000_000)&itr.ma_mean_amendment.notna()]
                if ite.empty or ih.empty: continue
                held+=len(ite); floor=mult*float(ih.ma_mean_amendment.min())
                if (ite.ma_mean_amendment<floor).any(): ok=False; break
            if ok and held>0: valid.append(mult)
        if train_hi.empty or not valid:
            diag.append({'outer_fy':int(ofy),'enabled':False,'changes':0}); continue
        mult=max(valid); floor=mult*float(train_hi.ma_mean_amendment.min())
        hit=test[test.ma_mean_amendment<floor]
        ids=[]
        for _,r in hit.iterrows():
            dn=int(r.disasterNumber); old=pred[dn]; pred[dn]='0-100K'; ids.append(dn)
            changed.append({'disasterNumber':dn,'state':r.state,'fyDeclared':int(r.fyDeclared),'target_clean':float(r.target_clean),'actual_band':r.actual_band,'old_pred':old,'new_pred':'0-100K','ma_mean_amendment':float(r.ma_mean_amendment),'multiplier':mult,'floor':floor})
        diag.append({'outer_fy':int(ofy),'enabled':True,'multiplier':mult,'floor':floor,'eligible_test_n':len(test),'changes':len(ids),'changed_ids':';'.join(map(str,ids))})

    out=df[['disasterNumber','state','incidentType','fyDeclared','target_clean','actual_band','candidate_pred']].copy()
    out['candidate_pred_new']=out.disasterNumber.map(pred)
    out.to_csv(OUT/'candidate_predictions.csv',index=False)
    pd.DataFrame(changed).to_csv(OUT/'changed_rows.csv',index=False)
    pd.DataFrame(diag).to_csv(OUT/'fold_diagnostics.csv',index=False)
    cand=int((out.candidate_pred_new==out.actual_band).sum())
    metric_input=out.rename(columns={'candidate_pred_new':'new_pred'})
    metrics=six_band_metrics(metric_input,'new_pred')
    protected=out[out.target_clean>=50_000_000]
    pchg=int((protected.candidate_pred!=protected.candidate_pred_new).sum())
    correct=sum(r['actual_band']=='0-100K' for r in changed); wrong=len(changed)-correct
    summary={'baseline_correct':base,'candidate_correct':cand,'total':len(out),'changed_count':len(changed),'correct_changes':correct,'wrong_changes':wrong,'protected_ge_50m_changes':pchg,'metrics':metrics}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Post-829 Hurricane amendment floor','',f'- Baseline: **{base}/912**',f'- Candidate: **{cand}/912**',f'- Changed rows: **{len(changed)}** ({correct} correct, {wrong} wrong)',f'- >=$50M prediction changes: **{pchg}**','','## Changed rows']
    if not changed: lines.append('- None')
    for r in changed:
        lines.append(f"- FEMA {r['disasterNumber']} {r['state']} FY{r['fyDeclared']}: ${r['target_clean']:,.2f} | actual {r['actual_band']} | {r['old_pred']} -> {r['new_pred']} | mean amendment={r['ma_mean_amendment']:.3f}, floor={r['floor']:.3f}, multiplier={r['multiplier']:.1f}")
    lines+=['','## Six-band recall']
    for b,v in metrics['per_band'].items(): lines.append(f"- {b}: **{v['correct']}/{v['total']} = {v['recall']:.1%}**")
    lines.append(f"- Macro recall: **{metrics['macro_recall']:.3f}**")
    (OUT/'summary.md').write_text('\n'.join(lines)); print('\n'.join(lines))
    if pchg: raise AssertionError('Changed protected >=50M rows')

if __name__=='__main__': main()
