#!/usr/bin/env python3
"""
Strict post-820 Hurricane mission-duration floor audit.

For each outer LFYO fold, choose the largest multiplier from the fixed grid
[0.5, 0.6, 0.7, 0.8, 0.9, 1.0] whose procedure preserves 100% of true-high
Hurricanes under inner LFYO. The outer floor is then:

    multiplier * min(sem_duration_mean among outer-training true-high Hurricanes)

Only accepted root-high Hurricanes below that floor are vetoed, and vetoed rows
return to the established outer-training-only low router. No upward promotions.
Biological remains excluded and no funding-derived predictor is used.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

from mission_semantic_audit import CURRENT_19, build_semantic_rollup, fetch_all_mission_assignments, normalize_master
from nonbio_all_ranges import funding_band, six_band_metrics, valid_cols
from nonbio_hurricane_boundary_confirm import build_low_predictor, data_hash

ROOT=Path(__file__).resolve().parents[1]
MASTER=ROOT/'master_openfema_40plus.xlsx'
ACCEPTED=ROOT/'audit_inputs'/'post820_hurr_duration'/'accepted'/'candidate_predictions.csv'
OUT=ROOT/'audit_outputs'/'nonbio_820_hurricane_duration_floor'
OUT.mkdir(parents=True,exist_ok=True)
HIGH=50_000_000.0
GRID=[0.5,0.6,0.7,0.8,0.9,1.0]
BASE_ROOT='new_root_pred_high'
BASE_PRED='new_pred'


def duration(df):
    return pd.to_numeric(df['sem_duration_mean'],errors='coerce')


def safe_multiplier(train):
    safe=[]
    h=train[train['incidentType']=='Hurricane'].copy()
    for mult in GRID:
        checks=[]
        for fy in sorted(h['fyDeclared'].astype(int).unique()):
            itr=h[h['fyDeclared'].astype(int)!=fy]
            ite=h[h['fyDeclared'].astype(int)==fy]
            tr_hi=itr[itr['target_clean']>=HIGH]
            te_hi=ite[ite['target_clean']>=HIGH]
            if tr_hi.empty or te_hi.empty:
                continue
            vals=duration(tr_hi).dropna()
            if vals.empty:
                continue
            th=float(mult*vals.min())
            kept=duration(te_hi).fillna(np.inf)>=th
            checks.extend(kept.tolist())
        if checks and all(checks):
            safe.append((mult,len(checks)))
    return max(safe,key=lambda x:x[0]) if safe else None


def main():
    master=normalize_master(pd.read_excel(MASTER))
    master['target_clean']=pd.to_numeric(master['totalObligatedFunding'],errors='coerce').fillna(0).clip(lower=0)
    master['actual_band']=master['target_clean'].map(funding_band)
    master['disasterNumber']=master['disasterNumber'].astype(int)
    ma=fetch_all_mission_assignments(); ma_hash=data_hash(ma)
    sem,_=build_semantic_rollup(master,ma)
    df=master.merge(sem,on='disasterNumber',how='left')
    df=df[df['incidentType']!='Biological'].copy().reset_index(drop=True)

    current=valid_cols(df,CURRENT_19)
    semcols=valid_cols(df,[c for c in df.columns if (c.startswith('sem_') or c.startswith('ma_')) and not any(b in c.lower() for b in ['oblig','fund','cost','amount','dollar'])])
    semfeat=list(dict.fromkeys(current+semcols))

    acc=pd.read_csv(ACCEPTED); acc['disasterNumber']=acc['disasterNumber'].astype(int)
    if len(acc)!=912: raise AssertionError(f'Expected 912 rows, got {len(acc)}')
    baseline_correct=int((acc[BASE_PRED]==acc['actual_band']).sum())
    if baseline_correct!=820: raise AssertionError(f'Expected 820 baseline, got {baseline_correct}')

    model=df.merge(acc[['disasterNumber',BASE_ROOT,BASE_PRED]],on='disasterNumber',how='inner',validate='one_to_one')
    if len(model)!=912: raise AssertionError(f'Expected 912 merged rows, got {len(model)}')
    model['root_actual_high']=(model['target_clean']>=HIGH).astype(int)
    roots={int(r.disasterNumber):int(getattr(r,BASE_ROOT)) for r in model.itertuples(index=False)}
    preds={int(r.disasterNumber):str(getattr(r,BASE_PRED)) for r in model.itertuples(index=False)}
    changed=[]; diags=[]

    for fy in sorted(model['fyDeclared'].astype(int).unique()):
        train=model[model['fyDeclared'].astype(int)!=fy].copy()
        test=model[model['fyDeclared'].astype(int)==fy].copy()
        sm=safe_multiplier(train)
        enabled=sm is not None
        mult=sm[0] if sm else None; inner_n=sm[1] if sm else 0
        hi=train[(train['incidentType']=='Hurricane')&(train['target_clean']>=HIGH)]
        vals=duration(hi).dropna()
        threshold=float(mult*vals.min()) if enabled and not vals.empty else None
        eligible=test[(test['incidentType']=='Hurricane')&(test[BASE_ROOT].astype(int)==1)].copy()
        if enabled:
            veto=eligible[duration(eligible).fillna(np.inf)<threshold].copy()
        else:
            veto=eligible.iloc[0:0].copy()
        if not veto.empty:
            low_predict=build_low_predictor(train,semfeat,int(fy))
            low_pred=low_predict(veto)
            for (_,r),lp in zip(veto.iterrows(),low_pred):
                dn=int(r['disasterNumber']); old=preds[dn]
                roots[dn]=0; preds[dn]=str(lp)
                changed.append({'disasterNumber':dn,'state':r['state'],'fyDeclared':int(r['fyDeclared']),'actual_band':r['actual_band'],'old_pred':old,'new_pred':str(lp),'sem_duration_mean':float(r['sem_duration_mean']),'multiplier':float(mult),'threshold':float(threshold)})
        diags.append({'outer_fy':int(fy),'enabled':enabled,'multiplier':mult,'inner_high_n':inner_n,'threshold':threshold,'training_high_n':int(len(hi)),'eligible_root_high_hurricane':int(len(eligible)),'veto_count':int(len(veto))})

    out=model[['disasterNumber','state','incidentType','fyDeclared','actual_band','target_clean','root_actual_high',BASE_ROOT,BASE_PRED,'sem_duration_mean']].copy()
    out['baseline_root_pred_high']=out[BASE_ROOT].astype(int); out['baseline_pred']=out[BASE_PRED].astype(str)
    out['new_root_pred_high']=out['disasterNumber'].map(roots).astype(int); out['new_pred']=out['disasterNumber'].map(preds).astype(str)
    out.to_csv(OUT/'candidate_predictions.csv',index=False)
    pd.DataFrame(changed).to_csv(OUT/'changed_rows.csv',index=False)
    pd.DataFrame(diags).to_csv(OUT/'fold_diagnostics.csv',index=False)

    y=out['root_actual_high'].to_numpy(int); br=out['baseline_root_pred_high'].to_numpy(int); nr=out['new_root_pred_high'].to_numpy(int)
    base_fn=set(out.loc[(y==1)&(br==0),'disasterNumber'].astype(int)); new_fn=set(out.loc[(y==1)&(nr==0),'disasterNumber'].astype(int))
    base_fp=int(((y==0)&(br==1)).sum()); new_fp=int(((y==0)&(nr==1)).sum())
    base_rec=float(recall_score(y,br,zero_division=0)); new_rec=float(recall_score(y,nr,zero_division=0))
    correct=int((out['new_pred']==out['actual_band']).sum())
    mm=out.rename(columns={'new_pred':'metric_pred'}); metrics=six_band_metrics(mm,'metric_pred')
    if base_fn: raise AssertionError(f'Accepted 820 should have no high FN, got {sorted(base_fn)}')

    summary={'mission_nonfinancial_sha256':ma_hash,'grid':GRID,'baseline':{'overall_correct':baseline_correct,'root_fp':base_fp,'root_high_recall':base_rec,'high_false_negatives':sorted(base_fn)},'candidate':{'overall_correct':correct,'root_fp':new_fp,'root_high_recall':new_rec,'high_false_negatives':sorted(new_fn),'end_to_end':metrics},'changed_rows':changed}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    md=['# Post-820 Hurricane mission-duration floor confirmation','',f'- Baseline: **{baseline_correct}/912**; root FP **{base_fp}**; high recall **{base_rec:.1%}**; FN **{sorted(base_fn)}**.',f'- Candidate: **{correct}/912**; root FP **{new_fp}**; high recall **{new_rec:.1%}**; FN **{sorted(new_fn)}**.','','## Changed rows','','| FEMA | State | Actual | Old | New | Duration mean | Multiplier | Floor |','|---:|---|---|---|---|---:|---:|---:|']
    for r in changed: md.append(f"| {r['disasterNumber']} | {r['state']} | {r['actual_band']} | {r['old_pred']} | {r['new_pred']} | {r['sem_duration_mean']:.2f} | {r['multiplier']:.1f} | {r['threshold']:.2f} |")
    md += ['',f"- Macro recall: **{metrics['macro_recall']:.3f}**"]
    for b,m in metrics['per_band'].items(): md.append(f"- {b}: **{m['correct']}/{m['total']} = {m['recall']:.1%}**")
    (OUT/'summary.md').write_text('\n'.join(md),encoding='utf-8'); print('\n'.join(md))

if __name__=='__main__': main()
