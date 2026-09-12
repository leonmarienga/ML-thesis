#!/usr/bin/env python3
"""
Strict post-822 Hurricane two-path mechanism audit.

Primary path: initial USACE + ESF-3 + DFA mechanism is present.
Secondary path: selected only from target-blind sem_topic_*_count features using
outer-training data. The component is disabled unless outer training itself
contains at least one true-high Hurricane lacking the primary mechanism and a
secondary topic covers every such exception.

The secondary topic is chosen to maximize rejection of outer-training low
Hurricanes while preserving every outer-training true-high through
(primary OR secondary). The selection procedure must also preserve 100% of
held-out true-high Hurricanes under inner LFYO; folds in which the selector has
no demonstrated alternate path simply disable the veto and are safe by design.

Only accepted root-high Hurricanes failing both paths are vetoed. Vetoed rows
return to the established outer-training-only low router. No promotions.
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
from nonbio_hazard_hierarchy import initial_mechanism_counts

ROOT=Path(__file__).resolve().parents[1]
MASTER=ROOT/'master_openfema_40plus.xlsx'
ACCEPTED=ROOT/'audit_inputs'/'post822_hurr_twopath'/'accepted'/'candidate_predictions.csv'
OUT=ROOT/'audit_outputs'/'nonbio_822_hurricane_two_path_mechanism'
OUT.mkdir(parents=True,exist_ok=True)
HIGH=50_000_000.0
BASE_ROOT='new_root_pred_high'; BASE_PRED='new_pred'


def primary(df):
    return pd.to_numeric(df['initial_usace_esf3_dfa_count'],errors='coerce').fillna(0)>0


def select_secondary(train, topic_cols):
    h=train[(train['incidentType']=='Hurricane')&(train['target_clean']>=HIGH)].copy()
    l=train[(train['incidentType']=='Hurricane')&(train['target_clean']<HIGH)].copy()
    uncovered=h[~primary(h)].copy()
    if uncovered.empty or l.empty:
        return None, {'uncovered_high_n':int(len(uncovered)),'low_n':int(len(l)),'rejected_low_n':0}
    best=None
    for c in topic_cols:
        sec_uncovered=pd.to_numeric(uncovered[c],errors='coerce').fillna(0)>0
        if not sec_uncovered.all():
            continue
        sec_h=pd.to_numeric(h[c],errors='coerce').fillna(0)>0
        cond_h=primary(h)|sec_h
        if not cond_h.all():
            continue
        sec_l=pd.to_numeric(l[c],errors='coerce').fillna(0)>0
        cond_l=primary(l)|sec_l
        rejected=int((~cond_l).sum())
        support=int(sec_h.sum())
        key=(rejected,support)
        if best is None or key>best[0] or (key==best[0] and c<best[1]):
            best=(key,c)
    if best is None:
        return None, {'uncovered_high_n':int(len(uncovered)),'low_n':int(len(l)),'rejected_low_n':0}
    return best[1], {'uncovered_high_n':int(len(uncovered)),'low_n':int(len(l)),'rejected_low_n':int(best[0][0]),'secondary_high_support':int(best[0][1])}


def selector_safe(train, topic_cols):
    h=train[train['incidentType']=='Hurricane'].copy()
    checks=[]; enabled_inner=0
    for fy in sorted(h['fyDeclared'].astype(int).unique()):
        itr=train[train['fyDeclared'].astype(int)!=fy].copy()
        ite=train[train['fyDeclared'].astype(int)==fy].copy()
        held=ite[(ite['incidentType']=='Hurricane')&(ite['target_clean']>=HIGH)].copy()
        if held.empty: continue
        sec,_=select_secondary(itr,topic_cols)
        if sec is None:
            checks.extend([True]*len(held))
            continue
        enabled_inner+=1
        cond=primary(held)|(pd.to_numeric(held[sec],errors='coerce').fillna(0)>0)
        checks.extend(cond.tolist())
    return bool(checks and all(checks)), enabled_inner, len(checks)


def main():
    master=normalize_master(pd.read_excel(MASTER))
    master['target_clean']=pd.to_numeric(master['totalObligatedFunding'],errors='coerce').fillna(0).clip(lower=0)
    master['actual_band']=master['target_clean'].map(funding_band); master['disasterNumber']=master['disasterNumber'].astype(int)
    ma=fetch_all_mission_assignments(); ma_hash=data_hash(ma)
    sem,_=build_semantic_rollup(master,ma); mech=initial_mechanism_counts(master,ma)
    df=master.merge(sem,on='disasterNumber',how='left').merge(mech,on='disasterNumber',how='left')
    df=df[df['incidentType']!='Biological'].copy().reset_index(drop=True)
    df['initial_usace_esf3_dfa_count']=pd.to_numeric(df['initial_usace_esf3_dfa_count'],errors='coerce').fillna(0)
    topic_cols=sorted([c for c in df.columns if c.startswith('sem_topic_') and c.endswith('_count') and pd.to_numeric(df[c],errors='coerce').nunique(dropna=True)>1])
    current=valid_cols(df,CURRENT_19)
    semcols=valid_cols(df,[c for c in df.columns if (c.startswith('sem_') or c.startswith('ma_')) and not any(b in c.lower() for b in ['oblig','fund','cost','amount','dollar'])])
    semfeat=list(dict.fromkeys(current+semcols))

    acc=pd.read_csv(ACCEPTED); acc['disasterNumber']=acc['disasterNumber'].astype(int)
    if len(acc)!=912: raise AssertionError(f'Expected 912 rows, got {len(acc)}')
    baseline_correct=int((acc[BASE_PRED]==acc['actual_band']).sum())
    if baseline_correct!=822: raise AssertionError(f'Expected 822 baseline, got {baseline_correct}')
    model=df.merge(acc[['disasterNumber',BASE_ROOT,BASE_PRED]],on='disasterNumber',how='inner',validate='one_to_one')
    if len(model)!=912: raise AssertionError(f'Expected 912 merged rows, got {len(model)}')
    model['root_actual_high']=(model['target_clean']>=HIGH).astype(int)
    roots={int(r.disasterNumber):int(getattr(r,BASE_ROOT)) for r in model.itertuples(index=False)}
    preds={int(r.disasterNumber):str(getattr(r,BASE_PRED)) for r in model.itertuples(index=False)}
    changed=[]; diags=[]

    for fy in sorted(model['fyDeclared'].astype(int).unique()):
        train=model[model['fyDeclared'].astype(int)!=fy].copy(); test=model[model['fyDeclared'].astype(int)==fy].copy()
        sec,sd=select_secondary(train,topic_cols); safe,inner_enabled,inner_n=selector_safe(train,topic_cols)
        enabled=bool(sec is not None and safe and sd.get('uncovered_high_n',0)>0)
        eligible=test[(test['incidentType']=='Hurricane')&(test[BASE_ROOT].astype(int)==1)].copy()
        if enabled:
            cond=primary(eligible)|(pd.to_numeric(eligible[sec],errors='coerce').fillna(0)>0)
            veto=eligible[~cond].copy()
        else:
            veto=eligible.iloc[0:0].copy()
        if not veto.empty:
            low_predict=build_low_predictor(train,semfeat,int(fy)); low_pred=low_predict(veto)
            for (_,r),lp in zip(veto.iterrows(),low_pred):
                dn=int(r['disasterNumber']); old=preds[dn]; roots[dn]=0; preds[dn]=str(lp)
                changed.append({'disasterNumber':dn,'state':r['state'],'fyDeclared':int(r['fyDeclared']),'actual_band':r['actual_band'],'old_pred':old,'new_pred':str(lp),'secondary_topic':sec,'primary_count':float(r['initial_usace_esf3_dfa_count']),'secondary_count':float(pd.to_numeric(r[sec],errors='coerce') if pd.notna(r[sec]) else 0.0)})
        diags.append({'outer_fy':int(fy),'enabled':enabled,'secondary_topic':sec,'inner_safe':safe,'inner_enabled_folds':inner_enabled,'inner_high_n':inner_n,**sd,'eligible_root_high_hurricane':int(len(eligible)),'veto_count':int(len(veto))})

    out=model[['disasterNumber','state','incidentType','fyDeclared','actual_band','target_clean','root_actual_high',BASE_ROOT,BASE_PRED,'initial_usace_esf3_dfa_count']].copy()
    out['baseline_root_pred_high']=out[BASE_ROOT].astype(int); out['baseline_pred']=out[BASE_PRED].astype(str)
    out['new_root_pred_high']=out['disasterNumber'].map(roots).astype(int); out['new_pred']=out['disasterNumber'].map(preds).astype(str)
    out.to_csv(OUT/'candidate_predictions.csv',index=False); pd.DataFrame(changed).to_csv(OUT/'changed_rows.csv',index=False); pd.DataFrame(diags).to_csv(OUT/'fold_diagnostics.csv',index=False)

    y=out['root_actual_high'].to_numpy(int); br=out['baseline_root_pred_high'].to_numpy(int); nr=out['new_root_pred_high'].to_numpy(int)
    base_fn=set(out.loc[(y==1)&(br==0),'disasterNumber'].astype(int)); new_fn=set(out.loc[(y==1)&(nr==0),'disasterNumber'].astype(int))
    base_fp=int(((y==0)&(br==1)).sum()); new_fp=int(((y==0)&(nr==1)).sum()); base_rec=float(recall_score(y,br,zero_division=0)); new_rec=float(recall_score(y,nr,zero_division=0))
    correct=int((out['new_pred']==out['actual_band']).sum()); metrics=six_band_metrics(out.rename(columns={'new_pred':'metric_pred'}),'metric_pred')
    if base_fn: raise AssertionError(f'Accepted 822 should have no high FNs, got {sorted(base_fn)}')
    summary={'mission_nonfinancial_sha256':ma_hash,'baseline':{'overall_correct':baseline_correct,'root_fp':base_fp,'root_high_recall':base_rec,'high_false_negatives':sorted(base_fn)},'candidate':{'overall_correct':correct,'root_fp':new_fp,'root_high_recall':new_rec,'high_false_negatives':sorted(new_fn),'end_to_end':metrics},'changed_rows':changed}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    md=['# Post-822 Hurricane two-path mechanism confirmation','',f'- Baseline: **{baseline_correct}/912**; root FP **{base_fp}**; high recall **{base_rec:.1%}**; FN **{sorted(base_fn)}**.',f'- Candidate: **{correct}/912**; root FP **{new_fp}**; high recall **{new_rec:.1%}**; FN **{sorted(new_fn)}**.','','## Changed rows','','| FEMA | State | Actual | Old | New | Secondary path | Primary count | Secondary count |','|---:|---|---|---|---|---|---:|---:|']
    for r in changed: md.append(f"| {r['disasterNumber']} | {r['state']} | {r['actual_band']} | {r['old_pred']} | {r['new_pred']} | {r['secondary_topic']} | {r['primary_count']:.0f} | {r['secondary_count']:.0f} |")
    md += ['',f"- Macro recall: **{metrics['macro_recall']:.3f}**"]
    for b,m in metrics['per_band'].items(): md.append(f"- {b}: **{m['correct']}/{m['total']} = {m['recall']:.1%}**")
    (OUT/'summary.md').write_text('\n'.join(md),encoding='utf-8'); print('\n'.join(md))

if __name__=='__main__': main()
