#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from mission_semantic_audit import normalize_master
from nonbio_all_ranges import funding_band, six_band_metrics

ROOT=Path(__file__).resolve().parents[1]
MASTER=ROOT/'master_openfema_40plus.xlsx'
ACCEPTED=ROOT/'audit_inputs'/'post827_hurricane_zero_mission'/'accepted'/'candidate_predictions.csv'
OUT=ROOT/'audit_outputs'/'nonbio_827_hurricane_zero_mission_prereq'
OUT.mkdir(parents=True,exist_ok=True)


def main():
    master=normalize_master(pd.read_excel(MASTER))
    master['disasterNumber']=master['disasterNumber'].astype(int)
    master['target_clean']=pd.to_numeric(master['totalObligatedFunding'],errors='coerce').fillna(0).clip(lower=0)
    master['actual_band']=master['target_clean'].map(funding_band)
    master['missionAssignmentCount']=pd.to_numeric(master['missionAssignmentCount'],errors='coerce').fillna(0)
    df=master[master['incidentType']!='Biological'].copy().reset_index(drop=True)

    acc=pd.read_csv(ACCEPTED)
    acc['disasterNumber']=acc['disasterNumber'].astype(int)
    if len(acc)!=912: raise AssertionError(f'Expected 912 rows, got {len(acc)}')
    pred_col='candidate_pred_new' if 'candidate_pred_new' in acc.columns else 'candidate_pred'
    base=int((acc[pred_col]==acc['actual_band']).sum())
    if base!=827: raise AssertionError(f'Expected 827, got {base}')
    df=df.merge(acc[['disasterNumber',pred_col]].rename(columns={pred_col:'accepted_pred'}),on='disasterNumber',how='inner',validate='one_to_one')
    if len(df)!=912: raise AssertionError(len(df))

    pred=dict(zip(df.disasterNumber.astype(int),df.accepted_pred.astype(str)))
    changed=[]; diag=[]
    hur=df[(df.incidentType=='Hurricane')&(df.target_clean<50_000_000)].copy()

    for ofy in sorted(hur.fyDeclared.astype(int).unique()):
        train=hur[hur.fyDeclared.astype(int)!=ofy].copy()
        test=hur[hur.fyDeclared.astype(int)==ofy].copy()

        zero=train[train.missionAssignmentCount==0]
        funded=train[train.target_clean>=100_000]
        outer_ok=(len(zero)>=5 and zero.fyDeclared.astype(int).nunique()>=2 and (zero.target_clean<100_000).all() and (funded.missionAssignmentCount>0).all())

        inner_valid_years=0; inner_zero_rows=0; inner_fail=False
        if outer_ok:
            for ify in sorted(train.fyDeclared.astype(int).unique()):
                itr=train[train.fyDeclared.astype(int)!=ify]
                ite=train[train.fyDeclared.astype(int)==ify]
                iz=itr[itr.missionAssignmentCount==0]
                ifunded=itr[itr.target_clean>=100_000]
                if len(iz)<5 or iz.fyDeclared.astype(int).nunique()<2 or not (iz.target_clean<100_000).all() or not (ifunded.missionAssignmentCount>0).all():
                    continue
                held=ite[ite.missionAssignmentCount==0]
                if held.empty:
                    continue
                inner_valid_years+=1; inner_zero_rows+=len(held)
                if not (held.target_clean<100_000).all():
                    inner_fail=True; break
        enabled=bool(outer_ok and not inner_fail and inner_valid_years>=2 and inner_zero_rows>=5)

        eligible=test[(test.accepted_pred.isin(['100K-1M','1M-50M'])) & (test.missionAssignmentCount==0)] if enabled else test.iloc[0:0]
        changed_ids=[]
        for _,r in eligible.iterrows():
            dn=int(r.disasterNumber); old=pred[dn]; pred[dn]='0-100K'; changed_ids.append(dn)
            changed.append({'disasterNumber':dn,'state':r.state,'fyDeclared':int(r.fyDeclared),'target_clean':float(r.target_clean),'actual_band':r.actual_band,'old_pred':old,'new_pred':'0-100K','missionAssignmentCount':float(r.missionAssignmentCount),'outer_zero_support_n':int(len(zero)),'outer_zero_support_years':int(zero.fyDeclared.astype(int).nunique()),'inner_valid_years':int(inner_valid_years),'inner_zero_rows':int(inner_zero_rows)})
        diag.append({'outer_fy':int(ofy),'outer_zero_support_n':int(len(zero)),'outer_zero_support_years':int(zero.fyDeclared.astype(int).nunique()),'outer_funded_n':int(len(funded)),'outer_ok':bool(outer_ok),'inner_valid_years':int(inner_valid_years),'inner_zero_rows':int(inner_zero_rows),'inner_fail':bool(inner_fail),'enabled':enabled,'changes':len(changed_ids),'changed_ids':';'.join(map(str,changed_ids))})

    out=df[['disasterNumber','state','incidentType','fyDeclared','target_clean','actual_band','accepted_pred']].copy()
    out['candidate_pred']=out.disasterNumber.map(pred)
    out.to_csv(OUT/'candidate_predictions.csv',index=False)
    pd.DataFrame(changed).to_csv(OUT/'changed_rows.csv',index=False)
    pd.DataFrame(diag).to_csv(OUT/'fold_diagnostics.csv',index=False)

    cand=int((out.candidate_pred==out.actual_band).sum())
    metrics=six_band_metrics(out,'candidate_pred')
    protected=out[out.target_clean>=50_000_000]
    pchg=int((protected.accepted_pred!=protected.candidate_pred).sum())
    correct=sum(r['actual_band']=='0-100K' for r in changed); wrong=len(changed)-correct
    summary={'baseline_correct':base,'candidate_correct':cand,'total':len(out),'changed_count':len(changed),'correct_changes':correct,'wrong_changes':wrong,'protected_ge_50m_changes':pchg,'metrics':metrics,'protocol':'Strict outer LFYO Hurricane active-response prerequisite. Per outer fold, missionAssignmentCount==0 must be supported by >=5 sub-$100K Hurricanes across >=2 fiscal years, all >=$100K training Hurricanes must have missionAssignmentCount>0, and inner LFYO must reproduce the zero-mission => sub-$100K relationship across >=2 held-out years and >=5 held-out zero-mission rows. Only accepted sub-$50M predictions above 0-100K are demoted.'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Post-827 Hurricane zero-mission prerequisite','',f'- Baseline: **{base}/912**',f'- Candidate: **{cand}/912**',f'- Changed rows: **{len(changed)}** ({correct} correct, {wrong} wrong)',f'- >=$50M prediction changes: **{pchg}**','','## Changed rows']
    if not changed: lines.append('- None')
    for r in changed:
        lines.append(f"- FEMA {r['disasterNumber']} {r['state']} FY{r['fyDeclared']}: ${r['target_clean']:,.2f} | actual {r['actual_band']} | {r['old_pred']} -> {r['new_pred']} | missions={r['missionAssignmentCount']:.0f} | training zero-mission support={r['outer_zero_support_n']} across {r['outer_zero_support_years']} years | inner validation={r['inner_zero_rows']} rows across {r['inner_valid_years']} years")
    lines+=['','## Six-band recall']
    for b,v in metrics['per_band'].items(): lines.append(f"- {b}: **{v['correct']}/{v['total']} = {v['recall']:.1%}**")
    lines.append(f"- Macro recall: **{metrics['macro_recall']:.3f}**")
    (OUT/'summary.md').write_text('\n'.join(lines)); print('\n'.join(lines))
    if pchg: raise AssertionError('Changed protected >=50M rows')

if __name__=='__main__': main()
