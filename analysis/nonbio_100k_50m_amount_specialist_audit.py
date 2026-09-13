#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

from mission_semantic_audit import CURRENT_19, build_semantic_rollup, fetch_all_mission_assignments, normalize_master
from nonbio_all_ranges import valid_cols
from nonbio_cross_1m_rescue import fit_base_reg, predict_reg_dollars

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / 'master_openfema_40plus.xlsx'
OUT = ROOT / 'audit_outputs' / 'nonbio_100k_50m_amount_specialist_audit'
OUT.mkdir(parents=True, exist_ok=True)


def metrics(y, p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    ly=np.log1p(y); lp=np.log1p(np.maximum(p,0))
    return {
        'n': int(len(y)),
        'R2': float(r2_score(y,p)) if len(y)>1 and np.var(y)>0 else None,
        'MAE': float(mean_absolute_error(y,p)),
        'RMSE': float(mean_squared_error(y,p)**0.5),
        'MedAE': float(median_absolute_error(y,p)),
        'log_R2': float(r2_score(ly,lp)) if len(y)>1 and np.var(ly)>0 else None,
        'log_MAE': float(mean_absolute_error(ly,lp)),
        'log_RMSE': float(mean_squared_error(ly,lp)**0.5),
    }


def main():
    master=normalize_master(pd.read_excel(MASTER))
    master['target_clean']=pd.to_numeric(master['totalObligatedFunding'],errors='coerce').fillna(0).clip(lower=0)
    ma=fetch_all_mission_assignments()
    sem,_=build_semantic_rollup(master,ma)
    df=master.merge(sem,on='disasterNumber',how='left')
    df=df[(df.incidentType!='Biological') & (df.target_clean>=100_000) & (df.target_clean<50_000_000)].copy().reset_index(drop=True)

    current=valid_cols(df,CURRENT_19)
    semcols=valid_cols(df,[c for c in df.columns if (c.startswith('sem_') or c.startswith('ma_')) and not any(b in c.lower() for b in ['oblig','fund','cost','amount','dollar'])])
    features=list(dict.fromkeys(current+semcols))

    rows=[]
    folds=[]
    for fy in sorted(df.fyDeclared.astype(int).unique()):
        tr=df[df.fyDeclared.astype(int)!=fy].copy()
        te=df[df.fyDeclared.astype(int)==fy].copy()
        if te.empty: continue
        model=fit_base_reg(tr,features,210000+int(fy))
        pred=predict_reg_dollars(model,te,features)
        for (_,r),p in zip(te.iterrows(),pred):
            band='100K-1M' if r.target_clean<1_000_000 else '1M-50M'
            rows.append({'disasterNumber':int(r.disasterNumber),'fyDeclared':int(fy),'state':r.state,'incidentType':r.incidentType,'target':float(r.target_clean),'band':band,'prediction':float(max(p,0))})
        fm=metrics(te.target_clean,pred)
        fm['fyDeclared']=int(fy); folds.append(fm)

    out=pd.DataFrame(rows).sort_values('disasterNumber').reset_index(drop=True)
    overall=metrics(out.target,out.prediction)
    low=out[out.band=='100K-1M']; mid=out[out.band=='1M-50M']
    summary={
        'model':'Preserved ExtraTreesRegressor from nonbio_cross_1m_rescue.py; 900 trees; log1p target; strict LFYO',
        'feature_count':len(features),
        'overall_100k_50m':overall,
        '100k_1m':metrics(low.target,low.prediction),
        '1m_50m':metrics(mid.target,mid.prediction),
        'within_factor_2':float(((out.prediction>=out.target/2)&(out.prediction<=out.target*2)).mean()),
        'within_factor_3':float(((out.prediction>=out.target/3)&(out.prediction<=out.target*3)).mean()),
        'notes':['Biological excluded.','Every test prediction excludes the entire held-out fiscal year.','No target-derived financial feature is used as an input.','This audits the preserved low/mid amount regressor directly rather than using it only as a $1M classification boundary.']
    }
    out.to_csv(OUT/'predictions.csv',index=False)
    pd.DataFrame(folds).to_csv(OUT/'fold_metrics.csv',index=False)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=['# Preserved $100K-$50M amount specialist audit','',f"- Overall: n={overall['n']}, R² **{overall['R2']:.4f}**, MAE **${overall['MAE']:,.0f}**, RMSE **${overall['RMSE']:,.0f}**, log MAE **{overall['log_MAE']:.4f}**",f"- $100K-$1M: n={summary['100k_1m']['n']}, R² **{summary['100k_1m']['R2']:.4f}**, MAE **${summary['100k_1m']['MAE']:,.0f}**, log MAE **{summary['100k_1m']['log_MAE']:.4f}**",f"- $1M-$50M: n={summary['1m_50m']['n']}, R² **{summary['1m_50m']['R2']:.4f}**, MAE **${summary['1m_50m']['MAE']:,.0f}**, log MAE **{summary['1m_50m']['log_MAE']:.4f}**",f"- Within factor 2: **{summary['within_factor_2']:.2%}**",f"- Within factor 3: **{summary['within_factor_3']:.2%}**"]
    (OUT/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines))

if __name__=='__main__': main()
