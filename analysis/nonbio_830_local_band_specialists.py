#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import recall_score, confusion_matrix

ROOT=Path(__file__).resolve().parents[1]
MASTER=ROOT/'master_openfema_40plus.xlsx'
ACCEPTED=ROOT/'audit_inputs'/'post830_local_specialists'/'accepted'/'candidate_predictions.csv'
OUT=ROOT/'audit_outputs'/'nonbio_830_local_band_specialists'; OUT.mkdir(parents=True,exist_ok=True)
BANDS=['0-100K','100K-1M','1M-50M','50-200M','200-500M','500M+']
LOW=BANDS[:3]; HIGH=BANDS[3:]
MARGINS=[0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.75,1.0]
FEATURES=['state','incidentType','expectedResourceLevel','disasterCategory','durationClass','durationDays','declarationDelayDays','fyDeclared','ihProgramDeclared','paProgramDeclared','hmProgramDeclared','expectedResourceScore','missionAssignmentCount','uniqueAgencyCount','uniqueMaTypeCount','uniquePriorityCount','responseComplexityScore','missionDensity','agencyDensity']

def prep_fit(train):
    cats=[c for c in FEATURES if c in {'state','incidentType','expectedResourceLevel','disasterCategory','durationClass'} or train[c].dtype=='object']
    nums=[c for c in FEATURES if c not in cats]
    p=ColumnTransformer([
      ('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cats),
      ('num',Pipeline([('imp',SimpleImputer(strategy='median'))]),nums)
    ])
    return p

def fit_local(train,bucket):
    allowed=LOW if bucket in LOW else HIGH
    t=train[(train.router_pred==bucket)&(train.actual_band.isin(allowed))].copy()
    if len(t)<8 or t.actual_band.nunique()<2: return None
    p=prep_fit(t); X=p.fit_transform(t[FEATURES]); y=t.actual_band.astype(str)
    m=ExtraTreesClassifier(n_estimators=240,max_depth=8,min_samples_leaf=2,max_features='sqrt',class_weight='balanced',random_state=100+BANDS.index(bucket),n_jobs=-1)
    m.fit(X,y)
    return p,m

def predict_local(model,test,bucket):
    if model is None or len(test)==0:
        return np.array([bucket]*len(test),dtype=object), np.zeros(len(test)), np.zeros(len(test))
    p,m=model; pr=m.predict_proba(p.transform(test[FEATURES])); classes=list(m.classes_)
    top_idx=np.argmax(pr,axis=1); cand=np.array([classes[i] for i in top_idx],dtype=object); top=pr[np.arange(len(test)),top_idx]
    if bucket in classes: stay=pr[:,classes.index(bucket)]
    else: stay=np.zeros(len(test))
    return cand,top,stay

def apply_margin(bucket,cand,top,stay,margin):
    out=[]
    for c,t,s in zip(cand,top,stay):
        if c!=bucket and (t-s)>=margin: out.append(c)
        else: out.append(bucket)
    return np.array(out,dtype=object)

def metric(y,p):
    y=np.array(y,dtype=object); p=np.array(p,dtype=object)
    rec=recall_score(y,p,labels=BANDS,average=None,zero_division=0)
    return {'correct':int((y==p).sum()),'n':len(y),'accuracy':float((y==p).mean()),'macro_recall':float(rec.mean()),'confusion':confusion_matrix(y,p,labels=BANDS).tolist(),'per_band':{b:{'correct':int(((y==b)&(p==b)).sum()),'n':int((y==b).sum()),'recall':float(rec[i])} for i,b in enumerate(BANDS)}}

def main():
    a=pd.read_csv(ACCEPTED); a.disasterNumber=a.disasterNumber.astype(int)
    pc='candidate_pred_new' if 'candidate_pred_new' in a.columns else 'candidate_pred'
    assert len(a)==912 and int((a[pc].astype(str)==a.actual_band.astype(str)).sum())==830
    master=pd.read_excel(MASTER); master.disasterNumber=master.disasterNumber.astype(int); master=master.sort_values('disasterNumber').drop_duplicates('disasterNumber')
    add=[c for c in FEATURES if c not in a.columns]
    d=a.merge(master[['disasterNumber']+add],on='disasterNumber',how='left',validate='one_to_one') if add else a.copy()
    d['router_pred']=d[pc].astype(str); d['actual_band']=d.actual_band.astype(str); d['fyDeclared']=pd.to_numeric(d.fyDeclared).astype(int)
    outs=[]; fold_diag=[]
    for ofy in sorted(d.fyDeclared.unique()):
        tr=d[d.fyDeclared!=ofy].copy(); te=d[d.fyDeclared==ofy].copy()
        selected={}
        for bucket in BANDS:
            inner_rows=[]
            for ify in sorted(tr.fyDeclared.unique()):
                tr2=tr[tr.fyDeclared!=ify].copy(); va=tr[(tr.fyDeclared==ify)&(tr.router_pred==bucket)].copy()
                if len(va)==0: continue
                model=fit_local(tr2,bucket); cand,top,stay=predict_local(model,va,bucket)
                part=va[['disasterNumber','actual_band']].copy(); part['cand']=cand; part['top']=top; part['stay']=stay; inner_rows.append(part)
            if not inner_rows:
                selected[bucket]=1.0; continue
            inn=pd.concat(inner_rows,ignore_index=True)
            best=None
            for mar in MARGINS:
                pp=apply_margin(bucket,inn.cand.values,inn.top.values,inn.stay.values,mar)
                cor=int((pp==inn.actual_band.astype(str).values).sum()); ch=int((pp!=bucket).sum())
                score=(cor,-ch,mar)
                if best is None or score>best[0]: best=(score,mar,cor,ch,len(inn))
            selected[bucket]=best[1]
            fold_diag.append({'outer_fy':ofy,'bucket':bucket,'selected_margin':best[1],'inner_correct':best[2],'inner_n':best[4],'inner_changes':best[3]})
        out=te[['disasterNumber','state','incidentType','fyDeclared','actual_band','router_pred']].copy()
        final=np.array(te.router_pred.values,dtype=object)
        for bucket in BANDS:
            idx=np.where(te.router_pred.values==bucket)[0]
            if len(idx)==0: continue
            sub=te.iloc[idx]
            model=fit_local(tr,bucket); cand,top,stay=predict_local(model,sub,bucket)
            pp=apply_margin(bucket,cand,top,stay,selected[bucket])
            final[idx]=pp
        out['local_specialist_pred']=final; outs.append(out)
    pred=pd.concat(outs,ignore_index=True).sort_values('disasterNumber').reset_index(drop=True)
    y=pred.actual_band.astype(str).values; r=pred.router_pred.astype(str).values; p=pred.local_specialist_pred.astype(str).values
    bm=metric(y,r); sm=metric(y,p)
    changed=pred[p!=r].copy(); changed['router_correct']=changed.router_pred==changed.actual_band; changed['specialist_correct']=changed.local_specialist_pred==changed.actual_band
    changed['effect']=np.where(~changed.router_correct & changed.specialist_correct,'fixed',np.where(changed.router_correct & ~changed.specialist_correct,'broken','sideways'))
    actual_high=np.isin(y,HIGH); pred_high=np.isin(p,HIGH)
    summary={'baseline':bm,'local_specialists':sm,'delta_correct':sm['correct']-bm['correct'],'changed_rows':len(changed),'fixed':int((changed.effect=='fixed').sum()),'broken':int((changed.effect=='broken').sum()),'sideways':int((changed.effect=='sideways').sum()),'true_high_recall':float((actual_high&pred_high).sum()/actual_high.sum()),'root_fp':int((~actual_high&pred_high).sum()),'root_membership_changes':int((np.isin(r,HIGH)!=pred_high).sum())}
    pred.to_csv(OUT/'candidate_predictions.csv',index=False); changed.to_csv(OUT/'changed_rows.csv',index=False); pd.DataFrame(fold_diag).to_csv(OUT/'fold_diagnostics.csv',index=False); (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Local per-router-band specialist stack','',f"- Router: **{bm['correct']}/{bm['n']} = {bm['accuracy']:.4%}**",f"- Router + local specialists: **{sm['correct']}/{sm['n']} = {sm['accuracy']:.4%}**",f"- Delta: **{summary['delta_correct']:+d}**",f"- Changed: **{len(changed)}** | fixed {summary['fixed']} | broken {summary['broken']} | sideways {summary['sideways']}",f"- Macro recall: **{sm['macro_recall']:.4%}**",f"- High root recall: **{summary['true_high_recall']:.4%}** | root FP {summary['root_fp']} | membership changes {summary['root_membership_changes']}",'','## Per-band']
    for b in BANDS:
        q=sm['per_band'][b]; qb=bm['per_band'][b]; lines.append(f"- {b}: **{q['correct']}/{q['n']} = {q['recall']:.2%}** (router {qb['correct']}/{qb['n']} = {qb['recall']:.2%})")
    (OUT/'summary.md').write_text('\n'.join(lines)); print('\n'.join(lines))
if __name__=='__main__': main()
