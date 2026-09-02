from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, cohen_kappa_score
from catboost import CatBoostClassifier
import full_funding_range_router_experiment as base

COMP=Path('mission_composition_results/mission_composition_features.csv')
OUT=Path('full_funding_range_router_v2_results'); OUT.mkdir(exist_ok=True)
BANDS=base.BANDS6
HIGH_HANDOFF_CUTOFF=.25
HIGH_HANDOFF_SCORE_FLOOR=1.8
EXTREME_NUMERIC_CUTOFF=.60
EXTREME_CATEGORICAL_CUTOFF=.50


def enrich(d):
    c=pd.read_csv(COMP)
    compact=[x for x in c if x.endswith('_entropy') or x.endswith('_hhi') or x.endswith('_top_share')]
    shares=[x for x in c if '_share__' in x and (pd.to_numeric(c[x],errors='coerce').fillna(0)!=0).mean()>=.08]
    z=d.merge(c[['disasterNumber']+compact+shares],on='disasterNumber',how='left')
    z[compact+shares]=z[compact+shares].fillna(0)
    numeric=['durationDays','declarationDelayDays','expectedResourceScore','missionAssignmentCount','uniqueAgencyCount','uniqueMaTypeCount','uniquePriorityCount','responseComplexityScore','missionDensity','agencyDensity','population2010','eventSize','missionAssignmentCount_event_pct_rank','responseComplexityScore_event_pct_rank','uniqueAgencyCount_event_pct_rank','population2010_event_pct_rank','missionAssignmentCount_relative_event_avg','responseComplexityScore_relative_event_avg','uniqueAgencyCount_relative_event_avg','population2010_relative_event_avg']+compact+shares
    cats=['incidentType','state','expectedResourceLevel','disasterCategory','durationClass']
    return z,numeric,cats


def fit_high_handoff(train,test,numeric):
    tr=train[train.target>1e6]
    X=tr[numeric].replace([np.inf,-np.inf],np.nan).fillna(0); Xe=test[numeric].replace([np.inf,-np.inf],np.nan).fillna(0)
    y=(tr.target>50e6).astype(int)
    m=RandomForestClassifier(n_estimators=300,min_samples_leaf=3,max_features=.8,class_weight='balanced',random_state=42,n_jobs=-1)
    m.fit(X,y); return m.predict_proba(Xe)[:,1]


def fit_extreme_categorical(train,test,numeric,cats):
    tr=train[train.target>50e6]
    feats=numeric+cats; X=tr[feats].copy(); Xe=test[feats].copy()
    for c in cats:
        X[c]=X[c].astype(str).fillna('MISSING'); Xe[c]=Xe[c].astype(str).fillna('MISSING')
    y=(tr.target>500e6).astype(int)
    if y.nunique()<2: return np.full(len(test),float(y.iloc[0]))
    n0=int((y==0).sum()); n1=int((y==1).sum())
    m=CatBoostClassifier(iterations=120,depth=3,learning_rate=.03,l2_leaf_reg=20,loss_function='Logloss',verbose=False,random_seed=42,allow_writing_files=False,class_weights=[1,n0/max(n1,1)])
    m.fit(X,y,cat_features=cats); return m.predict_proba(Xe)[:,1]


def metrics(y,p):
    pr,rc,_,_=precision_recall_fscore_support(y,p,labels=range(6),zero_division=0)
    return {'accuracy':float(accuracy_score(y,p)),'balanced_accuracy':float(balanced_accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,average='macro')),'weighted_f1':float(f1_score(y,p,average='weighted')),'within_one_band':float((np.abs(y-p)<=1).mean()),'mean_abs_band_error':float(np.abs(y-p).mean()),'quadratic_kappa':float(cohen_kappa_score(y,p,weights='quadratic')),'recall':{BANDS[i]:float(rc[i]) for i in range(6)},'precision':{BANDS[i]:float(pr[i]) for i in range(6)},'confusion_matrix':confusion_matrix(y,p,labels=range(6)).tolist()}


def main():
    raw=base.load_data(); d,numeric,cats=enrich(raw)
    base_pred=np.zeros(len(d),int); old_pred=np.zeros(len(d),int); new_pred=np.zeros(len(d),int)
    score=np.zeros(len(d)); highp=np.zeros(len(d)); extnum=np.zeros(len(d)); extcat=np.zeros(len(d)); lowp=np.zeros(len(d))
    for year in sorted(d.fyDeclared.astype(int).unique()):
        tr=d[d.fyDeclared.astype(int)!=year]; te=d[d.fyDeclared.astype(int)==year]
        bp,sc=base.fit_ordinal(tr,te,'band6_int',6); lp=base.fit_low_refiner(tr,te); en=base.fit_extreme_refiner(tr,te); hp=fit_high_handoff(tr,te,numeric); ec=fit_extreme_categorical(tr,te,numeric,cats)
        old=bp.copy(); lm=np.isin(bp,[0,1]); old[lm]=(lp[lm]>=base.LOW_CUTOFF).astype(int); old[(bp>=4)&(en>=base.EXTREME_CUTOFF)]=5
        new=bp.copy(); new[lm]=(lp[lm]>=base.LOW_CUTOFF).astype(int)
        promote=(new==2)&(hp>=HIGH_HANDOFF_CUTOFF)&(sc>=HIGH_HANDOFF_SCORE_FLOOR); new[promote]=3
        extreme=(bp>=4)&(en>=EXTREME_NUMERIC_CUTOFF)&(ec>=EXTREME_CATEGORICAL_CUTOFF); new[extreme]=5
        idx=te.index; base_pred[idx]=bp; old_pred[idx]=old; new_pred[idx]=new; score[idx]=sc; highp[idx]=hp; extnum[idx]=en; extcat[idx]=ec; lowp[idx]=lp
    y=d.band6_int.to_numpy()
    out=d[['disasterNumber','state','fyDeclared','incidentType','target','band6']].copy(); out['ordinal_score']=score; out['base_predicted_band']=[BANDS[x] for x in base_pred]; out['v1_predicted_band']=[BANDS[x] for x in old_pred]; out['high_handoff_probability_above_50M']=highp; out['numeric_extreme_probability_above_500M']=extnum; out['categorical_extreme_probability_above_500M']=extcat; out['v2_predicted_band']=[BANDS[x] for x in new_pred]; out['v2_band_distance']=np.abs(y-new_pred); out.to_csv(OUT/'strict_leave_year_out_predictions.csv',index=False)
    pd.DataFrame(confusion_matrix(y,new_pred),index=BANDS,columns=BANDS).to_csv(OUT/'confusion_matrix.csv')
    sensitivity={}
    for hc in [.20,.25,.30]:
        for sf in [1.8,2.0]:
            for ec in [.45,.50,.55]:
                p=base_pred.copy(); lm=np.isin(base_pred,[0,1]); p[lm]=(lowp[lm]>=base.LOW_CUTOFF).astype(int); p[(p==2)&(highp>=hc)&(score>=sf)]=3; p[(base_pred>=4)&(extnum>=EXTREME_NUMERIC_CUTOFF)&(extcat>=ec)]=5
                sensitivity[f'high={hc:.2f}|score={sf:.1f}|extcat={ec:.2f}']=metrics(y,p)
    summary={'v1_reference':metrics(y,old_pred),'v2_high_value_handoff':metrics(y,new_pred),'fixed_development_settings':{'low_refiner_cutoff':base.LOW_CUTOFF,'high_handoff_cutoff':HIGH_HANDOFF_CUTOFF,'high_handoff_ordinal_score_floor':HIGH_HANDOFF_SCORE_FLOOR,'extreme_numeric_cutoff':EXTREME_NUMERIC_CUTOFF,'extreme_categorical_cutoff':EXTREME_CATEGORICAL_CUTOFF},'sensitivity':sensitivity,'notes':['All test predictions exclude the entire held-out fiscal year from every fitted model.','The >50M handoff detector is trained only on >1M non-held-year support and may promote only cases already routed to 1M-50M by the ordinal backbone.','The 500M+ route now requires consensus between the original numeric extreme gate and a separate categorical high-value gate, reducing false extreme routing from the 200M-500M range.','Mission-composition features are target-free and come from the frozen MissionAssignments snapshot restored earlier in the workflow.','The cutoffs are development-selected on this dataset and are frozen here for future untouched evaluation.','The 500M-1B and 1B+ distinction remains diagnostic only because there are just three and four cases respectively.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
