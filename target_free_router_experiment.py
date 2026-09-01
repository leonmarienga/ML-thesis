from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, log_loss, r2_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostClassifier
import temporal_event_hierarchical_specialist_v2_experiment as v2

OUT=Path('target_free_router_results'); OUT.mkdir(exist_ok=True)
ROUTER_FEATURES=['incidentType','logPopulation2010','logMission','logComplexity','logDuration','logDeclaredArea','expectedResourceScore','missionDensity','agencyDensity','missionAssignmentCount_relative_event_avg','responseComplexityScore_relative_event_avg','uniqueAgencyCount_relative_event_avg','population2010_relative_event_avg']
ROUTER_CANDIDATES=[(1,150,10),(1,250,10),(2,150,10),(2,250,10),(2,250,20)]
FAMILY_SUPPORT_THRESHOLD=3


def metrics(y,p):
    return {'R2':float(r2_score(y,p)),'MAE':float(mean_absolute_error(y,p)),'RMSE':float(mean_squared_error(y,p)**.5)}


def router_prob(train,test,depth,iterations,l2):
    X=train[ROUTER_FEATURES].copy(); Xe=test[ROUTER_FEATURES].copy(); X['incidentType']=X.incidentType.astype(str); Xe['incidentType']=Xe.incidentType.astype(str)
    y=(train.target>300e6).astype(int); n0=int((y==0).sum()); n1=int((y==1).sum())
    model=CatBoostClassifier(iterations=iterations,depth=depth,learning_rate=.03,l2_leaf_reg=l2,loss_function='Logloss',verbose=False,random_seed=42,allow_writing_files=False,class_weights=[1,n0/max(n1,1)])
    model.fit(X,y,cat_features=['incidentType']); return model.predict_proba(Xe)[:,1]


def select_router(train):
    years=sorted(train.fyDeclared.astype(int).unique()); scores={}
    for cand in ROUTER_CANDIDATES:
        yy=[]; pp=[]
        for year in years:
            tr=train[train.fyDeclared.astype(int)!=year]; va=train[train.fyDeclared.astype(int)==year]
            if tr.empty or va.empty or (tr.target>300e6).astype(int).nunique()<2: continue
            p=router_prob(tr,va,*cand); yy.extend((va.target>300e6).astype(int).tolist()); pp.extend(p.tolist())
        scores[cand]=float(log_loss(yy,np.clip(pp,1e-6,1-1e-6),labels=[0,1])) if len(set(yy))==2 else 999.0
    return min(scores,key=scores.get),scores


def candidate_specialists(d,features,e,row,year):
    te=d.loc[[row.name]]; support=d[(d.target>100e6)&(d.target<=1e9)&(d.fyDeclared.astype(int)!=year)]; same=int((support.incidentType.astype(str)==str(row.incidentType)).sum())
    if same>=FAMILY_SUPPORT_THRESHOLD:
        lower=float(.5*v2.lower_direct(d,features,te,year,9000+year)[0]+.5*v2.lower_position(d,features,te,year,9000+year)[0]); lower_method='seen_family_lower_mixture'
    else:
        pv,alpha,_=v2.unseen_family_lower(d,te,year); lower=float(pv[0]); lower_method=f'unseen_family_lower_exposure_alpha_{alpha}'
    if same>=FAMILY_SUPPORT_THRESHOLD:
        upper=float(v2.seen_family_upper(d,te,year,9000+year)[0]); upper_method='seen_family_upper_static'
    else:
        upper,_=v2.unseen_event_upper(d,e,row,year); upper=float(upper); upper_method='unseen_family_upper_event_fallback'
    return lower,upper,lower_method,upper_method,same


def main():
    d,features=v2.load_data(); e=v2.event_table(d); ev=d[(d.target>200e6)&(d.target<=500e6)].sort_values('disasterNumber').reset_index()
    rows=[]
    for year in sorted(ev.fyDeclared.astype(int).unique()):
        train=d[(d.target>100e6)&(d.target<=1e9)&(d.fyDeclared.astype(int)!=year)].copy(); cand,inner=select_router(train); te=ev[ev.fyDeclared.astype(int)==year]
        p=router_prob(train,te,*cand)
        for k,(_,r) in enumerate(te.iterrows()):
            original=d.loc[r['index']]; lower,upper,lm,um,same=candidate_specialists(d,features,e,original,year); route_upper=bool(p[k]>=.5); final=upper if route_upper else lower
            rows.append({'disasterNumber':int(r.disasterNumber),'state':r.state,'fyDeclared':int(year),'incidentType':r.incidentType,'target':float(r.target),'router_probability_upper':float(p[k]),'router_predicted_band':'300-500M' if route_upper else '200-300M','actual_band_for_evaluation':'300-500M' if r.target>300e6 else '200-300M','lower_candidate':lower,'upper_candidate':upper,'final_routed_prediction':final,'lower_method':lm,'upper_method':um,'same_family_support':same,'selected_router_params':str(cand),'inner_router_logloss':json.dumps({str(a):float(b) for a,b in inner.items()})})
    out=pd.DataFrame(rows).sort_values('disasterNumber'); out.to_csv(OUT/'strict_target_free_routed_predictions.csv',index=False)
    y=(out.target>300e6).astype(int).to_numpy(); pred=(out.router_probability_upper>=.5).astype(int).to_numpy(); prob=out.router_probability_upper.to_numpy(); final=out.final_routed_prediction.to_numpy(float)
    threshold_sensitivity={}
    for t in [.3,.4,.5,.6,.7]:
        routed=np.where(prob>=t,out.upper_candidate,out.lower_candidate); threshold_sensitivity[str(t)]={'router_accuracy':float(accuracy_score(y,prob>=t)),'router_balanced_accuracy':float(balanced_accuracy_score(y,prob>=t)),'funding':metrics(out.target,routed)}
    summary={'router_support':'100M-1B excluding the entire held-out year','router_features':ROUTER_FEATURES,'router_model':'CatBoostClassifier with hyperparameters selected by inner leave-year-out log loss','router_accuracy':float(accuracy_score(y,pred)),'router_balanced_accuracy':float(balanced_accuracy_score(y,pred)),'router_auc':float(roc_auc_score(y,prob)),'target_free_routed_funding':metrics(out.target,final),'oracle_specialist_reference':metrics(out.target,np.where(out.target<=300e6,out.lower_candidate,out.upper_candidate)),'threshold_sensitivity':threshold_sensitivity,'misrouted_disasters':out.loc[pred!=y,['disasterNumber','target','router_probability_upper','router_predicted_band','actual_band_for_evaluation']].to_dict('records'),'notes':['No true funding band is used to route a test disaster.','Every router fit excludes all rows from the held-out fiscal year.','Router hyperparameters are selected only on the remaining support by inner leave-year-out log loss.','Both lower and upper specialist candidates are generated before routing; the CatBoost router chooses between them at probability 0.5.','The router training label target>300M is used only for non-held-year training examples.','Same-event target-free context remains available to the upper unseen-family specialist; this is not strictly declaration-time causal.','The nine-case 200M-500M evaluation set is small and this architecture was developed on the same dataset, so this remains development validation rather than a pristine final test.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
