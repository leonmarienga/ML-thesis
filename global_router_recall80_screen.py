from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from catboost import CatBoostClassifier
import full_funding_range_router_experiment as base

COMP=Path('mission_composition_results/mission_composition_features.csv')
OUT=Path('global_router_recall80_screen_results'); OUT.mkdir(exist_ok=True)
BANDS=base.BANDS6


def prep(X):
    cat=[c for c in X if not is_numeric_dtype(X[c])]; num=[c for c in X if c not in cat]
    return ColumnTransformer([('num',Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler())]),num),('cat',Pipeline([('i',SimpleImputer(strategy='most_frequent')),('o',OneHotEncoder(handle_unknown='ignore',min_frequency=2))]),cat)])


def event_table(d):
    e=d.groupby('event_key').agg(eventSize=('disasterNumber','size'),missionTotal=('missionAssignmentCount','sum'),complexityTotal=('responseComplexityScore','sum'),populationTotal=('population2010','sum'),agencyTotal=('uniqueAgencyCount','sum'),durationMax=('durationDays','max'),eventTarget=('target','sum'),incidentType=('incidentType','first'),minYear=('fyDeclared','min'),maxYear=('fyDeclared','max')).reset_index()
    for new,old in [('logEventSize','eventSize'),('logMissionTotal','missionTotal'),('logComplexityTotal','complexityTotal'),('logPopulationTotal','populationTotal'),('logAgencyTotal','agencyTotal'),('logDurationMax','durationMax')]: e[new]=np.log1p(pd.to_numeric(e[old],errors='coerce').fillna(0))
    return e
EVENTF=['incidentType','logEventSize','logMissionTotal','logComplexityTotal','logPopulationTotal','logAgencyTotal','logDurationMax']


def add_event_proxy_oof(d):
    z=d.copy(); z['event_proxy']=0.0; z['event_total_proxy']=0.0; z['event_share_proxy']=0.0
    e=event_table(z)
    g=z.groupby('event_key'); z['eventMissionTotal']=g.missionAssignmentCount.transform('sum'); z['eventComplexityTotal']=g.responseComplexityScore.transform('sum'); z['eventAgencyTotal']=g.uniqueAgencyCount.transform('sum'); z['eventPopulationTotal']=g.population2010.transform('sum'); z['eventTarget']=g.target.transform('sum')
    z['uniformShare']=1/z.eventSize.replace(0,np.nan); z['missionShare']=z.missionAssignmentCount/z.eventMissionTotal.replace(0,np.nan); z['complexityShare']=z.responseComplexityScore/z.eventComplexityTotal.replace(0,np.nan); z['agencyShare']=z.uniqueAgencyCount/z.eventAgencyTotal.replace(0,np.nan); z['populationShare']=z.population2010/z.eventPopulationTotal.replace(0,np.nan)
    SHAREF=['uniformShare','missionShare','complexityShare','agencyShare','populationShare']
    for year in sorted(z.fyDeclared.astype(int).unique()):
        touching=set(z.loc[z.fyDeclared.astype(int)==year,'event_key'])
        etr=e[(~e.event_key.isin(touching))&(e.eventTarget>0)].copy(); ete=e[e.event_key.isin(touching)].copy()
        p=prep(etr[EVENTF]); X=p.fit_transform(etr[EVENTF]); Xe=p.transform(ete[EVENTF]); m=Ridge(alpha=25); m.fit(X,np.log1p(etr.eventTarget)); total_pred=dict(zip(ete.event_key,np.maximum(0,np.expm1(m.predict(Xe)))))
        sr=z[(~z.event_key.isin(touching))&(z.eventTarget>0)].copy(); sr['targetShare']=sr.target/sr.eventTarget
        sm=LinearRegression(fit_intercept=False,positive=True); sm.fit(sr[SHAREF].fillna(0),sr.targetShare,sample_weight=(1/sr.eventSize).to_numpy())
        for key in touching:
            idx=z.index[z.event_key==key]; raw=np.maximum(0,sm.predict(z.loc[idx,SHAREF].fillna(0))); s=raw.sum(); sh=raw/s if s>0 else np.full(len(idx),1/len(idx)); tot=float(total_pred.get(key,0)); z.loc[idx,'event_total_proxy']=tot; z.loc[idx,'event_share_proxy']=sh; z.loc[idx,'event_proxy']=tot*sh
    z['logEventProxy']=np.log1p(z.event_proxy); z['logEventTotalProxy']=np.log1p(z.event_total_proxy)
    return z


def load():
    d=base.load_data(); c=pd.read_csv(COMP)
    compact=[x for x in c if x.endswith('_entropy') or x.endswith('_hhi') or x.endswith('_top_share')]
    shares=[x for x in c if '_share__' in x and (pd.to_numeric(c[x],errors='coerce').fillna(0)!=0).mean()>=.08]
    d=d.merge(c[['disasterNumber']+compact+shares],on='disasterNumber',how='left'); d[compact+shares]=d[compact+shares].fillna(0); d=add_event_proxy_oof(d)
    return d,compact+shares


def feature_set(d,mission):
    numeric=['logPopulation','logMission','logComplexity','logAgency','logDuration','logEventSize','expectedResourceScore','missionDensity','agencyDensity','missionAssignmentCount_event_pct_rank','responseComplexityScore_event_pct_rank','uniqueAgencyCount_event_pct_rank','population2010_event_pct_rank','missionAssignmentCount_relative_event_avg','responseComplexityScore_relative_event_avg','uniqueAgencyCount_relative_event_avg','population2010_relative_event_avg','logEventProxy','logEventTotalProxy','event_share_proxy','ihProgramDeclared','paProgramDeclared','hmProgramDeclared']+mission
    cats=['incidentType','state','expectedResourceLevel','disasterCategory','durationClass']
    return numeric,cats


def evaluate(y,pred):
    pr,rc,_,_=precision_recall_fscore_support(y,pred,labels=range(6),zero_division=0)
    return {'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro')),'min_band_recall':float(rc.min()),'recall':{BANDS[i]:float(rc[i]) for i in range(6)},'precision':{BANDS[i]:float(pr[i]) for i in range(6)},'confusion_matrix':confusion_matrix(y,pred,labels=range(6)).tolist()}


def oof_catboost(d,numeric,cats,power,loss,depth):
    probs=np.zeros((len(d),6)); pred=np.zeros(len(d),int); feats=numeric+cats
    counts=d.band6_int.value_counts(); weights=[float((len(d)/counts[i])**power) for i in range(6)]; mn=min(weights); weights=[w/mn for w in weights]
    for year in sorted(d.fyDeclared.astype(int).unique()):
        tr=d[d.fyDeclared.astype(int)!=year]; te=d[d.fyDeclared.astype(int)==year]; X=tr[feats].copy(); Xe=te[feats].copy()
        for c in cats: X[c]=X[c].astype(str).fillna('MISSING'); Xe[c]=Xe[c].astype(str).fillna('MISSING')
        m=CatBoostClassifier(iterations=180,depth=depth,learning_rate=.035,l2_leaf_reg=20,loss_function=loss,verbose=False,random_seed=42,allow_writing_files=False,class_weights=weights)
        m.fit(X,tr.band6_int,cat_features=cats); pp=m.predict_proba(Xe); probs[te.index]=pp; pred[te.index]=pp.argmax(1)
    return probs,pred,weights


def bias_search(y,probs):
    # Development diagnostic only: asks whether model probabilities contain enough signal for an 80% recall floor.
    bias=np.ones(6); best=evaluate(y,probs.argmax(1)); best_obj=(best['min_band_recall'],best['balanced_accuracy'])
    grid=[.35,.5,.7,1.0,1.4,2.0,2.8,4.0]
    for _ in range(4):
        changed=False
        for c in range(6):
            cur=bias[c]; local=(best_obj,cur,best)
            for v in grid:
                b=bias.copy(); b[c]=v; p=(probs*b).argmax(1); met=evaluate(y,p); obj=(met['min_band_recall'],met['balanced_accuracy'])
                if obj>local[0]: local=(obj,v,met)
            if local[1]!=cur: bias[c]=local[1]; best_obj=local[0]; best=local[2]; changed=True
        if not changed: break
    return bias,best


def main():
    d,mission=load(); numeric,cats=feature_set(d,mission); y=d.band6_int.to_numpy(); rows=[]; chosen=None
    configs=[]
    for loss in ['MultiClass','MultiClassOneVsAll']:
        for power in [.5,.75,1.0,1.25]:
            for depth in [2,3]: configs.append((loss,power,depth))
    for loss,power,depth in configs:
        probs,pred,cw=oof_catboost(d,numeric,cats,power,loss,depth); met=evaluate(y,pred); bias,bmet=bias_search(y,probs)
        rows.append({'loss':loss,'class_weight_power':power,'depth':depth,'metrics':met,'development_bias':bias.tolist(),'biased_metrics_diagnostic':bmet})
        np.save(OUT/f"probs_{loss}_{power}_{depth}.npy",probs)
        if chosen is None or (met['min_band_recall'],met['balanced_accuracy'])>(chosen[0],chosen[1]): chosen=(met['min_band_recall'],met['balanced_accuracy'],loss,power,depth,met,bias,bmet)
        print(loss,power,depth,met['min_band_recall'],met['balanced_accuracy'],met['recall'],'biased',bmet['min_band_recall'],bmet['recall'])
    pd.DataFrame([{'loss':r['loss'],'class_weight_power':r['class_weight_power'],'depth':r['depth'],'min_band_recall':r['metrics']['min_band_recall'],'balanced_accuracy':r['metrics']['balanced_accuracy'],'accuracy':r['metrics']['accuracy'],'biased_min_band_recall_diagnostic':r['biased_metrics_diagnostic']['min_band_recall'],'biased_balanced_accuracy_diagnostic':r['biased_metrics_diagnostic']['balanced_accuracy']} for r in rows]).sort_values(['min_band_recall','balanced_accuracy'],ascending=False).to_csv(OUT/'candidate_summary.csv',index=False)
    summary={'acceptance_target':'recall >= 0.80 in every operational funding band under strict leave-year-out validation','candidate_results':rows,'best_unbiased_candidate':{'loss':chosen[2],'class_weight_power':chosen[3],'depth':chosen[4],'metrics':chosen[5]},'best_candidate_bias_diagnostic':{'bias':chosen[6].tolist(),'metrics':chosen[7]},'notes':['Every candidate prediction excludes the entire held-out fiscal year.','Event funding proxy is built without held-year funding labels: event-total and state-share models exclude all events touching the held year.','The bias search is explicitly a development diagnostic on outer OOF probabilities and is NOT an unbiased validation result; it is used only to determine whether nested bias calibration is worth implementing next.','No model is accepted unless every band recall is at least 0.80 in a leakage-safe validation.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
