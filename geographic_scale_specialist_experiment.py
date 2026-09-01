from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMRegressor

MASTER=Path('master_openfema_40plus.xlsx')
COMP=Path('mission_composition_results/mission_composition_features.csv')
OUT=Path('geographic_scale_specialist_results'); OUT.mkdir(exist_ok=True)
DECL_URL='https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries'
CACHE=OUT/'DisasterDeclarationsSummaries_v2.jsonl'
SAFE=['state','fyDeclared','incidentType','ihProgramDeclared','paProgramDeclared','hmProgramDeclared','durationDays','declarationDelayDays','expectedResourceLevel','expectedResourceScore','disasterCategory','durationClass','missionAssignmentCount','uniqueAgencyCount','uniqueMaTypeCount','uniquePriorityCount','responseComplexityScore','missionDensity','agencyDensity']
BANDS=[(50e6,100e6),(100e6,200e6),(200e6,300e6),(300e6,500e6),(500e6,1e9)]
SEEDS=[42,123,7,99,2026,314,2718,17,29,53]


def met(y,p):
    return {'R2':float(r2_score(y,p)),'MAE':float(mean_absolute_error(y,p)),'RMSE':float(mean_squared_error(y,p)**0.5)}


def fetch_declarations(disaster_numbers,chunk_size=40):
    if CACHE.exists(): return pd.read_json(CACHE,lines=True)
    rows=[]; s=requests.Session()
    for start in range(0,len(disaster_numbers),chunk_size):
        chunk=disaster_numbers[start:start+chunk_size]
        filt=' or '.join(f'disasterNumber eq {int(n)}' for n in chunk)
        skip=0
        while True:
            r=s.get(DECL_URL,params={'$filter':filt,'$top':1000,'$skip':skip},timeout=60); r.raise_for_status()
            payload=r.json(); batch=[]
            for key in ['DisasterDeclarationsSummaries','DisasterDeclarationSummaries']:
                if isinstance(payload.get(key),list): batch=payload[key]; break
            if not batch:
                batch=next((v for v in payload.values() if isinstance(v,list)),[])
            rows.extend(batch)
            if len(batch)<1000: break
            skip+=1000; time.sleep(.05)
        print(f'Declarations {min(start+chunk_size,len(disaster_numbers))}/{len(disaster_numbers)}; rows={len(rows):,}')
    raw=pd.DataFrame(rows); raw.to_json(CACHE,orient='records',lines=True); return raw


def declaration_scale_features(raw,disaster_numbers):
    if raw.empty: raise RuntimeError('No DisasterDeclarationsSummaries rows downloaded')
    z=raw.copy(); z['disasterNumber']=pd.to_numeric(z['disasterNumber'],errors='coerce'); z=z[z.disasterNumber.notna()].copy(); z['disasterNumber']=z.disasterNumber.astype(int)
    area=next((c for c in ['designatedArea','declaredCountyArea'] if c in z.columns),None)
    if area is None: raise RuntimeError(f'No geographic area field found. Columns: {list(z.columns)}')
    z[area]=z[area].fillna('').astype(str).str.strip();
    if 'placeCode' not in z.columns: z['placeCode']=''
    z['placeCode']=z['placeCode'].fillna('').astype(str).str.strip()
    rows=[]
    for dn,g in z.groupby('disasterNumber'):
        areas=g.loc[g[area]!='',area]
        places=g.loc[g.placeCode!='','placeCode']
        rows.append({'disasterNumber':int(dn),'declaredAreaCount':int(areas.nunique()),'uniquePlaceCodeCount':int(places.nunique()),'declarationGeographyRowCount':int(len(g))})
    f=pd.DataFrame({'disasterNumber':disaster_numbers}).merge(pd.DataFrame(rows),on='disasterNumber',how='left').fillna(0)
    return f


def prep(X):
    cat=[c for c in X.columns if not is_numeric_dtype(X[c])]; num=[c for c in X.columns if c not in cat]
    return ColumnTransformer([('num',Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler())]),num),('cat',Pipeline([('i',SimpleImputer(strategy='most_frequent')),('o',OneHotEncoder(handle_unknown='ignore',min_frequency=2))]),cat)])


def lgb(seed):
    return LGBMRegressor(n_estimators=250,learning_rate=.035,num_leaves=7,max_depth=3,min_child_samples=5,reg_lambda=15,reg_alpha=1,verbosity=-1,random_state=seed,n_jobs=-1)


def fit_model(train,test,features,kind,target_kind,band,seed,sw=None,band_weight=1.0):
    pr=prep(train[features]); Xt=pr.fit_transform(train[features]); Xe=pr.transform(test[features]); y=train.target.to_numpy(float); yt=np.log1p(y) if target_kind=='log' else y
    if sw is None:
        sw=np.ones(len(train)); lo,hi=band; sw[(y>lo)&(y<=hi)]=band_weight
    model=lgb(seed) if kind=='lgbm' else Ridge(alpha=25.0); model.fit(Xt,yt,sample_weight=sw); z=model.predict(Xe)
    return np.maximum(np.expm1(z) if target_kind=='log' else z,0)


def direct_expert(data,features,test,held,seed):
    out=np.zeros(len(test))
    for band,mask in [((200e6,300e6),test.target<=300e6),((300e6,500e6),test.target>300e6)]:
        idx=np.where(mask.to_numpy())[0]
        if not len(idx): continue
        te=test.iloc[idx]
        if band[1]==300e6:
            tr=data[(data.target>100e6)&(data.target<=500e6)&(~data.disasterNumber.astype(int).isin(held))]
            p=fit_model(tr,te,features,'lgbm','raw',band,seed,band_weight=3.)
        else:
            tr=data[(data.target>200e6)&(data.target<=1e9)&(~data.disasterNumber.astype(int).isin(held))]
            p=.5*fit_model(tr,te,features,'lgbm','raw',band,seed)+.5*fit_model(tr,te,features,'ridge','log',band,seed)
        out[idx]=np.clip(p,band[0],band[1])
    return out


def add_position_target(df):
    z=df.copy(); vals=[]
    for y in z.target.to_numpy(float):
        val=None
        for lo,hi in BANDS:
            if y>lo and y<=hi: val=(np.log(y)-np.log(lo))/(np.log(hi)-np.log(lo)); break
        vals.append(val)
    z['u']=vals; return z.dropna(subset=['u'])


def position_expert(data,features,test,held,seed):
    tr=add_position_target(data[(data.target>50e6)&(data.target<=1e9)&(~data.disasterNumber.astype(int).isin(held))])
    pr=prep(tr[features]); Xt=pr.fit_transform(tr[features]); Xe=pr.transform(test[features])
    model=LGBMRegressor(n_estimators=220,learning_rate=.03,num_leaves=7,max_depth=3,min_child_samples=5,reg_lambda=20,reg_alpha=1,verbosity=-1,random_state=seed,n_jobs=-1); model.fit(Xt,tr.u.to_numpy()); u=np.clip(model.predict(Xe),0,1)
    out=[]
    for ui,y in zip(u,test.target.to_numpy(float)):
        lo,hi=(200e6,300e6) if y<=300e6 else (300e6,500e6); out.append(np.exp(np.log(lo)+ui*(np.log(hi)-np.log(lo))))
    return np.array(out)


def boundary_expert(data,features,test,held,seed):
    tr=data[(data.target>200e6)&(data.target<=1e9)&(~data.disasterNumber.astype(int).isin(held))]
    y=tr.target.to_numpy(float); dist=np.abs(np.log(np.maximum(y,1.)/300e6)); sw=1.+3.*((y>300e6)&(y<=500e6)).astype(float)+4.*np.exp(-dist/.5)
    p=fit_model(tr,test,features,'lgbm','raw',(300e6,500e6),seed,sw=sw)
    return np.clip(p,300e6,500e6)


def architecture_predict(data,features,test,held,seed):
    out=np.zeros(len(test)); lower=test.target<=300e6; upper=~lower
    if lower.any():
        te=test.loc[lower]; pd=direct_expert(data,features,te,held,seed); pp=position_expert(data,features,te,held,seed); out[np.where(lower.to_numpy())[0]]=.5*pd+.5*pp
    if upper.any():
        te=test.loc[upper]; pd=direct_expert(data,features,te,held,seed); pb=boundary_expert(data,features,te,held,seed); out[np.where(upper.to_numpy())[0]]=.5*pd+.5*pb
    return out


def validate(data,features,label):
    ev=data[(data.target>200e6)&(data.target<=500e6)].reset_index(drop=True)
    loo=np.zeros(len(ev))
    for i in range(len(ev)):
        te=ev.iloc[[i]]; held={int(te.disasterNumber.iloc[0])}; loo[i]=architecture_predict(data,features,te,held,1000+i)[0]
    lp=ev[['disasterNumber','target','fyDeclared']].copy(); lp['prediction']=loo; lp.to_csv(OUT/f'{label}_loo_predictions.csv',index=False)
    lower=ev.target<=300e6; upper=~lower
    summary={'LOO_overall':met(ev.target,loo),'LOO_200_300M':met(ev.loc[lower,'target'],loo[lower]),'LOO_300_500M':met(ev.loc[upper,'target'],loo[upper])}

    strat=(ev.target>300e6).astype(int).to_numpy(); reps=[]; avg=[]
    for seed in SEEDS:
        p=np.zeros(len(ev)); cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=seed)
        for fold,(_,teidx) in enumerate(cv.split(ev,strat)):
            te=ev.iloc[teidx]; held=set(te.disasterNumber.astype(int)); p[teidx]=architecture_predict(data,features,te,held,seed+fold)
        mm=met(ev.target,p); reps.append({'seed':seed,**mm}); avg.append(p)
    rdf=pd.DataFrame(reps); rdf.to_csv(OUT/f'{label}_repeated_3fold.csv',index=False); ap=np.vstack(avg).mean(axis=0)
    summary['repeated_3fold_average_prediction']=met(ev.target,ap); summary['repeat_R2']={'mean':float(rdf.R2.mean()),'min':float(rdf.R2.min()),'max':float(rdf.R2.max())}

    gp=np.zeros(len(ev)); grows=[]
    for year in sorted(ev.fyDeclared.astype(int).unique()):
        idx=np.where(ev.fyDeclared.astype(int).to_numpy()==year)[0]; te=ev.iloc[idx]; held=set(te.disasterNumber.astype(int)); p=architecture_predict(data,features,te,held,9000+year); gp[idx]=p
        grows.extend([{'held_year':int(year),'disasterNumber':int(te.iloc[j].disasterNumber),'target':float(te.iloc[j].target),'prediction':float(p[j])} for j in range(len(te))])
    pd.DataFrame(grows).to_csv(OUT/f'{label}_leave_year_out_predictions.csv',index=False)
    summary['leave_year_out_overall']=met(ev.target,gp); summary['leave_year_out_200_300M']=met(ev.loc[lower,'target'],gp[lower]); summary['leave_year_out_300_500M']=met(ev.loc[upper,'target'],gp[upper])
    return summary


def main():
    m=pd.read_excel(MASTER); m['target']=pd.to_numeric(m.totalObligatedFunding,errors='coerce').fillna(0).clip(lower=0); dns=m.disasterNumber.astype(int).tolist()
    c=pd.read_csv(COMP); compact=[x for x in c.columns if x.endswith('_entropy') or x.endswith('_hhi') or x.endswith('_top_share')]; shares=[x for x in c.columns if '_share__' in x and (pd.to_numeric(c[x],errors='coerce').fillna(0)!=0).mean()>=.08]; c=c[['disasterNumber']+compact+shares]
    base=m[['disasterNumber','target']+SAFE].merge(c,on='disasterNumber',how='left').fillna(0); base_features=[x for x in base.columns if x not in ['disasterNumber','target']]
    raw=fetch_declarations(dns); scale=declaration_scale_features(raw,dns); scale.to_csv(OUT/'declaration_scale_features.csv',index=False)
    aug=base.merge(scale,on='disasterNumber',how='left').fillna(0); aug['missionsPerDeclaredArea']=aug.missionAssignmentCount/aug.declaredAreaCount.clip(lower=1); aug['agenciesPerDeclaredArea']=aug.uniqueAgencyCount/aug.declaredAreaCount.clip(lower=1); aug['complexityPerDeclaredArea']=aug.responseComplexityScore/aug.declaredAreaCount.clip(lower=1); aug_features=[x for x in aug.columns if x not in ['disasterNumber','target']]
    baseline=validate(base,base_features,'baseline_no_geographic_scale'); augmented=validate(aug,aug_features,'with_geographic_scale')
    high=aug[(aug.target>200e6)&(aug.target<=500e6)][['disasterNumber','target','state','incidentType','declaredAreaCount','uniquePlaceCodeCount','declarationGeographyRowCount','missionsPerDeclaredArea','agenciesPerDeclaredArea','complexityPerDeclaredArea']].sort_values('target'); high.to_csv(OUT/'high_value_scale_diagnostics.csv',index=False)
    summary={'feature_counts':{'baseline':len(base_features),'augmented':len(aug_features)},'high_value_counts':{'200_300M':int(((aug.target>200e6)&(aug.target<=300e6)).sum()),'300_500M':int(((aug.target>300e6)&(aug.target<=500e6)).sum())},'baseline':baseline,'with_geographic_scale':augmented,'notes':['No obligation amount, funding-per-mission, or other funding-derived predictor is used.','Declared-area and place-code counts come from DisasterDeclarationsSummaries and are target-free geographic coverage descriptors.','The 200-300M expert still borrows from 100-500M plus pooled 50M-1B position data.','The 300-500M expert uses 200M-1B support with smooth target-side training weights around the 300M boundary; it is not trained on only four observations.','All reported specialist diagnostics still use the true funding band for oracle routing.','The boundary-weight configuration is development-selected; leave-year-out and repeated grouped folds are robustness checks, not a pristine final test set.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
