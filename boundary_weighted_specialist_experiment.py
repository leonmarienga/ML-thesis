from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
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
OUT=Path('boundary_weighted_specialist_results'); OUT.mkdir(exist_ok=True)
SAFE=['state','fyDeclared','incidentType','ihProgramDeclared','paProgramDeclared','hmProgramDeclared','durationDays','declarationDelayDays','expectedResourceLevel','expectedResourceScore','disasterCategory','durationClass','missionAssignmentCount','uniqueAgencyCount','uniqueMaTypeCount','uniquePriorityCount','responseComplexityScore','missionDensity','agencyDensity']
BANDS=[(50e6,100e6),(100e6,200e6),(200e6,300e6),(300e6,500e6),(500e6,1e9)]
SEEDS=[42,123,7,99,2026,314,2718,17,29,53]


def met(y,p):
    return {'R2':float(r2_score(y,p)),'MAE':float(mean_absolute_error(y,p)),'RMSE':float(mean_squared_error(y,p)**0.5)}


def prep(X):
    cat=[c for c in X.columns if not is_numeric_dtype(X[c])]; num=[c for c in X.columns if c not in cat]
    return ColumnTransformer([
        ('num',Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler())]),num),
        ('cat',Pipeline([('i',SimpleImputer(strategy='most_frequent')),('o',OneHotEncoder(handle_unknown='ignore',min_frequency=2))]),cat),
    ])


def lgb(seed):
    return LGBMRegressor(n_estimators=250,learning_rate=.035,num_leaves=7,max_depth=3,min_child_samples=5,reg_lambda=15,reg_alpha=1,verbosity=-1,random_state=seed,n_jobs=-1)


def fit_model(train,test,features,kind,target_kind,seed,sw=None,band=None,band_weight=1.0):
    pr=prep(train[features]); Xt=pr.fit_transform(train[features]); Xe=pr.transform(test[features])
    y=train.target.to_numpy(float); yt=np.log1p(y) if target_kind=='log' else y
    if sw is None:
        sw=np.ones(len(train))
        if band is not None:
            lo,hi=band; sw[(y>lo)&(y<=hi)]=band_weight
    model=lgb(seed) if kind=='lgbm' else Ridge(alpha=25.0)
    model.fit(Xt,yt,sample_weight=sw); z=model.predict(Xe)
    return np.maximum(np.expm1(z) if target_kind=='log' else z,0)


def direct_expert(data,features,test,held,seed):
    out=np.zeros(len(test))
    for band,mask in [((200e6,300e6),test.target<=300e6),((300e6,500e6),test.target>300e6)]:
        idx=np.where(mask.to_numpy())[0]
        if not len(idx): continue
        te=test.iloc[idx]
        if band[1]==300e6:
            tr=data[(data.target>100e6)&(data.target<=500e6)&(~data.disasterNumber.astype(int).isin(held))]
            p=fit_model(tr,te,features,'lgbm','raw',seed,band=band,band_weight=3.)
        else:
            tr=data[(data.target>200e6)&(data.target<=1e9)&(~data.disasterNumber.astype(int).isin(held))]
            p=.5*fit_model(tr,te,features,'lgbm','raw',seed)+.5*fit_model(tr,te,features,'ridge','log',seed)
        out[idx]=np.clip(p,band[0],band[1])
    return out


def add_position_target(df):
    z=df.copy(); vals=[]
    for y in z.target.to_numpy(float):
        val=None
        for lo,hi in BANDS:
            if y>lo and y<=hi:
                val=(np.log(y)-np.log(lo))/(np.log(hi)-np.log(lo)); break
        vals.append(val)
    z['u']=vals
    return z.dropna(subset=['u'])


def position_expert(data,features,test,held,seed):
    tr=data[(data.target>50e6)&(data.target<=1e9)&(~data.disasterNumber.astype(int).isin(held))]
    tr=add_position_target(tr)
    pr=prep(tr[features]); Xt=pr.fit_transform(tr[features]); Xe=pr.transform(test[features])
    model=LGBMRegressor(n_estimators=220,learning_rate=.03,num_leaves=7,max_depth=3,min_child_samples=5,reg_lambda=20,reg_alpha=1,verbosity=-1,random_state=seed,n_jobs=-1)
    model.fit(Xt,tr.u.to_numpy()); u=np.clip(model.predict(Xe),0,1)
    out=[]
    for ui,y in zip(u,test.target.to_numpy(float)):
        lo,hi=(200e6,300e6) if y<=300e6 else (300e6,500e6)
        out.append(np.exp(np.log(lo)+ui*(np.log(hi)-np.log(lo))))
    return np.array(out)


def boundary_expert(data,features,test,held,seed):
    tr=data[(data.target>200e6)&(data.target<=1e9)&(~data.disasterNumber.astype(int).isin(held))]
    y=tr.target.to_numpy(float)
    distance=np.abs(np.log(np.maximum(y,1.)/300e6))
    sw=1.+3.*((y>300e6)&(y<=500e6)).astype(float)+4.*np.exp(-distance/.5)
    p=fit_model(tr,test,features,'lgbm','raw',seed,sw=sw)
    return np.clip(p,300e6,500e6)


def predict_architecture(data,features,test,held,seed):
    out=np.zeros(len(test)); lower=test.target<=300e6; upper=~lower
    if lower.any():
        te=test.loc[lower]; pdirect=direct_expert(data,features,te,held,seed); pposition=position_expert(data,features,te,held,seed)
        out[np.where(lower.to_numpy())[0]]=.5*pdirect+.5*pposition
    if upper.any():
        te=test.loc[upper]; pdirect=direct_expert(data,features,te,held,seed); pboundary=boundary_expert(data,features,te,held,seed)
        out[np.where(upper.to_numpy())[0]]=.5*pdirect+.5*pboundary
    return out


def validate(data,features):
    ev=data[(data.target>200e6)&(data.target<=500e6)].reset_index(drop=True)
    lower=ev.target<=300e6; upper=~lower

    loo=np.zeros(len(ev))
    for i in range(len(ev)):
        te=ev.iloc[[i]]; held={int(te.disasterNumber.iloc[0])}
        loo[i]=predict_architecture(data,features,te,held,1000+i)[0]
    loo_df=ev[['disasterNumber','target','fyDeclared']].copy(); loo_df['prediction']=loo; loo_df.to_csv(OUT/'loo_predictions.csv',index=False)

    rep_rows=[]; rep_preds=[]; strat=(ev.target>300e6).astype(int).to_numpy()
    for seed in SEEDS:
        p=np.zeros(len(ev)); cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=seed)
        for fold,(_,teidx) in enumerate(cv.split(ev,strat)):
            te=ev.iloc[teidx]; held=set(te.disasterNumber.astype(int)); p[teidx]=predict_architecture(data,features,te,held,seed+fold)
        rep_rows.append({'seed':seed,**met(ev.target,p)}); rep_preds.append(p)
    rep=pd.DataFrame(rep_rows); rep.to_csv(OUT/'repeated_3fold.csv',index=False)
    avg=np.vstack(rep_preds).mean(axis=0); avg_df=ev[['disasterNumber','target']].copy(); avg_df['prediction']=avg; avg_df.to_csv(OUT/'repeated_3fold_average_predictions.csv',index=False)

    yearp=np.zeros(len(ev)); rows=[]
    for year in sorted(ev.fyDeclared.astype(int).unique()):
        idx=np.where(ev.fyDeclared.astype(int).to_numpy()==year)[0]; te=ev.iloc[idx]; held=set(te.disasterNumber.astype(int)); p=predict_architecture(data,features,te,held,9000+year); yearp[idx]=p
        for j in range(len(te)):
            rows.append({'held_year':int(year),'disasterNumber':int(te.iloc[j].disasterNumber),'target':float(te.iloc[j].target),'prediction':float(p[j])})
    pd.DataFrame(rows).to_csv(OUT/'leave_year_out_predictions.csv',index=False)

    return {
        'LOO_overall':met(ev.target,loo),
        'LOO_200_300M':met(ev.loc[lower,'target'],loo[lower]),
        'LOO_300_500M':met(ev.loc[upper,'target'],loo[upper]),
        'repeated_3fold_average_prediction':met(ev.target,avg),
        'repeat_R2':{'mean':float(rep.R2.mean()),'min':float(rep.R2.min()),'max':float(rep.R2.max())},
        'leave_year_out_overall':met(ev.target,yearp),
        'leave_year_out_200_300M':met(ev.loc[lower,'target'],yearp[lower]),
        'leave_year_out_300_500M':met(ev.loc[upper,'target'],yearp[upper]),
    }


def main():
    m=pd.read_excel(MASTER); m['target']=pd.to_numeric(m.totalObligatedFunding,errors='coerce').fillna(0).clip(lower=0)
    c=pd.read_csv(COMP)
    compact=[x for x in c.columns if x.endswith('_entropy') or x.endswith('_hhi') or x.endswith('_top_share')]
    shares=[x for x in c.columns if '_share__' in x and (pd.to_numeric(c[x],errors='coerce').fillna(0)!=0).mean()>=.08]
    c=c[['disasterNumber']+compact+shares]
    d=m[['disasterNumber','target']+SAFE].merge(c,on='disasterNumber',how='left').fillna(0); features=[x for x in d.columns if x not in ['disasterNumber','target']]
    metrics=validate(d,features)
    summary={
        'counts':{'200_300M':int(((d.target>200e6)&(d.target<=300e6)).sum()),'300_500M':int(((d.target>300e6)&(d.target<=500e6)).sum()),'support_200M_1B':int(((d.target>200e6)&(d.target<=1e9)).sum()),'pooled_position_support_50M_1B':int(((d.target>50e6)&(d.target<=1e9)).sum())},
        'feature_count':len(features),'metrics':metrics,'target_R2':0.80,
        'architecture':{'200_300M':'50/50 direct overlapping-window expert + pooled within-band-position expert','300_500M':'50/50 direct 200M-1B expert + boundary-weighted 200M-1B expert','boundary_weight':'1 + 3*I(300M<y<=500M) + 4*exp(-abs(log(y/300M))/0.5)'},
        'notes':['No standalone model is trained on only five or four target-band observations.','No obligation amount, funding-per-mission, geographic-scale feature, or other new funding-derived predictor is used.','Repeated grouped folds and leave-year-out are robustness checks.','Actual target still determines the 200-300M versus 300-500M oracle specialist band, so this is not deployable router performance.','Boundary weighting was development-selected and must remain frozen before any final untouched evaluation.']
    }
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
