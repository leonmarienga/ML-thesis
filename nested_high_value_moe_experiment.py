from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMRegressor

MASTER=Path('master_openfema_40plus.xlsx')
COMP=Path('mission_composition_results/mission_composition_features.csv')
OUT=Path('nested_high_value_moe_results'); OUT.mkdir(exist_ok=True)
SAFE=['state','fyDeclared','incidentType','ihProgramDeclared','paProgramDeclared','hmProgramDeclared','durationDays','declarationDelayDays','expectedResourceLevel','expectedResourceScore','disasterCategory','durationClass','missionAssignmentCount','uniqueAgencyCount','uniqueMaTypeCount','uniquePriorityCount','responseComplexityScore','missionDensity','agencyDensity']
BANDS=[(50e6,100e6),(100e6,200e6),(200e6,300e6),(300e6,500e6),(500e6,1e9)]
ALPHAS=[0.0,0.25,0.5,0.75,1.0]  # alpha*direct + (1-alpha)*pooled-position
SEEDS=[42,123,7,99,2026,314,2718,17,29,53]

def met(y,p):
    return dict(R2=float(r2_score(y,p)),MAE=float(mean_absolute_error(y,p)),RMSE=float(mean_squared_error(y,p)**0.5))

def prep(X):
    cat=[c for c in X.columns if not is_numeric_dtype(X[c])]; num=[c for c in X.columns if c not in cat]
    return ColumnTransformer([('num',Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler())]),num),('cat',Pipeline([('i',SimpleImputer(strategy='most_frequent')),('o',OneHotEncoder(handle_unknown='ignore',min_frequency=2))]),cat)])

def lgb(seed):
    return LGBMRegressor(n_estimators=250,learning_rate=.035,num_leaves=7,max_depth=3,min_child_samples=5,reg_lambda=15,reg_alpha=1,verbosity=-1,random_state=seed,n_jobs=-1)

def fit_direct(train,test,features,kind,target_kind,band,weight,seed):
    pr=prep(train[features]); Xt=pr.fit_transform(train[features]); Xe=pr.transform(test[features])
    y=train.target.to_numpy(float); yt=np.log1p(y) if target_kind=='log' else y
    sw=np.ones(len(train)); lo,hi=band; sw[(y>lo)&(y<=hi)]=weight
    if kind=='lgbm': model=lgb(seed)
    elif kind=='ridge': model=Ridge(alpha=25.0)
    else: raise ValueError(kind)
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
            p=fit_direct(tr,te,features,'lgbm','raw',band,3.0,seed)
        else:
            tr=data[(data.target>200e6)&(data.target<=1e9)&(~data.disasterNumber.astype(int).isin(held))]
            p1=fit_direct(tr,te,features,'lgbm','raw',band,1.0,seed)
            p2=fit_direct(tr,te,features,'ridge','log',band,1.0,seed)
            p=.5*p1+.5*p2
        out[idx]=np.clip(p,band[0],band[1])
    return out

def add_position_target(df):
    z=df.copy(); u=[]
    for y in z.target.to_numpy(float):
        found=None
        for lo,hi in BANDS:
            if y>lo and y<=hi:
                found=(np.log(y)-np.log(lo))/(np.log(hi)-np.log(lo)); break
        u.append(found)
    z['u']=u
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

def repeated_fixed(data,features,ev):
    strat=(ev.target>300e6).astype(int).to_numpy(); rows=[]; allp=[]
    for seed in SEEDS:
        cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=seed); p=np.zeros(len(ev))
        for fold,(_,teidx) in enumerate(cv.split(ev,strat)):
            te=ev.iloc[teidx]; held=set(te.disasterNumber.astype(int))
            pdirect=direct_expert(data,features,te,held,seed+fold)
            ppos=position_expert(data,features,te,held,seed+fold)
            p[teidx]=.5*pdirect+.5*ppos
        rows.append(dict(seed=seed,**met(ev.target,p))); allp.append(p)
    allp=np.vstack(allp); avg=allp.mean(axis=0)
    return pd.DataFrame(rows),avg,met(ev.target,avg)

def nested_loo(data,features,ev):
    preds=[]; choice_rows=[]
    for oi in range(len(ev)):
        outer=ev.iloc[[oi]]; outer_id=int(outer.disasterNumber.iloc[0])
        inner=ev.drop(index=oi).reset_index(drop=True); strat=(inner.target>300e6).astype(int).to_numpy()
        n_splits=min(3,int(np.bincount(strat).min()))
        pdirect=np.zeros(len(inner)); ppos=np.zeros(len(inner)); counts=np.zeros(len(inner))
        for seed in [42,123,7]:
            cv=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
            for fold,(_,teidx) in enumerate(cv.split(inner,strat)):
                te=inner.iloc[teidx]; held=set(te.disasterNumber.astype(int)); held.add(outer_id)
                pdirect[teidx]+=direct_expert(data,features,te,held,seed+fold)
                ppos[teidx]+=position_expert(data,features,te,held,seed+fold)
                counts[teidx]+=1
        pdirect/=counts; ppos/=counts
        scores={a:float(r2_score(inner.target,a*pdirect+(1-a)*ppos)) for a in ALPHAS}
        best=max(scores,key=scores.get)
        held={outer_id}; od=direct_expert(data,features,outer,held,1000+oi)[0]; op=position_expert(data,features,outer,held,1000+oi)[0]
        pred=best*od+(1-best)*op; preds.append(pred)
        choice_rows.append(dict(disasterNumber=outer_id,target=float(outer.target.iloc[0]),chosen_alpha_direct=float(best),direct_prediction=float(od),position_prediction=float(op),prediction=float(pred),inner_best_R2=float(scores[best]),inner_scores=json.dumps({str(k):v for k,v in scores.items()})))
    return np.array(preds),pd.DataFrame(choice_rows),met(ev.target,np.array(preds))

def main():
    m=pd.read_excel(MASTER); m['target']=pd.to_numeric(m.totalObligatedFunding,errors='coerce').fillna(0).clip(lower=0)
    c=pd.read_csv(COMP)
    compact=[x for x in c.columns if x.endswith('_entropy') or x.endswith('_hhi') or x.endswith('_top_share')]
    shares=[x for x in c.columns if '_share__' in x and (pd.to_numeric(c[x],errors='coerce').fillna(0)!=0).mean()>=.08]
    c=c[['disasterNumber']+compact+shares]
    d=m[['disasterNumber','target']+SAFE].merge(c,on='disasterNumber',how='left').fillna(0); features=[x for x in d.columns if x not in ['disasterNumber','target']]
    ev=d[(d.target>200e6)&(d.target<=500e6)].reset_index(drop=True)
    rep,avg,fixed= repeated_fixed(d,features,ev); rep.to_csv(OUT/'fixed_50_50_repeated_3fold.csv',index=False)
    pd.DataFrame({'disasterNumber':ev.disasterNumber,'target':ev.target,'prediction':avg}).to_csv(OUT/'fixed_50_50_average_predictions.csv',index=False)
    nested,choices,nmet=nested_loo(d,features,ev); choices.to_csv(OUT/'nested_loo_predictions.csv',index=False)
    summary={'counts':{'200_300M':int(((ev.target>200e6)&(ev.target<=300e6)).sum()),'300_500M':int(((ev.target>300e6)&(ev.target<=500e6)).sum()),'pooled_position_support_50M_1B':int(((d.target>50e6)&(d.target<=1e9)).sum())},'feature_count':len(features),'fixed_50_50_repeated_3fold_average_prediction_metrics':fixed,'fixed_50_50_repeat_R2':{'mean':float(rep.R2.mean()),'min':float(rep.R2.min()),'max':float(rep.R2.max())},'nested_leave_one_disaster_out_metrics':nmet,'target_R2':0.80,'notes':['No standalone model is trained on only the five 200-300M cases or four 300-500M cases.','The direct expert uses overlapping high-value support windows; the pooled-position expert learns relative within-band position from all 50M-1B cases.','In nested LOO, the outer disaster is excluded from all inner model fitting and blend-weight selection.','Actual target still determines whether evaluation uses the 200-300M or 300-500M specialist band, so these remain oracle-routing specialist diagnostics, not deployable router performance.','The base expert designs were developed using prior experiments on this dataset; nested LOO reduces selection leakage for the mixture weight but is not a fully untouched final test set.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
