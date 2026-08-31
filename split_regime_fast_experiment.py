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
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor=None

MASTER=Path('master_openfema_40plus.xlsx')
COMP=Path('mission_composition_results/mission_composition_features.csv')
OUT=Path('split_regime_fast_results'); OUT.mkdir(exist_ok=True)
SAFE=['state','fyDeclared','incidentType','ihProgramDeclared','paProgramDeclared','hmProgramDeclared','durationDays','declarationDelayDays','expectedResourceLevel','expectedResourceScore','disasterCategory','durationClass','missionAssignmentCount','uniqueAgencyCount','uniqueMaTypeCount','uniquePriorityCount','responseComplexityScore','missionDensity','agencyDensity']

def met(y,p):
    return dict(R2=float(r2_score(y,p)),MAE=float(mean_absolute_error(y,p)),RMSE=float(mean_squared_error(y,p)**0.5),MedAE=float(median_absolute_error(y,p)))

def prep(X):
    cat=[c for c in X.columns if not is_numeric_dtype(X[c])]; num=[c for c in X.columns if c not in cat]
    return ColumnTransformer([('num',Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler())]),num),('cat',Pipeline([('i',SimpleImputer(strategy='most_frequent')),('o',OneHotEncoder(handle_unknown='ignore',min_frequency=2))]),cat)])

def model(name,seed):
    if name=='ridge': return Ridge(alpha=25.0)
    if name=='extra': return ExtraTreesRegressor(n_estimators=400,max_depth=7,min_samples_leaf=2,max_features=.65,random_state=seed,n_jobs=-1)
    if name=='lgbm': return LGBMRegressor(n_estimators=300,learning_rate=.03,num_leaves=7,max_depth=3,min_child_samples=5,reg_lambda=15,reg_alpha=1,verbosity=-1,random_state=seed,n_jobs=-1)
    raise ValueError(name)

def fit(train,test,features,name,target_kind,seed,lo,hi,w):
    pr=prep(train[features]); Xt=pr.fit_transform(train[features]); Xe=pr.transform(test[features])
    y=train.target.to_numpy(float); yt=np.log1p(y) if target_kind=='log' else y
    sw=np.ones(len(train)); sw[(y>lo)&(y<=hi)]=w
    m=model(name,seed); m.fit(Xt,yt,sample_weight=sw)
    z=m.predict(Xe); return np.maximum(np.expm1(z) if target_kind=='log' else z,0)

def screen(data,features,region,elo,ehi,supports,weights,mode):
    ev=data[(data.target>elo)&(data.target<=ehi)].reset_index(drop=True)
    rows=[]; store={}
    specs=[]
    for name in ['ridge','extra']+(['lgbm'] if LGBMRegressor else []):
        for tk in ['raw','log']:
            specs.append((name,tk))
    if mode=='cv':
        strat=(ev.target>100e6).astype(int).to_numpy(); seeds=[42,123]
        for sn,slo,shi in supports:
          for w in weights:
           for name,tk in specs:
            reps=[]
            for seed in seeds:
             p=np.zeros(len(ev)); cv=StratifiedKFold(n_splits=4,shuffle=True,random_state=seed)
             for _,teidx in cv.split(ev,strat):
              te=ev.iloc[teidx]; ids=set(te.disasterNumber.astype(int)); tr=data[(data.target>slo)&(data.target<=shi)&(~data.disasterNumber.astype(int).isin(ids))]
              p[teidx]=fit(tr,te,features,name,tk,seed,elo,ehi,w)
             reps.append(p)
            avg=np.mean(reps,axis=0); key=f'{sn}|w{w}|{name}_{tk}'; store[key]=avg; rows.append(dict(region=region,support=sn,weight=w,model=f'{name}_{tk}',**met(ev.target,avg)))
    else:
        for sn,slo,shi in supports:
          for w in weights:
           for name,tk in specs:
            p=np.zeros(len(ev))
            for i in range(len(ev)):
             te=ev.iloc[[i]]; did=int(te.disasterNumber.iloc[0]); tr=data[(data.target>slo)&(data.target<=shi)&(data.disasterNumber.astype(int)!=did)]
             p[i]=fit(tr,te,features,name,tk,42+i,elo,ehi,w)[0]
            key=f'{sn}|w{w}|{name}_{tk}'; store[key]=p; rows.append(dict(region=region,support=sn,weight=w,model=f'{name}_{tk}',**met(ev.target,p)))
    res=pd.DataFrame(rows).sort_values('R2',ascending=False).reset_index(drop=True)
    # Blend top 5 only, development screen.
    keys=[f"{r.support}|w{r.weight}|{r.model}" for _,r in res.head(5).iterrows()]
    blends=[]
    for i in range(len(keys)):
      for j in range(i+1,len(keys)):
       for a in [.25,.5,.75]:
        p=a*store[keys[i]]+(1-a)*store[keys[j]]; blends.append((met(ev.target,p)['R2'],a,keys[i],keys[j],p))
    if blends:
      br=max(blends,key=lambda x:x[0]); bm=met(ev.target,br[4]); res=pd.concat([pd.DataFrame([dict(region=region,support='blend',weight=br[1],model=f'{br[1]}*{br[2]}+{1-br[1]}*{br[3]}',**bm)]),res],ignore_index=True).sort_values('R2',ascending=False); bestp=br[4] if res.iloc[0].support=='blend' else store[f"{res.iloc[0].support}|w{res.iloc[0].weight}|{res.iloc[0].model}"]
    else: bestp=store[f"{res.iloc[0].support}|w{res.iloc[0].weight}|{res.iloc[0].model}"]
    pred=ev[['disasterNumber','target']].copy(); pred['prediction']=bestp
    return res,pred,res.iloc[0].to_dict()

def main():
    m=pd.read_excel(MASTER); m['target']=pd.to_numeric(m.totalObligatedFunding,errors='coerce').fillna(0).clip(lower=0)
    c=pd.read_csv(COMP)
    compact=[x for x in c.columns if x.endswith('_entropy') or x.endswith('_hhi') or x.endswith('_top_share')]
    shares=[x for x in c.columns if '_share__' in x and (pd.to_numeric(c[x],errors='coerce').fillna(0)!=0).mean()>=.08]
    c=c[['disasterNumber']+compact+shares]
    d=m[['disasterNumber','target']+SAFE].merge(c,on='disasterNumber',how='left').fillna(0); features=[x for x in d.columns if x not in ['disasterNumber','target']]
    lr,lp,lb=screen(d,features,'50-200M',50e6,200e6,[('20-300M',20e6,300e6),('20-500M',20e6,500e6),('50-500M',50e6,500e6)],[1.,3.],'cv')
    ur,up,ub=screen(d,features,'200-500M',200e6,500e6,[('100M-1B',100e6,1e9),('50M-1B',50e6,1e9),('100M+',100e6,np.inf),('50M+',50e6,np.inf)],[1.,3.,6.],'loeo')
    lr.to_csv(OUT/'lower_screen.csv',index=False); ur.to_csv(OUT/'upper_screen.csv',index=False); lp.to_csv(OUT/'lower_best_predictions.csv',index=False); up.to_csv(OUT/'upper_best_predictions.csv',index=False)
    summary={'counts':{'50_200M':int(((d.target>50e6)&(d.target<=200e6)).sum()),'200_500M':int(((d.target>200e6)&(d.target<=500e6)).sum()),'500M_plus':int((d.target>500e6).sum())},'feature_count':len(features),'lower_best_development':lb,'upper_best_development':ub,'note':'Development screening. Actual funding defines regimes only; it is not used as an input feature. Final deployment still requires a target-free router.'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
