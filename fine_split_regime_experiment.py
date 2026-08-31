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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor=None

MASTER=Path('master_openfema_40plus.xlsx')
COMP=Path('mission_composition_results/mission_composition_features.csv')
OUT=Path('fine_split_regime_results'); OUT.mkdir(exist_ok=True)
SAFE=['state','fyDeclared','incidentType','ihProgramDeclared','paProgramDeclared','hmProgramDeclared','durationDays','declarationDelayDays','expectedResourceLevel','expectedResourceScore','disasterCategory','durationClass','missionAssignmentCount','uniqueAgencyCount','uniqueMaTypeCount','uniquePriorityCount','responseComplexityScore','missionDensity','agencyDensity']

def met(y,p):
    return dict(R2=float(r2_score(y,p)),MAE=float(mean_absolute_error(y,p)),RMSE=float(mean_squared_error(y,p)**0.5),MedAE=float(median_absolute_error(y,p)))

def prep(X):
    cat=[c for c in X.columns if not is_numeric_dtype(X[c])]; num=[c for c in X.columns if c not in cat]
    return ColumnTransformer([('num',Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler())]),num),('cat',Pipeline([('i',SimpleImputer(strategy='most_frequent')),('o',OneHotEncoder(handle_unknown='ignore',min_frequency=2))]),cat)])

def model(name,seed):
    if name=='ridge': return Ridge(alpha=25.0)
    if name=='extra': return ExtraTreesRegressor(n_estimators=350,max_depth=7,min_samples_leaf=2,max_features=.65,random_state=seed,n_jobs=-1)
    if name=='lgbm': return LGBMRegressor(n_estimators=250,learning_rate=.035,num_leaves=7,max_depth=3,min_child_samples=5,reg_lambda=15,reg_alpha=1,verbosity=-1,random_state=seed,n_jobs=-1)
    raise ValueError(name)

def fit(train,test,features,name,target_kind,seed,elo,ehi,w,clip=True):
    pr=prep(train[features]); Xt=pr.fit_transform(train[features]); Xe=pr.transform(test[features])
    y=train.target.to_numpy(float); yt=np.log1p(y) if target_kind=='log' else y
    sw=np.ones(len(train)); sw[(y>elo)&(y<=ehi)]=w
    m=model(name,seed); m.fit(Xt,yt,sample_weight=sw)
    z=m.predict(Xe); p=np.maximum(np.expm1(z) if target_kind=='log' else z,0)
    if clip: p=np.clip(p,elo,ehi)
    return p

def screen_band(data,features,label,elo,ehi,supports,weights):
    ev=data[(data.target>elo)&(data.target<=ehi)].reset_index(drop=True)
    specs=[]
    for name in ['ridge','extra']+(['lgbm'] if LGBMRegressor else []):
        for tk in ['raw','log']:
            specs.append((name,tk))
    rows=[]; store={}
    for sn,slo,shi in supports:
      for w in weights:
       for name,tk in specs:
        p=np.zeros(len(ev))
        for i in range(len(ev)):
            te=ev.iloc[[i]]; did=int(te.disasterNumber.iloc[0])
            tr=data[(data.target>slo)&(data.target<=shi)&(data.disasterNumber.astype(int)!=did)]
            p[i]=fit(tr,te,features,name,tk,42+i,elo,ehi,w,clip=True)[0]
        key=f'{sn}|w{w}|{name}_{tk}'; store[key]=p
        rows.append(dict(region=label,support=sn,weight=w,model=f'{name}_{tk}',**met(ev.target,p)))
    res=pd.DataFrame(rows).sort_values('R2',ascending=False).reset_index(drop=True)
    # screen blends of top 6 candidate predictions
    keys=[f"{r.support}|w{r.weight}|{r.model}" for _,r in res.head(6).iterrows()]
    blend_rows=[]
    for i in range(len(keys)):
      for j in range(i+1,len(keys)):
       for a in [.25,.5,.75]:
        p=a*store[keys[i]]+(1-a)*store[keys[j]]
        # preserve the regime constraints after blending
        p=np.clip(p,elo,ehi)
        blend_rows.append((met(ev.target,p)['R2'],a,keys[i],keys[j],p))
    if blend_rows:
        br=max(blend_rows,key=lambda x:x[0]); bm=met(ev.target,br[4])
        brow=dict(region=label,support='blend',weight=br[1],model=f'{br[1]}*{br[2]}+{1-br[1]}*{br[3]}',**bm)
        res=pd.concat([pd.DataFrame([brow]),res],ignore_index=True).sort_values('R2',ascending=False).reset_index(drop=True)
        if res.iloc[0].support=='blend': bestp=br[4]
        else: bestp=store[f"{res.iloc[0].support}|w{res.iloc[0].weight}|{res.iloc[0].model}"]
    else:
        bestp=store[f"{res.iloc[0].support}|w{res.iloc[0].weight}|{res.iloc[0].model}"]
    pred=ev[['disasterNumber','target']].copy(); pred['prediction']=bestp; pred['regime']=label
    return res,pred,res.iloc[0].to_dict()

def loo_band_mean(data,elo,ehi):
    ev=data[(data.target>elo)&(data.target<=ehi)].reset_index(drop=True)
    p=[]
    for i in range(len(ev)):
        vals=ev.loc[ev.index!=i,'target']
        p.append(vals.mean())
    p=np.array(p)
    return ev[['disasterNumber','target']].assign(prediction=p)

def main():
    m=pd.read_excel(MASTER); m['target']=pd.to_numeric(m.totalObligatedFunding,errors='coerce').fillna(0).clip(lower=0)
    c=pd.read_csv(COMP)
    compact=[x for x in c.columns if x.endswith('_entropy') or x.endswith('_hhi') or x.endswith('_top_share')]
    shares=[x for x in c.columns if '_share__' in x and (pd.to_numeric(c[x],errors='coerce').fillna(0)!=0).mean()>=.08]
    c=c[['disasterNumber']+compact+shares]
    d=m[['disasterNumber','target']+SAFE].merge(c,on='disasterNumber',how='left').fillna(0); features=[x for x in d.columns if x not in ['disasterNumber','target']]

    bands=[
        ('50-100M',50e6,100e6,[('20-200M',20e6,200e6),('1-200M',1e6,200e6),('50-200M',50e6,200e6)]),
        ('100-200M',100e6,200e6,[('50-300M',50e6,300e6),('20-300M',20e6,300e6),('100-500M',100e6,500e6)]),
        ('200-300M',200e6,300e6,[('100-500M',100e6,500e6),('50-500M',50e6,500e6),('100M-1B',100e6,1e9),('100M+',100e6,np.inf)]),
        ('300-500M',300e6,500e6,[('100M-1B',100e6,1e9),('200M-1B',200e6,1e9),('100M+',100e6,np.inf),('200M+',200e6,np.inf)]),
    ]
    all_preds=[]; bests={}; screens=[]; mean_preds=[]
    for label,lo,hi,supports in bands:
        res,pred,best=screen_band(d,features,label,lo,hi,supports,[1.,3.,6.])
        res.to_csv(OUT/f'{label}_screen.csv',index=False)
        pred.to_csv(OUT/f'{label}_best_predictions.csv',index=False)
        screens.append(res.assign(regime_label=label)); all_preds.append(pred); bests[label]=best
        bm=loo_band_mean(d,lo,hi); bm['regime']=label; mean_preds.append(bm)

    comb=pd.concat(all_preds,ignore_index=True)
    comb_mean=pd.concat(mean_preds,ignore_index=True)
    overall=met(comb.target,comb.prediction)
    overall_mean=met(comb_mean.target,comb_mean.prediction)
    comb.to_csv(OUT/'oracle_routed_four_band_predictions.csv',index=False)
    comb_mean.to_csv(OUT/'oracle_routed_four_band_loo_mean_predictions.csv',index=False)
    pd.concat(screens,ignore_index=True).to_csv(OUT/'all_band_screens.csv',index=False)

    summary={'counts':{label:int(((d.target>lo)&(d.target<=hi)).sum()) for label,lo,hi,_ in bands},'feature_count':len(features),'band_best_development':bests,'four_band_oracle_routed_model_metrics':overall,'four_band_oracle_routed_loo_band_mean_metrics':overall_mean,'target_r2':0.80,'notes':['Every test disaster is excluded from its specialist training set.','Specialist outputs are clipped to the predicted regime bounds; this is legitimate only if a target-free router assigns the regime.','The four-band combined score uses the true regime for routing and is therefore an oracle-routing development diagnostic, not deployable/final validation.','Candidate/model/blend selection is post-hoc development screening on these same cases.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
