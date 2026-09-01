from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

MASTER=Path('master_openfema_40plus.xlsx')
COMP=Path('mission_composition_results/mission_composition_features.csv')
SCALE=Path('frozen_declaration_scale_features.csv')
OUT=Path('temporal_event_hierarchical_specialist_v2_results'); OUT.mkdir(exist_ok=True)
SAFE=['state','fyDeclared','incidentType','ihProgramDeclared','paProgramDeclared','hmProgramDeclared','durationDays','declarationDelayDays','expectedResourceLevel','expectedResourceScore','disasterCategory','durationClass','missionAssignmentCount','uniqueAgencyCount','uniqueMaTypeCount','uniquePriorityCount','responseComplexityScore','missionDensity','agencyDensity']
BANDS=[(50e6,100e6),(100e6,200e6),(200e6,300e6),(300e6,500e6),(500e6,1e9)]
POP={'AL':4779736,'AK':710231,'AZ':6392017,'AR':2915918,'CA':37253956,'CO':5029196,'CT':3574097,'DE':897934,'DC':601723,'FL':18801310,'GA':9687653,'HI':1360301,'ID':1567582,'IL':12830632,'IN':6483802,'IA':3046355,'KS':2853118,'KY':4339367,'LA':4533372,'ME':1328361,'MD':5773552,'MA':6547629,'MI':9883640,'MN':5303925,'MS':2967297,'MO':5988927,'MT':989415,'NE':1826341,'NV':2700551,'NH':1316470,'NJ':8791894,'NM':2059179,'NY':19378102,'NC':9535483,'ND':672591,'OH':11536504,'OK':3751351,'OR':3831074,'PA':12702379,'RI':1052567,'SC':4625364,'SD':814180,'TN':6346105,'TX':25145561,'UT':2763885,'VT':625741,'VA':8001024,'WA':6724540,'WV':1852994,'WI':5686986,'WY':563626,'PR':3725789,'AS':55519,'GU':159358,'MP':53883,'VI':106405}
LOWER_ALPHAS=[10,25,50,75,100,150,250]
LOWER_FALLBACK_FEATURES=['logPopulation2010','logMission','logComplexity','logDeclaredArea','logPopPerArea','logMissionPerArea','logComplexityPerArea','missionAssignmentCount_relative_event_avg','responseComplexityScore_relative_event_avg','uniqueAgencyCount_relative_event_avg','population2010_relative_event_avg']


def met(y,p): return {'R2':float(r2_score(y,p)),'MAE':float(mean_absolute_error(y,p)),'RMSE':float(mean_squared_error(y,p)**.5)}
def prep(X):
    cat=[c for c in X if not is_numeric_dtype(X[c])]; num=[c for c in X if c not in cat]
    return ColumnTransformer([('num',Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler())]),num),('cat',Pipeline([('i',SimpleImputer(strategy='most_frequent')),('o',OneHotEncoder(handle_unknown='ignore',min_frequency=2))]),cat)])
def lgb(seed): return LGBMRegressor(n_estimators=250,learning_rate=.035,num_leaves=7,max_depth=3,min_child_samples=5,reg_lambda=15,reg_alpha=1,verbosity=-1,random_state=seed,n_jobs=-1)
def band_of(y):
    for b in BANDS:
        if y>b[0] and y<=b[1]: return b
    return None


def load_data():
    m=pd.read_excel(MASTER); m['target']=pd.to_numeric(m.totalObligatedFunding,errors='coerce').fillna(0).clip(lower=0)
    c=pd.read_csv(COMP); compact=[x for x in c if x.endswith('_entropy') or x.endswith('_hhi') or x.endswith('_top_share')]; shares=[x for x in c if '_share__' in x and (pd.to_numeric(c[x],errors='coerce').fillna(0)!=0).mean()>=.08]
    d=m.merge(c[['disasterNumber']+compact+shares],on='disasterNumber',how='left').merge(pd.read_csv(SCALE),on='disasterNumber',how='left')
    d[compact+shares]=d[compact+shares].fillna(0); d['declaredAreaCount']=d.declaredAreaCount.fillna(0); d['population2010']=d.state.map(POP).astype(float)
    d['event_key']=d.incidentType.astype(str)+'|'+d.incidentBeginDate.astype(str)+'|'+d.incidentEndDate.astype(str)
    for new,old in [('logPopulation2010','population2010'),('logMission','missionAssignmentCount'),('logComplexity','responseComplexityScore'),('logDuration','durationDays'),('logDeclaredArea','declaredAreaCount')]: d[new]=np.log1p(pd.to_numeric(d[old],errors='coerce').fillna(0))
    d['popPerArea']=d.population2010/d.declaredAreaCount.replace(0,np.nan); d['missionPerArea']=d.missionAssignmentCount/d.declaredAreaCount.replace(0,np.nan); d['complexityPerArea']=d.responseComplexityScore/d.declaredAreaCount.replace(0,np.nan)
    for new,old in [('logPopPerArea','popPerArea'),('logMissionPerArea','missionPerArea'),('logComplexityPerArea','complexityPerArea')]: d[new]=np.log1p(pd.to_numeric(d[old],errors='coerce').fillna(0))
    g=d.groupby('event_key'); d['eventSize']=g.disasterNumber.transform('size')
    for col in ['missionAssignmentCount','responseComplexityScore','uniqueAgencyCount','population2010']:
        total=g[col].transform('sum').replace(0,np.nan); d[col+'_event_share']=d[col]/total; d[col+'_relative_event_avg']=d[col+'_event_share']*d.eventSize
    return d,SAFE+compact+shares


def lower_direct(d,features,te,year,seed):
    tr=d[(d.target>100e6)&(d.target<=500e6)&(d.fyDeclared.astype(int)!=year)].copy(); pr=prep(tr[features]); X=pr.fit_transform(tr[features]); Xe=pr.transform(te[features]); y=tr.target.to_numpy(float); sw=np.ones(len(tr)); sw[(y>200e6)&(y<=300e6)]=3
    model=lgb(seed); model.fit(X,y,sample_weight=sw); return np.clip(model.predict(Xe),200e6,300e6)
def add_u(df):
    z=df.copy(); u=[]
    for y in z.target.to_numpy(float):
        b=band_of(y); u.append((np.log(y)-np.log(b[0]))/(np.log(b[1])-np.log(b[0])) if b else np.nan)
    z['u']=u; return z.dropna(subset=['u'])
def lower_position(d,features,te,year,seed):
    tr=add_u(d[(d.target>50e6)&(d.target<=1e9)&(d.fyDeclared.astype(int)!=year)].copy()); pr=prep(tr[features]); X=pr.fit_transform(tr[features]); Xe=pr.transform(te[features]); model=LGBMRegressor(n_estimators=220,learning_rate=.03,num_leaves=7,max_depth=3,min_child_samples=5,reg_lambda=20,reg_alpha=1,verbosity=-1,random_state=seed,n_jobs=-1); model.fit(X,tr.u); u=np.clip(model.predict(Xe),0,1); return np.exp(np.log(200e6)+u*(np.log(300e6)-np.log(200e6)))


def fit_lower_ridge(train,test,alpha):
    X=train[LOWER_FALLBACK_FEATURES].replace([np.inf,-np.inf],np.nan).fillna(0); Xe=test[LOWER_FALLBACK_FEATURES].replace([np.inf,-np.inf],np.nan).fillna(0)
    model=Pipeline([('scale',StandardScaler()),('ridge',Ridge(alpha=alpha))]); model.fit(X,train.target); return model.predict(Xe)
def select_lower_alpha(train):
    if len(train)<4: return 100,{}
    scores={}
    for alpha in LOWER_ALPHAS:
        err=[]
        for i in range(len(train)):
            tr=train.drop(train.index[i:i+1]); va=train.iloc[[i]]; p=float(fit_lower_ridge(tr,va,alpha)[0]); err.append(abs(p-float(va.target.iloc[0])))
        scores[alpha]=float(np.mean(err))
    return min(scores,key=scores.get),scores
def unseen_family_lower(d,te,year):
    tr=d[(d.target>100e6)&(d.target<=500e6)&(d.fyDeclared.astype(int)!=year)].copy(); alpha,scores=select_lower_alpha(tr); p=np.clip(fit_lower_ridge(tr,te,alpha),200e6,300e6)
    return p,alpha,scores

STATIC=['incidentType','logPopulation2010','logMission','logComplexity','logDuration','logDeclaredArea']
def seen_family_upper(d,te,year,seed):
    tr=d[(d.target>50e6)&(d.target<=1e9)&(d.fyDeclared.astype(int)!=year)].copy(); X=tr[STATIC].copy(); Xe=te[STATIC].copy(); X['incidentType']=X.incidentType.astype(str); Xe['incidentType']=Xe.incidentType.astype(str); y=tr.target.to_numpy(float); sw=np.ones(len(tr)); sw[(y>300e6)&(y<=500e6)]=4
    model=CatBoostRegressor(iterations=300,depth=2,learning_rate=.03,l2_leaf_reg=10,loss_function='RMSE',verbose=False,random_seed=seed,allow_writing_files=False); model.fit(X,y,cat_features=['incidentType'],sample_weight=sw); return np.clip(model.predict(Xe),300e6,500e6)

def event_table(d):
    e=d.groupby('event_key').agg(eventSize=('disasterNumber','size'),missionTotal=('missionAssignmentCount','sum'),complexityTotal=('responseComplexityScore','sum'),populationTotal=('population2010','sum'),agencyTotal=('uniqueAgencyCount','sum'),durationMax=('durationDays','max'),eventTarget=('target','sum'),incidentType=('incidentType','first'),minYear=('fyDeclared','min'),maxYear=('fyDeclared','max')).reset_index()
    for new,old in [('logEventSize','eventSize'),('logMissionTotal','missionTotal'),('logComplexityTotal','complexityTotal'),('logPopulationTotal','populationTotal'),('logAgencyTotal','agencyTotal'),('logDurationMax','durationMax')]: e[new]=np.log1p(pd.to_numeric(e[old],errors='coerce').fillna(0))
    return e
EVENTF=['incidentType','logEventSize','logMissionTotal','logComplexityTotal','logPopulationTotal','logAgencyTotal','logDurationMax']
def event_total(d,e,key,year):
    tr=e[(e.event_key!=key)&~((e.minYear<=year)&(e.maxYear>=year))&(e.eventTarget>0)].copy(); te=e[e.event_key==key]; pr=prep(tr[EVENTF]); X=pr.fit_transform(tr[EVENTF]); Xe=pr.transform(te[EVENTF]); model=Ridge(alpha=25); model.fit(X,np.log1p(tr.eventTarget)); return float(np.expm1(model.predict(Xe)[0]))
def mission_share(d,key,dn,year):
    z=d.copy(); g=z.groupby('event_key'); z['eventSize']=g.disasterNumber.transform('size'); z['eventTarget']=g.target.transform('sum'); z['eventMissionTotal']=g.missionAssignmentCount.transform('sum'); z['missionShare']=z.missionAssignmentCount/z.eventMissionTotal.replace(0,np.nan); touching=set(z.loc[z.fyDeclared.astype(int)==year,'event_key']); tr=z[(z.eventSize>1)&(z.eventTarget>0)&(z.target>0)&(~z.event_key.isin(touching))&(z.event_key!=key)].copy(); X=tr[['missionShare']].fillna(0); y=(tr.target/tr.eventTarget).to_numpy(float); sw=(1/tr.eventSize).to_numpy(float); model=LinearRegression(fit_intercept=False,positive=True); model.fit(X,y,sample_weight=sw); te=z[z.event_key==key]; ms=float(te.loc[te.disasterNumber.astype(int)==dn,'missionAssignmentCount'].iloc[0]/te.missionAssignmentCount.sum()); return float(model.predict(pd.DataFrame({'missionShare':[ms]}))[0]),ms,float(model.coef_[0])
def unseen_event_upper(d,e,row,year):
    total=event_total(d,e,row.event_key,year); share,ms,beta=mission_share(d,row.event_key,int(row.disasterNumber),year); return float(np.clip(total*share,300e6,500e6)),{'event_total_prediction':total,'predicted_share':share,'mission_share':ms,'share_beta':beta}


def evaluate(d,features,threshold):
    e=event_table(d); ev=d[(d.target>200e6)&(d.target<=500e6)].sort_values('disasterNumber').reset_index(drop=True); pred=np.zeros(len(ev)); methods=['']*len(ev); details=[{} for _ in range(len(ev))]
    for year in sorted(ev.fyDeclared.astype(int).unique()):
        idx=np.where(ev.fyDeclared.astype(int).to_numpy()==year)[0]; te=ev.iloc[idx]; low=te.target<=300e6; upper=~low; pp=np.zeros(len(te)); support=d[(d.target>100e6)&(d.target<=1e9)&(d.fyDeclared.astype(int)!=year)]
        if low.any():
            low_idx=np.where(low.to_numpy())[0]
            for j,(_,r) in zip(low_idx,te.loc[low].iterrows()):
                same=int((support.incidentType.astype(str)==str(r.incidentType)).sum())
                one=te.loc[[r.name]]
                if same>=threshold:
                    p=.5*lower_direct(d,features,one,year,9000+year)[0]+.5*lower_position(d,features,one,year,9000+year)[0]; method='seen_family_lower_mixture'; detail={'same_family_support':same}
                else:
                    pv,alpha,scores=unseen_family_lower(d,one,year); p=float(pv[0]); method='unseen_family_lower_exposure'; detail={'same_family_support':same,'selected_alpha':int(alpha),'inner_loo_mae_by_alpha':{str(k):float(v) for k,v in scores.items()}}
                pp[j]=p; methods[idx[j]]=method; details[idx[j]]=detail
        if upper.any():
            for j,(_,r) in zip(np.where(upper.to_numpy())[0],te.loc[upper].iterrows()):
                same=int((support.incidentType.astype(str)==str(r.incidentType)).sum())
                if same>=threshold:
                    p=float(seen_family_upper(d,te.loc[[r.name]],year,9000+year)[0]); method='seen_family_upper_static'; detail={'same_family_support':same}
                else:
                    p,detail=unseen_event_upper(d,e,r,year); detail['same_family_support']=same; method='unseen_family_upper_event_fallback'
                pp[j]=p; methods[idx[j]]=method; details[idx[j]]=detail
        pred[idx]=pp
    out=ev[['disasterNumber','state','fyDeclared','incidentType','target']].copy(); out['prediction']=pred; out['method']=methods; out['details']=[json.dumps(x) for x in details]
    lo=ev.target<=300e6; up=~lo
    return out,{'overall_200_500M':met(ev.target,pred),'subgroup_200_300M':met(ev.loc[lo,'target'],pred[lo]),'subgroup_300_500M':met(ev.loc[up,'target'],pred[up])}


def main():
    d,features=load_data(); sens={}; chosen=None
    for t in [2,3,4]:
        out,m=evaluate(d,features,t); sens[str(t)]=m; out.to_csv(OUT/f'strict_leave_year_out_threshold_{t}.csv',index=False)
        if t==3: chosen=(out,m)
    out,m=chosen; lower=out[out.target<=300e6][['disasterNumber','state','fyDeclared','incidentType','target','prediction','method','details']]; upper=out[out.target>300e6][['disasterNumber','state','fyDeclared','incidentType','target','prediction','method','details']]; lower.to_csv(OUT/'lower_200_300M_strict_predictions.csv',index=False); upper.to_csv(OUT/'upper_300_500M_strict_predictions.csv',index=False)
    summary={'family_support_threshold':3,'lower_alpha_grid':LOWER_ALPHAS,'threshold_sensitivity':sens,'selected_threshold_3_metrics':m,'counts':{'200_300M':int((out.target<=300e6).sum()),'300_500M':int((out.target>300e6).sum())},'notes':['Every supervised funding model excludes all rows from the held-out fiscal year.','For lower cases with enough same-family support, the frozen direct plus pooled-position mixture is retained.','For an unseen lower high-value family, Ridge uses static exposure plus scale-normalized within-event intensity; its alpha is selected only by inner leave-one-out MAE on the remaining 100M-500M support.','For upper seen families the frozen static CatBoost expert is retained; unseen upper families use the frozen exact-event hierarchical fallback.','No held-year funding label enters lower alpha selection, lower fallback fitting, upper event-total fitting, or upper share fitting.','Exact-event context uses target-free peer response descriptors and is not strictly declaration-time causal.','The true funding band still selects 200-300M versus 300-500M, so this remains oracle-band specialist validation rather than deployable router performance.','This v2 architecture was developed on the same small dataset; freeze it before target-free routing and any final untouched evaluation.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
