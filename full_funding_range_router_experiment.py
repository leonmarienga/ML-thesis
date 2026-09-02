from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, cohen_kappa_score
from catboost import CatBoostClassifier

MASTER=Path('master_openfema_40plus.xlsx')
OUT=Path('full_funding_range_router_results'); OUT.mkdir(exist_ok=True)
BANDS6=['0-100K','100K-1M','1M-50M','50M-200M','200M-500M','500M+']
BINS6=[-1,1e5,1e6,50e6,200e6,500e6,np.inf]
BANDS7=['0-100K','100K-1M','1M-50M','50M-200M','200M-500M','500M-1B','1B+']
BINS7=[-1,1e5,1e6,50e6,200e6,500e6,1e9,np.inf]
LOW_CUTOFF=.65
EXTREME_CUTOFF=.60
POP={'AL':4779736,'AK':710231,'AZ':6392017,'AR':2915918,'CA':37253956,'CO':5029196,'CT':3574097,'DE':897934,'DC':601723,'FL':18801310,'GA':9687653,'HI':1360301,'ID':1567582,'IL':12830632,'IN':6483802,'IA':3046355,'KS':2853118,'KY':4339367,'LA':4533372,'ME':1328361,'MD':5773552,'MA':6547629,'MI':9883640,'MN':5303925,'MS':2967297,'MO':5988927,'MT':989415,'NE':1826341,'NV':2700551,'NH':1316470,'NJ':8791894,'NM':2059179,'NY':19378102,'NC':9535483,'ND':672591,'OH':11536504,'OK':3751351,'OR':3831074,'PA':12702379,'RI':1052567,'SC':4625364,'SD':814180,'TN':6346105,'TX':25145561,'UT':2763885,'VT':625741,'VA':8001024,'WA':6724540,'WV':1852994,'WI':5686986,'WY':563626,'PR':3725789,'AS':55519,'GU':159358,'MP':53883,'VI':106405}


def load_data():
    d=pd.read_excel(MASTER)
    d['target']=pd.to_numeric(d.totalObligatedFunding,errors='coerce').fillna(0).clip(lower=0)
    d['band6']=pd.cut(d.target,bins=BINS6,labels=BANDS6,right=True).astype(str)
    d['band6_int']=d.band6.map({x:i for i,x in enumerate(BANDS6)}).astype(int)
    d['band7']=pd.cut(d.target,bins=BINS7,labels=BANDS7,right=True).astype(str)
    d['band7_int']=d.band7.map({x:i for i,x in enumerate(BANDS7)}).astype(int)
    d['population2010']=d.state.map(POP).astype(float)
    d['event_key']=d.incidentType.astype(str)+'|'+d.incidentBeginDate.astype(str)+'|'+d.incidentEndDate.astype(str)
    g=d.groupby('event_key')
    d['eventSize']=g.disasterNumber.transform('size')
    for new,old in [('logPopulation','population2010'),('logMission','missionAssignmentCount'),('logComplexity','responseComplexityScore'),('logAgency','uniqueAgencyCount'),('logDuration','durationDays'),('logEventSize','eventSize')]:
        d[new]=np.log1p(pd.to_numeric(d[old],errors='coerce').fillna(0))
    for col in ['missionAssignmentCount','responseComplexityScore','uniqueAgencyCount','population2010']:
        d[col+'_event_pct_rank']=g[col].rank(pct=True,method='average')
        total=g[col].transform('sum').replace(0,np.nan)
        d[col+'_relative_event_avg']=(d[col]/total*d.eventSize).replace([np.inf,-np.inf],np.nan).fillna(0)
    return d

ORDINAL_FEATURES=['logPopulation','logMission','logComplexity','logAgency','logDuration','logEventSize','expectedResourceScore','missionDensity','agencyDensity','missionAssignmentCount_event_pct_rank','responseComplexityScore_event_pct_rank','uniqueAgencyCount_event_pct_rank','population2010_event_pct_rank','missionAssignmentCount_relative_event_avg','responseComplexityScore_relative_event_avg','uniqueAgencyCount_relative_event_avg','population2010_relative_event_avg']
LOW_FEATURES=['state','incidentType','ihProgramDeclared','paProgramDeclared','hmProgramDeclared','durationDays','declarationDelayDays','expectedResourceLevel','expectedResourceScore','disasterCategory','durationClass','missionAssignmentCount','uniqueAgencyCount','uniqueMaTypeCount','uniquePriorityCount','responseComplexityScore','missionDensity','agencyDensity','population2010','eventSize','missionAssignmentCount_event_pct_rank','responseComplexityScore_event_pct_rank','uniqueAgencyCount_event_pct_rank','population2010_event_pct_rank','missionAssignmentCount_relative_event_avg','responseComplexityScore_relative_event_avg','uniqueAgencyCount_relative_event_avg','population2010_relative_event_avg']


def class_metrics(y,p,labels):
    pr,rc,_,_=precision_recall_fscore_support(y,p,labels=range(len(labels)),zero_division=0)
    return {'accuracy':float(accuracy_score(y,p)),'balanced_accuracy':float(balanced_accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,average='macro')),'weighted_f1':float(f1_score(y,p,average='weighted')),'within_one_band':float((np.abs(y-p)<=1).mean()),'mean_abs_band_error':float(np.abs(y-p).mean()),'quadratic_kappa':float(cohen_kappa_score(y,p,weights='quadratic')),'recall':{labels[i]:float(rc[i]) for i in range(len(labels))},'precision':{labels[i]:float(pr[i]) for i in range(len(labels))},'confusion_matrix':confusion_matrix(y,p,labels=range(len(labels))).tolist()}


def fit_ordinal(train,test,target_col,n_classes):
    X=train[ORDINAL_FEATURES].replace([np.inf,-np.inf],np.nan).fillna(0); Xe=test[ORDINAL_FEATURES].replace([np.inf,-np.inf],np.nan).fillna(0)
    counts=train[target_col].value_counts(); sw=train[target_col].map(lambda k:min(12.0,np.sqrt(len(train)/counts[k]))).to_numpy()
    model=ExtraTreesRegressor(n_estimators=300,min_samples_leaf=4,max_features=.8,random_state=42,n_jobs=-1)
    model.fit(X,train[target_col],sample_weight=sw); score=model.predict(Xe)
    return np.clip(np.rint(score),0,n_classes-1).astype(int),score


def prep_cat(df):
    X=df[LOW_FEATURES].copy(); cats=[]
    for c in LOW_FEATURES:
        if not pd.api.types.is_numeric_dtype(X[c]):
            cats.append(c); X[c]=X[c].astype(str).fillna('MISSING')
        else: X[c]=pd.to_numeric(X[c],errors='coerce')
    return X,cats


def fit_low_refiner(train,test):
    tr=train[train.target<=1e6]; X,cats=prep_cat(tr); Xe,_=prep_cat(test); y=(tr.target>1e5).astype(int)
    n0=int((y==0).sum()); n1=int((y==1).sum())
    model=CatBoostClassifier(iterations=120,depth=3,learning_rate=.04,l2_leaf_reg=15,loss_function='Logloss',verbose=False,random_seed=42,allow_writing_files=False,class_weights=[1,n0/max(n1,1)])
    model.fit(X,y,cat_features=cats); return model.predict_proba(Xe)[:,1]


def fit_extreme_refiner(train,test):
    tr=train[train.target>200e6]; X=tr[ORDINAL_FEATURES].fillna(0); Xe=test[ORDINAL_FEATURES].fillna(0); y=(tr.target>500e6).astype(int)
    if y.nunique()<2: return np.full(len(test),float(y.iloc[0]))
    model=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=2,max_features=.8,class_weight='balanced',random_state=42,n_jobs=-1)
    model.fit(X,y); return model.predict_proba(Xe)[:,1]


def operational_six_regime(d):
    base=np.zeros(len(d),int); final=np.zeros(len(d),int); score=np.zeros(len(d)); lowp=np.zeros(len(d)); extp=np.zeros(len(d))
    for year in sorted(d.fyDeclared.astype(int).unique()):
        tr=d[d.fyDeclared.astype(int)!=year]; te=d[d.fyDeclared.astype(int)==year]
        bp,sc=fit_ordinal(tr,te,'band6_int',6); lp=fit_low_refiner(tr,te); ep=fit_extreme_refiner(tr,te); fp=bp.copy()
        low_mask=np.isin(bp,[0,1]); fp[low_mask]=(lp[low_mask]>=LOW_CUTOFF).astype(int)
        extreme_mask=(bp>=4)&(ep>=EXTREME_CUTOFF); fp[extreme_mask]=5
        base[te.index]=bp; final[te.index]=fp; score[te.index]=sc; lowp[te.index]=lp; extp[te.index]=ep
    y=d.band6_int.to_numpy()
    out=d[['disasterNumber','state','fyDeclared','incidentType','target','band6']].copy(); out['ordinal_score']=score; out['base_predicted_band']=[BANDS6[x] for x in base]; out['low_refiner_probability_above_100K']=lowp; out['extreme_probability_above_500M']=extp; out['predicted_band']=[BANDS6[x] for x in final]; out['band_distance']=np.abs(y-final); out.to_csv(OUT/'operational_6_regime_predictions.csv',index=False)
    pd.DataFrame(confusion_matrix(y,final),index=BANDS6,columns=BANDS6).to_csv(OUT/'operational_6_regime_confusion.csv')
    sensitivity={}
    for lt in [.55,.60,.65,.70]:
        for et in [.50,.60,.70,.80]:
            p=base.copy(); lm=np.isin(base,[0,1]); p[lm]=(lowp[lm]>=lt).astype(int); em=(base>=4)&(extp>=et); p[em]=5
            sensitivity[f'low={lt:.2f}|extreme={et:.2f}']=class_metrics(y,p,BANDS6)
    return class_metrics(y,base,BANDS6),class_metrics(y,final,BANDS6),sensitivity


def diagnostic_seven_band(d):
    pred=np.zeros(len(d),int); score=np.zeros(len(d))
    for year in sorted(d.fyDeclared.astype(int).unique()):
        tr=d[d.fyDeclared.astype(int)!=year]; te=d[d.fyDeclared.astype(int)==year]; p,s=fit_ordinal(tr,te,'band7_int',7); pred[te.index]=p; score[te.index]=s
    y=d.band7_int.to_numpy(); out=d[['disasterNumber','state','fyDeclared','incidentType','target','band7']].copy(); out['ordinal_score']=score; out['predicted_band7']=[BANDS7[x] for x in pred]; out.to_csv(OUT/'diagnostic_7_band_predictions.csv',index=False); pd.DataFrame(confusion_matrix(y,pred),index=BANDS7,columns=BANDS7).to_csv(OUT/'diagnostic_7_band_confusion.csv'); return class_metrics(y,pred,BANDS7)


def main():
    d=load_data(); baseline,refined,sensitivity=operational_six_regime(d); diag7=diagnostic_seven_band(d)
    counts6={x:int((d.band6==x).sum()) for x in BANDS6}; counts7={x:int((d.band7==x).sum()) for x in BANDS7}
    summary={'counts_6_regime':counts6,'counts_original_7_band':counts7,'strict_leave_year_out':{'ordinal_backbone':baseline,'refined_operational_6_regime':refined,'original_7_band_diagnostic':diag7},'fixed_development_cutoffs':{'low_above_100K':LOW_CUTOFF,'extreme_above_500M':EXTREME_CUTOFF},'cutoff_sensitivity':sensitivity,'handoff':{'200M-500M':'Use the existing target-free 200M-300M vs 300M-500M sub-router, then the corresponding temporal specialist.','500M+':'Use the shared 500M+ extreme specialist; retain 500M-1B and 1B+ as reporting diagnostics until more support exists.'},'notes':['All predictions are target-free: the held-out disaster funding value and band are never model inputs.','Every outer fit removes the entire held-out fiscal year.','The global router intentionally avoids the live declaration-area snapshot; it uses only frozen master-table response descriptors, static 2010 population, and target-free within-event ranks/intensity.','The 0.65 low cutoff and 0.60 extreme cutoff were development-selected on this dataset; the sensitivity table is reported and these cutoffs must now be frozen before any final untouched evaluation.','The exact seven-band diagnostic shows that 3 cases in 500M-1B and 4 cases in 1B+ are not sufficient for a reliable standalone split; operational routing therefore combines them as 500M+.','The global six-regime router is a routing benchmark, not a funding-amount regressor. Range-specific specialists remain separate.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
