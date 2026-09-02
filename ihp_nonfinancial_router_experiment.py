from __future__ import annotations
import io, json
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from catboost import CatBoostClassifier
import full_funding_range_router_experiment as base

OUT=Path('ihp_nonfinancial_router_results'); OUT.mkdir(exist_ok=True)
URL='https://www.fema.gov/api/open/v2/RegistrationIntakeIndividualsHouseholdPrograms.csv'
COUNT_FIELDS=['totalValidRegistrations','validCallCenterRegistrations','validWebRegistrations','validMobileRegistrations','ihpReferrals','ihpEligible','haReferrals','haEligible','onaReferrals','onaEligible']


def fetch_ihp():
    r=requests.get(URL,timeout=180); r.raise_for_status()
    raw=pd.read_csv(io.BytesIO(r.content),low_memory=False)
    keep=['disasterNumber','county','city','zipCode']+[c for c in COUNT_FIELDS if c in raw.columns]
    raw=raw[keep].copy(); raw['disasterNumber']=pd.to_numeric(raw.disasterNumber,errors='coerce')
    for c in COUNT_FIELDS:
        if c in raw: raw[c]=pd.to_numeric(raw[c],errors='coerce').fillna(0)
    agg={c:'sum' for c in COUNT_FIELDS if c in raw}
    agg.update({'county':'nunique','city':'nunique','zipCode':'nunique'})
    f=raw.groupby('disasterNumber',as_index=False).agg(agg)
    f=f.rename(columns={'county':'ihpUniqueCountyCount','city':'ihpUniqueCityCount','zipCode':'ihpUniqueZipCount'})
    for c in [x for x in f.columns if x!='disasterNumber']:
        f['log_'+c]=np.log1p(pd.to_numeric(f[c],errors='coerce').fillna(0))
    f.to_csv(OUT/'ihp_nonfinancial_features.csv',index=False)
    return f


def operating(y,p):
    fpr,tpr,thr=roc_curve(y,p); tnr=1-fpr
    feasible=np.where((tpr>=.80)&(tnr>=.80))[0]; k=int(np.argmax(np.minimum(tpr,tnr)))
    j=int(feasible[np.argmax((tpr[feasible]+tnr[feasible])/2)]) if len(feasible) else k
    return {'auc':float(roc_auc_score(y,p)),'average_precision':float(average_precision_score(y,p)),'positive_recall':float(tpr[j]),'negative_recall':float(tnr[j]),'threshold':float(thr[j]),'has_80_80_point':bool(len(feasible)>0),'best_min_side_recall':float(min(tpr[k],tnr[k]))}


def main():
    d=base.load_data(); ihp=fetch_ihp(); d=d.merge(ihp,on='disasterNumber',how='left')
    ihpcols=[c for c in ihp.columns if c!='disasterNumber']
    d['ihpCoverage']=d[ihpcols].notna().any(axis=1).astype(int); d[ihpcols]=d[ihpcols].fillna(0)
    feats=list(dict.fromkeys(base.LOW_FEATURES+base.ORDINAL_FEATURES+ihpcols+['ihpCoverage','fyDeclared']))
    results=[]
    for lo,hi in [(1,2),(2,3),(3,4),(4,5)]:
        mask=d.band6_int.isin([lo,hi]); prob=np.full(len(d),np.nan)
        for year in sorted(d.fyDeclared.astype(int).unique()):
            tr=(d.fyDeclared.astype(int)!=year)&mask; te=(d.fyDeclared.astype(int)==year)&mask
            if te.sum()==0: continue
            X=d.loc[tr,feats].copy(); Xe=d.loc[te,feats].copy(); cats=[]
            for c in feats:
                if not pd.api.types.is_numeric_dtype(X[c]):
                    cats.append(c); X[c]=X[c].astype(str).fillna('MISSING'); Xe[c]=Xe[c].astype(str).fillna('MISSING')
                else:
                    X[c]=pd.to_numeric(X[c],errors='coerce').fillna(0); Xe[c]=pd.to_numeric(Xe[c],errors='coerce').fillna(0)
            y=(d.loc[tr,'band6_int']==hi).astype(int).to_numpy(); n0=int((y==0).sum()); n1=int((y==1).sum())
            m=CatBoostClassifier(iterations=140,depth=3,learning_rate=.03,l2_leaf_reg=20,loss_function='Logloss',verbose=False,random_seed=42,allow_writing_files=False,class_weights=[1,n0/max(n1,1)])
            m.fit(X,y,cat_features=cats); prob[te]=m.predict_proba(Xe)[:,1]
        yy=(d.loc[mask,'band6_int']==hi).astype(int).to_numpy(); met=operating(yy,prob[mask]); met.update({'lower_band':base.BANDS6[lo],'upper_band':base.BANDS6[hi],'n_lower':int((yy==0).sum()),'n_upper':int((yy==1).sum())}); results.append(met)
    pd.DataFrame(results).to_csv(OUT/'pairwise_results.csv',index=False)
    summary={'source':URL,'features_used':[c for c in ihpcols if not any(x in c.lower() for x in ['amount','dollar','obligat'])]+['ihpCoverage'],'excluded_financial_fields':['ihpAmount','haAmount','onaAmount'],'pairwise_results':results,'notes':['IHP dollar/amount fields are explicitly excluded.','IHP counts are external target-free impact descriptors observed for the disaster.','Every classifier removes the entire held-out fiscal year from model fitting.','This is a development separability screen; thresholds are selected on combined outer OOF predictions and must be nested before final validation.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
