from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from catboost import CatBoostClassifier
import full_funding_range_router_experiment as base

OUT=Path('global_router_pairwise_fullfeature_results'); OUT.mkdir(exist_ok=True)
BANDS=base.BANDS6
FEATURES=list(dict.fromkeys(base.LOW_FEATURES + base.ORDINAL_FEATURES))
CONFIGS=[
    {'iterations':100,'depth':2,'learning_rate':.04,'l2_leaf_reg':15},
    {'iterations':140,'depth':3,'learning_rate':.03,'l2_leaf_reg':20},
    {'iterations':180,'depth':2,'learning_rate':.025,'l2_leaf_reg':30},
]

def prep(df):
    X=df[FEATURES].copy(); cats=[]
    for c in FEATURES:
        if not pd.api.types.is_numeric_dtype(X[c]):
            cats.append(c); X[c]=X[c].astype(str).fillna('MISSING')
        else:
            X[c]=pd.to_numeric(X[c],errors='coerce')
    return X,cats

def best_operating_point(y,p):
    fpr,tpr,thr=roc_curve(y,p); tnr=1-fpr
    k=int(np.argmax(np.minimum(tpr,tnr)))
    feasible=np.where((tpr>=.80)&(tnr>=.80))[0]
    if len(feasible):
        # among points clearing 80/80 choose max mean recall, then higher minimum
        j=feasible[np.argmax((tpr[feasible]+tnr[feasible])/2)]
    else: j=k
    return {'threshold':float(thr[j]),'positive_recall':float(tpr[j]),'negative_recall':float(tnr[j]),'min_side_recall':float(min(tpr[j],tnr[j])),'has_80_80_point':bool(len(feasible)>0),'best_min_threshold':float(thr[k]),'best_min_side_recall':float(min(tpr[k],tnr[k]))}

def oof_pair(d,lo,hi,cfg):
    mask=d.band6_int.isin([lo,hi]).to_numpy(); prob=np.full(len(d),np.nan)
    years=sorted(d.fyDeclared.astype(int).unique())
    for year in years:
        tr=(d.fyDeclared.astype(int).to_numpy()!=year)&mask
        te=(d.fyDeclared.astype(int).to_numpy()==year)&mask
        if te.sum()==0: continue
        y=(d.loc[tr,'band6_int'].to_numpy()==hi).astype(int)
        if len(np.unique(y))<2:
            prob[te]=float(y[0]) if len(y) else .5; continue
        X,cats=prep(d.loc[tr]); Xe,_=prep(d.loc[te])
        n0=int((y==0).sum()); n1=int((y==1).sum())
        m=CatBoostClassifier(**cfg,loss_function='Logloss',verbose=False,random_seed=42,allow_writing_files=False,class_weights=[1,n0/max(n1,1)])
        m.fit(X,y,cat_features=cats); prob[te]=m.predict_proba(Xe)[:,1]
    yy=(d.loc[mask,'band6_int'].to_numpy()==hi).astype(int); pp=prob[mask]; ok=np.isfinite(pp); yy=yy[ok]; pp=pp[ok]
    op=best_operating_point(yy,pp)
    return {'auc':float(roc_auc_score(yy,pp)),'average_precision':float(average_precision_score(yy,pp)),'n_lower':int((yy==0).sum()),'n_upper':int((yy==1).sum()),**op},prob

def oof_one_vs_rest(d,c,cfg):
    prob=np.zeros(len(d)); years=sorted(d.fyDeclared.astype(int).unique())
    for year in years:
        tr=d.fyDeclared.astype(int).to_numpy()!=year; te=~tr; y=(d.loc[tr,'band6_int'].to_numpy()==c).astype(int)
        X,cats=prep(d.loc[tr]); Xe,_=prep(d.loc[te]); n0=int((y==0).sum()); n1=int((y==1).sum())
        m=CatBoostClassifier(**cfg,loss_function='Logloss',verbose=False,random_seed=42,allow_writing_files=False,class_weights=[1,n0/max(n1,1)])
        m.fit(X,y,cat_features=cats); prob[te]=m.predict_proba(Xe)[:,1]
    y=(d.band6_int.to_numpy()==c).astype(int); op=best_operating_point(y,prob)
    return {'auc':float(roc_auc_score(y,prob)),'average_precision':float(average_precision_score(y,prob)),'n_positive':int(y.sum()),**op},prob

def main():
    d=base.load_data(); pair_rows=[]; ovr_rows=[]
    best_pair_probs={}
    for g in range(5):
        best=None
        for ci,cfg in enumerate(CONFIGS):
            met,p=oof_pair(d,g,g+1,cfg); row={'lower_band':BANDS[g],'upper_band':BANDS[g+1],'config_index':ci,**cfg,**met}; pair_rows.append(row)
            key=(met['has_80_80_point'],met['min_side_recall'],met['auc'])
            if best is None or key>best[0]: best=(key,row,p)
        best_pair_probs[str(g)]=best[2].tolist()
    for c in range(6):
        best=None
        for ci,cfg in enumerate(CONFIGS):
            met,p=oof_one_vs_rest(d,c,cfg); row={'band':BANDS[c],'config_index':ci,**cfg,**met}; ovr_rows.append(row)
            key=(met['has_80_80_point'],met['min_side_recall'],met['auc'])
            if best is None or key>best[0]: best=(key,row,p)
    pair_df=pd.DataFrame(pair_rows); ovr_df=pd.DataFrame(ovr_rows); pair_df.to_csv(OUT/'pairwise_candidates.csv',index=False); ovr_df.to_csv(OUT/'ovr_candidates.csv',index=False)
    selected_pair=[]
    for g in range(5):
        sub=pair_df[pair_df.lower_band==BANDS[g]].copy(); sub['rank80']=sub.has_80_80_point.astype(int); r=sub.sort_values(['rank80','min_side_recall','auc'],ascending=False).iloc[0].to_dict(); selected_pair.append(r)
    selected_ovr=[]
    for c in range(6):
        sub=ovr_df[ovr_df.band==BANDS[c]].copy(); sub['rank80']=sub.has_80_80_point.astype(int); selected_ovr.append(sub.sort_values(['rank80','min_side_recall','auc'],ascending=False).iloc[0].to_dict())
    summary={'acceptance_target':'Each final funding band recall >=0.80; adjacent pair screen is a necessary separability diagnostic, not sufficient final validation.','features':FEATURES,'selected_adjacent_pair_results':selected_pair,'selected_one_vs_rest_results':selected_ovr,'all_adjacent_pairs_have_80_80':bool(all(bool(r['has_80_80_point']) for r in selected_pair)),'notes':['Every prediction excludes the entire held-out fiscal year.','Pairwise training uses only the two adjacent funding bands, so results measure local boundary separability directly.','Thresholds in this screen are selected on the combined outer OOF predictions and are development diagnostics only. If separability is sufficient, final thresholds must be selected inside each outer training fold.','No test funding value is used as an input feature.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
