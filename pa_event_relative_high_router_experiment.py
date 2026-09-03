from __future__ import annotations

import json
import time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MASTER = Path('master_openfema_40plus.xlsx')
SCALE = Path('frozen_declaration_scale_features.csv')
OUT = Path('pa_event_relative_high_router_results')
OUT.mkdir(exist_ok=True)
PA_URL = 'https://www.fema.gov/api/open/v1/PublicAssistanceFundedProjectsSummaries'

POP = {'AL':4779736,'AK':710231,'AZ':6392017,'AR':2915918,'CA':37253956,'CO':5029196,'CT':3574097,'DE':897934,'DC':601723,'FL':18801310,'GA':9687653,'HI':1360301,'ID':1567582,'IL':12830632,'IN':6483802,'IA':3046355,'KS':2853118,'KY':4339367,'LA':4533372,'ME':1328361,'MD':5773552,'MA':6547629,'MI':9883640,'MN':5303925,'MS':2967297,'MO':5988927,'MT':989415,'NE':1826341,'NV':2700551,'NH':1316470,'NJ':8791894,'NM':2059179,'NY':19378102,'NC':9535483,'ND':672591,'OH':11536504,'OK':3751351,'OR':3831074,'PA':12702379,'RI':1052567,'SC':4625364,'SD':814180,'TN':6346105,'TX':25145561,'UT':2763885,'VT':625741,'VA':8001024,'WA':6724540,'WV':1852994,'WI':5686986,'WY':563626,'PR':3725789,'AS':55519,'GU':159358,'MP':53883,'VI':106405}


def extract_rows(payload):
    if isinstance(payload,list): return payload
    if not isinstance(payload,dict): return []
    for k in ['PublicAssistanceFundedProjectsSummaries','items','data']:
        if isinstance(payload.get(k),list): return payload[k]
    vals=[v for v in payload.values() if isinstance(v,list)]
    return vals[0] if len(vals)==1 else []


def fetch_pa(nums, chunk_size=12):
    rows=[]; s=requests.Session()
    # IMPORTANT: federalObligatedAmount is intentionally not selected.
    select='disasterNumber,applicantName,county,educationApplicant,numberOfProjects,state'
    for start in range(0,len(nums),chunk_size):
        chunk=nums[start:start+chunk_size]
        filt=' or '.join(f'disasterNumber eq {int(x)}' for x in chunk)
        skip=0
        while True:
            params={'$filter':filt,'$select':select,'$top':1000,'$skip':skip,'$metadata':'off'}
            r=s.get(PA_URL,params=params,timeout=180,headers={'User-Agent':'ML-thesis-research/1.0'})
            r.raise_for_status(); batch=extract_rows(r.json()); rows.extend(batch)
            if len(batch)<1000: break
            skip+=1000
        print(f'PA event-peer fetch {min(start+chunk_size,len(nums))}/{len(nums)} rows={len(rows):,}',flush=True)
        time.sleep(.02)
    raw=pd.DataFrame(rows)
    raw.to_csv(OUT/'pa_event_peer_rows_nonfinancial.csv',index=False)
    return raw


def weighted_stats(labels, weights):
    c=Counter()
    for lab,w in zip(labels,weights):
        if pd.notna(lab) and float(w)>0: c[str(lab)]+=float(w)
    if not c: return 0.,0.,0.
    x=np.asarray(list(c.values()),float); p=x/x.sum()
    ent=float(-(p[p>0]*np.log(p[p>0])).sum()); hhi=float((p*p).sum()); top=float(p.max())
    return ent,hhi,top


def aggregate_pa(raw, nums):
    base=pd.DataFrame({'disasterNumber':nums})
    if raw.empty: return base
    r=raw.copy(); r['disasterNumber']=pd.to_numeric(r.disasterNumber,errors='coerce'); r=r[r.disasterNumber.notna()].copy(); r['disasterNumber']=r.disasterNumber.astype(int)
    r['numberOfProjects']=pd.to_numeric(r.numberOfProjects,errors='coerce').fillna(0).clip(lower=0)
    rows=[]
    for dn,g in r.groupby('disasterNumber'):
        projects=g.numberOfProjects.to_numpy(float)
        ae,ah,at=weighted_stats(g.applicantName,projects); ce,ch,ct=weighted_stats(g.county,projects)
        edu=g.educationApplicant.astype(str).str.lower().isin(['true','1','yes','y']).to_numpy()
        total=float(projects.sum())
        rows.append({'disasterNumber':int(dn),'paRowCount':int(len(g)),'paTotalProjectCount':total,
                     'paUniqueApplicantCount':int(g.applicantName.nunique(dropna=True)),'paUniqueCountyCount':int(g.county.nunique(dropna=True)),
                     'paProjectsPerApplicant':total/max(g.applicantName.nunique(dropna=True),1),'paProjectsPerCounty':total/max(g.county.nunique(dropna=True),1),
                     'paApplicantEntropy':ae,'paApplicantHHI':ah,'paApplicantTopShare':at,
                     'paCountyEntropy':ce,'paCountyHHI':ch,'paCountyTopShare':ct,
                     'paEducationProjectShare':float(projects[edu].sum()/total) if total>0 else 0.0})
    out=base.merge(pd.DataFrame(rows),on='disasterNumber',how='left').fillna(0)
    for c in [x for x in out.columns if x!='disasterNumber']:
        out['log_'+c]=np.log1p(pd.to_numeric(out[c],errors='coerce').fillna(0).clip(lower=0))
    out.to_csv(OUT/'pa_event_peer_structure_features.csv',index=False)
    return out


def op_point(y,p):
    fpr,tpr,thr=roc_curve(y,p); tnr=1-fpr
    feasible=np.where((tpr>=.80)&(tnr>=.80))[0]
    cand=feasible if len(feasible) else np.arange(len(thr))
    mins=np.minimum(tpr[cand],tnr[cand]); bestmin=mins.max(); cand2=cand[mins==bestmin]
    j=int(cand2[np.argmax((tpr[cand2]+tnr[cand2])/2)])
    return {'threshold':float(thr[j]),'lower_recall':float(tnr[j]),'upper_recall':float(tpr[j]),
            'min_recall':float(min(tnr[j],tpr[j])),'auc':float(roc_auc_score(y,p)),'pass80':bool(len(feasible))}


def prepare_data():
    m=pd.read_excel(MASTER); m['target']=pd.to_numeric(m.totalObligatedFunding,errors='coerce').fillna(0).clip(lower=0)
    m['band']=np.select([(m.target>50e6)&(m.target<=200e6),(m.target>200e6)&(m.target<=500e6),m.target>500e6],[3,4,5],default=-1).astype(int)
    m['event_key']=m.incidentType.astype(str)+'|'+m.incidentBeginDate.astype(str)+'|'+m.incidentEndDate.astype(str)
    high_events=set(m.loc[m.band>=3,'event_key']); peers=m[m.event_key.isin(high_events)].copy()
    nums=sorted(peers.disasterNumber.astype(int).unique().tolist()); print(f'High events={len(high_events)}, event-peer disasters={len(nums)}',flush=True)
    pa=aggregate_pa(fetch_pa(nums),nums)
    peers=peers.merge(pa,on='disasterNumber',how='left')
    pa_raw=[c for c in pa.columns if c!='disasterNumber']
    for c in pa_raw: peers[c]=pd.to_numeric(peers[c],errors='coerce').fillna(0)
    g=peers.groupby('event_key'); peers['eventPeerCount']=g.disasterNumber.transform('size')
    event_cols=['paTotalProjectCount','paUniqueApplicantCount','paUniqueCountyCount','paRowCount','paEducationProjectShare']
    event_feats=[]
    for c in event_cols:
        total=g[c].transform('sum').replace(0,np.nan)
        peers[c+'_event_share']=(peers[c]/total).replace([np.inf,-np.inf],np.nan).fillna(0)
        peers[c+'_relative_event_avg']=peers[c+'_event_share']*peers.eventPeerCount
        peers[c+'_event_rank']=g[c].rank(pct=True,method='average').fillna(.5)
        event_feats += [c+'_event_share',c+'_relative_event_avg',c+'_event_rank']
    event_table=peers[['disasterNumber','event_key','eventPeerCount']+pa_raw+event_feats]
    event_table.to_csv(OUT/'pa_event_relative_features_all_peers.csv',index=False)

    d=m[m.band>=3].copy().merge(event_table.drop(columns=['event_key']),on='disasterNumber',how='left')
    sc=pd.read_csv(SCALE); d=d.merge(sc[['disasterNumber','declaredAreaCount','uniquePlaceCodeCount','declarationGeographyRowCount']],on='disasterNumber',how='left')
    d['population2010']=d.state.map(POP).astype(float)
    full=m.copy(); full['population2010']=full.state.map(POP).astype(float); full=full.merge(sc[['disasterNumber','declaredAreaCount']],on='disasterNumber',how='left')
    gf=full.groupby('event_key'); full['eventSize']=gf.disasterNumber.transform('size')
    for c in ['missionAssignmentCount','responseComplexityScore','uniqueAgencyCount','population2010']:
        total=gf[c].transform('sum').replace(0,np.nan); full[c+'_relative_event_avg']=(full[c]/total*full.eventSize).replace([np.inf,-np.inf],np.nan).fillna(0)
    rel=full[['disasterNumber','eventSize']+[c+'_relative_event_avg' for c in ['missionAssignmentCount','responseComplexityScore','uniqueAgencyCount','population2010']]]
    d=d.merge(rel,on='disasterNumber',how='left')
    for new,old in [('logPopulation2010','population2010'),('logMission','missionAssignmentCount'),('logComplexity','responseComplexityScore'),('logDeclaredArea','declaredAreaCount'),('logDuration','durationDays')]:
        d[new]=np.log1p(pd.to_numeric(d[old],errors='coerce').fillna(0).clip(lower=0))
    exposure=['logPopulation2010','logMission','logComplexity','logDeclaredArea','logDuration','expectedResourceScore','missionDensity','agencyDensity','declarationDelayDays','uniqueAgencyCount','uniqueMaTypeCount','uniquePriorityCount','eventSize']+[c+'_relative_event_avg' for c in ['missionAssignmentCount','responseComplexityScore','uniqueAgencyCount','population2010']]
    for c in exposure+pa_raw+event_feats: d[c]=pd.to_numeric(d[c],errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(0)
    return d.reset_index(drop=True), exposure, pa_raw, event_feats


def make_model(kind):
    if kind=='et': return ExtraTreesClassifier(n_estimators=350,max_depth=4,min_samples_leaf=1,max_features=.75,class_weight='balanced',random_state=42,n_jobs=-1)
    if kind=='rf': return RandomForestClassifier(n_estimators=350,max_depth=4,min_samples_leaf=1,max_features=.75,class_weight='balanced',random_state=42,n_jobs=-1)
    return Pipeline([('s',StandardScaler()),('m',KNeighborsClassifier(n_neighbors=5,weights='distance'))])


def pair_oof(d,features,lo,hi,kind):
    mask=d.band.isin([lo,hi]).to_numpy(); p=np.full(len(d),np.nan)
    years=d.fyDeclared.astype(int).to_numpy(); X=d[features]
    for yr in sorted(np.unique(years[mask])):
        tr=mask&(years!=yr); te=mask&(years==yr); y=(d.loc[tr,'band']==hi).astype(int)
        model=make_model(kind); model.fit(X.loc[tr],y); p[te]=model.predict_proba(X.loc[te])[:,list(model.classes_).index(1)]
    yy=(d.loc[mask,'band']==hi).astype(int).to_numpy(); pp=p[mask]
    return op_point(yy,pp),p


def multi_oof(d,features,kind):
    years=d.fyDeclared.astype(int).to_numpy(); X=d[features]; pred=np.zeros(len(d),int)
    for yr in sorted(np.unique(years)):
        tr=years!=yr; te=~tr; model=make_model(kind); model.fit(X.loc[tr],d.loc[tr,'band']); pred[te]=model.predict(X.loc[te])
    rec={str(b):float((pred[d.band.to_numpy()==b]==b).mean()) for b in [3,4,5]}
    return pred,rec


def main():
    d,exposure,pa_raw,event_feats=prepare_data()
    sets={'exposure':exposure,'exposure_raw_pa':exposure+pa_raw,'exposure_event_pa':exposure+pa_raw+event_feats}
    results=[]; preds=[]
    for sn,features in sets.items():
        features=list(dict.fromkeys(features))
        for kind in ['et','rf','knn']:
            pred,rec=multi_oof(d,features,kind); results.append({'feature_set':sn,'model':'multi_'+kind,'r3':rec['3'],'r4':rec['4'],'r5':rec['5'],'min_recall':min(rec.values())})
        for lo,hi in [(3,4),(4,5),(3,5)]:
            for kind in ['et','rf','knn']:
                met,p=pair_oof(d,features,lo,hi,kind); results.append({'feature_set':sn,'model':f'pair_{lo}_{hi}_{kind}',**met})
                mm=d.band.isin([lo,hi]); q=d.loc[mm,['disasterNumber','state','fyDeclared','incidentType','target','band']].copy(); q['feature_set']=sn; q['model']=f'pair_{lo}_{hi}_{kind}'; q['prob_upper']=p[mm]; preds.append(q)
    pd.DataFrame(results).to_csv(OUT/'screen_results.csv',index=False)
    pd.concat(preds,ignore_index=True).to_csv(OUT/'pairwise_oof_predictions.csv',index=False)
    summary={'band_counts':{str(b):int((d.band==b).sum()) for b in [3,4,5]},'results':results,
             'excluded_fields':['federalObligatedAmount','declarationDate','hash','id'],
             'notes':['PA event-relative denominators use all master declarations sharing the same target-free incidentType+begin/end event key, not only high-value rows.','No dollar amount or obligation date is used as a predictor.','Predictions are strict leave-fiscal-year-out.','Thresholds are selected on combined OOF only for development screening; any winner must later be nested.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
