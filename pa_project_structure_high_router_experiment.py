from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MASTER = Path('master_openfema_40plus.xlsx')
SCALE = Path('frozen_declaration_scale_features.csv')
MISSION_COMP = Path('frozen_specialist_mission_features.csv')
OUT = Path('pa_project_structure_high_router_results')
OUT.mkdir(exist_ok=True)
PA_URL = 'https://www.fema.gov/api/open/v1/PublicAssistanceFundedProjectsDetails'

POP = {'AL':4779736,'AK':710231,'AZ':6392017,'AR':2915918,'CA':37253956,'CO':5029196,'CT':3574097,'DE':897934,'DC':601723,'FL':18801310,'GA':9687653,'HI':1360301,'ID':1567582,'IL':12830632,'IN':6483802,'IA':3046355,'KS':2853118,'KY':4339367,'LA':4533372,'ME':1328361,'MD':5773552,'MA':6547629,'MI':9883640,'MN':5303925,'MS':2967297,'MO':5988927,'MT':989415,'NE':1826341,'NV':2700551,'NH':1316470,'NJ':8791894,'NM':2059179,'NY':19378102,'NC':9535483,'ND':672591,'OH':11536504,'OK':3751351,'OR':3831074,'PA':12702379,'RI':1052567,'SC':4625364,'SD':814180,'TN':6346105,'TX':25145561,'UT':2763885,'VT':625741,'VA':8001024,'WA':6724540,'WV':1852994,'WI':5686986,'WY':563626,'PR':3725789,'AS':55519,'GU':159358,'MP':53883,'VI':106405}
BANDS = ['50M-200M','200M-500M','500M+']


def extract_list(payload):
    if isinstance(payload, list): return payload
    if not isinstance(payload, dict): raise RuntimeError(type(payload))
    for k in ['PublicAssistanceFundedProjectsDetails','PublicAssistanceFundedProjectDetails','items','data']:
        if isinstance(payload.get(k), list): return payload[k]
    lists=[v for v in payload.values() if isinstance(v,list)]
    if len(lists)==1: return lists[0]
    raise RuntimeError(f'Could not locate record list in keys {list(payload)}')


def fetch_pa(disaster_numbers, chunk_size=18):
    rows=[]; s=requests.Session()
    select='disasterNumber,pwNumber,applicantId,damageCategoryCode,damageCategory,county,state,stateCode'
    for start in range(0,len(disaster_numbers),chunk_size):
        chunk=disaster_numbers[start:start+chunk_size]
        filt=' or '.join(f'disasterNumber eq {int(x)}' for x in chunk)
        skip=0
        while True:
            params={'$filter':filt,'$select':select,'$top':1000,'$skip':skip,'$metadata':'off'}
            r=s.get(PA_URL,params=params,timeout=180,headers={'User-Agent':'ML-thesis-research/1.0'}); r.raise_for_status()
            batch=extract_list(r.json()); rows.extend(batch)
            if len(batch)<1000: break
            skip+=1000
        print(f'PA fetch {min(start+chunk_size,len(disaster_numbers))}/{len(disaster_numbers)} rows={len(rows):,}',flush=True)
        time.sleep(.03)
    raw=pd.DataFrame(rows); raw.to_csv(OUT/'pa_project_rows_nonfinancial.csv',index=False)
    return raw


def stats(counter):
    if not counter: return 0.,0.,0.
    x=np.array(list(counter.values()),float); p=x/x.sum()
    return float(-(p[p>0]*np.log(p[p>0])).sum()),float((p*p).sum()),float(p.max())


def aggregate_pa(raw, nums):
    by={int(n):[] for n in nums}
    if not raw.empty:
        raw['disasterNumber']=pd.to_numeric(raw.disasterNumber,errors='coerce')
        raw=raw[raw.disasterNumber.notna()].copy(); raw['disasterNumber']=raw.disasterNumber.astype(int)
        for _,r in raw.iterrows():
            if int(r.disasterNumber) in by: by[int(r.disasterNumber)].append(r)
    rows=[]
    for dn in nums:
        rr=by[int(dn)]; applicants=Counter(); counties=Counter(); cats=Counter()
        pws=set()
        for r in rr:
            if pd.notna(r.get('pwNumber')): pws.add(str(r.get('pwNumber')))
            if pd.notna(r.get('applicantId')): applicants[str(r.get('applicantId'))]+=1
            if pd.notna(r.get('county')): counties[str(r.get('county'))]+=1
            cat=r.get('damageCategoryCode') if pd.notna(r.get('damageCategoryCode')) else r.get('damageCategory')
            if pd.notna(cat): cats[str(cat).strip()]+=1
        ae,ah,at=stats(applicants); ce,ch,ct=stats(counties); de,dh,dt=stats(cats)
        row={'disasterNumber':int(dn),'paProjectRowCount':len(rr),'paUniqueProjectCount':len(pws),'paUniqueApplicantCount':len(applicants),'paUniqueCountyCount':len(counties),'paUniqueDamageCategoryCount':len(cats),'paApplicantEntropy':ae,'paApplicantHHI':ah,'paApplicantTopShare':at,'paCountyEntropy':ce,'paCountyHHI':ch,'paCountyTopShare':ct,'paCategoryEntropy':de,'paCategoryHHI':dh,'paCategoryTopShare':dt}
        total=sum(cats.values())
        for cat in ['A','B','C','D','E','F','G','Z']:
            row[f'paCategoryShare_{cat}']=cats.get(cat,0)/total if total else 0.
        rows.append(row)
    out=pd.DataFrame(rows); out.to_csv(OUT/'pa_project_structure_features.csv',index=False); return out


def load_data():
    m=pd.read_excel(MASTER); m['target']=pd.to_numeric(m.totalObligatedFunding,errors='coerce').fillna(0).clip(lower=0)
    m['band']=np.select([(m.target>50e6)&(m.target<=200e6),(m.target>200e6)&(m.target<=500e6),m.target>500e6],[3,4,5],default=-1).astype(int)
    d=m[m.band>=3].copy(); nums=d.disasterNumber.astype(int).tolist()
    pa=aggregate_pa(fetch_pa(nums),nums); d=d.merge(pa,on='disasterNumber',how='left')
    scale=pd.read_csv(SCALE); d=d.merge(scale[['disasterNumber','declaredAreaCount']],on='disasterNumber',how='left')
    comp=pd.read_csv(MISSION_COMP); d=d.merge(comp,on='disasterNumber',how='left')
    d['population2010']=d.state.map(POP).astype(float)
    d['event_key']=d.incidentType.astype(str)+'|'+d.incidentBeginDate.astype(str)+'|'+d.incidentEndDate.astype(str)
    for new,old in [('logPopulation2010','population2010'),('logMission','missionAssignmentCount'),('logComplexity','responseComplexityScore'),('logDeclaredArea','declaredAreaCount'),('logDuration','durationDays')]: d[new]=np.log1p(pd.to_numeric(d[old],errors='coerce').fillna(0))
    d['popPerArea']=d.population2010/d.declaredAreaCount.replace(0,np.nan); d['missionPerArea']=d.missionAssignmentCount/d.declaredAreaCount.replace(0,np.nan); d['complexityPerArea']=d.responseComplexityScore/d.declaredAreaCount.replace(0,np.nan)
    for new,old in [('logPopPerArea','popPerArea'),('logMissionPerArea','missionPerArea'),('logComplexityPerArea','complexityPerArea')]: d[new]=np.log1p(pd.to_numeric(d[old],errors='coerce').fillna(0))
    g=d.groupby('event_key'); d['eventSizeHigh']=g.disasterNumber.transform('size')
    for col in ['missionAssignmentCount','responseComplexityScore','uniqueAgencyCount','population2010']:
        total=g[col].transform('sum').replace(0,np.nan); d[col+'_relative_high_event_avg']=d[col]/total*d.eventSizeHigh
    pa_cols=[c for c in pa if c!='disasterNumber']; comp_cols=[c for c in comp if c!='disasterNumber']
    exposure=['logPopulation2010','logMission','logComplexity','logDeclaredArea','logPopPerArea','logMissionPerArea','logComplexityPerArea','missionAssignmentCount_relative_high_event_avg','responseComplexityScore_relative_high_event_avg','uniqueAgencyCount_relative_high_event_avg','population2010_relative_high_event_avg','logDuration','expectedResourceScore','missionDensity','agencyDensity','fyDeclared']
    for c in pa_cols+comp_cols+exposure: d[c]=pd.to_numeric(d[c],errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(0)
    return d.reset_index(drop=True), exposure, pa_cols, comp_cols


def recalls3(y,p): return [float((p[y==b]==b).mean()) for b in [3,4,5]]

def threshold_pair(y,p):
    fpr,tpr,thr=roc_curve(y,p); tnr=1-fpr; j=int(np.argmax(np.minimum(tpr,tnr)))
    return float(thr[j]),float(tnr[j]),float(tpr[j])


def main():
    d,exposure,pa_cols,comp_cols=load_data(); sets={'pa_only':pa_cols,'exposure_plus_pa':exposure+pa_cols,'exposure_pa_comp':exposure+pa_cols+comp_cols}
    result=[]; predrows=[]
    for setname,feats in sets.items():
      X=d[feats]
      for modelname in ['et','rf','knn']:
        pred=np.zeros(len(d),int); prob=np.zeros((len(d),3))
        for yr in sorted(d.fyDeclared.astype(int).unique()):
          tr=d.fyDeclared.astype(int)!=yr; te=~tr
          if modelname=='et': model=ExtraTreesClassifier(n_estimators=400,max_depth=5,min_samples_leaf=2,max_features=.7,class_weight='balanced',random_state=42,n_jobs=-1)
          elif modelname=='rf': model=RandomForestClassifier(n_estimators=400,max_depth=5,min_samples_leaf=2,max_features=.7,class_weight='balanced',random_state=42,n_jobs=-1)
          else: model=Pipeline([('sc',StandardScaler()),('m',KNeighborsClassifier(n_neighbors=5,weights='distance'))])
          model.fit(X.loc[tr],d.loc[tr,'band']); pred[te]=model.predict(X.loc[te]); prob[te]=model.predict_proba(X.loc[te])
        rr=recalls3(d.band.to_numpy(),pred); result.append({'feature_set':setname,'model':modelname,'r_50_200':rr[0],'r_200_500':rr[1],'r_500_plus':rr[2],'min_recall':min(rr),'accuracy':float((pred==d.band.to_numpy()).mean())})
        tmp=d[['disasterNumber','state','fyDeclared','incidentType','target','band']].copy(); tmp['feature_set']=setname; tmp['model']=modelname; tmp['prediction']=pred; predrows.append(tmp)
      # pairwise screen using ExtraTrees, for boundary separability only
      for lo,hi in [(3,4),(4,5)]:
        mask=d.band.isin([lo,hi]); pr=np.full(len(d),np.nan)
        for yr in sorted(d.loc[mask,'fyDeclared'].astype(int).unique()):
          tr=mask&(d.fyDeclared.astype(int)!=yr); te=mask&(d.fyDeclared.astype(int)==yr); y=(d.loc[tr,'band']==hi).astype(int)
          model=ExtraTreesClassifier(n_estimators=400,max_depth=4,min_samples_leaf=1,max_features=.8,class_weight='balanced',random_state=42,n_jobs=-1); model.fit(X.loc[tr],y); pr[te]=model.predict_proba(X.loc[te])[:,1]
        yy=(d.loc[mask,'band']==hi).astype(int).to_numpy(); pp=pr[mask]; th,lr,ur=threshold_pair(yy,pp); result.append({'feature_set':setname,'model':f'pair_{lo}_{hi}_et','r_50_200':lr if lo==3 else np.nan,'r_200_500':ur if hi==4 else lr,'r_500_plus':ur if hi==5 else np.nan,'min_recall':min(lr,ur),'accuracy':np.nan,'threshold':th,'auc':float(roc_auc_score(yy,pp))})
    pd.DataFrame(result).to_csv(OUT/'screen_results.csv',index=False); pd.concat(predrows,ignore_index=True).to_csv(OUT/'multiclass_predictions.csv',index=False)
    summary={'rows_high_value':int(len(d)),'band_counts':{BANDS[i-3]:int((d.band==i).sum()) for i in [3,4,5]},'results':result,'excluded_fields':['projectAmount','federalShareObligated','totalObligated','obligatedDate','projectSize','applicationTitle'],'notes':['PA project structure is retrospective and comes from a funded-project dataset, but no financial amount, obligation date, or amount-derived project-size field is used.','Every model prediction excludes the entire held-out fiscal year.','This is a development feature screen; any selected router must be nested/frozen before final thesis reporting.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
