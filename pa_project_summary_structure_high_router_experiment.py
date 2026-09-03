from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MASTER=Path('master_openfema_40plus.xlsx')
SCALE=Path('frozen_declaration_scale_features.csv')
OUT=Path('pa_project_summary_structure_high_router_results'); OUT.mkdir(exist_ok=True)
PA_URL='https://www.fema.gov/api/open/v1/PublicAssistanceFundedProjectsSummaries'
POP={'AL':4779736,'AK':710231,'AZ':6392017,'AR':2915918,'CA':37253956,'CO':5029196,'CT':3574097,'DE':897934,'DC':601723,'FL':18801310,'GA':9687653,'HI':1360301,'ID':1567582,'IL':12830632,'IN':6483802,'IA':3046355,'KS':2853118,'KY':4339367,'LA':4533372,'ME':1328361,'MD':5773552,'MA':6547629,'MI':9883640,'MN':5303925,'MS':2967297,'MO':5988927,'MT':989415,'NE':1826341,'NV':2700551,'NH':1316470,'NJ':8791894,'NM':2059179,'NY':19378102,'NC':9535483,'ND':672591,'OH':11536504,'OK':3751351,'OR':3831074,'PA':12702379,'RI':1052567,'SC':4625364,'SD':814180,'TN':6346105,'TX':25145561,'UT':2763885,'VT':625741,'VA':8001024,'WA':6724540,'WV':1852994,'WI':5686986,'WY':563626,'PR':3725789,'AS':55519,'GU':159358,'MP':53883,'VI':106405}


def extract(payload):
    if isinstance(payload,list): return payload
    for v in payload.values():
        if isinstance(v,list): return v
    return []


def fetch(nums,chunk_size=12):
    s=requests.Session(); rows=[]
    fields='disasterNumber,applicantName,county,educationApplicant,numberOfProjects,state'
    for start in range(0,len(nums),chunk_size):
        chunk=nums[start:start+chunk_size]; filt=' or '.join(f'disasterNumber eq {int(x)}' for x in chunk); skip=0
        while True:
            p={'$filter':filt,'$select':fields,'$top':1000,'$skip':skip,'$metadata':'off'}
            r=s.get(PA_URL,params=p,timeout=180,headers={'User-Agent':'ML-thesis-research/1.0'}); r.raise_for_status(); batch=extract(r.json()); rows.extend(batch)
            if len(batch)<1000: break
            skip+=1000
        print(f'PA summaries {min(start+chunk_size,len(nums))}/{len(nums)} rows={len(rows):,}',flush=True)
    raw=pd.DataFrame(rows); raw.to_csv(OUT/'pa_summary_rows_nonfinancial.csv',index=False); return raw


def concentration(weights):
    a=np.asarray(list(weights),float)
    if len(a)==0 or a.sum()<=0: return 0.,0.,0.
    p=a/a.sum(); return float(-(p[p>0]*np.log(p[p>0])).sum()),float((p*p).sum()),float(p.max())


def aggregate(raw,nums):
    rows=[]
    if raw.empty: raw=pd.DataFrame(columns=['disasterNumber','applicantName','county','educationApplicant','numberOfProjects'])
    raw['disasterNumber']=pd.to_numeric(raw.disasterNumber,errors='coerce'); raw=raw[raw.disasterNumber.notna()].copy(); raw['disasterNumber']=raw.disasterNumber.astype(int)
    raw['numberOfProjects']=pd.to_numeric(raw.numberOfProjects,errors='coerce').fillna(0).clip(lower=0)
    for dn in nums:
        g=raw[raw.disasterNumber==int(dn)].copy(); total=float(g.numberOfProjects.sum())
        app=g.groupby(g.applicantName.fillna('Unknown').astype(str)).numberOfProjects.sum() if len(g) else pd.Series(dtype=float)
        cnty=g.groupby(g.county.fillna('Unknown').astype(str)).numberOfProjects.sum() if len(g) else pd.Series(dtype=float)
        ae,ah,at=concentration(app.values); ce,ch,ct=concentration(cnty.values)
        edu=g.educationApplicant.fillna(False).astype(str).str.lower().isin(['true','1','yes','y']) if len(g) else pd.Series(dtype=bool)
        edu_projects=float(g.loc[edu,'numberOfProjects'].sum()) if len(g) else 0.
        rows.append({'disasterNumber':int(dn),'paSummaryRowCount':int(len(g)),'paTotalProjectCount':total,'paUniqueApplicantCount':int(g.applicantName.nunique()) if len(g) else 0,'paUniqueCountyCount':int(g.county.nunique()) if len(g) else 0,'paProjectsPerApplicant':total/max(int(g.applicantName.nunique()),1) if len(g) else 0.,'paProjectsPerCounty':total/max(int(g.county.nunique()),1) if len(g) else 0.,'paApplicantEntropy':ae,'paApplicantHHI':ah,'paApplicantTopShare':at,'paCountyEntropy':ce,'paCountyHHI':ch,'paCountyTopShare':ct,'paEducationApplicantRowShare':float(edu.mean()) if len(g) else 0.,'paEducationProjectShare':edu_projects/total if total>0 else 0.})
    out=pd.DataFrame(rows)
    for c in ['paTotalProjectCount','paUniqueApplicantCount','paUniqueCountyCount','paProjectsPerApplicant','paProjectsPerCounty']:
        out['log_'+c]=np.log1p(pd.to_numeric(out[c],errors='coerce').fillna(0))
    out.to_csv(OUT/'pa_summary_structure_features.csv',index=False); return out


def load():
    m=pd.read_excel(MASTER); m['target']=pd.to_numeric(m.totalObligatedFunding,errors='coerce').fillna(0).clip(lower=0)
    m['band']=np.select([(m.target>50e6)&(m.target<=200e6),(m.target>200e6)&(m.target<=500e6),m.target>500e6],[3,4,5],default=-1).astype(int); d=m[m.band>=3].copy(); nums=d.disasterNumber.astype(int).tolist()
    pa=aggregate(fetch(nums),nums); d=d.merge(pa,on='disasterNumber',how='left'); sc=pd.read_csv(SCALE); d=d.merge(sc[['disasterNumber','declaredAreaCount']],on='disasterNumber',how='left')
    d['population2010']=d.state.map(POP).astype(float); d['event_key']=d.incidentType.astype(str)+'|'+d.incidentBeginDate.astype(str)+'|'+d.incidentEndDate.astype(str)
    for new,old in [('logPopulation2010','population2010'),('logMission','missionAssignmentCount'),('logComplexity','responseComplexityScore'),('logDeclaredArea','declaredAreaCount'),('logDuration','durationDays')]: d[new]=np.log1p(pd.to_numeric(d[old],errors='coerce').fillna(0))
    d['popPerArea']=d.population2010/d.declaredAreaCount.replace(0,np.nan); d['missionPerArea']=d.missionAssignmentCount/d.declaredAreaCount.replace(0,np.nan); d['complexityPerArea']=d.responseComplexityScore/d.declaredAreaCount.replace(0,np.nan)
    for new,old in [('logPopPerArea','popPerArea'),('logMissionPerArea','missionPerArea'),('logComplexityPerArea','complexityPerArea')]: d[new]=np.log1p(pd.to_numeric(d[old],errors='coerce').fillna(0))
    gg=d.groupby('event_key'); d['eventSizeHigh']=gg.disasterNumber.transform('size')
    for col in ['missionAssignmentCount','responseComplexityScore','uniqueAgencyCount','population2010']:
        tot=gg[col].transform('sum').replace(0,np.nan); d[col+'_relative_high_event_avg']=d[col]/tot*d.eventSizeHigh
    exposure=['logPopulation2010','logMission','logComplexity','logDeclaredArea','logPopPerArea','logMissionPerArea','logComplexityPerArea','missionAssignmentCount_relative_high_event_avg','responseComplexityScore_relative_high_event_avg','uniqueAgencyCount_relative_high_event_avg','population2010_relative_high_event_avg','logDuration','expectedResourceScore','missionDensity','agencyDensity','fyDeclared']
    pa_cols=[c for c in pa.columns if c!='disasterNumber']
    for c in exposure+pa_cols: d[c]=pd.to_numeric(d[c],errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(0)
    return d.reset_index(drop=True),exposure,pa_cols


def pair_point(y,p):
    fpr,tpr,thr=roc_curve(y,p); tnr=1-fpr; feasible=np.where((tpr>=.8)&(tnr>=.8))[0]; j=int(np.argmax(np.minimum(tpr,tnr)))
    if len(feasible):
        mins=np.minimum(tpr[feasible],tnr[feasible]); cand=feasible[mins==mins.max()]; j=int(cand[np.argmax((tpr[cand]+tnr[cand])/2)])
    return float(thr[j]),float(tnr[j]),float(tpr[j]),bool(len(feasible))


def recalls(y,p): return {b:float((p[y==b]==b).mean()) for b in [3,4,5]}


def main():
    d,exposure,pa=load(); sets={'pa_summary_only':pa,'exposure_plus_pa_summary':exposure+pa}; rows=[]; preds=[]
    for setname,feats in sets.items():
        X=d[feats]
        for name in ['et','rf','knn']:
            pr=np.zeros(len(d),int)
            for yr in sorted(d.fyDeclared.astype(int).unique()):
                tr=d.fyDeclared.astype(int)!=yr; te=~tr
                if name=='et': model=ExtraTreesClassifier(n_estimators=500,max_depth=5,min_samples_leaf=2,max_features=.7,class_weight='balanced',random_state=42,n_jobs=-1)
                elif name=='rf': model=RandomForestClassifier(n_estimators=500,max_depth=5,min_samples_leaf=2,max_features=.7,class_weight='balanced',random_state=42,n_jobs=-1)
                else: model=Pipeline([('sc',StandardScaler()),('knn',KNeighborsClassifier(n_neighbors=5,weights='distance'))])
                model.fit(X.loc[tr],d.loc[tr,'band']); pr[te]=model.predict(X.loc[te])
            rr=recalls(d.band.to_numpy(),pr); rows.append({'feature_set':setname,'model':name,'recall_50_200':rr[3],'recall_200_500':rr[4],'recall_500_plus':rr[5],'min_recall':min(rr.values()),'accuracy':float((pr==d.band.to_numpy()).mean())})
            z=d[['disasterNumber','state','fyDeclared','incidentType','target','band']].copy(); z['feature_set']=setname; z['model']=name; z['prediction']=pr; preds.append(z)
        for lo,hi in [(3,4),(4,5)]:
            mask=d.band.isin([lo,hi]); prob=np.full(len(d),np.nan)
            for yr in sorted(d.loc[mask,'fyDeclared'].astype(int).unique()):
                tr=mask&(d.fyDeclared.astype(int)!=yr); te=mask&(d.fyDeclared.astype(int)==yr); y=(d.loc[tr,'band']==hi).astype(int)
                model=ExtraTreesClassifier(n_estimators=500,max_depth=4,min_samples_leaf=1,max_features=.8,class_weight='balanced',random_state=42,n_jobs=-1); model.fit(X.loc[tr],y); prob[te]=model.predict_proba(X.loc[te])[:,1]
            yy=(d.loc[mask,'band']==hi).astype(int).to_numpy(); pp=prob[mask]; th,lr,ur,ok=pair_point(yy,pp)
            rows.append({'feature_set':setname,'model':f'pair_{lo}_{hi}_et','lower_recall':lr,'upper_recall':ur,'min_recall':min(lr,ur),'has_80_80':ok,'auc':float(roc_auc_score(yy,pp)),'threshold':th})
    pd.DataFrame(rows).to_csv(OUT/'screen_results.csv',index=False); pd.concat(preds,ignore_index=True).to_csv(OUT/'multiclass_predictions.csv',index=False)
    summary={'band_counts':{str(b):int((d.band==b).sum()) for b in [3,4,5]},'results':rows,'excluded_fields':['federalObligatedAmount','declarationDate','hash','id'],'notes':['Uses OpenFEMA PublicAssistanceFundedProjectsSummaries non-dollar structure only.','numberOfProjects is a count, not a financial amount.','Every prediction removes the entire held-out fiscal year.','This is a development feature screen; any winning configuration requires nested/frozen final validation.']}; (OUT/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
