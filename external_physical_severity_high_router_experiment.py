from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score, roc_curve

MASTER = Path('master_openfema_40plus.xlsx')
OUT = Path('external_physical_severity_high_router_results')
OUT.mkdir(exist_ok=True)
NOAA_INDEX = 'https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/'
CDC_URL = 'https://data.cdc.gov/resource/pwn4-m3yp.json'

POP = {'AL':4779736,'AK':710231,'AZ':6392017,'AR':2915918,'CA':37253956,'CO':5029196,'CT':3574097,'DE':897934,'DC':601723,'FL':18801310,'GA':9687653,'HI':1360301,'ID':1567582,'IL':12830632,'IN':6483802,'IA':3046355,'KS':2853118,'KY':4339367,'LA':4533372,'ME':1328361,'MD':5773552,'MA':6547629,'MI':9883640,'MN':5303925,'MS':2967297,'MO':5988927,'MT':989415,'NE':1826341,'NV':2700551,'NH':1316470,'NJ':8791894,'NM':2059179,'NY':19378102,'NC':9535483,'ND':672591,'OH':11536504,'OK':3751351,'OR':3831074,'PA':12702379,'RI':1052567,'SC':4625364,'SD':814180,'TN':6346105,'TX':25145561,'UT':2763885,'VT':625741,'VA':8001024,'WA':6724540,'WV':1852994,'WI':5686986,'WY':563626,'PR':3725789,'VI':106405,'GU':159358,'AS':55519,'MP':53883}

STATE_NAME = {
'AL':'ALABAMA','AK':'ALASKA','AZ':'ARIZONA','AR':'ARKANSAS','CA':'CALIFORNIA','CO':'COLORADO','CT':'CONNECTICUT','DE':'DELAWARE','DC':'DISTRICT OF COLUMBIA','FL':'FLORIDA','GA':'GEORGIA','HI':'HAWAII','ID':'IDAHO','IL':'ILLINOIS','IN':'INDIANA','IA':'IOWA','KS':'KANSAS','KY':'KENTUCKY','LA':'LOUISIANA','ME':'MAINE','MD':'MARYLAND','MA':'MASSACHUSETTS','MI':'MICHIGAN','MN':'MINNESOTA','MS':'MISSISSIPPI','MO':'MISSOURI','MT':'MONTANA','NE':'NEBRASKA','NV':'NEVADA','NH':'NEW HAMPSHIRE','NJ':'NEW JERSEY','NM':'NEW MEXICO','NY':'NEW YORK','NC':'NORTH CAROLINA','ND':'NORTH DAKOTA','OH':'OHIO','OK':'OKLAHOMA','OR':'OREGON','PA':'PENNSYLVANIA','RI':'RHODE ISLAND','SC':'SOUTH CAROLINA','SD':'SOUTH DAKOTA','TN':'TENNESSEE','TX':'TEXAS','UT':'UTAH','VT':'VERMONT','VA':'VIRGINIA','WA':'WASHINGTON','WV':'WEST VIRGINIA','WI':'WISCONSIN','WY':'WYOMING','PR':'PUERTO RICO','VI':'VIRGIN ISLANDS','GU':'GUAM','AS':'AMERICAN SAMOA','MP':'NORTHERN MARIANA ISLANDS'}

NOAA_EVENT_MAP = {
    'HURRICANE': {'HURRICANE (TYPHOON)','TROPICAL STORM','STORM SURGE/TIDE','HIGH SURF','COASTAL FLOOD','FLASH FLOOD','FLOOD','HEAVY RAIN','HIGH WIND'},
    'TROPICAL STORM': {'TROPICAL STORM','HURRICANE (TYPHOON)','STORM SURGE/TIDE','HIGH SURF','COASTAL FLOOD','FLASH FLOOD','FLOOD','HEAVY RAIN','HIGH WIND'},
    'SEVERE STORM': {'THUNDERSTORM WIND','TORNADO','HAIL','FLASH FLOOD','FLOOD','HEAVY RAIN','HIGH WIND','LIGHTNING','STRONG WIND'},
    'SEVERE STORMS': {'THUNDERSTORM WIND','TORNADO','HAIL','FLASH FLOOD','FLOOD','HEAVY RAIN','HIGH WIND','LIGHTNING','STRONG WIND'},
    'FLOOD': {'FLOOD','FLASH FLOOD','COASTAL FLOOD','LAKESHORE FLOOD','HEAVY RAIN'},
    'FIRE': {'WILDFIRE'},
    'WILDFIRE': {'WILDFIRE'},
    'TORNADO': {'TORNADO'},
    'SNOWSTORM': {'HEAVY SNOW','WINTER STORM','BLIZZARD','LAKE-EFFECT SNOW'},
    'SEVERE ICE STORM': {'ICE STORM','WINTER STORM','FREEZING FOG','SLEET'},
    'WINTER STORM': {'WINTER STORM','HEAVY SNOW','ICE STORM','BLIZZARD','EXTREME COLD/WIND CHILL'},
    'COASTAL STORM': {'COASTAL FLOOD','HIGH SURF','HIGH WIND','HEAVY RAIN'},
    'STRAIGHT-LINE WINDS': {'THUNDERSTORM WIND','HIGH WIND','STRONG WIND'},
}

BASE_NUM = [
    'durationDays','declarationDelayDays','expectedResourceScore','missionAssignmentCount','uniqueAgencyCount',
    'uniqueMaTypeCount','uniquePriorityCount','responseComplexityScore','missionDensity','agencyDensity',
    'population2010','eventSize','logPopulation','logMission','logComplexity','logAgency','logDuration','logEventSize',
    'missionAssignmentCount_event_pct_rank','responseComplexityScore_event_pct_rank','uniqueAgencyCount_event_pct_rank','population2010_event_pct_rank',
    'missionAssignmentCount_relative_event_avg','responseComplexityScore_relative_event_avg','uniqueAgencyCount_relative_event_avg','population2010_relative_event_avg',
    'ihProgramDeclared','iaProgramDeclared','paProgramDeclared','hmProgramDeclared'
]
BASE_CAT = ['state','incidentType','expectedResourceLevel','disasterCategory','durationClass','eventScale']
EXT_NUM = [
    'extCoverage','extSeverityEventPctRank','extSeverityEventShare','extSeverityRelativeEventAvg',
    'extFatalityEventPctRank','extFatalityEventShare','extAcuteEventPctRank','extAcuteEventShare',
    'extMagnitudeEventPctRank','extRecordEventPctRank','extRecordEventShare',
    'extRawSeverity','extRawFatalityRate','extRawAcuteRate','extRawMagnitude','extRawRecordCount'
]
EXT_CAT = ['extSource']

def num(s):
    return pd.to_numeric(s, errors='coerce').replace([np.inf,-np.inf],np.nan)

def load_master():
    d=pd.read_excel(MASTER)
    d['target']=num(d.totalObligatedFunding).fillna(0).clip(lower=0)
    d['band']=np.select([(d.target>50e6)&(d.target<=200e6),(d.target>200e6)&(d.target<=500e6),d.target>500e6],[3,4,5],default=-1).astype(int)
    d['population2010']=d.state.map(POP).astype(float)
    d['event_key']=d.incidentType.astype(str)+'|'+d.incidentBeginDate.astype(str)+'|'+d.incidentEndDate.astype(str)
    g=d.groupby('event_key',dropna=False)
    d['eventSize']=g.disasterNumber.transform('size').astype(float)
    for new,old in [('logPopulation','population2010'),('logMission','missionAssignmentCount'),('logComplexity','responseComplexityScore'),('logAgency','uniqueAgencyCount'),('logDuration','durationDays'),('logEventSize','eventSize')]:
        d[new]=np.log1p(num(d[old]).fillna(0).clip(lower=0))
    for c in ['missionAssignmentCount','responseComplexityScore','uniqueAgencyCount','population2010']:
        d[c]=num(d[c]).fillna(0)
        d[c+'_event_pct_rank']=g[c].rank(pct=True,method='average').fillna(.5)
        total=g[c].transform('sum').replace(0,np.nan)
        d[c+'_relative_event_avg']=(d[c]/total*d.eventSize).replace([np.inf,-np.inf],np.nan).fillna(0)
    dd=pd.to_datetime(d['declarationDate'],errors='coerce',utc=True).dt.tz_convert(None) if 'declarationDate' in d.columns else pd.Series(pd.NaT,index=d.index,dtype='datetime64[ns]')
    begin=pd.to_datetime(d.incidentBeginDate,errors='coerce',utc=True).dt.tz_convert(None)
    delay=pd.to_timedelta(num(d.declarationDelayDays).fillna(0),unit='D')
    d['effectiveDeclarationDate']=dd.fillna(begin+delay)
    d['incidentBeginDateParsed']=begin
    d['incidentEndDateParsed']=pd.to_datetime(d.incidentEndDate,errors='coerce',utc=True).dt.tz_convert(None)
    return d

def latest_noaa_files(years):
    html=requests.get(NOAA_INDEX,timeout=180,headers={'User-Agent':'ML-thesis-research/1.0'}).text
    out={}
    for y in years:
        names=re.findall(rf'StormEvents_details-ftp_v1\.0_d{int(y)}_c\d+\.csv\.gz',html)
        if names:
            out[int(y)]=sorted(set(names))[-1]
    return out

def fetch_noaa(years):
    files=latest_noaa_files(years)
    frames=[]
    use=['STATE','EVENT_TYPE','BEGIN_DATE_TIME','END_DATE_TIME','INJURIES_DIRECT','INJURIES_INDIRECT','DEATHS_DIRECT','DEATHS_INDIRECT','MAGNITUDE']
    for y in sorted(files):
        url=NOAA_INDEX+files[y]
        print('NOAA',y,files[y],flush=True)
        raw=requests.get(url,timeout=300,headers={'User-Agent':'ML-thesis-research/1.0'}).content
        df=pd.read_csv(io.BytesIO(raw),compression='gzip',usecols=lambda c:c in use,low_memory=False)
        df['sourceYear']=y
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=use+['sourceYear'])
    n=pd.concat(frames,ignore_index=True)
    n['STATE']=n.STATE.astype(str).str.upper().str.strip()
    n['EVENT_TYPE']=n.EVENT_TYPE.astype(str).str.upper().str.strip()
    n['BEGIN_DT']=pd.to_datetime(n.BEGIN_DATE_TIME,errors='coerce',utc=True).dt.tz_convert(None)
    n['END_DT']=pd.to_datetime(n.END_DATE_TIME,errors='coerce',utc=True).dt.tz_convert(None)
    for c in ['INJURIES_DIRECT','INJURIES_INDIRECT','DEATHS_DIRECT','DEATHS_INDIRECT','MAGNITUDE']:
        n[c]=num(n[c]).fillna(0)
    return n

def fetch_cdc():
    params={'$limit':50000,'$select':'date_updated,state,start_date,end_date,tot_cases,new_cases,tot_deaths,new_deaths'}
    r=requests.get(CDC_URL,params=params,timeout=180,headers={'User-Agent':'ML-thesis-research/1.0'})
    r.raise_for_status()
    c=pd.DataFrame(r.json())
    if c.empty:
        return c
    for x in ['date_updated','start_date','end_date']:
        c[x]=pd.to_datetime(c[x],errors='coerce',utc=True).dt.tz_convert(None)
    for x in ['tot_cases','new_cases','tot_deaths','new_deaths']:
        c[x]=num(c[x]).fillna(0)
    c['state']=c.state.astype(str).str.upper().str.strip()
    return c

def noaa_types(incident):
    s=str(incident).upper().strip()
    if s in NOAA_EVENT_MAP:
        return NOAA_EVENT_MAP[s]
    return {x for vals in NOAA_EVENT_MAP.values() for x in vals if x==s}

def build_external_features(d):
    high_events=set(d.loc[d.band>=3,'event_key'])
    peers=d[d.event_key.isin(high_events)].copy()
    years=sorted(peers.fyDeclared.dropna().astype(int).unique().tolist())
    noaa=fetch_noaa(years)
    cdc=fetch_cdc()
    print('high events',len(high_events),'peer disasters',len(peers),'NOAA rows',len(noaa),'CDC rows',len(cdc),flush=True)
    rows=[]
    for _,r in peers.iterrows():
        st=str(r.state)
        inc=str(r.incidentType)
        decl=r.effectiveDeclarationDate
        begin=r.incidentBeginDateParsed
        pop=float(r.population2010) if pd.notna(r.population2010) and r.population2010>0 else np.nan
        source='NONE'
        coverage=0
        rec=0.
        fatal=0.
        acute=0.
        magnitude=0.
        raw_sev=0.
        if inc.upper().strip()=='BIOLOGICAL' and not cdc.empty and pd.notna(decl):
            cc=cdc[(cdc.state==st)&(cdc.end_date<=decl)].sort_values('end_date')
            if len(cc):
                z=cc.iloc[-1]
                source='CDC'
                coverage=1
                rec=1.
                cases=float(z.tot_cases)
                newcases=max(float(z.new_cases),0)
                deaths=float(z.tot_deaths)
                newdeaths=max(float(z.new_deaths),0)
                cases_rate=cases/pop*1e5 if pop else 0.
                newcase_rate=newcases/pop*1e5 if pop else 0.
                death_rate=deaths/pop*1e5 if pop else 0.
                newdeath_rate=newdeaths/pop*1e5 if pop else 0.
                fatal=death_rate
                acute=newcase_rate+5*newdeath_rate
                raw_sev=np.log1p(cases_rate)+1.5*np.log1p(death_rate)+.75*np.log1p(newcase_rate)+.75*np.log1p(newdeath_rate)
        else:
            types=noaa_types(inc)
            if types and pd.notna(decl) and pd.notna(begin) and len(noaa):
                state_name=STATE_NAME.get(st,'')
                nn=noaa[(noaa.STATE==state_name)&(noaa.EVENT_TYPE.isin(types))&(noaa.END_DT<=decl)&(noaa.BEGIN_DT>=begin-pd.Timedelta(days=21))]
                if len(nn):
                    source='NOAA'
                    coverage=1
                    rec=float(len(nn))
                    injuries=float((nn.INJURIES_DIRECT+nn.INJURIES_INDIRECT).sum())
                    deaths=float((nn.DEATHS_DIRECT+nn.DEATHS_INDIRECT).sum())
                    fatal=deaths/pop*1e6 if pop else deaths
                    acute=injuries/pop*1e6 if pop else injuries
                    magnitude=float(nn.MAGNITUDE.clip(lower=0).max())
                    raw_sev=np.log1p(rec)+1.25*np.log1p(max(acute,0))+2*np.log1p(max(fatal,0))+.25*np.log1p(max(magnitude,0))
        rows.append({'disasterNumber':int(r.disasterNumber),'event_key':r.event_key,'extSource':source,'extCoverage':coverage,'extRawRecordCount':rec,'extRawFatalityRate':fatal,'extRawAcuteRate':acute,'extRawMagnitude':magnitude,'extRawSeverity':raw_sev})
    e=pd.DataFrame(rows)
    gp=e.groupby('event_key',dropna=False)
    e['extEventPeerCount']=gp.disasterNumber.transform('size')
    for c,prefix in [('extRawSeverity','extSeverity'),('extRawFatalityRate','extFatality'),('extRawAcuteRate','extAcute'),('extRawMagnitude','extMagnitude'),('extRawRecordCount','extRecord')]:
        e[prefix+'EventPctRank']=gp[c].rank(pct=True,method='average').fillna(.5)
        total=gp[c].transform('sum').replace(0,np.nan)
        e[prefix+'EventShare']=(e[c]/total).replace([np.inf,-np.inf],np.nan).fillna(0)
        e[prefix+'RelativeEventAvg']=(e[prefix+'EventShare']*e.extEventPeerCount).fillna(0)
    e.to_csv(OUT/'external_severity_event_peer_features.csv',index=False)
    return e

def prep(df, nums, cats):
    X=df[nums+cats].copy()
    for c in nums:
        X[c]=num(X[c]).fillna(0)
    for c in cats:
        X[c]=X[c].astype(str).fillna('MISSING')
    return X

def fit_cat(Xtr,ytr,Xte,cats):
    model=CatBoostClassifier(iterations=220,depth=3,learning_rate=.03,l2_leaf_reg=30,loss_function='MultiClass' if ytr.nunique()>2 else 'Logloss',auto_class_weights='Balanced',verbose=False,random_seed=42,allow_writing_files=False)
    model.fit(Xtr,ytr,cat_features=cats)
    return model

def fit_et(Xtr,ytr,Xte,cats):
    A=Xtr.copy()
    B=Xte.copy()
    for c in cats:
        levels=sorted(set(A[c].astype(str)))
        mp={v:i for i,v in enumerate(levels)}
        A[c]=A[c].astype(str).map(mp).fillna(-1)
        B[c]=B[c].astype(str).map(mp).fillna(-1)
    model=ExtraTreesClassifier(n_estimators=500,max_depth=4,min_samples_leaf=1,max_features=.8,class_weight='balanced',random_state=42,n_jobs=-1)
    model.fit(A,ytr)
    return model,A,B

def metrics(y,p):
    recs=recall_score(y,p,labels=[3,4,5],average=None,zero_division=0)
    return {'r3':float(recs[0]),'r4':float(recs[1]),'r5':float(recs[2]),'min_recall':float(recs.min()),'balanced_accuracy':float(balanced_accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,average='macro',zero_division=0)),'confusion_matrix':confusion_matrix(y,p,labels=[3,4,5]).tolist()}

def op_point(y,p):
    fpr,tpr,thr=roc_curve(y,p)
    tnr=1-fpr
    feasible=np.where((tpr>=.8)&(tnr>=.8))[0]
    cand=feasible if len(feasible) else np.arange(len(thr))
    mn=np.minimum(tpr[cand],tnr[cand])
    best=mn.max()
    c2=cand[mn==best]
    j=int(c2[np.argmax((tpr[c2]+tnr[c2])/2)])
    return {'auc':float(roc_auc_score(y,p)),'threshold':float(thr[j]),'lower_recall':float(tnr[j]),'upper_recall':float(tpr[j]),'min_recall':float(min(tnr[j],tpr[j])),'pass80':bool(len(feasible))}

def screen(d):
    e=build_external_features(d)
    d=d.merge(e.drop(columns=['event_key']),on='disasterNumber',how='left')
    for c in EXT_NUM:
        if c not in d.columns:
            d[c]=0.
        d[c]=num(d[c]).fillna(0)
    d['extSource']=d.get('extSource','NONE').fillna('NONE').astype(str)
    high=d[d.band.isin([3,4,5])].copy().reset_index(drop=True)
    sets={'baseline':(BASE_NUM,BASE_CAT),'baseline_plus_external':(BASE_NUM+EXT_NUM,BASE_CAT+EXT_CAT)}
    rows=[]
    pred_rows=[]
    pair_rows=[]
    years=sorted(high.fyDeclared.astype(int).unique())
    for fs,(nums,cats) in sets.items():
        nums=[c for c in nums if c in high.columns]
        cats=[c for c in cats if c in high.columns]
        for family in ['cat','et']:
            pred=np.full(len(high),-99,int)
            for yr in years:
                tr=high.fyDeclared.astype(int)!=yr
                te=~tr
                Xtr=prep(high.loc[tr],nums,cats)
                Xte=prep(high.loc[te],nums,cats)
                ytr=high.loc[tr,'band'].astype(int)
                if family=='cat':
                    pred[te]=np.asarray(fit_cat(Xtr,ytr,Xte,cats).predict(Xte)).reshape(-1).astype(int)
                else:
                    m,A,B=fit_et(Xtr,ytr,Xte,cats)
                    pred[te]=m.predict(B).astype(int)
            met=metrics(high.band.astype(int).to_numpy(),pred)
            rows.append({'feature_set':fs,'model':family,**{k:v for k,v in met.items() if k!='confusion_matrix'},'confusion_matrix':json.dumps(met['confusion_matrix'])})
            q=high[['disasterNumber','state','fyDeclared','incidentType','target','band','extSource']].copy()
            q['feature_set']=fs
            q['model']=family
            q['predicted_band']=pred
            pred_rows.append(q)
        for lo,hi in [(3,4),(4,5),(3,5)]:
            mask=high.band.isin([lo,hi]).to_numpy()
            p=np.full(len(high),np.nan)
            for yr in sorted(high.loc[mask,'fyDeclared'].astype(int).unique()):
                tr=mask&(high.fyDeclared.astype(int).to_numpy()!=yr)
                te=mask&(high.fyDeclared.astype(int).to_numpy()==yr)
                Xtr=prep(high.loc[tr],nums,cats)
                Xte=prep(high.loc[te],nums,cats)
                ytr=(high.loc[tr,'band']==hi).astype(int)
                m=fit_cat(Xtr,ytr,Xte,cats)
                cls=list(m.classes_)
                p[te]=m.predict_proba(Xte)[:,cls.index(1)]
            yy=(high.loc[mask,'band']==hi).astype(int).to_numpy()
            met=op_point(yy,p[mask])
            pair_rows.append({'feature_set':fs,'pair':f'{lo}vs{hi}','model':'cat',**met})
    pd.DataFrame(rows).sort_values(['min_recall','balanced_accuracy'],ascending=False).to_csv(OUT/'multiclass_results.csv',index=False)
    pd.concat(pred_rows,ignore_index=True).to_csv(OUT/'multiclass_predictions.csv',index=False)
    pd.DataFrame(pair_rows).sort_values(['pair','min_recall'],ascending=[True,False]).to_csv(OUT/'pairwise_results.csv',index=False)
    bio=high[high.incidentType.astype(str).str.upper().eq('BIOLOGICAL')][['disasterNumber','state','fyDeclared','target','band','effectiveDeclarationDate','extSource']+[c for c in EXT_NUM if c in high.columns]].copy()
    bio.to_csv(OUT/'biological_high_case_features.csv',index=False)
    cov=high.groupby(['incidentType','extSource']).size().reset_index(name='n')
    cov.to_csv(OUT/'coverage_by_incident_source.csv',index=False)
    summary={'high_counts':{str(b):int((high.band==b).sum()) for b in [3,4,5]},'coverage':{'external_covered_high':int((high.extCoverage>0).sum()),'high_total':int(len(high)),'by_source':high.extSource.value_counts().to_dict()},'multiclass':rows,'pairwise':pair_rows,'notes':['No funding/obligation/project-dollar field is used as an external predictor.','NOAA rows are restricted to events ending on or before the FEMA declaration date and beginning no earlier than 21 days before the FEMA incident begin date.','CDC uses the latest weekly observation whose end_date is on or before the FEMA declaration date.','External raw severity is converted to target-free within-FEMA-event ranks/shares so the same dimensionless features can transfer across incident types.','This is a leave-fiscal-year-out development screen. Any apparent winner must be fully nested before a final claim.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str),flush=True)

if __name__=='__main__':
    d=load_master()
    screen(d)
