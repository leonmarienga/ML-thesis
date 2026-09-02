from __future__ import annotations

import io
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

import full_funding_range_router_experiment as base

OUT = Path('declared_population_exposure_router_results')
OUT.mkdir(exist_ok=True)
MISSION_JSONL = Path('mission_composition_results/MissionAssignments_v2.jsonl')
FEMA_DECL_URL = 'https://www.fema.gov/api/open/v1/DisasterDeclarationsSummaries.csv'
CENSUS_GAZ_URL = 'https://www2.census.gov/geo/docs/maps-data/data/gazetteer/Gaz_counties_national.zip'

STATE_FIPS = {
    'AL':'01','AK':'02','AZ':'04','AR':'05','CA':'06','CO':'08','CT':'09','DE':'10','DC':'11','FL':'12','GA':'13','HI':'15',
    'ID':'16','IL':'17','IN':'18','IA':'19','KS':'20','KY':'21','LA':'22','ME':'23','MD':'24','MA':'25','MI':'26','MN':'27',
    'MS':'28','MO':'29','MT':'30','NE':'31','NV':'32','NH':'33','NJ':'34','NM':'35','NY':'36','NC':'37','ND':'38','OH':'39',
    'OK':'40','OR':'41','PA':'42','RI':'44','SC':'45','SD':'46','TN':'47','TX':'48','UT':'49','VT':'50','VA':'51','WA':'53',
    'WV':'54','WI':'55','WY':'56','PR':'72','VI':'78','GU':'66','AS':'60','MP':'69'
}

BASE_NUMERIC = list(dict.fromkeys(base.ORDINAL_FEATURES + [
    'durationDays','declarationDelayDays','expectedResourceScore','missionAssignmentCount','uniqueAgencyCount',
    'uniqueMaTypeCount','uniquePriorityCount','responseComplexityScore','missionDensity','agencyDensity',
    'population2010','eventSize','fyDeclared','ihProgramDeclared','iaProgramDeclared','paProgramDeclared','hmProgramDeclared'
]))
BASE_CATS = [c for c in ['incidentType','state','expectedResourceLevel','disasterCategory','durationClass','eventScale','declarationType']]

POP_EXPOSURE_COLS = [
    'declaredCountyMatchedCount','declaredUniquePlaceCount','declaredGeographyRowCount','declaredCountyMatchRatio',
    'declaredPopulation2010','logDeclaredPopulation2010','declaredPopulationShareState',
    'declaredCountyPopulationMean','declaredCountyPopulationMedian','declaredCountyPopulationMax',
    'declaredLandSqMi','logDeclaredLandSqMi','declaredLandShareState','declaredPopulationDensity',
    'stateGazetteerPopulation2010','stateGazetteerLandSqMi','statewideDeclarationFlag'
]
COUNT_ONLY_COLS = [
    'declaredCountyMatchedCount','declaredUniquePlaceCount','declaredGeographyRowCount','declaredCountyMatchRatio','statewideDeclarationFlag'
]

MODEL_CONFIGS = [
    {'iterations': 80, 'depth': 2, 'learning_rate': 0.04, 'l2_leaf_reg': 20},
    {'iterations': 120, 'depth': 2, 'learning_rate': 0.03, 'l2_leaf_reg': 30},
    {'iterations': 120, 'depth': 3, 'learning_rate': 0.03, 'l2_leaf_reg': 30},
    {'iterations': 180, 'depth': 2, 'learning_rate': 0.025, 'l2_leaf_reg': 40},
]


def get_bytes(url: str, timeout: int = 180) -> bytes:
    r = requests.get(url, timeout=timeout, headers={'User-Agent': 'ML-thesis-research/1.0'})
    r.raise_for_status()
    return r.content


def load_census_gazetteer() -> pd.DataFrame:
    raw = get_bytes(CENSUS_GAZ_URL)
    with ZipFile(io.BytesIO(raw)) as z:
        names = [n for n in z.namelist() if not n.endswith('/')]
        if not names:
            raise RuntimeError('Census Gazetteer ZIP contained no file')
        with z.open(names[0]) as f:
            gaz = pd.read_csv(f, sep='\t', dtype=str, encoding='latin1')
    gaz.columns = [str(c).strip() for c in gaz.columns]
    # Some releases have padded field names/values.
    for c in gaz.columns:
        if gaz[c].dtype == object:
            gaz[c] = gaz[c].astype(str).str.strip()
    required = {'USPS','GEOID','POP10','ALAND_SQMI'}
    missing = required - set(gaz.columns)
    if missing:
        raise RuntimeError(f'Missing Gazetteer fields: {sorted(missing)}; got {gaz.columns.tolist()}')
    gaz['GEOID'] = gaz['GEOID'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(5)
    gaz['POP10'] = pd.to_numeric(gaz['POP10'], errors='coerce').fillna(0.0)
    gaz['ALAND_SQMI'] = pd.to_numeric(gaz['ALAND_SQMI'], errors='coerce').fillna(0.0)
    return gaz[['USPS','GEOID','POP10','ALAND_SQMI']].drop_duplicates('GEOID')


def normalize_place_code(x) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip()
    s = re.sub(r'\.0$', '', s)
    digits = re.sub(r'\D', '', s)
    if len(digits) < 3:
        return None
    return digits[-3:].zfill(3)


def build_population_exposure(master: pd.DataFrame) -> pd.DataFrame:
    wanted = set(pd.to_numeric(master['disasterNumber'], errors='coerce').dropna().astype(int).tolist())
    dec_bytes = get_bytes(FEMA_DECL_URL, timeout=240)
    dec = pd.read_csv(io.BytesIO(dec_bytes), dtype={'state':str,'placeCode':str}, low_memory=False)
    dec['disasterNumber'] = pd.to_numeric(dec['disasterNumber'], errors='coerce')
    dec = dec[dec['disasterNumber'].isin(wanted)].copy()
    dec['disasterNumber'] = dec['disasterNumber'].astype(int)
    dec['state'] = dec['state'].astype(str).str.strip()
    dec['county3'] = dec['placeCode'].map(normalize_place_code)
    dec['stateFips'] = dec['state'].map(STATE_FIPS)
    dec['candidateGEOID'] = np.where(
        dec['stateFips'].notna() & dec['county3'].notna(),
        dec['stateFips'].fillna('') + dec['county3'].fillna(''),
        None,
    )
    area_text = dec.get('declaredCountyArea', pd.Series('', index=dec.index)).fillna('').astype(str)
    dec['statewideFlagRow'] = area_text.str.contains('statewide', case=False, regex=False)

    gaz = load_census_gazetteer()
    valid_geoids = set(gaz['GEOID'])
    dec['matchedGEOID'] = dec['candidateGEOID'].where(dec['candidateGEOID'].isin(valid_geoids))

    state_tot = gaz.groupby('USPS', as_index=False).agg(
        stateGazetteerPopulation2010=('POP10','sum'),
        stateGazetteerLandSqMi=('ALAND_SQMI','sum'),
    )
    lookup = gaz.set_index('GEOID')[['POP10','ALAND_SQMI']]

    rows = []
    for disaster, g in dec.groupby('disasterNumber'):
        st = str(g['state'].dropna().iloc[0]) if g['state'].notna().any() else ''
        statewide = bool(g['statewideFlagRow'].any())
        geoids = sorted(set(g['matchedGEOID'].dropna().astype(str)))
        if statewide and st in set(state_tot['USPS']):
            sg = gaz[gaz['USPS'] == st]
            geoids = sorted(set(sg['GEOID'].astype(str)))
        vals = lookup.loc[geoids] if geoids else pd.DataFrame(columns=['POP10','ALAND_SQMI'])
        pops = pd.to_numeric(vals.get('POP10', pd.Series(dtype=float)), errors='coerce').fillna(0.0)
        lands = pd.to_numeric(vals.get('ALAND_SQMI', pd.Series(dtype=float)), errors='coerce').fillna(0.0)
        row = {
            'disasterNumber': int(disaster),
            'state': st,
            'declaredCountyMatchedCount': int(len(geoids)),
            'declaredUniquePlaceCount': int(g['placeCode'].astype(str).nunique()),
            'declaredGeographyRowCount': int(len(g)),
            'statewideDeclarationFlag': int(statewide),
            'declaredPopulation2010': float(pops.sum()),
            'declaredCountyPopulationMean': float(pops.mean()) if len(pops) else 0.0,
            'declaredCountyPopulationMedian': float(pops.median()) if len(pops) else 0.0,
            'declaredCountyPopulationMax': float(pops.max()) if len(pops) else 0.0,
            'declaredLandSqMi': float(lands.sum()),
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    out = out.merge(state_tot, left_on='state', right_on='USPS', how='left').drop(columns=['USPS'], errors='ignore')
    out['stateGazetteerPopulation2010'] = pd.to_numeric(out['stateGazetteerPopulation2010'], errors='coerce').fillna(0.0)
    out['stateGazetteerLandSqMi'] = pd.to_numeric(out['stateGazetteerLandSqMi'], errors='coerce').fillna(0.0)
    out['declaredCountyMatchRatio'] = out['declaredCountyMatchedCount'] / out['declaredUniquePlaceCount'].replace(0, np.nan)
    out['declaredCountyMatchRatio'] = out['declaredCountyMatchRatio'].replace([np.inf,-np.inf], np.nan).fillna(0.0).clip(0,1)
    out['declaredPopulationShareState'] = out['declaredPopulation2010'] / out['stateGazetteerPopulation2010'].replace(0,np.nan)
    out['declaredPopulationShareState'] = out['declaredPopulationShareState'].replace([np.inf,-np.inf],np.nan).fillna(0.0).clip(0,1.5)
    out['declaredLandShareState'] = out['declaredLandSqMi'] / out['stateGazetteerLandSqMi'].replace(0,np.nan)
    out['declaredLandShareState'] = out['declaredLandShareState'].replace([np.inf,-np.inf],np.nan).fillna(0.0).clip(0,1.5)
    out['declaredPopulationDensity'] = out['declaredPopulation2010'] / out['declaredLandSqMi'].replace(0,np.nan)
    out['declaredPopulationDensity'] = out['declaredPopulationDensity'].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    out['logDeclaredPopulation2010'] = np.log1p(out['declaredPopulation2010'].clip(lower=0))
    out['logDeclaredLandSqMi'] = np.log1p(out['declaredLandSqMi'].clip(lower=0))
    out.to_csv(OUT/'declared_population_exposure_features.csv', index=False)
    return out


def entropy_hhi_top(counter: Counter) -> tuple[float,float,float]:
    vals = np.asarray(list(counter.values()), dtype=float)
    if vals.size == 0 or vals.sum() <= 0:
        return 0.0, 0.0, 0.0
    p = vals / vals.sum()
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    hhi = float((p ** 2).sum())
    top = float(p.max())
    return ent, hhi, top


def build_operational_mission_features(master: pd.DataFrame) -> pd.DataFrame:
    if not MISSION_JSONL.exists():
        raise FileNotFoundError(f'Frozen MissionAssignments snapshot missing: {MISSION_JSONL}')
    wanted = set(pd.to_numeric(master['disasterNumber'], errors='coerce').dropna().astype(int))
    acc = {}
    def get_acc(dn):
        if dn not in acc:
            acc[dn] = {
                'rows':0,'maIds':set(),'actionIds':set(),'agencyIds':set(),
                'amendSum':0.0,'amendCount':0,'amendMax':0.0,'amendPositive':0,
                'fedPct':[],'sttPct':[],
                'agency':Counter(),'maType':Counter(),'priority':Counter(),'supportFunction':Counter(),
                'authority':Counter(),'region':Counter(),
            }
        return acc[dn]
    with MISSION_JSONL.open('r', encoding='utf-8') as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            dn = r.get('disasterNumber')
            try:
                dn = int(dn)
            except Exception:
                continue
            if dn not in wanted:
                continue
            a = get_acc(dn); a['rows'] += 1
            for field,key in [('maId','maIds'),('actionId','actionIds'),('agencyId','agencyIds')]:
                v = r.get(field)
                if v not in (None,''):
                    a[key].add(str(v))
            am = pd.to_numeric(pd.Series([r.get('maAmendNumber')]), errors='coerce').iloc[0]
            if pd.notna(am):
                am = float(am); a['amendSum'] += am; a['amendCount'] += 1; a['amendMax'] = max(a['amendMax'], am); a['amendPositive'] += int(am > 0)
            for field,key in [('fedCostSharePct','fedPct'),('sttCostSharePct','sttPct')]:
                v = pd.to_numeric(pd.Series([r.get(field)]), errors='coerce').iloc[0]
                if pd.notna(v): a[key].append(float(v))
            for field,key in [('agency','agency'),('maType','maType'),('priority','priority'),('supportFunction','supportFunction'),('authority','authority'),('region','region')]:
                v = r.get(field)
                if v not in (None,''):
                    a[key][str(v)] += 1
    rows = []
    for dn in wanted:
        a = acc.get(dn)
        if not a:
            rows.append({'disasterNumber':dn})
            continue
        row = {
            'disasterNumber':dn,'missionRowCount':a['rows'],'uniqueMaIdOperational':len(a['maIds']),
            'uniqueActionIdOperational':len(a['actionIds']),'uniqueAgencyIdOperational':len(a['agencyIds']),
            'maxMaAmendNumber':a['amendMax'],'meanMaAmendNumber':a['amendSum']/max(a['amendCount'],1),
            'amendmentRowShare':a['amendPositive']/max(a['amendCount'],1),'rowsPerMaId':a['rows']/max(len(a['maIds']),1),
            'fedCostSharePctMean':float(np.mean(a['fedPct'])) if a['fedPct'] else 0.0,
            'fedCostSharePctStd':float(np.std(a['fedPct'])) if a['fedPct'] else 0.0,
            'sttCostSharePctMean':float(np.mean(a['sttPct'])) if a['sttPct'] else 0.0,
            'sttCostSharePctStd':float(np.std(a['sttPct'])) if a['sttPct'] else 0.0,
        }
        for name,key in [('agency','agency'),('matype','maType'),('priority','priority'),('support','supportFunction'),('authority','authority'),('region','region')]:
            ent,hhi,top = entropy_hhi_top(a[key]); row[f'{name}_oper_entropy']=ent; row[f'{name}_oper_hhi']=hhi; row[f'{name}_oper_top_share']=top; row[f'{name}_oper_unique']=len(a[key])
        rows.append(row)
    out = pd.DataFrame(rows).fillna(0)
    out.to_csv(OUT/'operational_mission_features.csv', index=False)
    return out


def prep_cat(df: pd.DataFrame, features: list[str]):
    X = df[features].copy()
    cats = []
    for c in features:
        if not pd.api.types.is_numeric_dtype(X[c]):
            cats.append(c)
            X[c] = X[c].fillna('MISSING').astype(str)
        else:
            X[c] = pd.to_numeric(X[c], errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(0)
    return X, cats


def best_operating_point(y: np.ndarray, p: np.ndarray) -> dict:
    fpr,tpr,thr = roc_curve(y,p); tnr = 1-fpr
    feasible = np.where((tpr >= .80) & (tnr >= .80))[0]
    k = int(np.argmax(np.minimum(tpr,tnr)))
    if len(feasible):
        # Prefer the point with highest worst-side recall, then highest mean recall.
        mins = np.minimum(tpr[feasible], tnr[feasible])
        best_min = mins.max(); cand = feasible[mins == best_min]
        j = int(cand[np.argmax((tpr[cand]+tnr[cand])/2)])
    else:
        j = k
    return {
        'threshold':float(thr[j]),'upper_recall':float(tpr[j]),'lower_recall':float(tnr[j]),
        'min_side_recall':float(min(tpr[j],tnr[j])),'has_80_80_point':bool(len(feasible)>0),
        'best_min_side_recall':float(min(tpr[k],tnr[k])),'best_min_threshold':float(thr[k]),
    }


def run_pair_oof(d: pd.DataFrame, lo: int, hi: int, features: list[str], cfg: dict):
    pair = d['band6_int'].isin([lo,hi]).to_numpy()
    prob = np.full(len(d), np.nan)
    for year in sorted(d.loc[pair,'fyDeclared'].astype(int).unique()):
        tr = pair & (d['fyDeclared'].astype(int).to_numpy() != year)
        te = pair & (d['fyDeclared'].astype(int).to_numpy() == year)
        if te.sum() == 0:
            continue
        y = (d.loc[tr,'band6_int'].to_numpy() == hi).astype(int)
        if len(np.unique(y)) < 2:
            prob[te] = float(y[0]) if len(y) else .5
            continue
        X,cats = prep_cat(d.loc[tr], features); Xe,_ = prep_cat(d.loc[te], features)
        n0,n1 = int((y==0).sum()), int((y==1).sum())
        model = CatBoostClassifier(
            **cfg, loss_function='Logloss', verbose=False, random_seed=42,
            allow_writing_files=False, class_weights=[1.0, n0/max(n1,1)]
        )
        model.fit(X,y,cat_features=cats)
        prob[te] = model.predict_proba(Xe)[:,1]
    yy = (d.loc[pair,'band6_int'].to_numpy() == hi).astype(int)
    pp = prob[pair]
    ok = np.isfinite(pp); yy,pp = yy[ok],pp[ok]
    met = best_operating_point(yy,pp)
    met.update({
        'auc':float(roc_auc_score(yy,pp)),'average_precision':float(average_precision_score(yy,pp)),
        'n_lower':int((yy==0).sum()),'n_upper':int((yy==1).sum()),
    })
    return met, prob


def main():
    d = base.load_data()
    # Keep only features verified safe for routing; eventScale was retained after the notebook leakage audit.
    exposure = build_population_exposure(d)
    oper = build_operational_mission_features(d)
    d = d.merge(exposure.drop(columns=['state'], errors='ignore'), on='disasterNumber', how='left')
    d = d.merge(oper, on='disasterNumber', how='left')
    for c in POP_EXPOSURE_COLS + [c for c in oper.columns if c != 'disasterNumber']:
        if c not in d:
            d[c] = 0.0
        if pd.api.types.is_numeric_dtype(d[c]):
            d[c] = pd.to_numeric(d[c], errors='coerce').fillna(0.0)
    safe_cats = [c for c in BASE_CATS if c in d.columns]
    oper_cols = [c for c in oper.columns if c != 'disasterNumber']
    variants = {
        'counts_plus_operational': list(dict.fromkeys(BASE_NUMERIC + COUNT_ONLY_COLS + oper_cols + safe_cats)),
        'population_exposure_plus_operational': list(dict.fromkeys(BASE_NUMERIC + POP_EXPOSURE_COLS + oper_cols + safe_cats)),
    }
    results=[]; pred_rows=[]
    for variant,features in variants.items():
        features=[c for c in features if c in d.columns]
        for lo,hi in [(3,4),(4,5)]:
            for ci,cfg in enumerate(MODEL_CONFIGS):
                met,prob = run_pair_oof(d,lo,hi,features,cfg)
                row={'variant':variant,'lower_band':base.BANDS6[lo],'upper_band':base.BANDS6[hi],'config_index':ci,**cfg,**met}
                results.append(row)
                mask=d.band6_int.isin([lo,hi])
                tmp=d.loc[mask,['disasterNumber','state','fyDeclared','incidentType','target','band6']].copy()
                tmp['variant']=variant; tmp['pair']=f'{base.BANDS6[lo]}__vs__{base.BANDS6[hi]}'; tmp['config_index']=ci; tmp['prob_upper']=prob[mask]
                pred_rows.append(tmp)
                print(row)
    res=pd.DataFrame(results)
    res.to_csv(OUT/'pairwise_screen.csv',index=False)
    pd.concat(pred_rows,ignore_index=True).to_csv(OUT/'pairwise_oof_predictions.csv',index=False)
    selected=[]
    for (variant,lo,hi),g in res.groupby(['variant','lower_band','upper_band']):
        gg=g.copy(); gg['rank80']=gg['has_80_80_point'].astype(int)
        r=gg.sort_values(['rank80','min_side_recall','auc'],ascending=False).iloc[0].to_dict(); selected.append(r)
    sel=pd.DataFrame(selected); sel.to_csv(OUT/'selected_results.csv',index=False)
    summary={
        'objective':'Test whether full declared population/land exposure moves the two high-value boundaries above the 80% recall floor on both sides.',
        'data_sources':{'FEMA_declarations':FEMA_DECL_URL,'Census_2010_county_gazetteer':CENSUS_GAZ_URL,'MissionAssignments':'frozen cache snapshot'},
        'selected_results':selected,
        'population_exposure_coverage':{
            'disasters_with_any_matched_county':int((d['declaredCountyMatchedCount']>0).sum()),
            'of_total_disasters':int(len(d)),
            'mean_county_match_ratio':float(d['declaredCountyMatchRatio'].mean()),
            'median_county_match_ratio':float(d['declaredCountyMatchRatio'].median()),
        },
        'notes':[
            'No target funding value, obligation amount, funding-per-mission field, or obligation-derived timing is used as a model input.',
            'Population and land features come from 2010 Census Gazetteer county records joined to FEMA declaration place codes.',
            'FEMA placeCode is validated against a real state+county GEOID before a county is counted; unmatched tribal/special areas remain unmatched rather than being guessed.',
            'Mission operational structure excludes obligation amounts, cost-share amounts, dateObligated, assistanceRequested, and statementOfWork.',
            'Every prediction removes the entire held-out fiscal year from model fitting.',
            'Thresholds in this screen are selected on combined outer leave-year-out predictions and are development diagnostics; any winning architecture must later freeze or nest threshold selection.',
        ]
    }
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
