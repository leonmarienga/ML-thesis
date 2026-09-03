from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

MASTER = Path('master_openfema_40plus.xlsx')
SCALE = Path('frozen_declaration_scale_features.csv')
OUT = Path('systematic_feature_engineering_high_router_results')
OUT.mkdir(exist_ok=True)

POP = {'AL':4779736,'AK':710231,'AZ':6392017,'AR':2915918,'CA':37253956,'CO':5029196,'CT':3574097,'DE':897934,'DC':601723,'FL':18801310,'GA':9687653,'HI':1360301,'ID':1567582,'IL':12830632,'IN':6483802,'IA':3046355,'KS':2853118,'KY':4339367,'LA':4533372,'ME':1328361,'MD':5773552,'MA':6547629,'MI':9883640,'MN':5303925,'MS':2967297,'MO':5988927,'MT':989415,'NE':1826341,'NV':2700551,'NH':1316470,'NJ':8791894,'NM':2059179,'NY':19378102,'NC':9535483,'ND':672591,'OH':11536504,'OK':3751351,'OR':3831074,'PA':12702379,'RI':1052567,'SC':4625364,'SD':814180,'TN':6346105,'TX':25145561,'UT':2763885,'VT':625741,'VA':8001024,'WA':6724540,'WV':1852994,'WI':5686986,'WY':563626,'PR':3725789,'AS':55519,'GU':159358,'MP':53883,'VI':106405}

LEAKAGE_TOKENS = [
    'totalobligatedfunding','obligationamount','federalshareobligated','totalobligated',
    'avgobligation','medianobligation','maxobligation','mindateobligated','maxdateobligated',
    'daystofirstobligation','obligationspan','fundingpermission','fundingper','projectamount'
]

CATS = [
    'state','incidentType','expectedResourceLevel','disasterCategory','durationClass','eventScale'
]

RAW_CANDIDATES = [
    'ihProgramDeclared','iaProgramDeclared','paProgramDeclared','hmProgramDeclared',
    'durationDays','declarationDelayDays','expectedResourceScore',
    'missionAssignmentCount','uniqueAgencyCount','uniqueMaTypeCount','uniquePriorityCount',
    'responseComplexityScore','missionDensity','agencyDensity','population2010','eventSize',
    'declaredAreaCount','uniquePlaceCodeCount','declarationGeographyRowCount',
    'yearSince2010','yearNorm','yearNormSq'
]

NORM_SOURCES = [
    'logMission','logComplexity','logAgency','logPopulation','logDuration',
    'expectedResourceScore','missionDensity','agencyDensity',
    'missionAssignmentCount_relative_event_avg','responseComplexityScore_relative_event_avg',
    'uniqueAgencyCount_relative_event_avg','population2010_relative_event_avg',
    'missionsPerMillionPop','complexityPerMillionPop','missionsPerDay','complexityPerMission'
]


def safe_num(s):
    return pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan)


def ratio(a, b, scale=1.0):
    aa = safe_num(a).fillna(0.0)
    bb = safe_num(b)
    return (aa / bb.replace(0, np.nan) * scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def load_and_engineer_static():
    d = pd.read_excel(MASTER)
    d['target'] = safe_num(d['totalObligatedFunding']).fillna(0).clip(lower=0)
    d['band'] = np.select(
        [(d.target > 50e6) & (d.target <= 200e6), (d.target > 200e6) & (d.target <= 500e6), d.target > 500e6],
        [3,4,5], default=-1
    ).astype(int)
    d['population2010'] = d['state'].map(POP).astype(float)
    d['event_key'] = d['incidentType'].astype(str) + '|' + d['incidentBeginDate'].astype(str) + '|' + d['incidentEndDate'].astype(str)
    g = d.groupby('event_key', dropna=False)
    d['eventSize'] = g['disasterNumber'].transform('size').astype(float)

    # Geographic count descriptor already used in the project. Never use availability/missingness as a feature.
    if SCALE.exists():
        sc = pd.read_csv(SCALE)
        keep = ['disasterNumber'] + [c for c in ['declaredAreaCount','uniquePlaceCodeCount','declarationGeographyRowCount'] if c in sc.columns]
        d = d.merge(sc[keep].drop_duplicates('disasterNumber'), on='disasterNumber', how='left')
    for c in ['declaredAreaCount','uniquePlaceCodeCount','declarationGeographyRowCount']:
        if c not in d.columns:
            d[c] = np.nan

    # Numeric coercion only for safe, non-dollar descriptors.
    safe_sources = [
        'durationDays','declarationDelayDays','expectedResourceScore','missionAssignmentCount','uniqueAgencyCount',
        'uniqueMaTypeCount','uniquePriorityCount','responseComplexityScore','missionDensity','agencyDensity',
        'population2010','eventSize','declaredAreaCount','uniquePlaceCodeCount','declarationGeographyRowCount',
        'ihProgramDeclared','iaProgramDeclared','paProgramDeclared','hmProgramDeclared'
    ]
    for c in safe_sources:
        if c not in d.columns:
            d[c] = 0.0
        d[c] = safe_num(d[c])

    # Time is known at routing time and is not derived from funding.
    fy = safe_num(d['fyDeclared']).fillna(2010)
    d['yearSince2010'] = fy - 2010.0
    d['yearNorm'] = ((fy - 2010.0) / 14.0).clip(0, 2)
    d['yearNormSq'] = d['yearNorm'] ** 2

    # Log transforms make the very heavy-tailed response descriptors more manageable.
    log_map = {
        'logMission':'missionAssignmentCount','logComplexity':'responseComplexityScore','logAgency':'uniqueAgencyCount',
        'logPopulation':'population2010','logDuration':'durationDays','logEventSize':'eventSize',
        'logDeclaredArea':'declaredAreaCount','logPlaceCodes':'uniquePlaceCodeCount',
        'logMissionDensity':'missionDensity','logAgencyDensity':'agencyDensity'
    }
    for new, old in log_map.items():
        d[new] = np.log1p(safe_num(d[old]).fillna(0).clip(lower=0))

    # Target-free within-event context, computed over all declarations sharing the same event key.
    event_sources = ['missionAssignmentCount','responseComplexityScore','uniqueAgencyCount','population2010','durationDays']
    if d['declaredAreaCount'].notna().sum() > 0:
        event_sources.append('declaredAreaCount')
    event_feats = []
    for c in event_sources:
        x = safe_num(d[c]).fillna(0)
        d[c] = x
        d[c+'_event_pct_rank'] = g[c].rank(pct=True, method='average').fillna(.5)
        total = g[c].transform('sum').replace(0, np.nan)
        mx = g[c].transform('max').replace(0, np.nan)
        d[c+'_event_share'] = (x / total).replace([np.inf,-np.inf],np.nan).fillna(0)
        d[c+'_relative_event_avg'] = (d[c+'_event_share'] * d['eventSize']).fillna(0)
        d[c+'_event_max_ratio'] = (x / mx).replace([np.inf,-np.inf],np.nan).fillna(0)
        event_feats += [c+'_event_pct_rank',c+'_event_share',c+'_relative_event_avg',c+'_event_max_ratio']

    # Ratios/intensities: explicitly expose relationships that tiny tree samples may not discover reliably.
    d['missionsPerMillionPop'] = ratio(d.missionAssignmentCount, d.population2010, 1e6)
    d['complexityPerMillionPop'] = ratio(d.responseComplexityScore, d.population2010, 1e6)
    d['agenciesPerMillionPop'] = ratio(d.uniqueAgencyCount, d.population2010, 1e6)
    d['areasPerMillionPop'] = ratio(d.declaredAreaCount, d.population2010, 1e6)
    d['missionsPerArea'] = ratio(d.missionAssignmentCount, d.declaredAreaCount)
    d['complexityPerArea'] = ratio(d.responseComplexityScore, d.declaredAreaCount)
    d['agenciesPerArea'] = ratio(d.uniqueAgencyCount, d.declaredAreaCount)
    d['populationPerArea'] = ratio(d.population2010, d.declaredAreaCount)
    d['missionsPerDay'] = ratio(d.missionAssignmentCount, d.durationDays)
    d['complexityPerDay'] = ratio(d.responseComplexityScore, d.durationDays)
    d['agenciesPerDay'] = ratio(d.uniqueAgencyCount, d.durationDays)
    d['complexityPerMission'] = ratio(d.responseComplexityScore, d.missionAssignmentCount)
    d['agenciesPerMission'] = ratio(d.uniqueAgencyCount, d.missionAssignmentCount)
    d['maTypesPerMission'] = ratio(d.uniqueMaTypeCount, d.missionAssignmentCount)
    d['prioritiesPerMission'] = ratio(d.uniquePriorityCount, d.missionAssignmentCount)
    d['missionsPerAgency'] = ratio(d.missionAssignmentCount, d.uniqueAgencyCount)
    d['complexityPerAgency'] = ratio(d.responseComplexityScore, d.uniqueAgencyCount)
    d['missionsPerMaType'] = ratio(d.missionAssignmentCount, d.uniqueMaTypeCount)
    d['complexityPerPriority'] = ratio(d.responseComplexityScore, d.uniquePriorityCount)
    d['densityRatioMissionAgency'] = ratio(d.missionDensity, d.agencyDensity)
    d['agencyDensityPerMissionDensity'] = ratio(d.agencyDensity, d.missionDensity)

    ratio_feats = [
        'missionsPerMillionPop','complexityPerMillionPop','agenciesPerMillionPop','areasPerMillionPop',
        'missionsPerArea','complexityPerArea','agenciesPerArea','populationPerArea',
        'missionsPerDay','complexityPerDay','agenciesPerDay','complexityPerMission','agenciesPerMission',
        'maTypesPerMission','prioritiesPerMission','missionsPerAgency','complexityPerAgency','missionsPerMaType',
        'complexityPerPriority','densityRatioMissionAgency','agencyDensityPerMissionDensity'
    ]
    for c in ratio_feats:
        d['log1p_'+c] = np.log1p(safe_num(d[c]).fillna(0).clip(lower=0))

    # Explicit interactions/composites.
    interactions = {
        'mission_x_complexity': d.logMission * d.logComplexity,
        'mission_x_population': d.logMission * d.logPopulation,
        'complexity_x_population': d.logComplexity * d.logPopulation,
        'mission_x_area': d.logMission * d.logDeclaredArea,
        'complexity_x_area': d.logComplexity * d.logDeclaredArea,
        'population_x_area': d.logPopulation * d.logDeclaredArea,
        'agency_x_complexity': d.logAgency * d.logComplexity,
        'duration_x_complexity': d.logDuration * d.logComplexity,
        'expected_x_mission': safe_num(d.expectedResourceScore).fillna(0) * d.logMission,
        'expected_x_complexity': safe_num(d.expectedResourceScore).fillna(0) * d.logComplexity,
        'expected_x_population': safe_num(d.expectedResourceScore).fillna(0) * d.logPopulation,
        'expected_x_area': safe_num(d.expectedResourceScore).fillna(0) * d.logDeclaredArea,
        'density_product': safe_num(d.missionDensity).fillna(0) * safe_num(d.agencyDensity).fillna(0),
        'operational_load': (d.logMission * d.logComplexity) / (1.0 + d.logDuration),
        'geographic_operational_load': (d.logMission * d.logComplexity) / (1.0 + d.logDeclaredArea),
        'coordination_load': d.logComplexity * np.log1p(safe_num(d.uniqueAgencyCount).fillna(0)),
        'mission_complexity_event_product': d['missionAssignmentCount_relative_event_avg'] * d['responseComplexityScore_relative_event_avg'],
        'mission_population_event_product': d['missionAssignmentCount_relative_event_avg'] * d['population2010_relative_event_avg'],
        'complexity_population_event_product': d['responseComplexityScore_relative_event_avg'] * d['population2010_relative_event_avg'],
    }
    for c,v in interactions.items():
        d[c] = safe_num(v).fillna(0)

    raw = [c for c in RAW_CANDIDATES if c in d.columns]
    logs = list(log_map.keys())
    ratios = ratio_feats + ['log1p_'+c for c in ratio_feats]
    inter = list(interactions.keys())
    cats = [c for c in CATS if c in d.columns]

    # Never let a target-derived field enter a feature list by accident.
    all_static = list(dict.fromkeys(raw + logs + event_feats + ratios + inter))
    for c in all_static + cats:
        cl = c.lower()
        if any(tok in cl for tok in LEAKAGE_TOKENS):
            raise RuntimeError(f'Leakage-like feature blocked: {c}')

    metadata = {'raw':raw,'logs':logs,'event':event_feats,'ratios':ratios,'interactions':inter,'cats':cats}
    return d, metadata


def robust_group_features(reference, subset, group_col, sources, prefix):
    out = pd.DataFrame(index=subset.index)
    ref = reference.copy()
    ref[group_col] = ref[group_col].astype(str).fillna('MISSING')
    sub_groups = subset[group_col].astype(str).fillna('MISSING')
    for c in sources:
        if c not in ref.columns or c not in subset.columns:
            continue
        rv = safe_num(ref[c])
        sv = safe_num(subset[c]).fillna(0)
        global_vals = rv.dropna().to_numpy(float)
        if len(global_vals) == 0:
            out[f'{prefix}_{c}_rz'] = 0.0; out[f'{prefix}_{c}_pct'] = .5; continue
        gm = float(np.nanmedian(global_vals)); q1 = float(np.nanpercentile(global_vals,25)); q3 = float(np.nanpercentile(global_vals,75)); giqr = max(q3-q1, 1e-6)
        stats = ref.assign(_v=rv).groupby(group_col)['_v'].agg(['median', lambda x: np.nanpercentile(x.dropna(),25) if x.notna().any() else np.nan, lambda x: np.nanpercentile(x.dropna(),75) if x.notna().any() else np.nan, 'count'])
        stats.columns = ['median','q25','q75','count']
        med = sub_groups.map(stats['median']).fillna(gm).astype(float)
        q25 = sub_groups.map(stats['q25']).fillna(q1).astype(float)
        q75 = sub_groups.map(stats['q75']).fillna(q3).astype(float)
        cnt = sub_groups.map(stats['count']).fillna(0).astype(float)
        iqr = (q75-q25).where((q75-q25).abs()>1e-6, giqr)
        med = med.where(cnt>=4, gm); iqr = iqr.where(cnt>=4, giqr)
        out[f'{prefix}_{c}_rz'] = ((sv-med)/iqr).clip(-12,12).fillna(0)

        # Empirical percentile against training-reference values for the same group; fallback to global reference.
        global_sorted = np.sort(global_vals)
        pct = np.zeros(len(subset), float)
        for j,(idx,grp,val) in enumerate(zip(subset.index, sub_groups, sv)):
            arr = safe_num(ref.loc[ref[group_col].astype(str)==grp, c]).dropna().to_numpy(float)
            if len(arr) < 4: arr = global_sorted
            else: arr = np.sort(arr)
            pct[j] = np.searchsorted(arr, float(val), side='right') / max(len(arr),1)
        out[f'{prefix}_{c}_pct'] = pct
    return out


def fold_relative_features(full, train_years, subset_idx):
    ref = full[full['fyDeclared'].astype(int).isin(train_years)].copy()
    sub = full.loc[subset_idx].copy()
    sources = [c for c in NORM_SOURCES if c in full.columns]
    inc = robust_group_features(ref, sub, 'incidentType', sources, 'inc')
    state = robust_group_features(ref, sub, 'state', sources, 'state')

    # A smaller incident+state interaction baseline; only trusted when enough training examples exist.
    ref = ref.copy(); sub = sub.copy()
    ref['_incstate'] = ref['incidentType'].astype(str)+'|'+ref['state'].astype(str)
    sub['_incstate'] = sub['incidentType'].astype(str)+'|'+sub['state'].astype(str)
    incstate_sources = [c for c in ['logMission','logComplexity','logAgency','logPopulation','logDuration'] if c in full.columns]
    incstate = robust_group_features(ref, sub, '_incstate', incstate_sources, 'incstate')
    # Keep robust-z only for interaction group to limit dimensionality.
    incstate = incstate[[c for c in incstate.columns if c.endswith('_rz')]]
    return pd.concat([inc, state, incstate], axis=1)


@dataclass(frozen=True)
class ModelCfg:
    name: str
    family: str
    params: dict


MODELS = [
    ModelCfg('cat_d2_160','cat',{'iterations':160,'depth':2,'learning_rate':.035,'l2_leaf_reg':25}),
    ModelCfg('cat_d3_200','cat',{'iterations':200,'depth':3,'learning_rate':.03,'l2_leaf_reg':30}),
    ModelCfg('cat_d4_220','cat',{'iterations':220,'depth':4,'learning_rate':.025,'l2_leaf_reg':35}),
    ModelCfg('et_d3_l1','et',{'max_depth':3,'min_samples_leaf':1,'max_features':.75}),
    ModelCfg('et_d4_l1','et',{'max_depth':4,'min_samples_leaf':1,'max_features':.75}),
    ModelCfg('et_d5_l2','et',{'max_depth':5,'min_samples_leaf':2,'max_features':.85}),
    ModelCfg('rf_d4_l1','rf',{'max_depth':4,'min_samples_leaf':1,'max_features':.75}),
    ModelCfg('rf_d5_l2','rf',{'max_depth':5,'min_samples_leaf':2,'max_features':.85}),
    ModelCfg('logit_c01','logit',{'C':.1}),
    ModelCfg('logit_c1','logit',{'C':1.0}),
    ModelCfg('logit_c10','logit',{'C':10.0}),
    ModelCfg('svm_rbf_c05','svm',{'C':.5}),
    ModelCfg('svm_rbf_c2','svm',{'C':2.0}),
]

PAIR_MODELS = [m for m in MODELS if m.name in {'cat_d2_160','cat_d3_200','et_d3_l1','et_d4_l1','logit_c1','svm_rbf_c2'}]


def prep_frame(df, num_cols, cat_cols):
    X = df[num_cols + cat_cols].copy()
    for c in num_cols: X[c] = safe_num(X[c])
    for c in cat_cols: X[c] = X[c].astype(str).fillna('MISSING')
    return X


def fit_predict(cfg, Xtr, ytr, Xte, num_cols, cat_cols, proba=False):
    if cfg.family == 'cat':
        X1 = Xtr.copy(); X2 = Xte.copy()
        for c in num_cols:
            med = safe_num(X1[c]).median()
            X1[c] = safe_num(X1[c]).fillna(med if pd.notna(med) else 0)
            X2[c] = safe_num(X2[c]).fillna(med if pd.notna(med) else 0)
        for c in cat_cols:
            X1[c] = X1[c].astype(str).fillna('MISSING'); X2[c] = X2[c].astype(str).fillna('MISSING')
        model = CatBoostClassifier(loss_function='MultiClass' if ytr.nunique()>2 else 'Logloss', auto_class_weights='Balanced', verbose=False, random_seed=42, allow_writing_files=False, **cfg.params)
        model.fit(X1, ytr, cat_features=cat_cols)
        if proba:
            return model.predict_proba(X2), list(model.classes_)
        return np.asarray(model.predict(X2)).reshape(-1), None

    scale = cfg.family in {'logit','svm'}
    num_steps = [('imp',SimpleImputer(strategy='median'))]
    if scale: num_steps.append(('scale',StandardScaler()))
    prep = ColumnTransformer([
        ('num',Pipeline(num_steps),num_cols),
        ('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat_cols),
    ], remainder='drop')
    if cfg.family == 'et':
        model = ExtraTreesClassifier(n_estimators=500,class_weight='balanced',random_state=42,n_jobs=-1,**cfg.params)
    elif cfg.family == 'rf':
        model = RandomForestClassifier(n_estimators=500,class_weight='balanced',random_state=42,n_jobs=-1,**cfg.params)
    elif cfg.family == 'logit':
        model = LogisticRegression(class_weight='balanced',max_iter=5000,solver='lbfgs',**cfg.params)
    else:
        model = SVC(kernel='rbf',class_weight='balanced',probability=True,gamma='scale',**cfg.params)
    pipe = Pipeline([('prep',prep),('model',model)])
    pipe.fit(Xtr,ytr)
    if proba:
        return pipe.predict_proba(Xte), list(pipe.named_steps['model'].classes_)
    return pipe.predict(Xte), None


def feature_sets(meta, fold_cols):
    raw = meta['raw']; logs=meta['logs']; event=meta['event']; ratios=meta['ratios']; inter=meta['interactions']; cats=meta['cats']
    original_event = [c for c in event if c.endswith('_event_pct_rank') or c.endswith('_relative_event_avg')]
    compact_static = list(dict.fromkeys([
        'logPopulation','logMission','logComplexity','logAgency','logDuration','logEventSize','logDeclaredArea',
        'expectedResourceScore','missionDensity','agencyDensity','yearNorm','yearNormSq',
        'missionsPerMillionPop','complexityPerMillionPop','missionsPerArea','complexityPerArea','missionsPerDay',
        'complexityPerDay','complexityPerMission','missionsPerAgency','complexityPerAgency',
        'mission_x_complexity','mission_x_population','complexity_x_population','mission_x_area','complexity_x_area',
        'expected_x_mission','expected_x_complexity','operational_load','geographic_operational_load',
        'missionAssignmentCount_relative_event_avg','responseComplexityScore_relative_event_avg','uniqueAgencyCount_relative_event_avg','population2010_relative_event_avg',
        'missionAssignmentCount_event_pct_rank','responseComplexityScore_event_pct_rank','uniqueAgencyCount_event_pct_rank','population2010_event_pct_rank'
    ]))
    compact_static = [c for c in compact_static if c in (raw+logs+event+ratios+inter)]
    return {
        'raw_context': (list(dict.fromkeys(raw+original_event)), cats),
        'logs_ratios': (list(dict.fromkeys(raw+logs+event+ratios)), cats),
        'interactions': (list(dict.fromkeys(raw+logs+event+ratios+inter)), cats),
        'compact_intensity': (list(dict.fromkeys(compact_static+fold_cols)), cats),
        'full_fold_relative': (list(dict.fromkeys(raw+logs+event+ratios+inter+fold_cols)), cats),
        'full_numeric_no_cats': (list(dict.fromkeys(raw+logs+event+ratios+inter+fold_cols)), []),
    }


def metrics(y, pred):
    rec = recall_score(y,pred,labels=[3,4,5],average=None,zero_division=0)
    return {
        'accuracy':float(accuracy_score(y,pred)),
        'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
        'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
        'r3':float(rec[0]),'r4':float(rec[1]),'r5':float(rec[2]),'min_recall':float(np.min(rec)),
        'pass80':bool(np.min(rec)>=.80),
        'confusion_matrix':confusion_matrix(y,pred,labels=[3,4,5]).tolist(),
    }


def multiclass_screen(d, meta):
    high = d.index[d.band.isin([3,4,5])].to_numpy()
    years_all = sorted(d.loc[high,'fyDeclared'].astype(int).unique())
    pred_store = {(fs,cfg.name):np.full(len(d),-99,int) for fs in ['raw_context','logs_ratios','interactions','compact_intensity','full_fold_relative','full_numeric_no_cats'] for cfg in MODELS}
    feature_counts = {}

    for yr in years_all:
        train_years = [x for x in sorted(d.fyDeclared.astype(int).unique()) if x != yr]
        tr_idx = d.index[(d.band.isin([3,4,5])) & (d.fyDeclared.astype(int)!=yr)]
        te_idx = d.index[(d.band.isin([3,4,5])) & (d.fyDeclared.astype(int)==yr)]
        if len(te_idx)==0: continue
        fold = fold_relative_features(d,train_years,np.concatenate([tr_idx.to_numpy(),te_idx.to_numpy()]))
        for c in fold.columns: d.loc[fold.index,c] = fold[c]
        fs_map = feature_sets(meta,list(fold.columns))
        for fs,(nums,cats) in fs_map.items():
            feature_counts[fs]=(len(nums),len(cats))
            Xtr=prep_frame(d.loc[tr_idx],nums,cats); Xte=prep_frame(d.loc[te_idx],nums,cats); ytr=d.loc[tr_idx,'band'].astype(int)
            for cfg in MODELS:
                try:
                    p,_=fit_predict(cfg,Xtr,ytr,Xte,nums,cats,proba=False)
                    pred_store[(fs,cfg.name)][te_idx]=np.asarray(p,int)
                except Exception as e:
                    print(f'WARN multiclass {yr} {fs} {cfg.name}: {type(e).__name__}: {e}',flush=True)

    y=d.loc[high,'band'].astype(int).to_numpy(); rows=[]; pred_rows=[]
    for (fs,name),pall in pred_store.items():
        p=pall[high]; valid=p!=-99
        if valid.sum()!=len(high): continue
        m=metrics(y,p); rows.append({'feature_set':fs,'model':name,'n_numeric':feature_counts.get(fs,(0,0))[0],'n_categorical':feature_counts.get(fs,(0,0))[1],**{k:v for k,v in m.items() if k!='confusion_matrix'},'confusion_matrix':json.dumps(m['confusion_matrix'])})
        q=d.loc[high,['disasterNumber','state','fyDeclared','incidentType','target','band']].copy(); q['feature_set']=fs; q['model']=name; q['predicted_band']=p; pred_rows.append(q)
    res=pd.DataFrame(rows).sort_values(['min_recall','balanced_accuracy','macro_f1'],ascending=False)
    res.to_csv(OUT/'multiclass_screen_results.csv',index=False)
    if pred_rows: pd.concat(pred_rows,ignore_index=True).to_csv(OUT/'multiclass_oof_predictions.csv',index=False)
    return res


def op_point(y,p):
    fpr,tpr,thr=roc_curve(y,p); tnr=1-fpr
    feasible=np.where((tpr>=.80)&(tnr>=.80))[0]
    cand=feasible if len(feasible) else np.arange(len(thr))
    mins=np.minimum(tpr[cand],tnr[cand]); best=mins.max(); cand2=cand[mins==best]
    j=int(cand2[np.argmax((tpr[cand2]+tnr[cand2])/2)])
    return {'threshold':float(thr[j]),'lower_recall':float(tnr[j]),'upper_recall':float(tpr[j]),'min_recall':float(min(tnr[j],tpr[j])),'auc':float(roc_auc_score(y,p)),'pass80':bool(len(feasible))}


def pairwise_screen(d,meta):
    results=[]; pred_rows=[]
    all_years=sorted(d.fyDeclared.astype(int).unique())
    for fs_name in ['compact_intensity','full_fold_relative']:
        for lo,hi in [(3,4),(4,5),(3,5)]:
            pair_idx=d.index[d.band.isin([lo,hi])].to_numpy(); pair_years=sorted(d.loc[pair_idx,'fyDeclared'].astype(int).unique())
            for cfg in PAIR_MODELS:
                probs=np.full(len(d),np.nan)
                for yr in pair_years:
                    train_years=[x for x in all_years if x!=yr]
                    tr_idx=d.index[(d.band.isin([lo,hi]))&(d.fyDeclared.astype(int)!=yr)]
                    te_idx=d.index[(d.band.isin([lo,hi]))&(d.fyDeclared.astype(int)==yr)]
                    if len(te_idx)==0: continue
                    fold=fold_relative_features(d,train_years,np.concatenate([tr_idx.to_numpy(),te_idx.to_numpy()]))
                    for c in fold.columns: d.loc[fold.index,c]=fold[c]
                    fs_map=feature_sets(meta,list(fold.columns)); nums,cats=fs_map[fs_name]
                    Xtr=prep_frame(d.loc[tr_idx],nums,cats); Xte=prep_frame(d.loc[te_idx],nums,cats); ytr=(d.loc[tr_idx,'band']==hi).astype(int)
                    try:
                        pr,classes=fit_predict(cfg,Xtr,ytr,Xte,nums,cats,proba=True)
                        pos=classes.index(1); probs[te_idx]=pr[:,pos]
                    except Exception as e:
                        print(f'WARN pair {yr} {fs_name} {lo}/{hi} {cfg.name}: {type(e).__name__}: {e}',flush=True)
                yy=(d.loc[pair_idx,'band']==hi).astype(int).to_numpy(); pp=probs[pair_idx]
                if np.isnan(pp).any(): continue
                met=op_point(yy,pp); results.append({'feature_set':fs_name,'pair':f'{lo}vs{hi}','model':cfg.name,**met})
                q=d.loc[pair_idx,['disasterNumber','state','fyDeclared','incidentType','target','band']].copy(); q['feature_set']=fs_name; q['pair']=f'{lo}vs{hi}'; q['model']=cfg.name; q['prob_upper']=pp; pred_rows.append(q)
    res=pd.DataFrame(results).sort_values(['pair','min_recall','auc'],ascending=[True,False,False]); res.to_csv(OUT/'pairwise_screen_results.csv',index=False)
    if pred_rows: pd.concat(pred_rows,ignore_index=True).to_csv(OUT/'pairwise_oof_predictions.csv',index=False)
    return res


def main():
    d,meta=load_and_engineer_static()
    counts={str(b):int((d.band==b).sum()) for b in [3,4,5]}
    print('High counts',counts,flush=True)
    multi=multiclass_screen(d,meta)
    pair=pairwise_screen(d,meta)
    best_multi=multi.head(12).to_dict('records') if len(multi) else []
    best_pairs={p:pair[pair.pair==p].head(6).to_dict('records') for p in ['3vs4','4vs5','3vs5']} if len(pair) else {}
    dev_pass=multi[multi.min_recall>=.80] if len(multi) else pd.DataFrame()
    summary={
        'band_counts':counts,
        'development_multiclass_pass_count':int(len(dev_pass)),
        'best_multiclass':best_multi,
        'best_pairwise':best_pairs,
        'feature_engineering':{
            'logs':meta['logs'],'event_relative_count':len(meta['event']),'ratio_feature_count':len(meta['ratios']),
            'interaction_feature_count':len(meta['interactions']),
            'fold_safe_normalization':'Incident-type and state robust-z/empirical-percentile baselines are fit using only non-held fiscal years. Incident+state robust-z uses the same training-year-only reference.'
        },
        'validation_notes':[
            'Every reported prediction is leave-fiscal-year-out: the entire held fiscal year is excluded from model fitting.',
            'Funding-derived columns are explicitly excluded from all predictors.',
            'Within-event descriptors use only target-free response/exposure variables and the target-free incidentType+begin/end event key.',
            'The geographic count artifact is used only as a value descriptor inside the already-established >50M branch; artifact availability/missingness is never used as a feature.',
            'This is a development screen because feature-set/model selection is judged on the combined leave-year-out OOF predictions. Any apparent >=80% three-band winner must be rerun with fully nested outer/inner year validation before a final claim.'
        ]
    }
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    main()
