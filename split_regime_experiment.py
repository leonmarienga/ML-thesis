from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

MASTER = Path('master_openfema_40plus.xlsx')
COMP = Path('mission_composition_results/mission_composition_features.csv')
OUT = Path('split_regime_results')
OUT.mkdir(exist_ok=True)

SAFE_BASE = [
    'state','fyDeclared','incidentType','ihProgramDeclared','paProgramDeclared','hmProgramDeclared',
    'durationDays','declarationDelayDays','expectedResourceLevel','expectedResourceScore',
    'disasterCategory','durationClass','missionAssignmentCount','uniqueAgencyCount','uniqueMaTypeCount',
    'uniquePriorityCount','responseComplexityScore','missionDensity','agencyDensity'
]


def metrics(y, p):
    return {
        'R2': float(r2_score(y, p)),
        'MAE': float(mean_absolute_error(y, p)),
        'RMSE': float(mean_squared_error(y, p) ** 0.5),
        'MedAE': float(median_absolute_error(y, p)),
    }


def transform_target(y, kind):
    if kind == 'raw':
        return y
    if kind == 'log':
        return np.log1p(y)
    if kind == 'sqrt':
        return np.sqrt(y)
    if kind == 'p035':
        return np.power(y, 0.35)
    raise ValueError(kind)


def inverse_target(z, kind):
    if kind == 'raw':
        return np.maximum(z, 0)
    if kind == 'log':
        return np.maximum(np.expm1(z), 0)
    if kind == 'sqrt':
        return np.maximum(z, 0) ** 2
    if kind == 'p035':
        return np.maximum(z, 0) ** (1 / 0.35)
    raise ValueError(kind)


def make_pre(X, svd_components=None):
    cat = [c for c in X.columns if not is_numeric_dtype(X[c])]
    num = [c for c in X.columns if c not in cat]
    steps = [
        ('ct', ColumnTransformer([
            ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num),
            ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('oh', OneHotEncoder(handle_unknown='ignore', min_frequency=2))]), cat),
        ]))
    ]
    if svd_components is not None:
        steps.append(('svd', TruncatedSVD(n_components=svd_components, random_state=42)))
    return Pipeline(steps)


def candidate_specs(region):
    # Conservative models because the target regimes are small.
    specs = []
    for target_kind in ['raw', 'log', 'sqrt', 'p035']:
        specs += [
            {'name': f'ridge_svd10_{target_kind}', 'family': 'ridge', 'target': target_kind, 'svd': 10},
            {'name': f'ridge_svd20_{target_kind}', 'family': 'ridge', 'target': target_kind, 'svd': 20},
            {'name': f'extratrees_{target_kind}', 'family': 'extratrees', 'target': target_kind, 'svd': None},
            {'name': f'rf_{target_kind}', 'family': 'rf', 'target': target_kind, 'svd': None},
            {'name': f'xgb_{target_kind}', 'family': 'xgb', 'target': target_kind, 'svd': None},
        ]
        if LGBMRegressor is not None:
            specs.append({'name': f'lgbm_{target_kind}', 'family': 'lgbm', 'target': target_kind, 'svd': None})
        if CatBoostRegressor is not None:
            specs.append({'name': f'catboost_{target_kind}', 'family': 'catboost', 'target': target_kind, 'svd': None})
    return specs


def build_model(family, seed):
    if family == 'ridge':
        return Ridge(alpha=20.0)
    if family == 'extratrees':
        return ExtraTreesRegressor(n_estimators=700, max_depth=7, min_samples_leaf=2, max_features=0.65, random_state=seed, n_jobs=-1)
    if family == 'rf':
        return RandomForestRegressor(n_estimators=700, max_depth=7, min_samples_leaf=2, max_features=0.7, random_state=seed, n_jobs=-1)
    if family == 'xgb':
        return XGBRegressor(n_estimators=600, max_depth=3, learning_rate=0.025, subsample=0.85, colsample_bytree=0.7, reg_lambda=20, reg_alpha=1, objective='reg:squarederror', random_state=seed, n_jobs=-1)
    if family == 'lgbm':
        return LGBMRegressor(n_estimators=450, learning_rate=0.025, num_leaves=7, max_depth=3, min_child_samples=5, reg_lambda=15, reg_alpha=1, verbosity=-1, random_state=seed, n_jobs=-1)
    if family == 'catboost':
        return CatBoostRegressor(iterations=600, depth=4, learning_rate=0.03, loss_function='RMSE', l2_leaf_reg=15, verbose=False, random_seed=seed)
    raise ValueError(family)


def fit_predict(train, test, feature_cols, spec, seed, target_lo, target_hi, target_weight):
    Xtr = train[feature_cols]
    Xte = test[feature_cols]
    ytr = train['target'].to_numpy(float)
    pre = make_pre(Xtr, spec['svd'])
    Xt = pre.fit_transform(Xtr)
    Xe = pre.transform(Xte)
    model = build_model(spec['family'], seed)
    yt = transform_target(ytr, spec['target'])
    w = np.ones(len(train), dtype=float)
    primary = (train['target'].to_numpy(float) > target_lo) & (train['target'].to_numpy(float) <= target_hi)
    w[primary] = target_weight
    try:
        model.fit(Xt, yt, sample_weight=w)
    except TypeError:
        model.fit(Xt, yt)
    pred = inverse_target(model.predict(Xe), spec['target'])
    return pred


def run_lower(data, feature_cols):
    # Evaluate 50-200M. Broader windows provide support but evaluation remains only on the 39 target cases.
    eval_mask = (data.target > 50e6) & (data.target <= 200e6)
    eval_data = data.loc[eval_mask].copy().reset_index(drop=True)
    strat = (eval_data.target > 100e6).astype(int).to_numpy()
    supports = [
        ('20-300M', 20e6, 300e6),
        ('10-300M', 10e6, 300e6),
        ('20-500M', 20e6, 500e6),
        ('1-300M', 1e6, 300e6),
        ('50-500M', 50e6, 500e6),
    ]
    specs = candidate_specs('lower')
    seeds = [42, 123, 777]
    result_rows = []
    pred_store = {}

    for sname, slo, shi in supports:
        for tw in [1.0, 2.0, 4.0]:
            for spec in specs:
                repeated = []
                for seed in seeds:
                    pred = np.zeros(len(eval_data))
                    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                    for tr_idx, te_idx in folds.split(eval_data, strat):
                        te = eval_data.iloc[te_idx]
                        held_ids = set(te.disasterNumber.astype(int))
                        train = data[(data.target > slo) & (data.target <= shi) & (~data.disasterNumber.astype(int).isin(held_ids))].copy()
                        pred[te_idx] = fit_predict(train, te, feature_cols, spec, seed, 50e6, 200e6, tw)
                    repeated.append(pred)
                avg = np.mean(repeated, axis=0)
                key = f'{sname}|w{tw:g}|{spec["name"]}'
                pred_store[key] = avg
                result_rows.append({'region':'50-200M','support':sname,'weight':tw,'model':spec['name'], **metrics(eval_data.target, avg)})

    # Development blend screen among top 8 single models.
    res = pd.DataFrame(result_rows).sort_values('R2', ascending=False).reset_index(drop=True)
    top_keys = [f'{r.support}|w{r.weight:g}|{r.model}' for _, r in res.head(8).iterrows()]
    blends = []
    for i in range(len(top_keys)):
        for j in range(i+1, len(top_keys)):
            for w in [0.25,0.5,0.75]:
                p = w*pred_store[top_keys[i]] + (1-w)*pred_store[top_keys[j]]
                blends.append({'region':'50-200M','support':'blend','weight':w,'model':f'{w:.2f}*{top_keys[i]} + {1-w:.2f}*{top_keys[j]}', **metrics(eval_data.target, p), '_pred':p})
    if blends:
        best_blend = max(blends, key=lambda x:x['R2'])
        res = pd.concat([res, pd.DataFrame([{k:v for k,v in best_blend.items() if k!='_pred'}])], ignore_index=True).sort_values('R2',ascending=False)
        pred_store['BEST_BLEND'] = best_blend['_pred']
    best = res.iloc[0].to_dict()
    if best['support']=='blend':
        best_pred = pred_store['BEST_BLEND']
    else:
        best_pred = pred_store[f"{best['support']}|w{best['weight']:g}|{best['model']}"]
    outp = eval_data[['disasterNumber','target']].copy()
    outp['prediction'] = best_pred
    return res, outp, best


def run_upper(data, feature_cols):
    # Evaluate 200-500M using leave-one-disaster-out. Training may use broader high-value support.
    eval_mask = (data.target > 200e6) & (data.target <= 500e6)
    eval_data = data.loc[eval_mask].copy().reset_index(drop=True)
    supports = [
        ('100M-1B', 100e6, 1e9),
        ('50M-1B', 50e6, 1e9),
        ('100M+', 100e6, np.inf),
        ('50M+', 50e6, np.inf),
        ('150M+', 150e6, np.inf),
    ]
    specs = candidate_specs('upper')
    result_rows = []
    pred_store = {}
    seed = 42

    for sname, slo, shi in supports:
        for tw in [1.0, 2.0, 4.0, 8.0]:
            for spec in specs:
                pred = np.zeros(len(eval_data))
                for i in range(len(eval_data)):
                    te = eval_data.iloc[[i]]
                    did = int(te.disasterNumber.iloc[0])
                    train = data[(data.target > slo) & (data.target <= shi) & (data.disasterNumber.astype(int) != did)].copy()
                    pred[i] = fit_predict(train, te, feature_cols, spec, seed+i, 200e6, 500e6, tw)[0]
                key = f'{sname}|w{tw:g}|{spec["name"]}'
                pred_store[key] = pred
                result_rows.append({'region':'200-500M','support':sname,'weight':tw,'model':spec['name'], **metrics(eval_data.target, pred)})

    res = pd.DataFrame(result_rows).sort_values('R2', ascending=False).reset_index(drop=True)
    top_keys = [f'{r.support}|w{r.weight:g}|{r.model}' for _, r in res.head(10).iterrows()]
    blends = []
    for i in range(len(top_keys)):
        for j in range(i+1, len(top_keys)):
            for w in [0.25,0.5,0.75]:
                p = w*pred_store[top_keys[i]] + (1-w)*pred_store[top_keys[j]]
                blends.append({'region':'200-500M','support':'blend','weight':w,'model':f'{w:.2f}*{top_keys[i]} + {1-w:.2f}*{top_keys[j]}', **metrics(eval_data.target, p), '_pred':p})
    if blends:
        best_blend = max(blends, key=lambda x:x['R2'])
        res = pd.concat([res, pd.DataFrame([{k:v for k,v in best_blend.items() if k!='_pred'}])], ignore_index=True).sort_values('R2',ascending=False)
        pred_store['BEST_BLEND'] = best_blend['_pred']
    best = res.iloc[0].to_dict()
    if best['support']=='blend':
        best_pred = pred_store['BEST_BLEND']
    else:
        best_pred = pred_store[f"{best['support']}|w{best['weight']:g}|{best['model']}"]
    outp = eval_data[['disasterNumber','target']].copy()
    outp['prediction'] = best_pred
    return res, outp, best


def main():
    master = pd.read_excel(MASTER)
    master['target'] = pd.to_numeric(master['totalObligatedFunding'], errors='coerce').fillna(0).clip(lower=0)
    comp = pd.read_csv(COMP)

    # Compact mission-composition features reduce the 752 raw columns to robust summary statistics,
    # plus recurring category shares. This avoids fitting 770 independent parameters to tiny bands.
    compact_cols = [c for c in comp.columns if c.endswith('_entropy') or c.endswith('_hhi') or c.endswith('_top_share')]
    share_cols = [c for c in comp.columns if '_share__' in c]
    # Keep only share features present in at least 5% of disasters, using nonzero prevalence.
    keep_share = [c for c in share_cols if (pd.to_numeric(comp[c], errors='coerce').fillna(0) != 0).mean() >= 0.05]
    comp_use = comp[['disasterNumber'] + compact_cols + keep_share].copy()

    data = master[['disasterNumber','target'] + SAFE_BASE].merge(comp_use, on='disasterNumber', how='left').fillna(0)
    feature_cols = [c for c in data.columns if c not in ['disasterNumber','target']]

    lower_res, lower_pred, lower_best = run_lower(data, feature_cols)
    upper_res, upper_pred, upper_best = run_upper(data, feature_cols)

    lower_res.to_csv(OUT/'lower_50_200_screen.csv', index=False)
    upper_res.to_csv(OUT/'upper_200_500_screen.csv', index=False)
    lower_pred.to_csv(OUT/'lower_50_200_best_predictions.csv', index=False)
    upper_pred.to_csv(OUT/'upper_200_500_best_predictions.csv', index=False)

    summary = {
        'counts': {
            '50_200M': int(((data.target>50e6)&(data.target<=200e6)).sum()),
            '200_500M': int(((data.target>200e6)&(data.target<=500e6)).sum()),
            '500M_plus': int((data.target>500e6).sum()),
        },
        'feature_count': len(feature_cols),
        'lower_best_development': lower_best,
        'upper_best_development': upper_best,
        'notes': [
            '50-200M uses repeated 5-fold OOF evaluation; support windows may include neighboring funding ranges but held-out target cases are excluded from training.',
            '200-500M uses leave-one-disaster-out evaluation because only nine target cases exist.',
            'Model/support/weight/blend selection is a development screen on the same target cases and is not final unbiased validation.',
            'Actual funding is used only to define training/evaluation regimes, never as a predictor feature. A deployable system still requires a target-free router.'
        ]
    }
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
