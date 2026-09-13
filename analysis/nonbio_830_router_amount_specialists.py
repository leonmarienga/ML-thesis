#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

import legacy_tempv2 as v2

ROOT = Path(__file__).resolve().parents[1]
ACCEPTED = ROOT / "audit_inputs" / "post830_amount_specialists" / "accepted" / "candidate_predictions.csv"
OUT = ROOT / "audit_outputs" / "nonbio_830_router_amount_specialists"
OUT.mkdir(parents=True, exist_ok=True)

BANDS = ["0-100K", "100K-1M", "1M-50M", "50-200M", "200-500M", "500M+"]
BOUNDS = {
    "0-100K": (0.0, 1e5),
    "100K-1M": (1e5, 1e6),
    "1M-50M": (1e6, 50e6),
    "50-200M": (50e6, 200e6),
    "200-500M": (200e6, 500e6),
    "500M+": (500e6, np.inf),
}

SUBROUTER_FEATURES = [
    "incidentType", "logPopulation2010", "logMission", "logComplexity", "logDuration",
    "logDeclaredArea", "expectedResourceScore", "missionDensity", "agencyDensity",
    "missionAssignmentCount_relative_event_avg", "responseComplexityScore_relative_event_avg",
    "uniqueAgencyCount_relative_event_avg", "population2010_relative_event_avg",
]
SUBROUTER_CANDIDATES = [(1,150,10),(1,250,10),(2,150,10),(2,250,10),(2,250,20)]
FAMILY_SUPPORT_THRESHOLD = 3

ANALOG_FEATURES = [
    "logPopulation2010", "logMission", "logComplexity", "logDuration", "logDeclaredArea",
    "expectedResourceScore", "missionDensity", "agencyDensity",
]


def funding_metrics(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    ly = np.log1p(np.maximum(y, 0))
    lp = np.log1p(np.maximum(p, 0))
    out = {
        "n": int(len(y)),
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(mean_squared_error(y, p) ** 0.5),
        "MedAE": float(median_absolute_error(y, p)),
        "log_MAE": float(mean_absolute_error(ly, lp)),
        "log_RMSE": float(mean_squared_error(ly, lp) ** 0.5),
    }
    out["R2"] = float(r2_score(y, p)) if len(y) >= 2 and np.var(y) > 0 else None
    out["log_R2"] = float(r2_score(ly, lp)) if len(y) >= 2 and np.var(ly) > 0 else None
    return out


def fit_preprocessed_regressor(train, test, features, model, target_kind="log", sample_weight=None):
    pre = v2.prep(train[features])
    X = pre.fit_transform(train[features])
    Xe = pre.transform(test[features])
    y = train.target.to_numpy(float)
    yt = np.log1p(y) if target_kind == "log" else y
    try:
        model.fit(X, yt, sample_weight=sample_weight)
    except TypeError:
        model.fit(X, yt)
    z = model.predict(Xe)
    return np.maximum(np.expm1(z) if target_kind == "log" else z, 0.0)


def predict_generic_band(train, test, band, features, seed):
    if len(test) == 0:
        return np.array([], float)
    lo, hi = BOUNDS[band]

    if band == "0-100K":
        tr = train[(train.target >= 0) & (train.target <= 1e5)].copy()
        pre = v2.prep(tr[features])
        X = pre.fit_transform(tr[features])
        Xe = pre.transform(test[features])
        funded = (tr.target > 0).astype(int).to_numpy()
        if len(np.unique(funded)) < 2:
            prob = np.full(len(test), float(funded[0]))
        else:
            clf = ExtraTreesClassifier(
                n_estimators=350, max_depth=8, min_samples_leaf=2, max_features=.7,
                class_weight="balanced", random_state=seed, n_jobs=-1,
            )
            clf.fit(X, funded)
            prob = clf.predict_proba(Xe)[:, 1]
        pos = tr[tr.target > 0].copy()
        if len(pos) < 3:
            amount = np.full(len(test), float(pos.target.median()) if len(pos) else 0.0)
        else:
            amount = fit_preprocessed_regressor(
                pos, test, features,
                ExtraTreesRegressor(n_estimators=400, max_depth=8, min_samples_leaf=2, max_features=.7, random_state=seed+1, n_jobs=-1),
                "log",
            )
        return np.where(prob >= .5, np.clip(amount, 1.0, hi), 0.0)

    if band in ("100K-1M", "1M-50M"):
        tr = train[(train.target > lo) & (train.target <= hi)].copy()
        if len(tr) < 4:
            return np.full(len(test), float(np.clip(tr.target.median() if len(tr) else lo, lo, hi)))
        p1 = fit_preprocessed_regressor(
            tr, test, features,
            ExtraTreesRegressor(n_estimators=450, max_depth=9, min_samples_leaf=2, max_features=.7, random_state=seed, n_jobs=-1),
            "log",
        )
        p2 = fit_preprocessed_regressor(tr, test, features, Ridge(alpha=25.0), "log")
        w = .75 if band == "1M-50M" else .65
        return np.clip(w*p1 + (1-w)*p2, lo, hi)

    if band == "50-200M":
        # Preserved high-value specialist pattern: broad support with heavy target-band weighting.
        tr = train[(train.target > 20e6) & (train.target <= 300e6)].copy()
        if len(tr) < 5:
            return np.full(len(test), float(np.clip(tr.target.median() if len(tr) else 100e6, lo, hi)))
        y = tr.target.to_numpy(float)
        sw = np.ones(len(tr)); sw[(y > 50e6) & (y <= 200e6)] = 6.0
        p1 = fit_preprocessed_regressor(
            tr, test, features,
            ExtraTreesRegressor(n_estimators=450, max_depth=7, min_samples_leaf=2, max_features=.65, random_state=seed, n_jobs=-1),
            "raw", sw,
        )
        p2 = fit_preprocessed_regressor(
            tr, test, features,
            LGBMRegressor(n_estimators=260, learning_rate=.035, num_leaves=7, max_depth=3, min_child_samples=5, reg_lambda=15, reg_alpha=1, verbosity=-1, random_state=seed, n_jobs=-1),
            "log", sw,
        )
        return np.clip(.5*p1 + .5*p2, lo, hi)

    raise ValueError(f"generic specialist not defined for {band}")


def subrouter_prob(train, test, depth, iterations, l2):
    X = train[SUBROUTER_FEATURES].copy()
    Xe = test[SUBROUTER_FEATURES].copy()
    X["incidentType"] = X.incidentType.astype(str)
    Xe["incidentType"] = Xe.incidentType.astype(str)
    y = (train.target > 300e6).astype(int)
    n0 = int((y == 0).sum()); n1 = int((y == 1).sum())
    if y.nunique() < 2:
        return np.full(len(test), float(y.iloc[0]))
    model = CatBoostClassifier(
        iterations=iterations, depth=depth, learning_rate=.03, l2_leaf_reg=l2,
        loss_function="Logloss", verbose=False, random_seed=42, allow_writing_files=False,
        class_weights=[1, n0/max(n1,1)],
    )
    model.fit(X, y, cat_features=["incidentType"])
    return model.predict_proba(Xe)[:, 1]


def select_subrouter(train_support):
    years = sorted(train_support.fyDeclared.astype(int).unique())
    scores = {}
    for cand in SUBROUTER_CANDIDATES:
        yy, pp = [], []
        for year in years:
            tr = train_support[train_support.fyDeclared.astype(int) != year]
            va = train_support[train_support.fyDeclared.astype(int) == year]
            if tr.empty or va.empty or (tr.target > 300e6).astype(int).nunique() < 2:
                continue
            p = subrouter_prob(tr, va, *cand)
            yy.extend((va.target > 300e6).astype(int).tolist())
            pp.extend(p.tolist())
        if len(set(yy)) == 2:
            scores[cand] = float(log_loss(yy, np.clip(pp, 1e-6, 1-1e-6), labels=[0,1]))
    if not scores:
        return (2,250,10), {}
    return min(scores, key=scores.get), scores


def mid_high_candidates(d, features, e, row, year):
    one = d.loc[[row.name]]
    support = d[(d.target > 100e6) & (d.target <= 1e9) & (d.fyDeclared.astype(int) != year)]
    same = int((support.incidentType.astype(str) == str(row.incidentType)).sum())
    if same >= FAMILY_SUPPORT_THRESHOLD:
        lower = float(.5*v2.lower_direct(d, features, one, year, 9000+year)[0] + .5*v2.lower_position(d, features, one, year, 9000+year)[0])
    else:
        lower = float(v2.unseen_family_lower(d, one, year)[0][0])
    if same >= FAMILY_SUPPORT_THRESHOLD:
        upper = float(v2.seen_family_upper(d, one, year, 9000+year)[0])
    else:
        upper = float(v2.unseen_event_upper(d, e, row, year)[0])
    return lower, upper


def predict_200_500(d, features, test, year):
    if len(test) == 0:
        return np.array([], float), []
    train_support = d[(d.target > 100e6) & (d.target <= 1e9) & (d.fyDeclared.astype(int) != year)].copy()
    cand, _ = select_subrouter(train_support)
    prob = subrouter_prob(train_support, test, *cand)
    e = v2.event_table(d)
    out, methods = [], []
    for k, (_, r) in enumerate(test.iterrows()):
        lower, upper = mid_high_candidates(d, features, e, r, year)
        if prob[k] >= .5:
            out.append(upper); methods.append("target_free_300_500_specialist")
        else:
            out.append(lower); methods.append("target_free_200_300_specialist")
    return np.clip(np.asarray(out, float), 200e6, 500e6), methods


def extreme_direct(train, test, features, seed):
    tr = train[train.target > 200e6].copy()
    if len(tr) < 4:
        return np.full(len(test), max(500e6, float(tr.target.median()) if len(tr) else 1e9))
    pre = v2.prep(tr[features])
    X = pre.fit_transform(tr[features]); Xe = pre.transform(test[features])
    y = np.log1p(tr.target.to_numpy(float))
    sw = np.ones(len(tr)); sw[tr.target.to_numpy(float) > 500e6] = 6.0
    model = LGBMRegressor(
        objective="quantile", alpha=.75, n_estimators=320, learning_rate=.03,
        num_leaves=7, max_depth=3, min_child_samples=3, reg_lambda=20, reg_alpha=1,
        verbosity=-1, random_state=seed, n_jobs=-1,
    )
    model.fit(X, y, sample_weight=sw)
    return np.clip(np.expm1(model.predict(Xe)), 500e6, 6e9)


def extreme_analogue(train, test):
    ref = train[train.target > 500e6].copy()
    if len(ref) == 0:
        ref = train[train.target > 200e6].copy()
    vals = []
    for _, row in test.iterrows():
        pool = ref[ref.incidentType.astype(str) == str(row.incidentType)]
        if len(pool) < 1:
            pool = ref
        X = pool[ANALOG_FEATURES].replace([np.inf,-np.inf], np.nan).fillna(0).astype(float)
        q = row[ANALOG_FEATURES].replace([np.inf,-np.inf], np.nan).fillna(0).astype(float)
        mu = X.mean(axis=0); sd = X.std(axis=0).replace(0,1).fillna(1)
        dist = np.sqrt((((X-mu)/sd - (q-mu)/sd)**2).mean(axis=1)).to_numpy(float)
        order = np.argsort(dist)[:min(2, len(pool))]
        chosen = pool.iloc[order]
        weights = 1.0 / np.maximum(dist[order], .10)
        base = np.average(np.log1p(chosen.target.to_numpy(float)), weights=weights)
        # Historical analogue scale adjustment using response intensity only.
        cand_m = np.maximum(chosen.missionAssignmentCount.to_numpy(float), 1)
        cand_c = np.maximum(chosen.responseComplexityScore.to_numpy(float), 1)
        rm = max(float(row.missionAssignmentCount),1) / cand_m
        rc = max(float(row.responseComplexityScore),1) / cand_c
        scale = np.clip(np.sqrt(rm*rc), .5, 2.0)
        scaled_logs = np.log1p(chosen.target.to_numpy(float) * np.sqrt(scale))
        val = float(np.expm1(np.average(scaled_logs, weights=weights)))
        vals.append(np.clip(val, 500e6, 6e9))
    return np.asarray(vals, float)


def select_extreme_blend(train, features):
    extremes = train[train.target > 500e6].copy()
    if len(extremes) < 3:
        return .5
    rows = []
    for _, r in extremes.iterrows():
        yr = int(r.fyDeclared)
        inner = train[train.fyDeclared.astype(int) != yr].copy()
        te = train.loc[[r.name]]
        if (inner.target > 500e6).sum() < 1:
            continue
        d = extreme_direct(inner, te, features, 7000+yr)[0]
        a = extreme_analogue(inner, te)[0]
        rows.append((float(r.target), d, a))
    if len(rows) < 2:
        return .5
    y = np.array([x[0] for x in rows]); direct = np.array([x[1] for x in rows]); analog = np.array([x[2] for x in rows])
    grid = [.25,.5,.75]
    return min(grid, key=lambda w: mean_absolute_error(y, w*direct + (1-w)*analog))


def predict_500_plus(train, test, features, year):
    if len(test) == 0:
        return np.array([], float), .5
    w = select_extreme_blend(train, features)
    direct = extreme_direct(train, test, features, 8000+year)
    analog = extreme_analogue(train, test)
    return np.clip(w*direct + (1-w)*analog, 500e6, 6e9), w


def predict_band(d, features, train, test, band, year):
    if len(test) == 0:
        return np.array([], float), []
    if band in ("0-100K", "100K-1M", "1M-50M", "50-200M"):
        p = predict_generic_band(train, test, band, features, 1000+year+BANDS.index(band)*31)
        return p, [f"{band}_amount_specialist"]*len(test)
    if band == "200-500M":
        return predict_200_500(d, features, test, year)
    if band == "500M+":
        p, w = predict_500_plus(train, test, features, year)
        return p, [f"reconstructed_500Mplus_hybrid_w{w:.2f}"]*len(test)
    raise ValueError(band)


def main():
    accepted = pd.read_csv(ACCEPTED)
    accepted.disasterNumber = accepted.disasterNumber.astype(int)
    pred_col = "candidate_pred_new" if "candidate_pred_new" in accepted.columns else "candidate_pred"
    assert len(accepted) == 912
    assert int((accepted[pred_col].astype(str) == accepted.actual_band.astype(str)).sum()) == 830

    d, features = v2.load_data()
    d.disasterNumber = d.disasterNumber.astype(int)
    d = d[d.disasterNumber.isin(set(accepted.disasterNumber))].copy()
    d = d.merge(accepted[["disasterNumber", "actual_band", pred_col]], on="disasterNumber", how="inner", validate="one_to_one")
    d = d.rename(columns={pred_col:"router_pred"})
    d.router_pred = d.router_pred.astype(str); d.actual_band = d.actual_band.astype(str)
    assert len(d) == 912
    assert not (d.incidentType.astype(str).str.lower() == "biological").any()

    pred_router = pd.Series(index=d.index, dtype=float)
    pred_oracle = pd.Series(index=d.index, dtype=float)
    pred_median = pd.Series(index=d.index, dtype=float)
    pred_oracle_median = pd.Series(index=d.index, dtype=float)
    method_router = pd.Series(index=d.index, dtype=object)

    fold_rows = []
    for year in sorted(d.fyDeclared.astype(int).unique()):
        print(f"OUTER FY {year}", flush=True)
        train = d[d.fyDeclared.astype(int) != year].copy()
        test = d[d.fyDeclared.astype(int) == year].copy()

        medians = {}
        for band in BANDS:
            vals = train.loc[train.actual_band == band, "target"]
            medians[band] = float(vals.median()) if len(vals) else float(np.median(train.target))
        for idx, r in test.iterrows():
            pred_median.loc[idx] = medians[str(r.router_pred)]
            pred_oracle_median.loc[idx] = medians[str(r.actual_band)]

        for band in BANDS:
            ridx = test.index[test.router_pred == band]
            if len(ridx):
                p, methods = predict_band(d, features, train, d.loc[ridx], band, year)
                pred_router.loc[ridx] = p
                method_router.loc[ridx] = methods
            oidx = test.index[test.actual_band == band]
            if len(oidx):
                p, _ = predict_band(d, features, train, d.loc[oidx], band, year)
                pred_oracle.loc[oidx] = p

        fold_rows.append({
            "outer_fy": int(year), "n": int(len(test)),
            "router_correct": int((test.router_pred == test.actual_band).sum()),
            "router_accuracy": float((test.router_pred == test.actual_band).mean()),
        })

    for s in [pred_router, pred_oracle, pred_median, pred_oracle_median]:
        if s.isna().any():
            raise AssertionError(f"missing predictions: {int(s.isna().sum())}")

    out = d[["disasterNumber","state","fyDeclared","incidentType","target","actual_band","router_pred"]].copy()
    out["router_band_median_prediction"] = pred_median
    out["oracle_band_median_prediction"] = pred_oracle_median
    out["router_plus_specialist_prediction"] = pred_router
    out["oracle_band_specialist_prediction"] = pred_oracle
    out["router_specialist_method"] = method_router
    out["router_correct"] = out.router_pred == out.actual_band
    out["abs_error"] = np.abs(out.target - out.router_plus_specialist_prediction)
    out["log_abs_error"] = np.abs(np.log1p(out.target) - np.log1p(out.router_plus_specialist_prediction))
    out = out.sort_values("disasterNumber").reset_index(drop=True)

    y = out.target.to_numpy(float)
    summary = {
        "router_band_accuracy": {"correct":830,"n":912,"accuracy":830/912},
        "router_band_median_baseline": funding_metrics(y, out.router_band_median_prediction),
        "router_plus_amount_specialists": funding_metrics(y, out.router_plus_specialist_prediction),
        "oracle_band_median_ceiling": funding_metrics(y, out.oracle_band_median_prediction),
        "oracle_band_specialist_ceiling": funding_metrics(y, out.oracle_band_specialist_prediction),
        "correctly_routed_subset": funding_metrics(out.loc[out.router_correct,"target"], out.loc[out.router_correct,"router_plus_specialist_prediction"]),
        "misrouted_subset": funding_metrics(out.loc[~out.router_correct,"target"], out.loc[~out.router_correct,"router_plus_specialist_prediction"]),
        "per_actual_band": {},
        "notes": [
            "The accepted 830/912 router predictions are frozen and never modified by the amount specialists.",
            "Every amount specialist excludes the entire held-out fiscal year from supervised fitting.",
            "The 200M-500M amount stage reproduces the preserved target-free 200M-300M versus 300M-500M sub-router pattern and temporal specialist functions on non-Biological rows only.",
            "The 500M+ code was not preserved on the branch; it is explicitly reconstructed as high-quantile LightGBM plus a response-intensity historical analogue, with its blend selected only inside outer-training data.",
            "Oracle-band results use the true six-band label only to choose which amount specialist to invoke and are a ceiling diagnostic, not deployable performance.",
        ],
    }
    for band in BANDS:
        m = out.actual_band == band
        summary["per_actual_band"][band] = funding_metrics(out.loc[m,"target"], out.loc[m,"router_plus_specialist_prediction"])

    out.to_csv(OUT/"predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT/"fold_diagnostics.csv", index=False)
    (OUT/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    s = summary["router_plus_amount_specialists"]
    bm = summary["router_band_median_baseline"]
    oc = summary["oracle_band_specialist_ceiling"]
    lines = [
        "# Frozen 830 router + amount specialists",
        "",
        "## Routing stage",
        "- Frozen router: **830/912 = 91.01%**",
        "",
        "## End-to-end dollar prediction",
        f"- Router-band median baseline: R² **{bm['R2']:.4f}**, MAE **${bm['MAE']:,.0f}**, RMSE **${bm['RMSE']:,.0f}**, log MAE **{bm['log_MAE']:.4f}**",
        f"- Router + amount specialists: R² **{s['R2']:.4f}**, MAE **${s['MAE']:,.0f}**, RMSE **${s['RMSE']:,.0f}**, log MAE **{s['log_MAE']:.4f}**",
        f"- Oracle-band specialist ceiling: R² **{oc['R2']:.4f}**, MAE **${oc['MAE']:,.0f}**, RMSE **${oc['RMSE']:,.0f}**, log MAE **{oc['log_MAE']:.4f}**",
        "",
        "## Per actual band (router + specialists)",
    ]
    for band in BANDS:
        q = summary["per_actual_band"][band]
        r2 = "NA" if q["R2"] is None else f"{q['R2']:.4f}"
        lines.append(f"- {band}: n={q['n']}, R² **{r2}**, MAE **${q['MAE']:,.0f}**, RMSE **${q['RMSE']:,.0f}**, log MAE **{q['log_MAE']:.4f}**")
    (OUT/"summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
