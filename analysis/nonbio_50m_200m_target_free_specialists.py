#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
)

import legacy_tempv2 as v2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "audit_outputs" / "nonbio_50m_200m_target_free_specialists"
OUT.mkdir(parents=True, exist_ok=True)

ROUTER_FEATURES = [
    "incidentType",
    "logPopulation2010",
    "logMission",
    "logComplexity",
    "logDuration",
    "logDeclaredArea",
    "expectedResourceScore",
    "missionDensity",
    "agencyDensity",
    "missionAssignmentCount_relative_event_avg",
    "responseComplexityScore_relative_event_avg",
    "uniqueAgencyCount_relative_event_avg",
    "population2010_relative_event_avg",
]

ROUTER_SUPPORTS = {
    "20M-300M": (20e6, 300e6),
    "50M-300M": (50e6, 300e6),
    "20M-200M": (20e6, 200e6),
    "50M-200M": (50e6, 200e6),
}
ROUTER_PARAMS = [
    (1, 150, 10),
    (1, 250, 10),
    (2, 150, 10),
    (2, 250, 10),
    (2, 250, 20),
]
THRESHOLD_GRID = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]


def funding_metrics(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    ly = np.log1p(np.maximum(y, 0))
    lp = np.log1p(np.maximum(p, 0))
    rel = np.abs(p - y) / np.maximum(y, 1.0)
    ratio = np.maximum(p / np.maximum(y, 1.0), y / np.maximum(p, 1.0))
    out = {
        "n": int(len(y)),
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(mean_squared_error(y, p) ** 0.5),
        "MedAE": float(median_absolute_error(y, p)),
        "log_MAE": float(mean_absolute_error(ly, lp)),
        "log_RMSE": float(mean_squared_error(ly, lp) ** 0.5),
        "within_20pct": float(np.mean(rel <= 0.20)),
        "within_30pct": float(np.mean(rel <= 0.30)),
        "within_50pct": float(np.mean(rel <= 0.50)),
        "within_factor2": float(np.mean(ratio <= 2.0)),
        "within_factor3": float(np.mean(ratio <= 3.0)),
    }
    out["R2"] = float(r2_score(y, p)) if len(y) >= 2 and np.var(y) > 0 else None
    out["log_R2"] = float(r2_score(ly, lp)) if len(y) >= 2 and np.var(ly) > 0 else None
    return out


def filter_support(df, support_name):
    lo, hi = ROUTER_SUPPORTS[support_name]
    return df[(df.target > lo) & (df.target <= hi)].copy()


def router_prob(train, test, depth, iterations, l2):
    X = train[ROUTER_FEATURES].copy()
    Xe = test[ROUTER_FEATURES].copy()
    X["incidentType"] = X.incidentType.astype(str)
    Xe["incidentType"] = Xe.incidentType.astype(str)
    y = (train.target > 100e6).astype(int)
    if y.nunique() < 2:
        return np.full(len(test), float(y.iloc[0]))
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    model = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=0.03,
        l2_leaf_reg=l2,
        loss_function="Logloss",
        verbose=False,
        random_seed=42,
        allow_writing_files=False,
        class_weights=[1.0, n0 / max(n1, 1)],
    )
    model.fit(X, y, cat_features=["incidentType"])
    return model.predict_proba(Xe)[:, 1]


def select_router(outer_train):
    years = sorted(outer_train.fyDeclared.astype(int).unique())
    candidate_rows = []
    cached = {}
    for support_name in ROUTER_SUPPORTS:
        for params in ROUTER_PARAMS:
            yy, pp = [], []
            for year in years:
                inner_train_all = outer_train[outer_train.fyDeclared.astype(int) != year]
                train = filter_support(inner_train_all, support_name)
                valid = outer_train[
                    (outer_train.fyDeclared.astype(int) == year)
                    & (outer_train.target > 50e6)
                    & (outer_train.target <= 200e6)
                ].copy()
                if train.empty or valid.empty or (train.target > 100e6).astype(int).nunique() < 2:
                    continue
                prob = router_prob(train, valid, *params)
                yy.extend((valid.target > 100e6).astype(int).tolist())
                pp.extend(prob.tolist())
            key = (support_name, params)
            cached[key] = (np.asarray(yy, int), np.asarray(pp, float))
            if len(set(yy)) == 2:
                ll = float(log_loss(yy, np.clip(pp, 1e-6, 1 - 1e-6), labels=[0, 1]))
                auc = float(roc_auc_score(yy, pp))
            else:
                ll, auc = 999.0, None
            candidate_rows.append(
                {
                    "support": support_name,
                    "params": str(params),
                    "inner_n": int(len(yy)),
                    "inner_logloss": ll,
                    "inner_auc": auc,
                }
            )
    table = pd.DataFrame(candidate_rows).sort_values(["inner_logloss", "support", "params"]).reset_index(drop=True)
    best_row = table.iloc[0]
    best_support = str(best_row.support)
    best_params = eval(str(best_row.params), {"__builtins__": {}})
    y, p = cached[(best_support, best_params)]
    threshold_scores = []
    for t in THRESHOLD_GRID:
        pred = p >= t
        if len(np.unique(y)) < 2:
            bal = 0.0
            acc = float(accuracy_score(y, pred)) if len(y) else 0.0
        else:
            bal = float(balanced_accuracy_score(y, pred))
            acc = float(accuracy_score(y, pred))
        threshold_scores.append({"threshold": t, "balanced_accuracy": bal, "accuracy": acc})
    # Training-only threshold selection: maximize balanced accuracy; ties prefer 0.5.
    threshold_scores = sorted(
        threshold_scores,
        key=lambda z: (-z["balanced_accuracy"], abs(z["threshold"] - 0.5), -z["accuracy"]),
    )
    threshold = float(threshold_scores[0]["threshold"])
    return best_support, best_params, threshold, table, threshold_scores


def fit_model(train, test, features, model_name, target_kind, seed, elo, ehi, weight):
    pre = v2.prep(train[features])
    X = pre.fit_transform(train[features])
    Xe = pre.transform(test[features])
    y = train.target.to_numpy(float)
    yt = np.log1p(y) if target_kind == "log" else y
    sw = np.ones(len(train), float)
    sw[(y > elo) & (y <= ehi)] = weight
    if model_name == "extra":
        model = ExtraTreesRegressor(
            n_estimators=350,
            max_depth=7,
            min_samples_leaf=2,
            max_features=0.65,
            random_state=seed,
            n_jobs=-1,
        )
    elif model_name == "lgbm":
        model = LGBMRegressor(
            n_estimators=250,
            learning_rate=0.035,
            num_leaves=7,
            max_depth=3,
            min_child_samples=5,
            reg_lambda=15,
            reg_alpha=1,
            verbosity=-1,
            random_state=seed,
            n_jobs=-1,
        )
    elif model_name == "ridge":
        model = Ridge(alpha=25.0)
    else:
        raise ValueError(model_name)
    model.fit(X, yt, sample_weight=sw)
    z = model.predict(Xe)
    p = np.expm1(z) if target_kind == "log" else z
    return np.clip(np.maximum(p, 0), elo, ehi)


def lower_candidate(d, features, test, held_year):
    train = d[
        (d.target > 20e6)
        & (d.target <= 200e6)
        & (d.fyDeclared.astype(int) != held_year)
    ].copy()
    p1 = fit_model(train, test, features, "extra", "raw", 1000 + held_year, 50e6, 100e6, 6.0)
    p2 = fit_model(train, test, features, "lgbm", "log", 2000 + held_year, 50e6, 100e6, 6.0)
    return np.clip(0.5 * p1 + 0.5 * p2, 50e6, 100e6)


def upper_candidate(d, features, test, held_year):
    train = d[
        (d.target > 50e6)
        & (d.target <= 300e6)
        & (d.fyDeclared.astype(int) != held_year)
    ].copy()
    p1 = fit_model(train, test, features, "extra", "raw", 3000 + held_year, 100e6, 200e6, 6.0)
    p2 = fit_model(train, test, features, "ridge", "raw", 4000 + held_year, 100e6, 200e6, 6.0)
    return np.clip(0.75 * p1 + 0.25 * p2, 100e6, 200e6)


def training_medians(outer_train):
    broad = outer_train[(outer_train.target > 50e6) & (outer_train.target <= 200e6)]
    lower = outer_train[(outer_train.target > 50e6) & (outer_train.target <= 100e6)]
    upper = outer_train[(outer_train.target > 100e6) & (outer_train.target <= 200e6)]
    return {
        "broad": float(broad.target.median()),
        "lower": float(lower.target.median()),
        "upper": float(upper.target.median()),
    }


def main():
    d, features = v2.load_data()
    d = d[d.incidentType.astype(str).str.lower() != "biological"].copy()
    d["fyDeclared"] = pd.to_numeric(d.fyDeclared, errors="coerce").astype(int)
    ev = d[(d.target > 50e6) & (d.target <= 200e6)].copy().sort_values("disasterNumber")

    rows = []
    inner_tables = []
    threshold_tables = []

    for year in sorted(ev.fyDeclared.unique()):
        outer_train = d[d.fyDeclared != year].copy()
        test = ev[ev.fyDeclared == year].copy()
        support_name, params, threshold, table, threshold_scores = select_router(outer_train)
        table = table.copy()
        table["outer_held_year"] = int(year)
        inner_tables.append(table)
        threshold_tables.append(
            pd.DataFrame(threshold_scores).assign(
                outer_held_year=int(year), selected_support=support_name, selected_params=str(params)
            )
        )

        route_train = filter_support(outer_train, support_name)
        prob = router_prob(route_train, test, *params)
        lower = lower_candidate(d, features, test, year)
        upper = upper_candidate(d, features, test, year)
        med = training_medians(outer_train)

        for k, (_, r) in enumerate(test.iterrows()):
            route_upper = bool(prob[k] >= threshold)
            routed_specialist = float(upper[k] if route_upper else lower[k])
            routed_median = float(med["upper"] if route_upper else med["lower"])
            oracle_specialist = float(upper[k] if r.target > 100e6 else lower[k])
            oracle_median = float(med["upper"] if r.target > 100e6 else med["lower"])
            rows.append(
                {
                    "disasterNumber": int(r.disasterNumber),
                    "state": r.state,
                    "fyDeclared": int(year),
                    "incidentType": r.incidentType,
                    "target": float(r.target),
                    "actual_subband": "100-200M" if r.target > 100e6 else "50-100M",
                    "router_probability_upper": float(prob[k]),
                    "selected_threshold": float(threshold),
                    "router_predicted_subband": "100-200M" if route_upper else "50-100M",
                    "selected_router_support": support_name,
                    "selected_router_params": str(params),
                    "lower_candidate": float(lower[k]),
                    "upper_candidate": float(upper[k]),
                    "routed_specialist_prediction": routed_specialist,
                    "broad_band_median_prediction": float(med["broad"]),
                    "routed_subband_median_prediction": routed_median,
                    "oracle_specialist_prediction": oracle_specialist,
                    "oracle_subband_median_prediction": oracle_median,
                }
            )

    out = pd.DataFrame(rows).sort_values("disasterNumber").reset_index(drop=True)
    out.to_csv(OUT / "predictions.csv", index=False)
    pd.concat(inner_tables, ignore_index=True).to_csv(OUT / "inner_router_candidate_scores.csv", index=False)
    pd.concat(threshold_tables, ignore_index=True).to_csv(OUT / "inner_threshold_scores.csv", index=False)

    y_route = (out.target > 100e6).astype(int).to_numpy()
    pred_route = (out.router_predicted_subband == "100-200M").astype(int).to_numpy()
    prob = out.router_probability_upper.to_numpy(float)

    routed = out.routed_specialist_prediction.to_numpy(float)
    broad_med = out.broad_band_median_prediction.to_numpy(float)
    routed_med = out.routed_subband_median_prediction.to_numpy(float)
    oracle = out.oracle_specialist_prediction.to_numpy(float)
    oracle_med = out.oracle_subband_median_prediction.to_numpy(float)
    target = out.target.to_numpy(float)

    summary = {
        "scope": "non-Biological disasters with 50M < funding <= 200M",
        "n": int(len(out)),
        "actual_subband_counts": out.actual_subband.value_counts().to_dict(),
        "router": {
            "accuracy": float(accuracy_score(y_route, pred_route)),
            "balanced_accuracy": float(balanced_accuracy_score(y_route, pred_route)),
            "auc": float(roc_auc_score(y_route, prob)) if len(np.unique(y_route)) == 2 else None,
            "confusion": pd.crosstab(out.actual_subband, out.router_predicted_subband).to_dict(),
            "misrouted": out.loc[
                out.actual_subband != out.router_predicted_subband,
                [
                    "disasterNumber",
                    "state",
                    "fyDeclared",
                    "target",
                    "router_probability_upper",
                    "selected_threshold",
                    "router_predicted_subband",
                    "actual_subband",
                ],
            ].to_dict("records"),
        },
        "funding": {
            "broad_band_median_baseline": funding_metrics(target, broad_med),
            "routed_subband_median_baseline": funding_metrics(target, routed_med),
            "target_free_routed_specialists": funding_metrics(target, routed),
            "oracle_subband_median_reference": funding_metrics(target, oracle_med),
            "oracle_specialist_reference": funding_metrics(target, oracle),
            "actual_50_100M_routed_specialists": funding_metrics(
                out.loc[out.actual_subband == "50-100M", "target"],
                out.loc[out.actual_subband == "50-100M", "routed_specialist_prediction"],
            ),
            "actual_100_200M_routed_specialists": funding_metrics(
                out.loc[out.actual_subband == "100-200M", "target"],
                out.loc[out.actual_subband == "100-200M", "routed_specialist_prediction"],
            ),
        },
        "protocol": [
            "Biological/COVID rows are excluded from this audit.",
            "Every supervised router and amount fit excludes the entire outer held fiscal year.",
            "Router support window and CatBoost hyperparameters are selected only inside the outer-training data by inner leave-fiscal-year-out log loss.",
            "The routing threshold is selected only inside the outer-training data by inner balanced accuracy; ties prefer 0.5.",
            "No held-out disaster's true funding sub-band is used to choose its routed specialist.",
            "The lower and upper specialist formulas are the preserved fine-split development formulas, but are refit here on non-Biological data with full held-year exclusion.",
            "The oracle metrics are diagnostics only and are not deployable claims.",
            "The specialist formulas were previously developed on this dataset, so this is strict nested development validation rather than a pristine external test.",
        ],
    }

    # Useful error-reduction comparisons.
    bmae = summary["funding"]["broad_band_median_baseline"]["MAE"]
    rmae = summary["funding"]["routed_subband_median_baseline"]["MAE"]
    smae = summary["funding"]["target_free_routed_specialists"]["MAE"]
    summary["comparisons"] = {
        "specialist_MAE_reduction_vs_broad_median_pct": float(100 * (bmae - smae) / bmae),
        "specialist_MAE_reduction_vs_routed_subband_median_pct": float(100 * (rmae - smae) / rmae),
    }

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# Non-Biological 50M-200M target-free specialist audit",
        "",
        f"n = {summary['n']}",
        f"router accuracy = {summary['router']['accuracy']:.4f}",
        f"router balanced accuracy = {summary['router']['balanced_accuracy']:.4f}",
        f"router AUC = {summary['router']['auc']}",
        "",
    ]
    for name, metrics in summary["funding"].items():
        lines.append(f"## {name}")
        for k, val in metrics.items():
            lines.append(f"- {k}: {val}")
        lines.append("")
    lines.append("## comparisons")
    for k, val in summary["comparisons"].items():
        lines.append(f"- {k}: {val}")
    (OUT / "summary.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
