#!/usr/bin/env python3
"""
Dedicated strict nested-LFYO amount specialist audit for the non-Biological
$100K-$1M funding band.

For each outer fiscal year, the entire year is held out. Inside the remaining
training years, candidate regressors and a nonzero shrinkage weight are selected
using inner LFYO. The selected specialist is then refit on all outer-training
$100K-$1M cases and applied to the untouched outer year.

The frozen 830/912 router is never modified.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

from mission_semantic_audit import (
    CURRENT_19,
    build_semantic_rollup,
    fetch_all_mission_assignments,
    normalize_master,
    normalize_model_frame,
    prep_pipeline,
)
from nonbio_all_ranges import valid_cols

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
ACCEPTED = ROOT / "audit_inputs" / "100k_1m_ml_specialist" / "accepted" / "candidate_predictions.csv"
OUT = ROOT / "audit_outputs" / "nonbio_100k_1m_dedicated_ml_specialist"
OUT.mkdir(parents=True, exist_ok=True)

LO = 100_000.0
HI = 1_000_000.0
BAND = "100K-1M"
SHRINK_WEIGHTS = [0.25, 0.50, 0.75, 1.00]


def metric_block(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if len(y) == 0:
        return {"n": 0}
    ae = np.abs(p - y)
    ly = np.log1p(y)
    lp = np.log1p(np.maximum(p, 0))
    rel = ae / np.maximum(y, 1.0)
    ratio = np.maximum(p / np.maximum(y, 1.0), y / np.maximum(p, 1.0))
    return {
        "n": int(len(y)),
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(mean_squared_error(y, p) ** 0.5),
        "MedAE": float(median_absolute_error(y, p)),
        "R2": float(r2_score(y, p)) if len(y) >= 2 and float(np.var(y)) > 0 else None,
        "log_MAE": float(mean_absolute_error(ly, lp)),
        "log_RMSE": float(mean_squared_error(ly, lp) ** 0.5),
        "within_20pct": float(np.mean(rel <= 0.20)),
        "within_30pct": float(np.mean(rel <= 0.30)),
        "within_factor2": float(np.mean(ratio <= 2.0)),
        "unique_predictions_rounded_dollar": int(len(np.unique(np.round(p, 0)))),
    }


def clip_band(p):
    return np.clip(np.asarray(p, float), LO, HI - 1.0)


def model_specs():
    return {
        "et_log_sem": {"features": "sem", "target": "log", "kind": "et"},
        "et_raw_sem": {"features": "sem", "target": "raw", "kind": "et"},
        "et_log_current": {"features": "current", "target": "log", "kind": "et"},
        "rf_log_sem": {"features": "sem", "target": "log", "kind": "rf"},
        "rf_raw_sem": {"features": "sem", "target": "raw", "kind": "rf"},
        "lgbm_log_sem": {"features": "sem", "target": "log", "kind": "lgbm"},
        "ridge_log_sem": {"features": "sem", "target": "log", "kind": "ridge"},
    }


def build_model(kind, seed):
    if kind == "et":
        return ExtraTreesRegressor(
            n_estimators=600, random_state=seed, max_features="sqrt",
            min_samples_leaf=1, n_jobs=-1,
        )
    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=600, random_state=seed, max_features="sqrt",
            min_samples_leaf=1, n_jobs=-1,
        )
    if kind == "lgbm":
        return LGBMRegressor(
            n_estimators=400, learning_rate=0.03, num_leaves=7, max_depth=3,
            min_child_samples=8, reg_lambda=10.0, reg_alpha=1.0,
            verbosity=-1, random_state=seed, n_jobs=-1,
        )
    if kind == "ridge":
        return Ridge(alpha=10.0, solver="lsqr")
    raise ValueError(kind)


def fit_candidate(train, features, spec, seed):
    X = normalize_model_frame(train[features])
    y = train["target_clean"].to_numpy(float)
    yt = np.log1p(y) if spec["target"] == "log" else y
    pipe = prep_pipeline(X, build_model(spec["kind"], seed))
    pipe.fit(X, yt)
    return pipe


def predict_candidate(model, df, features, spec):
    if df.empty:
        return np.array([], dtype=float)
    z = model.predict(normalize_model_frame(df[features]))
    p = np.expm1(z) if spec["target"] == "log" else z
    return clip_band(p)


def in_band(df):
    return df[(df["target_clean"] >= LO) & (df["target_clean"] < HI)].copy()


def inner_oof_for_spec(outer_train, feature_sets, name, spec, seedbase):
    rows = []
    band_train = in_band(outer_train)
    years = sorted(band_train["fyDeclared"].astype(int).unique())
    features = feature_sets[spec["features"]]

    for inner_fy in years:
        tr = band_train[band_train["fyDeclared"].astype(int) != inner_fy].copy()
        va = band_train[band_train["fyDeclared"].astype(int) == inner_fy].copy()
        if tr.empty or va.empty:
            continue
        med = float(np.median(tr["target_clean"].to_numpy(float)))
        model = fit_candidate(tr, features, spec, seedbase + int(inner_fy))
        pp = predict_candidate(model, va, features, spec)
        for (_, r), p in zip(va.iterrows(), pp):
            rows.append({
                "disasterNumber": int(r["disasterNumber"]),
                "inner_fy": int(inner_fy),
                "actual": float(r["target_clean"]),
                "ml_pred": float(p),
                "inner_median": med,
                "candidate": name,
            })
    return pd.DataFrame(rows)


def select_candidate(outer_train, feature_sets, outer_fy):
    scores = []
    for j, (name, spec) in enumerate(model_specs().items()):
        oof = inner_oof_for_spec(
            outer_train, feature_sets, name, spec,
            100_000 + int(outer_fy) * 100 + j * 10_000,
        )
        if oof.empty:
            continue
        y = oof["actual"].to_numpy(float)
        med = oof["inner_median"].to_numpy(float)
        med_mae = float(mean_absolute_error(y, med))
        med_log = float(mean_absolute_error(np.log1p(y), np.log1p(med)))
        for w in SHRINK_WEIGHTS:
            pred = clip_band(w * oof["ml_pred"].to_numpy(float) + (1.0 - w) * med)
            scores.append({
                "candidate": name,
                "weight": float(w),
                "inner_MAE": float(mean_absolute_error(y, pred)),
                "inner_log_MAE": float(mean_absolute_error(np.log1p(y), np.log1p(pred))),
                "inner_RMSE": float(mean_squared_error(y, pred) ** 0.5),
                "inner_n": int(len(y)),
                "median_inner_MAE": med_mae,
                "median_inner_log_MAE": med_log,
            })
    if not scores:
        raise RuntimeError(f"No valid inner OOF candidates for outer FY {outer_fy}")
    sdf = pd.DataFrame(scores).sort_values(
        ["inner_MAE", "inner_log_MAE", "weight"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    best = sdf.iloc[0].to_dict()
    best["beats_inner_median"] = bool(float(best["inner_MAE"]) < float(best["median_inner_MAE"]))
    return best, sdf


def fit_outer_selected(outer_train, feature_sets, selected, outer_fy):
    name = str(selected["candidate"])
    spec = model_specs()[name]
    features = feature_sets[spec["features"]]
    bt = in_band(outer_train)
    median = float(np.median(bt["target_clean"].to_numpy(float)))
    model = fit_candidate(bt, features, spec, 900_000 + int(outer_fy))
    return model, features, spec, median


def apply_outer(model, df, features, spec, median, weight):
    ml = predict_candidate(model, df, features, spec)
    blended = clip_band(weight * ml + (1.0 - weight) * median)
    return ml, blended


def main():
    accepted = pd.read_csv(ACCEPTED)
    accepted["disasterNumber"] = accepted["disasterNumber"].astype(int)
    pred_col = "candidate_pred_new" if "candidate_pred_new" in accepted.columns else "candidate_pred"
    assert len(accepted) == 912

    master = normalize_master(pd.read_excel(MASTER))
    master["target_clean"] = pd.to_numeric(
        master["totalObligatedFunding"], errors="coerce"
    ).fillna(0).clip(lower=0)

    ma = fetch_all_mission_assignments()
    sem, _ = build_semantic_rollup(master, ma)

    df = master.merge(sem, on="disasterNumber", how="left")
    df["disasterNumber"] = df["disasterNumber"].astype(int)
    df = df[df["incidentType"] != "Biological"].copy()
    df = df.merge(
        accepted[["disasterNumber", "actual_band", pred_col]],
        on="disasterNumber", how="inner", validate="one_to_one",
    ).rename(columns={pred_col: "router_pred"})
    df["router_pred"] = df["router_pred"].astype(str)
    df["actual_band"] = df["actual_band"].astype(str)
    df["fyDeclared"] = pd.to_numeric(df["fyDeclared"], errors="coerce").astype(int)
    assert len(df) == 912

    current = valid_cols(df, CURRENT_19)
    semcols = valid_cols(
        df,
        [
            c for c in df.columns
            if (c.startswith("sem_") or c.startswith("ma_"))
            and not any(b in c.lower() for b in ["oblig", "fund", "cost", "amount", "dollar"])
        ],
    )
    feature_sets = {
        "current": list(dict.fromkeys(current)),
        "sem": list(dict.fromkeys(current + semcols)),
    }

    direct_rows, routed_rows, fold_rows, score_rows = [], [], [], []

    for outer_fy in sorted(df["fyDeclared"].astype(int).unique()):
        outer_train = df[df["fyDeclared"].astype(int) != outer_fy].copy()
        outer_test = df[df["fyDeclared"].astype(int) == outer_fy].copy()
        bt = in_band(outer_train)
        if bt.empty:
            continue

        selected, sdf = select_candidate(outer_train, feature_sets, int(outer_fy))
        sdf["outer_fy"] = int(outer_fy)
        score_rows.extend(sdf.to_dict("records"))

        model, features, spec, outer_median = fit_outer_selected(
            outer_train, feature_sets, selected, int(outer_fy)
        )
        weight = float(selected["weight"])

        direct_test = in_band(outer_test)
        if len(direct_test):
            ml, pred = apply_outer(model, direct_test, features, spec, outer_median, weight)
            for (_, r), pm, pp in zip(direct_test.iterrows(), ml, pred):
                direct_rows.append({
                    "disasterNumber": int(r["disasterNumber"]),
                    "fyDeclared": int(outer_fy),
                    "state": r["state"],
                    "incidentType": r["incidentType"],
                    "actual": float(r["target_clean"]),
                    "outer_training_median": outer_median,
                    "raw_ml_prediction": float(pm),
                    "selected_ml_prediction": float(pp),
                    "selected_candidate": str(selected["candidate"]),
                    "selected_weight": weight,
                    "beats_inner_median": bool(selected["beats_inner_median"]),
                })

        routed_test = outer_test[outer_test["router_pred"] == BAND].copy()
        if len(routed_test):
            ml, pred = apply_outer(model, routed_test, features, spec, outer_median, weight)
            for (_, r), pm, pp in zip(routed_test.iterrows(), ml, pred):
                routed_rows.append({
                    "disasterNumber": int(r["disasterNumber"]),
                    "fyDeclared": int(outer_fy),
                    "state": r["state"],
                    "incidentType": r["incidentType"],
                    "actual": float(r["target_clean"]),
                    "actual_band": r["actual_band"],
                    "router_pred": r["router_pred"],
                    "outer_training_median": outer_median,
                    "raw_ml_prediction": float(pm),
                    "selected_ml_prediction": float(pp),
                    "selected_candidate": str(selected["candidate"]),
                    "selected_weight": weight,
                    "beats_inner_median": bool(selected["beats_inner_median"]),
                })

        fold_rows.append({
            "outer_fy": int(outer_fy),
            "outer_training_band_n": int(len(bt)),
            "outer_training_band_median": outer_median,
            "selected_candidate": str(selected["candidate"]),
            "selected_weight": weight,
            "inner_MAE": float(selected["inner_MAE"]),
            "inner_log_MAE": float(selected["inner_log_MAE"]),
            "inner_median_MAE": float(selected["median_inner_MAE"]),
            "beats_inner_median": bool(selected["beats_inner_median"]),
            "direct_test_n": int(len(direct_test)),
            "routed_test_n": int(len(routed_test)),
        })

    direct = pd.DataFrame(direct_rows)
    routed = pd.DataFrame(routed_rows)
    folds = pd.DataFrame(fold_rows)
    scores = pd.DataFrame(score_rows)

    direct.to_csv(OUT / "direct_predictions.csv", index=False)
    routed.to_csv(OUT / "routed_predictions.csv", index=False)
    folds.to_csv(OUT / "folds.csv", index=False)
    scores.to_csv(OUT / "inner_candidate_scores.csv", index=False)

    direct_metrics = {
        "outer_training_band_median": metric_block(direct["actual"], direct["outer_training_median"]),
        "dedicated_ml_selected": metric_block(direct["actual"], direct["selected_ml_prediction"]),
        "raw_selected_ml_before_shrinkage": metric_block(direct["actual"], direct["raw_ml_prediction"]),
    }
    routed_metrics = {
        "outer_training_band_median": metric_block(routed["actual"], routed["outer_training_median"]),
        "dedicated_ml_selected": metric_block(routed["actual"], routed["selected_ml_prediction"]),
        "raw_selected_ml_before_shrinkage": metric_block(routed["actual"], routed["raw_ml_prediction"]),
    }

    model_counts = Counter(folds["selected_candidate"].astype(str))
    weight_counts = Counter(folds["selected_weight"].astype(float))
    inner_win_rate = float(folds["beats_inner_median"].mean()) if len(folds) else None

    med_mae = direct_metrics["outer_training_band_median"]["MAE"]
    ml_mae = direct_metrics["dedicated_ml_selected"]["MAE"]
    med_rmse = direct_metrics["outer_training_band_median"]["RMSE"]
    ml_rmse = direct_metrics["dedicated_ml_selected"]["RMSE"]

    summary = {
        "scope": "Dedicated non-Biological $100K-$1M amount specialist",
        "protocol": [
            "Strict outer leave-fiscal-year-out.",
            "Candidate model and nonzero shrinkage weight selected only from inner LFYO predictions within outer-training $100K-$1M cases.",
            "Entire outer fiscal year excluded from fitting, selection, and training-band median calculation.",
            "Target-blind current19 and Mission Assignment semantic features only.",
            "Frozen 830/912 router is not modified.",
            "Direct audit evaluates true-band cases; routed audit evaluates cases the frozen router actually sends to this band.",
        ],
        "direct_true_band": direct_metrics,
        "deployable_routed_to_band": routed_metrics,
        "selection": {
            "fold_count": int(len(folds)),
            "selected_model_counts": dict(model_counts),
            "selected_weight_counts": {str(k): int(v) for k, v in weight_counts.items()},
            "fraction_of_folds_inner_ml_beats_median": inner_win_rate,
        },
        "comparison_direct": {
            "MAE_change_pct_vs_median": float(100.0 * (ml_mae - med_mae) / med_mae),
            "RMSE_change_pct_vs_median": float(100.0 * (ml_rmse - med_rmse) / med_rmse),
            "ML_wins_MAE": bool(ml_mae < med_mae),
            "ML_wins_RMSE": bool(ml_rmse < med_rmse),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    dmed = direct_metrics["outer_training_band_median"]
    dml = direct_metrics["dedicated_ml_selected"]
    rmed = routed_metrics["outer_training_band_median"]
    rml = routed_metrics["dedicated_ml_selected"]

    lines = [
        "# Dedicated $100K-$1M ML amount specialist",
        "",
        "## Direct strict-LFYO audit on true band cases",
        f"- n: **{dml['n']}**",
        f"- Median baseline MAE: **USD {dmed['MAE']:,.0f}**",
        f"- Dedicated ML MAE: **USD {dml['MAE']:,.0f}**",
        f"- Median baseline RMSE: **USD {dmed['RMSE']:,.0f}**",
        f"- Dedicated ML RMSE: **USD {dml['RMSE']:,.0f}**",
        f"- Median baseline log-MAE: **{dmed['log_MAE']:.4f}**",
        f"- Dedicated ML log-MAE: **{dml['log_MAE']:.4f}**",
        f"- Dedicated ML unique predictions: **{dml['unique_predictions_rounded_dollar']}**",
        "",
        "## Frozen-router deployable audit for rows routed to $100K-$1M",
        f"- n: **{rml['n']}**",
        f"- Median baseline MAE: **USD {rmed['MAE']:,.0f}**",
        f"- Dedicated ML MAE: **USD {rml['MAE']:,.0f}**",
        f"- Median baseline RMSE: **USD {rmed['RMSE']:,.0f}**",
        f"- Dedicated ML RMSE: **USD {rml['RMSE']:,.0f}**",
        "",
        "## Model selection",
        f"- Selected models by fold: **{dict(model_counts)}**",
        f"- Selected shrinkage weights by fold: **{dict(weight_counts)}**",
        f"- Fraction of folds where inner-selected ML beat inner median: **{inner_win_rate:.1%}**",
        "",
        "## Decision",
        "- **Dedicated ML beats the strict-LFYO median on direct MAE.**"
        if summary["comparison_direct"]["ML_wins_MAE"]
        else "- **Dedicated ML does not yet beat the strict-LFYO median on direct MAE.**",
        "",
        "The frozen router remains unchanged; this audit only tests the amount specialist.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
