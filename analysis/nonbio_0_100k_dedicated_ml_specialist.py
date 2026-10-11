#!/usr/bin/env python3
"""
Dedicated strict nested-LFYO amount specialist audit for non-Biological
$0-$100K funding cases.

Architecture
------------
Two-stage specialist:
1) funded-vs-zero classifier
2) positive-amount regressor for cases predicted funded

For every OUTER held fiscal year:
- exclude the entire outer year;
- use only outer-training rows with 0 <= funding < 100K;
- run INNER LFYO to generate honest OOF probabilities and amount predictions;
- select classifier, funding threshold, positive regressor, and optional amount shrinkage
  using inner OOF dollar MAE;
- refit selected components on all outer-training band rows;
- predict untouched outer-year cases.

The frozen 830/912 router is never modified.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
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
ACCEPTED = ROOT / "audit_inputs" / "0_100k_ml_specialist" / "accepted" / "candidate_predictions.csv"
OUT = ROOT / "audit_outputs" / "nonbio_0_100k_dedicated_ml_specialist"
OUT.mkdir(parents=True, exist_ok=True)

LO = 0.0
HI = 100_000.0
BAND = "0-100K"
THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
SHRINK_WEIGHTS = [0.50, 0.75, 1.00]


def metric_block(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if len(y) == 0:
        return {"n": 0}
    ly = np.log1p(np.maximum(y, 0))
    lp = np.log1p(np.maximum(p, 0))
    return {
        "n": int(len(y)),
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(mean_squared_error(y, p) ** 0.5),
        "MedAE": float(median_absolute_error(y, p)),
        "R2": float(r2_score(y, p)) if len(y) >= 2 and float(np.var(y)) > 0 else None,
        "log_MAE": float(mean_absolute_error(ly, lp)),
        "log_RMSE": float(mean_squared_error(ly, lp) ** 0.5),
        "zero_accuracy": float(np.mean((y == 0) == (p <= 0.5))),
        "predicted_zero_share": float(np.mean(p <= 0.5)),
        "actual_zero_share": float(np.mean(y == 0)),
        "unique_predictions_rounded_dollar": int(len(np.unique(np.round(p, 0)))),
    }


def in_band(df):
    return df[(df["target_clean"] >= LO) & (df["target_clean"] < HI)].copy()


def clip_positive(p):
    return np.clip(np.asarray(p, float), 1.0, HI - 1.0)


def classifier_specs():
    return {
        "logistic_sem": {"features": "sem", "kind": "logistic"},
        "rf_sem": {"features": "sem", "kind": "rf"},
        "et_sem": {"features": "sem", "kind": "et"},
        "lgbm_sem": {"features": "sem", "kind": "lgbm"},
        "et_current": {"features": "current", "kind": "et"},
    }


def regressor_specs():
    return {
        "et_log_sem": {"features": "sem", "target": "log", "kind": "et"},
        "rf_log_sem": {"features": "sem", "target": "log", "kind": "rf"},
        "lgbm_log_sem": {"features": "sem", "target": "log", "kind": "lgbm"},
        "ridge_log_sem": {"features": "sem", "target": "log", "kind": "ridge"},
        "et_raw_sem": {"features": "sem", "target": "raw", "kind": "et"},
        "et_log_current": {"features": "current", "target": "log", "kind": "et"},
    }


def build_classifier(kind, seed):
    if kind == "logistic":
        return LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5)
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=500, random_state=seed, class_weight="balanced_subsample",
            max_features="sqrt", min_samples_leaf=1, n_jobs=-1
        )
    if kind == "et":
        return ExtraTreesClassifier(
            n_estimators=500, random_state=seed, class_weight="balanced",
            max_features="sqrt", min_samples_leaf=1, n_jobs=-1
        )
    if kind == "lgbm":
        return LGBMClassifier(
            n_estimators=350, learning_rate=0.03, num_leaves=7, max_depth=3,
            min_child_samples=10, reg_lambda=10.0, reg_alpha=1.0,
            class_weight="balanced", verbosity=-1, random_state=seed, n_jobs=-1
        )
    raise ValueError(kind)


def build_regressor(kind, seed):
    if kind == "et":
        return ExtraTreesRegressor(
            n_estimators=500, random_state=seed, max_features="sqrt",
            min_samples_leaf=1, n_jobs=-1
        )
    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=500, random_state=seed, max_features="sqrt",
            min_samples_leaf=1, n_jobs=-1
        )
    if kind == "lgbm":
        return LGBMRegressor(
            n_estimators=350, learning_rate=0.03, num_leaves=7, max_depth=3,
            min_child_samples=8, reg_lambda=10.0, reg_alpha=1.0,
            verbosity=-1, random_state=seed, n_jobs=-1
        )
    if kind == "ridge":
        return Ridge(alpha=10.0, solver="lsqr")
    raise ValueError(kind)


def fit_classifier(train, features, spec, seed):
    X = normalize_model_frame(train[features])
    y = (train["target_clean"].to_numpy(float) > 0).astype(int)
    pipe = prep_pipeline(X, build_classifier(spec["kind"], seed))
    pipe.fit(X, y)
    return pipe


def predict_funded_prob(model, df, features):
    if df.empty:
        return np.array([], dtype=float)
    p = model.predict_proba(normalize_model_frame(df[features]))
    classes = list(model[-1].classes_)
    return p[:, classes.index(1)] if 1 in classes else np.zeros(len(df), dtype=float)


def fit_regressor(train_pos, features, spec, seed):
    X = normalize_model_frame(train_pos[features])
    y = train_pos["target_clean"].to_numpy(float)
    yt = np.log1p(y) if spec["target"] == "log" else y
    pipe = prep_pipeline(X, build_regressor(spec["kind"], seed))
    pipe.fit(X, yt)
    return pipe


def predict_positive(model, df, features, spec):
    if df.empty:
        return np.array([], dtype=float)
    z = model.predict(normalize_model_frame(df[features]))
    p = np.expm1(z) if spec["target"] == "log" else z
    return clip_positive(p)


def inner_oof(outer_train, feature_sets, outer_fy):
    band_train = in_band(outer_train)
    years = sorted(band_train["fyDeclared"].astype(int).unique())
    rows = []

    for inner_fy in years:
        tr = band_train[band_train["fyDeclared"].astype(int) != inner_fy].copy()
        va = band_train[band_train["fyDeclared"].astype(int) == inner_fy].copy()
        if tr.empty or va.empty:
            continue

        pos_tr = tr[tr["target_clean"] > 0].copy()
        pos_median = float(np.median(pos_tr["target_clean"])) if len(pos_tr) else 1.0

        cl_probs = {}
        for j, (name, spec) in enumerate(classifier_specs().items()):
            ytr = (tr["target_clean"] > 0).astype(int)
            if ytr.nunique() < 2:
                cl_probs[name] = np.full(len(va), float(ytr.iloc[0]))
            else:
                features = feature_sets[spec["features"]]
                model = fit_classifier(tr, features, spec, 100_000 + outer_fy * 100 + inner_fy + j * 1000)
                cl_probs[name] = predict_funded_prob(model, va, features)

        reg_preds = {}
        for j, (name, spec) in enumerate(regressor_specs().items()):
            if len(pos_tr) < 4:
                reg_preds[name] = np.full(len(va), pos_median)
            else:
                features = feature_sets[spec["features"]]
                model = fit_regressor(pos_tr, features, spec, 200_000 + outer_fy * 100 + inner_fy + j * 1000)
                reg_preds[name] = predict_positive(model, va, features, spec)

        for k, (_, r) in enumerate(va.iterrows()):
            row = {
                "disasterNumber": int(r["disasterNumber"]),
                "inner_fy": int(inner_fy),
                "actual": float(r["target_clean"]),
                "positive_median": pos_median,
            }
            for name, p in cl_probs.items():
                row[f"prob__{name}"] = float(p[k])
            for name, p in reg_preds.items():
                row[f"reg__{name}"] = float(p[k])
            rows.append(row)
    return pd.DataFrame(rows)


def select_config(outer_train, feature_sets, outer_fy):
    oof = inner_oof(outer_train, feature_sets, outer_fy)
    if oof.empty:
        raise RuntimeError(f"No inner OOF rows for outer FY {outer_fy}")

    y = oof["actual"].to_numpy(float)
    band_train = in_band(outer_train)
    outer_band_median = float(np.median(band_train["target_clean"].to_numpy(float)))
    baseline = np.full(len(y), outer_band_median, dtype=float)

    scores = []
    for cname in classifier_specs():
        prob = oof[f"prob__{cname}"].to_numpy(float)
        for rname in regressor_specs():
            raw_amt = oof[f"reg__{rname}"].to_numpy(float)
            med_amt = oof["positive_median"].to_numpy(float)
            for shrink in SHRINK_WEIGHTS:
                amt = clip_positive(shrink * raw_amt + (1.0 - shrink) * med_amt)
                for thr in THRESHOLDS:
                    pred = np.where(prob >= thr, amt, 0.0)
                    scores.append({
                        "classifier": cname,
                        "regressor": rname,
                        "threshold": float(thr),
                        "shrink_weight": float(shrink),
                        "inner_MAE": float(mean_absolute_error(y, pred)),
                        "inner_RMSE": float(mean_squared_error(y, pred) ** 0.5),
                        "inner_log_MAE": float(mean_absolute_error(np.log1p(y), np.log1p(pred))),
                        "inner_n": int(len(y)),
                        "outer_training_band_median": outer_band_median,
                        "median_baseline_MAE": float(mean_absolute_error(y, baseline)),
                    })

    sdf = pd.DataFrame(scores).sort_values(
        ["inner_MAE", "inner_log_MAE", "inner_RMSE"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    best = sdf.iloc[0].to_dict()
    best["beats_inner_median"] = bool(float(best["inner_MAE"]) < float(best["median_baseline_MAE"]))
    return best, sdf


def fit_outer_selected(outer_train, feature_sets, selected, outer_fy):
    band_train = in_band(outer_train)
    pos_train = band_train[band_train["target_clean"] > 0].copy()

    cspec = classifier_specs()[str(selected["classifier"])]
    rspec = regressor_specs()[str(selected["regressor"])]
    cfeatures = feature_sets[cspec["features"]]
    rfeatures = feature_sets[rspec["features"]]

    clf = fit_classifier(band_train, cfeatures, cspec, 900_000 + outer_fy)
    pos_median = float(np.median(pos_train["target_clean"])) if len(pos_train) else 1.0
    if len(pos_train) < 4:
        reg = None
    else:
        reg = fit_regressor(pos_train, rfeatures, rspec, 910_000 + outer_fy)

    return clf, reg, cfeatures, rfeatures, rspec, pos_median


def apply_outer(clf, reg, df, cfeatures, rfeatures, rspec, pos_median, selected):
    prob = predict_funded_prob(clf, df, cfeatures)
    raw_amt = np.full(len(df), pos_median, dtype=float) if reg is None else predict_positive(reg, df, rfeatures, rspec)
    shrink = float(selected["shrink_weight"])
    amt = clip_positive(shrink * raw_amt + (1.0 - shrink) * pos_median)
    pred = np.where(prob >= float(selected["threshold"]), amt, 0.0)
    return prob, raw_amt, pred


def main():
    accepted = pd.read_csv(ACCEPTED)
    accepted["disasterNumber"] = accepted["disasterNumber"].astype(int)
    pred_col = "candidate_pred_new" if "candidate_pred_new" in accepted.columns else "candidate_pred"
    assert len(accepted) == 912

    master = normalize_master(pd.read_excel(MASTER))
    master["target_clean"] = pd.to_numeric(master["totalObligatedFunding"], errors="coerce").fillna(0).clip(lower=0)

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
        band_train = in_band(outer_train)
        if band_train.empty:
            continue

        selected, sdf = select_config(outer_train, feature_sets, int(outer_fy))
        sdf["outer_fy"] = int(outer_fy)
        score_rows.extend(sdf.to_dict("records"))

        clf, reg, cfeatures, rfeatures, rspec, pos_median = fit_outer_selected(
            outer_train, feature_sets, selected, int(outer_fy)
        )
        outer_band_median = float(np.median(band_train["target_clean"].to_numpy(float)))

        direct_test = in_band(outer_test)
        if len(direct_test):
            prob, raw, pred = apply_outer(
                clf, reg, direct_test, cfeatures, rfeatures, rspec, pos_median, selected
            )
            for (_, r), pr, ra, pp in zip(direct_test.iterrows(), prob, raw, pred):
                direct_rows.append({
                    "disasterNumber": int(r["disasterNumber"]),
                    "fyDeclared": int(outer_fy),
                    "state": r["state"],
                    "incidentType": r["incidentType"],
                    "actual": float(r["target_clean"]),
                    "outer_training_band_median": outer_band_median,
                    "funded_probability": float(pr),
                    "raw_positive_prediction": float(ra),
                    "selected_ml_prediction": float(pp),
                    "classifier": str(selected["classifier"]),
                    "regressor": str(selected["regressor"]),
                    "threshold": float(selected["threshold"]),
                    "shrink_weight": float(selected["shrink_weight"]),
                })

        routed_test = outer_test[outer_test["router_pred"] == BAND].copy()
        if len(routed_test):
            prob, raw, pred = apply_outer(
                clf, reg, routed_test, cfeatures, rfeatures, rspec, pos_median, selected
            )
            for (_, r), pr, ra, pp in zip(routed_test.iterrows(), prob, raw, pred):
                routed_rows.append({
                    "disasterNumber": int(r["disasterNumber"]),
                    "fyDeclared": int(outer_fy),
                    "state": r["state"],
                    "incidentType": r["incidentType"],
                    "actual": float(r["target_clean"]),
                    "actual_band": r["actual_band"],
                    "router_pred": r["router_pred"],
                    "outer_training_band_median": outer_band_median,
                    "funded_probability": float(pr),
                    "raw_positive_prediction": float(ra),
                    "selected_ml_prediction": float(pp),
                    "classifier": str(selected["classifier"]),
                    "regressor": str(selected["regressor"]),
                    "threshold": float(selected["threshold"]),
                    "shrink_weight": float(selected["shrink_weight"]),
                })

        fold_rows.append({
            "outer_fy": int(outer_fy),
            "outer_training_band_n": int(len(band_train)),
            "outer_training_band_median": outer_band_median,
            "actual_zero_share_train": float(np.mean(band_train["target_clean"] == 0)),
            "classifier": str(selected["classifier"]),
            "regressor": str(selected["regressor"]),
            "threshold": float(selected["threshold"]),
            "shrink_weight": float(selected["shrink_weight"]),
            "inner_MAE": float(selected["inner_MAE"]),
            "inner_median_MAE": float(selected["median_baseline_MAE"]),
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
        "outer_training_band_median": metric_block(direct["actual"], direct["outer_training_band_median"]),
        "dedicated_two_stage_ml": metric_block(direct["actual"], direct["selected_ml_prediction"]),
    }
    routed_metrics = {
        "outer_training_band_median": metric_block(routed["actual"], routed["outer_training_band_median"]),
        "dedicated_two_stage_ml": metric_block(routed["actual"], routed["selected_ml_prediction"]),
    }

    classifier_counts = Counter(folds["classifier"].astype(str))
    regressor_counts = Counter(folds["regressor"].astype(str))
    threshold_counts = Counter(folds["threshold"].astype(float))
    shrink_counts = Counter(folds["shrink_weight"].astype(float))

    med_mae = direct_metrics["outer_training_band_median"]["MAE"]
    ml_mae = direct_metrics["dedicated_two_stage_ml"]["MAE"]
    med_rmse = direct_metrics["outer_training_band_median"]["RMSE"]
    ml_rmse = direct_metrics["dedicated_two_stage_ml"]["RMSE"]

    summary = {
        "scope": "Dedicated non-Biological $0-$100K two-stage amount specialist",
        "protocol": [
            "Strict outer leave-fiscal-year-out.",
            "Classifier, classifier threshold, positive-amount regressor, and shrinkage weight selected only from inner LFYO.",
            "Entire outer fiscal year excluded from fitting, selection, and baseline median calculation.",
            "Target-blind current19 and Mission Assignment semantic features only.",
            "Frozen 830/912 router is not modified.",
        ],
        "direct_true_band": direct_metrics,
        "deployable_routed_to_band": routed_metrics,
        "selection": {
            "fold_count": int(len(folds)),
            "classifier_counts": dict(classifier_counts),
            "regressor_counts": dict(regressor_counts),
            "threshold_counts": {str(k): int(v) for k, v in threshold_counts.items()},
            "shrinkage_counts": {str(k): int(v) for k, v in shrink_counts.items()},
            "fraction_of_folds_inner_ml_beats_median": float(folds["beats_inner_median"].mean()) if len(folds) else None,
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
    dml = direct_metrics["dedicated_two_stage_ml"]
    lines = [
        "# Dedicated $0-$100K two-stage ML amount specialist",
        "",
        f"- Direct n: **{dml['n']}**",
        f"- Median baseline MAE: **USD {dmed['MAE']:,.0f}**",
        f"- Two-stage ML MAE: **USD {dml['MAE']:,.0f}**",
        f"- Median baseline RMSE: **USD {dmed['RMSE']:,.0f}**",
        f"- Two-stage ML RMSE: **USD {dml['RMSE']:,.0f}**",
        f"- Median zero accuracy: **{dmed['zero_accuracy']:.1%}**",
        f"- ML zero accuracy: **{dml['zero_accuracy']:.1%}**",
        f"- ML unique predictions: **{dml['unique_predictions_rounded_dollar']}**",
        "",
        "Decision: " + (
            "**ML beats the strict-LFYO median on direct MAE.**"
            if summary["comparison_direct"]["ML_wins_MAE"]
            else "**ML does not beat the strict-LFYO median on direct MAE.**"
        ),
    ]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
