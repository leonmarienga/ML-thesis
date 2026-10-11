#!/usr/bin/env python3
"""
Strict nested-LFYO individualized amount-specialist audit for the non-Biological
$50M-$200M band.

Because only 13 true cases exist, the specialist is not fit on those 13 alone.
Supervised candidates use broader target-blind support ($20M-$300M) with
training rows in the target band upweighted. Historical analogue candidates
also draw from the broader support pool.

For each outer held fiscal year:
- hide the entire outer year;
- use inner LFYO on outer-training $50M-$200M cases to select the method;
- refit the selected method using only outer-training data;
- predict the untouched outer-year cases.

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
ACCEPTED = ROOT / "audit_inputs" / "50_200m_ml_specialist" / "accepted" / "candidate_predictions.csv"
OUT = ROOT / "audit_outputs" / "nonbio_50_200m_dedicated_ml_specialist"
OUT.mkdir(parents=True, exist_ok=True)

LO = 50_000_000.0
HI = 200_000_000.0
SUPPORT_LO = 20_000_000.0
SUPPORT_HI = 300_000_000.0
BAND = "50-200M"
SHRINK_WEIGHTS = [0.25, 0.50, 0.75, 1.00]
ANALOG_K = [1, 2, 3]


def metric_block(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if len(y) == 0:
        return {"n": 0}
    ly = np.log1p(y)
    lp = np.log1p(np.maximum(p, 0))
    rel = np.abs(p - y) / np.maximum(y, 1.0)
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
        "within_50pct": float(np.mean(rel <= 0.50)),
        "unique_predictions_rounded_dollar": int(len(np.unique(np.round(p, 0)))),
    }


def in_target_band(df):
    return df[(df["target_clean"] >= LO) & (df["target_clean"] < HI)].copy()


def support_rows(df):
    return df[(df["target_clean"] >= SUPPORT_LO) & (df["target_clean"] < SUPPORT_HI)].copy()


def clip_band(p):
    return np.clip(np.asarray(p, float), LO, HI - 1.0)


def model_specs():
    return {
        "et_log_sem": {"kind": "et", "target": "log", "features": "sem"},
        "et_raw_sem": {"kind": "et", "target": "raw", "features": "sem"},
        "rf_log_sem": {"kind": "rf", "target": "log", "features": "sem"},
        "lgbm_log_sem": {"kind": "lgbm", "target": "log", "features": "sem"},
        "ridge_log_sem": {"kind": "ridge", "target": "log", "features": "sem"},
        "et_log_current": {"kind": "et", "target": "log", "features": "current"},
    }


def build_model(kind, seed):
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
            min_child_samples=5, reg_lambda=15.0, reg_alpha=1.0,
            verbosity=-1, random_state=seed, n_jobs=-1
        )
    if kind == "ridge":
        return Ridge(alpha=25.0, solver="lsqr")
    raise ValueError(kind)


def fit_model(train_support, features, spec, seed):
    X = normalize_model_frame(train_support[features])
    y = train_support["target_clean"].to_numpy(float)
    yt = np.log1p(y) if spec["target"] == "log" else y
    sw = np.ones(len(train_support), dtype=float)
    sw[(y >= LO) & (y < HI)] = 6.0
    pipe = prep_pipeline(X, build_model(spec["kind"], seed))
    try:
        pipe.fit(X, yt, model__sample_weight=sw)
    except Exception:
        pipe.fit(X, yt)
    return pipe


def predict_model(model, df, features, spec):
    if df.empty:
        return np.array([], dtype=float)
    z = model.predict(normalize_model_frame(df[features]))
    p = np.expm1(z) if spec["target"] == "log" else z
    return clip_band(p)


def analogue_features(df, sem_features):
    cols = []
    preferred = [
        "durationDays", "declarationDelayDays", "expectedResourceScore",
        "missionAssignmentCount", "uniqueAgencyCount", "uniqueMaTypeCount",
        "uniquePriorityCount", "responseComplexityScore", "missionDensity", "agencyDensity",
        "sem_unique_missions", "sem_unique_agencies", "sem_unique_esf", "sem_unique_types",
        "sem_duration_mean", "sem_duration_max", "sem_long_30d_count", "sem_long_90d_count",
        "ma_raw_row_count", "ma_amendment_rows", "ma_amendment_share",
    ]
    for c in preferred:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c].dtype):
            cols.append(c)
    if len(cols) < 8:
        for c in sem_features:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c].dtype) and c not in cols:
                cols.append(c)
            if len(cols) >= 24:
                break
    return cols[:24]


def analogue_predict(train_support, test, features, k):
    out = []
    for _, row in test.iterrows():
        same = train_support[train_support["incidentType"].astype(str) == str(row["incidentType"])].copy()
        pool = same if len(same) >= 2 else train_support.copy()
        X = pool[features].replace([np.inf, -np.inf], np.nan).apply(pd.to_numeric, errors="coerce")
        q = pd.to_numeric(row[features], errors="coerce")
        med = X.median(axis=0)
        X = X.fillna(med).fillna(0.0)
        q = q.fillna(med).fillna(0.0)
        mu = X.mean(axis=0)
        sd = X.std(axis=0).replace(0, 1).fillna(1)
        dist = np.sqrt((((X - mu) / sd - (q - mu) / sd) ** 2).mean(axis=1)).to_numpy(float)
        order = np.argsort(dist)[: min(k, len(pool))]
        chosen = pool.iloc[order]
        weights = 1.0 / np.maximum(dist[order], 0.10)
        pred = float(np.expm1(np.average(np.log1p(chosen["target_clean"].to_numpy(float)), weights=weights)))
        out.append(pred)
    return clip_band(out)


def inner_candidate_predictions(outer_train, feature_sets, analog_cols, outer_fy):
    target = in_target_band(outer_train)
    years = sorted(target["fyDeclared"].astype(int).unique())
    rows = []

    for inner_fy in years:
        tr_all = outer_train[outer_train["fyDeclared"].astype(int) != inner_fy].copy()
        va = target[target["fyDeclared"].astype(int) == inner_fy].copy()
        if va.empty:
            continue
        tr_support = support_rows(tr_all)
        tr_band = in_target_band(tr_all)
        if tr_support.empty or tr_band.empty:
            continue

        band_median = float(np.median(tr_band["target_clean"].to_numpy(float)))

        preds = {}
        for j, (name, spec) in enumerate(model_specs().items()):
            feats = feature_sets[spec["features"]]
            if len(tr_support) < 5:
                preds[name] = np.full(len(va), band_median)
            else:
                model = fit_model(tr_support, feats, spec, 100_000 + outer_fy * 100 + inner_fy + j * 1000)
                preds[name] = predict_model(model, va, feats, spec)

        for k in ANALOG_K:
            preds[f"analogue_k{k}"] = analogue_predict(tr_support, va, analog_cols, k)

        for idx, (_, r) in enumerate(va.iterrows()):
            rec = {
                "disasterNumber": int(r["disasterNumber"]),
                "inner_fy": int(inner_fy),
                "actual": float(r["target_clean"]),
                "inner_band_median": band_median,
            }
            for name, pp in preds.items():
                rec[f"pred__{name}"] = float(pp[idx])
            rows.append(rec)
    return pd.DataFrame(rows)


def select_candidate(outer_train, feature_sets, analog_cols, outer_fy):
    oof = inner_candidate_predictions(outer_train, feature_sets, analog_cols, outer_fy)
    if oof.empty:
        raise RuntimeError(f"No inner OOF target-band rows for outer FY {outer_fy}")

    y = oof["actual"].to_numpy(float)
    med = oof["inner_band_median"].to_numpy(float)
    median_mae = float(mean_absolute_error(y, med))
    scores = []

    names = list(model_specs()) + [f"analogue_k{k}" for k in ANALOG_K]
    for name in names:
        raw = oof[f"pred__{name}"].to_numpy(float)
        for w in SHRINK_WEIGHTS:
            pred = clip_band(w * raw + (1.0 - w) * med)
            scores.append({
                "candidate": name,
                "weight": float(w),
                "inner_MAE": float(mean_absolute_error(y, pred)),
                "inner_RMSE": float(mean_squared_error(y, pred) ** 0.5),
                "inner_log_MAE": float(mean_absolute_error(np.log1p(y), np.log1p(pred))),
                "inner_n": int(len(y)),
                "median_inner_MAE": median_mae,
            })

    sdf = pd.DataFrame(scores).sort_values(
        ["inner_MAE", "inner_log_MAE", "inner_RMSE"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    best = sdf.iloc[0].to_dict()
    best["beats_inner_median"] = bool(float(best["inner_MAE"]) < median_mae)
    return best, sdf


def fit_outer_selected(outer_train, feature_sets, analog_cols, selected, outer_fy):
    tr_support = support_rows(outer_train)
    tr_band = in_target_band(outer_train)
    median = float(np.median(tr_band["target_clean"].to_numpy(float)))
    name = str(selected["candidate"])
    if name.startswith("analogue_k"):
        return {"kind": "analogue", "support": tr_support, "k": int(name.split("k")[-1]), "median": median}
    spec = model_specs()[name]
    feats = feature_sets[spec["features"]]
    model = fit_model(tr_support, feats, spec, 900_000 + outer_fy)
    return {"kind": "model", "model": model, "features": feats, "spec": spec, "median": median}


def apply_outer(fitted, df, analog_cols, selected):
    if fitted["kind"] == "analogue":
        raw = analogue_predict(fitted["support"], df, analog_cols, fitted["k"])
    else:
        raw = predict_model(fitted["model"], df, fitted["features"], fitted["spec"])
    pred = clip_band(float(selected["weight"]) * raw + (1.0 - float(selected["weight"])) * fitted["median"])
    return raw, pred


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
    analog_cols = analogue_features(df, feature_sets["sem"])

    direct_rows, routed_rows, fold_rows, score_rows = [], [], [], []

    for outer_fy in sorted(df["fyDeclared"].astype(int).unique()):
        outer_train = df[df["fyDeclared"].astype(int) != outer_fy].copy()
        outer_test = df[df["fyDeclared"].astype(int) == outer_fy].copy()
        tr_band = in_target_band(outer_train)
        if tr_band.empty:
            continue

        selected, sdf = select_candidate(outer_train, feature_sets, analog_cols, int(outer_fy))
        sdf["outer_fy"] = int(outer_fy)
        score_rows.extend(sdf.to_dict("records"))
        fitted = fit_outer_selected(outer_train, feature_sets, analog_cols, selected, int(outer_fy))

        direct_test = in_target_band(outer_test)
        if len(direct_test):
            raw, pred = apply_outer(fitted, direct_test, analog_cols, selected)
            for (_, r), rr, pp in zip(direct_test.iterrows(), raw, pred):
                direct_rows.append({
                    "disasterNumber": int(r["disasterNumber"]),
                    "fyDeclared": int(outer_fy),
                    "state": r["state"],
                    "incidentType": r["incidentType"],
                    "actual": float(r["target_clean"]),
                    "outer_training_band_median": float(fitted["median"]),
                    "raw_specialist_prediction": float(rr),
                    "selected_ml_prediction": float(pp),
                    "candidate": str(selected["candidate"]),
                    "weight": float(selected["weight"]),
                })

        routed_test = outer_test[outer_test["router_pred"] == BAND].copy()
        if len(routed_test):
            raw, pred = apply_outer(fitted, routed_test, analog_cols, selected)
            for (_, r), rr, pp in zip(routed_test.iterrows(), raw, pred):
                routed_rows.append({
                    "disasterNumber": int(r["disasterNumber"]),
                    "fyDeclared": int(outer_fy),
                    "state": r["state"],
                    "incidentType": r["incidentType"],
                    "actual": float(r["target_clean"]),
                    "actual_band": r["actual_band"],
                    "router_pred": r["router_pred"],
                    "outer_training_band_median": float(fitted["median"]),
                    "raw_specialist_prediction": float(rr),
                    "selected_ml_prediction": float(pp),
                    "candidate": str(selected["candidate"]),
                    "weight": float(selected["weight"]),
                })

        fold_rows.append({
            "outer_fy": int(outer_fy),
            "outer_training_target_band_n": int(len(tr_band)),
            "outer_training_support_n": int(len(support_rows(outer_train))),
            "outer_training_band_median": float(fitted["median"]),
            "candidate": str(selected["candidate"]),
            "weight": float(selected["weight"]),
            "inner_MAE": float(selected["inner_MAE"]),
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
        "outer_training_band_median": metric_block(direct["actual"], direct["outer_training_band_median"]),
        "nested_selected_specialist": metric_block(direct["actual"], direct["selected_ml_prediction"]),
    }
    routed_metrics = {
        "outer_training_band_median": metric_block(routed["actual"], routed["outer_training_band_median"]),
        "nested_selected_specialist": metric_block(routed["actual"], routed["selected_ml_prediction"]),
    }

    cand_counts = Counter(folds["candidate"].astype(str))
    weight_counts = Counter(folds["weight"].astype(float))
    med_mae = direct_metrics["outer_training_band_median"]["MAE"]
    ml_mae = direct_metrics["nested_selected_specialist"]["MAE"]
    med_rmse = direct_metrics["outer_training_band_median"]["RMSE"]
    ml_rmse = direct_metrics["nested_selected_specialist"]["RMSE"]

    summary = {
        "scope": "Dedicated non-Biological $50M-$200M individualized amount specialist",
        "protocol": [
            "Strict outer leave-fiscal-year-out.",
            "Method and shrinkage weight selected only by inner LFYO on outer-training true $50M-$200M cases.",
            "Supervised candidates use broader $20M-$300M training support with target-band rows upweighted 6x.",
            "Historical analogue candidates use the same target-blind broader support.",
            "Entire outer fiscal year excluded from fitting, model selection, and median calculation.",
            "Frozen 830/912 router is not modified.",
        ],
        "analogue_features": analog_cols,
        "direct_true_band": direct_metrics,
        "deployable_routed_to_band": routed_metrics,
        "selection": {
            "fold_count": int(len(folds)),
            "candidate_counts": dict(cand_counts),
            "weight_counts": {str(k): int(v) for k, v in weight_counts.items()},
            "fraction_of_folds_inner_specialist_beats_median": float(folds["beats_inner_median"].mean()) if len(folds) else None,
        },
        "comparison_direct": {
            "MAE_change_pct_vs_median": float(100.0 * (ml_mae - med_mae) / med_mae),
            "RMSE_change_pct_vs_median": float(100.0 * (ml_rmse - med_rmse) / med_rmse),
            "specialist_wins_MAE": bool(ml_mae < med_mae),
            "specialist_wins_RMSE": bool(ml_rmse < med_rmse),
        },
        "caution": "Only 13 true non-Biological $50M-$200M cases exist; all metrics are high variance.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    dmed = direct_metrics["outer_training_band_median"]
    dml = direct_metrics["nested_selected_specialist"]
    lines = [
        "# Dedicated $50M-$200M individualized amount specialist",
        "",
        f"- Direct n: **{dml['n']}**",
        f"- Median baseline MAE: **USD {dmed['MAE']:,.0f}**",
        f"- Specialist MAE: **USD {dml['MAE']:,.0f}**",
        f"- Median baseline RMSE: **USD {dmed['RMSE']:,.0f}**",
        f"- Specialist RMSE: **USD {dml['RMSE']:,.0f}**",
        f"- Median baseline R2: **{dmed['R2']:.4f}**",
        f"- Specialist R2: **{dml['R2']:.4f}**",
        f"- Specialist unique predictions: **{dml['unique_predictions_rounded_dollar']}**",
        "",
        "Decision: " + (
            "**Specialist beats the strict-LFYO median on direct MAE.**"
            if summary["comparison_direct"]["specialist_wins_MAE"]
            else "**Specialist does not beat the strict-LFYO median on direct MAE.**"
        ),
        "",
        "Caution: only 13 true cases exist.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
