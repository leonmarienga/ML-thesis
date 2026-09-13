#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

import legacy_tempv2 as v2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "audit_outputs" / "nonbio_500mplus_hybrid_audit"
OUT.mkdir(parents=True, exist_ok=True)

ANALOG_FEATURES = [
    "logPopulation2010",
    "logMission",
    "logComplexity",
    "logDuration",
    "logDeclaredArea",
    "expectedResourceScore",
    "missionDensity",
    "agencyDensity",
]

DIRECT_SUPPORTS = [100e6, 200e6]
DIRECT_ALPHAS = [0.50, 0.65, 0.75, 0.85]
DIRECT_WEIGHTS = [3.0, 6.0, 10.0]
ANALOG_SUPPORTS = [200e6, 300e6, 500e6]
ANALOG_K = [1, 2, 3]
ANALOG_SCALE_EXP = [0.0, 0.5, 1.0]
BLEND_WEIGHTS = [0.25, 0.50, 0.75]


def metrics(y, p):
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


def direct_predict(train, test, features, support_lo, alpha, extreme_weight, seed):
    tr = train[train.target > support_lo].copy()
    if len(tr) < 4:
        ext = train[train.target > 500e6]
        fallback = float(np.exp(np.log(np.maximum(ext.target, 1.0)).mean())) if len(ext) else 1e9
        return np.full(len(test), np.clip(fallback, 500e6, 6e9))
    pre = v2.prep(tr[features])
    X = pre.fit_transform(tr[features])
    Xe = pre.transform(test[features])
    y = np.log1p(tr.target.to_numpy(float))
    sw = np.ones(len(tr), float)
    sw[tr.target.to_numpy(float) > 500e6] = extreme_weight
    model = LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=320,
        learning_rate=0.03,
        num_leaves=7,
        max_depth=3,
        min_child_samples=3,
        reg_lambda=20,
        reg_alpha=1,
        verbosity=-1,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X, y, sample_weight=sw)
    return np.clip(np.expm1(model.predict(Xe)), 500e6, 6e9)


def analogue_predict(train, test, support_lo, k, scale_exp):
    ref = train[train.target > support_lo].copy()
    if len(ref) == 0:
        ref = train[train.target > 200e6].copy()
    out = []
    for _, row in test.iterrows():
        pool = ref[ref.incidentType.astype(str) == str(row.incidentType)].copy()
        if len(pool) == 0:
            pool = ref.copy()
        X = pool[ANALOG_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
        q = row[ANALOG_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
        mu = X.mean(axis=0)
        sd = X.std(axis=0).replace(0, 1).fillna(1)
        dist = np.sqrt((((X - mu) / sd - (q - mu) / sd) ** 2).mean(axis=1)).to_numpy(float)
        order = np.argsort(dist)[: min(k, len(pool))]
        chosen = pool.iloc[order]
        weights = 1.0 / np.maximum(dist[order], 0.10)

        # Historical analogue scaling: adjust historical target by the geometric
        # response-intensity ratio, with scale_exp selected only on training data.
        cand_m = np.maximum(chosen.missionAssignmentCount.to_numpy(float), 1.0)
        cand_c = np.maximum(chosen.responseComplexityScore.to_numpy(float), 1.0)
        rm = max(float(row.missionAssignmentCount), 1.0) / cand_m
        rc = max(float(row.responseComplexityScore), 1.0) / cand_c
        intensity_ratio = np.clip(np.sqrt(rm * rc), 0.5, 2.0)
        scaled_target = chosen.target.to_numpy(float) * np.power(intensity_ratio, scale_exp)
        pred = float(np.expm1(np.average(np.log1p(scaled_target), weights=weights)))
        out.append(np.clip(pred, 500e6, 6e9))
    return np.asarray(out, float)


def extreme_baselines(train, n):
    ext = train[train.target > 500e6].target.to_numpy(float)
    if len(ext) == 0:
        return {
            "median": np.full(n, 1e9),
            "geomean": np.full(n, 1e9),
            "mean": np.full(n, 1e9),
        }
    return {
        "median": np.full(n, float(np.median(ext))),
        "geomean": np.full(n, float(np.exp(np.mean(np.log(np.maximum(ext, 1.0)))))),
        "mean": np.full(n, float(np.mean(ext))),
    }


def inner_predictions(outer_train, features):
    extreme = outer_train[outer_train.target > 500e6].copy()
    years = sorted(extreme.fyDeclared.astype(int).unique())
    direct_store = {(s, a, w): [] for s in DIRECT_SUPPORTS for a in DIRECT_ALPHAS for w in DIRECT_WEIGHTS}
    analog_store = {(s, k, e): [] for s in ANALOG_SUPPORTS for k in ANALOG_K for e in ANALOG_SCALE_EXP}
    actual = []

    for yr in years:
        tr = outer_train[outer_train.fyDeclared.astype(int) != yr].copy()
        te = extreme[extreme.fyDeclared.astype(int) == yr].copy()
        if te.empty:
            continue
        actual.extend(te.target.astype(float).tolist())
        for cfg in direct_store:
            s, a, w = cfg
            direct_store[cfg].extend(direct_predict(tr, te, features, s, a, w, 5000 + yr).tolist())
        for cfg in analog_store:
            s, k, e = cfg
            analog_store[cfg].extend(analogue_predict(tr, te, s, k, e).tolist())
    return np.asarray(actual, float), direct_store, analog_store


def choose_components(outer_train, features):
    y, direct_store, analog_store = inner_predictions(outer_train, features)
    if len(y) < 2:
        return (200e6, 0.75, 6.0), (500e6, 2, 0.5), 0.5, {}

    direct_scores = {}
    for cfg, pred in direct_store.items():
        p = np.asarray(pred, float)
        direct_scores[cfg] = float(mean_absolute_error(np.log1p(y), np.log1p(p)))
    analog_scores = {}
    for cfg, pred in analog_store.items():
        p = np.asarray(pred, float)
        analog_scores[cfg] = float(mean_absolute_error(np.log1p(y), np.log1p(p)))

    best_direct = min(direct_scores, key=direct_scores.get)
    best_analog = min(analog_scores, key=analog_scores.get)
    dp = np.asarray(direct_store[best_direct], float)
    ap = np.asarray(analog_store[best_analog], float)
    blend_scores = {}
    for w in BLEND_WEIGHTS:
        p = np.clip(w * dp + (1 - w) * ap, 500e6, 6e9)
        blend_scores[w] = float(mean_absolute_error(np.log1p(y), np.log1p(p)))
    best_blend = min(blend_scores, key=blend_scores.get)

    details = {
        "inner_n": int(len(y)),
        "best_direct": str(best_direct),
        "best_direct_log_MAE": direct_scores[best_direct],
        "best_analogue": str(best_analog),
        "best_analogue_log_MAE": analog_scores[best_analog],
        "blend_log_MAE": {str(k): float(v) for k, v in blend_scores.items()},
    }
    return best_direct, best_analog, float(best_blend), details


def legacy_fixed_hybrid(train, test, features, held_year):
    # Reproduce the reconstructed recipe used in the earlier mixed-stack diagnostic.
    direct = direct_predict(train, test, features, 200e6, 0.75, 6.0, 8000 + held_year)
    analog = analogue_predict(train, test, 500e6, 2, 0.5)

    extreme = train[train.target > 500e6].copy()
    inner_rows = []
    for yr in sorted(extreme.fyDeclared.astype(int).unique()):
        tr = train[train.fyDeclared.astype(int) != yr].copy()
        te = extreme[extreme.fyDeclared.astype(int) == yr].copy()
        if te.empty or (tr.target > 500e6).sum() < 1:
            continue
        d = direct_predict(tr, te, features, 200e6, 0.75, 6.0, 7000 + yr)
        a = analogue_predict(tr, te, 500e6, 2, 0.5)
        for yy, dd, aa in zip(te.target.to_numpy(float), d, a):
            inner_rows.append((yy, dd, aa))
    if len(inner_rows) >= 2:
        yy = np.array([x[0] for x in inner_rows], float)
        dd = np.array([x[1] for x in inner_rows], float)
        aa = np.array([x[2] for x in inner_rows], float)
        scores = {w: mean_absolute_error(yy, w * dd + (1 - w) * aa) for w in BLEND_WEIGHTS}
        blend = min(scores, key=scores.get)
    else:
        blend = 0.5
    return np.clip(blend * direct + (1 - blend) * analog, 500e6, 6e9), float(blend)


def main():
    d, features = v2.load_data()
    d = d[d.incidentType.astype(str).str.lower() != "biological"].copy()
    d["fyDeclared"] = pd.to_numeric(d.fyDeclared, errors="coerce").astype(int)
    ev = d[d.target > 500e6].copy().sort_values("disasterNumber")

    rows = []
    for year in sorted(ev.fyDeclared.unique()):
        train = d[d.fyDeclared != year].copy()
        test = ev[ev.fyDeclared == year].copy()
        if test.empty:
            continue

        baseline = extreme_baselines(train, len(test))
        best_direct_cfg, best_analog_cfg, blend_weight, inner = choose_components(train, features)
        ds, da, dw = best_direct_cfg
        asup, ak, ae = best_analog_cfg
        direct = direct_predict(train, test, features, ds, da, dw, 9000 + year)
        analog = analogue_predict(train, test, asup, ak, ae)
        hybrid = np.clip(blend_weight * direct + (1 - blend_weight) * analog, 500e6, 6e9)
        legacy, legacy_blend = legacy_fixed_hybrid(train, test, features, year)

        for k, (_, r) in enumerate(test.iterrows()):
            rows.append({
                "disasterNumber": int(r.disasterNumber),
                "state": r.state,
                "fyDeclared": int(year),
                "incidentType": r.incidentType,
                "target": float(r.target),
                "median_baseline": float(baseline["median"][k]),
                "geomean_baseline": float(baseline["geomean"][k]),
                "mean_baseline": float(baseline["mean"][k]),
                "selected_direct_prediction": float(direct[k]),
                "selected_analogue_prediction": float(analog[k]),
                "nested_hybrid_prediction": float(hybrid[k]),
                "legacy_reconstructed_hybrid_prediction": float(legacy[k]),
                "selected_direct_cfg": str(best_direct_cfg),
                "selected_analogue_cfg": str(best_analog_cfg),
                "selected_blend_weight": float(blend_weight),
                "legacy_blend_weight": float(legacy_blend),
                "inner_selection": json.dumps(inner),
            })

    out = pd.DataFrame(rows).sort_values("disasterNumber").reset_index(drop=True)
    out.to_csv(OUT / "predictions.csv", index=False)
    y = out.target.to_numpy(float)
    method_cols = {
        "training_extreme_median": "median_baseline",
        "training_extreme_geomean": "geomean_baseline",
        "training_extreme_mean": "mean_baseline",
        "nested_selected_direct": "selected_direct_prediction",
        "nested_selected_analogue": "selected_analogue_prediction",
        "nested_selected_hybrid": "nested_hybrid_prediction",
        "legacy_reconstructed_hybrid": "legacy_reconstructed_hybrid_prediction",
    }
    result = {name: metrics(y, out[col].to_numpy(float)) for name, col in method_cols.items()}
    ranking = sorted(result, key=lambda x: (result[x]["log_MAE"], result[x]["MAE"]))

    summary = {
        "scope": "non-Biological disasters with funding > $500M",
        "n": int(len(out)),
        "cases": out[["disasterNumber", "state", "fyDeclared", "incidentType", "target"]].to_dict("records"),
        "metrics": result,
        "ranking_by_log_MAE_then_MAE": ranking,
        "best_method": ranking[0] if ranking else None,
        "protocol": [
            "Biological/COVID rows are excluded.",
            "Every outer prediction excludes every row from the held fiscal year.",
            "Direct-model support, quantile, extreme weighting, analogue support, analogue k, analogue scaling, and hybrid blend are selected only from the outer-training data.",
            "Inner component selection uses leave-fiscal-year-out predictions on training extremes and optimizes log-MAE; blend weights are selected on the same inner training-only predictions.",
            "The legacy reconstructed hybrid is reproduced separately for comparison and is not represented as the unrecovered original Billion-dollar Hybrid Expert.",
            "Only five non-Biological >$500M cases are available, so all metrics are high-variance development-validation estimates rather than a definitive external test.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = ["# Non-Biological $500M+ strict hybrid audit", "", f"n = {len(out)}", ""]
    for name in ranking:
        m = result[name]
        lines.append(f"## {name}")
        for k, v in m.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
