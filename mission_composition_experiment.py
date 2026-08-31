from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

MASTER = Path("master_openfema_40plus.xlsx")
OUT = Path("mission_composition_results")
OUT.mkdir(exist_ok=True)
MA_URL = "https://www.fema.gov/api/open/v2/MissionAssignments"
CACHE = OUT / "MissionAssignments_v2.jsonl"

SAFE_BASE_FEATURES = [
    "state", "fyDeclared", "incidentType", "ihProgramDeclared", "paProgramDeclared",
    "hmProgramDeclared", "durationDays", "declarationDelayDays", "expectedResourceLevel",
    "expectedResourceScore", "disasterCategory", "durationClass", "missionAssignmentCount",
    "uniqueAgencyCount", "uniqueMaTypeCount", "uniquePriorityCount", "responseComplexityScore",
    "missionDensity", "agencyDensity",
]


def fetch_openfema_for_disasters(disaster_numbers: list[int], chunk_size: int = 40) -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_json(CACHE, lines=True)

    rows: list[dict] = []
    session = requests.Session()
    for start in range(0, len(disaster_numbers), chunk_size):
        chunk = disaster_numbers[start:start + chunk_size]
        filt = " or ".join(f"disasterNumber eq {int(n)}" for n in chunk)
        skip = 0
        while True:
            params = {"$filter": filt, "$top": 1000, "$skip": skip}
            r = session.get(MA_URL, params=params, timeout=60)
            r.raise_for_status()
            payload = r.json()
            batch = payload.get("MissionAssignments", [])
            rows.extend(batch)
            if len(batch) < 1000:
                break
            skip += 1000
            time.sleep(0.05)
        print(f"Downloaded MissionAssignments for {min(start+chunk_size, len(disaster_numbers))}/{len(disaster_numbers)} disasters; rows={len(rows):,}")

    raw = pd.DataFrame(rows)
    raw.to_json(CACHE, orient="records", lines=True)
    return raw


def entropy_from_counts(counts: pd.Series) -> float:
    x = counts.to_numpy(float)
    s = x.sum()
    if s <= 0:
        return 0.0
    p = x / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def hhi_from_counts(counts: pd.Series) -> float:
    x = counts.to_numpy(float)
    s = x.sum()
    if s <= 0:
        return 0.0
    p = x / s
    return float(np.square(p).sum())


def top_share(counts: pd.Series) -> float:
    s = counts.sum()
    return float(counts.max() / s) if s > 0 else 0.0


def composition_features(raw: pd.DataFrame, disaster_numbers: list[int]) -> pd.DataFrame:
    required = ["disasterNumber", "agency", "maType", "priority"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise RuntimeError(f"MissionAssignments is missing required non-target fields: {missing}")

    m = raw.copy()
    m["disasterNumber"] = pd.to_numeric(m["disasterNumber"], errors="coerce")
    m = m[m["disasterNumber"].notna()].copy()
    m["disasterNumber"] = m["disasterNumber"].astype(int)
    m = m[m["disasterNumber"].isin(disaster_numbers)].copy()

    # IMPORTANT: obligationAmount/dateObligated and every funding-derived field are intentionally ignored.
    for c in ["agency", "maType", "priority"]:
        m[c] = m[c].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")

    base = pd.DataFrame({"disasterNumber": disaster_numbers}).drop_duplicates()

    for field, prefix in [("agency", "agency"), ("maType", "matype"), ("priority", "priority")]:
        ct = pd.crosstab(m["disasterNumber"], m[field])
        totals = ct.sum(axis=1).replace(0, np.nan)

        # Retain categories seen in at least 3 disasters to avoid an enormous sparse feature space.
        keep = ct.columns[(ct > 0).sum(axis=0) >= 3]
        ct = ct[keep]
        share = ct.div(totals, axis=0).fillna(0)

        ct.columns = [f"{prefix}_count__{str(c)[:60]}" for c in ct.columns]
        share.columns = [c.replace(f"{prefix}_count__", f"{prefix}_share__") for c in ct.columns]

        ct = ct.reset_index()
        share = share.reset_index()
        base = base.merge(ct, how="left", on="disasterNumber").merge(share, how="left", on="disasterNumber")

        stats = m.groupby("disasterNumber")[field].value_counts().rename("n").reset_index()
        agg = stats.groupby("disasterNumber")["n"].agg(
            **{
                f"{prefix}_entropy": entropy_from_counts,
                f"{prefix}_hhi": hhi_from_counts,
                f"{prefix}_top_share": top_share,
            }
        ).reset_index()
        base = base.merge(agg, how="left", on="disasterNumber")

    # Cross-composition: agency x MA type and MA type x priority. Keep recurring combinations only.
    for left, right, prefix in [("agency", "maType", "agency_matype"), ("maType", "priority", "matype_priority")]:
        combo = m[left].astype(str) + "__" + m[right].astype(str)
        tmp = pd.DataFrame({"disasterNumber": m["disasterNumber"], "combo": combo})
        ct = pd.crosstab(tmp["disasterNumber"], tmp["combo"])
        keep = ct.columns[(ct > 0).sum(axis=0) >= 4]
        ct = ct[keep]
        totals = ct.sum(axis=1).replace(0, np.nan)
        sh = ct.div(totals, axis=0).fillna(0)
        ct.columns = [f"{prefix}_count__{str(c)[:70]}" for c in ct.columns]
        sh.columns = [c.replace(f"{prefix}_count__", f"{prefix}_share__") for c in ct.columns]
        base = base.merge(ct.reset_index(), how="left", on="disasterNumber").merge(sh.reset_index(), how="left", on="disasterNumber")

    return base.fillna(0)


def make_preprocessor(X: pd.DataFrame):
    cat = [c for c in X.columns if X[c].dtype == "object"]
    num = [c for c in X.columns if c not in cat]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=2))]), cat),
    ]), cat, num


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "R2": float(r2_score(y, p)),
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(mean_squared_error(y, p) ** 0.5),
        "MedAE": float(median_absolute_error(y, p)),
    }


def main() -> None:
    master = pd.read_excel(MASTER)
    master["target"] = pd.to_numeric(master["totalObligatedFunding"], errors="coerce").fillna(0).clip(lower=0)
    disaster_numbers = master["disasterNumber"].astype(int).tolist()

    raw = fetch_openfema_for_disasters(disaster_numbers)
    feats = composition_features(raw, disaster_numbers)
    feats.to_csv(OUT / "mission_composition_features.csv", index=False)

    data = master[["disasterNumber", "target"] + SAFE_BASE_FEATURES].merge(feats, on="disasterNumber", how="left").fillna(0)

    # Main evaluation region requested in the thesis development work.
    mask = (data["target"] > 50_000_000) & (data["target"] <= 500_000_000)
    d = data.loc[mask].reset_index(drop=True)
    y = d["target"].to_numpy(float)
    X = d.drop(columns=["target", "disasterNumber"])
    strat = np.digitize(y, [100e6, 200e6, 300e6])

    seeds = [42, 123, 777, 2026, 99]
    model_names = ["ridge", "extratrees", "xgb"]
    if LGBMRegressor is not None:
        model_names.append("lgbm")
    if CatBoostRegressor is not None:
        model_names.append("catboost")

    all_predictions = {name: [] for name in model_names}
    run_rows = []

    for seed in seeds:
        folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
        pred = {name: np.zeros(len(y)) for name in model_names}

        for tr, te in folds.split(X, strat):
            pre, _, _ = make_preprocessor(X.iloc[tr])
            xt = pre.fit_transform(X.iloc[tr])
            xe = pre.transform(X.iloc[te])
            yt_log = np.log1p(y[tr])

            models = {
                "ridge": Ridge(alpha=25.0),
                "extratrees": ExtraTreesRegressor(n_estimators=800, max_depth=8, min_samples_leaf=2, max_features=0.7, random_state=seed, n_jobs=-1),
                "xgb": XGBRegressor(n_estimators=700, max_depth=3, learning_rate=0.02, subsample=0.85, colsample_bytree=0.75, reg_lambda=20, reg_alpha=1, objective="reg:squarederror", random_state=seed, n_jobs=-1),
            }
            if LGBMRegressor is not None:
                models["lgbm"] = LGBMRegressor(n_estimators=500, learning_rate=0.02, num_leaves=7, max_depth=3, min_child_samples=5, reg_lambda=15, reg_alpha=1, verbosity=-1, random_state=seed, n_jobs=-1)
            if CatBoostRegressor is not None:
                models["catboost"] = CatBoostRegressor(iterations=700, depth=4, learning_rate=0.025, loss_function="RMSE", l2_leaf_reg=15, verbose=False, random_seed=seed)

            for name, model in models.items():
                model.fit(xt, yt_log)
                pred[name][te] = np.expm1(model.predict(xe)).clip(min=0)

        for name in model_names:
            all_predictions[name].append(pred[name])
            row = {"seed": seed, "model": name, **metrics(y, pred[name])}
            run_rows.append(row)

    # Repeated-OOF average per model.
    avg_preds = {}
    for name in model_names:
        avg_preds[name] = np.mean(all_predictions[name], axis=0)
        run_rows.append({"seed": "avg", "model": name, **metrics(y, avg_preds[name])})

    # Blend search uses only OOF predictions; report as development screening, not untouched validation.
    names = list(avg_preds)
    blend_candidates = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for w in [0.25, 0.5, 0.75]:
                p = w * avg_preds[names[i]] + (1 - w) * avg_preds[names[j]]
                blend_candidates.append({"blend": f"{w:.2f}*{names[i]}+{1-w:.2f}*{names[j]}", **metrics(y, p), "prediction": p})
    best_blend = max(blend_candidates, key=lambda x: x["R2"])

    pd.DataFrame(run_rows).to_csv(OUT / "repeated_oof_model_results.csv", index=False)
    pd.DataFrame([{k: v for k, v in b.items() if k != "prediction"} for b in blend_candidates]).sort_values("R2", ascending=False).to_csv(OUT / "blend_screen.csv", index=False)

    outpred = pd.DataFrame({"disasterNumber": d["disasterNumber"], "actual": y})
    for name, p in avg_preds.items():
        outpred[f"pred_{name}"] = p
    outpred["pred_best_blend"] = best_blend["prediction"]
    outpred.to_csv(OUT / "repeated_oof_predictions.csv", index=False)

    # Diagnose whether $200M–$500M remains the bottleneck.
    rows = []
    p = best_blend["prediction"]
    for label, lo, hi in [("50-100M", 50e6, 100e6), ("100-200M", 100e6, 200e6), ("200-300M", 200e6, 300e6), ("300-500M", 300e6, 500e6), ("200-500M", 200e6, 500e6), ("50-500M", 50e6, 500e6)]:
        mm = (y > lo) & (y <= hi)
        if mm.sum() >= 2:
            rows.append({"range": label, "n": int(mm.sum()), **metrics(y[mm], p[mm])})
    pd.DataFrame(rows).to_csv(OUT / "range_diagnostics.csv", index=False)

    summary = {
        "best_blend": {k: v for k, v in best_blend.items() if k != "prediction"},
        "target_r2": 0.80,
        "feature_count": int(X.shape[1]),
        "mission_rows": int(len(raw)),
        "evaluation_cases_50_500m": int(len(d)),
        "note": "All MissionAssignments funding/obligation fields are excluded from predictors. Blend selection is development screening; freeze before final validation.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
