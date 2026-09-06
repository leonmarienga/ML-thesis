#!/usr/bin/env python3
"""
Mission-semantic enrichment audit for the 971-row disaster-relief thesis dataset.

Purpose
-------
1. Reconstruct how the prepared master table relates to raw OpenFEMA
   MissionAssignments v2 records.
2. Build strictly non-financial mission-semantic features.
3. Test whether those features improve separation of the non-Biological
   high-value funding bands, especially the $500M+ tail.

This script intentionally excludes all raw MissionAssignments financial fields
from predictors. Financial fields are used only in a target-reconstruction
audit to understand how the historical target was created.
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import requests
from scipy.stats import mannwhitneyu
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MASTER_PATH = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "mission_semantic"
OUT.mkdir(parents=True, exist_ok=True)

API_URL = "https://www.fema.gov/api/open/v2/MissionAssignments"
PAGE_SIZE = 1000

CURRENT_19 = [
    "state",
    "incidentType",
    "expectedResourceLevel",
    "disasterCategory",
    "durationClass",
    "durationDays",
    "declarationDelayDays",
    "fyDeclared",
    "ihProgramDeclared",
    "paProgramDeclared",
    "hmProgramDeclared",
    "expectedResourceScore",
    "missionAssignmentCount",
    "uniqueAgencyCount",
    "uniqueMaTypeCount",
    "uniquePriorityCount",
    "responseComplexityScore",
    "missionDensity",
    "agencyDensity",
]

FINANCIAL_FIELDS = {
    "obligationAmount",
    "sttCostSharePct",
    "fedCostSharePct",
    "sttCostShareAmt",
    "fedCostShareAmt",
    "maSttCostSharePct",
    "maFedCostSharePct",
    "maSttCostShareAmount",
    "maFedCostShareAmount",
    "dateObligated",
}

TEXT_CATEGORIES: Dict[str, str] = {
    "power": r"\b(power|generator|generators|electric|electricity|energy|grid)\b",
    "debris": r"\bdebris\b",
    "housing_shelter": r"\b(housing|shelter|sheltering|lodging|temporary housing)\b",
    "logistics_commodities": r"\b(logistics|commodity|commodities|warehouse|warehousing|staging|distribution center|supply chain)\b",
    "transportation": r"\b(transportation|transport|trucking|truck|vehicle|vehicles|airlift|aviation|air transport)\b",
    "medical_health": r"\b(medical|health|hospital|hospitals|ambulance|ems|public health|patient)\b",
    "search_rescue": r"\b(search and rescue|search & rescue|urban search|usar|us&r|rescue team)\b",
    "engineering_infrastructure": r"\b(engineering|infrastructure|public works|bridge|bridges|road|roads|facility|facilities|structural)\b",
    "water_wastewater": r"\b(water system|drinking water|wastewater|sewer|sewage|water infrastructure)\b",
    "firefighting": r"\b(firefighting|fire fighting|fire suppression|wildfire|fire management)\b",
    "communications": r"\b(communications|communication|radio|telecom|telecommunications|satellite communications)\b",
    "mass_care_feeding": r"\b(mass care|feeding|meals|food service|food distribution)\b",
    "security_law_enforcement": r"\b(security|law enforcement|police|crowd control)\b",
    "environment_hazmat": r"\b(environmental|hazmat|hazardous material|hazardous materials|pollution|epa)\b",
    "damage_assessment": r"\b(damage assessment|assessment team|preliminary damage|assess damage)\b",
    "mortuary": r"\b(mortuary|fatality management|human remains)\b",
    "civil_affairs": r"\b(civil affairs|community liaison)\b",
}


def fetch_all_mission_assignments() -> pd.DataFrame:
    cache = OUT / "MissionAssignments_v2_raw.jsonl.gz"
    if cache.exists():
        return pd.read_json(cache, lines=True, compression="gzip")

    rows: List[dict] = []
    skip = 0
    session = requests.Session()
    session.headers.update({"User-Agent": "thesis-mission-semantic-audit/1.0"})

    while True:
        params = {
            "$top": PAGE_SIZE,
            "$skip": skip,
            "$orderby": "actionId asc",
        }
        for attempt in range(6):
            try:
                r = session.get(API_URL, params=params, timeout=90)
                r.raise_for_status()
                payload = r.json()
                batch = payload.get("MissionAssignments", [])
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        rows.extend(batch)
        print(f"Fetched {len(rows):,} MissionAssignments rows")
        if len(batch) < PAGE_SIZE:
            break
        skip += PAGE_SIZE
        if skip > 200_000:
            raise RuntimeError("Pagination safety limit exceeded")

    df = pd.DataFrame(rows)
    df.to_json(cache, orient="records", lines=True, compression="gzip")
    return df


def normalize_master(master: pd.DataFrame) -> pd.DataFrame:
    master = master.copy()
    master["disasterNumber"] = pd.to_numeric(master["disasterNumber"], errors="coerce").astype("Int64")
    master["totalObligatedFunding"] = pd.to_numeric(master["totalObligatedFunding"], errors="coerce")
    master["declarationDate_dt"] = pd.to_datetime(master["declarationDate"], errors="coerce", utc=True)
    return master


def latest_assignment_state(ma: pd.DataFrame) -> pd.DataFrame:
    x = ma.copy()
    x["maAmendNumber_num"] = pd.to_numeric(x["maAmendNumber"], errors="coerce").fillna(-1)
    x["actionId_num"] = pd.to_numeric(x["actionId"], errors="coerce").fillna(-1)
    x = x.sort_values(["disasterNumber", "maId", "maAmendNumber_num", "actionId_num"])
    return x.groupby(["disasterNumber", "maId"], as_index=False, dropna=False).tail(1)


def safe_days(a: pd.Series, b: pd.Series) -> pd.Series:
    aa = pd.to_datetime(a, errors="coerce", utc=True)
    bb = pd.to_datetime(b, errors="coerce", utc=True)
    return (bb - aa).dt.total_seconds() / 86400.0


def sanitize(s: object) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(s).strip()).strip("_").lower()
    return text[:60] or "unknown"


def build_semantic_rollup(master: pd.DataFrame, ma_all: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    master_ids = set(master["disasterNumber"].dropna().astype(int).tolist())
    ma = ma_all.copy()
    ma["disasterNumber"] = pd.to_numeric(ma["disasterNumber"], errors="coerce").astype("Int64")
    ma = ma[ma["disasterNumber"].isin(master_ids)].copy()

    # Raw transaction/amendment diagnostics.
    raw = ma.groupby("disasterNumber").agg(
        ma_raw_row_count=("maId", "size"),
        ma_unique_id_count=("maId", "nunique"),
        ma_raw_unique_agency=("agencyId", "nunique"),
        ma_raw_unique_esf=("supportFunction", "nunique"),
    ).reset_index()

    amend = ma.copy()
    amend["maAmendNumber_num"] = pd.to_numeric(amend["maAmendNumber"], errors="coerce").fillna(0)
    amend_stats = amend.groupby("disasterNumber").agg(
        ma_amendment_rows=("maAmendNumber_num", lambda s: int((s > 0).sum())),
        ma_max_amendment=("maAmendNumber_num", "max"),
        ma_mean_amendment=("maAmendNumber_num", "mean"),
    ).reset_index()
    raw = raw.merge(amend_stats, on="disasterNumber", how="outer")
    raw["ma_amendment_share"] = raw["ma_amendment_rows"] / raw["ma_raw_row_count"].replace(0, np.nan)

    latest = latest_assignment_state(ma)

    latest["ma_duration_days"] = safe_days(latest["popStartDate"], latest["popEndDate"])
    latest["ma_duration_days"] = latest["ma_duration_days"].clip(lower=0, upper=3650)
    latest["dateReceived_dt"] = pd.to_datetime(latest["dateReceived"], errors="coerce", utc=True)

    text = (
        latest["assistanceRequested"].fillna("").astype(str)
        + " "
        + latest["statementOfWork"].fillna("").astype(str)
    ).str.lower()
    for cat, pat in TEXT_CATEGORIES.items():
        latest[f"text_{cat}"] = text.str.contains(pat, regex=True, na=False).astype(int)

    # General latest-state rollups.
    sem = latest.groupby("disasterNumber").agg(
        sem_unique_missions=("maId", "nunique"),
        sem_unique_agencies=("agencyId", "nunique"),
        sem_unique_esf=("supportFunction", "nunique"),
        sem_unique_types=("maType", "nunique"),
        sem_unique_priorities=("priority", "nunique"),
        sem_duration_mean=("ma_duration_days", "mean"),
        sem_duration_median=("ma_duration_days", "median"),
        sem_duration_max=("ma_duration_days", "max"),
        sem_duration_p90=("ma_duration_days", lambda s: s.quantile(0.90)),
        sem_long_30d_count=("ma_duration_days", lambda s: int((s >= 30).sum())),
        sem_long_90d_count=("ma_duration_days", lambda s: int((s >= 90).sum())),
    ).reset_index()

    # Mission types.
    for value in sorted(latest["maType"].dropna().astype(str).unique()):
        c = latest.assign(_v=(latest["maType"].astype(str) == value).astype(int)).groupby("disasterNumber")["_v"].sum()
        name = f"sem_type_{sanitize(value)}_count"
        sem = sem.merge(c.rename(name), on="disasterNumber", how="left")
        sem[f"{name[:-6]}share"] = sem[name] / sem["sem_unique_missions"].replace(0, np.nan)

    # ESF/supportFunction composition.
    sf = pd.to_numeric(latest["supportFunction"], errors="coerce")
    for value in sorted(sf.dropna().astype(int).unique()):
        c = latest.assign(_v=(sf == value).astype(int)).groupby("disasterNumber")["_v"].sum()
        name = f"sem_esf_{value}_count"
        sem = sem.merge(c.rename(name), on="disasterNumber", how="left")
        sem[f"sem_esf_{value}_share"] = sem[name] / sem["sem_unique_missions"].replace(0, np.nan)

    # Top agency IDs chosen without using the target.
    agency_counts = latest["agencyId"].fillna("UNKNOWN").astype(str).value_counts()
    top_agencies = agency_counts.head(25).index.tolist()
    for value in top_agencies:
        c = latest.assign(_v=(latest["agencyId"].fillna("UNKNOWN").astype(str) == value).astype(int)).groupby("disasterNumber")["_v"].sum()
        key = sanitize(value)
        name = f"sem_agency_{key}_count"
        sem = sem.merge(c.rename(name), on="disasterNumber", how="left")
        sem[f"sem_agency_{key}_share"] = sem[name] / sem["sem_unique_missions"].replace(0, np.nan)

    # Priority composition.
    for value in sorted(latest["priority"].dropna().astype(str).unique()):
        c = latest.assign(_v=(latest["priority"].astype(str) == value).astype(int)).groupby("disasterNumber")["_v"].sum()
        key = sanitize(value)
        name = f"sem_priority_{key}_count"
        sem = sem.merge(c.rename(name), on="disasterNumber", how="left")
        sem[f"sem_priority_{key}_share"] = sem[name] / sem["sem_unique_missions"].replace(0, np.nan)

    # Rule-based task topics.
    for cat in TEXT_CATEGORIES:
        col = f"text_{cat}"
        c = latest.groupby("disasterNumber")[col].sum()
        name = f"sem_topic_{cat}_count"
        sem = sem.merge(c.rename(name), on="disasterNumber", how="left")
        sem[f"sem_topic_{cat}_share"] = sem[name] / sem["sem_unique_missions"].replace(0, np.nan)

    sem = raw.merge(sem, on="disasterNumber", how="outer")
    sem = sem.replace([np.inf, -np.inf], np.nan)
    return sem, latest


def build_timing_rollup(master: pd.DataFrame, latest: pd.DataFrame) -> pd.DataFrame:
    decl = master[["disasterNumber", "declarationDate_dt"]].drop_duplicates()
    x = latest.merge(decl, on="disasterNumber", how="left")
    x["days_received_from_declaration"] = (
        x["dateReceived_dt"] - x["declarationDate_dt"]
    ).dt.total_seconds() / 86400.0

    out = decl[["disasterNumber"]].copy()
    for day in [0, 3, 7, 14, 30]:
        g = x[x["days_received_from_declaration"] <= day].groupby("disasterNumber")
        vals = g.agg(
            missions=("maId", "nunique"),
            agencies=("agencyId", "nunique"),
            esf=("supportFunction", "nunique"),
        ).rename(columns=lambda c: f"t{day}_{c}")
        out = out.merge(vals, on="disasterNumber", how="left")
    return out.fillna(0)


def target_reconstruction(master: pd.DataFrame, ma: pd.DataFrame, latest: pd.DataFrame) -> pd.DataFrame:
    x = ma.copy()
    for c in ["obligationAmount", "fedCostShareAmt", "sttCostShareAmt"]:
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)

    sums = x.groupby("disasterNumber").agg(
        api_sum_obligation_all=("obligationAmount", "sum"),
        api_sum_fed_share_all=("fedCostShareAmt", "sum"),
        api_sum_state_share_all=("sttCostShareAmt", "sum"),
        api_raw_rows=("maId", "size"),
        api_unique_ma_ids=("maId", "nunique"),
    ).reset_index()

    lat = latest.copy()
    lat["obligationAmount"] = pd.to_numeric(lat["obligationAmount"], errors="coerce").fillna(0.0)
    latest_sums = lat.groupby("disasterNumber")["obligationAmount"].sum().rename("api_sum_obligation_latest").reset_index()

    z = master[
        [
            "disasterNumber",
            "totalObligatedFunding",
            "missionAssignmentCount",
            "uniqueAgencyCount",
            "uniqueMaTypeCount",
            "uniquePriorityCount",
        ]
    ].merge(sums, on="disasterNumber", how="left").merge(latest_sums, on="disasterNumber", how="left")

    for c in ["api_sum_obligation_all", "api_sum_fed_share_all", "api_sum_obligation_latest"]:
        z[f"delta_{c}"] = z[c] - z["totalObligatedFunding"]
        z[f"abs_delta_{c}"] = z[f"delta_{c}"].abs()
        denom = z["totalObligatedFunding"].abs().clip(lower=1.0)
        z[f"rel_delta_{c}"] = z[f"abs_delta_{c}"] / denom

    z["count_delta_raw"] = z["api_raw_rows"] - z["missionAssignmentCount"]
    z["count_delta_unique"] = z["api_unique_ma_ids"] - z["missionAssignmentCount"]
    return z


def normalize_model_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Make pandas booleans sklearn-1.8-safe without changing their information."""
    X = X.copy()
    for c in X.columns:
        if pd.api.types.is_bool_dtype(X[c].dtype):
            X[c] = X[c].astype("int8")
    return X


def prep_pipeline(X: pd.DataFrame, model):
    # pandas 2.x may use StringDtype instead of object for text columns.
    cats = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c].dtype)]
    nums = [c for c in X.columns if c not in cats]
    pre = ColumnTransformer(
        [
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore")),
            ]), cats),
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scale", StandardScaler(with_mean=False)),
            ]), nums),
        ],
        remainder="drop",
    )
    return Pipeline([("pre", pre), ("model", model)])


def loo_classification(df: pd.DataFrame, features: List[str], target: pd.Series, model_kind: str) -> Dict:
    y = np.asarray(target)
    pred = np.empty(len(df), dtype=int)
    proba = np.empty(len(df), dtype=float)

    for i in range(len(df)):
        train = np.arange(len(df)) != i
        Xtr = normalize_model_frame(df.iloc[train][features])
        Xte = normalize_model_frame(df.iloc[[i]][features])
        ytr = y[train]

        if model_kind == "logistic":
            model = LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5)
        else:
            model = RandomForestClassifier(
                n_estimators=500,
                random_state=42 + i,
                class_weight="balanced_subsample",
                min_samples_leaf=1,
                max_features="sqrt",
            )
        pipe = prep_pipeline(Xtr, model)
        pipe.fit(Xtr, ytr)
        pred[i] = int(pipe.predict(Xte)[0])
        if len(np.unique(y)) == 2 and hasattr(pipe[-1], "predict_proba"):
            cls = list(pipe[-1].classes_)
            p = pipe.predict_proba(Xte)[0]
            proba[i] = p[cls.index(1)] if 1 in cls else 0.0
        else:
            proba[i] = np.nan

    return {
        "model": model_kind,
        "n": int(len(df)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "recall_extreme": float(recall_score(y, pred, pos_label=1, zero_division=0)) if len(np.unique(y)) == 2 else None,
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "predictions": pred.tolist(),
        "prob_extreme": [None if np.isnan(v) else float(v) for v in proba],
    }


def loo_regression(df: pd.DataFrame, features: List[str]) -> Dict:
    y_dollar = df["totalObligatedFunding"].to_numpy(dtype=float)
    y = np.log1p(np.clip(y_dollar, 0, None))
    pred_log = np.zeros(len(df))
    for i in range(len(df)):
        train = np.arange(len(df)) != i
        Xtr = normalize_model_frame(df.iloc[train][features])
        Xte = normalize_model_frame(df.iloc[[i]][features])
        model = RandomForestRegressor(
            n_estimators=600,
            random_state=142 + i,
            min_samples_leaf=1,
            max_features=0.7,
        )
        pipe = prep_pipeline(Xtr, model)
        pipe.fit(Xtr, y[train])
        pred_log[i] = float(pipe.predict(Xte)[0])
    pred_dollar = np.expm1(pred_log)
    return {
        "n": len(df),
        "log_mae": float(mean_absolute_error(y, pred_log)),
        "log_r2": float(r2_score(y, pred_log)),
        "dollar_mae": float(mean_absolute_error(y_dollar, pred_dollar)),
        "predictions": pred_dollar.tolist(),
    }


def effect_table(high: pd.DataFrame, semantic_cols: List[str]) -> pd.DataFrame:
    a = high[high["is_extreme"] == 1]
    b = high[high["is_extreme"] == 0]
    rows = []
    for c in semantic_cols:
        xa = pd.to_numeric(a[c], errors="coerce").dropna()
        xb = pd.to_numeric(b[c], errors="coerce").dropna()
        if len(xa) < 2 or len(xb) < 2:
            continue
        if xa.nunique() <= 1 and xb.nunique() <= 1 and xa.iloc[0] == xb.iloc[0]:
            continue
        try:
            u, p = mannwhitneyu(xa, xb, alternative="two-sided")
        except Exception:
            u, p = np.nan, np.nan
        pooled = pd.concat([xa, xb]).std(ddof=1)
        effect = (xa.mean() - xb.mean()) / pooled if pooled and np.isfinite(pooled) else np.nan
        rows.append(
            {
                "feature": c,
                "extreme_median": xa.median(),
                "lower_median": xb.median(),
                "extreme_mean": xa.mean(),
                "lower_mean": xb.mean(),
                "std_mean_difference": effect,
                "mannwhitney_p": p,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_effect"] = out["std_mean_difference"].abs()
        out = out.sort_values(["abs_effect", "mannwhitney_p"], ascending=[False, True])
    return out


def main():
    print("Reading master workbook...")
    master = normalize_master(pd.read_excel(MASTER_PATH))
    assert len(master) == 971, f"Expected 971 rows, found {len(master)}"
    assert master["disasterNumber"].nunique() == 971, "disasterNumber must be unique"

    print("Fetching OpenFEMA MissionAssignments v2...")
    ma_all = fetch_all_mission_assignments()
    print("MissionAssignments rows:", len(ma_all))

    sem, latest = build_semantic_rollup(master, ma_all)
    timing = build_timing_rollup(master, latest)
    recon = target_reconstruction(
        master,
        ma_all[pd.to_numeric(ma_all["disasterNumber"], errors="coerce").isin(master["disasterNumber"])].copy(),
        latest,
    )

    enriched = master.merge(sem, on="disasterNumber", how="left").merge(timing, on="disasterNumber", how="left")
    semantic_cols = [c for c in enriched.columns if c.startswith("sem_") or c.startswith("ma_")]
    enriched[semantic_cols] = enriched[semantic_cols].fillna(0)

    # Coverage.
    coverage = {
        "master_rows": int(len(master)),
        "master_with_any_api_ma": int((enriched["ma_raw_row_count"].fillna(0) > 0).sum()),
        "coverage_pct": float((enriched["ma_raw_row_count"].fillna(0) > 0).mean() * 100),
        "api_rows_matching_master_disasters": int(enriched["ma_raw_row_count"].fillna(0).sum()),
        "unique_latest_missions_matching_master": int(enriched["sem_unique_missions"].fillna(0).sum()),
        "top_agencies": latest["agencyId"].fillna("UNKNOWN").astype(str).value_counts().head(25).to_dict(),
        "support_functions": sorted(pd.to_numeric(latest["supportFunction"], errors="coerce").dropna().astype(int).unique().tolist()),
    }

    # Reconstruction metrics.
    rvalid = recon.dropna(subset=["totalObligatedFunding"]).copy()
    reconstruct = {}
    for c in ["api_sum_obligation_all", "api_sum_fed_share_all", "api_sum_obligation_latest"]:
        ok = rvalid[c].notna()
        rel = rvalid.loc[ok, f"rel_delta_{c}"]
        reconstruct[c] = {
            "n": int(ok.sum()),
            "median_abs_delta": float(rvalid.loc[ok, f"abs_delta_{c}"].median()),
            "median_relative_delta": float(rel.median()),
            "within_1pct": float((rel <= 0.01).mean()),
            "within_5pct": float((rel <= 0.05).mean()),
            "within_10pct": float((rel <= 0.10).mean()),
        }
    reconstruct["mission_count_vs_raw_rows_exact"] = float((recon["count_delta_raw"].fillna(np.inf) == 0).mean())
    reconstruct["mission_count_vs_unique_ids_exact"] = float((recon["count_delta_unique"].fillna(np.inf) == 0).mean())

    # High-band non-Biological set.
    high = enriched[
        (enriched["incidentType"] != "Biological")
        & (enriched["totalObligatedFunding"] >= 50_000_000)
    ].copy()
    high["funding_band"] = pd.cut(
        high["totalObligatedFunding"],
        bins=[50_000_000, 200_000_000, 500_000_000, np.inf],
        labels=["50-200M", "200-500M", "500M+"],
        right=False,
    ).astype(str)
    high["is_extreme"] = (high["totalObligatedFunding"] >= 500_000_000).astype(int)
    print("High-band counts:", high["funding_band"].value_counts().to_dict())

    # Keep semantic model compact and target-blind.
    compact_sem = [
        c for c in semantic_cols
        if (
            c in {
                "ma_raw_row_count", "ma_unique_id_count", "ma_amendment_rows",
                "ma_amendment_share", "ma_max_amendment",
                "sem_unique_missions", "sem_unique_agencies", "sem_unique_esf",
                "sem_unique_types", "sem_duration_mean", "sem_duration_median",
                "sem_duration_max", "sem_duration_p90", "sem_long_30d_count",
                "sem_long_90d_count",
            }
            or c.startswith("sem_type_")
            or c.startswith("sem_esf_")
            or c.startswith("sem_topic_")
        )
    ]
    compact_sem = [c for c in compact_sem if high[c].nunique(dropna=False) > 1]

    current_features = [c for c in CURRENT_19 if c in high.columns]
    added_features = current_features + compact_sem

    diagnostics = {
        "binary_extreme_vs_lower": {},
        "high_band_regression": {},
    }
    for kind in ["logistic", "rf"]:
        diagnostics["binary_extreme_vs_lower"][f"current19_{kind}"] = loo_classification(
            high.reset_index(drop=True), current_features, high["is_extreme"].reset_index(drop=True), kind
        )
        diagnostics["binary_extreme_vs_lower"][f"current19_plus_semantic_{kind}"] = loo_classification(
            high.reset_index(drop=True), added_features, high["is_extreme"].reset_index(drop=True), kind
        )

    diagnostics["high_band_regression"]["current19_rf"] = loo_regression(high.reset_index(drop=True), current_features)
    diagnostics["high_band_regression"]["current19_plus_semantic_rf"] = loo_regression(high.reset_index(drop=True), added_features)

    effects = effect_table(high, compact_sem)
    effects.to_csv(OUT / "semantic_effects_extreme_vs_lower.csv", index=False)

    # Pairwise/case view.
    case_ids = [4671, 4339, 4344, 4724, 4407, 4353, 4652, 4332, 4611]
    case_cols = [
        "disasterNumber", "state", "incidentType", "fyDeclared", "totalObligatedFunding",
        "missionAssignmentCount", "ma_raw_row_count", "ma_unique_id_count",
        "ma_amendment_rows", "ma_amendment_share", "sem_unique_missions",
        "sem_unique_agencies", "sem_unique_esf", "sem_duration_mean",
        "sem_duration_max", "t0_missions", "t3_missions", "t7_missions",
        "t14_missions", "t30_missions",
    ] + [c for c in compact_sem if c.startswith("sem_topic_") and c.endswith("_share")]
    case_view = enriched[enriched["disasterNumber"].isin(case_ids)][case_cols].sort_values("totalObligatedFunding")
    case_view.to_csv(OUT / "key_cases_semantics.csv", index=False)

    high_out_cols = [
        "disasterNumber", "state", "incidentType", "fyDeclared", "totalObligatedFunding",
        "funding_band", "is_extreme",
    ] + current_features + compact_sem + [
        "t0_missions", "t3_missions", "t7_missions", "t14_missions", "t30_missions"
    ]
    high[high_out_cols].to_csv(OUT / "high_band_mission_semantics.csv", index=False)
    recon.to_csv(OUT / "target_reconstruction_audit.csv", index=False)
    timing.to_csv(OUT / "mission_timing_coverage.csv", index=False)

    # Top feature differences for markdown summary.
    top_effects = effects.head(20).to_dict(orient="records") if not effects.empty else []

    summary = {
        "coverage": coverage,
        "target_reconstruction": reconstruct,
        "high_band_counts": high["funding_band"].value_counts().to_dict(),
        "semantic_feature_count": len(compact_sem),
        "diagnostics": diagnostics,
        "top_extreme_effects": top_effects,
        "key_case_ids_requested": case_ids,
        "financial_predictor_fields_excluded": sorted(FINANCIAL_FIELDS),
        "notes": [
            "All semantic predictors are non-financial.",
            "Financial fields are used only for target reconstruction diagnostics.",
            "The final OpenFEMA snapshot may contain amendments made after the original master file was created.",
            "The LOOCV high-band diagnostics are exploratory because n=23 and the extreme class has n=5.",
        ],
    }
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # Human-readable summary.
    b = diagnostics["binary_extreme_vs_lower"]
    reg = diagnostics["high_band_regression"]
    md = [
        "# Mission-semantic enrichment audit",
        "",
        f"- Master rows: **{coverage['master_rows']}**",
        f"- Rows matched to current OpenFEMA MissionAssignments: **{coverage['master_with_any_api_ma']} ({coverage['coverage_pct']:.1f}%)**",
        f"- High-band counts: **{summary['high_band_counts']}**",
        f"- Compact non-financial semantic features tested: **{len(compact_sem)}**",
        "",
        "## Target reconstruction",
        "",
        f"- missionAssignmentCount == current raw API row count: **{reconstruct['mission_count_vs_raw_rows_exact']:.1%}** of all rows",
        f"- missionAssignmentCount == current unique maId count: **{reconstruct['mission_count_vs_unique_ids_exact']:.1%}** of all rows",
        f"- all-row obligation sum within 5% of master target: **{reconstruct['api_sum_obligation_all']['within_5pct']:.1%}**",
        "",
        "## Extreme-band LOOCV ($500M+ vs $50M-$500M)",
        "",
        f"- Current 19 / logistic: balanced accuracy **{b['current19_logistic']['balanced_accuracy']:.3f}**, extreme recall **{b['current19_logistic']['recall_extreme']:.3f}**",
        f"- + semantics / logistic: balanced accuracy **{b['current19_plus_semantic_logistic']['balanced_accuracy']:.3f}**, extreme recall **{b['current19_plus_semantic_logistic']['recall_extreme']:.3f}**",
        f"- Current 19 / random forest: balanced accuracy **{b['current19_rf']['balanced_accuracy']:.3f}**, extreme recall **{b['current19_rf']['recall_extreme']:.3f}**",
        f"- + semantics / random forest: balanced accuracy **{b['current19_plus_semantic_rf']['balanced_accuracy']:.3f}**, extreme recall **{b['current19_plus_semantic_rf']['recall_extreme']:.3f}**",
        "",
        "## High-band regression LOOCV",
        "",
        f"- Current 19 RF log MAE: **{reg['current19_rf']['log_mae']:.3f}**",
        f"- + semantic RF log MAE: **{reg['current19_plus_semantic_rf']['log_mae']:.3f}**",
        "",
        "## Important caution",
        "",
        "This is a diagnostic final-snapshot experiment. A deployment-safe model must rebuild these features at a defined t0 using only mission records available by that time.",
    ]
    (OUT / "audit_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
