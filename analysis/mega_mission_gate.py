#!/usr/bin/env python3
"""
Strict-LFYO mega Mission Assignment detector.

Question
--------
Can the initial NON-FINANCIAL characteristics of individual Mission
Assignments predict the rare assignments that ultimately contribute >= $500M,
and does that out-of-year signal improve disaster-level $500M+ routing?

Leakage controls
----------------
* obligationAmount is used ONLY to construct the mission-level training label.
* The feature row is the INITIAL action/amendment for each maId.
* maId/actionId/dateObligated/cost-share dollar fields are excluded.
* All digits are removed from free text so explicit cost estimates or resource
  quantities cannot leak scale through the SOW/request wording.
* For every held-out fiscal year, the mission model is trained only on missions
  belonging to OTHER fiscal years.
* The disaster-level gate uses only out-of-year mega-MA probabilities.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mission_semantic_audit import (
    CURRENT_19,
    build_semantic_rollup,
    fetch_all_mission_assignments,
    normalize_master,
    normalize_model_frame,
    prep_pipeline,
)
from external_severity_ablation import build_external

ROOT = Path(__file__).resolve().parents[1]
MASTER_PATH = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "mega_mission_gate"
OUT.mkdir(parents=True, exist_ok=True)

MEGA_THRESHOLD = 500_000_000.0

FINANCIAL_OR_ID_EXCLUSIONS = {
    "obligationAmount", "fedCostShareAmt", "sttCostShareAmt",
    "dateObligated", "actionId", "maId",
}

MISSION_CATS = ["agencyId", "supportFunction", "maType", "priority", "state", "incidentType"]
MISSION_NUMS = [
    "initial_pop_duration_days",
    "initial_required_lead_days",
    "initial_days_received_from_declaration",
]

def redact_text(x: object) -> str:
    s = "" if pd.isna(x) else str(x)
    s = s.lower()
    # Remove URLs/emails, currency symbols, all digits and compact amounts.
    s = re.sub(r"https?://\\S+|www\\.\\S+|\\S+@\\S+", " ", s)
    s = re.sub(r"\\$|usd|dollars?|million|billion|thousand", " ", s)
    s = re.sub(r"\\d+(?:[.,]\\d+)*", " ", s)
    s = re.sub(r"[^a-z\\s/-]", " ", s)
    s = re.sub(r"\\s+", " ", s).strip()
    return s

def prepare_missions(master: pd.DataFrame, ma: pd.DataFrame) -> pd.DataFrame:
    x = ma.copy()
    x["disasterNumber"] = pd.to_numeric(x["disasterNumber"], errors="coerce").astype("Int64")
    x = x[x["disasterNumber"].isin(master["disasterNumber"])].copy()

    x["obligationAmount_num"] = pd.to_numeric(x["obligationAmount"], errors="coerce").fillna(0.0)
    net = (
        x.groupby(["disasterNumber", "maId"], dropna=False)["obligationAmount_num"]
        .sum()
        .rename("mission_net_obligation")
        .reset_index()
    )

    x["maAmendNumber_num"] = pd.to_numeric(x["maAmendNumber"], errors="coerce").fillna(-1)
    x["actionId_num"] = pd.to_numeric(x["actionId"], errors="coerce").fillna(-1)
    x = x.sort_values(["disasterNumber", "maId", "maAmendNumber_num", "actionId_num"])
    initial = x.groupby(["disasterNumber", "maId"], dropna=False, as_index=False).head(1).copy()
    initial = initial.merge(net, on=["disasterNumber", "maId"], how="left")

    meta = master[
        ["disasterNumber", "fyDeclared", "state", "incidentType", "declarationDate"]
    ].copy()
    meta["declarationDate_dt"] = pd.to_datetime(meta["declarationDate"], errors="coerce", utc=True)
    initial = initial.merge(meta, on="disasterNumber", how="left", suffixes=("", "_master"))
    # Use master state/incident type for consistency.
    if "state_master" in initial:
        initial["state"] = initial["state_master"]
    if "incidentType_master" in initial:
        initial["incidentType"] = initial["incidentType_master"]

    pop_start = pd.to_datetime(initial["popStartDate"], errors="coerce", utc=True)
    pop_end = pd.to_datetime(initial["popEndDate"], errors="coerce", utc=True)
    received = pd.to_datetime(initial["dateReceived"], errors="coerce", utc=True)
    required = pd.to_datetime(initial["dateRequired"], errors="coerce", utc=True)

    initial["initial_pop_duration_days"] = ((pop_end - pop_start).dt.total_seconds() / 86400.0).clip(0, 3650)
    initial["initial_required_lead_days"] = ((required - received).dt.total_seconds() / 86400.0).clip(-365, 3650)
    initial["initial_days_received_from_declaration"] = (
        (received - initial["declarationDate_dt"]).dt.total_seconds() / 86400.0
    ).clip(-365, 3650)

    initial["clean_text"] = (
        initial["assistanceRequested"].fillna("").astype(str)
        + " "
        + initial["statementOfWork"].fillna("").astype(str)
    ).map(redact_text)

    initial["is_mega_ma"] = (initial["mission_net_obligation"] >= MEGA_THRESHOLD).astype(int)

    # A small interpretable mechanism flag, computed only from non-financial fields.
    ag = initial["agencyId"].fillna("").astype(str).str.upper()
    initial["is_usace_esf3_dfa"] = (
        ag.str.contains("COE|USACE", regex=True)
        & (pd.to_numeric(initial["supportFunction"], errors="coerce") == 3)
        & initial["maType"].fillna("").astype(str).str.upper().str.contains("DFA")
    ).astype(int)

    return initial.reset_index(drop=True)

def fit_mission_model(train: pd.DataFrame, use_text: bool = True):
    cat_cols = MISSION_CATS
    num_cols = MISSION_NUMS + ["is_usace_esf3_dfa"]

    cat = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
    ])
    num = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=False)),
    ])
    structured = ColumnTransformer([
        ("cat", cat, cat_cols),
        ("num", num, num_cols),
    ])

    Xs = structured.fit_transform(train)
    vectorizer = None
    if use_text:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.995,
            sublinear_tf=True,
            max_features=12000,
        )
        Xt = vectorizer.fit_transform(train["clean_text"])
        X = sparse.hstack([Xs, Xt], format="csr")
    else:
        X = sparse.csr_matrix(Xs)

    y = train["is_mega_ma"].to_numpy(int)
    model = LogisticRegression(
        max_iter=7000,
        class_weight="balanced",
        C=0.35,
        solver="liblinear",
    )
    model.fit(X, y)
    return structured, vectorizer, model

def predict_mission(model_tuple, test: pd.DataFrame) -> np.ndarray:
    structured, vectorizer, model = model_tuple
    Xs = structured.transform(test)
    if vectorizer is not None:
        Xt = vectorizer.transform(test["clean_text"])
        X = sparse.hstack([Xs, Xt], format="csr")
    else:
        X = sparse.csr_matrix(Xs)
    return model.predict_proba(X)[:, 1]

def safe_auc(y: np.ndarray, p: np.ndarray) -> Tuple[float | None, float | None]:
    if len(np.unique(y)) < 2:
        return None, None
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))

def mission_lfyo(missions: pd.DataFrame, use_text: bool) -> Tuple[pd.DataFrame, Dict]:
    out = missions[
        ["disasterNumber", "maId", "fyDeclared", "mission_net_obligation",
         "is_mega_ma", "agencyId", "supportFunction", "maType", "priority",
         "is_usace_esf3_dfa", "clean_text"]
    ].copy()
    out["mega_probability"] = np.nan

    fold_metrics = []
    years = sorted(missions["fyDeclared"].dropna().astype(int).unique())
    for year in years:
        te = missions["fyDeclared"].astype(int).to_numpy() == year
        tr = ~te
        ytr = missions.loc[tr, "is_mega_ma"].to_numpy(int)
        # A fold is only learnable if historical training contains both classes.
        if len(np.unique(ytr)) < 2:
            continue
        fitted = fit_mission_model(missions.loc[tr], use_text=use_text)
        p = predict_mission(fitted, missions.loc[te])
        out.loc[te, "mega_probability"] = p
        yte = missions.loc[te, "is_mega_ma"].to_numpy(int)
        roc, ap = safe_auc(yte, p)
        fold_metrics.append({
            "heldout_fy": int(year),
            "train_missions": int(tr.sum()),
            "test_missions": int(te.sum()),
            "train_mega": int(ytr.sum()),
            "test_mega": int(yte.sum()),
            "roc_auc": roc,
            "pr_auc": ap,
            "max_prob": float(np.max(p)) if len(p) else None,
        })

    valid = out["mega_probability"].notna()
    y = out.loc[valid, "is_mega_ma"].to_numpy(int)
    p = out.loc[valid, "mega_probability"].to_numpy(float)
    roc, ap = safe_auc(y, p)
    summary = {
        "use_text": use_text,
        "missions_total": int(len(missions)),
        "mega_missions_total": int(missions["is_mega_ma"].sum()),
        "oof_missions": int(valid.sum()),
        "oof_mega_missions": int(y.sum()),
        "oof_roc_auc": roc,
        "oof_pr_auc": ap,
        "folds": fold_metrics,
    }
    return out, summary

def aggregate_disaster_probs(mission_oof: pd.DataFrame, suffix: str) -> pd.DataFrame:
    x = mission_oof.dropna(subset=["mega_probability"]).copy()
    if x.empty:
        return pd.DataFrame(columns=["disasterNumber"])
    def top3mean(s: pd.Series) -> float:
        vals = np.sort(s.to_numpy(float))
        return float(vals[-min(3, len(vals)):].mean())
    agg = x.groupby("disasterNumber").agg(
        **{
            f"mega_maxprob_{suffix}": ("mega_probability", "max"),
            f"mega_meanprob_{suffix}": ("mega_probability", "mean"),
            f"mega_top3prob_{suffix}": ("mega_probability", top3mean),
            f"mega_prob_gt25_count_{suffix}": ("mega_probability", lambda s: int((s >= 0.25).sum())),
            f"mega_prob_gt50_count_{suffix}": ("mega_probability", lambda s: int((s >= 0.50).sum())),
            f"usace_esf3_dfa_initial_count_{suffix}": ("is_usace_esf3_dfa", "sum"),
        }
    ).reset_index()
    return agg

def lfyo_extreme_gate(df: pd.DataFrame, features: List[str]) -> Dict:
    y = (df["totalObligatedFunding"].to_numpy(float) >= MEGA_THRESHOLD).astype(int)
    pred = np.zeros(len(df), dtype=int)
    prob = np.zeros(len(df), dtype=float)
    for year in sorted(df["fyDeclared"].astype(int).unique()):
        te = df["fyDeclared"].astype(int).to_numpy() == year
        tr = ~te
        Xtr = normalize_model_frame(df.loc[tr, features])
        Xte = normalize_model_frame(df.loc[te, features])
        model = LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5)
        pipe = prep_pipeline(Xtr, model)
        pipe.fit(Xtr, y[tr])
        pred[te] = pipe.predict(Xte)
        prob[te] = pipe.predict_proba(Xte)[:, 1]
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "extreme_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "predictions": pred.tolist(),
        "probabilities": prob.tolist(),
    }

def main():
    print("Reading master...")
    master = normalize_master(pd.read_excel(MASTER_PATH))
    assert len(master) == 971

    print("Loading MissionAssignments...")
    ma = fetch_all_mission_assignments()
    missions = prepare_missions(master, ma)
    print("Initial unique missions:", len(missions))
    print("Mega missions >= $500M:", int(missions["is_mega_ma"].sum()))

    mega_cases = missions[missions["is_mega_ma"] == 1][
        ["disasterNumber", "maId", "fyDeclared", "state", "incidentType",
         "agencyId", "supportFunction", "maType", "priority",
         "mission_net_obligation", "is_usace_esf3_dfa", "clean_text"]
    ].sort_values("mission_net_obligation", ascending=False)
    mega_cases.to_csv(OUT / "mega_missions.csv", index=False)

    print("Running mission LFYO without text...")
    struct_oof, struct_summary = mission_lfyo(missions, use_text=False)
    print("Running mission LFYO with redacted text...")
    text_oof, text_summary = mission_lfyo(missions, use_text=True)

    struct_oof.to_csv(OUT / "mission_oof_structured.csv", index=False)
    text_oof.to_csv(OUT / "mission_oof_text.csv", index=False)

    struct_agg = aggregate_disaster_probs(struct_oof, "struct")
    text_agg = aggregate_disaster_probs(text_oof, "text")

    print("Building disaster semantic rollup...")
    sem, _ = build_semantic_rollup(master, ma)

    print("Building corrected external severity layer...")
    ext, match_audit = build_external(master)
    match_audit.to_csv(OUT / "external_match_audit.csv", index=False)

    enriched = (
        master.merge(sem, on="disasterNumber", how="left")
        .merge(ext, on="disasterNumber", how="left")
        .merge(struct_agg, on="disasterNumber", how="left")
        .merge(text_agg, on="disasterNumber", how="left")
    )

    mega_cols = [c for c in enriched.columns if c.startswith("mega_") or c.startswith("usace_esf3_dfa_initial_count_")]
    enriched[mega_cols] = enriched[mega_cols].fillna(0.0)

    high = enriched[
        (enriched["incidentType"] != "Biological")
        & (enriched["totalObligatedFunding"] >= 50_000_000)
    ].copy().reset_index(drop=True)
    assert len(high) == 23

    current = [c for c in CURRENT_19 if c in high.columns]
    sem_cols = [
        c for c in high.columns
        if (c.startswith("sem_") or c.startswith("ma_"))
        and high[c].notna().sum() >= 2 and high[c].nunique(dropna=True) > 1
    ]
    ext_cols = [
        c for c in high.columns
        if (c.startswith("nhc_") or c.startswith("noaa_") or c.startswith("calfire_"))
        and high[c].notna().sum() >= 2 and high[c].nunique(dropna=True) > 1
    ]

    sets = {
        "A_current19": current,
        "A_plus_mega_struct": current + [c for c in mega_cols if c.endswith("_struct")],
        "A_plus_mega_text": current + [c for c in mega_cols if c.endswith("_text")],
        "B_semantics": current + sem_cols,
        "B_semantics_plus_mega_text": current + sem_cols + [c for c in mega_cols if c.endswith("_text")],
        "D_semantics_external": current + sem_cols + ext_cols,
        "D_semantics_external_plus_mega_struct": current + sem_cols + ext_cols + [c for c in mega_cols if c.endswith("_struct")],
        "D_semantics_external_plus_mega_text": current + sem_cols + ext_cols + [c for c in mega_cols if c.endswith("_text")],
    }

    results = {}
    pred_table = high[
        ["disasterNumber", "state", "incidentType", "fyDeclared", "totalObligatedFunding"]
    ].copy()
    pred_table["actual_extreme"] = (pred_table["totalObligatedFunding"] >= MEGA_THRESHOLD).astype(int)

    for name, feats in sets.items():
        print(f"Extreme gate {name}: {len(feats)} features")
        r = lfyo_extreme_gate(high, feats)
        results[name] = {"feature_count": len(feats), **r}
        pred_table[f"{name}_pred"] = r["predictions"]
        pred_table[f"{name}_prob"] = r["probabilities"]

    pred_table.to_csv(OUT / "high23_mega_gate_predictions.csv", index=False)

    # Attach interpretable mega probabilities for the key 23.
    key_cols = ["disasterNumber"] + mega_cols
    high[["disasterNumber", "state", "incidentType", "fyDeclared", "totalObligatedFunding"] + mega_cols].to_csv(
        OUT / "high23_mega_features.csv", index=False
    )

    summary = {
        "mission_definition": "initial non-financial MA record -> eventual net mission obligation >= $500M",
        "mega_threshold": MEGA_THRESHOLD,
        "mission_counts": {
            "unique_initial_missions": int(len(missions)),
            "mega_missions": int(missions["is_mega_ma"].sum()),
            "mega_disasters": int(missions.loc[missions["is_mega_ma"] == 1, "disasterNumber"].nunique()),
            "mega_usace_esf3_dfa_share": float(
                missions.loc[missions["is_mega_ma"] == 1, "is_usace_esf3_dfa"].mean()
            ) if missions["is_mega_ma"].sum() else None,
        },
        "mission_lfyo_structured": struct_summary,
        "mission_lfyo_redacted_text": text_summary,
        "disaster_extreme_gate": results,
        "leakage_controls": [
            "Financial fields are used only to create the mission-level training label.",
            "Features are taken from the initial MA action/amendment.",
            "All digits and currency terms are removed from free text.",
            "Held-out fiscal-year Mission Assignments never enter mission-model training.",
            "Disaster-level gates receive only out-of-year mega-MA probabilities.",
        ],
        "timing_note": (
            "Initial Mission Assignment features are later than declaration for many events; "
            "this experiment is an early-mission-stage mechanism test, not declaration-time."
        ),
    }
    (OUT / "mega_gate_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    md = [
        "# Mega Mission Assignment gate",
        "",
        f"- Unique initial MAs: **{len(missions):,}**",
        f"- Mega MAs >= $500M: **{int(missions['is_mega_ma'].sum())}**",
        f"- Mega disasters: **{int(missions.loc[missions['is_mega_ma']==1, 'disasterNumber'].nunique())}**",
        f"- Mega MAs matching USACE×ESF3×DFA: **{summary['mission_counts']['mega_usace_esf3_dfa_share']:.1%}**",
        "",
        "## Mission-level strict LFYO",
        "",
        f"- Structured ROC-AUC: **{struct_summary['oof_roc_auc']}**",
        f"- Structured PR-AUC: **{struct_summary['oof_pr_auc']}**",
        f"- + redacted text ROC-AUC: **{text_summary['oof_roc_auc']}**",
        f"- + redacted text PR-AUC: **{text_summary['oof_pr_auc']}**",
        "",
        "## Disaster-level $500M+ strict LFYO",
        "",
        "| Feature set | Balanced accuracy | Extreme recall | Confusion matrix [TN,FP;FN,TP] |",
        "|---|---:|---:|---|",
    ]
    for name, r in results.items():
        md.append(
            f"| {name} | {r['balanced_accuracy']:.3f} | {r['extreme_recall']:.1%} | {r['confusion_matrix']} |"
        )
    md += [
        "",
        "## Timing",
        "",
        summary["timing_note"],
    ]
    (OUT / "mega_gate_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

if __name__ == "__main__":
    main()
