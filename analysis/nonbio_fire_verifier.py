#!/usr/bin/env python3
"""
Non-Biological fire verifier audit.

Goal
----
Improve the lower $50M-$200M band without sacrificing the current 80% recall
for $200M-$500M and $500M+.

Architecture
------------
1. Baseline extreme gate: current 19 + mission semantics + corrected external severity.
2. Fire-only verifier:
   - invoked ONLY if baseline gate predicts $500M+ AND incidentType == Fire
   - learns an ESF-4 mission-share threshold from OTHER fiscal years' high-value fires
   - rejects candidate extreme fires below that threshold back to the lower router
3. Lower router: current 19 + mission semantics, trained only on < $500M cases.

The verifier threshold is learned without using the held-out fiscal year.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mission_semantic_audit import (
    CURRENT_19,
    build_semantic_rollup,
    fetch_all_mission_assignments,
    normalize_master,
)
from external_severity_ablation import build_external

ROOT = Path(__file__).resolve().parents[1]
MASTER_PATH = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "nonbio_fire_verifier"
OUT.mkdir(parents=True, exist_ok=True)

def make_pipe(train: pd.DataFrame, features: List[str], y: pd.Series):
    X = train[features]
    cats = [c for c in features if not pd.api.types.is_numeric_dtype(X[c])]
    nums = [c for c in features if c not in cats]
    pre = ColumnTransformer([
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore")),
        ]), cats),
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler(with_mean=False)),
        ]), nums),
    ])
    pipe = Pipeline([
        ("pre", pre),
        ("m", LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5)),
    ])
    pipe.fit(X, y)
    return pipe

def band(v: float) -> str:
    if v < 200_000_000:
        return "50-200M"
    if v < 500_000_000:
        return "200-500M"
    return "500M+"

def main():
    master = normalize_master(pd.read_excel(MASTER_PATH))
    ma = fetch_all_mission_assignments()
    sem, _ = build_semantic_rollup(master, ma)
    ext, audit = build_external(master)
    audit.to_csv(OUT / "external_match_audit.csv", index=False)

    df = (
        master.merge(sem, on="disasterNumber", how="left")
        .merge(ext, on="disasterNumber", how="left")
    )
    high = df[
        (df["incidentType"] != "Biological")
        & (df["totalObligatedFunding"] >= 50_000_000)
    ].copy().reset_index(drop=True)
    high["funding_band"] = high["totalObligatedFunding"].map(band)
    assert len(high) == 23

    current = [c for c in CURRENT_19 if c in high.columns]
    sem_cols = [
        c for c in high.columns
        if (c.startswith("sem_") or c.startswith("ma_"))
        and high[c].notna().sum() >= 2
        and high[c].nunique(dropna=True) > 1
    ]
    ext_cols = [
        c for c in high.columns
        if (c.startswith("nhc_") or c.startswith("noaa_") or c.startswith("calfire_"))
        and high[c].notna().sum() >= 2
        and high[c].nunique(dropna=True) > 1
    ]

    d_features = current + sem_cols + ext_cols
    lower_features = current + sem_cols

    rows = []
    thresholds = []

    for outer in sorted(high["fyDeclared"].astype(int).unique()):
        train = high[high["fyDeclared"].astype(int) != outer].copy()
        test = high[high["fyDeclared"].astype(int) == outer].copy()

        # Extreme gate.
        y_gate = (train["totalObligatedFunding"] >= 500_000_000).astype(int)
        gate = make_pipe(train, d_features, y_gate)
        gate_pred = gate.predict(test[d_features])

        # Lower router.
        lower_train = train[train["totalObligatedFunding"] < 500_000_000].copy()
        y_lower = (lower_train["totalObligatedFunding"] >= 200_000_000).astype(int)
        lower = make_pipe(lower_train, lower_features, y_lower)
        lower_pred = lower.predict(test[lower_features])

        # Fire verifier threshold learned from OTHER-year fires only.
        fire_train = train[train["incidentType"] == "Fire"].copy()
        neg = fire_train.loc[
            fire_train["totalObligatedFunding"] < 500_000_000, "sem_esf_4_share"
        ].dropna()
        pos = fire_train.loc[
            fire_train["totalObligatedFunding"] >= 500_000_000, "sem_esf_4_share"
        ].dropna()

        if len(neg) and len(pos):
            max_neg = float(neg.max())
            min_pos = float(pos.min())
            if max_neg < min_pos:
                threshold = (max_neg + min_pos) / 2.0
            else:
                threshold = min_pos
        else:
            threshold = 0.0

        thresholds.append({
            "outer_fy": int(outer),
            "threshold": float(threshold),
            "fire_train_n": int(len(fire_train)),
            "fire_train_extreme_n": int((fire_train["totalObligatedFunding"] >= 500_000_000).sum()),
        })

        for (_, r), gp, lp in zip(test.iterrows(), gate_pred, lower_pred):
            accept_extreme = int(gp)
            if int(gp) == 1 and r["incidentType"] == "Fire":
                esf4_share = float(r["sem_esf_4_share"]) if pd.notna(r["sem_esf_4_share"]) else 0.0
                accept_extreme = int(esf4_share >= threshold)

            if accept_extreme:
                pred = "500M+"
            else:
                pred = "200-500M" if int(lp) == 1 else "50-200M"

            rows.append({
                "disasterNumber": int(r["disasterNumber"]),
                "state": r["state"],
                "incidentType": r["incidentType"],
                "fyDeclared": int(r["fyDeclared"]),
                "totalObligatedFunding": float(r["totalObligatedFunding"]),
                "actual_band": r["funding_band"],
                "gate_pred": int(gp),
                "fire_verifier_threshold": float(threshold),
                "sem_esf_4_share": float(r["sem_esf_4_share"]) if pd.notna(r["sem_esf_4_share"]) else 0.0,
                "final_pred": pred,
            })

    pred = pd.DataFrame(rows)
    pred.to_csv(OUT / "fire_verifier_predictions.csv", index=False)
    pd.DataFrame(thresholds).to_csv(OUT / "fire_verifier_thresholds.csv", index=False)

    per_band = {}
    for b in ["50-200M", "200-500M", "500M+"]:
        m = pred["actual_band"] == b
        correct = int((pred.loc[m, "actual_band"] == pred.loc[m, "final_pred"]).sum())
        total = int(m.sum())
        per_band[b] = {
            "correct": correct,
            "total": total,
            "recall": correct / total if total else None,
        }

    summary = {
        "n": int(len(pred)),
        "overall_correct": int((pred["actual_band"] == pred["final_pred"]).sum()),
        "overall_accuracy": float((pred["actual_band"] == pred["final_pred"]).mean()),
        "per_band": per_band,
        "errors": pred.loc[
            pred["actual_band"] != pred["final_pred"],
            ["disasterNumber", "state", "incidentType", "actual_band", "final_pred"]
        ].to_dict(orient="records"),
        "protocol": (
            "Strict outer leave-fiscal-year-out. Fire verifier threshold learned only "
            "from high-value Fire cases in non-outer fiscal years."
        ),
    }
    (OUT / "fire_verifier_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Non-Biological fire verifier",
        "",
        f"- Overall: **{summary['overall_correct']}/{summary['n']} = {summary['overall_accuracy']:.1%}**",
        f"- $50M-$200M: **{per_band['50-200M']['correct']}/{per_band['50-200M']['total']} = {per_band['50-200M']['recall']:.1%}**",
        f"- $200M-$500M: **{per_band['200-500M']['correct']}/{per_band['200-500M']['total']} = {per_band['200-500M']['recall']:.1%}**",
        f"- $500M+: **{per_band['500M+']['correct']}/{per_band['500M+']['total']} = {per_band['500M+']['recall']:.1%}**",
        "",
        "## Remaining errors",
    ]
    for e in summary["errors"]:
        md.append(
            f"- FEMA {e['disasterNumber']} {e['state']} {e['incidentType']}: "
            f"{e['actual_band']} -> {e['final_pred']}"
        )
    (OUT / "fire_verifier_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

if __name__ == "__main__":
    main()
