#!/usr/bin/env python3
"""
Final-development non-Biological hazard hierarchy audit.

Architecture
------------
1. Extreme candidate gate:
   current19 + mission semantics + corrected external severity.
2. Fire verifier:
   if Fire is called extreme, require ESF-4 share consistent with other-year
   extreme fires.
3. Mechanism prerequisite:
   if inner temporal folds show that all known true extreme candidates contain
   >=1 initial USACE + ESF-3 + DFA mission, reject an outer extreme candidate
   with zero such missions.
4. Lower router:
   - Hurricane: hurricane-only current19 + semantics logistic router.
   - Other hazards: general current19 + semantics logistic router.

Everything is evaluated by outer leave-fiscal-year-out.
The mechanism prerequisite is activated using inner out-of-year candidate
predictions from the outer-training data only.

NOTE: This is still a development-set nested temporal audit, not a fresh
external holdout. Mission aggregates are retrospective unless t0 is defined.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

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
MASTER = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "nonbio_hazard_hierarchy"
OUT.mkdir(parents=True, exist_ok=True)

EXTREME = 500_000_000.0

def band(v: float) -> str:
    if v < 200_000_000:
        return "50-200M"
    if v < 500_000_000:
        return "200-500M"
    return "500M+"

def fit_binary(train: pd.DataFrame, features: List[str], y: pd.Series):
    X = normalize_model_frame(train[features])
    model = LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5)
    pipe = prep_pipeline(X, model)
    pipe.fit(X, y)
    return pipe

def initial_mechanism_counts(master: pd.DataFrame, ma: pd.DataFrame) -> pd.DataFrame:
    x = ma.copy()
    x["disasterNumber"] = pd.to_numeric(x["disasterNumber"], errors="coerce").astype("Int64")
    ids = set(master["disasterNumber"].dropna().astype(int))
    x = x[x["disasterNumber"].isin(ids)].copy()
    x["amend"] = pd.to_numeric(x["maAmendNumber"], errors="coerce").fillna(-1)
    x["action"] = pd.to_numeric(x["actionId"], errors="coerce").fillna(-1)
    initial = (
        x.sort_values(["disasterNumber", "maId", "amend", "action"])
         .groupby(["disasterNumber", "maId"], dropna=False)
         .head(1)
         .copy()
    )
    agency = initial["agencyId"].fillna("").astype(str).str.upper()
    is_coe = agency.str.contains(r"(?:^|[^A-Z])(?:COE|USACE)(?:[^A-Z]|$)", regex=True)
    is_esf3 = pd.to_numeric(initial["supportFunction"], errors="coerce").eq(3)
    is_dfa = initial["maType"].fillna("").astype(str).str.upper().str.contains("DFA")
    initial["mechanism"] = (is_coe & is_esf3 & is_dfa).astype(int)
    return (
        initial.groupby("disasterNumber")["mechanism"]
        .sum()
        .rename("initial_usace_esf3_dfa_count")
        .reset_index()
    )

def fire_threshold(train: pd.DataFrame) -> float:
    f = train[train["incidentType"] == "Fire"]
    neg = f.loc[f["totalObligatedFunding"] < EXTREME, "sem_esf_4_share"].dropna()
    pos = f.loc[f["totalObligatedFunding"] >= EXTREME, "sem_esf_4_share"].dropna()
    if len(neg) and len(pos):
        max_neg, min_pos = float(neg.max()), float(pos.min())
        return (max_neg + min_pos) / 2 if max_neg < min_pos else min_pos
    return 0.0

def extreme_candidates(train: pd.DataFrame, test: pd.DataFrame, gate_features: List[str]):
    y = (train["totalObligatedFunding"] >= EXTREME).astype(int)
    gate = fit_binary(train, gate_features, y)
    pred = gate.predict(normalize_model_frame(test[gate_features]))

    fth = fire_threshold(train)
    out = []
    for (_, r), p in zip(test.iterrows(), pred):
        extreme = int(p)
        if extreme == 1 and r["incidentType"] == "Fire":
            esf4 = float(r["sem_esf_4_share"]) if pd.notna(r["sem_esf_4_share"]) else 0.0
            if esf4 < fth:
                extreme = 0
        out.append(extreme)
    return np.asarray(out, dtype=int)

def mechanism_active_from_inner(
    outer_train: pd.DataFrame,
    gate_features: List[str],
) -> tuple[bool, list[dict]]:
    rows = []
    for inner_fy in sorted(outer_train["fyDeclared"].astype(int).unique()):
        inner_train = outer_train[outer_train["fyDeclared"].astype(int) != inner_fy].copy()
        inner_test = outer_train[outer_train["fyDeclared"].astype(int) == inner_fy].copy()

        y = (inner_train["totalObligatedFunding"] >= EXTREME).astype(int)
        if y.nunique() < 2:
            continue

        candidate = extreme_candidates(inner_train, inner_test, gate_features)
        for (_, r), c in zip(inner_test.iterrows(), candidate):
            if int(c) == 1:
                rows.append({
                    "disasterNumber": int(r["disasterNumber"]),
                    "true_extreme": int(r["totalObligatedFunding"] >= EXTREME),
                    "mechanism_count": int(r["initial_usace_esf3_dfa_count"]),
                    "inner_fy": int(inner_fy),
                })

    candidate_df = pd.DataFrame(rows)
    if candidate_df.empty:
        return False, rows

    true_candidates = candidate_df[candidate_df["true_extreme"] == 1]
    # The prerequisite is enabled only when the outer-training inner folds
    # contain at least one correctly proposed true extreme, and every such
    # candidate exhibits the mechanism.
    active = (
        len(true_candidates) >= 1
        and (true_candidates["mechanism_count"] >= 1).all()
    )
    return bool(active), rows

def fit_lower_general(train: pd.DataFrame, features: List[str]):
    t = train[train["totalObligatedFunding"] < EXTREME].copy()
    y = (t["totalObligatedFunding"] >= 200_000_000).astype(int)
    return fit_binary(t, features, y)

def fit_lower_hurricane(train: pd.DataFrame, features: List[str]):
    t = train[
        (train["incidentType"] == "Hurricane")
        & (train["totalObligatedFunding"] < EXTREME)
    ].copy()
    y = (t["totalObligatedFunding"] >= 200_000_000).astype(int)
    if t.empty or y.nunique() < 2:
        return None
    return fit_binary(t, features, y)

def main():
    master = normalize_master(pd.read_excel(MASTER))
    ma = fetch_all_mission_assignments()
    sem, _ = build_semantic_rollup(master, ma)
    ext, match_audit = build_external(master)
    mech = initial_mechanism_counts(master, ma)

    match_audit.to_csv(OUT / "external_match_audit.csv", index=False)

    df = (
        master.merge(sem, on="disasterNumber", how="left")
        .merge(ext, on="disasterNumber", how="left")
        .merge(mech, on="disasterNumber", how="left")
    )
    df["initial_usace_esf3_dfa_count"] = (
        df["initial_usace_esf3_dfa_count"].fillna(0).astype(int)
    )

    high = df[
        (df["incidentType"] != "Biological")
        & (df["totalObligatedFunding"] >= 50_000_000)
    ].copy().reset_index(drop=True)
    high["actual_band"] = high["totalObligatedFunding"].map(band)
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

    gate_features = current + sem_cols + ext_cols
    lower_features = current + sem_cols

    rows = []
    folds = []

    for outer_fy in sorted(high["fyDeclared"].astype(int).unique()):
        train = high[high["fyDeclared"].astype(int) != outer_fy].copy()
        test = high[high["fyDeclared"].astype(int) == outer_fy].copy()

        active, inner_candidates = mechanism_active_from_inner(train, gate_features)

        candidate = extreme_candidates(train, test, gate_features)
        general_lower = fit_lower_general(train, lower_features)
        general_pred = general_lower.predict(
            normalize_model_frame(test[lower_features])
        )
        hurricane_lower = fit_lower_hurricane(train, lower_features)

        folds.append({
            "outer_fy": int(outer_fy),
            "mechanism_prerequisite_active": bool(active),
            "inner_candidate_count": len(inner_candidates),
            "inner_true_candidate_count": sum(
                int(x["true_extreme"]) for x in inner_candidates
            ),
        })

        for (_, r), extreme0, lower0 in zip(
            test.iterrows(), candidate, general_pred
        ):
            extreme = int(extreme0)

            # Mechanism prerequisite can only reject an extreme call.
            if (
                extreme == 1
                and active
                and int(r["initial_usace_esf3_dfa_count"]) < 1
            ):
                extreme = 0

            if extreme == 1:
                final = "500M+"
            else:
                if r["incidentType"] == "Hurricane" and hurricane_lower is not None:
                    hp = int(
                        hurricane_lower.predict(
                            normalize_model_frame(
                                test.loc[[r.name], lower_features]
                            )
                        )[0]
                    )
                    final = "200-500M" if hp else "50-200M"
                else:
                    final = "200-500M" if int(lower0) else "50-200M"

            rows.append({
                "disasterNumber": int(r["disasterNumber"]),
                "state": r["state"],
                "incidentType": r["incidentType"],
                "fyDeclared": int(r["fyDeclared"]),
                "totalObligatedFunding": float(r["totalObligatedFunding"]),
                "actual_band": r["actual_band"],
                "initial_extreme_candidate": int(extreme0),
                "mechanism_count": int(r["initial_usace_esf3_dfa_count"]),
                "mechanism_active": bool(active),
                "final_pred": final,
            })

    pred = pd.DataFrame(rows)
    pred.to_csv(OUT / "predictions.csv", index=False)
    pd.DataFrame(folds).to_csv(OUT / "folds.csv", index=False)

    per_band = {}
    for b in ["50-200M", "200-500M", "500M+"]:
        m = pred["actual_band"] == b
        correct = int((pred.loc[m, "final_pred"] == b).sum())
        total = int(m.sum())
        per_band[b] = {
            "correct": correct,
            "total": total,
            "recall": correct / total if total else None,
        }

    correct = int((pred["actual_band"] == pred["final_pred"]).sum())
    errors = pred[pred["actual_band"] != pred["final_pred"]][
        ["disasterNumber", "state", "incidentType", "actual_band", "final_pred"]
    ].to_dict(orient="records")

    summary = {
        "overall_correct": correct,
        "overall_total": int(len(pred)),
        "overall_accuracy": correct / len(pred),
        "per_band": per_band,
        "errors": errors,
        "protocol": (
            "Outer leave-fiscal-year-out. Fire verification learned from outer-training "
            "fire cases. Mechanism prerequisite activated only from inner out-of-year "
            "true extreme candidates. Hurricane lower router trained only on outer-training "
            "high-value hurricanes below $500M."
        ),
        "cautions": [
            "This is a nested temporal development audit, not a fresh untouched external test.",
            "Mission-based features remain retrospective unless a consistent t0 is defined.",
            "The mechanism hypothesis was developed from exploratory analysis of this dataset.",
        ],
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    md = [
        "# Non-Biological hazard hierarchy",
        "",
        f"- Overall: **{correct}/{len(pred)} = {correct/len(pred):.1%}**",
        f"- $50M-$200M: **{per_band['50-200M']['correct']}/{per_band['50-200M']['total']} = {per_band['50-200M']['recall']:.1%}**",
        f"- $200M-$500M: **{per_band['200-500M']['correct']}/{per_band['200-500M']['total']} = {per_band['200-500M']['recall']:.1%}**",
        f"- $500M+: **{per_band['500M+']['correct']}/{per_band['500M+']['total']} = {per_band['500M+']['recall']:.1%}**",
        "",
        "## Remaining errors",
    ]
    for e in errors:
        md.append(
            f"- FEMA {e['disasterNumber']} {e['state']} {e['incidentType']}: "
            f"{e['actual_band']} -> {e['final_pred']}"
        )
    md += [
        "",
        "## Architecture",
        "",
        "Extreme gate -> Fire ESF-4 verifier -> USACE/ESF-3/DFA mechanism prerequisite -> hazard-specific lower router.",
        "",
        "This result is developmental and nested-temporal, not an untouched external holdout.",
    ]
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

if __name__ == "__main__":
    main()
