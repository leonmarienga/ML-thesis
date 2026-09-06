#!/usr/bin/env python3
"""
Fully nested LFYO high-confidence mega-Mission override.

Outer FY is untouched. Inside each outer fold:
1. Build inner-LFYO predictions for the D = current19+semantics+external gate.
2. Build inner-LFYO mission mega probabilities using ONLY non-outer/non-inner FY missions.
3. Aggregate mission max probability to disaster.
4. Choose a mission override threshold on outer-training disasters only, maximizing
   balanced accuracy of OR(baseline_gate, mega_prob >= threshold), tie -> higher threshold.
5. Fit baseline and mission models on all outer-training years and evaluate outer FY.

No threshold is selected using outer-year outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score

from mission_semantic_audit import (
    CURRENT_19,
    build_semantic_rollup,
    fetch_all_mission_assignments,
    normalize_master,
    normalize_model_frame,
    prep_pipeline,
)
from external_severity_ablation import build_external
from mega_mission_gate import (
    MEGA_THRESHOLD,
    fit_mission_model,
    predict_mission,
    prepare_missions,
)

ROOT = Path(__file__).resolve().parents[1]
MASTER_PATH = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "nested_mega_override"
OUT.mkdir(parents=True, exist_ok=True)


def build_high(master: pd.DataFrame, ma: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    sem, _ = build_semantic_rollup(master, ma)
    ext, audit = build_external(master)
    audit.to_csv(OUT / "external_match_audit.csv", index=False)
    e = master.merge(sem, on="disasterNumber", how="left").merge(ext, on="disasterNumber", how="left")

    current = [c for c in CURRENT_19 if c in e.columns]
    sem_cols = [
        c for c in e.columns
        if (c.startswith("sem_") or c.startswith("ma_"))
        and e[c].notna().sum() >= 2 and e[c].nunique(dropna=True) > 1
    ]
    ext_cols = [
        c for c in e.columns
        if (c.startswith("nhc_") or c.startswith("noaa_") or c.startswith("calfire_"))
        and e[c].notna().sum() >= 2 and e[c].nunique(dropna=True) > 1
    ]
    high = e[
        (e["incidentType"] != "Biological")
        & (e["totalObligatedFunding"] >= 50_000_000)
    ].copy().reset_index(drop=True)
    return high, current + sem_cols + ext_cols


def fit_gate(train: pd.DataFrame, features: List[str]):
    y = (train["totalObligatedFunding"].to_numpy(float) >= MEGA_THRESHOLD).astype(int)
    X = normalize_model_frame(train[features])
    model = LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5)
    pipe = prep_pipeline(X, model)
    pipe.fit(X, y)
    return pipe


def pred_gate(pipe, test: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X = normalize_model_frame(test[features])
    p = pipe.predict_proba(X)[:,1]
    return (p >= 0.5).astype(int), p


def aggregate_max_prob(missions: pd.DataFrame, probs: np.ndarray) -> Dict[int, float]:
    z = missions[["disasterNumber"]].copy()
    z["p"] = probs
    return z.groupby("disasterNumber")["p"].max().to_dict()


def choose_threshold(y: np.ndarray, baseline_pred: np.ndarray, mega_p: np.ndarray) -> Tuple[float, float]:
    # Candidates are observed training probabilities plus sentinel 1.01.
    vals = np.unique(np.r_[mega_p[np.isfinite(mega_p)], 0.0, 1.01])
    best_ba, best_th = -1.0, 1.01
    for th in vals:
        pred = ((baseline_pred == 1) | (mega_p >= th)).astype(int)
        ba = balanced_accuracy_score(y, pred)
        if ba > best_ba + 1e-12 or (abs(ba-best_ba) <= 1e-12 and th > best_th):
            best_ba, best_th = float(ba), float(th)
    return best_th, best_ba


def main():
    master = normalize_master(pd.read_excel(MASTER_PATH))
    ma = fetch_all_mission_assignments()
    missions = prepare_missions(master, ma)
    high, d_features = build_high(master, ma)

    years = sorted(high["fyDeclared"].astype(int).unique())
    y_all = (high["totalObligatedFunding"].to_numpy(float) >= MEGA_THRESHOLD).astype(int)

    final_base = np.zeros(len(high), dtype=int)
    final_over = np.zeros(len(high), dtype=int)
    final_base_p = np.zeros(len(high), dtype=float)
    final_mega_p = np.zeros(len(high), dtype=float)
    thresholds = []

    for outer in years:
        print(f"OUTER FY {outer}")
        outer_te = high["fyDeclared"].astype(int).to_numpy() == outer
        outer_tr = ~outer_te

        train_high = high.loc[outer_tr].copy()
        test_high = high.loc[outer_te].copy()
        train_indices = np.where(outer_tr)[0]
        test_indices = np.where(outer_te)[0]

        # Inner OOF predictions on outer-training high disasters.
        inner_base = np.zeros(len(train_high), dtype=int)
        inner_mega = np.full(len(train_high), np.nan, dtype=float)
        tr_years = sorted(train_high["fyDeclared"].astype(int).unique())

        for inner in tr_years:
            inner_te_mask = train_high["fyDeclared"].astype(int).to_numpy() == inner
            inner_tr_mask = ~inner_te_mask

            # Baseline inner gate.
            if len(np.unique((train_high.loc[inner_tr_mask, "totalObligatedFunding"].to_numpy(float) >= MEGA_THRESHOLD).astype(int))) < 2:
                continue
            g = fit_gate(train_high.loc[inner_tr_mask], d_features)
            bp, _ = pred_gate(g, train_high.loc[inner_te_mask], d_features)
            inner_base[inner_te_mask] = bp

            # Mission inner model excludes BOTH outer and inner FY.
            mtr = missions[
                (missions["fyDeclared"].astype(int) != outer)
                & (missions["fyDeclared"].astype(int) != inner)
            ]
            mte = missions[
                (missions["fyDeclared"].astype(int) == inner)
            ]
            if mte.empty or mtr["is_mega_ma"].nunique() < 2:
                continue
            mm = fit_mission_model(mtr, use_text=False)
            mp = predict_mission(mm, mte)
            amap = aggregate_max_prob(mte, mp)
            rows = np.where(inner_te_mask)[0]
            for r in rows:
                dn = int(train_high.iloc[r]["disasterNumber"])
                inner_mega[r] = float(amap.get(dn, 0.0))

        # Any training disaster with no mission prediction gets zero.
        inner_mega = np.nan_to_num(inner_mega, nan=0.0)
        y_train = (train_high["totalObligatedFunding"].to_numpy(float) >= MEGA_THRESHOLD).astype(int)
        th, inner_ba = choose_threshold(y_train, inner_base, inner_mega)

        # Outer baseline gate fit only non-outer FY.
        g_outer = fit_gate(train_high, d_features)
        bp_outer, bprob_outer = pred_gate(g_outer, test_high, d_features)

        # Outer mission model fit only non-outer FY.
        mtr = missions[missions["fyDeclared"].astype(int) != outer]
        mte = missions[missions["fyDeclared"].astype(int) == outer]
        if mtr["is_mega_ma"].nunique() >= 2 and not mte.empty:
            mm_outer = fit_mission_model(mtr, use_text=False)
            mp_outer = predict_mission(mm_outer, mte)
            amap = aggregate_max_prob(mte, mp_outer)
        else:
            amap = {}

        mega_outer = np.array([
            float(amap.get(int(dn), 0.0))
            for dn in test_high["disasterNumber"]
        ])
        over_outer = ((bp_outer == 1) | (mega_outer >= th)).astype(int)

        final_base[test_indices] = bp_outer
        final_over[test_indices] = over_outer
        final_base_p[test_indices] = bprob_outer
        final_mega_p[test_indices] = mega_outer

        thresholds.append({
            "outer_fy": int(outer),
            "threshold": float(th),
            "inner_balanced_accuracy": float(inner_ba),
            "outer_cases": int(len(test_high)),
            "outer_extremes": int((test_high["totalObligatedFunding"] >= MEGA_THRESHOLD).sum()),
        })

    base_ba = float(balanced_accuracy_score(y_all, final_base))
    over_ba = float(balanced_accuracy_score(y_all, final_over))
    base_rec = float(recall_score(y_all, final_base, pos_label=1, zero_division=0))
    over_rec = float(recall_score(y_all, final_over, pos_label=1, zero_division=0))
    base_cm = confusion_matrix(y_all, final_base, labels=[0,1]).tolist()
    over_cm = confusion_matrix(y_all, final_over, labels=[0,1]).tolist()

    pred = high[[
        "disasterNumber","state","incidentType","fyDeclared","totalObligatedFunding"
    ]].copy()
    pred["actual_extreme"] = y_all
    pred["baseline_pred"] = final_base
    pred["baseline_prob"] = final_base_p
    pred["mega_maxprob_nested"] = final_mega_p
    pred["override_pred"] = final_over
    threshold_map = {x["outer_fy"]:x["threshold"] for x in thresholds}
    pred["fold_threshold"] = pred["fyDeclared"].astype(int).map(threshold_map)
    pred.to_csv(OUT / "nested_override_predictions.csv", index=False)
    pd.DataFrame(thresholds).to_csv(OUT / "nested_thresholds.csv", index=False)

    summary = {
        "baseline": {
            "balanced_accuracy": base_ba,
            "extreme_recall": base_rec,
            "confusion_matrix": base_cm,
        },
        "nested_mega_override": {
            "balanced_accuracy": over_ba,
            "extreme_recall": over_rec,
            "confusion_matrix": over_cm,
        },
        "thresholds": thresholds,
        "protocol": (
            "Outer LFYO. Override threshold selected from inner LFYO predictions on "
            "outer-training years only; structured initial-MA mega model excludes "
            "outer and inner fiscal years during threshold calibration."
        ),
    }
    (OUT / "nested_override_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Fully nested mega-Mission override",
        "",
        f"- Baseline D balanced accuracy: **{base_ba:.3f}**",
        f"- Baseline D extreme recall: **{base_rec:.1%}**",
        f"- Baseline confusion: **{base_cm}**",
        "",
        f"- Nested override balanced accuracy: **{over_ba:.3f}**",
        f"- Nested override extreme recall: **{over_rec:.1%}**",
        f"- Nested override confusion: **{over_cm}**",
        "",
        "## Thresholds by untouched outer fiscal year",
        "",
    ]
    for x in thresholds:
        md.append(
            f"- FY{x['outer_fy']}: threshold {x['threshold']:.4f}; "
            f"inner BA {x['inner_balanced_accuracy']:.3f}; "
            f"outer n={x['outer_cases']}, extremes={x['outer_extremes']}"
        )
    (OUT / "nested_override_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

if __name__ == "__main__":
    main()
