#!/usr/bin/env python3
"""
Strict LFYO one-way hurricane mega-Mission verifier layered on the fire verifier.

The hurricane verifier may REJECT a baseline $500M+ call but may never create
a new extreme call. Its threshold is calibrated only on inner out-of-year
candidate hurricane predictions and is constrained to retain all inner true
extremes when possible.

This targets false extreme hurricane alarms such as Helene GA without risking
the current extreme recall by aggressive overrides.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

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
from mega_mission_gate import (
    fit_mission_model,
    predict_mission,
    prepare_missions,
)

ROOT = Path(__file__).resolve().parents[1]
MASTER_PATH = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "hurricane_oneway_verifier"
OUT.mkdir(parents=True, exist_ok=True)

EXTREME = 500_000_000.0

def band(v: float) -> str:
    if v < 200_000_000: return "50-200M"
    if v < 500_000_000: return "200-500M"
    return "500M+"

def fit_gate(train: pd.DataFrame, features: List[str]):
    y = (train["totalObligatedFunding"] >= EXTREME).astype(int)
    X = normalize_model_frame(train[features])
    model = LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5)
    pipe = prep_pipeline(X, model)
    pipe.fit(X, y)
    return pipe

def fit_lower(train: pd.DataFrame, features: List[str]):
    t = train[train["totalObligatedFunding"] < EXTREME].copy()
    y = (t["totalObligatedFunding"] >= 200_000_000).astype(int)
    X = normalize_model_frame(t[features])
    model = LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5)
    pipe = prep_pipeline(X, model)
    pipe.fit(X, y)
    return pipe

def agg_max(missions: pd.DataFrame, probs: np.ndarray) -> Dict[int, float]:
    z = missions[["disasterNumber"]].copy()
    z["p"] = probs
    return z.groupby("disasterNumber")["p"].max().to_dict()

def fire_threshold(train: pd.DataFrame) -> float:
    f = train[train["incidentType"] == "Fire"]
    neg = f.loc[f["totalObligatedFunding"] < EXTREME, "sem_esf_4_share"].dropna()
    pos = f.loc[f["totalObligatedFunding"] >= EXTREME, "sem_esf_4_share"].dropna()
    if len(neg) and len(pos):
        mx, mn = float(neg.max()), float(pos.min())
        return (mx + mn) / 2.0 if mx < mn else mn
    return 0.0

def choose_hurricane_reject_threshold(candidates: pd.DataFrame) -> float:
    """
    Candidate rows already satisfy baseline extreme==1.
    Keep candidate iff mega_prob >= threshold.
    Constraint: retain all inner true extremes if any exist.
    Among thresholds satisfying that, maximize false-positive rejection.
    """
    if candidates.empty:
        return 0.0
    vals = sorted(set([0.0, 1.01] + candidates["mega_prob"].astype(float).tolist()))
    best = 0.0
    best_rejects = -1
    true = candidates["actual_extreme"].to_numpy(int)
    p = candidates["mega_prob"].to_numpy(float)
    for th in vals:
        keep = (p >= th).astype(int)
        # Do not reject an observed true extreme.
        if true.sum() > 0 and np.any((true == 1) & (keep == 0)):
            continue
        rejects = int(np.sum((true == 0) & (keep == 0)))
        # Prefer more FP rejection, then higher threshold.
        if rejects > best_rejects or (rejects == best_rejects and th > best):
            best_rejects = rejects
            best = float(th)
    return best

def main():
    master = normalize_master(pd.read_excel(MASTER_PATH))
    ma = fetch_all_mission_assignments()
    missions = prepare_missions(master, ma)
    sem, _ = build_semantic_rollup(master, ma)
    ext, match_audit = build_external(master)
    match_audit.to_csv(OUT / "external_match_audit.csv", index=False)

    df = master.merge(sem, on="disasterNumber", how="left").merge(ext, on="disasterNumber", how="left")
    high = df[
        (df["incidentType"] != "Biological")
        & (df["totalObligatedFunding"] >= 50_000_000)
    ].copy().reset_index(drop=True)
    high["actual_band"] = high["totalObligatedFunding"].map(band)

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
    gate_features = current + sem_cols + ext_cols
    lower_features = current + sem_cols

    rows = []
    fold_info = []

    for outer in sorted(high["fyDeclared"].astype(int).unique()):
        train = high[high["fyDeclared"].astype(int) != outer].copy()
        test = high[high["fyDeclared"].astype(int) == outer].copy()

        # Inner calibration dataset for hurricane verifier.
        inner_candidates = []
        for inner in sorted(train["fyDeclared"].astype(int).unique()):
            itr = train[train["fyDeclared"].astype(int) != inner].copy()
            ite = train[train["fyDeclared"].astype(int) == inner].copy()
            if itr.empty or ite.empty:
                continue
            yitr = (itr["totalObligatedFunding"] >= EXTREME).astype(int)
            if yitr.nunique() < 2:
                continue
            g = fit_gate(itr, gate_features)
            gp = g.predict(normalize_model_frame(ite[gate_features]))

            # Mission model excludes outer and inner FY.
            mtr = missions[
                (missions["fyDeclared"].astype(int) != outer)
                & (missions["fyDeclared"].astype(int) != inner)
            ].copy()
            mte = missions[missions["fyDeclared"].astype(int) == inner].copy()
            amap = {}
            if not mte.empty and mtr["is_mega_ma"].nunique() >= 2:
                mm = fit_mission_model(mtr, use_text=False)
                mp = predict_mission(mm, mte)
                amap = agg_max(mte, mp)

            for (_, r), pred in zip(ite.iterrows(), gp):
                if r["incidentType"] == "Hurricane" and int(pred) == 1:
                    inner_candidates.append({
                        "disasterNumber": int(r["disasterNumber"]),
                        "actual_extreme": int(r["totalObligatedFunding"] >= EXTREME),
                        "mega_prob": float(amap.get(int(r["disasterNumber"]), 0.0)),
                    })

        inner_df = pd.DataFrame(inner_candidates)
        hth = choose_hurricane_reject_threshold(inner_df) if not inner_df.empty else 0.0

        # Outer baseline models.
        gate = fit_gate(train, gate_features)
        lower = fit_lower(train, lower_features)
        gp = gate.predict(normalize_model_frame(test[gate_features]))
        lp = lower.predict(normalize_model_frame(test[lower_features]))

        # Fire verifier threshold from outer-training years.
        fth = fire_threshold(train)

        # Outer mission mega probabilities.
        mtr = missions[missions["fyDeclared"].astype(int) != outer].copy()
        mte = missions[missions["fyDeclared"].astype(int) == outer].copy()
        amap = {}
        if not mte.empty and mtr["is_mega_ma"].nunique() >= 2:
            mm = fit_mission_model(mtr, use_text=False)
            mp = predict_mission(mm, mte)
            amap = agg_max(mte, mp)

        fold_info.append({
            "outer_fy": int(outer),
            "hurricane_threshold": float(hth),
            "inner_hurricane_candidates": int(len(inner_df)),
            "inner_true_extreme_candidates": int(inner_df["actual_extreme"].sum()) if not inner_df.empty else 0,
            "fire_threshold": float(fth),
        })

        for (_, r), gpred, lpred in zip(test.iterrows(), gp, lp):
            extreme = int(gpred)
            mega_prob = float(amap.get(int(r["disasterNumber"]), 0.0))

            # Existing fire verifier.
            if extreme == 1 and r["incidentType"] == "Fire":
                esf4 = float(r["sem_esf_4_share"]) if pd.notna(r["sem_esf_4_share"]) else 0.0
                if esf4 < fth:
                    extreme = 0

            # Conservative hurricane one-way verifier.
            if extreme == 1 and r["incidentType"] == "Hurricane":
                if mega_prob < hth:
                    extreme = 0

            final = "500M+" if extreme else ("200-500M" if int(lpred) else "50-200M")
            rows.append({
                "disasterNumber": int(r["disasterNumber"]),
                "state": r["state"],
                "incidentType": r["incidentType"],
                "fyDeclared": int(r["fyDeclared"]),
                "totalObligatedFunding": float(r["totalObligatedFunding"]),
                "actual_band": r["actual_band"],
                "baseline_gate": int(gpred),
                "mega_prob_struct": mega_prob,
                "hurricane_threshold": float(hth),
                "final_pred": final,
            })

    pred = pd.DataFrame(rows)
    pred.to_csv(OUT / "predictions.csv", index=False)
    pd.DataFrame(fold_info).to_csv(OUT / "fold_thresholds.csv", index=False)

    per_band = {}
    for b in ["50-200M","200-500M","500M+"]:
        m = pred["actual_band"] == b
        c = int((pred.loc[m,"actual_band"] == pred.loc[m,"final_pred"]).sum())
        n = int(m.sum())
        per_band[b] = {"correct":c,"total":n,"recall":c/n if n else None}

    summary = {
        "overall_correct": int((pred["actual_band"] == pred["final_pred"]).sum()),
        "overall_total": int(len(pred)),
        "overall_accuracy": float((pred["actual_band"] == pred["final_pred"]).mean()),
        "per_band": per_band,
        "errors": pred[pred["actual_band"] != pred["final_pred"]][
            ["disasterNumber","state","incidentType","actual_band","final_pred","mega_prob_struct","hurricane_threshold"]
        ].to_dict(orient="records"),
        "protocol": (
            "Outer LFYO. Hurricane reject threshold calibrated from inner LFYO baseline-extreme "
            "hurricane candidates using mission models excluding both outer and inner FY. "
            "Verifier may reject but never create an extreme call."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Hurricane one-way mega-Mission verifier",
        "",
        f"- Overall: **{summary['overall_correct']}/{summary['overall_total']} = {summary['overall_accuracy']:.1%}**",
    ]
    for b in ["50-200M","200-500M","500M+"]:
        x=per_band[b]
        md.append(f"- {b}: **{x['correct']}/{x['total']} = {x['recall']:.1%}**")
    md += ["","## Remaining errors"]
    for e in summary["errors"]:
        md.append(
            f"- FEMA {e['disasterNumber']} {e['state']} {e['incidentType']}: "
            f"{e['actual_band']} -> {e['final_pred']} "
            f"(mega={e['mega_prob_struct']:.3f}, th={e['hurricane_threshold']:.3f})"
        )
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

if __name__ == "__main__":
    main()
