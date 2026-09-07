#!/usr/bin/env python3
"""
Strict nested-LFYO recall-first >=$50M root-gate audit.

Biological remains excluded/frozen. The confirmed 22/23 high-value hierarchy
is frozen and reused unchanged.

Purpose
-------
The full-range audit showed that the main end-to-end bottleneck is the root
decision <50M vs >=50M. This experiment compares:

A. current19 logistic at 0.50 (reference)
B. current19 logistic with a threshold learned by INNER LFYO to achieve 100%
   recall on outer-training >=50M cases while maximizing precision
C. OR ensemble: current19 logistic OR semantics+external logistic
D. OR ensemble with both component thresholds calibrated by inner LFYO
E. global current19 root OR hazard-specific Fire/Hurricane logistic specialists

All threshold selection is performed using outer-training data only.
No held-out fiscal-year outcome is used for calibration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    precision_score, recall_score,
)

from mission_semantic_audit import (
    CURRENT_19, build_semantic_rollup, fetch_all_mission_assignments,
    normalize_master, normalize_model_frame, prep_pipeline,
)
from external_severity_ablation import build_external
from nonbio_hazard_hierarchy import initial_mechanism_counts
from nonbio_outage_rescue import build_eaglei_all
from nonbio_all_ranges import (
    BANDS, funding_band, high22_predict, fit_low_multiclass,
    six_band_metrics, valid_cols,
)

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "nonbio_root_recall"
OUT.mkdir(parents=True, exist_ok=True)


def fit_log(train: pd.DataFrame, features: List[str]):
    y = (train["target_clean"] >= 50_000_000).astype(int)
    X = normalize_model_frame(train[features])
    m = LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5)
    p = prep_pipeline(X, m)
    p.fit(X, y)
    return p


def proba(model, df: pd.DataFrame, features: List[str]) -> np.ndarray:
    return model.predict_proba(normalize_model_frame(df[features]))[:, 1]


def inner_oof_scores(outer_train: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    rows = []
    for fy in sorted(outer_train["fyDeclared"].astype(int).unique()):
        tr = outer_train[outer_train["fyDeclared"].astype(int) != fy].copy()
        te = outer_train[outer_train["fyDeclared"].astype(int) == fy].copy()
        ytr = (tr["target_clean"] >= 50_000_000).astype(int)
        if ytr.nunique() < 2:
            continue
        model = fit_log(tr, features)
        pp = proba(model, te, features)
        for (_, r), p in zip(te.iterrows(), pp):
            rows.append({
                "disasterNumber": int(r["disasterNumber"]),
                "fy": int(fy),
                "actual_high": int(r["target_clean"] >= 50_000_000),
                "prob": float(p),
            })
    return pd.DataFrame(rows)


def recall_first_threshold(oof: pd.DataFrame) -> Tuple[float, Dict]:
    """
    Highest threshold that preserves 100% recall on OOF positives.
    This maximizes precision among thresholds with perfect observed training recall.
    """
    if oof.empty or int(oof["actual_high"].sum()) == 0:
        return 0.5, {"reason": "no_oof_positives", "recall": None, "precision": None}

    pos = oof.loc[oof["actual_high"] == 1, "prob"]
    threshold = float(pos.min())
    pred = (oof["prob"] >= threshold).astype(int)
    return threshold, {
        "oof_n": int(len(oof)),
        "oof_positive_n": int(oof["actual_high"].sum()),
        "threshold": threshold,
        "recall": float(recall_score(oof["actual_high"], pred, pos_label=1, zero_division=0)),
        "precision": float(precision_score(oof["actual_high"], pred, pos_label=1, zero_division=0)),
        "fp": int(((oof["actual_high"] == 0) & (pred == 1)).sum()),
    }


def fit_hazard_specialist(train: pd.DataFrame, hazard: str, features: List[str]):
    t = train[train["incidentType"] == hazard].copy()
    y = (t["target_clean"] >= 50_000_000).astype(int)
    if len(t) < 10 or y.nunique() < 2:
        return None
    X = normalize_model_frame(t[features])
    m = LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5)
    p = prep_pipeline(X, m)
    p.fit(X, y)
    return p


def root_metrics(df: pd.DataFrame) -> Dict:
    y = df["root_actual_high"].to_numpy(int)
    p = df["root_pred_high"].to_numpy(int)
    return {
        "accuracy": float(accuracy_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "high_recall": float(recall_score(y, p, pos_label=1, zero_division=0)),
        "low_recall": float(recall_score(y, p, pos_label=0, zero_division=0)),
        "high_precision": float(precision_score(y, p, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, p, labels=[0, 1]).tolist(),
        "false_negative_high": df.loc[
            (df["root_actual_high"] == 1) & (df["root_pred_high"] == 0),
            ["disasterNumber", "state", "incidentType", "actual_band", "root_score"],
        ].to_dict(orient="records"),
        "false_positive_count": int(
            ((df["root_actual_high"] == 0) & (df["root_pred_high"] == 1)).sum()
        ),
    }


def main():
    master = normalize_master(pd.read_excel(MASTER))
    master["target_clean"] = pd.to_numeric(
        master["totalObligatedFunding"], errors="coerce"
    ).fillna(0).clip(lower=0)
    master["actual_band"] = master["target_clean"].map(funding_band)

    ma = fetch_all_mission_assignments()
    sem, _ = build_semantic_rollup(master, ma)
    ext, ext_audit = build_external(master)
    mech = initial_mechanism_counts(master, ma)
    print("Building EAGLE-I features...", flush=True)
    eag = build_eaglei_all(master)

    df = (
        master.merge(sem, on="disasterNumber", how="left")
        .merge(ext, on="disasterNumber", how="left")
        .merge(mech, on="disasterNumber", how="left")
        .merge(eag, on="disasterNumber", how="left")
    )
    df["initial_usace_esf3_dfa_count"] = (
        df["initial_usace_esf3_dfa_count"].fillna(0).astype(int)
    )
    nonbio = df[df["incidentType"] != "Biological"].copy().reset_index(drop=True)

    current = valid_cols(nonbio, CURRENT_19)
    semcols = valid_cols(
        nonbio, [c for c in nonbio.columns if c.startswith("sem_") or c.startswith("ma_")]
    )
    extcols = valid_cols(
        nonbio, [c for c in nonbio.columns
                 if c.startswith("nhc_") or c.startswith("noaa_") or c.startswith("calfire_")]
    )
    semext = list(dict.fromkeys(current + semcols + extcols))

    high_all = nonbio[nonbio["target_clean"] >= 50_000_000].copy()
    high_current = valid_cols(high_all, CURRENT_19)
    high_sem = valid_cols(
        high_all, [c for c in high_all.columns if c.startswith("sem_") or c.startswith("ma_")]
    )
    high_ext = valid_cols(
        high_all, [c for c in high_all.columns
                   if c.startswith("nhc_") or c.startswith("noaa_") or c.startswith("calfire_")]
    )
    high_gate_features = high_current + high_sem + high_ext
    high_lower_features = high_current + high_sem

    variants = [
        "current19_default",
        "current19_nested_recall",
        "or_current_semext_default",
        "or_current_semext_nested",
        "global_or_hazard_specialists",
    ]
    rows = {v: [] for v in variants}
    calibration_rows = []

    for fy in sorted(nonbio["fyDeclared"].astype(int).unique()):
        train = nonbio[nonbio["fyDeclared"].astype(int) != fy].copy()
        test = nonbio[nonbio["fyDeclared"].astype(int) == fy].copy()
        train_high = train[train["target_clean"] >= 50_000_000].copy()

        # Global models.
        cur = fit_log(train, current)
        semx = fit_log(train, semext)
        pcur = proba(cur, test, current)
        psemx = proba(semx, test, semext)

        # Nested recall-first thresholds.
        oof_cur = inner_oof_scores(train, current)
        th_cur, diag_cur = recall_first_threshold(oof_cur)
        oof_semx = inner_oof_scores(train, semext)
        th_semx, diag_semx = recall_first_threshold(oof_semx)
        calibration_rows.append({
            "outer_fy": int(fy),
            "current_threshold": th_cur,
            "current_oof_precision": diag_cur.get("precision"),
            "current_oof_fp": diag_cur.get("fp"),
            "semext_threshold": th_semx,
            "semext_oof_precision": diag_semx.get("precision"),
            "semext_oof_fp": diag_semx.get("fp"),
        })

        # Hazard specialists. Their use is one-way rescue only.
        fire_model = fit_hazard_specialist(train, "Fire", semext)
        hurr_model = fit_hazard_specialist(train, "Hurricane", semext)
        fire_prob = np.zeros(len(test), dtype=float)
        hurr_prob = np.zeros(len(test), dtype=float)
        if fire_model is not None:
            m = test["incidentType"].eq("Fire").to_numpy()
            if m.any():
                fire_prob[m] = proba(fire_model, test.loc[m], semext)
        if hurr_model is not None:
            m = test["incidentType"].eq("Hurricane").to_numpy()
            if m.any():
                hurr_prob[m] = proba(hurr_model, test.loc[m], semext)

        root_defs = {
            "current19_default": (pcur >= 0.5),
            "current19_nested_recall": (pcur >= th_cur),
            "or_current_semext_default": ((pcur >= 0.5) | (psemx >= 0.5)),
            "or_current_semext_nested": ((pcur >= th_cur) | (psemx >= th_semx)),
            "global_or_hazard_specialists": (
                (pcur >= 0.5)
                | (psemx >= 0.5)
                | (fire_prob >= 0.5)
                | (hurr_prob >= 0.5)
            ),
        }

        # Keep low specialist fixed so only root behavior changes.
        low_model = fit_low_multiclass(
            train, current, "log", seed=30000 + int(fy)
        )

        for v in variants:
            root_pred = root_defs[v].astype(int)
            pred_high_rows = test.loc[root_pred == 1].copy()
            high_map = high22_predict(
                train_high, pred_high_rows, high_gate_features, high_lower_features
            ) if not pred_high_rows.empty else {}

            pred_low_rows = test.loc[root_pred == 0].copy()
            low_map = {}
            if not pred_low_rows.empty:
                lp = low_model.predict(
                    normalize_model_frame(pred_low_rows[current])
                )
                low_map = {
                    int(dn): str(p)
                    for dn, p in zip(pred_low_rows["disasterNumber"], lp)
                }

            for j, (_, r) in enumerate(test.iterrows()):
                dn = int(r["disasterNumber"])
                rp = int(root_pred[j])
                final = high_map[dn] if rp else low_map[dn]
                if v.startswith("or_current_semext"):
                    score = max(float(pcur[j]), float(psemx[j]))
                elif v == "global_or_hazard_specialists":
                    score = max(float(pcur[j]), float(psemx[j]), float(fire_prob[j]), float(hurr_prob[j]))
                else:
                    score = float(pcur[j])

                rows[v].append({
                    "disasterNumber": dn,
                    "state": r["state"],
                    "incidentType": r["incidentType"],
                    "fyDeclared": int(r["fyDeclared"]),
                    "target_clean": float(r["target_clean"]),
                    "actual_band": r["actual_band"],
                    "root_actual_high": int(r["target_clean"] >= 50_000_000),
                    "root_pred_high": rp,
                    "root_score": score,
                    "p_current19": float(pcur[j]),
                    "p_semext": float(psemx[j]),
                    "p_fire_specialist": float(fire_prob[j]),
                    "p_hurricane_specialist": float(hurr_prob[j]),
                    "final_pred": final,
                })

    results = {}
    for v in variants:
        p = pd.DataFrame(rows[v])
        p.to_csv(OUT / f"{v}_predictions.csv", index=False)
        results[v] = {
            "root": root_metrics(p),
            "end_to_end": six_band_metrics(p, "final_pred"),
        }

    pd.DataFrame(calibration_rows).to_csv(OUT / "nested_thresholds.csv", index=False)
    ext_audit.to_csv(OUT / "external_match_audit.csv", index=False)

    summary = {
        "scope": "912 non-Biological declarations; Biological excluded",
        "conditional_high_branch_frozen": "22/23",
        "results": results,
        "notes": [
            "Nested recall thresholds are learned solely from inner-LFYO outer-training predictions.",
            "Hazard-specific models are one-way entry rescues only.",
            "The low specialist is fixed to current19 logistic to isolate root-gate changes.",
            "This remains developmental temporal validation, not external holdout validation.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Recall-first >=$50M root gate audit",
        "",
        "- Biological excluded/frozen.",
        "- High hierarchy frozen at **22/23 conditional accuracy**.",
        "- Evaluation: outer LFYO; recall thresholds calibrated by inner LFYO only.",
        "",
        "| Variant | High recall | High precision | Root FP | End-to-end | Macro recall | 0-100K | 100K-1M | 1M-50M | 50-200M | 200-500M | 500M+ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for v in variants:
        r = results[v]
        g = r["root"]
        e = r["end_to_end"]
        pb = e["per_band"]
        md.append(
            f"| {v} | {g['high_recall']:.1%} | {g['high_precision']:.1%} | "
            f"{g['false_positive_count']} | "
            f"{e['overall_correct']}/{e['overall_total']} ({e['overall_accuracy']:.1%}) | "
            f"{e['macro_recall']:.1%} | "
            + " | ".join(
                f"{pb[b]['correct']}/{pb[b]['total']} ({pb[b]['recall']:.1%})"
                for b in BANDS
            )
            + " |"
        )

    md += ["", "## Remaining root false-negative >=$50M cases"]
    for v in variants:
        errs = results[v]["root"]["false_negative_high"]
        md.append(f"### {v}")
        if not errs:
            md.append("- **None**")
        else:
            for e in errs:
                md.append(
                    f"- FEMA {e['disasterNumber']} {e['state']} {e['incidentType']} "
                    f"{e['actual_band']} score={e['root_score']:.4f}"
                )

    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
