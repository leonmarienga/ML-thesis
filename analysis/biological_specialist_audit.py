#!/usr/bin/env python3
"""
Biological specialist audit.

Goals
-----
1) Quantify true cold-start performance on FY2020 Biological high-value cases
   when NO FY2020 labels are allowed into training.
2) Quantify whether Biological cases are internally separable if the model is
   allowed to learn from other Biological cases (leave-one-out within FY2020).
3) Compare a simple specialist branch against the non-Biological hierarchy.

Important:
- Biological high-value cases are all FY2020 in this dataset.
- Therefore strict LFYO means zero labeled Biological high-value training examples.
- Any within-Biological result is diagnostic, not temporal generalization evidence.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score, accuracy_score
from sklearn.model_selection import LeaveOneOut

from mission_semantic_audit import (
    CURRENT_19,
    build_semantic_rollup,
    fetch_all_mission_assignments,
    normalize_master,
    normalize_model_frame,
    prep_pipeline,
)

ROOT = Path(__file__).resolve().parents[1]
MASTER_PATH = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "biological_specialist"
OUT.mkdir(parents=True, exist_ok=True)

def band(v: float) -> str:
    if v < 200_000_000: return "50-200M"
    if v < 500_000_000: return "200-500M"
    return "500M+"

def metrics(y, pred, labels):
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "per_band": {}
    }
    for lab in labels:
        m = np.asarray(y) == lab
        out["per_band"][lab] = {
            "correct": int((np.asarray(pred)[m] == np.asarray(y)[m]).sum()),
            "total": int(m.sum()),
            "recall": float((np.asarray(pred)[m] == np.asarray(y)[m]).mean()) if m.any() else None,
        }
    return out

def fit_predict(train, test, features, kind="rf"):
    Xtr = normalize_model_frame(train[features])
    Xte = normalize_model_frame(test[features])
    ytr = train["funding_band"].astype(str).to_numpy()
    if kind == "logistic":
        model = LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5)
    else:
        model = RandomForestClassifier(
            n_estimators=600,
            random_state=42,
            class_weight="balanced_subsample",
            max_features="sqrt",
            min_samples_leaf=1,
        )
    pipe = prep_pipeline(Xtr, model)
    pipe.fit(Xtr, ytr)
    return pipe.predict(Xte)

def main():
    master = normalize_master(pd.read_excel(MASTER_PATH))
    ma = fetch_all_mission_assignments()
    sem, _ = build_semantic_rollup(master, ma)
    df = master.merge(sem, on="disasterNumber", how="left")

    high = df[df["totalObligatedFunding"] >= 50_000_000].copy()
    high["funding_band"] = high["totalObligatedFunding"].map(band)

    bio = high[high["incidentType"] == "Biological"].copy().reset_index(drop=True)
    nonbio = high[high["incidentType"] != "Biological"].copy().reset_index(drop=True)

    assert len(bio) == 32, len(bio)
    assert set(bio["fyDeclared"].astype(int).unique()) == {2020}
    labels = ["50-200M","200-500M","500M+"]

    current = [c for c in CURRENT_19 if c in high.columns]
    sem_cols = [
        c for c in high.columns
        if (c.startswith("sem_") or c.startswith("ma_"))
        and high[c].notna().sum() >= 2
        and high[c].nunique(dropna=True) > 1
    ]
    features = current + sem_cols

    # 1) Strict cold-start: train on non-Biological high-value disasters only.
    cold_results = {}
    for kind in ["rf","logistic"]:
        pred = fit_predict(nonbio, bio, features, kind=kind)
        cold_results[kind] = metrics(bio["funding_band"].astype(str).to_numpy(), pred, labels)

    # 2) Simplest strict prior-only baseline:
    # use the nonbio class prior / majority class for every Biological case.
    majority = nonbio["funding_band"].value_counts().idxmax()
    prior_pred = np.array([majority] * len(bio), dtype=object)
    cold_results["majority_nonbio"] = metrics(
        bio["funding_band"].astype(str).to_numpy(), prior_pred, labels
    )

    # 3) Within-Biological LOO diagnostic.
    loo_results = {}
    for kind in ["rf","logistic"]:
        preds = np.empty(len(bio), dtype=object)
        loo = LeaveOneOut()
        for tr_idx, te_idx in loo.split(bio):
            tr = bio.iloc[tr_idx].copy()
            te = bio.iloc[te_idx].copy()
            # If one fold loses the only representative of a class, sklearn may
            # still fit remaining classes; record prediction honestly.
            preds[te_idx[0]] = fit_predict(tr, te, features, kind=kind)[0]
        loo_results[kind] = metrics(
            bio["funding_band"].astype(str).to_numpy(), preds, labels
        )

    # 4) Rule-based Biological specialist: ranking by final mission structure only.
    # This is retrospective diagnostic because final mission aggregates are used.
    # We search thresholds only within bio using leave-one-out rank logic:
    # top k predicted as 500M+, next m as 200-500M, rest lower.
    # k/m are fixed to training counts inside each LOO fold, not test labels.
    rule_preds = np.empty(len(bio), dtype=object)
    score_cols = [c for c in [
        "responseComplexityScore","missionAssignmentCount","uniqueAgencyCount",
        "uniqueMaTypeCount","uniquePriorityCount","missionDensity","agencyDensity"
    ] if c in bio.columns]
    for i in range(len(bio)):
        tr = bio.drop(index=i).copy()
        te = bio.iloc[[i]].copy()
        med = tr[score_cols].median(numeric_only=True)
        mad = (tr[score_cols] - med).abs().median().replace(0, 1.0)
        tr_score = ((tr[score_cols] - med) / mad).fillna(0).mean(axis=1)
        te_score = float(((te[score_cols] - med) / mad).fillna(0).mean(axis=1).iloc[0])

        n_ext = int((tr["funding_band"] == "500M+").sum())
        n_mid = int((tr["funding_band"] == "200-500M").sum())
        ordered = np.sort(tr_score.to_numpy())[::-1]
        th_ext = ordered[n_ext-1] if n_ext > 0 else np.inf
        th_mid = ordered[min(n_ext+n_mid-1, len(ordered)-1)] if n_mid > 0 else np.inf

        if te_score >= th_ext:
            rule_preds[i] = "500M+"
        elif te_score >= th_mid:
            rule_preds[i] = "200-500M"
        else:
            rule_preds[i] = "50-200M"

    rule_result = metrics(
        bio["funding_band"].astype(str).to_numpy(), rule_preds, labels
    )

    case_table = bio[[
        "disasterNumber","state","fyDeclared","totalObligatedFunding","funding_band"
    ] + score_cols].copy()
    case_table.to_csv(OUT / "biological_cases.csv", index=False)

    summary = {
        "bio_count": len(bio),
        "bio_band_counts": bio["funding_band"].value_counts().to_dict(),
        "strict_cold_start": cold_results,
        "within_bio_loo_diagnostic": loo_results,
        "within_bio_rank_rule_diagnostic": rule_result,
        "interpretation": {
            "strict_cold_start": (
                "Scientifically valid temporal test: no FY2020 Biological labels in training."
            ),
            "within_bio_loo": (
                "Diagnostic only: shows internal separability if other FY2020 Biological labels are available."
            ),
        },
    }
    (OUT / "biological_specialist_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    md = [
        "# Biological specialist audit",
        "",
        f"- Biological high-value cases: **{len(bio)}**",
        f"- Bands: **{bio['funding_band'].value_counts().to_dict()}**",
        "",
        "## Strict FY2020 cold-start",
        "",
    ]
    for name, r in cold_results.items():
        md.append(
            f"- {name}: BA={r['balanced_accuracy']:.3f}; "
            f"50-200={r['per_band']['50-200M']['correct']}/{r['per_band']['50-200M']['total']}; "
            f"200-500={r['per_band']['200-500M']['correct']}/{r['per_band']['200-500M']['total']}; "
            f"500M+={r['per_band']['500M+']['correct']}/{r['per_band']['500M+']['total']}"
        )

    md += ["", "## Within-Biological LOO diagnostic", ""]
    for name, r in loo_results.items():
        md.append(
            f"- {name}: BA={r['balanced_accuracy']:.3f}; "
            f"50-200={r['per_band']['50-200M']['correct']}/{r['per_band']['50-200M']['total']}; "
            f"200-500={r['per_band']['200-500M']['correct']}/{r['per_band']['200-500M']['total']}; "
            f"500M+={r['per_band']['500M+']['correct']}/{r['per_band']['500M+']['total']}"
        )
    md += [
        "",
        "## Within-Biological rank rule diagnostic",
        "",
        f"- BA={rule_result['balanced_accuracy']:.3f}; "
        f"50-200={rule_result['per_band']['50-200M']['correct']}/{rule_result['per_band']['50-200M']['total']}; "
        f"200-500={rule_result['per_band']['200-500M']['correct']}/{rule_result['per_band']['200-500M']['total']}; "
        f"500M+={rule_result['per_band']['500M+']['correct']}/{rule_result['per_band']['500M+']['total']}",
        "",
        "Strict cold-start is the only result that can be used as temporal generalization evidence."
    ]
    (OUT / "biological_specialist_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

if __name__ == "__main__":
    main()
