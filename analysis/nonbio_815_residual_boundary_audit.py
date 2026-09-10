#!/usr/bin/env python3
"""
Residual $1M-boundary audit on the accepted 815/912 non-Biological router.

Scope
-----
Only rows that accepted 815 STILL predicts as $100K-$1M are examined for:
- Flood
- Severe Storm

This is the exact candidate pool a second one-way rescue would see. Negatives are
all candidates with actual funding <$1M (including true 0-$100K); positives are
actual $1M-$50M. No >=$50M row is eligible.

The audit ranks target-blind CURRENT_19 + MissionAssignment semantic features by
univariate ROC-AUC, reporting directionless separation max(AUC, 1-AUC), coverage,
class medians, and direction. This is diagnostic only; it does not tune a router.
Biological remains excluded/frozen.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from mission_semantic_audit import (
    CURRENT_19, build_semantic_rollup, fetch_all_mission_assignments,
    normalize_master,
)
from nonbio_all_ranges import valid_cols

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
ACCEPTED = ROOT / "audit_inputs" / "residual815" / "accepted" / "integrated_predictions.csv"
OUT = ROOT / "audit_outputs" / "nonbio_815_residual_boundary_audit"
OUT.mkdir(parents=True, exist_ok=True)
PRED_COL = "hurricane_integrated_pred"
HAZARDS = ["Flood", "Severe Storm"]


def rank_features(cand, features):
    y = (cand["target_clean"] >= 1_000_000).astype(int).to_numpy()
    rows = []
    for f in features:
        x = pd.to_numeric(cand[f], errors="coerce")
        m = x.notna().to_numpy()
        if int(m.sum()) < 8 or len(np.unique(y[m])) < 2:
            continue
        vals = x.to_numpy(float)[m]
        yy = y[m]
        try:
            auc = float(roc_auc_score(yy, vals))
        except Exception:
            continue
        sep = max(auc, 1.0 - auc)
        direction = "higher=>mid" if auc >= 0.5 else "lower=>mid"
        neg = vals[yy == 0]
        pos = vals[yy == 1]
        rows.append({
            "feature": f,
            "auc": auc,
            "separation_auc": sep,
            "direction": direction,
            "coverage": int(m.sum()),
            "coverage_pct": float(m.mean()),
            "negative_n": int((yy == 0).sum()),
            "positive_n": int((yy == 1).sum()),
            "negative_median": float(np.nanmedian(neg)) if len(neg) else None,
            "positive_median": float(np.nanmedian(pos)) if len(pos) else None,
            "negative_mean": float(np.nanmean(neg)) if len(neg) else None,
            "positive_mean": float(np.nanmean(pos)) if len(pos) else None,
        })
    return pd.DataFrame(rows).sort_values(
        ["separation_auc", "coverage"], ascending=[False, False]
    ) if rows else pd.DataFrame()


def main():
    master = normalize_master(pd.read_excel(MASTER))
    master["target_clean"] = pd.to_numeric(
        master["totalObligatedFunding"], errors="coerce"
    ).fillna(0).clip(lower=0)
    master = master[master["incidentType"] != "Biological"].copy()
    master["disasterNumber"] = master["disasterNumber"].astype(int)

    ma = fetch_all_mission_assignments()
    sem, _ = build_semantic_rollup(master, ma)
    df = master.merge(sem, on="disasterNumber", how="left")

    acc = pd.read_csv(ACCEPTED)
    acc["disasterNumber"] = acc["disasterNumber"].astype(int)
    assert PRED_COL in acc.columns
    df = df.merge(acc[["disasterNumber", PRED_COL]], on="disasterNumber", how="inner", validate="one_to_one")
    assert len(df) == 912

    current = valid_cols(df, CURRENT_19)
    semcols = valid_cols(
        df,
        [c for c in df.columns
         if (c.startswith("sem_") or c.startswith("ma_"))
         and not any(bad in c.lower() for bad in ["oblig", "fund", "cost", "amount", "dollar"])],
    )
    features = list(dict.fromkeys(current + semcols))

    summary = {"accepted_total": len(df), "feature_count": len(features), "hazards": {}}
    md = [
        "# Accepted 815 residual $1M-boundary audit",
        "",
        "Candidate pool = accepted 815 prediction $100K-$1M; positives = actual $1M-$50M; negatives = actual <$1M.",
        "",
    ]

    for hazard in HAZARDS:
        cand = df[
            (df["incidentType"] == hazard)
            & (df[PRED_COL] == "100K-1M")
            & (df["target_clean"] < 50_000_000)
        ].copy()
        cand["residual_label"] = np.where(cand["target_clean"] >= 1_000_000, "mid_miss", "true_below_1m")
        cand.to_csv(OUT / f"{hazard.lower().replace(' ','_')}_candidates.csv", index=False)

        n_pos = int((cand["target_clean"] >= 1_000_000).sum())
        n_neg = int((cand["target_clean"] < 1_000_000).sum())
        actual_counts = cand.assign(
            actual_group=np.where(cand["target_clean"] < 100_000, "0-100K", np.where(cand["target_clean"] < 1_000_000, "100K-1M", "1M-50M"))
        )["actual_group"].value_counts().to_dict()

        ranking = rank_features(cand, features)
        ranking.to_csv(OUT / f"{hazard.lower().replace(' ','_')}_feature_ranking.csv", index=False)
        top = ranking.head(15).to_dict(orient="records") if not ranking.empty else []

        summary["hazards"][hazard] = {
            "candidate_n": int(len(cand)),
            "positive_mid_miss_n": n_pos,
            "negative_below_1m_n": n_neg,
            "actual_group_counts": actual_counts,
            "top_features": top,
        }

        md += [
            f"## {hazard}",
            "",
            f"Candidates: **{len(cand)}**; remaining true $1M-$50M misses: **{n_pos}**; actual <$1M negatives: **{n_neg}**.",
            "",
            "| Feature | Separation AUC | Direction | Coverage | Negative median | Mid-miss median |",
            "|---|---:|---|---:|---:|---:|",
        ]
        for r in top:
            md.append(
                f"| {r['feature']} | {r['separation_auc']:.3f} | {r['direction']} | "
                f"{r['coverage']}/{len(cand)} | {r['negative_median']:.4g} | {r['positive_median']:.4g} |"
            )
        md.append("")

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
