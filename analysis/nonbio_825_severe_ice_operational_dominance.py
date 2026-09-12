#!/usr/bin/env python3
"""
Post-825 Severe Ice Storm rescue at the $1M boundary.

A held-out Severe Ice Storm currently predicted $100K-$1M is promoted to
$1M-$50M only if it strictly exceeds every sub-$1M Severe Ice Storm in the
outer-training years on BOTH:
  1) missionAssignmentCount, and
  2) ma_max_amendment.

The fold is enabled only when every true $1M-$50M Severe Ice Storm in the
outer-training years also strictly exceeds those two sub-$1M maxima.

This is a fixed two-dimensional operational-dominance mechanism. Thresholds
are learned from outer-training negatives only; no test-year labels are used.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mission_semantic_audit import build_semantic_rollup, fetch_all_mission_assignments, normalize_master
from nonbio_all_ranges import funding_band, six_band_metrics

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
ACCEPTED = ROOT / "audit_inputs" / "post825_severe_ice" / "accepted" / "candidate_predictions.csv"
OUT = ROOT / "audit_outputs" / "nonbio_825_severe_ice_operational_dominance"
OUT.mkdir(parents=True, exist_ok=True)

HAZARD = "Severe Ice Storm"
BASE_PRED = "candidate_pred"
LOWER_BAND = "100K-1M"
UPPER_BAND = "1M-50M"


def num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def main():
    master = normalize_master(pd.read_excel(MASTER))
    master["disasterNumber"] = master["disasterNumber"].astype(int)
    master["target_clean"] = num(master["totalObligatedFunding"]).clip(lower=0.0)
    master["actual_band"] = master["target_clean"].map(funding_band)

    ma = fetch_all_mission_assignments()
    sem, _ = build_semantic_rollup(master, ma)
    df = master.merge(sem, on="disasterNumber", how="left")
    df = df[df["incidentType"] != "Biological"].copy().reset_index(drop=True)

    accepted = pd.read_csv(ACCEPTED)
    accepted["disasterNumber"] = accepted["disasterNumber"].astype(int)
    if len(accepted) != 912:
        raise AssertionError(f"Expected 912 accepted rows, got {len(accepted)}")
    baseline_correct = int((accepted[BASE_PRED] == accepted["actual_band"]).sum())
    if baseline_correct != 825:
        raise AssertionError(f"Expected accepted score 825, got {baseline_correct}")

    df = df.merge(accepted[["disasterNumber", BASE_PRED]], on="disasterNumber", how="inner", validate="one_to_one")
    if len(df) != 912:
        raise AssertionError(f"Expected 912 merged rows, got {len(df)}")

    for c in ["missionAssignmentCount", "ma_max_amendment"]:
        df[c] = num(df[c])

    pred = dict(zip(df["disasterNumber"].astype(int), df[BASE_PRED].astype(str)))
    changed = []
    fold_rows = []

    ice = df[(df["incidentType"] == HAZARD) & (df["target_clean"] < 50_000_000)].copy()

    for outer_fy in sorted(df["fyDeclared"].astype(int).unique()):
        train = ice[ice["fyDeclared"].astype(int) != outer_fy].copy()
        test = ice[(ice["fyDeclared"].astype(int) == outer_fy) & (ice[BASE_PRED] == LOWER_BAND)].copy()

        low = train[train["target_clean"] < 1_000_000].copy()
        high = train[(train["target_clean"] >= 1_000_000) & (train["target_clean"] < 50_000_000)].copy()

        enabled = False
        max_low_missions = None
        max_low_amendment = None
        if not low.empty and not high.empty:
            max_low_missions = float(low["missionAssignmentCount"].max())
            max_low_amendment = float(low["ma_max_amendment"].max())
            enabled = bool(
                ((high["missionAssignmentCount"] > max_low_missions)
                 & (high["ma_max_amendment"] > max_low_amendment)).all()
            )

        promoted = []
        if enabled and not test.empty:
            hit = (
                (test["missionAssignmentCount"] > max_low_missions)
                & (test["ma_max_amendment"] > max_low_amendment)
            )
            for _, r in test[hit].iterrows():
                dn = int(r["disasterNumber"])
                old = pred[dn]
                pred[dn] = UPPER_BAND
                promoted.append(dn)
                changed.append({
                    "disasterNumber": dn,
                    "state": r["state"],
                    "fyDeclared": int(r["fyDeclared"]),
                    "target_clean": float(r["target_clean"]),
                    "actual_band": r["actual_band"],
                    "old_pred": old,
                    "new_pred": UPPER_BAND,
                    "missionAssignmentCount": float(r["missionAssignmentCount"]),
                    "ma_max_amendment": float(r["ma_max_amendment"]),
                    "train_sub1m_max_missions": max_low_missions,
                    "train_sub1m_max_amendment": max_low_amendment,
                })

        fold_rows.append({
            "outer_fy": int(outer_fy),
            "train_sub1m_n": int(len(low)),
            "train_1m_50m_n": int(len(high)),
            "eligible_test_n": int(len(test)),
            "enabled": bool(enabled),
            "train_sub1m_max_missions": max_low_missions,
            "train_sub1m_max_amendment": max_low_amendment,
            "promotions": int(len(promoted)),
            "promoted_disaster_numbers": ";".join(str(x) for x in promoted),
        })

    out = df[["disasterNumber", "state", "incidentType", "fyDeclared", "target_clean", "actual_band", BASE_PRED]].copy()
    out["candidate_pred"] = out["disasterNumber"].map(pred)
    out.to_csv(OUT / "candidate_predictions.csv", index=False)
    pd.DataFrame(changed).to_csv(OUT / "changed_rows.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "fold_diagnostics.csv", index=False)

    candidate_correct = int((out["candidate_pred"] == out["actual_band"]).sum())
    metrics = six_band_metrics(out, "candidate_pred")
    protected = out[out["target_clean"] >= 50_000_000]
    protected_changes = int((protected[BASE_PRED] != protected["candidate_pred"]).sum())
    correct_changes = int(sum(r["actual_band"] == UPPER_BAND for r in changed))
    wrong_changes = int(len(changed) - correct_changes)

    summary = {
        "baseline_correct": baseline_correct,
        "candidate_correct": candidate_correct,
        "total": int(len(out)),
        "changed_count": int(len(changed)),
        "correct_changes": correct_changes,
        "wrong_changes": wrong_changes,
        "protected_ge_50m_changes": protected_changes,
        "metrics": metrics,
        "protocol": (
            "Strict outer LFYO Severe Ice Storm operational-dominance rescue. "
            "Per fold, sub-$1M training maxima define both thresholds. The rule is enabled only "
            "when every outer-training $1M-$50M Severe Ice case exceeds both maxima. Only held-out "
            "Severe Ice cases already predicted $100K-$1M can be promoted."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Post-825 Severe Ice operational-dominance rescue",
        "",
        f"- Baseline: **{baseline_correct}/912**",
        f"- Candidate: **{candidate_correct}/912**",
        f"- Changed rows: **{len(changed)}** ({correct_changes} correct, {wrong_changes} wrong)",
        f"- >=$50M prediction changes: **{protected_changes}**",
        "",
        "## Changed rows",
    ]
    if not changed:
        md.append("- None")
    for r in changed:
        md.append(
            f"- FEMA {r['disasterNumber']} {r['state']} FY{r['fyDeclared']}: "
            f"${r['target_clean']:,.2f} | actual {r['actual_band']} | {r['old_pred']} -> {r['new_pred']} | "
            f"missions={r['missionAssignmentCount']:.0f} > train-low max {r['train_sub1m_max_missions']:.0f}; "
            f"max amendment={r['ma_max_amendment']:.0f} > train-low max {r['train_sub1m_max_amendment']:.0f}"
        )
    md += ["", "## Six-band recall"]
    for band, vals in metrics["per_band"].items():
        md.append(f"- {band}: **{vals['correct']}/{vals['total']} = {vals['recall']:.1%}**")
    md.append(f"- Macro recall: **{metrics['macro_recall']:.3f}**")
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

    if protected_changes != 0:
        raise AssertionError("Candidate changed protected >=$50M predictions")


if __name__ == "__main__":
    main()
