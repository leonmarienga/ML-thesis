#!/usr/bin/env python3
"""
Strict post-824 Severe Storm rescue at the $100K boundary.

Frozen architecture:
- accepted 824 six-band predictions are the baseline;
- no row predicted >=$50M can be changed;
- only Severe Storm rows currently predicted 0-100K are eligible;
- eligible rows may only be promoted to 100K-1M.

Discovery is target-blind with respect to each outer fiscal year. Candidate
numeric operational/mission-semantic features are selected from OTHER years
using nested LFYO. For each feature and direction, the inner-fold threshold is
constructed from inner-training true 0-100K Severe Storms so that none of those
training negatives can be promoted. A feature is eligible only when its inner
held-out predictions produce zero false promotions, at least two true rescues,
and true rescues in at least two distinct held-out fiscal years. The outer
threshold is then reconstructed from all outer-training 0-100K Severe Storms.

This intentionally favors precision over recall.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mission_semantic_audit import (
    CURRENT_19, build_semantic_rollup, fetch_all_mission_assignments,
    normalize_master,
)
from nonbio_all_ranges import funding_band, six_band_metrics, valid_cols

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
ACCEPTED = (
    ROOT / "audit_inputs" / "post824_storm_100k_rescue" / "accepted"
    / "candidate_predictions.csv"
)
OUT = ROOT / "audit_outputs" / "nonbio_824_storm_100k_zero_fp_rescue"
OUT.mkdir(parents=True, exist_ok=True)

HAZARD = "Severe Storm"
BASE_PRED = "candidate_pred"
LOW = "0-100K"
MID = "100K-1M"

# Current19 categorical/time fields are deliberately excluded from univariate
# threshold discovery. Semantic rollup fields are non-financial by construction.
EXCLUDE = {
    "fyDeclared", "state", "incidentType", "expectedResourceLevel",
    "disasterCategory", "durationClass",
}


def trigger(values: pd.Series, threshold: float, direction: str) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    if direction == "ge":
        return x >= threshold
    return x <= threshold


def threshold_from_negatives(neg_values: pd.Series, direction: str):
    x = pd.to_numeric(neg_values, errors="coerce").dropna()
    if x.empty:
        return None
    if direction == "ge":
        v = float(x.max())
        return float(np.nextafter(v, np.inf))
    v = float(x.min())
    return float(np.nextafter(v, -np.inf))


def inner_score(train: pd.DataFrame, feature: str, direction: str):
    rows = []
    for fy in sorted(train["fyDeclared"].astype(int).unique()):
        tr = train[train["fyDeclared"].astype(int) != fy].copy()
        te = train[train["fyDeclared"].astype(int) == fy].copy()
        neg = tr[tr["target_clean"] < 100_000]
        pos = tr[(tr["target_clean"] >= 100_000) & (tr["target_clean"] < 1_000_000)]
        te = te[te["target_clean"] < 1_000_000].copy()
        if neg.empty or pos.empty or te.empty:
            continue
        th = threshold_from_negatives(neg[feature], direction)
        if th is None:
            continue
        hit = trigger(te[feature], th, direction)
        for (_, r), h in zip(te.iterrows(), hit):
            if not bool(h):
                continue
            rows.append({
                "fy": int(fy),
                "is_positive": bool(100_000 <= float(r["target_clean"]) < 1_000_000),
            })
    if not rows:
        return {"tp": 0, "fp": 0, "tp_years": 0}
    rr = pd.DataFrame(rows)
    tp_rows = rr[rr["is_positive"]]
    return {
        "tp": int(rr["is_positive"].sum()),
        "fp": int((~rr["is_positive"]).sum()),
        "tp_years": int(tp_rows["fy"].nunique()),
    }


def main():
    master = normalize_master(pd.read_excel(MASTER))
    master["disasterNumber"] = master["disasterNumber"].astype(int)
    master["target_clean"] = pd.to_numeric(
        master["totalObligatedFunding"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
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
    if baseline_correct != 824:
        raise AssertionError(f"Expected accepted score 824, got {baseline_correct}")

    df = df.merge(
        accepted[["disasterNumber", BASE_PRED]],
        on="disasterNumber", how="inner", validate="one_to_one",
    )
    if len(df) != 912:
        raise AssertionError(f"Expected 912 merged rows, got {len(df)}")

    current = [c for c in valid_cols(df, CURRENT_19) if c not in EXCLUDE]
    semantic = [
        c for c in df.columns
        if (c.startswith("sem_") or c.startswith("ma_")) and c not in EXCLUDE
    ]
    candidates = []
    for c in list(dict.fromkeys(current + semantic)):
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().sum() >= 20 and x.nunique(dropna=True) >= 2:
            candidates.append(c)

    pred = dict(zip(df["disasterNumber"].astype(int), df[BASE_PRED].astype(str)))
    changed = []
    fold_rows = []

    for outer_fy in sorted(df["fyDeclared"].astype(int).unique()):
        train = df[
            (df["fyDeclared"].astype(int) != outer_fy)
            & (df["incidentType"] == HAZARD)
            & (df["target_clean"] < 1_000_000)
        ].copy()
        test = df[
            (df["fyDeclared"].astype(int) == outer_fy)
            & (df["incidentType"] == HAZARD)
            & (df[BASE_PRED] == LOW)
        ].copy()

        train_neg = train[train["target_clean"] < 100_000]
        train_pos = train[(train["target_clean"] >= 100_000) & (train["target_clean"] < 1_000_000)]
        best = None
        diagnostics = []

        if not train_neg.empty and not train_pos.empty:
            for feature in candidates:
                for direction in ("ge", "le"):
                    score = inner_score(train, feature, direction)
                    if score["fp"] != 0 or score["tp"] < 2 or score["tp_years"] < 2:
                        continue
                    th = threshold_from_negatives(train_neg[feature], direction)
                    if th is None:
                        continue
                    outer_train_pos_hits = int(trigger(train_pos[feature], th, direction).sum())
                    if outer_train_pos_hits < 2:
                        continue
                    rec = {
                        "feature": feature,
                        "direction": direction,
                        "threshold": float(th),
                        "inner_tp": int(score["tp"]),
                        "inner_fp": int(score["fp"]),
                        "inner_tp_years": int(score["tp_years"]),
                        "outer_train_pos_hits": outer_train_pos_hits,
                    }
                    diagnostics.append(rec)
                    # Prefer more held-out true rescues, support across more years,
                    # then more full outer-training positives covered.
                    key = (score["tp"], score["tp_years"], outer_train_pos_hits)
                    if best is None or key > best[0]:
                        best = (key, rec)

        promoted = []
        if best is not None and not test.empty:
            rec = best[1]
            hits = trigger(test[rec["feature"]], rec["threshold"], rec["direction"])
            for (_, r), h in zip(test.iterrows(), hits):
                if not bool(h):
                    continue
                dn = int(r["disasterNumber"])
                old = pred[dn]
                pred[dn] = MID
                promoted.append(dn)
                changed.append({
                    "disasterNumber": dn,
                    "state": r["state"],
                    "fyDeclared": int(r["fyDeclared"]),
                    "actual_band": r["actual_band"],
                    "old_pred": old,
                    "new_pred": MID,
                    "feature": rec["feature"],
                    "direction": rec["direction"],
                    "threshold": rec["threshold"],
                    "feature_value": float(pd.to_numeric(pd.Series([r[rec["feature"]]]), errors="coerce").iloc[0]),
                    "inner_tp": rec["inner_tp"],
                    "inner_fp": rec["inner_fp"],
                    "inner_tp_years": rec["inner_tp_years"],
                })

        fold_rows.append({
            "outer_fy": int(outer_fy),
            "train_negative_n": int(len(train_neg)),
            "train_positive_n": int(len(train_pos)),
            "eligible_test_n": int(len(test)),
            "selected_feature": None if best is None else best[1]["feature"],
            "selected_direction": None if best is None else best[1]["direction"],
            "selected_threshold": None if best is None else best[1]["threshold"],
            "inner_tp": None if best is None else best[1]["inner_tp"],
            "inner_fp": None if best is None else best[1]["inner_fp"],
            "inner_tp_years": None if best is None else best[1]["inner_tp_years"],
            "outer_train_pos_hits": None if best is None else best[1]["outer_train_pos_hits"],
            "promotions": int(len(promoted)),
            "candidate_feature_count": int(len(diagnostics)),
        })

    out = df[[
        "disasterNumber", "state", "incidentType", "fyDeclared", "target_clean",
        "actual_band", BASE_PRED,
    ]].copy()
    out["candidate_pred"] = out["disasterNumber"].map(pred)
    out.to_csv(OUT / "candidate_predictions.csv", index=False)
    pd.DataFrame(changed).to_csv(OUT / "changed_rows.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "fold_diagnostics.csv", index=False)

    candidate_correct = int((out["candidate_pred"] == out["actual_band"]).sum())
    metrics = six_band_metrics(out, "candidate_pred")
    protected = out[df["target_clean"] >= 50_000_000]
    protected_changes = int((protected[BASE_PRED] != protected["candidate_pred"]).sum())
    correct_changes = int(sum(r["actual_band"] == MID for r in changed))
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
            "Strict outer LFYO; per-fold univariate feature/direction selected by inner LFYO "
            "with zero held-out false promotions, >=2 held-out true rescues, and support in "
            ">=2 held-out fiscal years. Outer threshold is defined solely by outer-training "
            "true 0-100K Severe Storms. Only accepted 0-100K Severe Storm predictions can "
            "be promoted to 100K-1M."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Post-824 Severe Storm $100K zero-FP rescue",
        "",
        f"- Baseline: **{baseline_correct}/912**",
        f"- Candidate: **{candidate_correct}/912**",
        f"- Changed rows: **{len(changed)}** ({correct_changes} correct, {wrong_changes} wrong)",
        f"- >=$50M prediction changes: **{protected_changes}**",
        "",
        "## Changed rows",
    ]
    for r in changed:
        md.append(
            f"- FEMA {r['disasterNumber']} {r['state']} FY{r['fyDeclared']}: "
            f"{r['actual_band']} | {r['old_pred']} -> {r['new_pred']} | "
            f"{r['feature']} {r['direction']} {r['threshold']:.6g} "
            f"(value={r['feature_value']:.6g}; inner TP={r['inner_tp']}, FP={r['inner_fp']}, years={r['inner_tp_years']})"
        )
    md += ["", "## Six-band recall"]
    for b, m in metrics["per_band"].items():
        md.append(f"- {b}: **{m['correct']}/{m['total']} = {m['recall']:.1%}**")
    md.append(f"- Macro recall: **{metrics['macro_recall']:.3f}**")
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

    if protected_changes != 0:
        raise AssertionError("Candidate changed protected >=$50M predictions")


if __name__ == "__main__":
    main()
