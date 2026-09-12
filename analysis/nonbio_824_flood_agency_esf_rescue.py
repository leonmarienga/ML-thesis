#!/usr/bin/env python3
"""
Strict post-824 Flood rescue across the $1M boundary.

Mechanism hypothesis
--------------------
A Flood currently routed to $100K-$1M is promoted to $1M-$50M only when:

1. It contains a specialist federal agency whose presence is learned entirely
   from OTHER fiscal years.  An agency enters the outer-fold whitelist only if,
   among outer-training Floods below $50M, that agency:
      - appears in at least 2 true $1M-$50M Floods,
      - appears across at least 2 distinct fiscal years, and
      - appears in ZERO Floods below $1M.
2. The held-out Flood also has at least one ESF-coded mission assignment.

The agency identity is therefore not hardcoded.  The ESF key is fixed and
mechanistic: the specialist-agency evidence must be accompanied by an actual
Emergency Support Function assignment.

Safety constraints
------------------
- Baseline is the accepted 824/912 router artifact.
- Only Flood rows currently predicted $100K-$1M are eligible.
- Promotion can only be to $1M-$50M.
- No prediction >=$50M can change.
- Biological disasters remain excluded.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mission_semantic_audit import (
    build_semantic_rollup,
    fetch_all_mission_assignments,
    normalize_master,
)
from nonbio_all_ranges import funding_band, six_band_metrics

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
ACCEPTED = (
    ROOT / "audit_inputs" / "post824_flood_agency_esf" / "accepted"
    / "candidate_predictions.csv"
)
OUT = ROOT / "audit_outputs" / "nonbio_824_flood_agency_esf_rescue"
OUT.mkdir(parents=True, exist_ok=True)

HAZARD = "Flood"
BASE_PRED = "candidate_pred"
LOWER_BAND = "100K-1M"
UPPER_BAND = "1M-50M"


def numeric_presence(frame: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(frame[col], errors="coerce").fillna(0).to_numpy() > 0


def main() -> None:
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

    agency_cols = sorted(
        c for c in df.columns
        if c.startswith("sem_agency_") and c.endswith("_count")
    )
    esf_cols = sorted(
        c for c in df.columns
        if c.startswith("sem_esf_") and c.endswith("_count")
    )
    if not agency_cols or not esf_cols:
        raise AssertionError("Expected semantic agency and ESF count features")

    pred = dict(zip(df["disasterNumber"].astype(int), df[BASE_PRED].astype(str)))
    changed = []
    fold_rows = []
    whitelist_rows = []

    for outer_fy in sorted(df["fyDeclared"].astype(int).unique()):
        train = df[
            (df["fyDeclared"].astype(int) != outer_fy)
            & (df["incidentType"] == HAZARD)
            & (df["target_clean"] < 50_000_000)
        ].copy()
        test = df[
            (df["fyDeclared"].astype(int) == outer_fy)
            & (df["incidentType"] == HAZARD)
            & (df[BASE_PRED] == LOWER_BAND)
        ].copy()

        train_high = train[
            (train["target_clean"] >= 1_000_000)
            & (train["target_clean"] < 50_000_000)
        ].copy()
        train_low = train[train["target_clean"] < 1_000_000].copy()

        whitelist = []
        for col in agency_cols:
            high_hit = numeric_presence(train_high, col)
            low_hit = numeric_presence(train_low, col)
            high_n = int(high_hit.sum())
            low_n = int(low_hit.sum())
            high_years = int(
                train_high.loc[high_hit, "fyDeclared"].astype(int).nunique()
            )
            if low_n == 0 and high_n >= 2 and high_years >= 2:
                whitelist.append(col)
                whitelist_rows.append({
                    "outer_fy": int(outer_fy),
                    "agency_feature": col,
                    "training_high_n": high_n,
                    "training_high_years": high_years,
                    "training_sub1m_n": low_n,
                })

        specialist_hit = np.zeros(len(test), dtype=bool)
        esfs_hit = np.zeros(len(test), dtype=bool)
        if whitelist and len(test):
            for col in whitelist:
                specialist_hit |= numeric_presence(test, col)
            for col in esf_cols:
                esfs_hit |= numeric_presence(test, col)

        promote = specialist_hit & esfs_hit
        promotions = []
        for (_, row), hit in zip(test.iterrows(), promote):
            if not bool(hit):
                continue
            dn = int(row["disasterNumber"])
            old = pred[dn]
            pred[dn] = UPPER_BAND

            matched_agencies = [
                c for c in whitelist
                if float(pd.to_numeric(pd.Series([row[c]]), errors="coerce").fillna(0).iloc[0]) > 0
            ]
            matched_esfs = [
                c for c in esf_cols
                if float(pd.to_numeric(pd.Series([row[c]]), errors="coerce").fillna(0).iloc[0]) > 0
            ]
            rec = {
                "disasterNumber": dn,
                "state": row["state"],
                "fyDeclared": int(row["fyDeclared"]),
                "actual_band": row["actual_band"],
                "target_clean": float(row["target_clean"]),
                "old_pred": old,
                "new_pred": UPPER_BAND,
                "matched_training_safe_agencies": ";".join(matched_agencies),
                "matched_esfs": ";".join(matched_esfs),
                "outer_whitelist_size": int(len(whitelist)),
            }
            changed.append(rec)
            promotions.append(dn)

        fold_rows.append({
            "outer_fy": int(outer_fy),
            "training_flood_n": int(len(train)),
            "training_sub1m_n": int(len(train_low)),
            "training_1m_50m_n": int(len(train_high)),
            "eligible_test_n": int(len(test)),
            "whitelist_size": int(len(whitelist)),
            "whitelist": ";".join(whitelist),
            "specialist_agency_hits": int(specialist_hit.sum()),
            "esf_hits": int(esfs_hit.sum()),
            "promotions": int(len(promotions)),
            "promoted_disaster_numbers": ";".join(str(x) for x in promotions),
        })

    out = df[[
        "disasterNumber", "state", "incidentType", "fyDeclared", "target_clean",
        "actual_band", BASE_PRED,
    ]].copy()
    out["candidate_pred"] = out["disasterNumber"].map(pred)
    out.to_csv(OUT / "candidate_predictions.csv", index=False)
    pd.DataFrame(changed).to_csv(OUT / "changed_rows.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "fold_diagnostics.csv", index=False)
    pd.DataFrame(whitelist_rows).to_csv(OUT / "agency_whitelists.csv", index=False)

    candidate_correct = int((out["candidate_pred"] == out["actual_band"]).sum())
    metrics = six_band_metrics(out, "candidate_pred")

    protected_mask = out["target_clean"] >= 50_000_000
    protected_changes = int((
        out.loc[protected_mask, BASE_PRED]
        != out.loc[protected_mask, "candidate_pred"]
    ).sum())
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
            "Strict outer LFYO. In each outer fold, build a Flood specialist-agency whitelist "
            "from outer-training rows only. Agency presence must occur in >=2 true $1M-$50M "
            "Floods across >=2 fiscal years and in zero sub-$1M Floods. Promote only held-out "
            "Floods already predicted $100K-$1M that contain a whitelisted agency AND at least "
            "one ESF-coded mission assignment."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Post-824 Flood specialist-agency + ESF rescue",
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
            f"${r['target_clean']:,.2f} | actual {r['actual_band']} | "
            f"{r['old_pred']} -> {r['new_pred']} | "
            f"agencies={r['matched_training_safe_agencies']} | ESFs={r['matched_esfs']}"
        )
    md += ["", "## Six-band recall"]
    for band, vals in metrics["per_band"].items():
        md.append(
            f"- {band}: **{vals['correct']}/{vals['total']} = {vals['recall']:.1%}**"
        )
    md.append(f"- Macro recall: **{metrics['macro_recall']:.3f}**")
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

    if protected_changes != 0:
        raise AssertionError("Candidate changed protected >=$50M predictions")


if __name__ == "__main__":
    main()
