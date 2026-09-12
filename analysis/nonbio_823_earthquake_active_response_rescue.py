#!/usr/bin/env python3
"""
Strict post-823 Earthquake active-response rescue audit.

The >=$50M architecture is frozen. This component can only promote an
Earthquake that the accepted 823 router currently places in $100K-$1M.

Fixed target-blind signature:
    missionAssignmentCount > 0

For each outer fiscal-year fold the rescue is enabled only if, among OTHER-year
sub-$50M Earthquakes, both sides of the $1M boundary are represented and the
fixed signature perfectly separates >=$1M from <$1M Earthquakes. No threshold
is tuned on the held-out year.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mission_semantic_audit import normalize_master
from nonbio_all_ranges import funding_band, six_band_metrics

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
ACCEPTED = (
    ROOT / "audit_inputs" / "post823_earthquake_active" / "accepted"
    / "candidate_predictions.csv"
)
OUT = ROOT / "audit_outputs" / "nonbio_823_earthquake_active_response_rescue"
OUT.mkdir(parents=True, exist_ok=True)

BASE_PRED = "new_pred"
BASE_ROOT = "new_root_pred_high"


def active_response(df: pd.DataFrame) -> pd.Series:
    x = pd.to_numeric(df["missionAssignmentCount"], errors="coerce").fillna(0.0)
    return x > 0


def main():
    master = normalize_master(pd.read_excel(MASTER))
    master["disasterNumber"] = master["disasterNumber"].astype(int)
    master["target_clean"] = pd.to_numeric(
        master["totalObligatedFunding"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    master["actual_band"] = master["target_clean"].map(funding_band)
    model = master[master["incidentType"] != "Biological"].copy().reset_index(drop=True)

    accepted = pd.read_csv(ACCEPTED)
    accepted["disasterNumber"] = accepted["disasterNumber"].astype(int)
    for col in [BASE_PRED, BASE_ROOT, "actual_band"]:
        if col not in accepted.columns:
            raise RuntimeError(f"Accepted 823 artifact missing {col}")
    if len(accepted) != 912:
        raise AssertionError(f"Expected 912 accepted rows, got {len(accepted)}")
    baseline_correct = int((accepted[BASE_PRED] == accepted["actual_band"]).sum())
    if baseline_correct != 823:
        raise AssertionError(f"Expected accepted score 823, got {baseline_correct}")

    model = model.merge(
        accepted[["disasterNumber", BASE_PRED, BASE_ROOT]],
        on="disasterNumber", how="inner", validate="one_to_one",
    )
    if len(model) != 912:
        raise AssertionError(f"Expected 912 merged rows, got {len(model)}")

    pred = {int(r.disasterNumber): str(getattr(r, BASE_PRED)) for r in model.itertuples(index=False)}
    changed = []
    folds = []

    for fy in sorted(model["fyDeclared"].astype(int).unique()):
        train = model[model["fyDeclared"].astype(int) != fy].copy()
        test = model[model["fyDeclared"].astype(int) == fy].copy()

        eq = train[(train["incidentType"] == "Earthquake") & (train["target_clean"] < 50_000_000)].copy()
        y = (eq["target_clean"] >= 1_000_000).astype(int)
        sig = active_response(eq).astype(int)
        high_n = int(y.sum())
        low_n = int((y == 0).sum())
        enabled = bool(
            high_n > 0 and low_n > 0
            and bool((sig[y == 1] == 1).all())
            and bool((sig[y == 0] == 0).all())
        )

        eligible = test[
            (test["incidentType"] == "Earthquake")
            & (test[BASE_PRED] == "100K-1M")
            & active_response(test)
        ].copy()
        if not enabled:
            eligible = eligible.iloc[0:0].copy()

        for _, r in eligible.iterrows():
            dn = int(r["disasterNumber"])
            old = pred[dn]
            pred[dn] = "1M-50M"
            changed.append({
                "disasterNumber": dn,
                "state": r["state"],
                "fyDeclared": int(r["fyDeclared"]),
                "actual_band": r["actual_band"],
                "old_pred": old,
                "new_pred": "1M-50M",
                "missionAssignmentCount": float(pd.to_numeric(r["missionAssignmentCount"], errors="coerce")),
                "outer_gate_enabled": enabled,
            })

        folds.append({
            "outer_fy": int(fy),
            "gate_enabled": enabled,
            "train_earthquake_n": int(len(eq)),
            "train_high_n": high_n,
            "train_low_n": low_n,
            "train_high_signature_recall": float((sig[y == 1] == 1).mean()) if high_n else None,
            "train_low_signature_specificity": float((sig[y == 0] == 0).mean()) if low_n else None,
            "promotions": int(len(eligible)),
        })

    out = model[[
        "disasterNumber", "state", "incidentType", "fyDeclared", "target_clean",
        "actual_band", "missionAssignmentCount", BASE_ROOT, BASE_PRED,
    ]].copy()
    out["candidate_pred"] = out["disasterNumber"].map(pred)
    out.to_csv(OUT / "candidate_predictions.csv", index=False)
    pd.DataFrame(changed).to_csv(OUT / "changed_rows.csv", index=False)
    pd.DataFrame(folds).to_csv(OUT / "fold_diagnostics.csv", index=False)

    candidate_correct = int((out["candidate_pred"] == out["actual_band"]).sum())
    metrics = six_band_metrics(out, "candidate_pred")
    protected = out[out["target_clean"] >= 50_000_000]
    protected_changes = int((protected[BASE_PRED] != protected["candidate_pred"]).sum())
    false_promotions = int(sum(r["actual_band"] != "1M-50M" for r in changed))

    summary = {
        "baseline_correct": baseline_correct,
        "candidate_correct": candidate_correct,
        "total": int(len(out)),
        "changed_rows": changed,
        "protected_ge_50m_changes": protected_changes,
        "false_promotions": false_promotions,
        "metrics": metrics,
        "protocol": (
            "Strict outer LFYO. Fixed Earthquake signature missionAssignmentCount>0; "
            "outer fold enabled only when OTHER-year sub-$50M Earthquakes contain both "
            "classes and perfectly separate at the fixed signature. Rescue is one-way "
            "from accepted $100K-$1M to $1M-$50M."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Post-823 Earthquake active-response rescue",
        "",
        f"- Baseline: **{baseline_correct}/912**",
        f"- Candidate: **{candidate_correct}/912**",
        f"- >=$50M prediction changes: **{protected_changes}**",
        f"- False promotions among changed rows: **{false_promotions}**",
        "",
        "## Changed rows",
    ]
    for r in changed:
        md.append(
            f"- FEMA {r['disasterNumber']} {r['state']} FY{r['fyDeclared']}: "
            f"{r['actual_band']} | {r['old_pred']} -> {r['new_pred']} | "
            f"MissionAssignments={r['missionAssignmentCount']:.0f}"
        )
    md += ["", "## Six-band recall"]
    for b, m in metrics["per_band"].items():
        md.append(f"- {b}: **{m['correct']}/{m['total']} = {m['recall']:.1%}**")
    md.append(f"- Macro recall: **{metrics['macro_recall']:.3f}**")
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

    if candidate_correct <= baseline_correct:
        raise AssertionError("Candidate did not improve 823 baseline")
    if protected_changes != 0:
        raise AssertionError("Candidate changed protected >=$50M predictions")
    if false_promotions != 0:
        raise AssertionError("Candidate produced a false lower-band promotion")


if __name__ == "__main__":
    main()
