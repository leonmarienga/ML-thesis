#!/usr/bin/env python3
"""
Post-824 Severe Storm high-confidence <$100K veto.

The accepted 824 router is frozen. This component is allowed to change only
rows that:
  * are Severe Storm,
  * are currently predicted $100K-$1M, and
  * are below a Stage-1 probability floor calibrated entirely inside the
    outer-training years.

Stage 1 is the previously validated semantics Random Forest classifier for
<$100K vs >=$100K. For each outer FY, its veto floor is the minimum INNER-LFYO
probability among true >=$100K Severe Storms in the outer-training data. Thus,
the veto region has 100% inner-LFYO recall for >=$100K Severe Storms. The rule
is enabled only when at least one inner-held-out true <$100K Severe Storm lies
strictly below that floor, proving that the safe region is useful in training.

No >=$50M prediction can be changed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mission_semantic_audit import (
    CURRENT_19,
    build_semantic_rollup,
    fetch_all_mission_assignments,
    normalize_master,
)
from nonbio_all_ranges import funding_band, six_band_metrics, valid_cols
from nonbio_low_thresholds import fit_binary, positive_proba, inner_binary_oof

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
ACCEPTED = (
    ROOT / "audit_inputs" / "post824_storm_stage1_veto" / "accepted"
    / "candidate_predictions.csv"
)
OUT = ROOT / "audit_outputs" / "nonbio_824_storm_stage1_safe_veto"
OUT.mkdir(parents=True, exist_ok=True)

BASE_PRED = "candidate_pred"
EPS = 1e-12


def main():
    master = normalize_master(pd.read_excel(MASTER))
    master["disasterNumber"] = master["disasterNumber"].astype(int)
    master["target_clean"] = pd.to_numeric(
        master["totalObligatedFunding"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    master["actual_band"] = master["target_clean"].map(funding_band)

    ma = fetch_all_mission_assignments()
    sem, _ = build_semantic_rollup(master, ma)
    model = master.merge(sem, on="disasterNumber", how="left")
    model = model[model["incidentType"] != "Biological"].copy().reset_index(drop=True)

    current = valid_cols(model, CURRENT_19)
    semcols = valid_cols(
        model,
        [
            c for c in model.columns
            if (c.startswith("sem_") or c.startswith("ma_"))
            and not any(
                bad in c.lower()
                for bad in ["oblig", "fund", "cost", "amount", "dollar"]
            )
        ],
    )
    semfeat = list(dict.fromkeys(current + semcols))

    accepted = pd.read_csv(ACCEPTED)
    accepted["disasterNumber"] = accepted["disasterNumber"].astype(int)
    if len(accepted) != 912:
        raise AssertionError(f"Expected 912 accepted rows, got {len(accepted)}")
    for col in [BASE_PRED, "actual_band"]:
        if col not in accepted.columns:
            raise RuntimeError(f"Accepted 824 artifact missing {col}")
    baseline_correct = int((accepted[BASE_PRED] == accepted["actual_band"]).sum())
    if baseline_correct != 824:
        raise AssertionError(f"Expected accepted score 824, got {baseline_correct}")

    model = model.merge(
        accepted[["disasterNumber", BASE_PRED]],
        on="disasterNumber", how="inner", validate="one_to_one",
    )
    if len(model) != 912:
        raise AssertionError(f"Expected 912 merged rows, got {len(model)}")

    pred = {int(r.disasterNumber): str(getattr(r, BASE_PRED)) for r in model.itertuples(index=False)}
    changed = []
    folds = []

    for outer_fy in sorted(model["fyDeclared"].astype(int).unique()):
        train = model[model["fyDeclared"].astype(int) != outer_fy].copy()
        test = model[model["fyDeclared"].astype(int) == outer_fy].copy()

        # Inner LFYO probabilities on the outer-training set only.
        oof = inner_binary_oof(
            train, semfeat, stage=1, kind="rf", seedbase=110000
        )
        storm_truth = train[["disasterNumber", "incidentType", "target_clean"]].copy()
        oof = oof.merge(storm_truth, on="disasterNumber", how="left", validate="one_to_one")
        ss = oof[
            (oof["incidentType"] == "Severe Storm")
            & (oof["target_clean"] < 50_000_000)
        ].copy()
        pos = ss[ss["target_clean"] >= 100_000]
        neg = ss[ss["target_clean"] < 100_000]

        enabled = False
        veto_floor = None
        inner_safe_negatives = 0
        if not pos.empty and not neg.empty:
            veto_floor = float(pos["p"].min())
            inner_safe_negatives = int((neg["p"] < veto_floor - EPS).sum())
            enabled = inner_safe_negatives > 0

        low_train = train[train["target_clean"] < 50_000_000].copy()
        y1 = (low_train["target_clean"] >= 100_000).astype(int)
        s1 = fit_binary(
            low_train, semfeat, y1, "rf", 130000 + int(outer_fy)
        )

        eligible = test[
            (test["incidentType"] == "Severe Storm")
            & (test[BASE_PRED] == "100K-1M")
        ].copy()
        probs = positive_proba(s1, eligible, semfeat) if not eligible.empty else np.array([])

        veto_count = 0
        if enabled and veto_floor is not None:
            for (_, r), p in zip(eligible.iterrows(), probs):
                if float(p) < veto_floor - EPS:
                    dn = int(r["disasterNumber"])
                    old = pred[dn]
                    pred[dn] = "0-100K"
                    veto_count += 1
                    changed.append({
                        "disasterNumber": dn,
                        "state": r["state"],
                        "fyDeclared": int(r["fyDeclared"]),
                        "actual_band": r["actual_band"],
                        "old_pred": old,
                        "new_pred": "0-100K",
                        "stage1_probability_ge100k": float(p),
                        "inner_safe_veto_floor": float(veto_floor),
                    })

        folds.append({
            "outer_fy": int(outer_fy),
            "enabled": bool(enabled),
            "inner_storm_positive_n": int(len(pos)),
            "inner_storm_negative_n": int(len(neg)),
            "inner_safe_veto_floor": veto_floor,
            "inner_safe_negative_n": int(inner_safe_negatives),
            "eligible_test_n": int(len(eligible)),
            "veto_count": int(veto_count),
        })

    out = model[[
        "disasterNumber", "state", "incidentType", "fyDeclared",
        "target_clean", "actual_band", BASE_PRED,
    ]].copy()
    out["veto_candidate_pred"] = out["disasterNumber"].map(pred)
    out.to_csv(OUT / "candidate_predictions.csv", index=False)
    pd.DataFrame(changed).to_csv(OUT / "changed_rows.csv", index=False)
    pd.DataFrame(folds).to_csv(OUT / "fold_diagnostics.csv", index=False)

    candidate_correct = int((out["veto_candidate_pred"] == out["actual_band"]).sum())
    metrics = six_band_metrics(out, "veto_candidate_pred")
    protected_changes = int((
        (out["target_clean"] >= 50_000_000)
        & (out[BASE_PRED] != out["veto_candidate_pred"])
    ).sum())
    correct_changes = int(sum(r["actual_band"] == "0-100K" for r in changed))
    wrong_changes = int(len(changed) - correct_changes)

    summary = {
        "baseline_correct": baseline_correct,
        "candidate_correct": candidate_correct,
        "total": int(len(out)),
        "changed_rows": changed,
        "correct_changes": correct_changes,
        "wrong_changes": wrong_changes,
        "protected_ge50m_changes": protected_changes,
        "metrics": metrics,
        "protocol": (
            "Severe Storm only. Stage-1 semantics RF. Per outer FY, veto floor is "
            "minimum inner-LFYO probability among true >=$100K Severe Storms in "
            "outer training; enabled only if inner validation contains at least one "
            "true <$100K Storm below the safe floor. Applied only to accepted "
            "$100K-$1M predictions."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Post-824 Severe Storm Stage-1 safe veto",
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
            f"actual {r['actual_band']} | {r['old_pred']} -> {r['new_pred']} | "
            f"p(>=100K)={r['stage1_probability_ge100k']:.3f}, "
            f"safe floor={r['inner_safe_veto_floor']:.3f}"
        )
    md += ["", "## Six-band recall"]
    for b, m in metrics["per_band"].items():
        md.append(f"- {b}: **{m['correct']}/{m['total']} = {m['recall']:.1%}**")
    md.append(f"- Macro recall: **{metrics['macro_recall']:.3f}**")
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

    if protected_changes != 0:
        raise AssertionError("Veto changed protected >=$50M prediction")


if __name__ == "__main__":
    main()
