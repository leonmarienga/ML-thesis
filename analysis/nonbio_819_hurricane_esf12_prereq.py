#!/usr/bin/env python3
"""
Strict post-819 Hurricane ESF-12 prerequisite audit.

Candidate
---------
For a Hurricane currently entering the >=$50M router, require at least one
ESF-12 Mission Assignment. This is a one-way veto only; it never promotes a
low prediction.

For each outer leave-fiscal-year-out fold the prerequisite is enabled only if:
- every true >=$50M Hurricane in outer training has ESF-12 count > 0;
- outer training contains at least five sub-$50M Hurricanes with ESF-12 count 0;
- an inner-LFYO safety check over outer training preserves 100% of held-out
  true high Hurricanes wherever an inner high case exists.

Vetoed rows re-enter the established outer-training-only low router.
Biological remains excluded. No funding-derived predictor is used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

from mission_semantic_audit import (
    CURRENT_19, build_semantic_rollup, fetch_all_mission_assignments,
    normalize_master,
)
from nonbio_all_ranges import funding_band, six_band_metrics, valid_cols
from nonbio_hurricane_boundary_confirm import build_low_predictor, data_hash

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
ACCEPTED = ROOT / "audit_inputs" / "post819_hurr_esf12" / "accepted" / "candidate_predictions.csv"
OUT = ROOT / "audit_outputs" / "nonbio_819_hurricane_esf12_prereq"
OUT.mkdir(parents=True, exist_ok=True)

HIGH_CUTOFF = 50_000_000.0
BASE_ROOT_COL = "candidate_root_pred_high"
BASE_PRED_COL = "candidate_pred"


def esf12_present(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["sem_esf_12_count"], errors="coerce").fillna(0.0) > 0


def gate(train: pd.DataFrame):
    h = train[train["incidentType"] == "Hurricane"].copy()
    hi = h[h["target_clean"] >= HIGH_CUTOFF].copy()
    lo = h[h["target_clean"] < HIGH_CUTOFF].copy()
    high_all_present = bool(len(hi) > 0 and esf12_present(hi).all())
    low_absent_n = int((~esf12_present(lo)).sum())

    inner_checks = []
    for inner_fy in sorted(h["fyDeclared"].astype(int).unique()):
        inner_train = h[h["fyDeclared"].astype(int) != inner_fy].copy()
        inner_test = h[h["fyDeclared"].astype(int) == inner_fy].copy()
        inner_hi_train = inner_train[inner_train["target_clean"] >= HIGH_CUTOFF]
        held_hi = inner_test[inner_test["target_clean"] >= HIGH_CUTOFF]
        if held_hi.empty or inner_hi_train.empty:
            continue
        train_safe = bool(esf12_present(inner_hi_train).all())
        kept = esf12_present(held_hi)
        for (_, r), k in zip(held_hi.iterrows(), kept.to_numpy(bool)):
            inner_checks.append({
                "inner_fy": int(inner_fy),
                "disasterNumber": int(r["disasterNumber"]),
                "training_prereq_safe": train_safe,
                "heldout_kept": bool(k),
            })

    inner_recall = (
        float(np.mean([x["heldout_kept"] for x in inner_checks]))
        if inner_checks else None
    )
    inner_training_safe = bool(
        inner_checks and all(x["training_prereq_safe"] for x in inner_checks)
    )
    enabled = bool(
        high_all_present
        and low_absent_n >= 5
        and inner_training_safe
        and inner_recall is not None
        and inner_recall >= 1.0 - 1e-12
    )
    return enabled, {
        "hurricane_train_n": int(len(h)),
        "hurricane_train_high_n": int(len(hi)),
        "hurricane_train_low_n": int(len(lo)),
        "training_high_all_esf12_present": high_all_present,
        "training_low_esf12_absent_n": low_absent_n,
        "inner_high_n": int(len(inner_checks)),
        "inner_high_recall": inner_recall,
        "inner_training_safe": inner_training_safe,
    }


def main():
    master = normalize_master(pd.read_excel(MASTER))
    master["target_clean"] = pd.to_numeric(master["totalObligatedFunding"], errors="coerce").fillna(0).clip(lower=0)
    master["actual_band"] = master["target_clean"].map(funding_band)
    master["disasterNumber"] = master["disasterNumber"].astype(int)

    ma = fetch_all_mission_assignments()
    ma_hash = data_hash(ma)
    sem, _ = build_semantic_rollup(master, ma)
    df = master.merge(sem, on="disasterNumber", how="left")
    df = df[df["incidentType"] != "Biological"].copy().reset_index(drop=True)
    df["sem_esf_12_count"] = pd.to_numeric(df["sem_esf_12_count"], errors="coerce").fillna(0.0)

    current = valid_cols(df, CURRENT_19)
    semcols = valid_cols(df, [
        c for c in df.columns
        if (c.startswith("sem_") or c.startswith("ma_"))
        and not any(b in c.lower() for b in ["oblig", "fund", "cost", "amount", "dollar"])
    ])
    semfeat = list(dict.fromkeys(current + semcols))

    accepted = pd.read_csv(ACCEPTED)
    accepted["disasterNumber"] = accepted["disasterNumber"].astype(int)
    if len(accepted) != 912:
        raise AssertionError(f"Expected 912 accepted rows, got {len(accepted)}")
    baseline_correct = int((accepted[BASE_PRED_COL] == accepted["actual_band"]).sum())
    if baseline_correct != 819:
        raise AssertionError(f"Expected accepted score 819, got {baseline_correct}")

    model = df.merge(
        accepted[["disasterNumber", BASE_ROOT_COL, BASE_PRED_COL]],
        on="disasterNumber", how="inner", validate="one_to_one"
    )
    if len(model) != 912:
        raise AssertionError(f"Expected 912 merged rows, got {len(model)}")
    model["root_actual_high"] = (model["target_clean"] >= HIGH_CUTOFF).astype(int)

    cand_root = {int(r.disasterNumber): int(getattr(r, BASE_ROOT_COL)) for r in model.itertuples(index=False)}
    cand_pred = {int(r.disasterNumber): str(getattr(r, BASE_PRED_COL)) for r in model.itertuples(index=False)}
    changed = []
    diagnostics = []

    for fy in sorted(model["fyDeclared"].astype(int).unique()):
        train = model[model["fyDeclared"].astype(int) != fy].copy()
        test = model[model["fyDeclared"].astype(int) == fy].copy()
        enabled, diag = gate(train)

        eligible = test[
            (test["incidentType"] == "Hurricane")
            & (test[BASE_ROOT_COL].astype(int) == 1)
        ].copy()
        veto = eligible[enabled & (~esf12_present(eligible))].copy() if not eligible.empty else eligible.copy()

        if not veto.empty:
            low_predict = build_low_predictor(train, semfeat, int(fy))
            low_pred = low_predict(veto)
            for (_, r), lp in zip(veto.iterrows(), low_pred):
                dn = int(r["disasterNumber"])
                old = cand_pred[dn]
                cand_root[dn] = 0
                cand_pred[dn] = str(lp)
                changed.append({
                    "disasterNumber": dn,
                    "state": r["state"],
                    "fyDeclared": int(r["fyDeclared"]),
                    "actual_band": r["actual_band"],
                    "old_pred": old,
                    "new_pred": str(lp),
                    "sem_esf_12_count": float(r["sem_esf_12_count"]),
                    "gate_enabled": bool(enabled),
                })

        diagnostics.append({
            "outer_fy": int(fy), "gate_enabled": bool(enabled), **diag,
            "eligible_root_high_hurricane": int(len(eligible)),
            "veto_count": int(len(veto)),
        })

    out = model[[
        "disasterNumber", "state", "incidentType", "fyDeclared", "actual_band",
        "target_clean", "root_actual_high", BASE_ROOT_COL, BASE_PRED_COL,
        "sem_esf_12_count"
    ]].copy()
    out["baseline_root_pred_high"] = out[BASE_ROOT_COL].astype(int)
    out["baseline_pred"] = out[BASE_PRED_COL].astype(str)
    out["new_root_pred_high"] = out["disasterNumber"].map(cand_root).astype(int)
    out["new_pred"] = out["disasterNumber"].map(cand_pred).astype(str)
    out.to_csv(OUT / "candidate_predictions.csv", index=False)
    pd.DataFrame(changed).to_csv(OUT / "changed_rows.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUT / "fold_diagnostics.csv", index=False)

    y = out["root_actual_high"].to_numpy(int)
    br = out["baseline_root_pred_high"].to_numpy(int)
    nr = out["new_root_pred_high"].to_numpy(int)
    base_fn = set(out.loc[(y == 1) & (br == 0), "disasterNumber"].astype(int))
    new_fn = set(out.loc[(y == 1) & (nr == 0), "disasterNumber"].astype(int))
    base_fp = int(((y == 0) & (br == 1)).sum())
    new_fp = int(((y == 0) & (nr == 1)).sum())
    base_recall = float(recall_score(y, br, zero_division=0))
    new_recall = float(recall_score(y, nr, zero_division=0))
    new_correct = int((out["new_pred"] == out["actual_band"]).sum())
    metrics = six_band_metrics(out.rename(columns={"new_pred":"candidate_for_metrics"}), "candidate_for_metrics")

    if base_fn:
        raise AssertionError(f"Accepted 819 should have no high FNs, got {sorted(base_fn)}")

    summary = {
        "mission_nonfinancial_sha256": ma_hash,
        "rule": "One-way Hurricane root-high veto if sem_esf_12_count == 0, training-gated with inner LFYO safety.",
        "baseline": {"overall_correct": baseline_correct, "root_fp": base_fp, "root_high_recall": base_recall, "high_false_negatives": sorted(base_fn)},
        "candidate": {"overall_correct": new_correct, "root_fp": new_fp, "root_high_recall": new_recall, "high_false_negatives": sorted(new_fn), "end_to_end": metrics},
        "changed_rows": changed,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Post-819 Hurricane ESF-12 prerequisite confirmation", "",
        f"- Baseline: **{baseline_correct}/912**; root FP **{base_fp}**; high recall **{base_recall:.1%}**; FN **{sorted(base_fn)}**.",
        f"- Candidate: **{new_correct}/912**; root FP **{new_fp}**; high recall **{new_recall:.1%}**; FN **{sorted(new_fn)}**.",
        "", "## Changed rows", "",
        "| FEMA | State | Actual | Old | New | ESF-12 count |",
        "|---:|---|---|---|---|---:|",
    ]
    for r in changed:
        md.append(f"| {r['disasterNumber']} | {r['state']} | {r['actual_band']} | {r['old_pred']} | {r['new_pred']} | {r['sem_esf_12_count']:.0f} |")
    md += ["", f"- Macro recall: **{metrics['macro_recall']:.3f}**"]
    for b,m in metrics["per_band"].items():
        md.append(f"- {b}: **{m['correct']}/{m['total']} = {m['recall']:.1%}**")
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

if __name__ == "__main__":
    main()
