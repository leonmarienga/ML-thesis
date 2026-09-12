#!/usr/bin/env python3
"""
Strict post-817 Fire two-key boundary audit.

Starting point
--------------
The accepted non-Biological router is 817/912 after the rare-hazard
operational-floor component.

Candidate Fire mechanism
------------------------
A Fire is considered to have the two-key high-value response signature when:
1. initial_usace_esf3_dfa_count > 0
2. sem_priority_life_sustaining_count > 0

The rule is never enabled from held-out-year labels. For each outer LFYO fold,
it is enabled only when Fire cases in the OTHER fiscal years contain both
classes and the fixed two-key signature perfectly separates >=$50M from <$50M
within that outer-training set.

When enabled:
- accepted root-high Fire + signature absent -> veto to the established low router
- accepted root-low Fire + signature present -> conservative rescue to $50M-$200M

No Biological cases are included. No funding-derived predictor is used.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

from mission_semantic_audit import (
    CURRENT_19,
    build_semantic_rollup,
    fetch_all_mission_assignments,
    normalize_master,
)
from nonbio_all_ranges import funding_band, six_band_metrics, valid_cols
from nonbio_hurricane_boundary_confirm import build_low_predictor, data_hash
from nonbio_hazard_hierarchy import initial_mechanism_counts

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
ACCEPTED = (
    ROOT
    / "audit_inputs"
    / "post817_fire_two_key"
    / "accepted"
    / "candidate_predictions.csv"
)
OUT = ROOT / "audit_outputs" / "nonbio_817_fire_two_key_boundary"
OUT.mkdir(parents=True, exist_ok=True)

HIGH_CUTOFF = 50_000_000.0
BASE_PRED_COL = "candidate_pred"
BASE_ROOT_COL = "candidate_root_pred_high"


def fire_signature(df: pd.DataFrame) -> pd.Series:
    usace = pd.to_numeric(
        df["initial_usace_esf3_dfa_count"], errors="coerce"
    ).fillna(0.0)
    life = pd.to_numeric(
        df["sem_priority_life_sustaining_count"], errors="coerce"
    ).fillna(0.0)
    return (usace > 0) & (life > 0)


def training_gate(train: pd.DataFrame):
    fires = train[train["incidentType"] == "Fire"].copy()
    if fires.empty:
        return False, {
            "fire_train_n": 0,
            "fire_train_high_n": 0,
            "fire_train_low_n": 0,
            "train_accuracy": None,
            "train_high_recall": None,
            "train_low_specificity": None,
        }

    y = (fires["target_clean"] >= HIGH_CUTOFF).to_numpy(int)
    sig = fire_signature(fires).to_numpy(int)
    high_n = int(y.sum())
    low_n = int((y == 0).sum())

    if high_n == 0 or low_n == 0:
        return False, {
            "fire_train_n": int(len(fires)),
            "fire_train_high_n": high_n,
            "fire_train_low_n": low_n,
            "train_accuracy": None,
            "train_high_recall": None,
            "train_low_specificity": None,
        }

    acc = float(np.mean(sig == y))
    high_recall = float(np.mean(sig[y == 1] == 1))
    low_specificity = float(np.mean(sig[y == 0] == 0))
    enabled = (
        acc >= 1.0 - 1e-12
        and high_recall >= 1.0 - 1e-12
        and low_specificity >= 1.0 - 1e-12
    )
    return bool(enabled), {
        "fire_train_n": int(len(fires)),
        "fire_train_high_n": high_n,
        "fire_train_low_n": low_n,
        "train_accuracy": acc,
        "train_high_recall": high_recall,
        "train_low_specificity": low_specificity,
    }


def main():
    master = normalize_master(pd.read_excel(MASTER))
    master["target_clean"] = pd.to_numeric(
        master["totalObligatedFunding"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    master["actual_band"] = master["target_clean"].map(funding_band)
    master["disasterNumber"] = master["disasterNumber"].astype(int)

    ma = fetch_all_mission_assignments()
    ma_hash = data_hash(ma)
    sem, _ = build_semantic_rollup(master, ma)
    mech = initial_mechanism_counts(master, ma)

    df = (
        master.merge(sem, on="disasterNumber", how="left")
        .merge(mech, on="disasterNumber", how="left")
    )
    df = df[df["incidentType"] != "Biological"].copy().reset_index(drop=True)
    df["initial_usace_esf3_dfa_count"] = pd.to_numeric(
        df["initial_usace_esf3_dfa_count"], errors="coerce"
    ).fillna(0.0)
    df["sem_priority_life_sustaining_count"] = pd.to_numeric(
        df["sem_priority_life_sustaining_count"], errors="coerce"
    ).fillna(0.0)

    current = valid_cols(df, CURRENT_19)
    semcols = valid_cols(
        df,
        [
            c for c in df.columns
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
    for col in ["actual_band", "root_actual_high", BASE_ROOT_COL, BASE_PRED_COL]:
        if col not in accepted.columns:
            raise RuntimeError(f"Accepted 817 artifact missing {col}")

    baseline_correct = int((accepted[BASE_PRED_COL] == accepted["actual_band"]).sum())
    if baseline_correct != 817:
        raise AssertionError(f"Expected accepted score 817, got {baseline_correct}")

    model = df.merge(
        accepted[["disasterNumber", BASE_ROOT_COL, BASE_PRED_COL]],
        on="disasterNumber",
        how="inner",
        validate="one_to_one",
    )
    if len(model) != 912:
        raise AssertionError(f"Expected 912 merged rows, got {len(model)}")
    model["root_actual_high"] = (model["target_clean"] >= HIGH_CUTOFF).astype(int)

    candidate_pred = {
        int(r.disasterNumber): str(getattr(r, BASE_PRED_COL))
        for r in model.itertuples(index=False)
    }
    candidate_root = {
        int(r.disasterNumber): int(getattr(r, BASE_ROOT_COL))
        for r in model.itertuples(index=False)
    }

    diagnostics = []
    changed_rows = []

    for fy in sorted(model["fyDeclared"].astype(int).unique()):
        train = model[model["fyDeclared"].astype(int) != fy].copy()
        test = model[model["fyDeclared"].astype(int) == fy].copy()

        enabled, gate_diag = training_gate(train)
        fire_test = test[test["incidentType"] == "Fire"].copy()
        sig = fire_signature(fire_test) if not fire_test.empty else pd.Series(dtype=bool)

        low_predict = None
        veto_rows = fire_test[
            enabled
            & (fire_test[BASE_ROOT_COL].astype(int) == 1)
            & (~sig)
        ].copy() if not fire_test.empty else fire_test.copy()

        if not veto_rows.empty:
            low_predict = build_low_predictor(train, semfeat, int(fy))
            low_pred = low_predict(veto_rows)
            for (_, row), lp in zip(veto_rows.iterrows(), low_pred):
                dn = int(row["disasterNumber"])
                old_pred = candidate_pred[dn]
                candidate_root[dn] = 0
                candidate_pred[dn] = str(lp)
                changed_rows.append({
                    "disasterNumber": dn,
                    "state": row["state"],
                    "incidentType": row["incidentType"],
                    "fyDeclared": int(row["fyDeclared"]),
                    "actual_band": row["actual_band"],
                    "old_root": int(row[BASE_ROOT_COL]),
                    "new_root": 0,
                    "old_pred": old_pred,
                    "new_pred": str(lp),
                    "action": "fire_two_key_veto",
                    "life_count": float(row["sem_priority_life_sustaining_count"]),
                    "usace_esf3_dfa_count": float(row["initial_usace_esf3_dfa_count"]),
                    "gate_enabled": bool(enabled),
                })

        rescue_rows = fire_test[
            enabled
            & (fire_test[BASE_ROOT_COL].astype(int) == 0)
            & sig
        ].copy() if not fire_test.empty else fire_test.copy()

        for _, row in rescue_rows.iterrows():
            dn = int(row["disasterNumber"])
            old_pred = candidate_pred[dn]
            candidate_root[dn] = 1
            candidate_pred[dn] = "50-200M"
            changed_rows.append({
                "disasterNumber": dn,
                "state": row["state"],
                "incidentType": row["incidentType"],
                "fyDeclared": int(row["fyDeclared"]),
                "actual_band": row["actual_band"],
                "old_root": int(row[BASE_ROOT_COL]),
                "new_root": 1,
                "old_pred": old_pred,
                "new_pred": "50-200M",
                "action": "fire_two_key_rescue",
                "life_count": float(row["sem_priority_life_sustaining_count"]),
                "usace_esf3_dfa_count": float(row["initial_usace_esf3_dfa_count"]),
                "gate_enabled": bool(enabled),
            })

        diagnostics.append({
            "outer_fy": int(fy),
            "gate_enabled": bool(enabled),
            **gate_diag,
            "fire_test_n": int(len(fire_test)),
            "fire_test_signature_n": int(sig.sum()) if len(sig) else 0,
            "veto_count": int(len(veto_rows)),
            "rescue_count": int(len(rescue_rows)),
        })

    out = model[[
        "disasterNumber", "state", "incidentType", "fyDeclared",
        "actual_band", "target_clean", "root_actual_high",
        BASE_ROOT_COL, BASE_PRED_COL,
        "sem_priority_life_sustaining_count",
        "initial_usace_esf3_dfa_count",
    ]].copy()
    out["candidate_root_pred_high"] = out["disasterNumber"].map(candidate_root).astype(int)
    out["candidate_pred"] = out["disasterNumber"].map(candidate_pred).astype(str)
    out.to_csv(OUT / "candidate_predictions.csv", index=False)

    changed = pd.DataFrame(changed_rows)
    changed.to_csv(OUT / "changed_rows.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUT / "fold_diagnostics.csv", index=False)

    y = out["root_actual_high"].to_numpy(int)
    base_root = out[BASE_ROOT_COL].to_numpy(int)
    cand_root = out["candidate_root_pred_high"].to_numpy(int)

    baseline_fn = set(out.loc[(y == 1) & (base_root == 0), "disasterNumber"].astype(int))
    candidate_fn = set(out.loc[(y == 1) & (cand_root == 0), "disasterNumber"].astype(int))
    baseline_fp = int(((y == 0) & (base_root == 1)).sum())
    candidate_fp = int(((y == 0) & (cand_root == 1)).sum())
    baseline_recall = float(recall_score(y, base_root, pos_label=1, zero_division=0))
    candidate_recall = float(recall_score(y, cand_root, pos_label=1, zero_division=0))

    candidate_correct = int((out["candidate_pred"] == out["actual_band"]).sum())
    candidate_metrics = six_band_metrics(out, "candidate_pred")

    if baseline_fn != {4353}:
        raise AssertionError(f"Unexpected accepted 817 high FN set: {sorted(baseline_fn)}")

    fire_all = out[out["incidentType"] == "Fire"].copy()
    fire_all["signature"] = fire_signature(fire_all).astype(int)
    fire_signature_table = (
        fire_all.groupby(["root_actual_high", "signature"], dropna=False)
        .size().reset_index(name="n")
    )
    fire_signature_table.to_csv(OUT / "fire_signature_table.csv", index=False)

    summary = {
        "mission_nonfinancial_sha256": ma_hash,
        "rule": (
            "Fire signature = initial_usace_esf3_dfa_count>0 AND "
            "sem_priority_life_sustaining_count>0; enabled per outer FY only "
            "when it perfectly separates high/low Fire cases in outer training."
        ),
        "baseline": {
            "overall_correct": baseline_correct,
            "overall_total": int(len(out)),
            "root_fp": baseline_fp,
            "root_high_recall": baseline_recall,
            "high_false_negatives": sorted(baseline_fn),
        },
        "candidate": {
            "overall_correct": candidate_correct,
            "overall_total": int(len(out)),
            "root_fp": candidate_fp,
            "root_high_recall": candidate_recall,
            "high_false_negatives": sorted(candidate_fn),
            "end_to_end": candidate_metrics,
        },
        "changed_rows": changed_rows,
        "fire_signature_table": fire_signature_table.to_dict(orient="records"),
        "acceptance_rule": (
            "Accept only if end-to-end six-band correctness improves over 817, "
            "no previously correct >=$50M case becomes wrong, and root high recall "
            "does not decrease."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Post-817 Fire two-key boundary confirmation",
        "",
        f"- Baseline: **{baseline_correct}/{len(out)}**; root FP **{baseline_fp}**; high recall **{baseline_recall:.1%}**; high FN **{sorted(baseline_fn)}**.",
        f"- Candidate: **{candidate_correct}/{len(out)}**; root FP **{candidate_fp}**; high recall **{candidate_recall:.1%}**; high FN **{sorted(candidate_fn)}**.",
        "",
        "## Changed rows",
        "",
        "| FEMA | State | Actual | Old | New | Action | Life | USACE-ESF3-DFA |",
        "|---:|---|---|---|---|---|---:|---:|",
    ]
    for r in changed_rows:
        md.append(
            f"| {r['disasterNumber']} | {r['state']} | {r['actual_band']} | "
            f"{r['old_pred']} | {r['new_pred']} | {r['action']} | "
            f"{r['life_count']:.0f} | {r['usace_esf3_dfa_count']:.0f} |"
        )

    md += [
        "",
        "## Six-band recall",
        "",
        f"- Macro recall: **{candidate_metrics['macro_recall']:.3f}**",
    ]
    for b, m in candidate_metrics["per_band"].items():
        md.append(f"- {b}: **{m['correct']}/{m['total']} = {m['recall']:.1%}**")

    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
