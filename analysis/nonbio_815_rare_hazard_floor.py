#!/usr/bin/env python3
"""
Strict post-815 rare-hazard operational-floor audit.

Purpose
-------
Test one conservative, interpretable one-way veto after the accepted 815/912
non-Biological router. The veto applies only to accepted >=$50M predictions for
hazards other than Hurricane and Fire.

For each outer leave-fiscal-year-out (LFYO) fold:
1. Using outer-training data only, define an operational floor as
   0.5 * the minimum missionAssignmentCount among true >=$50M rare-hazard cases.
2. Enable that floor only if an inner LFYO safety audit on the outer-training
   data preserves 100% of true >=$50M rare-hazard cases.
3. Veto only accepted root-high rare-hazard test rows below the enabled floor.
4. Route vetoed rows through the accepted low-band predictor. Flood rows also
   receive the already accepted guard70 Flood $1M rescue.

The 0.5 multiplier is fixed a priori for this confirmation run; it is not tuned
against end-to-end accuracy. No funding-derived field is used as a predictor.
Biological remains excluded/frozen.
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
    normalize_model_frame,
)
from nonbio_all_ranges import funding_band, six_band_metrics, valid_cols
from nonbio_hurricane_boundary_confirm import build_low_predictor, data_hash
from nonbio_flood_1m_rescue import (
    inner_base_and_flood_scores,
    choose_rescue_threshold,
    fit_flood_log,
)

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
ACCEPTED = (
    ROOT
    / "audit_inputs"
    / "post815_rare_floor"
    / "accepted"
    / "integrated_predictions.csv"
)
OUT = ROOT / "audit_outputs" / "nonbio_815_rare_hazard_floor"
OUT.mkdir(parents=True, exist_ok=True)

PRED_COL = "hurricane_integrated_pred"
FLOOR_MULTIPLIER = 0.50
HIGH_CUTOFF = 50_000_000.0


def rare_scope(df: pd.DataFrame) -> pd.Series:
    """Rare-hazard scope used by this veto: non-Hurricane, non-Fire, non-Biological."""
    return ~df["incidentType"].isin(["Hurricane", "Fire", "Biological"])


def mission_values(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["missionAssignmentCount"], errors="coerce").fillna(0.0)


def operational_floor(outer_train: pd.DataFrame):
    """Return outer-training floor plus an inner-LFYO safety decision."""
    scoped = outer_train.loc[rare_scope(outer_train)].copy()
    highs = scoped[scoped["target_clean"] >= HIGH_CUTOFF].copy()

    if highs.empty:
        return 0.0, False, {
            "enabled": False,
            "inner_high_n": 0,
            "inner_high_recall": None,
            "outer_training_high_n": 0,
        }

    floor = FLOOR_MULTIPLIER * float(mission_values(highs).min())

    inner_checks = []
    for inner_fy in sorted(scoped["fyDeclared"].astype(int).unique()):
        inner_train = scoped[scoped["fyDeclared"].astype(int) != inner_fy].copy()
        inner_test = scoped[scoped["fyDeclared"].astype(int) == inner_fy].copy()
        heldout_high = inner_test[inner_test["target_clean"] >= HIGH_CUTOFF].copy()
        train_high = inner_train[inner_train["target_clean"] >= HIGH_CUTOFF].copy()

        if heldout_high.empty or train_high.empty:
            continue

        inner_floor = FLOOR_MULTIPLIER * float(mission_values(train_high).min())
        kept = mission_values(heldout_high) >= inner_floor
        for (_, row), is_kept in zip(heldout_high.iterrows(), kept.to_numpy(bool)):
            inner_checks.append({
                "inner_fy": int(inner_fy),
                "disasterNumber": int(row["disasterNumber"]),
                "missionAssignmentCount": float(
                    pd.to_numeric(row["missionAssignmentCount"], errors="coerce")
                    if pd.notna(row["missionAssignmentCount"]) else 0.0
                ),
                "inner_floor": float(inner_floor),
                "kept": bool(is_kept),
            })

    if not inner_checks:
        return floor, False, {
            "enabled": False,
            "inner_high_n": 0,
            "inner_high_recall": None,
            "outer_training_high_n": int(len(highs)),
        }

    inner_high_recall = float(np.mean([x["kept"] for x in inner_checks]))
    enabled = inner_high_recall >= 1.0 - 1e-12
    return floor, enabled, {
        "enabled": bool(enabled),
        "inner_high_n": int(len(inner_checks)),
        "inner_high_recall": inner_high_recall,
        "outer_training_high_n": int(len(highs)),
        "inner_checks": inner_checks,
    }


def main():
    master = normalize_master(pd.read_excel(MASTER))
    master["target_clean"] = pd.to_numeric(
        master["totalObligatedFunding"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    master["actual_band"] = master["target_clean"].map(funding_band)
    master["disasterNumber"] = master["disasterNumber"].astype(int)

    # Only target-blind mission semantics are added. Biological is frozen/excluded.
    ma = fetch_all_mission_assignments()
    ma_hash = data_hash(ma)
    sem, _ = build_semantic_rollup(master, ma)
    df = master.merge(sem, on="disasterNumber", how="left")
    df = df[df["incidentType"] != "Biological"].copy().reset_index(drop=True)

    if "missionAssignmentCount" not in df.columns:
        raise RuntimeError("missionAssignmentCount missing from model frame")
    df["missionAssignmentCount"] = mission_values(df)

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
    flood_features = current

    accepted = pd.read_csv(ACCEPTED)
    accepted["disasterNumber"] = accepted["disasterNumber"].astype(int)
    if len(accepted) != 912:
        raise AssertionError(f"Expected 912 accepted rows, got {len(accepted)}")
    for col in ["root_actual_high", "root_pred_high", "actual_band", PRED_COL]:
        if col not in accepted.columns:
            raise RuntimeError(f"Accepted artifact missing {col}")

    baseline_correct = int((accepted[PRED_COL] == accepted["actual_band"]).sum())
    if baseline_correct != 815:
        raise AssertionError(f"Expected accepted score 815, got {baseline_correct}")

    model = df.merge(
        accepted[["disasterNumber", "root_pred_high", PRED_COL]],
        on="disasterNumber",
        how="inner",
        validate="one_to_one",
    )
    if len(model) != 912:
        raise AssertionError(f"Expected 912 merged rows, got {len(model)}")

    candidate_pred = {
        int(r.disasterNumber): str(getattr(r, PRED_COL))
        for r in model.itertuples(index=False)
    }
    candidate_root = {
        int(r.disasterNumber): int(r.root_pred_high)
        for r in model.itertuples(index=False)
    }

    diagnostics = []
    changed_rows = []

    for fy in sorted(model["fyDeclared"].astype(int).unique()):
        train = model[model["fyDeclared"].astype(int) != fy].copy()
        test = model[model["fyDeclared"].astype(int) == fy].copy()

        floor, enabled, floor_diag = operational_floor(train)

        test_missions = mission_values(test).to_numpy(float)
        eligible = (
            (test["root_pred_high"].to_numpy(int) == 1)
            & rare_scope(test).to_numpy(bool)
        )
        veto = eligible & enabled & (test_missions < floor)
        veto_rows = test.loc[veto].copy()

        # Compute the accepted low branch strictly from outer-training only.
        low_predict = build_low_predictor(train, semfeat, int(fy))
        low_pred = low_predict(veto_rows) if not veto_rows.empty else np.array([], dtype=str)

        # Apply the already accepted Flood guard70 to newly-demoted Flood rows.
        flood_th = None
        flood_enabled = False
        if not veto_rows.empty and np.any(veto_rows["incidentType"].to_numpy() == "Flood"):
            flood_inner = inner_base_and_flood_scores(
                train[train["target_clean"] < HIGH_CUTOFF].copy(),
                semfeat,
                flood_features,
                int(fy),
            )
            flood_th, flood_diag = choose_rescue_threshold(flood_inner, 0.70)

            flood_train = train[
                (train["incidentType"] == "Flood")
                & (train["target_clean"] >= 100_000)
                & (train["target_clean"] < HIGH_CUTOFF)
            ].copy()
            flood_y = (flood_train["target_clean"] >= 1_000_000).astype(int)
            if len(flood_train) >= 12 and flood_y.nunique() >= 2:
                flood_model = fit_flood_log(
                    flood_train, flood_features, 900000 + int(fy)
                )
                rescue_idx = np.flatnonzero(
                    (low_pred == "100K-1M")
                    & (veto_rows["incidentType"].to_numpy() == "Flood")
                )
                if len(rescue_idx):
                    probs = flood_model.predict_proba(
                        normalize_model_frame(veto_rows.iloc[rescue_idx][flood_features])
                    )[:, 1]
                    low_pred[rescue_idx[probs >= flood_th]] = "1M-50M"
                flood_enabled = True
        else:
            flood_diag = {}

        for (_, row), lp in zip(veto_rows.iterrows(), low_pred):
            dn = int(row["disasterNumber"])
            old = candidate_pred[dn]
            candidate_root[dn] = 0
            candidate_pred[dn] = str(lp)
            changed_rows.append({
                "disasterNumber": dn,
                "state": row["state"],
                "incidentType": row["incidentType"],
                "fyDeclared": int(row["fyDeclared"]),
                "actual_band": row["actual_band"],
                "old_pred": old,
                "new_pred": str(lp),
                "missionAssignmentCount": float(row["missionAssignmentCount"]),
                "operational_floor": float(floor),
                "floor_enabled": bool(enabled),
            })

        diagnostics.append({
            "outer_fy": int(fy),
            "operational_floor": float(floor),
            "floor_enabled": bool(enabled),
            "inner_high_n": floor_diag.get("inner_high_n"),
            "inner_high_recall": floor_diag.get("inner_high_recall"),
            "outer_training_high_n": floor_diag.get("outer_training_high_n"),
            "eligible_root_high_rare": int(eligible.sum()),
            "veto_count": int(veto.sum()),
            "flood_guard_threshold": None if flood_th is None else float(flood_th),
            "flood_guard_enabled": bool(flood_enabled),
        })

    out = model[[
        "disasterNumber", "state", "incidentType", "fyDeclared",
        "actual_band", "target_clean", "root_actual_high", "root_pred_high",
        PRED_COL,
    ]].copy()
    out["candidate_root_pred_high"] = out["disasterNumber"].map(candidate_root).astype(int)
    out["candidate_pred"] = out["disasterNumber"].map(candidate_pred).astype(str)
    out.to_csv(OUT / "candidate_predictions.csv", index=False)

    changed = pd.DataFrame(changed_rows)
    changed.to_csv(OUT / "changed_rows.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUT / "fold_diagnostics.csv", index=False)

    base_y = out["root_actual_high"].to_numpy(int)
    base_yp = out["root_pred_high"].to_numpy(int)
    cand_yp = out["candidate_root_pred_high"].to_numpy(int)

    baseline_fn = set(out.loc[(base_y == 1) & (base_yp == 0), "disasterNumber"].astype(int))
    candidate_fn = set(out.loc[(base_y == 1) & (cand_yp == 0), "disasterNumber"].astype(int))
    baseline_fp = int(((base_y == 0) & (base_yp == 1)).sum())
    candidate_fp = int(((base_y == 0) & (cand_yp == 1)).sum())

    candidate_metrics = six_band_metrics(out, "candidate_pred")
    candidate_correct = int((out["candidate_pred"] == out["actual_band"]).sum())
    high_recall = float(recall_score(base_y, cand_yp, pos_label=1, zero_division=0))

    if baseline_fn != {4353}:
        raise AssertionError(f"Unexpected accepted high FN set: {sorted(baseline_fn)}")

    summary = {
        "mission_nonfinancial_sha256": ma_hash,
        "floor_multiplier": FLOOR_MULTIPLIER,
        "scope": "non-Hurricane, non-Fire, non-Biological accepted root-high rows",
        "baseline": {
            "overall_correct": baseline_correct,
            "overall_total": int(len(out)),
            "root_fp": baseline_fp,
            "high_false_negatives": sorted(baseline_fn),
        },
        "candidate": {
            "overall_correct": candidate_correct,
            "overall_total": int(len(out)),
            "root_fp": candidate_fp,
            "root_high_recall": high_recall,
            "high_false_negatives": sorted(candidate_fn),
            "safe_same_high_fn_set": candidate_fn == baseline_fn,
            "end_to_end": candidate_metrics,
        },
        "changed_rows": changed_rows,
        "acceptance_rule": (
            "Accept only if the exact high-value false-negative set remains unchanged "
            "and end-to-end six-band correctness improves over 815/912."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Post-815 rare-hazard operational-floor confirmation",
        "",
        f"- Fixed floor multiplier: **{FLOOR_MULTIPLIER:.2f}**",
        f"- Baseline: **{baseline_correct}/{len(out)}**; root FP **{baseline_fp}**; high FN set **{sorted(baseline_fn)}**.",
        f"- Candidate: **{candidate_correct}/{len(out)}**; root FP **{candidate_fp}**; high recall **{high_recall:.1%}**.",
        f"- Candidate high FN set: **{sorted(candidate_fn)}**; same as baseline: **{candidate_fn == baseline_fn}**.",
        "",
        "## Changed rows",
        "",
        "| FEMA | State | Hazard | Actual | Old | New | Missions | Floor |",
        "|---:|---|---|---|---|---|---:|---:|",
    ]
    for r in changed_rows:
        md.append(
            f"| {r['disasterNumber']} | {r['state']} | {r['incidentType']} | "
            f"{r['actual_band']} | {r['old_pred']} | {r['new_pred']} | "
            f"{r['missionAssignmentCount']:.0f} | {r['operational_floor']:.1f} |"
        )

    e = candidate_metrics
    md += [
        "",
        "## Candidate six-band metrics",
        "",
        f"Overall: **{e['overall_correct']}/{e['overall_total']} ({e['overall_accuracy']:.1%})**; macro recall **{e['macro_recall']:.1%}**.",
        "",
        "| Band | Correct | Total | Recall |",
        "|---|---:|---:|---:|",
    ]
    for band, b in e["per_band"].items():
        md.append(f"| {band} | {b['correct']} | {b['total']} | {b['recall']:.1%} |")

    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
