#!/usr/bin/env python3
"""
Exploratory audit of the 9 remaining Hurricane root false positives versus the
15 true >=$50M Hurricanes under the confirmed binary Hurricane veto.

Purpose
-------
Identify physically/operationally interpretable non-financial features that
might support a second one-way Hurricane veto.

Important:
- This script is diagnostic only and does NOT modify the router.
- Features are built for all Hurricane declarations before the fixed focus
  groups are attached.
- No obligation/funding amount is used as a predictor.
- Funding labels are attached only after feature construction for audit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from mission_semantic_audit import (
    CURRENT_19, build_semantic_rollup, fetch_all_mission_assignments,
    normalize_master,
)
from external_severity_ablation import build_external
from nonbio_hazard_hierarchy import initial_mechanism_counts
from nonbio_outage_rescue import build_eaglei_all

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "hurricane_remaining_fp_audit"
OUT.mkdir(parents=True, exist_ok=True)

REMAINING_FP = {4338, 4285, 4336, 4286, 4335, 4734, 4828, 4817, 4798}
TRUE_HIGH = {4086, 4085, 4337, 4339, 4332, 4340, 4393, 4399, 4400, 4559, 4611, 4673, 4671, 4715, 4830}


def valid_numeric(df: pd.DataFrame, cols):
    out = []
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= 4 and s.nunique(dropna=True) > 1:
            out.append(c)
    return out


def auc_oriented(y, x):
    m = np.isfinite(x)
    if m.sum() < 6 or len(np.unique(y[m])) < 2:
        return None, None
    auc = roc_auc_score(y[m], x[m])
    if auc >= 0.5:
        return float(auc), "higher_high"
    return float(1.0 - auc), "lower_high"


def main():
    master = normalize_master(pd.read_excel(MASTER))
    ma = fetch_all_mission_assignments()
    sem, _ = build_semantic_rollup(master, ma)
    ext, ext_audit = build_external(master)
    mech = initial_mechanism_counts(master, ma)
    print("Building all-Hurricane EAGLE-I features...", flush=True)
    eag = build_eaglei_all(master)

    # Build ALL Hurricane features first, target-blind.
    h = master[master["incidentType"] == "Hurricane"].copy()
    feat = (
        h.merge(sem, on="disasterNumber", how="left")
         .merge(ext, on="disasterNumber", how="left")
         .merge(mech, on="disasterNumber", how="left")
         .merge(eag, on="disasterNumber", how="left")
    )
    feat["initial_usace_esf3_dfa_count"] = (
        feat["initial_usace_esf3_dfa_count"].fillna(0)
    )

    # Save full target-blind feature table before focus labels are attached.
    target_blind_cols = [
        c for c in feat.columns
        if c not in {"totalObligatedFunding"}
    ]
    feat[target_blind_cols].to_csv(
        OUT / "all_hurricane_features_target_blind.csv", index=False
    )

    # Attach fixed diagnostic groups only now.
    focus = feat[feat["disasterNumber"].isin(REMAINING_FP | TRUE_HIGH)].copy()
    focus["diagnostic_group"] = np.where(
        focus["disasterNumber"].isin(TRUE_HIGH), "true_high", "remaining_fp"
    )
    focus["y_high"] = (focus["diagnostic_group"] == "true_high").astype(int)

    candidate_cols = []
    candidate_cols += [c for c in CURRENT_19 if c in focus.columns]
    candidate_cols += [c for c in focus.columns if c.startswith("sem_") or c.startswith("ma_")]
    candidate_cols += [c for c in focus.columns if c.startswith("nhc_") or c.startswith("noaa_") or c.startswith("calfire_")]
    candidate_cols += [c for c in focus.columns if c.startswith("eaglei_")]
    candidate_cols += ["initial_usace_esf3_dfa_count"]
    candidate_cols = list(dict.fromkeys(candidate_cols))
    candidate_cols = valid_numeric(focus, candidate_cols)

    rows = []
    y = focus["y_high"].to_numpy(int)

    for c in candidate_cols:
        x = pd.to_numeric(focus[c], errors="coerce").to_numpy(float)
        auc, direction = auc_oriented(y, x)
        hi = focus.loc[focus["y_high"] == 1, c]
        lo = focus.loc[focus["y_high"] == 0, c]
        hi_num = pd.to_numeric(hi, errors="coerce")
        lo_num = pd.to_numeric(lo, errors="coerce")

        rows.append({
            "feature": c,
            "oriented_auc": auc,
            "direction_for_true_high": direction,
            "high_n": int(hi_num.notna().sum()),
            "fp_n": int(lo_num.notna().sum()),
            "high_median": float(hi_num.median()) if hi_num.notna().any() else None,
            "fp_median": float(lo_num.median()) if lo_num.notna().any() else None,
            "high_min": float(hi_num.min()) if hi_num.notna().any() else None,
            "high_max": float(hi_num.max()) if hi_num.notna().any() else None,
            "fp_min": float(lo_num.min()) if lo_num.notna().any() else None,
            "fp_max": float(lo_num.max()) if lo_num.notna().any() else None,
        })

    sep = pd.DataFrame(rows).sort_values(
        ["oriented_auc", "high_n", "fp_n"],
        ascending=[False, False, False],
        na_position="last",
    )
    sep.to_csv(OUT / "feature_separation.csv", index=False)

    display_cols = [
        "disasterNumber", "state", "fyDeclared", "diagnostic_group",
        "missionAssignmentCount", "durationDays", "declarationDelayDays",
        "initial_usace_esf3_dfa_count",
    ]
    display_cols += [
        c for c in [
            "eaglei_peak_outage_share",
            "eaglei_customer_hours_per_customer",
            "eaglei_residual_share_3d",
            "eaglei_residual_share_7d",
            "nhc_max_wind_kt",
            "nhc_min_pressure_mb",
        ] if c in focus.columns
    ]
    display_cols += [
        c for c in focus.columns
        if c.startswith("sem_") and c in sep.head(12)["feature"].tolist()
    ]
    display_cols = list(dict.fromkeys([c for c in display_cols if c in focus.columns]))
    focus[display_cols].sort_values(
        ["diagnostic_group", "fyDeclared", "state"]
    ).to_csv(OUT / "focus_24_cases.csv", index=False)

    # Top features with adequate coverage in both groups.
    adequate = sep[
        (sep["high_n"] >= 8)
        & (sep["fp_n"] >= 5)
        & sep["oriented_auc"].notna()
    ].head(20)

    summary = {
        "remaining_fp_n": len(REMAINING_FP),
        "true_high_n": len(TRUE_HIGH),
        "all_hurricane_n": int(len(feat)),
        "candidate_feature_n": len(candidate_cols),
        "top_adequate_features": adequate.to_dict(orient="records"),
        "note": (
            "Diagnostic only. Any proposed verifier must be pre-specified from an "
            "interpretable mechanism and then evaluated in a fresh strict nested-LFYO workflow."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Remaining Hurricane false-positive mechanism audit",
        "",
        f"- Remaining Hurricane false positives: **{len(REMAINING_FP)}**",
        f"- True >=$50M Hurricanes: **{len(TRUE_HIGH)}**",
        f"- All Hurricane declarations enriched target-blind: **{len(feat)}**",
        "",
        "## Top non-financial separators with adequate coverage",
        "",
        "| Feature | Oriented AUC | Direction for true high | High median | FP median | Coverage high / FP |",
        "|---|---:|---|---:|---:|---:|",
    ]

    for _, r in adequate.iterrows():
        md.append(
            f"| {r['feature']} | {r['oriented_auc']:.3f} | {r['direction_for_true_high']} | "
            f"{r['high_median']:.4g} | {r['fp_median']:.4g} | "
            f"{int(r['high_n'])}/{len(TRUE_HIGH)} / {int(r['fp_n'])}/{len(REMAINING_FP)} |"
        )

    md += [
        "",
        "## Method note",
        "",
        "This is exploratory separation only. No feature or threshold from this table is "
        "allowed to become a router rule without a new strict nested-LFYO workflow.",
    ]
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
