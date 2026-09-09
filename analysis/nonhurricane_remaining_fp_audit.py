#!/usr/bin/env python3
"""
Diagnostic audit of remaining non-Hurricane false-high entries versus true
non-Hurricane >=$50M cases under the accepted Hurricane two-key router.

This script is diagnostic only. It does not change the router.

Features are built target-blind for all non-Biological, non-Hurricane
declarations first. Fixed diagnostic groups are attached only after feature
construction.
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

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "nonhurricane_remaining_fp_audit"
OUT.mkdir(parents=True, exist_ok=True)

REMAINING_FP = {
    4413, 4240, 4562, 4795, 4421, 4420, 4463, 4663,
    4683, 4720, 4781, 4699, 4697, 4806, 4707,
}
TRUE_HIGH = {
    4344, 4353, 4407, 4652, 4724, 4277, 4827, 4404,
}


def valid_numeric(df: pd.DataFrame, cols):
    out = []
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= 4 and s.nunique(dropna=True) > 1:
            out.append(c)
    return out


def oriented_auc(y, x):
    mask = np.isfinite(x)
    if mask.sum() < 6 or len(np.unique(y[mask])) < 2:
        return None, None
    auc = roc_auc_score(y[mask], x[mask])
    if auc >= 0.5:
        return float(auc), "higher_high"
    return float(1.0 - auc), "lower_high"


def main():
    master = normalize_master(pd.read_excel(MASTER))
    ma = fetch_all_mission_assignments()
    sem, _ = build_semantic_rollup(master, ma)
    ext, ext_audit = build_external(master)
    mech = initial_mechanism_counts(master, ma)

    # Target-blind feature construction for all non-Bio, non-Hurricane rows.
    base = master[
        (master["incidentType"] != "Biological")
        & (master["incidentType"] != "Hurricane")
    ].copy()

    feat = (
        base.merge(sem, on="disasterNumber", how="left")
        .merge(ext, on="disasterNumber", how="left")
        .merge(mech, on="disasterNumber", how="left")
    )
    feat["initial_usace_esf3_dfa_count"] = (
        feat["initial_usace_esf3_dfa_count"].fillna(0)
    )

    target_blind_cols = [
        c for c in feat.columns
        if c != "totalObligatedFunding"
    ]
    feat[target_blind_cols].to_csv(
        OUT / "all_nonhurricane_features_target_blind.csv",
        index=False,
    )

    focus = feat[
        feat["disasterNumber"].isin(REMAINING_FP | TRUE_HIGH)
    ].copy()
    focus["diagnostic_group"] = np.where(
        focus["disasterNumber"].isin(TRUE_HIGH),
        "true_high",
        "remaining_fp",
    )
    focus["y_high"] = (
        focus["diagnostic_group"] == "true_high"
    ).astype(int)

    candidate_cols = []
    candidate_cols += [c for c in CURRENT_19 if c in focus.columns]
    candidate_cols += [
        c for c in focus.columns
        if c.startswith("sem_") or c.startswith("ma_")
    ]
    candidate_cols += [
        c for c in focus.columns
        if c.startswith("nhc_")
        or c.startswith("noaa_")
        or c.startswith("calfire_")
    ]
    candidate_cols += ["initial_usace_esf3_dfa_count"]
    candidate_cols = list(dict.fromkeys(candidate_cols))
    candidate_cols = valid_numeric(focus, candidate_cols)

    y = focus["y_high"].to_numpy(int)
    rows = []

    for c in candidate_cols:
        x = pd.to_numeric(
            focus[c], errors="coerce"
        ).to_numpy(float)
        auc, direction = oriented_auc(y, x)

        hi = pd.to_numeric(
            focus.loc[focus["y_high"] == 1, c],
            errors="coerce",
        )
        lo = pd.to_numeric(
            focus.loc[focus["y_high"] == 0, c],
            errors="coerce",
        )

        rows.append({
            "feature": c,
            "oriented_auc": auc,
            "direction_for_true_high": direction,
            "high_n": int(hi.notna().sum()),
            "fp_n": int(lo.notna().sum()),
            "high_median": float(hi.median()) if hi.notna().any() else None,
            "fp_median": float(lo.median()) if lo.notna().any() else None,
            "high_min": float(hi.min()) if hi.notna().any() else None,
            "high_max": float(hi.max()) if hi.notna().any() else None,
            "fp_min": float(lo.min()) if lo.notna().any() else None,
            "fp_max": float(lo.max()) if lo.notna().any() else None,
        })

    sep = pd.DataFrame(rows).sort_values(
        ["oriented_auc", "high_n", "fp_n"],
        ascending=[False, False, False],
        na_position="last",
    )
    sep.to_csv(
        OUT / "feature_separation.csv",
        index=False,
    )

    focus_cols = [
        "disasterNumber", "state", "incidentType", "fyDeclared",
        "diagnostic_group", "missionAssignmentCount",
        "responseComplexityScore", "initial_usace_esf3_dfa_count",
    ]
    top_features = sep.head(15)["feature"].tolist()
    focus_cols += top_features
    focus_cols = list(dict.fromkeys([
        c for c in focus_cols if c in focus.columns
    ]))
    focus[focus_cols].sort_values(
        ["diagnostic_group", "incidentType", "fyDeclared"]
    ).to_csv(
        OUT / "focus_cases.csv",
        index=False,
    )

    adequate = sep[
        (sep["high_n"] >= 6)
        & (sep["fp_n"] >= 10)
        & sep["oriented_auc"].notna()
    ].head(20)

    high_by_hazard = (
        focus[focus["y_high"] == 1]["incidentType"]
        .value_counts()
        .to_dict()
    )
    fp_by_hazard = (
        focus[focus["y_high"] == 0]["incidentType"]
        .value_counts()
        .to_dict()
    )

    summary = {
        "remaining_fp_n": len(REMAINING_FP),
        "true_high_n": len(TRUE_HIGH),
        "all_nonhurricane_n": int(len(feat)),
        "high_by_hazard": high_by_hazard,
        "fp_by_hazard": fp_by_hazard,
        "top_adequate_features": adequate.to_dict(orient="records"),
        "note": (
            "Diagnostic only. Any pooled verifier must be tested in a fresh "
            "strict nested-LFYO workflow and must not worsen the high-value FN set."
        ),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    md = [
        "# Remaining non-Hurricane false-positive mechanism audit",
        "",
        f"- Remaining false positives: **{len(REMAINING_FP)}**",
        f"- True non-Hurricane >=$50M cases: **{len(TRUE_HIGH)}**",
        f"- All non-Bio non-Hurricane declarations enriched target-blind: **{len(feat)}**",
        "",
        f"- High hazards: {high_by_hazard}",
        f"- FP hazards: {fp_by_hazard}",
        "",
        "## Top non-financial separators with adequate coverage",
        "",
        "| Feature | Oriented AUC | Direction | High median | FP median | Coverage high / FP |",
        "|---|---:|---|---:|---:|---:|",
    ]

    for _, r in adequate.iterrows():
        md.append(
            f"| {r['feature']} | {r['oriented_auc']:.3f} | "
            f"{r['direction_for_true_high']} | {r['high_median']:.4g} | "
            f"{r['fp_median']:.4g} | "
            f"{int(r['high_n'])}/{len(TRUE_HIGH)} / "
            f"{int(r['fp_n'])}/{len(REMAINING_FP)} |"
        )

    md += [
        "",
        "## Method note",
        "",
        "Exploratory separation only. No threshold from this table becomes a router rule directly.",
    ]

    (OUT / "summary.md").write_text(
        "\n".join(md),
        encoding="utf-8",
    )
    print("\n".join(md))


if __name__ == "__main__":
    main()
