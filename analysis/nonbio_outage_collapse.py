#!/usr/bin/env python3
"""
Strict-LFYO audit of a target-blind EAGLE-I persistent grid-collapse verifier.

Hypothesis fixed BEFORE evaluation
----------------------------------
A Hurricane missed by the confirmed non-Biological extreme gate may be promoted
to $500M+ only when all of the following hold:

1. Existing mechanism prerequisite is present:
   >=1 initial USACE + ESF-3 + DFA Mission Assignment.
2. EAGLE-I source coverage exists.
3. Peak outage penetration >= 50% of the state's MCC customer base.
4. Seven-day residual >= 50% of the event's peak outage.

The 50/50 thresholds are operational, not target-fitted: a majority of the
customer base loses power, and a majority of the peak outage remains after one
week. The rule is one-way rescue only; it never rejects an already-correct
extreme candidate and never changes non-Hurricane routing.

All EAGLE-I features are built for all 108 Hurricane declarations using only
state + incident dates before the >=$50M evaluation subset is selected.
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
from external_severity_ablation import build_external
from nonbio_hazard_hierarchy import band, initial_mechanism_counts
from nonbio_outage_rescue import (
    build_eaglei_all,
    metrics,
    predict_confirmed_fold,
)

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "nonbio_outage_collapse"
OUT.mkdir(parents=True, exist_ok=True)

PEAK_THRESHOLD = 0.50
RESIDUAL_7D_THRESHOLD = 0.50


def main():
    master = normalize_master(pd.read_excel(MASTER))
    ma = fetch_all_mission_assignments()

    print("Building target-blind EAGLE-I features for all Hurricane declarations...", flush=True)
    eag = build_eaglei_all(master)
    eag.to_csv(OUT / "all108_eaglei_features.csv", index=False)

    hurricane_ref = (
        master.loc[
            master["incidentType"] == "Hurricane",
            ["disasterNumber", "fyDeclared", "state", "totalObligatedFunding"],
        ]
        .merge(eag, on="disasterNumber", how="left")
    )
    hurricane_ref["persistent_grid_collapse_50_50"] = (
        (hurricane_ref["eaglei_coverage"].fillna(0) == 1)
        & (hurricane_ref["eaglei_peak_outage_share"] >= PEAK_THRESHOLD)
        & (hurricane_ref["eaglei_residual_share_7d"] >= RESIDUAL_7D_THRESHOLD)
    )
    hurricane_ref.to_csv(OUT / "all108_collapse_audit.csv", index=False)

    sem, _ = build_semantic_rollup(master, ma)
    ext, match_audit = build_external(master)
    mech = initial_mechanism_counts(master, ma)
    match_audit.to_csv(OUT / "external_match_audit.csv", index=False)

    df = (
        master.merge(sem, on="disasterNumber", how="left")
        .merge(ext, on="disasterNumber", how="left")
        .merge(mech, on="disasterNumber", how="left")
        .merge(eag, on="disasterNumber", how="left")
    )
    df["initial_usace_esf3_dfa_count"] = (
        df["initial_usace_esf3_dfa_count"].fillna(0).astype(int)
    )

    high = df[
        (df["incidentType"] != "Biological")
        & (df["totalObligatedFunding"] >= 50_000_000)
    ].copy().reset_index(drop=True)
    high["actual_band"] = high["totalObligatedFunding"].map(band)
    assert len(high) == 23

    current = [c for c in CURRENT_19 if c in high.columns]
    sem_cols = [
        c for c in high.columns
        if (c.startswith("sem_") or c.startswith("ma_"))
        and high[c].notna().sum() >= 2
        and high[c].nunique(dropna=True) > 1
    ]
    ext_cols = [
        c for c in high.columns
        if (c.startswith("nhc_") or c.startswith("noaa_") or c.startswith("calfire_"))
        and high[c].notna().sum() >= 2
        and high[c].nunique(dropna=True) > 1
    ]
    gate_features = current + sem_cols + ext_cols
    lower_features = current + sem_cols

    rows = []
    for outer_fy in sorted(high["fyDeclared"].astype(int).unique()):
        train = high[high["fyDeclared"].astype(int) != outer_fy].copy()
        test = high[high["fyDeclared"].astype(int) == outer_fy].copy()

        base_rows, _ = predict_confirmed_fold(
            train,
            test,
            gate_features,
            lower_features,
            lower_features,
            use_outage_lower=False,
        )
        base_map = {x["disasterNumber"]: x["baseline_final"] for x in base_rows}

        for _, r in test.iterrows():
            dn = int(r["disasterNumber"])
            baseline = base_map[dn]
            coverage = int(pd.to_numeric(
                pd.Series([r.get("eaglei_coverage")]), errors="coerce"
            ).fillna(0).iloc[0])

            peak = pd.to_numeric(
                pd.Series([r.get("eaglei_peak_outage_share")]), errors="coerce"
            ).iloc[0]
            residual7 = pd.to_numeric(
                pd.Series([r.get("eaglei_residual_share_7d")]), errors="coerce"
            ).iloc[0]

            collapse = (
                r["incidentType"] == "Hurricane"
                and baseline != "500M+"
                and int(r["initial_usace_esf3_dfa_count"]) >= 1
                and coverage == 1
                and pd.notna(peak)
                and pd.notna(residual7)
                and float(peak) >= PEAK_THRESHOLD
                and float(residual7) >= RESIDUAL_7D_THRESHOLD
            )

            rows.append({
                "disasterNumber": dn,
                "state": r["state"],
                "incidentType": r["incidentType"],
                "fyDeclared": int(r["fyDeclared"]),
                "totalObligatedFunding": float(r["totalObligatedFunding"]),
                "actual_band": r["actual_band"],
                "mechanism_count": int(r["initial_usace_esf3_dfa_count"]),
                "eaglei_coverage": coverage,
                "peak_outage_share": float(peak) if pd.notna(peak) else np.nan,
                "residual_share_7d": float(residual7) if pd.notna(residual7) else np.nan,
                "customer_hours_per_customer": (
                    float(r["eaglei_customer_hours_per_customer"])
                    if pd.notna(r.get("eaglei_customer_hours_per_customer")) else np.nan
                ),
                "confirmed_pred": baseline,
                "collapse_rescue": bool(collapse),
                "collapse_pred": "500M+" if collapse else baseline,
            })

    pred = pd.DataFrame(rows)
    pred.to_csv(OUT / "predictions.csv", index=False)

    result = {
        "confirmed_20of23": metrics(pred, "confirmed_pred"),
        "persistent_grid_collapse_50_50": metrics(pred, "collapse_pred"),
    }

    # Freeze the previously reproduced hierarchy. Fail loudly if unrelated code drifted.
    base = result["confirmed_20of23"]
    assert base["overall_correct"] == 20
    assert base["per_band"]["50-200M"]["correct"] == 12
    assert base["per_band"]["200-500M"]["correct"] == 4
    assert base["per_band"]["500M+"]["correct"] == 4

    # Target-blind threshold sensitivity: counts/identities only, not performance selection.
    sens_rows = []
    covered = hurricane_ref[hurricane_ref["eaglei_coverage"].fillna(0) == 1].copy()
    for p in [0.40, 0.50, 0.60]:
        for q in [0.40, 0.50, 0.60]:
            hit = covered[
                (covered["eaglei_peak_outage_share"] >= p)
                & (covered["eaglei_residual_share_7d"] >= q)
            ]
            sens_rows.append({
                "peak_threshold": p,
                "residual7_threshold": q,
                "hit_count_all_covered_hurricanes": int(len(hit)),
                "disaster_numbers": ",".join(map(str, hit["disasterNumber"].astype(int).tolist())),
            })
    pd.DataFrame(sens_rows).to_csv(OUT / "target_blind_sensitivity.csv", index=False)

    all_hits = hurricane_ref[hurricane_ref["persistent_grid_collapse_50_50"]].copy()
    hit_cols = [
        "disasterNumber", "state", "fyDeclared",
        "eaglei_peak_outage_share", "eaglei_residual_share_7d",
        "eaglei_customer_hours_per_customer",
    ]

    summary = {
        "rule": {
            "peak_outage_share_gte": PEAK_THRESHOLD,
            "residual_share_7d_gte": RESIDUAL_7D_THRESHOLD,
            "mechanism_required": ">=1 initial USACE + ESF-3 + DFA Mission Assignment",
            "interpretation": (
                "majority of customer base out at peak AND majority of peak outage "
                "still unresolved after seven days"
            ),
        },
        "eaglei": {
            "all_hurricanes": int(len(hurricane_ref)),
            "coverage": int(hurricane_ref["eaglei_coverage"].fillna(0).sum()),
            "all_covered_rule_hits": int(len(all_hits)),
            "all_covered_rule_hit_rows": all_hits[hit_cols].to_dict(orient="records"),
        },
        "variants": result,
        "cautions": [
            "Rule was fixed on operational semantics, not optimized against funding labels.",
            "Sensitivity table reports target-blind hit identities only; it is not used to select thresholds.",
            "EAGLE-I remains retrospective external_final evidence until a prediction-time cutoff t0 is defined.",
            "Missing EAGLE-I coverage is never coded as zero and never causes rejection of an existing extreme prediction.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Persistent EAGLE-I grid-collapse verifier",
        "",
        "- Fixed hypothesis: **peak outage share >= 50% AND 7-day residual >= 50%**",
        "- Mechanism prerequisite: **>=1 initial USACE + ESF-3 + DFA MA**",
        f"- EAGLE-I coverage: **{summary['eaglei']['coverage']}/{summary['eaglei']['all_hurricanes']} Hurricanes**",
        f"- Target-blind all-covered 50/50 hits: **{summary['eaglei']['all_covered_rule_hits']}**",
        "",
        "| Variant | Overall | $50-200M | $200-500M | $500M+ |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, rr in result.items():
        b = rr["per_band"]
        md.append(
            f"| {name} | {rr['overall_correct']}/{rr['overall_total']} "
            f"({rr['overall_accuracy']:.1%}) | "
            f"{b['50-200M']['correct']}/{b['50-200M']['total']} "
            f"({b['50-200M']['recall']:.1%}) | "
            f"{b['200-500M']['correct']}/{b['200-500M']['total']} "
            f"({b['200-500M']['recall']:.1%}) | "
            f"{b['500M+']['correct']}/{b['500M+']['total']} "
            f"({b['500M+']['recall']:.1%}) |"
        )

    md += ["", "## 50/50 rule hits across all covered Hurricanes"]
    for rr in summary["eaglei"]["all_covered_rule_hit_rows"]:
        md.append(
            f"- FEMA {rr['disasterNumber']} {rr['state']} FY{rr['fyDeclared']}: "
            f"peak={rr['eaglei_peak_outage_share']:.1%}, "
            f"7d residual={rr['eaglei_residual_share_7d']:.1%}, "
            f"customer-hours/customer={rr['eaglei_customer_hours_per_customer']:.1f}"
        )

    md += ["", "## Remaining errors after collapse verifier"]
    for e in result["persistent_grid_collapse_50_50"]["errors"]:
        md.append(
            f"- FEMA {e['disasterNumber']} {e['state']} {e['incidentType']}: "
            f"{e['actual_band']} -> {e['collapse_pred']}"
        )

    md += [
        "",
        "Strict outer LFYO is preserved for the supervised hierarchy. "
        "The 50/50 outage rule is fixed and target-blind.",
    ]
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
