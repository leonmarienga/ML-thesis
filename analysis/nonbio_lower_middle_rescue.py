#!/usr/bin/env python3
"""
Strict-LFYO audit of a Hurricane lower->middle outage rescue plus Sandy target reconstruction.

Frozen hierarchy:
- confirmed non-Biological hierarchy = 20/23
- persistent EAGLE-I grid-collapse verifier = 21/23

New hypothesis (fixed on operational semantics before scoring):
A missed Hurricane lower-band prediction may be promoted from $50M-$200M to
$200M-$500M when:
  1) it is not already classified as $500M+,
  2) it has >=1 initial USACE + ESF-3 + DFA Mission Assignment,
  3) EAGLE-I coverage exists, and
  4) at least one-third of the jurisdiction's MCC customer base is still out
     three days after the event's peak outage.

The one-third threshold has a physical interpretation: severe persistent
infrastructure disruption affecting >=33.3% of the entire customer base after
72 hours. It is not fitted from funding labels.

Sandy target reconstruction is FINANCIAL AUDIT ONLY. It never enters a model or
decision rule. It compares the historical master target against the current
OpenFEMA MissionAssignments v2 obligation sum for FEMA 4085/4086 and Ida 4611.
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
    latest_assignment_state,
)
from external_severity_ablation import build_external
from nonbio_hazard_hierarchy import band, initial_mechanism_counts
from nonbio_outage_rescue import build_eaglei_all, metrics, predict_confirmed_fold

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "nonbio_lower_middle_rescue"
OUT.mkdir(parents=True, exist_ok=True)

EXTREME_PEAK = 0.50
EXTREME_RES7 = 0.50
MIDDLE_BASE3 = 1.0 / 3.0


def target_reconstruction_cases(master: pd.DataFrame, ma: pd.DataFrame) -> pd.DataFrame:
    """Financial audit only; these fields are never merged into predictors."""
    ids = [4085, 4086, 4611]
    x = ma.copy()
    x["disasterNumber"] = pd.to_numeric(x["disasterNumber"], errors="coerce").astype("Int64")
    x = x[x["disasterNumber"].isin(ids)].copy()
    x["obligationAmount_num"] = pd.to_numeric(x["obligationAmount"], errors="coerce").fillna(0.0)
    x["dateObligated_dt"] = pd.to_datetime(x["dateObligated"], errors="coerce", utc=True)

    latest = latest_assignment_state(x)
    latest["obligationAmount_num"] = pd.to_numeric(
        latest["obligationAmount"], errors="coerce"
    ).fillna(0.0)

    all_agg = x.groupby("disasterNumber").agg(
        current_api_raw_rows=("maId", "size"),
        current_api_unique_ma=("maId", "nunique"),
        current_api_sum_obligation_all=("obligationAmount_num", "sum"),
        current_api_positive_obligation=("obligationAmount_num", lambda s: float(s[s > 0].sum())),
        current_api_negative_obligation=("obligationAmount_num", lambda s: float(s[s < 0].sum())),
        current_api_max_date_obligated=("dateObligated_dt", "max"),
    ).reset_index()

    lat_agg = latest.groupby("disasterNumber").agg(
        current_api_sum_obligation_latest=("obligationAmount_num", "sum"),
        current_api_latest_unique_ma=("maId", "nunique"),
    ).reset_index()

    z = master.loc[
        master["disasterNumber"].astype(int).isin(ids),
        ["disasterNumber", "state", "fyDeclared", "missionAssignmentCount", "totalObligatedFunding"],
    ].copy()
    z = z.merge(all_agg, on="disasterNumber", how="left").merge(
        lat_agg, on="disasterNumber", how="left"
    )
    z["all_sum_minus_master"] = (
        z["current_api_sum_obligation_all"] - z["totalObligatedFunding"]
    )
    z["all_sum_relative_delta"] = (
        z["all_sum_minus_master"].abs()
        / z["totalObligatedFunding"].abs().clip(lower=1.0)
    )
    z["raw_row_count_delta"] = (
        z["current_api_raw_rows"] - z["missionAssignmentCount"]
    )
    return z


def main():
    master = normalize_master(pd.read_excel(MASTER))
    ma = fetch_all_mission_assignments()

    # Financial target audit is kept isolated.
    recon = target_reconstruction_cases(master, ma)
    recon.to_csv(OUT / "sandy_ida_target_reconstruction.csv", index=False)

    print("Building target-blind EAGLE-I features for all Hurricanes...", flush=True)
    eag = build_eaglei_all(master)

    href = master.loc[
        master["incidentType"] == "Hurricane",
        ["disasterNumber", "fyDeclared", "state", "totalObligatedFunding"],
    ].merge(eag, on="disasterNumber", how="left")
    href["eaglei_customer_base_out_3d"] = (
        href["eaglei_peak_outage_share"] * href["eaglei_residual_share_3d"]
    )
    href["middle_persistent_outage_1of3"] = (
        (href["eaglei_coverage"].fillna(0) == 1)
        & (href["eaglei_customer_base_out_3d"] >= MIDDLE_BASE3)
    )
    href.to_csv(OUT / "all108_middle_outage_audit.csv", index=False)

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
            train, test, gate_features, lower_features, lower_features,
            use_outage_lower=False,
        )
        base_map = {x["disasterNumber"]: x["baseline_final"] for x in base_rows}

        for _, r in test.iterrows():
            dn = int(r["disasterNumber"])
            confirmed = base_map[dn]
            coverage = int(pd.to_numeric(
                pd.Series([r.get("eaglei_coverage")]), errors="coerce"
            ).fillna(0).iloc[0])
            peak = pd.to_numeric(
                pd.Series([r.get("eaglei_peak_outage_share")]), errors="coerce"
            ).iloc[0]
            res3 = pd.to_numeric(
                pd.Series([r.get("eaglei_residual_share_3d")]), errors="coerce"
            ).iloc[0]
            res7 = pd.to_numeric(
                pd.Series([r.get("eaglei_residual_share_7d")]), errors="coerce"
            ).iloc[0]

            # Frozen 50/50 extreme rescue.
            collapse = (
                r["incidentType"] == "Hurricane"
                and confirmed != "500M+"
                and int(r["initial_usace_esf3_dfa_count"]) >= 1
                and coverage == 1
                and pd.notna(peak)
                and pd.notna(res7)
                and float(peak) >= EXTREME_PEAK
                and float(res7) >= EXTREME_RES7
            )
            collapse_pred = "500M+" if collapse else confirmed

            base3 = (
                float(peak) * float(res3)
                if pd.notna(peak) and pd.notna(res3) else np.nan
            )

            # New one-way lower -> middle rescue.
            middle_rescue = (
                r["incidentType"] == "Hurricane"
                and collapse_pred == "50-200M"
                and int(r["initial_usace_esf3_dfa_count"]) >= 1
                and coverage == 1
                and np.isfinite(base3)
                and base3 >= MIDDLE_BASE3
            )
            final = "200-500M" if middle_rescue else collapse_pred

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
                "residual_share_3d": float(res3) if pd.notna(res3) else np.nan,
                "residual_share_7d": float(res7) if pd.notna(res7) else np.nan,
                "customer_base_out_3d": base3,
                "confirmed_pred": confirmed,
                "collapse_pred": collapse_pred,
                "middle_rescue": bool(middle_rescue),
                "final_pred": final,
            })

    pred = pd.DataFrame(rows)
    pred.to_csv(OUT / "predictions.csv", index=False)

    result = {
        "confirmed_20of23": metrics(pred, "confirmed_pred"),
        "persistent_grid_collapse_21of23": metrics(pred, "collapse_pred"),
        "collapse_plus_middle_outage_rescue": metrics(pred, "final_pred"),
    }

    # Hard freeze checks: unrelated router behavior must reproduce exactly.
    b = result["confirmed_20of23"]
    assert b["overall_correct"] == 20
    assert b["per_band"]["50-200M"]["correct"] == 12
    assert b["per_band"]["200-500M"]["correct"] == 4
    assert b["per_band"]["500M+"]["correct"] == 4

    c = result["persistent_grid_collapse_21of23"]
    assert c["overall_correct"] == 21
    assert c["per_band"]["50-200M"]["correct"] == 12
    assert c["per_band"]["200-500M"]["correct"] == 4
    assert c["per_band"]["500M+"]["correct"] == 5

    hits = href[href["middle_persistent_outage_1of3"]].copy()
    hit_cols = [
        "disasterNumber", "state", "fyDeclared",
        "eaglei_peak_outage_share", "eaglei_residual_share_3d",
        "eaglei_customer_base_out_3d",
    ]

    summary = {
        "fixed_middle_rule": {
            "customer_base_out_3d_gte": MIDDLE_BASE3,
            "mechanism_required": ">=1 initial USACE + ESF-3 + DFA MA",
            "application": "one-way 50-200M -> 200-500M only after extreme routing",
        },
        "target_blind_all_hurricane_hits": hits[hit_cols].to_dict(orient="records"),
        "target_blind_hit_count": int(len(hits)),
        "variants": result,
        "target_reconstruction": recon.to_dict(orient="records"),
        "cautions": [
            "EAGLE-I is retrospective external_final evidence until a t0 is defined.",
            "The 1/3 threshold is operationally interpretable and not fitted to funding labels.",
            "Financial target reconstruction is audit-only and never enters prediction.",
            "Sandy remains unresolved unless an independently defensible mechanism is found.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    md = [
        "# Hurricane lower-middle persistent-outage rescue",
        "",
        "- Frozen extreme collapse hierarchy reproduced before new scoring.",
        "- Fixed lower->middle rule: **>= 1/3 of entire customer base still out 3 days after peak**",
        "- Mechanism prerequisite: **>=1 initial USACE + ESF-3 + DFA MA**",
        f"- Target-blind all-covered Hurricane hits: **{len(hits)}**",
        "",
        "| Variant | Overall | $50-200M | $200-500M | $500M+ |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, rr in result.items():
        pb = rr["per_band"]
        md.append(
            f"| {name} | {rr['overall_correct']}/{rr['overall_total']} "
            f"({rr['overall_accuracy']:.1%}) | "
            f"{pb['50-200M']['correct']}/{pb['50-200M']['total']} "
            f"({pb['50-200M']['recall']:.1%}) | "
            f"{pb['200-500M']['correct']}/{pb['200-500M']['total']} "
            f"({pb['200-500M']['recall']:.1%}) | "
            f"{pb['500M+']['correct']}/{pb['500M+']['total']} "
            f"({pb['500M+']['recall']:.1%}) |"
        )

    md += ["", "## Target-blind 1/3 customer-base outage hits"]
    for rr in summary["target_blind_all_hurricane_hits"]:
        md.append(
            f"- FEMA {rr['disasterNumber']} {rr['state']} FY{rr['fyDeclared']}: "
            f"peak={rr['eaglei_peak_outage_share']:.1%}, "
            f"3d residual={rr['eaglei_residual_share_3d']:.1%}, "
            f"entire-base still out at 3d={rr['eaglei_customer_base_out_3d']:.1%}"
        )

    md += ["", "## Remaining errors after new rescue"]
    for e in result["collapse_plus_middle_outage_rescue"]["errors"]:
        md.append(
            f"- FEMA {e['disasterNumber']} {e['state']} {e['incidentType']}: "
            f"{e['actual_band']} -> {e['final_pred']}"
        )

    md += ["", "## Sandy / Ida target reconstruction (audit only)"]
    for rr in summary["target_reconstruction"]:
        md.append(
            f"- FEMA {rr['disasterNumber']} {rr['state']}: master=${rr['totalObligatedFunding']:,.2f}; "
            f"current API all-row sum=${rr['current_api_sum_obligation_all']:,.2f}; "
            f"relative delta={rr['all_sum_relative_delta']:.2%}; "
            f"master rows={rr['missionAssignmentCount']}, current rows={rr['current_api_raw_rows']}"
        )

    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
