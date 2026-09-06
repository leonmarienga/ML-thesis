#!/usr/bin/env python3
"""
Strict-LFYO non-Biological hierarchy with EAGLE-I outage rescue.

This freezes the workflow-confirmed 20/23 hierarchy and tests two additions:

A) OUTAGE RESCUE ONLY
   A missed Hurricane can be promoted to $500M+ only when:
   - it has >=1 initial USACE + ESF-3 + DFA Mission Assignment, AND
   - EAGLE-I customer-hours-per-customer is a robust outlier relative to
     ALL other-fiscal-year Hurricane declarations with outage coverage.
   Robust outlier rule: modified z-score > 3.5 (standard Iglewicz-Hoaglin rule).

B) OUTAGE-AUGMENTED HURRICANE LOWER ROUTER
   Adds normalized EAGLE-I outage metrics to the existing Hurricane
   $50M-$200M vs $200M-$500M logistic specialist. Missing source coverage
   remains NaN and no coverage indicator is exposed to the model.

Leakage / selection controls:
- EAGLE-I features are built for ALL 108 Hurricane declarations before
  selecting the >=$50M evaluation set.
- EAGLE matching uses state + incident dates only.
- Outer FY is excluded from every supervised model.
- The outage anomaly reference distribution uses ALL covered Hurricane
  declarations from non-outer fiscal years, irrespective of funding.
- No target-derived financial field enters an EAGLE feature.
- The confirmed fire verifier and mechanism prerequisite are unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from mission_semantic_audit import (
    CURRENT_19,
    build_semantic_rollup,
    fetch_all_mission_assignments,
    normalize_master,
    normalize_model_frame,
)
from external_severity_ablation import build_external
from nonbio_hazard_hierarchy import (
    EXTREME,
    band,
    initial_mechanism_counts,
    mechanism_active_from_inner,
    extreme_candidates,
    fit_lower_general,
    fit_lower_hurricane,
    fit_binary,
)
from eaglei_all_hurricanes import (
    EAGLEI_FILE_IDS,
    MCC_FALLBACK,
    STATE_NAMES,
    STATE_FIPS,
    METRIC_COLS,
    load_mcc,
    scan_year,
    compute,
)

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "nonbio_outage_rescue"
OUT.mkdir(parents=True, exist_ok=True)

OUTAGE_FEATURES = [
    "eaglei_peak_outage_share",
    "eaglei_customer_hours_per_customer",
    "eaglei_residual_share_3d",
    "eaglei_residual_share_7d",
    "eaglei_residual_share_14d",
    "eaglei_half_life_hours",
    "eaglei_restore90_hours",
]


def build_eaglei_all(master: pd.DataFrame) -> pd.DataFrame:
    """Build target-blind EAGLE-I features for every Hurricane declaration."""
    h = master[master["incidentType"] == "Hurricane"].copy()
    h["begin"] = pd.to_datetime(h["incidentBeginDate"], errors="coerce", utc=True)
    h["event_year"] = h["begin"].dt.year

    mcc = load_mcc(MCC_FALLBACK)

    rows = []
    for dn in h["disasterNumber"].astype(int):
        row = {"disasterNumber": dn, "eaglei_coverage": 0}
        row.update({c: np.nan for c in METRIC_COLS})
        rows.append(row)
    feat = pd.DataFrame(rows).set_index("disasterNumber")

    eligible = h[
        h["event_year"].isin(EAGLEI_FILE_IDS.keys()) & h["begin"].notna()
    ].copy()

    for year, g in eligible.groupby("event_year"):
        year = int(year)
        states = sorted({
            STATE_NAMES.get(str(s))
            for s in g["state"]
            if STATE_NAMES.get(str(s))
        })
        if not states:
            continue

        start = g["begin"].min() - pd.Timedelta(days=2)
        end = g["begin"].max() + pd.Timedelta(days=30)
        url = f"https://ndownloader.figshare.com/files/{EAGLEI_FILE_IDS[year]}"
        print(f"EAGLE-I {year}: {len(g)} Hurricane rows; states={states}", flush=True)
        yr = scan_year(url, states, start, end)
        yr["t"] = pd.to_datetime(yr["run_start_time"], errors="coerce", utc=True)

        for _, r in g.iterrows():
            state = str(r["state"])
            state_name = STATE_NAMES.get(state)
            fips = STATE_FIPS.get(state)
            if not state_name or not fips:
                continue

            s = r["begin"] - pd.Timedelta(days=2)
            e = r["begin"] + pd.Timedelta(days=30)
            z = yr[
                (yr["state"] == state_name)
                & (yr["t"] >= s)
                & (yr["t"] <= e)
            ].copy()

            total = float(
                mcc.loc[mcc["fips"].str.startswith(fips), "Customers"].sum()
            )
            m = compute(z, total)
            dn = int(r["disasterNumber"])
            if m:
                for k, v in m.items():
                    feat.loc[dn, k] = v
                feat.loc[dn, "eaglei_coverage"] = 1

    return feat.reset_index()


def modified_z(value: float, reference: pd.Series) -> Tuple[float, float, float, int]:
    vals = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(float)
    if len(vals) < 5 or not np.isfinite(value):
        return np.nan, np.nan, np.nan, int(len(vals))

    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    if mad <= 0:
        return np.nan, med, mad, int(len(vals))

    z = float(0.6745 * (value - med) / mad)
    return z, med, mad, int(len(vals))


def predict_confirmed_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    gate_features: List[str],
    lower_features: List[str],
    hurricane_lower_features: List[str],
    use_outage_lower: bool,
):
    """Reproduce confirmed hierarchy, optionally changing only Hurricane lower router."""
    active, inner_candidates = mechanism_active_from_inner(train, gate_features)
    candidate = extreme_candidates(train, test, gate_features)

    general_lower = fit_lower_general(train, lower_features)
    general_pred = general_lower.predict(
        normalize_model_frame(test[lower_features])
    )

    h_features = hurricane_lower_features if use_outage_lower else lower_features
    hurricane_lower = fit_lower_hurricane(train, h_features)

    rows = []
    for (_, r), extreme0, lower0 in zip(test.iterrows(), candidate, general_pred):
        extreme = int(extreme0)

        if (
            extreme == 1
            and active
            and int(r["initial_usace_esf3_dfa_count"]) < 1
        ):
            extreme = 0

        if extreme == 1:
            final = "500M+"
        else:
            if r["incidentType"] == "Hurricane" and hurricane_lower is not None:
                hp = int(
                    hurricane_lower.predict(
                        normalize_model_frame(
                            test.loc[[r.name], h_features]
                        )
                    )[0]
                )
                final = "200-500M" if hp else "50-200M"
            else:
                final = "200-500M" if int(lower0) else "50-200M"

        rows.append({
            "disasterNumber": int(r["disasterNumber"]),
            "baseline_final": final,
            "initial_extreme_candidate": int(extreme0),
            "mechanism_active": bool(active),
        })
    return rows, inner_candidates


def metrics(pred: pd.DataFrame, col: str) -> Dict:
    per_band = {}
    for b in ["50-200M", "200-500M", "500M+"]:
        m = pred["actual_band"] == b
        correct = int((pred.loc[m, col] == b).sum())
        total = int(m.sum())
        per_band[b] = {
            "correct": correct,
            "total": total,
            "recall": correct / total if total else None,
        }
    correct = int((pred[col] == pred["actual_band"]).sum())
    return {
        "overall_correct": correct,
        "overall_total": int(len(pred)),
        "overall_accuracy": correct / len(pred),
        "per_band": per_band,
        "errors": pred.loc[
            pred[col] != pred["actual_band"],
            ["disasterNumber", "state", "incidentType", "actual_band", col]
        ].to_dict(orient="records"),
    }


def main():
    master = normalize_master(pd.read_excel(MASTER))
    ma = fetch_all_mission_assignments()

    print("Building all-Hurricane EAGLE-I features...", flush=True)
    eag = build_eaglei_all(master)
    eag.to_csv(OUT / "all108_eaglei_features.csv", index=False)

    # Keep complete Hurricane population for outer-training anomaly reference.
    hurricane_ref = (
        master[master["incidentType"] == "Hurricane"][
            ["disasterNumber", "fyDeclared", "state", "totalObligatedFunding"]
        ]
        .merge(eag, on="disasterNumber", how="left")
    )
    hurricane_ref.to_csv(OUT / "all108_outage_reference.csv", index=False)

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
    outage_cols = [
        c for c in OUTAGE_FEATURES
        if c in high.columns
        and high[c].notna().sum() >= 2
        and high[c].nunique(dropna=True) > 1
    ]

    gate_features = current + sem_cols + ext_cols
    lower_features = current + sem_cols
    hurricane_lower_outage = lower_features + outage_cols

    all_rows = []
    fold_rows = []

    for outer_fy in sorted(high["fyDeclared"].astype(int).unique()):
        train = high[high["fyDeclared"].astype(int) != outer_fy].copy()
        test = high[high["fyDeclared"].astype(int) == outer_fy].copy()

        # Variant 1: exact confirmed hierarchy.
        base_rows, inner_candidates = predict_confirmed_fold(
            train, test, gate_features, lower_features,
            hurricane_lower_outage, use_outage_lower=False
        )
        base_map = {x["disasterNumber"]: x for x in base_rows}

        # Variant 2: same hierarchy, Hurricane lower model gets outage metrics.
        outage_lower_rows, _ = predict_confirmed_fold(
            train, test, gate_features, lower_features,
            hurricane_lower_outage, use_outage_lower=True
        )
        outage_lower_map = {x["disasterNumber"]: x for x in outage_lower_rows}

        # Robust anomaly reference = ALL Hurricane declarations from OTHER FYs.
        ref = hurricane_ref[
            (hurricane_ref["fyDeclared"].astype(int) != outer_fy)
            & (hurricane_ref["eaglei_coverage"].fillna(0) == 1)
        ]["eaglei_customer_hours_per_customer"]

        for _, r in test.iterrows():
            dn = int(r["disasterNumber"])
            baseline = base_map[dn]["baseline_final"]
            lower_outage = outage_lower_map[dn]["baseline_final"]

            value = pd.to_numeric(
                pd.Series([r.get("eaglei_customer_hours_per_customer")]),
                errors="coerce"
            ).iloc[0]
            z, med, mad, nref = modified_z(value, ref)

            rescue = (
                r["incidentType"] == "Hurricane"
                and baseline != "500M+"
                and int(r["initial_usace_esf3_dfa_count"]) >= 1
                and pd.notna(r.get("eaglei_coverage"))
                and int(r.get("eaglei_coverage", 0)) == 1
                and np.isfinite(z)
                and z > 3.5
            )

            rescue_from_outage_lower = (
                r["incidentType"] == "Hurricane"
                and lower_outage != "500M+"
                and int(r["initial_usace_esf3_dfa_count"]) >= 1
                and pd.notna(r.get("eaglei_coverage"))
                and int(r.get("eaglei_coverage", 0)) == 1
                and np.isfinite(z)
                and z > 3.5
            )

            all_rows.append({
                "disasterNumber": dn,
                "state": r["state"],
                "incidentType": r["incidentType"],
                "fyDeclared": int(r["fyDeclared"]),
                "totalObligatedFunding": float(r["totalObligatedFunding"]),
                "actual_band": r["actual_band"],
                "mechanism_count": int(r["initial_usace_esf3_dfa_count"]),
                "eaglei_coverage": int(r.get("eaglei_coverage", 0) or 0),
                "customer_hours_per_customer": (
                    float(value) if np.isfinite(value) else np.nan
                ),
                "outage_modified_z": float(z) if np.isfinite(z) else np.nan,
                "outage_reference_n": nref,
                "outage_reference_median": med,
                "outage_reference_mad": mad,
                "confirmed_pred": baseline,
                "outage_rescue_pred": "500M+" if rescue else baseline,
                "outage_lower_pred": lower_outage,
                "outage_lower_plus_rescue_pred": (
                    "500M+" if rescue_from_outage_lower else lower_outage
                ),
            })

        fold_rows.append({
            "outer_fy": int(outer_fy),
            "outage_reference_n": int(pd.to_numeric(ref, errors="coerce").notna().sum()),
            "inner_candidate_count": len(inner_candidates),
        })

    pred = pd.DataFrame(all_rows)
    pred.to_csv(OUT / "predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "folds.csv", index=False)

    variants = {
        "confirmed_20of23": "confirmed_pred",
        "confirmed_plus_outage_rescue": "outage_rescue_pred",
        "outage_augmented_hurricane_lower": "outage_lower_pred",
        "outage_lower_plus_rescue": "outage_lower_plus_rescue_pred",
    }

    result = {name: metrics(pred, col) for name, col in variants.items()}

    # Audit the robust outlier rule across ALL 108 hurricanes out-of-year.
    anomaly_rows = []
    hfull = hurricane_ref.copy()
    for _, r in hfull.iterrows():
        if int(r.get("eaglei_coverage", 0) or 0) != 1:
            continue
        fy = int(r["fyDeclared"])
        ref = hfull[
            (hfull["fyDeclared"].astype(int) != fy)
            & (hfull["eaglei_coverage"].fillna(0) == 1)
        ]["eaglei_customer_hours_per_customer"]
        value = float(r["eaglei_customer_hours_per_customer"])
        z, med, mad, nref = modified_z(value, ref)
        anomaly_rows.append({
            "disasterNumber": int(r["disasterNumber"]),
            "state": r["state"],
            "fyDeclared": fy,
            "totalObligatedFunding_audit_only": float(r["totalObligatedFunding"]),
            "customer_hours_per_customer": value,
            "modified_z": z,
            "is_outlier_gt3_5": bool(np.isfinite(z) and z > 3.5),
            "reference_n": nref,
        })
    anomaly_df = pd.DataFrame(anomaly_rows)
    anomaly_df.to_csv(OUT / "all108_outlier_audit.csv", index=False)

    summary = {
        "eaglei": {
            "all_hurricanes": int(len(hurricane_ref)),
            "coverage": int(hurricane_ref["eaglei_coverage"].fillna(0).sum()),
            "high_value_hurricane_coverage": int(
                high.loc[high["incidentType"] == "Hurricane", "eaglei_coverage"]
                .fillna(0).sum()
            ),
        },
        "outlier_rule": (
            "modified z = 0.6745*(x-median)/MAD; rescue if z>3.5, "
            "using all covered non-outer-FY Hurricane declarations as reference; "
            "also require >=1 initial USACE+ESF3+DFA mission"
        ),
        "variants": result,
        "all108_outliers_gt3_5": int(
            anomaly_df["is_outlier_gt3_5"].sum()
        ) if len(anomaly_df) else 0,
        "cautions": [
            "EAGLE-I features are retrospective external_final evidence until t0 is defined.",
            "EAGLE-I geographic/utility coverage varies; missing coverage is never coded as zero.",
            "The outage-rescue mechanism is developmental and should be externally validated.",
        ],
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    md = [
        "# Non-Biological hierarchy + EAGLE-I outage rescue",
        "",
        f"- EAGLE-I coverage: **{summary['eaglei']['coverage']}/{summary['eaglei']['all_hurricanes']} Hurricane declarations**",
        f"- High-value Hurricane coverage: **{summary['eaglei']['high_value_hurricane_coverage']}/15**",
        "",
        "| Variant | Overall | $50-200M | $200-500M | $500M+ |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, r in result.items():
        b = r["per_band"]
        md.append(
            f"| {name} | {r['overall_correct']}/{r['overall_total']} "
            f"({r['overall_accuracy']:.1%}) | "
            f"{b['50-200M']['correct']}/{b['50-200M']['total']} "
            f"({b['50-200M']['recall']:.1%}) | "
            f"{b['200-500M']['correct']}/{b['200-500M']['total']} "
            f"({b['200-500M']['recall']:.1%}) | "
            f"{b['500M+']['correct']}/{b['500M+']['total']} "
            f"({b['500M+']['recall']:.1%}) |"
        )

    md += [
        "",
        "## Remaining errors: best variant",
    ]
    best_name = max(
        result,
        key=lambda n: (
            min(x["recall"] for x in result[n]["per_band"].values()),
            result[n]["overall_accuracy"],
        )
    )
    for e in result[best_name]["errors"]:
        md.append(
            f"- FEMA {e['disasterNumber']} {e['state']} {e['incidentType']}: "
            f"{e['actual_band']} -> {e[variants[best_name]]}"
        )
    md += [
        "",
        f"All-108 Hurricane robust outage outliers (>3.5): **{summary['all108_outliers_gt3_5']}**",
        "",
        "This is a strict outer-LFYO developmental audit. EAGLE-I remains retrospective until t0 is defined.",
    ]
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
