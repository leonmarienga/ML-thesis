#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
ACCEPTED = ROOT / "audit_inputs" / "post830_ia_impact" / "accepted" / "candidate_predictions.csv"
OUT = ROOT / "audit_outputs" / "nonbio_830_ia_impact_highband_diagnostic"
OUT.mkdir(parents=True, exist_ok=True)

API = "https://gis.fema.gov/arcgis/rest/services/FEMA/IA_Applicants/FeatureServer/1/query"
HIGH_BANDS = ["50-200M", "200-500M", "500M+"]
COUNT_FIELDS = [
    "applicants_all",
    "applicants_valid",
    "inspections_issued",
    "inspections_returned",
    "ihp_eligible",
    "ha_eligible",
    "ona_eligible",
]


def funding_band(v: float) -> str:
    if v < 100_000:
        return "0-100K"
    if v < 1_000_000:
        return "100K-1M"
    if v < 50_000_000:
        return "1M-50M"
    if v < 200_000_000:
        return "50-200M"
    if v < 500_000_000:
        return "200-500M"
    return "500M+"


def fetch_one(disaster_number: int) -> dict:
    params = {
        "where": f"disaster_number={int(disaster_number)}",
        "outFields": "disaster_number,st_abbr,county," + ",".join(COUNT_FIELDS),
        "returnGeometry": "false",
        "f": "json",
        "resultRecordCount": 2000,
    }
    r = requests.get(API, params=params, timeout=60)
    r.raise_for_status()
    payload = r.json()
    if "error" in payload:
        raise RuntimeError(payload["error"])
    feats = [x.get("attributes", {}) for x in payload.get("features", [])]
    row = {
        "disasterNumber": int(disaster_number),
        "ia_county_rows": len(feats),
        "ia_counties_nonnull": len({str(x.get("county")) for x in feats if x.get("county") not in (None, "")}),
    }
    for f in COUNT_FIELDS:
        vals = [x.get(f) for x in feats]
        nums = [float(v) for v in vals if v is not None]
        row[f"ia_{f}_sum"] = float(np.sum(nums)) if nums else 0.0
        row[f"ia_{f}_max"] = float(np.max(nums)) if nums else 0.0
    return row


def main():
    accepted = pd.read_csv(ACCEPTED)
    accepted["disasterNumber"] = accepted["disasterNumber"].astype(int)
    pred_col = "candidate_pred_new" if "candidate_pred_new" in accepted.columns else "candidate_pred"
    baseline_correct = int((accepted[pred_col] == accepted["actual_band"]).sum())
    if baseline_correct != 830:
        raise AssertionError(f"Expected accepted 830 baseline, got {baseline_correct}")

    high = accepted[accepted[pred_col].isin(HIGH_BANDS)].copy()
    disaster_numbers = sorted(high["disasterNumber"].astype(int).unique())

    rows = []
    for i, dn in enumerate(disaster_numbers, start=1):
        print(f"Fetching IA counts {i}/{len(disaster_numbers)} for FEMA {dn}", flush=True)
        rows.append(fetch_one(dn))
        time.sleep(0.05)

    ia = pd.DataFrame(rows)
    joined = high.merge(ia, on="disasterNumber", how="left", validate="one_to_one")
    joined.to_csv(OUT / "highband_ia_impact.csv", index=False)

    # Restrict diagnostic comparison to exact 50-200M vs 200-500M boundary.
    boundary = joined[joined["actual_band"].isin(["50-200M", "200-500M"])].copy()
    feature_rows = []
    for c in [x for x in joined.columns if x.startswith("ia_") and x.endswith(("_sum", "_max"))] + ["ia_county_rows", "ia_counties_nonnull"]:
        a = pd.to_numeric(boundary.loc[boundary.actual_band == "50-200M", c], errors="coerce").fillna(0.0)
        b = pd.to_numeric(boundary.loc[boundary.actual_band == "200-500M", c], errors="coerce").fillna(0.0)
        err = pd.to_numeric(boundary.loc[boundary.disasterNumber == 4086, c], errors="coerce")
        feature_rows.append({
            "feature": c,
            "fema4086": float(err.iloc[0]) if len(err) else None,
            "band50_200_min": float(a.min()) if len(a) else None,
            "band50_200_median": float(a.median()) if len(a) else None,
            "band50_200_max": float(a.max()) if len(a) else None,
            "band200_500_min": float(b.min()) if len(b) else None,
            "band200_500_median": float(b.median()) if len(b) else None,
            "band200_500_max": float(b.max()) if len(b) else None,
        })
    pd.DataFrame(feature_rows).to_csv(OUT / "ia_feature_boundary_summary.csv", index=False)

    err = joined[joined.disasterNumber == 4086]
    sandy_pair = joined[joined.disasterNumber.isin([4085, 4086])].copy()
    sandy_pair.to_csv(OUT / "sandy_ny_nj_ia_comparison.csv", index=False)

    summary = {
        "baseline_correct": baseline_correct,
        "high_prediction_rows": int(len(high)),
        "rows_with_ia_county_data": int((joined["ia_county_rows"] > 0).sum()),
        "fema4086_rows": int(len(err)),
        "fema4086_ia_county_rows": int(err["ia_county_rows"].iloc[0]) if len(err) else 0,
        "note": "Diagnostic only. Uses non-dollar FEMA Individual Assistance applicant/inspection/eligibility counts; does not modify predictions.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Post-830 IA impact high-band diagnostic",
        "",
        f"- Accepted baseline: **{baseline_correct}/912**",
        f"- Current high-band predictions audited: **{len(high)}**",
        f"- High-band rows with IA county data: **{summary['rows_with_ia_county_data']}**",
        "- Prediction changes: **0** (diagnostic only)",
        "",
        "## Sandy NY/NJ comparison",
    ]
    for _, r in sandy_pair.sort_values("disasterNumber").iterrows():
        lines.append(
            f"- FEMA {int(r.disasterNumber)} {r.state} | actual {r.actual_band} | pred {r[pred_col]} | "
            f"valid applicants={r.get('ia_applicants_valid_sum', 0):,.0f} | inspections={r.get('ia_inspections_issued_sum', 0):,.0f} | "
            f"IHP eligible={r.get('ia_ihp_eligible_sum', 0):,.0f} | county rows={int(r.get('ia_county_rows', 0))}"
        )
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
