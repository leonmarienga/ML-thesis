#!/usr/bin/env python3
"""
Targeted EAGLE-I outage mechanism audit for the two remaining extreme-routing
contrast cases:
- FEMA 4671 Puerto Rico / Hurricane Fiona (true $500M+ miss)
- FEMA 4830 Georgia / Hurricane Helene (former false $500M+ candidate)

The annual EAGLE-I files are public ORNL data mirrored on Figshare (CC BY 4.0).
We read only state + disaster-window rows and compute NON-FINANCIAL outage metrics.

This is external_final / retrospective diagnostic evidence, not declaration-time
prediction evidence until a t0 snapshot is defined.
"""

from __future__ import annotations
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "fiona_helene_outages"
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    2022: "https://ndownloader.figshare.com/files/42547897",
    2024: "https://ndownloader.figshare.com/files/53581661",
}
MCC_URL = "https://ndownloader.figshare.com/files/42547708"
CASES = {
    4671: {"state_name": "Puerto Rico", "fips_prefix": "72", "label": "Fiona PR"},
    4830: {"state_name": "Georgia", "fips_prefix": "13", "label": "Helene GA"},
}

def load_mcc():
    p = OUT / "MCC.csv"
    if not p.exists():
        r = requests.get(MCC_URL, timeout=120)
        r.raise_for_status()
        p.write_bytes(r.content)
    m = pd.read_csv(p, encoding="utf-8-sig")
    m.columns = [c.strip() for c in m.columns]
    m["County_FIPS"] = pd.to_numeric(m["County_FIPS"], errors="coerce")
    m = m[m["County_FIPS"].notna()].copy()
    m["fips"] = m["County_FIPS"].astype(int).astype(str).str.zfill(5)
    m["Customers"] = pd.to_numeric(m["Customers"], errors="coerce")
    return m

def query_case(url: str, state: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    con = duckdb.connect()
    # Conservative CSV parse: source schema is fips_code, county, state,
    # customers_out, run_start_time (+ optional other columns).
    q = """
        SELECT
            CAST(run_start_time AS VARCHAR) AS run_start_time,
            SUM(TRY_CAST(customers_out AS DOUBLE)) AS state_out,
            COUNT(*) AS county_rows
        FROM read_csv_auto(?, header=true, all_varchar=true, sample_size=200000)
        WHERE state = ?
          AND TRY_CAST(run_start_time AS TIMESTAMP) >= ?
          AND TRY_CAST(run_start_time AS TIMESTAMP) <= ?
        GROUP BY run_start_time
        ORDER BY TRY_CAST(run_start_time AS TIMESTAMP)
    """
    return con.execute(
        q,
        [
            url, state,
            start.tz_localize(None).to_pydatetime(),
            end.tz_localize(None).to_pydatetime()
        ],
    ).df()

def metrics(ts: pd.DataFrame, total_customers: float):
    if ts.empty:
        return {}
    x = ts.copy()
    x["t"] = pd.to_datetime(x["run_start_time"], errors="coerce", utc=True)
    x["out"] = pd.to_numeric(x["state_out"], errors="coerce")
    x = x.dropna(subset=["t","out"]).sort_values("t")
    if x.empty:
        return {}

    peak_i = x["out"].idxmax()
    peak = float(x.loc[peak_i, "out"])
    peak_t = x.loc[peak_i, "t"]

    # EAGLE-I is 15-minute cadence; integrate actual gaps capped at one hour
    # to avoid over-crediting source gaps.
    dt = x["t"].diff().dt.total_seconds().div(3600).fillna(0).clip(0,1)
    cust_hours = float((x["out"] * dt).sum())

    def nearest_after(days):
        t = peak_t + pd.Timedelta(days=days)
        z = x[x["t"] >= t]
        return float(z.iloc[0]["out"]) if len(z) else np.nan

    after = x[x["t"] >= peak_t]
    h50 = after[after["out"] <= peak * 0.50]
    h10 = after[after["out"] <= peak * 0.10]

    return {
        "peak_customers_out": peak,
        "peak_time": str(peak_t),
        "total_customers_mcc": float(total_customers),
        "peak_outage_share": float(peak / total_customers) if total_customers > 0 else None,
        "customer_hours_out": cust_hours,
        "customer_hours_per_customer": float(cust_hours / total_customers) if total_customers > 0 else None,
        "out_3d_after_peak": nearest_after(3),
        "out_7d_after_peak": nearest_after(7),
        "out_14d_after_peak": nearest_after(14),
        "residual_share_3d": float(nearest_after(3)/peak) if peak > 0 and np.isfinite(nearest_after(3)) else None,
        "residual_share_7d": float(nearest_after(7)/peak) if peak > 0 and np.isfinite(nearest_after(7)) else None,
        "residual_share_14d": float(nearest_after(14)/peak) if peak > 0 and np.isfinite(nearest_after(14)) else None,
        "half_life_hours": float((h50.iloc[0]["t"]-peak_t).total_seconds()/3600) if len(h50) else None,
        "restore90_hours": float((h10.iloc[0]["t"]-peak_t).total_seconds()/3600) if len(h10) else None,
        "observations": int(len(x)),
    }

def main():
    master = pd.read_excel(MASTER)
    mcc = load_mcc()
    rows = []
    series_dir = OUT / "series"
    series_dir.mkdir(exist_ok=True)

    for dn, cfg in CASES.items():
        r = master[master["disasterNumber"] == dn].iloc[0]
        begin = pd.to_datetime(r["incidentBeginDate"], errors="coerce", utc=True)
        # Use a standard window around incident onset for both cases:
        # 2 days before through 30 days after onset.
        start = begin - pd.Timedelta(days=2)
        end = begin + pd.Timedelta(days=30)
        year = int(begin.year)

        total = float(
            mcc.loc[mcc["fips"].str.startswith(cfg["fips_prefix"]), "Customers"].sum()
        )

        print(f"Querying {cfg['label']} {year}: {start} .. {end}", flush=True)
        ts = query_case(FILES[year], cfg["state_name"], start, end)
        ts.to_csv(series_dir / f"{dn}.csv", index=False)
        feat = metrics(ts, total)

        rows.append({
            "disasterNumber": dn,
            "label": cfg["label"],
            "state": r["state"],
            "incidentBeginDate": str(begin),
            "target_for_audit_only": float(r["totalObligatedFunding"]),
            **feat,
        })
        print(rows[-1], flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "fiona_helene_outage_features.csv", index=False)

    # Comparison ratios are diagnostic only.
    f = out.set_index("disasterNumber")
    ratios = {}
    if 4671 in f.index and 4830 in f.index:
        for c in ["peak_outage_share","customer_hours_per_customer","residual_share_3d",
                  "residual_share_7d","residual_share_14d","half_life_hours","restore90_hours"]:
            a = pd.to_numeric(pd.Series([f.loc[4671,c]]), errors="coerce").iloc[0]
            b = pd.to_numeric(pd.Series([f.loc[4830,c]]), errors="coerce").iloc[0]
            ratios[c] = float(a/b) if pd.notna(a) and pd.notna(b) and b != 0 else None

    summary = {
        "source": "ORNL EAGLE-I annual outage files via Figshare",
        "license": "CC BY 4.0",
        "file_ids": {"2022":42547897,"2024":53581661,"MCC":42547708},
        "window": "incident begin -2 days through +30 days",
        "comparison_ratios_fiona_over_helene": ratios,
        "timing_note": "Retrospective external_final diagnostic; not t0-valid yet.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = ["# Fiona vs Helene grid-disruption audit",""]
    for _,r in out.iterrows():
        md += [
            f"## {r['label']}",
            f"- Peak customers out: **{r.get('peak_customers_out',np.nan):,.0f}**",
            f"- Peak outage share: **{r.get('peak_outage_share',np.nan):.1%}**",
            f"- Customer-hours per customer: **{r.get('customer_hours_per_customer',np.nan):,.1f}**",
            f"- Residual outage at 3d: **{r.get('residual_share_3d',np.nan):.1%}**",
            f"- Residual outage at 7d: **{r.get('residual_share_7d',np.nan):.1%}**",
            f"- Residual outage at 14d: **{r.get('residual_share_14d',np.nan):.1%}**",
            f"- 50% restoration time: **{r.get('half_life_hours',np.nan):,.1f} h**",
            f"- 90% restoration time: **{r.get('restore90_hours',np.nan):,.1f} h**",
            "",
        ]
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

if __name__ == "__main__":
    main()
