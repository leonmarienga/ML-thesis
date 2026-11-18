#!/usr/bin/env python3
"""Reconcile signed historical public actions at distinct elapsed horizons."""
import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from thesis_forward_models import YEARS, read_table, split
from thesis_historical_coverage import START, HistoricalSource, aggregate, get_audit

THRESHOLDS = (50_000_000, 200_000_000, 500_000_000)


def fold_support(rollup, frozen):
    rows = []
    for year in YEARS:
        train, held = split(frozen, year)
        first = min(held.start.min(), held.declaration.min()).tz_localize("UTC")
        prior = rollup[rollup.disasterNumber.isin(train.disasterNumber)]
        for name, selection in [("published_current", prior.snapshot_status.eq("published_actions")),
                                ("elapsed_36m_plus_90d_at_origin", prior.release_review_date.lt(first) &
                                 prior.snapshot_status.eq("published_actions"))]:
            use = prior[selection]
            for j, boundary in enumerate(THRESHOLDS):
                value = "net_obligation_usd" if name == "published_current" else "net_through_36m_usd"
                y = use[use[value].ge(boundary)]
                bands = y if j == len(THRESHOLDS)-1 else y[y[value].lt(THRESHOLDS[j+1])]
                cells = bands.groupby(["state", "incidentType"]).event_group.nunique()
                rows.append({"held_fy": year, "policy": name, "first_held_safety_cutoff": first.isoformat(),
                  "published_label_rows": len(use), "published_label_episodes": use.event_group.nunique(),
                  "threshold_usd": boundary, "exceedance_cases": len(y), "exceedance_episodes": y.event_group.nunique(),
                  "band_cases": len(bands), "band_episodes": bands.event_group.nunique(),
                  "max_same_state_hazard_episodes": cells.max() if len(cells) else 0,
                  "same_state_hazard_cells_ge_2": cells.ge(2).sum(),
                  "same_state_hazard_cells_ge_8": cells.ge(8).sum()})
    return pd.DataFrame(rows)


def cuts(rollup, snapshot):
    rows = []
    complete = rollup[rollup.snapshot_status.eq("published_actions")]
    for months in (12, 24, 36, 60):
        eligible = complete[complete.declaration_date.add(pd.DateOffset(months=months)).le(snapshot)]
        value = eligible[f"net_through_{months}m_usd"]
        dif = (value-eligible.net_obligation_usd).abs()
        for boundary in THRESHOLDS:
            rows.append({"horizon_months": months, "compared_cases": len(eligible),
                "different_from_eventual_over_cent": dif.gt(.010001).sum(), "median_abs_delta_usd": dif.median(),
                "max_abs_delta_usd": dif.max(), "cutoff_usd": boundary, "fixed_horizon_cases": value.ge(boundary).sum(),
                "eventual_cases": eligible.net_obligation_usd.ge(boundary).sum(),
                "fixed_above_eventual_below": (value.ge(boundary) & eligible.net_obligation_usd.lt(boundary)).sum(),
                "fixed_below_eventual_above": (value.lt(boundary) & eligible.net_obligation_usd.ge(boundary)).sum()})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cohort = read_table(Path("research/results/hazards/cohort_features.csv"))
    historical = cohort[cohort.declaration.dt.tz_localize("UTC").ge(START)].copy()
    assert len(historical) == 689 and historical.fyDeclared.max() == 2024 and historical.declarationType.eq("DR").all()
    roster = pd.read_csv("research/continuous/coverage/audit/declaration_roster.csv")
    assert set(historical.disasterNumber) <= set(roster.disasterNumber)
    protected = set(pd.read_csv("research/results/follow_up/fy2025_target_free_roster.csv", usecols=["disasterNumber"]).disasterNumber)
    source = HistoricalSource(args.raw)
    actions = get_audit(source, historical.disasterNumber, protected)
    manifest = pd.DataFrame(source.manifest)
    snapshot = pd.to_datetime(manifest.retrieved_utc, utc=True).min().normalize()
    prior = historical[["disasterNumber", "fyDeclared", "state", "incidentType", "event_group",
                        "declaration", "target"]].rename(columns={"target": "frozen_workbook_target_usd"})
    prior["declaration_date"] = pd.to_datetime(prior.declaration, utc=True)
    rollup = aggregate(actions, prior)
    rollup["fixed_36m_end"] = rollup.declaration_date + pd.DateOffset(months=36)
    rollup["release_review_date"] = rollup.fixed_36m_end + timedelta(days=90)
    rollup["audit_snapshot_date"] = snapshot
    eligible = rollup.snapshot_status.eq("published_actions") & rollup.release_review_date.le(snapshot)
    rollup["horizon_36m_status"] = np.select(
        [rollup.snapshot_status.ne("published_actions"), eligible], [rollup.snapshot_status, "elapsed_under_current_archive"],
        default="elapsed_window_or_reporting_buffer_pending")
    rollup["eventual_delta_vs_workbook_usd"] = rollup.net_obligation_usd - rollup.frozen_workbook_target_usd
    tables = {"action_source_manifest.csv": manifest, "historical_actions.csv.gz": actions,
              "historical_horizons.csv": rollup, "threshold_transitions.csv": cuts(rollup, snapshot),
              "past_only_support.csv": fold_support(rollup, cohort)}
    for name, frame in tables.items():
        compression = {"method": "gzip", "mtime": 0} if name.endswith(".gz") else None
        frame.to_csv(args.output / name, index=False, compression=compression)
    print(f"Recorded {len(actions):,} signed actions, {int(eligible.sum())} elapsed historical horizons with published action structure.")


if __name__ == "__main__":
    main()
