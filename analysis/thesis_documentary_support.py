#!/usr/bin/env python3
"""Measure chronological support for versioned declaration features and elapsed targets."""
import argparse
from pathlib import Path

import pandas as pd

from thesis_forward_models import YEARS, split
from thesis_obligation_horizon_audit import THRESHOLDS

ROOT = Path("research/continuous")
HORIZONS = (0, 24, 48, 72)
GENERAL_PRIOR_MINIMUM = 30
GENERAL_HELD_MINIMUM = 15
TAIL_PRIOR_MINIMUM = 8
TAIL_HELD_MINIMUM = 5
ORIGIN_MINIMUM = 2


def load_cohort():
    dates = ["start", "end", "declaration"]
    keys = ["disasterNumber", "incidentType", "event_group", "fyDeclared", *dates]
    cohort = pd.read_csv("research/results/hazards/cohort_features.csv", parse_dates=dates, usecols=keys)
    cohort = cohort[cohort.incidentType.ne("Biological")].copy()
    assert len(cohort) == 912 and cohort.fyDeclared.max() == 2024 and cohort.disasterNumber.is_unique
    maturity = pd.read_csv(ROOT / "horizons/results/historical_horizons.csv", usecols=[
        "disasterNumber", "release_review_date", "horizon_36m_status", "net_through_36m_usd",
    ])
    maturity["release_review_date"] = pd.to_datetime(maturity.release_review_date, utc=True)
    result = cohort.merge(maturity, on="disasterNumber", how="left", validate="one_to_one")
    elapsed = result.horizon_36m_status.eq("elapsed_under_current_archive")
    assert elapsed.sum() == 324 and result.loc[elapsed, "net_through_36m_usd"].notna().all()
    return result


def source_sets(older):
    historical = pd.read_csv(older, usecols=["disasterNumber", "status", "capture_elapsed_upper_hours"])
    matched = historical[historical.status.eq("matched")]
    fields = pd.read_csv(ROOT / "announcement_fields/results/field_evidence.csv", usecols=["disasterNumber"])
    metadata = pd.read_csv(ROOT / "announcements/results/case_availability.csv",
                           usecols=["disasterNumber", "clock", "hours", "status"])
    bounded = metadata[metadata.clock.eq("archived_modified_utc") & metadata.status.eq("covered")]
    for hour in HORIZONS:
        exact = set(matched.loc[matched.capture_elapsed_upper_hours.le(hour), "disasterNumber"].astype(int))
        newer = set(bounded.loc[bounded.hours.eq(hour), "disasterNumber"]) & set(fields.disasterNumber)
        assert not exact & newer
        yield hour, {
            "exact_captures_2017_2020": exact,
            "archived_edit_metadata_2021_2024": newer,
            "combined_distinct_evidence_tiers": exact | newer,
            "maturity_only_ceiling": None,
        }


def count_origins(cohort, availability):
    rows = []
    for year in YEARS:
        prior, held = split(cohort, year)
        cutoff = min(held.start.min(), held.declaration.min()).tz_localize("UTC")
        prior = prior[prior.horizon_36m_status.eq("elapsed_under_current_archive") & prior.release_review_date.lt(cutoff)]
        held = held[held.horizon_36m_status.eq("elapsed_under_current_archive")]
        for hour, tiers in availability:
            for tier, eligible in tiers.items():
                p = prior if eligible is None else prior[prior.disasterNumber.isin(eligible)]
                h = held if eligible is None else held[held.disasterNumber.isin(eligible)]
                pn, hn = p.event_group.nunique(), h.event_group.nunique()
                row = {"held_fy": year, "horizon_hours": hour, "evidence_tier": tier,
                       "safe_origin_cutoff_utc": cutoff.isoformat(), "prior_cases": len(p), "prior_groups": pn,
                       "held_cases": len(h), "held_groups": hn,
                       "general_gate_at_origin": pn >= GENERAL_PRIOR_MINIMUM and hn >= GENERAL_HELD_MINIMUM}
                for usd in THRESHOLDS:
                    suffix = str(usd // 1_000_000) + "m"
                    row["prior_groups_ge_" + suffix] = p.loc[p.net_through_36m_usd.ge(usd), "event_group"].nunique()
                    row["held_groups_ge_" + suffix] = h.loc[h.net_through_36m_usd.ge(usd), "event_group"].nunique()
                row["rare_gate_at_origin"] = (row["general_gate_at_origin"] and row["prior_groups_ge_50m"] >= TAIL_PRIOR_MINIMUM
                                               and row["held_groups_ge_50m"] >= TAIL_HELD_MINIMUM)
                rows.append(row)
    return pd.DataFrame(rows)


def count_hazards(cohort, availability):
    rows = []
    for hour, tiers in availability:
        for tier, eligible in tiers.items():
            if eligible is None:
                continue
            qualified = cohort[cohort.disasterNumber.isin(eligible)]
            for hazard, cases in qualified.groupby("incidentType", sort=True):
                rows.append({"horizon_hours": hour, "evidence_tier": tier, "incident_type": hazard,
                             "cases": len(cases), "groups": cases.event_group.nunique(),
                             "elapsed_label_eligible_cases": cases.horizon_36m_status.eq("elapsed_under_current_archive").sum()})
    return pd.DataFrame(rows)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("document_identity", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = load_cohort()
    availability = list(source_sets(args.document_identity))
    origins = count_origins(cases, availability)
    summary = origins.groupby(["evidence_tier", "horizon_hours"], sort=True).agg(
        general_eligible_origins=("general_gate_at_origin", "sum"),
        rare_eligible_origins=("rare_gate_at_origin", "sum"),
    ).reset_index()
    summary["general_gate_met"] = summary.general_eligible_origins.ge(ORIGIN_MINIMUM)
    summary["rare_gate_met"] = summary.rare_eligible_origins.ge(ORIGIN_MINIMUM)
    origins.to_csv(args.output / "documentary_support_by_origin.csv", index=False)
    summary.to_csv(args.output / "documentary_gate_summary.csv", index=False)
    count_hazards(cases, availability).to_csv(args.output / "documentary_hazard_counts.csv", index=False)


if __name__ == "__main__":
    main()
