#!/usr/bin/env python3
"""Show retrospective evidence and a separate simulated allocation briefing."""
import argparse
import json
from pathlib import Path

import pandas as pd

from thesis_hazard_features import BANDS, PHYSICAL


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("disaster_number", type=int)
    p.add_argument("--results", type=Path, default=Path("research/results"))
    p.add_argument("--scenario-event", type=int, default=1)
    args = p.parse_args()
    root = args.results
    events = pd.read_csv(root/"hazards/cohort_features.csv").set_index("disasterNumber")
    event = events.loc[args.disaster_number]
    prediction = pd.read_csv(root/"forward/forward_predictions.csv").query(
        "variant == 'nested_metadata' and disasterNumber == @args.disaster_number")
    intervals = pd.read_csv(root/"uncertainty/forward_prediction_intervals.csv").query(
        "variant == 'nested_metadata' and method == 'dollar' and disasterNumber == @args.disaster_number")
    briefing = {
        "historical_disaster": {
            "id": args.disaster_number, "hazard": event.incidentType, "state": event.state,
            "recorded_dates": {"start": event.start, "end": event.end if pd.notna(event.end) else None},
            "historical_fema_obligations_usd": event.target,
            "evidence_fields_available": [f for f in PHYSICAL if pd.notna(event[f])],
            "evidence_fields_missing": [f for f in PHYSICAL if pd.isna(event[f])],
            "evidence_provenance": "research/SOURCE_AND_FEATURE_MANIFEST.csv",
        },
        "retrospective_model_estimate": {"status": "outside evaluated forward cohort"},
        "simulated_humanitarian_allocation": {"status": "independent synthetic scenario, expressed in service units"},
    }
    if len(prediction):
        row = prediction.iloc[0]
        briefing["retrospective_model_estimate"] = {
            "historical_obligation_estimate_usd": row.predicted_amount,
            "band": BANDS[row.predicted_band],
            "band_probabilities": {name: float(row[f"p_band_{i}"]) for i, name in enumerate(BANDS)},
            "intervals": intervals[["level", "lower_usd", "upper_usd"]].to_dict("records"),
            "human_review": True,
            "availability_limit": "later-year retrospective test, exact original decision-time versions unverified",
        }
    simulation = pd.read_csv(root/"allocation/synthetic_allocation_demonstration.csv").query(
        "scenario_event == @args.scenario_event")
    briefing["simulated_humanitarian_allocation"]["policy_comparison"] = simulation[
        ["hazard", "estimated_need_units", "simulated_need_units", "allocation_units", "vulnerability", "policy"]
    ].to_dict("records")
    print(json.dumps(briefing, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
