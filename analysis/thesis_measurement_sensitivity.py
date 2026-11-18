#!/usr/bin/env python3
"""Stress test linked severity and exposure inputs in the matched forward model."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from thesis_forward_models import ARMS, YEARS, fit_predict, read_table, split

PERTURBATIONS = {
    "wind_minus_10_percent": {"cyclone_max_wind_kt": .90},
    "wind_plus_10_percent": {"cyclone_max_wind_kt": 1.10},
    "population_minus_20_percent": {"designated_2020_population": .80},
    "population_plus_20_percent": {"designated_2020_population": 1.20},
    "fire_area_minus_20_percent": {"mtbs_burned_acres": .80},
    "fire_area_plus_20_percent": {"mtbs_burned_acres": 1.20},
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cohort", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cohort = read_table(args.cohort)
    fields = ARMS["verified_link_only"]
    rows = []
    for year in YEARS:
        train, test = split(cohort, year)
        trials = [test]
        for multipliers in PERTURBATIONS.values():
            changed = test.copy()
            for name, ratio in multipliers.items():
                changed[name] *= ratio
            trials.append(changed)
        p, a = fit_predict(train, pd.concat(trials, ignore_index=True), fields, "extra_trees", 6)
        prob = p.reshape(len(trials), len(test), -1)
        amounts = a.reshape(len(trials), len(test))
        for i, (scenario, cols) in enumerate(PERTURBATIONS.items(), 1):
            active = test[list(cols)].notna().all(axis=1).to_numpy()
            active_idx = np.flatnonzero(active)
            result = test.iloc[active_idx][["disasterNumber", "fyDeclared", "incidentType", "actual_band"]].copy()
            result = result.assign(
                scenario=scenario,
                baseline_band=prob[0, active_idx].argmax(axis=1),
                perturbed_band=prob[i, active_idx].argmax(axis=1),
                baseline_amount_usd=amounts[0, active_idx],
                perturbed_amount_usd=amounts[i, active_idx],
            )
            rows.append(result)
    results = pd.concat(rows, ignore_index=True)
    results["band_changed"] = results.baseline_band.ne(results.perturbed_band)
    results["absolute_change_usd"] = (results.perturbed_amount_usd - results.baseline_amount_usd).abs()
    results.to_csv(args.output/"measurement_sensitivity_predictions.csv", index=False)
    summary = results.groupby("scenario").agg(
        affected_cases=("disasterNumber", "size"), band_change_rate=("band_changed", "mean"),
        median_absolute_change_usd=("absolute_change_usd", "median"),
        p90_absolute_change_usd=("absolute_change_usd", lambda values: values.quantile(.9)),
        max_absolute_change_usd=("absolute_change_usd", "max"),
    )
    summary.to_csv(args.output/"measurement_sensitivity_summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
