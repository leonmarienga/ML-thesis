#!/usr/bin/env python3
"""Audit the augmentation ledgers and check an optional replay against the archive."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from thesis_augmentation_generators import CATEGORY, CLASS_ONLY, DELAY, key_seed
from thesis_forward_models import read_table
from thesis_hazard_features import CUTS


def verify(directory):
    cohort = read_table(Path("research/results/hazards/cohort_features.csv")).set_index("disasterNumber")
    membership = pd.read_csv(directory / "membership.csv.gz")
    registry = pd.read_csv(directory / "registry.csv.gz")
    rows = pd.read_csv(directory / "synthetic.csv.gz")
    study = pd.read_csv(directory / "predictions.csv.gz")
    inner = pd.read_csv(directory / "inner_predictions.csv.gz")
    for _, part in membership.groupby("fold"):
        train = part[part.role.eq("train")]
        held = part[part.role.eq("real_holdout")]
        assert train.fyDeclared.max() < held.fyDeclared.min()
        assert set(train.event_group).isdisjoint(held.event_group)
        assert set(train.disasterNumber).isdisjoint(held.disasterNumber)
    assert membership.fyDeclared.max() == study.fyDeclared.max() == 2024
    for origin in range(2019, 2025):
        earlier = inner[inner.fold.str.startswith(f"{origin}/") & inner.used_for_selection_or_calibration]
        calibrating = earlier[earlier.fold.str.contains("/calibration/")]
        tuning = earlier[earlier.fold.str.contains("/tuning/")]
        assert set(calibrating.event_group).isdisjoint(tuning.event_group)
        assert set(earlier.event_group).isdisjoint(study[study.fyDeclared.eq(origin)].event_group)
    assert not study.duplicated(["variant", "disasterNumber"]).any()
    assert rows.synthetic_id.is_unique
    assert not rows.uses_real_fema_identifier.any()
    assert not rows.county_or_coordinates_synthesized.any()
    registry_by_trial = registry.groupby("trial_id")
    train_membership = membership[membership.role.eq("train")].groupby("fold")
    for trial, generated in rows.groupby("trial_id"):
        fold = generated.fold.iloc[0]
        permitted = set(train_membership.get_group(fold).disasterNumber)
        method = generated.method.iloc[0]
        line = registry_by_trial.get_group(trial)
        assert line.accepted.sum() == len(generated)
        for record in generated.to_dict("records"):
            a, b, cat = (record[key] for key in ["source_a", "source_b", "category_source"])
            assert {a, b, cat} <= permitted
            idx = record["synthetic_funding_band_index"]
            assert cohort.loc[[a, b, cat], "actual_band"].eq(idx).all()
            for column in CATEGORY:
                assert record[column] == cohort.loc[cat, column]
            assert record["episode_a"] == cohort.loc[a, "event_group"]
            assert record["episode_b"] == cohort.loc[b, "event_group"]
            amount = record["synthetic_mission_obligation_proxy_usd"]
            if method == "random":
                assert a == b == cat and record["feature_duplicate"] and record["joint_duplicate"]
                assert np.isclose(amount, cohort.loc[a, "target"], rtol=0., atol=1e-5)
                stratum = line[line.band.eq(idx)].iloc[0]
            else:
                assert method == "copula" or record["episode_a"] != record["episode_b"]
                assert cohort.loc[[a, b, cat], "state"].eq(record["state"]).all()
                assert cohort.loc[[a, b, cat], "incidentType"].eq(record["incidentType"]).all()
                stratum = line[line.band.eq(idx) & line.state.eq(record["state"]) & line.hazard.eq(record["incidentType"])].iloc[0]
                assert not record["joint_duplicate"]
            sources = set(map(int, stratum.source_ids.split("|")))
            assert {a, b, cat} <= sources <= permitted
            assert record["seed"] == key_seed(f"{stratum.seed_namespace}|{method}|{idx}|{stratum.state}|{stratum.hazard}")
            assert stratum.training_delay_min-1e-10 <= record[DELAY] <= stratum.training_delay_max+1e-10
            if method in CLASS_ONLY:
                assert pd.isna(amount) and not record["feature_duplicate"]
            else:
                assert int(np.searchsorted(CUTS, amount, side="right")) == idx
                assert stratum.training_target_min-1e-5 <= amount <= stratum.training_target_max+1e-5
    assert registry.accepted.sum() == len(rows)
    print(f"Verified {len(rows):,} generated rows across {registry.trial_id.nunique():,} fold/configuration trials.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    verify(args.results)
    if args.reference:
        outputs = sorted(args.reference.glob("*.csv")) + sorted(args.reference.glob("*.csv.gz"))
        outputs = [file for file in outputs if not file.name.startswith("physics_")]
        for file in outputs:
            pd.testing.assert_frame_equal(pd.read_csv(args.results / file.name), pd.read_csv(file),
                                          check_exact=False, rtol=1e-6, atol=1e-6)
        print(f"Replayed {len(outputs)} augmentation ledgers and metric tables.")


if __name__ == "__main__":
    main()
