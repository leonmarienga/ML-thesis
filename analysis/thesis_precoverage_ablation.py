#!/usr/bin/env python3
"""Separate public-ledger coverage from a recorded zero in past-only funding fits."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.preprocessing import OneHotEncoder

from thesis_augmentation_generators import CATEGORY, DELAY, PREDICTORS
from thesis_augmentation_study import raw_fit
from thesis_forward_models import YEARS, estimators, read_table, score, split

COVERAGE_START = pd.Timestamp("2012-10-01")
THRESHOLDS = ((3, "50m"), (4, "200m"), (5, "500m"))
SETTINGS = [("real", 1), ("weight", 20)]
SEED = 27812


def positive_probabilities(train, hold):
    encoder = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORY),
        ("delay", "passthrough", [DELAY]),
    ])
    classifier, _ = estimators("extra_trees", 6)
    classifier.set_params(class_weight=None)
    features = encoder.fit_transform(train[PREDICTORS])
    classifier.fit(features, train.target.gt(0))
    return classifier.predict_proba(encoder.transform(hold[PREDICTORS]))[:, 1]


def fit(cohort):
    cases, membership = [], []
    for year in YEARS:
        training, hold = split(cohort, year)
        eligible = training.declaration.ge(COVERAGE_START)
        assert training.loc[~eligible, "target"].eq(0).all()
        for policy, train in (("archive_all", training), ("coverage_screened", training[eligible])):
            identity = f"{year}/{policy}/{year}"
            d, _, _, roster = raw_fit(train, hold, identity, configs=SETTINGS)
            pred_positive = pd.Series(positive_probabilities(train, hold), index=hold.disasterNumber)
            cases.append(d.assign(training_policy=policy, predicted_positive=d.disasterNumber.map(pred_positive)))
            membership.append(roster)
    return pd.concat(cases, ignore_index=True), pd.concat(membership, ignore_index=True)


def metrics(frame):
    p = frame[[f"p{i}" for i in range(6)]].to_numpy()
    amount = frame.amount.to_numpy()
    proper = score(frame, amount, p)
    err = abs(amount-frame.target.to_numpy())
    result = {}
    for band, label in THRESHOLDS:
        actual = frame.actual_band.ge(band)
        risk = p[:, band:].sum(axis=1)
        flagged = risk >= .10
        result[f"cases_{label}"] = int(actual.sum())
        result[f"ap_{label}"] = average_precision_score(actual, risk) if actual.any() else np.nan
        result[f"brier_{label}"] = brier_score_loss(actual, risk)
        result[f"mae_{label}_usd"] = err[actual].mean() if actual.any() else np.nan
        result[f"alerts_true_{label}"] = int((actual & flagged).sum())
        result[f"alerts_false_{label}"] = int((~actual & flagged).sum())
    positive = frame.target.gt(0)
    predicted = frame.predicted_positive
    result["positive_ap"] = average_precision_score(positive, predicted)
    result["positive_brier"] = brier_score_loss(positive, predicted)
    result["mean_prob_positive"] = predicted.mean()
    result["observed_positive_fraction"] = positive.mean()
    return {**proper, **result}


def paired(cases, draws=1000):
    result = []
    setup = cases[cases.training_policy.eq("archive_all") & cases.config.eq("real_1x")].sort_values("disasterNumber")
    groups = list(setup.reset_index(drop=True).groupby("event_group").indices.values())
    use = np.random.default_rng(SEED).integers(0, len(groups), size=(draws, len(groups)))
    indexes = [np.concatenate([groups[j] for j in row]) for row in use]
    indexes = [np.arange(len(setup)), *indexes]

    def summarize(frame):
        p = frame[[f"p{i}" for i in range(6)]].to_numpy()
        bands = frame.actual_band.to_numpy()
        err = abs(frame.amount.to_numpy() - frame.target.to_numpy())
        tail = bands >= 3
        positive = frame.target.to_numpy() > 0
        risk = p[:, 3:].sum(axis=1)
        pos_risk = frame.predicted_positive.to_numpy()

        def calculate(i):
            high = tail[i]
            return [
                (p[i].argmax(axis=1) == bands[i]).mean(),
                err[i].mean(),
                err[i][high].mean() if high.any() else np.nan,
                average_precision_score(high, risk[i]) if high.any() else np.nan,
                ((pos_risk[i] - positive[i])**2).mean(),
            ]

        return np.array([calculate(i) for i in indexes])

    for config, case in cases.groupby("config"):
        earlier = case[case.training_policy.eq("archive_all")].sort_values("disasterNumber")
        screened = case[case.training_policy.eq("coverage_screened")].sort_values("disasterNumber")
        assert np.array_equal(screened.disasterNumber, earlier.disasterNumber)
        difference = summarize(screened) - summarize(earlier)
        metrics_to_compare = ("accuracy", "mae_usd", "mae_50m_usd", "ap_50m", "positive_brier")
        for metric, value, sample in zip(metrics_to_compare, difference[0], difference[1:].T, strict=True):
            lo, hi = np.nanquantile(sample, [.025, .975])
            result.append({"config": config, "metric": metric, "difference": value, "p025": lo, "p975": hi,
                           "episode_groups": len(groups), "draws": int(np.isfinite(sample).sum())})
    return pd.DataFrame(result)


def verify_reference(cases):
    archive = pd.read_csv("research/augmentation/results/predictions.csv.gz")
    replacement = cases[cases.training_policy.eq("archive_all")].set_index(["config", "disasterNumber"])
    old = archive[archive.variant.isin(replacement.index.get_level_values("config"))].set_index(["config", "disasterNumber"])
    assert len(old) == len(replacement) == 752
    compared = ["amount", *(f"p{i}" for i in range(6))]
    pd.testing.assert_frame_equal(replacement[compared].sort_index(), old[compared].sort_index(),
                                  check_exact=False, rtol=1e-6, atol=1e-6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cohort = read_table(Path("research/results/hazards/cohort_features.csv"))
    assert cohort.fyDeclared.max() == 2024
    cases, membership = fit(cohort)
    verify_reference(cases)
    train = membership[membership.role.eq("train")].merge(cohort[["disasterNumber", "target", "declaration"]],
                on="disasterNumber", validate="many_to_one")
    train["before_coverage"] = train.declaration.lt(COVERAGE_START)
    train["positive"] = train.target.gt(0)
    train["high_group"] = train.event_group.where(train.target.ge(50e6))
    support = train.groupby("fold").agg(rows=("disasterNumber", "size"), episodes=("event_group", "nunique"),
         precoverage_rows=("before_coverage", "sum"), positive_rows=("positive", "sum"),
         high_episodes=("high_group", "nunique")).reset_index()
    metrics_by_group = [{"policy": policy, "config": config, "scope": scope, **metrics(frame)}
        for (policy, config), group in cases.groupby(["training_policy", "config"])
        for scope, frame in [("pooled", group), *group.groupby("fyDeclared")]]
    tables = {
        "metrics.csv": pd.DataFrame(metrics_by_group),
        "paired_episode_bootstrap.csv": paired(cases),
        "training_support.csv": support,
        "predictions.csv.gz": cases,
        "membership.csv.gz": membership,
    }
    for name, frame in tables.items():
        compression = {"method": "gzip", "mtime": 0} if name.endswith(".gz") else None
        frame.to_csv(args.output / name, index=False, compression=compression)
    print("Reproduced both archived controls and evaluated the source-coverage policy on 376 cases.")


if __name__ == "__main__":
    main()
