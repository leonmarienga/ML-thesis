#!/usr/bin/env python3
"""Audit outcome observability and replay nested coverage-aware funding forecasts."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from thesis_augmentation_generators import CATEGORY, DELAY, PREDICTORS
from thesis_augmentation_study import raw_fit
from thesis_forward_models import YEARS, estimators, read_table, split
from thesis_hazard_features import CUTS
from thesis_precoverage_ablation import COVERAGE_START

TAIL = 50_000_000
REAL = "real_only"
WEIGHTED = "weighted_20x"
HURDLE = "positive_hurdle"
GATES = {"hazard_gate_10": .10, "hazard_gate_20": .20}
ENDPOINTS = ("eventual_proxy", "signed_elapsed")


def archive():
    cases = read_table(Path("research/results/hazards/cohort_features.csv"))
    assert cases.fyDeclared.max() == 2024
    cols = ["disasterNumber", "snapshot_status", "horizon_36m_status", "release_review_date",
            "net_through_36m_usd", "first_obligation_utc", "fixed_36m_end"]
    d = cases.merge(pd.read_csv("research/continuous/horizons/results/historical_horizons.csv", usecols=cols),
                    on="disasterNumber", how="left", validate="one_to_one")
    d["release_review_date"] = pd.to_datetime(d.release_review_date, utc=True)
    signed = d.net_through_36m_usd.round(2)
    elapsed = d.horizon_36m_status.eq("elapsed_under_current_archive")
    assert signed[elapsed].ge(0).all()
    recorded_zero = elapsed & signed.eq(0)
    assert pd.to_datetime(d.loc[recorded_zero, "first_obligation_utc"], utc=True).le(
        pd.to_datetime(d.loc[recorded_zero, "fixed_36m_end"], utc=True)
    ).all()
    d["observation_status"] = np.select(
        [d.declaration.lt(COVERAGE_START), d.snapshot_status.eq("no_public_rows"),
         d.snapshot_status.eq("missing_obligation_date"),
         elapsed & signed.eq(0), elapsed & signed.gt(0)],
        ["precoverage_unavailable", "in_scope_no_public_actions", "unusable_action_dates",
         "signed_elapsed_net_zero", "signed_elapsed_positive"],
        default="horizon_pending",
    )
    assert d.observation_status.eq("precoverage_unavailable").sum() == 223
    assert d.loc[d.observation_status.eq("precoverage_unavailable"), "target"].eq(0).all()
    d["signed_target"] = signed.where(elapsed)
    return d


def get_split(d, year, endpoint):
    train, held = split(d, year)
    cutoff = min(held.start.min(), held.declaration.min()).tz_localize("UTC")
    if endpoint == "eventual_proxy":
        train = train[train.declaration.ge(COVERAGE_START)].copy()
    else:
        train = train[train.signed_target.notna() & train.release_review_date.lt(cutoff)].copy()
        held = held[held.signed_target.notna()].copy()
        for frame in (train, held):
            frame["target"] = frame.signed_target
            frame["actual_band"] = np.searchsorted(CUTS, frame.target, side="right")
    return train, held


def counts(frame):
    return frame.event_group.nunique(), frame.loc[frame.target.ge(TAIL), "event_group"].nunique()


def fit_arms(train, held, key):
    old, _, _, _ = raw_fit(train, held, key, configs=[("real", 1), ("weight", 20)])
    base = held[["disasterNumber", "event_group", "fyDeclared", "target", "observation_status"]].reset_index(drop=True)

    def result(name, amount, risk, active=True):
        return base.assign(arm=name, amount=amount, risk=risk, active=active)

    arrays = {}
    for config, name in [("real_1x", REAL), ("weight_20x", WEIGHTED)]:
        rows = old[old.config.eq(config)].set_index("disasterNumber").reindex(base.disasterNumber)
        assert rows.amount.notna().all()
        arrays[name] = result(name, rows.amount.to_numpy(), rows[["p3", "p4", "p5"]].sum(axis=1).to_numpy())

    transform = ColumnTransformer([
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORY),
        ("delay", StandardScaler(), [DELAY]),
    ])
    a, b = transform.fit_transform(train[PREDICTORS]), transform.transform(held[PREDICTORS])
    positives = train.target.gt(0).to_numpy()
    classifier, _ = estimators("extra_trees", 6)
    classifier.set_params(class_weight=None)
    p = (classifier.fit(a, positives).predict_proba(b)[:, 1] if len(np.unique(positives)) == 2
         else np.full(len(held), float(positives.mean())))
    _, reg = estimators("extra_trees", 6)
    reg.fit(a[positives], np.log1p(train.target.to_numpy()[positives]))
    amount = p * np.expm1(reg.predict(b))
    arrays[HURDLE] = result(HURDLE, amount, arrays[REAL].risk)
    rare = train[train.target.ge(TAIL)]
    if counts(train)[1] >= 8:
        model = LogisticRegression(C=1, max_iter=500).fit(a, train.target.ge(TAIL))
        risk = model.predict_proba(b)[:, 1]
        pooled = np.log1p(rare.groupby("event_group").target.median()).median()
        by_hazard = np.log1p(rare.groupby(["incidentType", "event_group"]).target.median()).groupby(level=0)
        ns = by_hazard.size()
        fitted = (ns * by_hazard.median() + 8 * pooled) / (ns + 8)
        tail_amount = np.expm1(held.incidentType.map(fitted).fillna(pooled).to_numpy())
        for name, threshold in GATES.items():
            pred = np.where(risk >= threshold, .5 * arrays[REAL].amount + .5 * tail_amount, arrays[REAL].amount)
            arrays[name] = result(name, pred, risk)
    else:
        for name in GATES:
            arrays[name] = arrays[REAL].assign(arm=name, active=False)
    return pd.concat(arrays.values(), ignore_index=True)


def summarize(rows):
    y = rows.target.to_numpy()
    p = rows.risk.to_numpy()
    error = abs(rows.amount.to_numpy() - y)
    tail = y >= TAIL
    return {"n": len(y), "tail_n": int(tail.sum()), "tail_groups": rows.loc[tail, "event_group"].nunique(),
            "mae": error.mean(), "tail_mae": error[tail].mean() if tail.any() else np.nan,
            "rmse": np.sqrt((error**2).mean()), "tail_ap": average_precision_score(tail, p) if tail.any() else np.nan,
            "tail_brier": brier_score_loss(tail, p), "alerts_true_10pct": int((tail & (p >= .1)).sum()),
            "alerts_false_10pct": int((~tail & (p >= .1)).sum()), "active_cases": int(rows.active.sum())}


def choose(d, outer_train, outer, endpoint):
    collected = []
    for history in range(2016, outer):
        train, validation = get_split(d, history, endpoint)
        permitted = set(outer_train.disasterNumber)
        train = train[train.disasterNumber.isin(permitted)]
        validation = validation[validation.disasterNumber.isin(permitted)]
        if len(train) < 25 or len(validation) < 8 or not train.target.gt(0).any():
            continue
        assert not set(train.event_group) & set(validation.event_group)
        assert train.fyDeclared.max() < validation.fyDeclared.min()
        collected.append(fit_arms(train, validation, f"{endpoint}/{outer}/inner/{history}").assign(origin=history))
    if not collected:
        return REAL, []
    observations = pd.concat(collected, ignore_index=True)
    baseline = observations[observations.arm.eq(REAL)].set_index("disasterNumber")
    records = []
    for name, frame in observations[observations.arm.ne(REAL) & observations.active].groupby("arm"):
        base = baseline.loc[frame.disasterNumber]
        out, ref = summarize(frame), summarize(base)
        num_origins = frame.origin.nunique()
        alert_gate = name not in GATES or frame.risk.ge(GATES[name]).any()
        accepted = (num_origins >= 2 and out["tail_groups"] >= 5 and alert_gate and
                    out["tail_mae"] <= .95 * ref["tail_mae"] and out["mae"] <= 1.05 * ref["mae"])
        records.append({"endpoint": endpoint, "outer_origin": outer, "arm": name, "inner_origins": num_origins,
                        "inner_validation_n": len(frame), "inner_tail_groups": out["tail_groups"],
                        "inner_mae": out["mae"], "inner_tail_mae": out["tail_mae"], "reference_mae": ref["mae"],
                        "reference_tail_mae": ref["tail_mae"], "selected_by_gate": accepted})
    eligible = [r for r in records if r["selected_by_gate"]]
    return (min(eligible, key=lambda r: (r["inner_tail_mae"], r["arm"]))["arm"] if eligible else REAL), records


def evaluate(d):
    results, choices, selections, support = [], [], [], []
    for endpoint in ENDPOINTS:
        for year in YEARS:
            train, held = get_split(d, year, endpoint)
            _, tails = counts(train)
            strata = train[train.target.ge(TAIL)].groupby(["state", "incidentType", "actual_band"]).event_group.nunique()
            support.append({"endpoint": endpoint, "held_fy": year, "train_n": len(train), "held_n": len(held),
                            "train_tail_groups": tails, "held_tail_groups": counts(held)[1],
                            "largest_compatible_tail_stratum": int(strata.max()) if len(strata) else 0,
                            "same_strata_with_eight_groups": int(strata.ge(8).sum()),
                            **{f"training_groups_band_{band}": train.loc[train.actual_band.eq(band),
                                                                        "event_group"].nunique()
                               for band in (3, 4, 5)}})
            if held.empty:
                continue
            selected, inner = choose(d, train, year, endpoint)
            predictions = fit_arms(train, held, f"{endpoint}/{year}/outer")
            chosen = predictions[predictions.arm.eq(selected)].assign(arm="nested_selected")
            assert chosen.active.all()
            results.append(pd.concat([predictions, chosen], ignore_index=True).assign(endpoint=endpoint))
            choices.append({"endpoint": endpoint, "held_fy": year, "selected_arm": selected})
            selections.extend(inner)
    return pd.concat(results, ignore_index=True), pd.DataFrame(choices), pd.DataFrame(selections), pd.DataFrame(support)


def metrics(predictions):
    table = []
    for (endpoint, arm), rows in predictions.groupby(["endpoint", "arm"]):
        populations = [("pooled", rows), *((str(y), r) for y, r in rows.groupby("fyDeclared"))]
        for name, sample in populations:
            table.append({"endpoint": endpoint, "arm": arm, "population": name, **summarize(sample)})
    return pd.DataFrame(table)


def paired(predictions):
    results = []
    for endpoint, part in predictions.groupby("endpoint"):
        base = part[part.arm.eq(REAL)].sort_values("disasterNumber")
        groups = list(base.reset_index(drop=True).groupby("event_group").indices.values())
        rng = np.random.default_rng(20261125)
        choices = rng.integers(0, len(groups), size=(2000, len(groups)))
        samples = [np.concatenate([groups[j] for j in row]) for row in choices]
        tail = base.target.to_numpy() >= TAIL
        for name, arm in part[part.arm.ne(REAL)].groupby("arm"):
            arm = arm.sort_values("disasterNumber")
            assert np.array_equal(arm.disasterNumber, base.disasterNumber)
            delta = abs(arm.amount.to_numpy() - base.target.to_numpy()) - abs(base.amount.to_numpy() - base.target.to_numpy())
            for metric, selected in [("mae", np.ones(len(tail), dtype=bool)), ("tail_mae", tail)]:
                diffs = [delta[idx][selected[idx]].mean() for idx in samples if selected[idx].any()]
                low, high = np.quantile(diffs, [.025, .975])
                results.append({"endpoint": endpoint, "arm": name, "metric": metric, "change": delta[selected].mean(),
                                "lower_95": low, "upper_95": high, "draws": len(diffs), "pooled_episode_groups": len(groups)})
    return pd.DataFrame(results)


def verify_retained(predictions):
    old = pd.read_csv("research/continuous/coverage/ablation/predictions.csv.gz")
    old = old[old.training_policy.eq("coverage_screened")].sort_values("disasterNumber")
    for config, arm in [("real_1x", REAL), ("weight_20x", WEIGHTED)]:
        current = predictions[predictions.endpoint.eq("eventual_proxy") & predictions.arm.eq(arm)].sort_values("disasterNumber")
        reference = old[old.config.eq(config)]
        assert np.array_equal(current.disasterNumber, reference.disasterNumber)
        assert np.allclose(current.amount, reference.amount, rtol=1e-7, atol=1e-6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = archive()
    predictions, selections, inner, support = evaluate(data)
    verify_retained(predictions)
    statuses = data[["disasterNumber", "fyDeclared", "event_group", "observation_status", "signed_target", "release_review_date"]]
    for name, table in [("label_observability", statuses), ("predictions", predictions), ("scores", metrics(predictions)),
                        ("nested_choices", selections), ("inner_selection", inner), ("fold_support", support),
                        ("paired_changes", paired(predictions))]:
        table.to_csv(args.output / f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
