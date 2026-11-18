#!/usr/bin/env python3
"""Replay sustained observed county-outage trajectories and training-only transfers."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

from thesis_outage_observability import COUNTY_SOURCE, FUTURE_STEPS, PERIOD, THRESHOLD, retain_eaglei_rows, valid_county_rows

YEARS = (2018, 2020, 2022, 2023)
ORIGINS = (2022, 2023)
EDGES = np.array([250, 500, 1000, 2000, 5000, 10000, 50000])
N_STATES = len(EDGES) + 1
THRESHOLD_STATE = np.searchsorted(EDGES, THRESHOLD, side="right")
TAIL_STEPS = 12
N_SYNTHETIC = 8
SYNTHETIC_WEIGHT = .25
PHASE_MASS = 20
PHASES = (slice(0, 4), slice(4, 16), slice(16, FUTURE_STEPS))
REAL = "real_logistic"
HYBRID = "hybrid_logistic"


def raw_slice(year):
    return COUNTY_SOURCE if year == 2018 else f"eaglei_{year}_24_counties.csv.gz"


def reextract(national, raw):
    release = json.loads((raw / "figshare_release_v2.json").read_text())
    assert release["doi"] == "10.6084/m9.figshare.24237376.v2"
    sources = {row["name"]: row for row in release["files"]}
    files = {year: sources[f"eaglei_outages_{year}.csv"] for year in YEARS[1:]}
    assert sum(file["size"] for file in files.values()) < 3_600_000_000
    for year, file in files.items():
        source = raw / raw_slice(year)
        manifest = raw / f"slice_manifest_{year}.json"
        if source.exists() and manifest.exists():
            registered = json.loads(manifest.read_text())
            if registered["annual_md5"] == file["supplied_md5"] and registered["slice_sha256"] == sha256(source.read_bytes()).hexdigest():
                continue
        details = retain_eaglei_rows(national / file["name"], source, file, release)
        manifest.write_text(json.dumps(details, indent=2) + "\n")


def county_spells(year, observations):
    quarter = PERIOD.value
    cooldown = 48 * 4 * quarter
    rows, sequences, summary = [], [], []
    for fips, data in observations:
        times = pd.DatetimeIndex(data.utc).as_unit("ns")
        clock = times.asi8
        counts = data.customers_out.to_numpy()
        entries, accepted, unrecoverable_onset = 0, 0, 0
        last = None
        for i in np.flatnonzero(counts >= THRESHOLD):
            if i == 0 or clock[i] - clock[i - 1] != quarter:
                unrecoverable_onset += 1
                continue
            if counts[i - 1] >= THRESHOLD:
                continue
            entries += 1
            now = clock[i]
            if last is not None and now - last < cooldown:
                continue
            last = now
            accepted += 1
            limit = now + FUTURE_STEPS * quarter
            stop = i + FUTURE_STEPS
            complete = stop < len(clock) and clock[stop] == limit
            record = {"year": year, "fips_code": fips, "onset_utc": times[i], "prior_count": int(counts[i - 1]),
                      "onset_count": int(counts[i]), "observed_complete": complete, "last_3h_below": None}
            if complete:
                values = counts[i:stop + 1]
                record["last_3h_below"] = bool((values[-TAIL_STEPS:] < THRESHOLD).all())
                record["sequence_row"] = len(sequences)
                sequences.append(np.searchsorted(EDGES, values, side="right"))
            rows.append(record)
        summary.append({"year": year, "fips_code": fips, "entry_candidates": entries, "retained_entries": accepted,
                        "high_snapshots_with_unknown_predecessor": unrecoverable_onset})
    return pd.DataFrame(rows), np.array(sequences, dtype=np.int8), pd.DataFrame(summary)


def prepare(raw, exclude_zero_counts=False):
    candidates, data_quality, selection, state_groups = [], [], [], []
    for year in YEARS:
        manifest = json.loads((raw / f"slice_manifest_{year}.json").read_text())
        source = raw / raw_slice(year)
        assert source.name == manifest["slice"] and sha256(source.read_bytes()).hexdigest() == manifest["slice_sha256"]
        clock, counties = valid_county_rows(raw, year=year, source=source.name)
        if exclude_zero_counts:
            counties = [(county, rows[rows.customers_out.ne(0)]) for county, rows in counties]
        spells, states, summary = county_spells(year, counties)
        assert len(states) == spells.observed_complete.sum()
        completeness_column = f"usable_fraction_of_{year}_quarters"
        data_quality.append(clock.rename(columns={completeness_column: "usable_fraction_of_year_quarters"}).assign(year=year))
        observed = spells.loc[spells.observed_complete, "sequence_row"].astype(int)
        spells.loc[spells.observed_complete, "sequence_row"] = observed + sum(map(len, state_groups))
        state_groups.append(states)
        candidates.append(spells)
        selection.append(summary)
    records = pd.concat(candidates, ignore_index=True)
    records["window_id"] = records.year.astype(str) + "-" + records.fips_code + "-" + records.onset_utc.dt.strftime("%Y%m%d%H%M")
    assert records.window_id.is_unique
    return records, np.concatenate(state_groups), pd.concat(data_quality, ignore_index=True), pd.concat(selection, ignore_index=True)


def features(records):
    begin = pd.to_datetime(records.onset_utc.dt.year.astype(str) + "-01-01", utc=True)
    end = begin + pd.offsets.YearBegin()
    fraction = (records.onset_utc - begin) / (end - begin)
    return np.column_stack([np.log1p(records.onset_count), np.log1p(records.prior_count),
                            np.log1p(records.onset_count - records.prior_count),
                            np.sin(2 * np.pi * fraction), np.cos(2 * np.pi * fraction)])


def learn_transitions(training):
    counts = []
    for phase in PHASES:
        left = training[:, :-1][:, phase].astype(int).ravel()
        right = training[:, 1:][:, phase].ravel()
        counts.append(np.bincount(left * N_STATES + right, minlength=N_STATES**2).reshape(N_STATES, N_STATES))
    counts = np.array(counts)
    pooled = counts.sum(axis=0)
    totals = pooled.sum(axis=1, keepdims=True)
    target = np.divide(pooled, totals, out=np.eye(N_STATES), where=totals > 0)
    shrunk = counts + PHASE_MASS * target
    probabilities = shrunk / shrunk.sum(axis=2, keepdims=True)
    return counts, probabilities


def simulate(training, probability, seed):
    rng = np.random.default_rng(seed)
    state = np.repeat(training[:, 0], N_SYNTHETIC)
    donors = np.repeat(np.arange(len(training)), N_SYNTHETIC)
    cumulative = probability.cumsum(axis=2)
    cumulative[:, :, -1] = 1.0
    sustained = np.ones(len(state), dtype=bool)
    transitions = np.zeros((len(PHASES), N_STATES, N_STATES), dtype=np.int64)
    phase_by_step = np.repeat(np.arange(len(PHASES)), [phase.stop - phase.start for phase in PHASES])
    for step, phase in enumerate(phase_by_step):
        nxt = (rng.random(len(state))[:, None] >= cumulative[phase, state]).sum(axis=1)
        np.add.at(transitions[phase], (state, nxt), 1)
        state = nxt
        if step >= FUTURE_STEPS - TAIL_STEPS:
            sustained &= state < THRESHOLD_STATE
    return donors, sustained, transitions


def origin_split(windows, origin):
    start = pd.Timestamp(year=origin, month=1, day=1, tz="UTC")
    training_label_cutoff = start - pd.Timedelta(14, unit="D")
    complete = windows[windows.observed_complete]
    train = complete[complete.year.lt(origin) & (complete.onset_utc + FUTURE_STEPS * PERIOD).lt(training_label_cutoff)]
    test = complete[complete.year.eq(origin)]
    y = test.last_3h_below.astype(bool)
    blocks = ((test.onset_utc.dt.normalize() - start).dt.days // 14).nunique()
    qualifies = len(test) >= 50 and min(int(y.sum()), int((~y).sum())) >= 20 and blocks >= 12
    return train, test, qualifies, blocks


def fit_models(train, test, states, origin, windows):
    scaler = StandardScaler().fit(features(train))
    fit_x = scaler.transform(features(train))
    test_x = scaler.transform(features(test))
    fit_y = train.last_3h_below.astype(bool)
    actual_states = states[train.sequence_row.to_numpy(int)]
    actual_counts, probability = learn_transitions(actual_states)
    donors, generated_y, generated_counts = simulate(actual_states, probability, 80719 + origin)
    unobserved = actual_counts.sum(axis=0) == 0
    new_pair = generated_counts.sum(axis=0) > 0
    represented = actual_counts.sum(axis=(0, 2)) > 0
    assert not (unobserved & new_pair & represented[:, None]).any()
    assert np.isclose(probability.sum(axis=2), 1).all()
    augmented_weights = np.r_[np.ones(len(train)), np.full(len(donors), SYNTHETIC_WEIGHT / N_SYNTHETIC)]
    real = LogisticRegression(C=1, max_iter=1000).fit(fit_x, fit_y)
    hybrid = LogisticRegression(C=1, max_iter=1000).fit(np.vstack([fit_x, fit_x[donors]]),
                                                      np.r_[fit_y.to_numpy(), generated_y], sample_weight=augmented_weights)
    probability_by_model = {
        "pooled_prevalence": np.full(len(test), fit_y.mean()),
        REAL: real.predict_proba(test_x)[:, 1],
        HYBRID: hybrid.predict_proba(test_x)[:, 1],
    }
    predictions = test[["year", "window_id", "fips_code", "onset_utc", "last_3h_below"]].copy()
    for label, p in probability_by_model.items():
        predictions[label] = p
    source_year = pd.Timestamp(year=origin, month=1, day=1, tz="UTC")
    days = (predictions.onset_utc.dt.normalize() - source_year).dt.days
    predictions["block14"] = days // 14
    predictions["block28"] = days // 28
    manifest = train[["year", "window_id", "fips_code", "onset_utc"]].copy()
    manifest["held_origin"] = origin
    manifest["synthetic_trajectories"] = N_SYNTHETIC
    manifest["synthetic_positive_trajectories"] = generated_y.reshape(-1, N_SYNTHETIC).sum(axis=1)
    support = {"year": origin, "train_rows": len(train), "train_positive": int(fit_y.sum()),
               "held_retained_entries": int(windows.year.eq(origin).sum()), "held_complete_rows": len(test),
               "held_positive": int(test.last_3h_below.eq(True).sum()), "synthetic_rows": len(donors),
               "synthetic_positive": int(generated_y.sum()), "synthetic_relative_weight": SYNTHETIC_WEIGHT,
               "generator_seed": 80719 + origin}
    transition_rows = [{"origin": origin, "phase": phase + 1, "from_bin": i, "to_bin": j,
                        "real_training_count": actual_counts[phase, i, j], "generation_probability": probability[phase, i, j],
                        "synthetic_count": generated_counts[phase, i, j]}
                       for phase in range(len(PHASES)) for i in range(N_STATES) for j in range(N_STATES)]
    return predictions, manifest, support, transition_rows


def evaluate(predictions):
    metrics, calibration = [], []
    models = ("pooled_prevalence", REAL, HYBRID)
    for group, rows in [("all_held", predictions), *((str(y), r) for y, r in predictions.groupby("year"))]:
        actual = rows.last_3h_below.astype(int).to_numpy()
        for model in models:
            estimate = rows[model].to_numpy()
            metrics.append({"population": group, "model": model, "rows": len(rows), "positives": int(actual.sum()),
                            "brier": brier_score_loss(actual, estimate), "log_loss": log_loss(actual, estimate),
                            "average_precision": average_precision_score(actual, estimate)})
            bins = np.minimum(4, (estimate * 5).astype(int))
            for i in range(5):
                inside = bins == i
                if inside.any():
                    calibration.append({"population": group, "model": model, "bin_lower": i / 5, "bin_upper": (i + 1) / 5,
                                        "rows": int(inside.sum()), "mean_probability": estimate[inside].mean(),
                                        "observed_rate": actual[inside].mean()})
    return pd.DataFrame(metrics), pd.DataFrame(calibration)


def paired_bootstrap(predictions):
    actual = predictions.last_3h_below.astype(int)
    loss_delta = (predictions[HYBRID] - actual)**2 - (predictions[REAL] - actual)**2
    rows = predictions.assign(delta=loss_delta)
    results = []
    for days in (14, 28):
        key = f"block{days}"
        rng = np.random.default_rng(29117)
        combined = np.zeros((4000, 2))
        n_blocks = 0
        for _, group in rows.groupby("year"):
            paired = group.groupby(key).delta.agg(["sum", "size"]).to_numpy()
            n = len(paired)
            n_blocks += n
            combined += paired[rng.integers(0, n, size=(len(combined), n))].sum(axis=1)
        lo, hi = np.quantile(combined[:, 0] / combined[:, 1], [.025, .975])
        results.append({"block_days": days, "held_blocks": n_blocks, "hybrid_minus_real_brier": loss_delta.mean(),
                        "paired_95_lower": lo, "paired_95_upper": hi})
    return pd.DataFrame(results)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("raw", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--national", type=Path, help="Verify and extract source CSVs for 2020, 2022 and 2023")
    cli.add_argument("--exclude-zero-counts", action="store_true", help="Sensitivity to unresolved provenance of explicit zeros")
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.national:
        reextract(args.national, args.raw)
    windows, states, quality, selection = prepare(args.raw, args.exclude_zero_counts)
    windows.drop(columns="sequence_row").to_csv(args.output / "window_outcomes.csv", index=False)
    quality.to_csv(args.output / "county_clock_audit.csv", index=False)
    selection.to_csv(args.output / "entry_selection.csv", index=False)
    scores, membership, support, transition = [], [], [], []
    for year in ORIGINS:
        train, test, gate, blocks = origin_split(windows, year)
        assert gate, f"Insufficient held observation support in {year}: {len(test)} rows, {blocks} blocks"
        assert len(train) >= 50 and train.last_3h_below.nunique() == 2
        p, m, s, t = fit_models(train, test, states, year, windows)
        scores.append(p)
        membership.append(m)
        support.append({**s, "held_14_day_blocks": blocks})
        transition.extend(t)
    held = pd.concat(scores, ignore_index=True)
    metrics, calibration = evaluate(held)
    results = {"held_predictions": held, "origin_support": pd.DataFrame(support), "scores": metrics,
               "calibration": calibration, "paired_brier_intervals": paired_bootstrap(held),
               "training_trajectory_membership": pd.concat(membership, ignore_index=True),
               "transition_support": pd.DataFrame(transition)}
    for name, table in results.items():
        table.to_csv(args.output / f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
