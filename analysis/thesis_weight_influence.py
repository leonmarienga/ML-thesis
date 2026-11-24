#!/usr/bin/env python3
"""Replay fixed signed-endpoint weights under prespecified training episode omissions."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from thesis_augmentation_study import raw_fit
from thesis_coverage_specialists import REAL, TAIL, WEIGHTED, archive, get_split, paired, summarize
from thesis_forward_models import YEARS


def fitted(train, held, origin, rank):
    estimates, _, _, _ = raw_fit(train, held, f"signed_elapsed/{origin}/outer", configs=[("real", 1), ("weight", 20)])
    return estimates.assign(arm=estimates.config.map({"real_1x": REAL, "weight_20x": WEIGHTED}),
                            risk=estimates[["p3", "p4", "p5"]].sum(axis=1), rank=rank)


def compare(old):
    history = archive()
    held = old[old.endpoint.eq("signed_elapsed") & old.arm.isin([REAL, WEIGHTED])].copy()
    all_results, exclusions = [], []
    for origin in YEARS:
        train, cases = get_split(history, origin, "signed_elapsed")
        if cases.empty:
            continue
        ordered = train[train.target.ge(TAIL)].groupby("event_group", as_index=False).target.max()
        groups = ordered.sort_values(["target", "event_group"], ascending=[False, True]).head(3)
        initial = fitted(train, cases, origin, 0)
        ref = held[held.fyDeclared.eq(origin)].set_index(["disasterNumber", "arm"])
        match = initial.set_index(["disasterNumber", "arm"]).sort_index()
        assert match.index.equals(ref.sort_index().index)
        for metric in ("amount", "risk"):
            assert np.allclose(match[metric], ref.sort_index()[metric], rtol=1e-7, atol=1e-7)
        all_results.append(initial)
        for rank in (1, 2, 3):
            omitted = groups.iloc[rank - 1].event_group if len(groups) >= rank else None
            drop = train.event_group.eq(omitted)
            exclusions.append({"held_fy": origin, "rank": rank, "omitted_event": omitted,
                               "removed_declarations": int(drop.sum()),
                               "train_groups_after_removal": train.loc[~drop, "event_group"].nunique()})
            if omitted is None:
                all_results.append(initial.assign(rank=rank))
            else:
                all_results.append(fitted(train[~drop], cases, origin, rank))
    columns = ["disasterNumber", "event_group", "fyDeclared", "target", "arm", "amount", "risk", "active", "rank"]
    return pd.concat(all_results, ignore_index=True)[columns], pd.DataFrame(exclusions), held


def report(arms):
    rows = []
    for (rank, arm), frame in arms.groupby(["rank", "arm"]):
        losses = frame.assign(loss=abs(frame.target - frame.amount))
        tails = losses[losses.target.ge(TAIL)]
        rows.append({"rank": rank, "arm": arm, **summarize(frame),
                     "episode_mae": losses.groupby("event_group").loss.mean().mean(),
                     "episode_tail_mae": tails.groupby("event_group").loss.mean().mean()})
    summary = pd.DataFrame(rows)
    base = summary[summary.arm.eq(REAL)].set_index("rank")
    specialist = summary[summary.arm.eq(WEIGHTED)].set_index("rank")
    metrics = ["mae", "tail_mae", "episode_mae", "episode_tail_mae"]
    return summary, (specialist[metrics] - base[metrics]).reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    old = pd.read_csv("research/continuous/coverage_specialists/results/predictions.csv")
    predictions, exclusions, held = compare(old)
    summary, sensitivity = report(predictions)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, result in [("predictions", predictions), ("omitted_groups", exclusions), ("metrics", summary),
                         ("paired_differences", sensitivity), ("original_bootstrap", paired(held))]:
        result.to_csv(args.output / (name + ".csv"), index=False)


if __name__ == "__main__":
    main()
