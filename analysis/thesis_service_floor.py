#!/usr/bin/env python3
"""Test inventory-feasible service floors under matched sequential decisions."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from thesis_service_scenarios import (ANCHORS, FAMILIES, audit_anchors, dispatch, proportional, recourse,
                                      scenario_inputs, summarized, verify)

FLOOR = .70
PRIORITIES = ("myopic", "expected", "cvar")
METRICS = ("served_fraction", "high_vulnerability_served_fraction", "lowest_zone_fraction")


def run_case(phase, family, case, held_seed, plan_seed, stock_ratio):
    shelter, _, _, inputs = scenario_inputs(phase, family, case, held_seed, plan_seed, stock_ratio)
    rows = []
    for _, first, second, opening, delivery, future, planning_stock in inputs:
        policies = [("equal_fraction", 0.)] + [(priority, minimum) for priority in PRIORITIES
                                               for minimum in (0., FLOOR)]
        for policy, floor in policies:
            equal = policy == "equal_fraction"
            if policy in ("myopic", "equal_fraction"):
                x = dispatch(first, opening, equal, floor)
            else:
                x = recourse(first, opening, future, planning_stock, policy == "cvar", floor)
            available = max(0, opening - x.sum()) + delivery
            y = dispatch(second, available, equal, floor)
            verify(first, second, x, y, opening, available)
            assert (x >= floor * proportional(first, opening) - 1e-6).all()
            assert (y >= floor * proportional(second, available) - 1e-6).all()
            rows.extend((policy, floor, zone, need, served)
                        for zone, (need, served) in enumerate(zip(first + second, x + y, strict=True)))
    frame = pd.DataFrame(rows, columns=["policy", "floor", "zone", "need", "served"])
    scores = [{"policy": policy, "floor": floor, **summarized(part)}
              for (policy, floor), part in frame.groupby(["policy", "floor"], sort=False)]
    return shelter, scores


def paired(scores):
    rng = np.random.default_rng(92219)
    decisions = scores[scores.policy.isin(PRIORITIES)]
    keys = ["phase", "family", "nominal_supply"]
    results = []
    for label, group in [("pooled", decisions), *decisions.groupby("nominal_supply")]:
        wide = group.pivot(index=[*keys, "case"], columns=["policy", "floor"], values=list(METRICS))
        strata = list(wide.groupby(level=[0, 1, 2]).indices.values())
        sampled = np.concatenate([rng.choice(idx, size=(2000, len(idx))) for idx in strata], axis=1)
        for policy in PRIORITIES:
            for metric in METRICS:
                deltas = (wide[(metric, policy, FLOOR)] - wide[(metric, policy, 0.)]).to_numpy()
                low, high = np.quantile(deltas[sampled].mean(axis=1), [.025, .975])
                results.append((label, policy, metric, deltas.mean(), low, high, len(deltas)))
    return pd.DataFrame(results, columns=["nominal_supply", "policy", "metric", "paired_difference", "lower_95",
                                          "upper_95", "scenarios"])


def simulate(pdf, output):
    audit_anchors(pdf)
    output.mkdir(parents=True, exist_ok=True)
    rows, registry = [], []
    for p, phase in enumerate(ANCHORS):
        for f, family in enumerate(FAMILIES):
            for case in range(48):
                num = 1000 * p + 100 * f + case
                held_seed, plan_seed = 148000 + num, 196000 + num
                supply = (.55, .72, .90)[case // 3 % 3]
                shelter, scores = run_case(phase, family, case, held_seed, plan_seed, supply)
                key = dict(phase=phase, family=family, case=case, nominal_supply=supply, shelter=shelter)
                registry.append({**key, "held_seed": held_seed, "planning_seed": plan_seed})
                rows.extend({**key, **value} for value in scores)
    register = pd.DataFrame(registry)
    assert register.held_seed.is_unique and register.planning_seed.is_unique
    assert not set(register.held_seed) & set(register.planning_seed)
    assert register[["phase", "family", "nominal_supply", "shelter"]].drop_duplicates().shape[0] == 36
    scored = pd.DataFrame(rows)
    for name, table in (("scenario_registry", register), ("scenario_metrics", scored), ("paired_floors", paired(scored))):
        table.to_csv(output / f"{name}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("casper_pdf", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    simulate(args.casper_pdf, args.output)
