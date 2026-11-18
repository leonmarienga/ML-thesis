#!/usr/bin/env python3
"""Simulate constrained incremental allocation with independent synthetic need."""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261117
POLICIES = ("equal", "proportional_need", "severity_priority", "exposure_priority",
            "efficient_expected", "equity_expected", "equity_robust")
HAZARDS = np.array(["flood", "cyclone", "wildfire", "storm", "earthquake", "winter"])


def capped_proportional(weight, caps, budget):
    """Redistribute excess shares until demand caps or the budget bind."""
    allocation = np.zeros(len(caps))
    available = caps > 1e-8
    while available.any() and budget > 1e-8:
        shares = weight[available]
        shares = shares/shares.sum() if shares.sum() else np.ones(len(shares))/len(shares)
        increment = np.minimum(caps[available]-allocation[available], budget*shares)
        if increment.sum() <= 1e-8:
            break
        allocation[available] += increment
        budget -= increment.sum()
        available = caps-allocation > 1e-8
    return allocation


def priority(scores, caps, budget):
    assignment = np.zeros(len(caps))
    for i in np.argsort(-scores, kind="stable"):
        assignment[i] = min(caps[i], budget)
        budget -= assignment[i]
    return assignment


def risk_allocation(assigned, mean, vulnerability, budget, floor, robust):
    """Allocate to the highest expected marginal unmet-need reductions."""
    if robust:
        factors, probs = np.array([.65, 1, 1.55]), np.array([.1, .35, .55])
    else:
        factors, probs = np.array([.65, 1, 1.55]), np.array([.25, .5, .25])
    scenarios = mean[:, None] * factors
    caps = (scenarios[:, -1] - assigned).clip(0)
    # Common planned service floor with a bounded share of this wave budget.
    floor_need = (.65 * mean - assigned).clip(0)
    baseline = capped_proportional(floor_need, floor_need, budget*floor)
    used = baseline.copy()
    segments = []
    for i, levels in enumerate(scenarios):
        edges = np.r_[0, levels- assigned[i] - baseline[i]].clip(0)
        for j, length in enumerate(np.diff(edges)):
            if length > 0:
                objective_weight = 1 + vulnerability[i] if floor else 1.0
                segments.append((objective_weight*probs[j:].sum(), i, length))
    budget -= baseline.sum()
    for _, i, length in sorted(segments, key=lambda q: -q[0]):
        value = min(length, budget)
        used[i] += value
        budget -= value
    assert (used <= caps + 1e-7).all()
    return used


def scenario(rng, n, ratio, shock, arrival):
    hazard = rng.choice(HAZARDS, n)
    exposure = rng.lognormal(7.0, .85, n)
    severity = rng.beta(2, 2, n)
    vulnerability = rng.beta(2, 2, n)
    hazard_scale = dict(zip(HAZARDS, [1.1, 1.2, .7, .8, 1.35, .9], strict=True))
    mean = exposure * (.08 + .45*severity) * np.array([hazard_scale[h] for h in hazard])
    true = mean * rng.lognormal(0, shock, n)
    if arrival == "simultaneous":
        wave = np.zeros(n, int)
    else:
        wave = rng.integers(0, 3, n)
    # Budget ratios are relative to planned aggregate need used for scenario construction.
    # The policies receive only realized incremental availability, not the multiplier or future waves.
    budget = mean.sum()*ratio*rng.uniform(.75, 1.25, 3)*np.array([.45, .3, .25])
    return (hazard, exposure, severity, vulnerability, mean, true, wave, budget)


def evaluate_policy(values, policy):
    hazard, exposure, severity, vulnerability, mean, need, arrivals, budgets = values
    assigned = np.zeros(len(need))
    weighted_wait = 0.0
    remaining = 0.0
    for t, capacity in enumerate(budgets):
        remaining += capacity
        indices = arrivals <= t
        views = assigned[indices], mean[indices], vulnerability[indices]
        if indices.any():
            cap = np.maximum(1.55*views[1]-views[0], 0)
            if policy in ("equal", "proportional_need"):
                weights = np.ones(len(cap)) if policy == "equal" else views[1]
                step = capped_proportional(weights, cap, remaining)
            elif policy in ("severity_priority", "exposure_priority"):
                scores = severity if policy == "severity_priority" else exposure
                step = priority(scores[indices], cap, remaining)
            else:
                equity = .35 if policy.startswith("equity") else 0
                step = risk_allocation(*views, remaining, equity, policy.endswith("robust"))
            remaining -= step.sum()
            assigned[indices] += step
            weighted_wait += (np.maximum(need[indices]-assigned[indices], 0)*(1+vulnerability[indices])).sum()
        assert remaining >= -1e-7
    served = np.minimum(need, assigned)
    service = served / need
    top = vulnerability >= np.median(vulnerability)
    assert assigned.sum() <= budgets.sum() + 1e-7
    return {
        "need_met": served.sum()/need.sum(),
        "vulnerability_weighted_need_met": np.average(service, weights=need*(1+vulnerability)),
        "min_service_fraction": service.min(),
        "p10_service_fraction": np.quantile(service,.1),
        "high_vulnerability_service": np.average(service[top], weights=need[top]),
        "high_minus_low_vulnerability_service": np.average(service[top], weights=need[top])-np.average(service[~top], weights=need[~top]),
        "weighted_wait_units": weighted_wait,
        "wasted_budget_fraction": (assigned-served).sum()/budgets.sum(),
        "spent": assigned.sum(),
        "budget": budgets.sum(),
    }, assigned


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--scenarios-per-cell", type=int, default=120)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    records, demonstration = [], []
    for ratio, shock, arrivals in itertools.product((.4, .65, .9), (.25, .6), ("simultaneous", "sequential")):
        for replication in range(args.scenarios_per_cell):
            values = scenario(rng, 12, ratio, shock, arrivals)
            for name in POLICIES:
                result, assigned = evaluate_policy(values, name)
                records.append({"planned_budget_ratio": ratio, "need_log_noise": shock,
                                "arrival_type": arrivals, "replication": replication, "policy": name, **result})
                if ratio == .65 and shock == .6 and arrivals == "sequential" and replication == 0:
                    h, e, s, v, mean, real, wave, _ = values
                    demonstration.extend({"scenario_event": i+1, "hazard": h[i], "population_context": e[i],
                                          "severity_score": s[i], "vulnerability": v[i], "estimated_need_units": mean[i],
                                          "simulated_need_units": real[i], "arrival_wave": wave[i], "policy": name,
                                          "allocation_units": assigned[i]} for i in range(len(h)))
    d = pd.DataFrame(records)
    dimensions = ["planned_budget_ratio", "need_log_noise", "arrival_type"]
    d.to_csv(args.output/"synthetic_allocation_scenarios.csv", index=False)
    d.groupby(dimensions + ["policy"]).mean(numeric_only=True).drop(columns="replication").to_csv(args.output/"synthetic_allocation_metrics.csv")
    sample = pd.DataFrame(demonstration)
    sample.to_csv(args.output/"synthetic_allocation_demonstration.csv", index=False)


if __name__ == "__main__":
    main()
