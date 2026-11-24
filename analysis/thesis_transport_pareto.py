#!/usr/bin/env python3
"""Replay transport-constrained allocation and enumerate sampled Pareto frontiers."""
import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from thesis_service_scenarios import ANCHORS, FAMILIES, HIGH, VULNERABILITY, audit_anchors, scenario_inputs

CARGO = np.array([1., 1.8, 2.5, .25, 2.2])
VOLUME = np.array([1., 1.2, 1.8, .4, 2.8])
SERVICES, ZONES = len(CARGO), len(VULNERABILITY)
SIZE = SERVICES * ZONES
FLOORS = np.arange(11) / 10
FUTURES = 4
METRICS = ("overall", "high_vulnerability", "least_served")
STATIC = ("efficiency", "vulnerability", "geography", "mixed", "maxmin")
SEQUENTIAL = ("expected", "worst_branch", "cvar")
PREFERENCES = {
    "total service": (1., 0., 0.),
    "vulnerability priority": (.25, .65, .10),
    "minimum coverage priority": (.20, .20, .60),
    "balanced": (.40, .30, .30),
}
STOCK_MATRIX = np.repeat(np.eye(SERVICES), ZONES, axis=1)
ROUTE_MATRIX = np.tile(np.eye(ZONES), (1, SERVICES)) * np.repeat(CARGO, ZONES)
VOLUME_ROW = np.repeat(VOLUME, ZONES)
VULNERABILITY_WEIGHT = np.broadcast_to(1 + 2 * VULNERABILITY, (SERVICES, ZONES))


@dataclass
class Stage:
    need: np.ndarray
    inlet: np.ndarray
    routes: np.ndarray
    arrival: np.ndarray
    storage: float


def stage(need, inlet, ordinary_routes, nominal_handling, rng, regime):
    scales = {"normal": (.83, 1.16, .92, 1., .94, 1.),
              "delay": (.55, 1.02, .60, .92, .48, .88),
              "closure": (.40, .95, .68, .98, .64, .92)}
    low, high, arrival_low, arrival_high, intake_low, intake_high = scales[regime]
    routes = ordinary_routes * rng.uniform(low, high, ZONES)
    if regime == "closure":
        routes[rng.choice(ZONES, 2, replace=False)] = 0
    arrival = rng.uniform(arrival_low, arrival_high, (SERVICES, ZONES))
    available = np.floor(inlet * rng.uniform(intake_low, intake_high, SERVICES))
    storage = nominal_handling * rng.uniform(.68, 1.12)
    return Stage(need, available, routes, arrival, storage)


def proportional(state):
    eligible = state.need * (state.routes > 0) * (state.inlet[:, None] > 0)
    full = eligible / state.arrival
    by_zone = CARGO @ full
    by_service = full.sum(axis=1)
    route_rate = np.divide(state.routes, by_zone, out=np.full(ZONES, np.inf), where=by_zone > 0)
    stock_rate = np.divide(state.inlet, by_service, out=np.full(SERVICES, np.inf), where=by_service > 0)
    baseline = min(1., route_rate.min(), stock_rate.min(), state.storage / max(1., VOLUME @ by_service))
    return full * baseline


def limits(state):
    return np.r_[state.inlet, state.routes, state.storage]


def constraints():
    return np.vstack([STOCK_MATRIX, ROUTE_MATRIX, VOLUME_ROW])


def bounds(state, floor):
    return list(zip((floor * proportional(state)).ravel(), (state.need / state.arrival).ravel(), strict=True))


def solve(cost, matrix, limits_, bounds_):
    result = linprog(cost, A_ub=matrix, b_ub=limits_, bounds=bounds_, method="highs")
    if not result.success:
        raise ValueError(result.message)
    assert (matrix @ result.x <= np.asarray(limits_) + 1e-5).all()
    return result.x


def dispatch(state, objective, floor):
    if objective == "proportional":
        return proportional(state)
    weight = {
        "efficiency": np.ones((SERVICES, ZONES)),
        "vulnerability": VULNERABILITY_WEIGHT,
        "geography": np.broadcast_to(1 / np.maximum(state.need.sum(axis=0), 1), (SERVICES, ZONES)),
        "mixed": np.broadcast_to(1 + VULNERABILITY, (SERVICES, ZONES)),
    }
    a, b = constraints(), limits(state)
    box = bounds(state, floor)
    if objective != "maxmin":
        c = -(weight[objective] * state.arrival).ravel()
        return solve(c, a, b, box).reshape(SERVICES, ZONES)

    active = (state.need * (state.inlet[:, None] > 0)).sum(axis=0)
    active *= state.routes > 0
    zone = np.tile(np.eye(ZONES), (1, SERVICES)) * state.arrival.ravel()
    extra = np.column_stack([-zone, active])
    matrix = np.vstack([np.column_stack([a, np.zeros(len(a))]), extra])
    rhs = np.r_[b, np.zeros(ZONES)]
    best = solve(np.r_[np.zeros(SIZE), -1.], matrix, rhs, [*box, (0., 1.)])[-1]
    efficient = solve(np.r_[-state.arrival.ravel(), 0.], matrix, rhs, [*box, (max(0., best - 1e-8), best)])
    return efficient[:SIZE].reshape(SERVICES, ZONES)


def recourse(initial, outlooks, objective, floor):
    is_robust = objective == "worst_branch"
    is_cvar = objective == "cvar"
    flow = SIZE * (1 + FUTURES)
    extra = (FUTURES + 1) if is_cvar else int(is_robust)
    total = flow + extra
    base_matrix = constraints()
    n = len(base_matrix)
    matrix = np.zeros((n * (FUTURES + 1) + (FUTURES if extra else 0), total))
    rhs = np.zeros(len(matrix))
    c = np.zeros(total)
    w0 = (VULNERABILITY_WEIGHT * initial.arrival).ravel()
    matrix[:n, :SIZE] = base_matrix
    rhs[:n] = limits(initial)
    expectation = .05 if is_robust else 1.
    c[:SIZE] = -expectation * w0
    box = bounds(initial, floor)
    if extra:
        c[flow] = 1. if is_robust else .60
        if is_cvar:
            c[flow + 1:] = .60 / (.25 * FUTURES)

    for i, future in enumerate(outlooks):
        section = slice((i + 1) * n, (i + 2) * n)
        sl = slice((i + 1) * SIZE, (i + 2) * SIZE)
        matrix[section.start:section.start + SERVICES, :SIZE] = STOCK_MATRIX
        matrix[section, sl] = base_matrix
        rhs[section] = limits(future)
        rhs[section.start:section.start + SERVICES] += initial.inlet
        wj = (VULNERABILITY_WEIGHT * future.arrival).ravel()
        c[sl] = -expectation * .95 * wj / FUTURES
        box.extend(bounds(future, floor))
        if extra:
            row = n * (FUTURES + 1) + i
            matrix[row, :SIZE] = -w0
            matrix[row, sl] = -wj
            matrix[row, flow] = -1
            if is_cvar:
                matrix[row, flow + 1 + i] = -1
            rhs[row] = -np.sum(VULNERABILITY_WEIGHT * (initial.need + future.need))
    box.extend([(0., None)] * extra)
    return solve(c, matrix, rhs, box)[:SIZE].reshape(SERVICES, ZONES)


def verify(state, allocation, floor):
    assert (allocation >= -1e-6).all()
    served = state.arrival * allocation
    assert (served <= state.need + 1e-6).all()
    assert (allocation.sum(axis=1) <= state.inlet + 1e-6).all()
    assert (CARGO @ allocation <= state.routes + 1e-6).all()
    assert np.sum(VOLUME @ allocation) <= state.storage + 1e-6
    assert (allocation + 1e-6 >= floor * proportional(state)).all()
    return served


def world(phase, family, case, stock_ratio, seeds, regime):
    held, planning, routes_held, routes_plan = seeds
    shelter, _, _, supplies = scenario_inputs(phase, family, case, held, planning, stock_ratio)
    first, later, opening, deliveries, expected, planned_deliveries = map(np.asarray, zip(
        *(record[1:] for record in supplies), strict=True))
    actual = np.random.default_rng(routes_held)
    plan = np.random.default_rng(routes_plan)
    normal_routes = (1.20 + .45 * (1 - VULNERABILITY)) * np.maximum(CARGO @ first, 5)
    handling = max(1., VOLUME @ opening)
    early_regime = "normal" if regime == "normal" else "delay"
    initial = stage(first, opening, normal_routes, handling, actual, early_regime)
    later_stock = max(handling, VOLUME @ deliveries)
    observed = stage(later, deliveries, normal_routes * 1.05, later_stock, actual, regime)
    expected_regimes = {"normal": ("normal", "normal", "delay", "delay"),
                        "delay": ("normal", "delay", "delay", "closure"),
                        "closure": ("delay", "closure", "closure", "closure")}
    fixed_storage = max(handling, VOLUME @ np.mean(planned_deliveries, axis=1))
    outlooks = [
        stage(expected[:, j], planned_deliveries[:, j], normal_routes * 1.05, fixed_storage, plan, alternative)
        for j, alternative in enumerate(expected_regimes[regime])
    ]
    return shelter, initial, observed, outlooks


def evaluate(initial, observed, outlooks):
    rows = []
    options = [("proportional", 1.)] + [(policy, f) for policy in (*STATIC, *SEQUENTIAL) for f in FLOORS]
    need = initial.need + observed.need
    for policy, floor in options:
        sequential = policy in SEQUENTIAL
        x = recourse(initial, outlooks, policy, floor) if sequential else dispatch(initial, policy, floor)
        delivered1 = verify(initial, x, floor)
        carry = np.maximum(0., initial.inlet - x.sum(axis=1))
        next_stage = Stage(observed.need, observed.inlet + carry, observed.routes, observed.arrival, observed.storage)
        y = dispatch(next_stage, "vulnerability" if sequential else policy, floor)
        delivered = delivered1 + verify(next_stage, y, floor)
        zone_need, zone_served = need.sum(axis=0), delivered.sum(axis=0)
        coverage = zone_served[zone_need > 0] / zone_need[zone_need > 0]
        rows.append({"policy": policy, "floor": floor, "overall": delivered.sum() / need.sum(),
                     "high_vulnerability": zone_served[HIGH].sum() / zone_need[HIGH].sum(),
                     "least_served": coverage.min(), "unfilled_units": need.sum() - delivered.sum(),
                     "stage1_sent": x.sum(), "stage2_sent": y.sum()})
    return rows


def frontier(metrics):
    means = metrics.groupby(["policy", "floor"], as_index=False)[list(METRICS)].mean()
    objectives = means[list(METRICS)].to_numpy()
    means["nondominated"] = [
        not (((objectives >= row - 1e-9).all(axis=1)) & ((objectives > row + 1e-9).any(axis=1))).any()
        for row in objectives
    ]
    supported = means[means.nondominated]
    preferred = []
    for name, weights in PREFERENCES.items():
        scores = supported[list(METRICS)].to_numpy() @ weights
        idx = scores.argmax()
        preferred.append({"preference": name, **supported.iloc[idx].to_dict(), "score": scores[idx]})
    return means, pd.DataFrame(preferred)


def paired(metrics):
    rng = np.random.default_rng(82130)
    strata = ["phase", "family", "case"]
    shown = metrics[metrics.policy.ne("proportional") & metrics.floor.isin([0., 1.])]
    wide = shown.pivot(index=strata, columns=["policy", "floor"], values=list(METRICS))
    groups = wide.groupby(level=[0, 1]).indices.values()
    sample = np.concatenate([rng.choice(idx, (2000, len(idx))) for idx in groups], axis=1)
    output = []
    for policy in (*STATIC, *SEQUENTIAL):
        for metric in METRICS:
            delta = (wide[(metric, policy, 1.)] - wide[(metric, policy, 0.)]).to_numpy()
            lo, hi = np.quantile(delta[sample].mean(axis=1), [.025, .975])
            output.append((policy, metric, delta.mean(), lo, hi))
    return pd.DataFrame(output, columns=["policy", "metric", "difference", "lower_95", "upper_95"])


def plots(means, sensitivity, output):
    plt.rcParams.update({"font.size": 10, "axes.spines.right": False, "axes.spines.top": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained")
    selected = means[means.nondominated]
    for ax, y, color, label, colorbar in (
        (axes[0], "overall", "high_vulnerability", "Total requests served (%)", "High-vulnerability service (%)"),
        (axes[1], "high_vulnerability", "overall", "High-vulnerability service (%)", "Total requests served (%)"),
    ):
        ax.scatter(100 * means.least_served, 100 * means[y], s=15, color="#c0cbd0", alpha=.75)
        dots = ax.scatter(100 * selected.least_served, 100 * selected[y], c=100 * selected[color], s=32,
                          cmap="viridis", edgecolors="#202c3a", linewidth=.35)
        ax.set(xlabel="Least-served zone coverage (%)", ylabel=label)
        fig.colorbar(dots, ax=ax, label=colorbar, location="bottom", shrink=.86)
    fig.suptitle("Transport-constrained allocation frontier across 48 simulated cases")
    fig.savefig(output / "pareto_frontiers.png", dpi=175)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.9), sharex=True, layout="constrained")
    wanted = sensitivity[sensitivity.policy.eq("vulnerability")]
    palette = {"normal": "#137c7c", "delay": "#c2583c", "closure": "#4541a2"}
    for ax, metric, label in zip(axes, METRICS, ["Total service", "High-vulnerability service", "Least-served zone"],
                                 strict=True):
        for regime, group in wanted.groupby("regime"):
            ax.plot(100 * group.floor, 100 * group[metric], marker="o", markersize=2, lw=1.6,
                    label=regime.title(), color=palette[regime])
        ax.set(title=label, xlabel="Proportional floor (%)", ylabel="Requests served (%)")
    axes[1].legend(loc="best", frameon=False)
    fig.suptitle("Minimum service floors under normal, delayed and closed routes")
    fig.savefig(output / "floor_sensitivity.png", dpi=175)
    plt.close(fig)


def simulate(pdf, output):
    audit_anchors(pdf)
    output.mkdir(parents=True, exist_ok=True)
    rows, seeds = [], []
    for p, phase in enumerate(ANCHORS):
        for f, family in enumerate(FAMILIES):
            for case in range(12):
                label = 1000 * p + 100 * f + case
                streams = [i + label for i in (248000, 296000, 348000, 396000)]
                regime = ("normal", "delay", "closure")[(case // 3 + p + f) % 3]
                ratio = (.55, .72, .90)[(case + case // 3 + p + 2 * f) % 3]
                shelter, initial, realized, planning = world(phase, family, case, ratio, streams, regime)
                assert len(planning) == FUTURES
                key = {"phase": phase, "family": family, "case": case, "regime": regime}
                seeds.append({**key, "stock_ratio": ratio, "shelter": shelter, "closed_stage2_routes": sum(realized.routes == 0),
                              **dict(zip(("held_demand", "plan_demand", "held_network", "plan_network"), streams, strict=True))})
                rows.extend({**key, **score} for score in evaluate(initial, realized, planning))
    registry = pd.DataFrame(seeds)
    assert len(set(registry[["held_demand", "plan_demand", "held_network", "plan_network"]].to_numpy().ravel())) == 192
    measurements = pd.DataFrame(rows)
    means, preferences = frontier(measurements)
    groups = measurements.groupby(["regime", "policy", "floor"], as_index=False)[list(METRICS)].mean()
    for name, frame in [("scenario_registry", registry), ("case_metrics", measurements), ("pareto_points", means),
                        ("preference_choices", preferences), ("disruption_sensitivity", groups), ("paired_floors", paired(measurements))]:
        frame.to_csv(output / (name + ".csv"), index=False)
    plots(means, groups, output)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("casper_pdf", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    simulate(args.casper_pdf, args.output)
