#!/usr/bin/env python3
"""Compare two-day clean-water allocation policies under synthetic shared disruptions."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

HUBS, ZONES = 2, 6
N_ARCS = HUBS * ZONES
SEED = 360192
DRAW_N = 240
PROBABILITIES = np.array([.30, .35, .25, .10])
LITRES_PER_PERSON = 15.
STATES = ("restoration", "routine", "regional_disruption", "widespread_disruption")
POLICIES = ("myopic_equity", "myopic_equal_fraction", "recourse_expected", "recourse_cvar")
HUB_FLOW = np.repeat(np.eye(HUBS), ZONES, axis=1)
ZONE_FLOW = np.tile(np.eye(ZONES), HUBS)


def scenario(rng):
    population = rng.integers(900, 3501, ZONES)
    target = population * LITRES_PER_PERSON / 1000
    vulnerability = rng.beta(2, 2, ZONES)
    normal_supply = rng.uniform(.18, .64, ZONES)
    day1 = target * (1-normal_supply)
    site = np.array([-1, -1, -1, 1, 1, 1])
    preferred = np.vstack([site < 0, site > 0])
    initial = day1.sum() * rng.uniform(.63, 1.12) * rng.dirichlet([6, 6])
    road = np.where(preferred, .85, .39) * target * rng.uniform(.65, 1.2, (HUBS, ZONES))
    road *= rng.random((HUBS, ZONES)) > .08
    later, arrivals, routes = [], [], []
    for j, (restored, replenish, route_factor) in enumerate([
            (.30, .85, 1.20), (.10, .48, 1.), (-.09, .26, .72), (-.24, .12, .51)]):
        regional = rng.choice((-1, 1))
        exposed = site == regional
        local = np.clip(normal_supply + restored + rng.uniform(-.06, .06, ZONES)
                        - j * .045 * exposed, 0, 1)
        later.append(target*(1-local))
        fraction = rng.dirichlet([5, 5])
        arrivals.append(day1.sum()*replenish*fraction*rng.uniform(.8, 1.2))
        capacity = road*route_factor*rng.uniform(.75, 1.25, (HUBS, ZONES))
        capacity *= np.where(exposed, max(.25, 1-j*.18), 1)
        routes.append(capacity)
    return {"population": population, "vulnerability": vulnerability, "demand1": day1, "stock1": initial,
            "routes1": road, "demand2": np.array(later), "resupply2": np.array(arrivals),
            "routes2": np.array(routes)}


def optimize(c, constraints, bounds):
    matrix, rhs = constraints
    fitted = linprog(c, A_ub=matrix, b_ub=rhs, bounds=bounds, method="highs")
    if not fitted.success:
        raise ValueError(fitted.message)
    assert (matrix @ fitted.x <= rhs+1e-6).all()
    return fitted.x


def dispatch(demand, inventory, roads, priority):
    matrix = np.vstack([HUB_FLOW, ZONE_FLOW])
    rhs = np.r_[inventory, demand]
    c = -np.tile(priority, HUBS) + 1e-7*np.arange(N_ARCS)
    result = optimize(c, (matrix, rhs), list(zip(np.zeros(N_ARCS), roads.ravel(), strict=True)))
    return result.reshape(HUBS, ZONES)


def plan(case, priority, risk):
    scenarios = len(PROBABILITIES)
    flow_size = (1+scenarios)*N_ARCS
    n = flow_size+(1+scenarios if risk else 0)
    idx = slice(0, N_ARCS)
    c = np.zeros(n)
    per_arc = np.tile(priority, HUBS)
    c[idx] = -per_arc
    matrix = np.zeros((HUBS+ZONES + scenarios*(HUBS+ZONES+int(risk)), n))
    first = HUBS+ZONES
    matrix[:first, idx] = np.vstack([HUB_FLOW, ZONE_FLOW])
    rhs = list(np.r_[case["stock1"], case["demand1"]])
    bounds = [(0, cap) for cap in case["routes1"].ravel()]
    row = first
    for j, p in enumerate(PROBABILITIES):
        y = slice((j+1)*N_ARCS, (j+2)*N_ARCS)
        c[y] = -p*per_arc
        matrix[row:row+HUBS, idx] = HUB_FLOW
        matrix[row:row+HUBS, y] = HUB_FLOW
        matrix[row+HUBS:row+first, y] = ZONE_FLOW
        rhs.extend(case["stock1"]+case["resupply2"][j])
        rhs.extend(case["demand2"][j])
        bounds.extend((0, cap) for cap in case["routes2"][j].ravel())
        row += first
        if risk:
            matrix[row, idx] = -per_arc
            matrix[row, y] = -per_arc
            matrix[row, flow_size] = -1.
            matrix[row, flow_size+1+j] = -1.
            rhs.append(-float(priority @ (case["demand1"]+case["demand2"][j])))
            row += 1
    if risk:
        c[flow_size] = .50
        c[flow_size+1:] = .50*PROBABILITIES/.25
        bounds.extend([(None, None), *[(0, None)]*scenarios])
    c[:flow_size] += 1e-7*np.tile(np.arange(N_ARCS), 1+scenarios)
    return optimize(c, (matrix, np.array(rhs)), bounds)[idx].reshape(HUBS, ZONES)


def first_day(case, variant, priority):
    if variant.startswith("recourse"):
        return plan(case, priority, risk=variant=="recourse_cvar")
    p = priority if variant=="myopic_equity" else 1/np.maximum(case["demand1"], 1e-6)
    return dispatch(case["demand1"], case["stock1"], case["routes1"], p)


def measure(case, variant, shipments, realized, priority):
    stock = (case["stock1"]-shipments.sum(axis=1)).clip(0)+case["resupply2"][realized]
    d1, d2 = case["demand1"], case["demand2"][realized]
    p = priority if variant!="myopic_equal_fraction" else 1/np.maximum(d2, 1e-6)
    later = dispatch(d2, stock, case["routes2"][realized], p)
    upper = d1 + d2
    assert (shipments >= -1e-6).all() and (later >= -1e-6).all()
    assert (shipments <= case["routes1"]+1e-6).all()
    assert (later <= case["routes2"][realized]+1e-6).all()
    assert (shipments.sum(axis=1) <= case["stock1"]+1e-6).all()
    assert (shipments.sum(axis=0) <= d1+1e-6).all() and (later.sum(axis=0) <= d2+1e-6).all()
    assert ((shipments+later).sum(axis=1) <= case["stock1"]+case["resupply2"][realized]+1e-6).all()
    delivered = (shipments+later).sum(axis=0)
    high = case["vulnerability"] >= np.median(case["vulnerability"])
    indicators = {"service_fraction": delivered.sum()/upper.sum(),
                  "vulnerability_weighted_service": (delivered@priority)/(upper@priority),
                  "high_vulnerability_service": delivered[high].sum()/upper[high].sum(),
                  "day1_service_fraction": shipments.sum()/d1.sum(),
                  "prepositioned_stock_carry_m3": (case["stock1"]-shipments.sum(axis=1)).sum()}
    return indicators, later


def summarize(results):
    variables = ("service_fraction", "vulnerability_weighted_service", "high_vulnerability_service", "day1_service_fraction")
    summary = results.groupby("policy")[list(variables)].mean()
    summary["p10_weighted_service"] = results.groupby("policy").vulnerability_weighted_service.quantile(.10)
    summary = summary.reset_index()
    wide = results.pivot(index="scenario", columns="policy", values=list(variables))
    n = len(wide)
    draws = np.random.default_rng(SEED+1).integers(0, n, size=(1600, n))
    rows = []
    for policy in POLICIES[1:]:
        for var in ("service_fraction", "high_vulnerability_service", "p10_weighted_service"):
            family = "vulnerability_weighted_service" if var.startswith("p10") else var
            new, reference = wide[(family, policy)].to_numpy(), wide[(family, "myopic_equity")].to_numpy()
            if var.startswith("p10"):
                differ = np.quantile(new[draws], .10, axis=1)-np.quantile(reference[draws], .10, axis=1)
                value = np.quantile(new, .10)-np.quantile(reference, .10)
            else:
                differ = (new-reference)[draws].mean(axis=1)
                value = (new-reference).mean()
            lo, hi = np.quantile(differ, [.025, .975])
            rows.append({"policy": policy, "metric": var, "difference": value, "p025": lo, "p975": hi,
                         "paired_scenarios": n})
    return summary, pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    results, scenario_inputs, example = [], [], []
    for i in range(DRAW_N):
        case = scenario(rng)
        priority = 1 + case["vulnerability"]
        planned = {name: first_day(case, name, priority) for name in POLICIES}
        # The realized disruption is picked only after every day-one plan is fixed.
        realized = int(rng.choice(len(PROBABILITIES), p=PROBABILITIES))
        for name, shipments in planned.items():
            indicators, later = measure(case, name, shipments, realized, priority)
            results.append({"scenario": i, "policy": name, "realized_state": STATES[realized], **indicators})
            if i < 3:
                for h in range(HUBS):
                    for z in range(ZONES):
                        example.append({"scenario": i, "policy": name, "hub": h+1, "zone": z+1,
                             "shipped_day1_m3": shipments[h,z], "shipped_day2_m3": later[h,z]})
        scenario_inputs.append({"scenario": i, "realized_state": STATES[realized],
            **{field: value.tolist() for field, value in case.items()}})
    results = pd.DataFrame(results)
    summary, difference = summarize(results)
    for name, table in (("simulation_cases", results), ("policy_metrics", summary),
                        ("paired_scenario_intervals", difference), ("example_dispatch", pd.DataFrame(example))):
        table.to_csv(args.output/f"{name}.csv", index=False)
    pd.DataFrame(scenario_inputs).to_json(args.output/"scenario_inputs.jsonl.gz", orient="records", lines=True,
                                           double_precision=15, compression={"method": "gzip", "mtime": 0})
    print(f"Passed feasibility checks for all {len(results)} paired scenario-policy evaluations.")


if __name__ == "__main__":
    main()
