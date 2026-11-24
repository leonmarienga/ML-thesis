#!/usr/bin/env python3
"""Replay survey-informed, synthetic household service allocation comparisons."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

from thesis_precoverage_documentary import text_pages
from thesis_water_recourse import optimize

HOUSEHOLDS = np.array([700, 560, 880, 420, 1100, 620, 580, 360])
VULNERABILITY = np.array([.12, .24, .35, .47, .56, .69, .81, .92])
SUSCEPTIBILITY = np.array([-.4, .7, -.5, -.3, .5, .2, .8, -.2])
WEIGHT = 1 + 2 * VULNERABILITY
HIGH = VULNERABILITY >= .69
DRAWS = 8
FAMILIES = ("beta_anchor_binomial_counts", "logit_normal_overdispersed_counts")
POLICIES = ("equal_fraction", "vulnerability_myopic", "expected_recourse", "cvar_recourse")
ANCHORS = {
    "response": {"food": (.210, .157, .262), "water": (.232, .176, .289),
                 "electricity": (.745, .659, .831), "medication": (.172, .128, .217)},
    "recovery": {"food": (.111, .057, .165), "water": (.109, .064, .155),
                 "electricity": (.033, .003, .064), "medication": (.098, .058, .139)},
}


def audit_anchors(pdf):
    pages = text_pages(pdf)
    source = " ".join(" ".join(p.split()) for p in pages)
    for row in ("17.2 12.8–21.7 4253 9.8 5.8–13.9", "21.0 15.7–26.2",
                "23.2 17.6–28.9", "25.5 16.9–34.1", "96.7 93.6–99.7",
                "11.1 5.7–16.5", "10.9 6.4–15.5"):
        assert row in source, row


def demand(rng, anchor, severity, family):
    mean, low, high = anchor
    if family == FAMILIES[0]:
        variance = ((high - low) / 3.92)**2
        concentration = mean * (1 - mean) / variance - 1
        marginal = rng.beta(mean * concentration, (1 - mean) * concentration)
    else:
        marginal = expit(rng.normal(logit(mean), (logit(high) - logit(low)) / 3.92))
    target = expit(logit(np.clip(marginal, 1e-6, 1 - 1e-6)) + .45 * severity)
    tilt = 1.05 * (VULNERABILITY - .5) + .5 * severity * SUSCEPTIBILITY + rng.normal(0, .17, len(HOUSEHOLDS))
    offset = brentq(lambda value: np.average(expit(value + tilt), weights=HOUSEHOLDS) - target, -40, 40)
    p = expit(offset + tilt)
    assert abs(np.average(p, weights=HOUSEHOLDS) - target) < 1e-9
    if family == FAMILIES[1]:
        p = rng.beta(32 * p, 32 * (1 - p))
    all_requests = rng.binomial(HOUSEHOLDS, p)
    first = rng.binomial(all_requests, .48)
    return first, all_requests - first


def proportional(requests, stock):
    return min(1, stock / max(1, requests.sum())) * requests


def dispatch(requests, stock, equal, floor=.25):
    if equal:
        return proportional(requests, stock)
    out = floor * proportional(requests, stock)
    for zone in np.argsort(-WEIGHT):
        out[zone] += min(requests[zone] - out[zone], max(0, stock - out.sum()))
    return out


def recourse(first, opening, future, replenishments, risk, floor=0.):
    zones = len(first)
    flow = zones * (1 + DRAWS)
    size = flow + (DRAWS + 1 if risk else 0)
    c = np.zeros(size)
    c[:zones] = -WEIGHT
    c[zones:flow] = -.92 * np.tile(WEIGHT, DRAWS) / DRAWS
    rows = DRAWS + 1 + DRAWS * int(risk)
    matrix = np.zeros((rows, size))
    matrix[0, :zones] = 1
    rhs = np.empty(rows)
    rhs[0] = opening
    lower = [floor * proportional(first, opening)]
    lower.extend(floor * proportional(need, stock) for need, stock in zip(future, replenishments, strict=True))
    bounds = list(zip(np.concatenate(lower), np.r_[first, future.ravel()], strict=True))
    if risk:
        c[flow:] = [.60, *np.repeat(.60 / (DRAWS * .25), DRAWS)]
        bounds.extend([(0, None)] * (DRAWS + 1))
    for j in range(DRAWS):
        idx = slice(zones * (1 + j), zones * (2 + j))
        matrix[1 + j, :zones] = 1
        matrix[1 + j, idx] = 1
        rhs[1 + j] = opening + replenishments[j]
        if risk:
            row = 1 + DRAWS + j
            matrix[row, :zones] = -WEIGHT
            matrix[row, idx] = -WEIGHT
            matrix[row, [flow, flow + 1 + j]] = -1
            rhs[row] = -WEIGHT @ (first + future[j])
    return optimize(c, (matrix, rhs), bounds)[:zones]


def scenario_inputs(phase, family, number, held_seed, plan_seed, stock_ratio):
    held = np.random.default_rng(held_seed)
    planning = np.random.default_rng(plan_seed)
    severity = held.normal(0, .7)
    signal = severity + held.normal(0, .25)
    posterior_mean = signal * .7**2 / (.7**2 + .25**2)
    posterior_sd = .7 * .25 / np.hypot(.7, .25)
    shelter = (.04, .08, .16)[number % 3]
    anchors = dict(ANCHORS[phase], shelter=(shelter, shelter * .7, shelter * 1.3))
    inputs = []
    for service, anchor in anchors.items():
        first, second = demand(held, anchor, severity, family)
        stock = int(stock_ratio * anchor[0] * HOUSEHOLDS.sum())
        opening = int(.48 * stock)
        future_stock = stock - opening
        observed_supply = round(future_stock * held.uniform(.75, 1.25))
        future = np.array([demand(planning, anchor, s, family)[1]
                           for s in planning.normal(posterior_mean, posterior_sd, DRAWS)])
        planning_supply = np.rint(future_stock * planning.uniform(.75, 1.25, DRAWS))
        inputs.append((service, first, second, opening, observed_supply, future, planning_supply))
    return shelter, severity, signal, inputs


def verify(first, second, served1, served2, opening, available):
    assert (served1 >= -1e-6).all() and (served2 >= -1e-6).all()
    assert (served1 <= first + 1e-6).all() and (served2 <= second + 1e-6).all()
    assert (first + second <= HOUSEHOLDS).all()
    assert served1.sum() <= opening + 1e-6 and served2.sum() <= available + 1e-6


def scenario(phase, family, number, held_seed, plan_seed):
    shelter, severity, signal, inputs = scenario_inputs(phase, family, number, held_seed, plan_seed, .72)
    details = []
    for service, first, second, opening, observed_supply, future, planning_supply in inputs:
        decisions = (dispatch(first, opening, True), dispatch(first, opening, False),
                     recourse(first, opening, future, planning_supply, False),
                     recourse(first, opening, future, planning_supply, True))
        for policy, served1 in zip(POLICIES, decisions, strict=True):
            available = max(0., opening - served1.sum()) + observed_supply
            served2 = dispatch(second, available, policy == POLICIES[0])
            verify(first, second, served1, served2, opening, available)
            for zone, n1, n2, x1, x2 in zip(range(len(first)), first, second, served1, served2, strict=True):
                details.append((service, policy, zone, n1, n2, x1, x2, opening, observed_supply))
    return shelter, severity, signal, details


def summarized(frame):
    zone = frame.groupby("zone")[["need", "served"]].sum()
    positive = zone.need.gt(0)
    high = zone.loc[np.flatnonzero(HIGH)]
    assert high.need.sum() > 0
    return {"served_fraction": zone.served.sum() / zone.need.sum(),
            "high_vulnerability_served_fraction": high.served.sum() / high.need.sum(),
            "lowest_zone_fraction": (zone.served[positive] / zone.need[positive]).min()}


def paired(scores, metrics):
    rng = np.random.default_rng(73723)
    rows = []
    keys = ["phase", "family"]
    for context, group in [("pooled", scores), *scores.groupby(keys, sort=False)]:
        pivot = group.pivot(index=["phase", "family", "case"], columns="policy", values=metrics)
        strata = [np.array(indices) for indices in pivot.groupby(level=[0, 1]).indices.values()]
        samples = np.concatenate([rng.choice(idx, size=(2000, len(idx))) for idx in strata], axis=1)
        for policy in POLICIES[1:]:
            for metric in metrics:
                value = (pivot[(metric, policy)] - pivot[(metric, POLICIES[0])]).to_numpy()
                low, high = np.quantile(value[samples].mean(axis=1), [.025, .975])
                rows.append({"phase": context[0] if isinstance(context, tuple) else context,
                             "family": context[1] if isinstance(context, tuple) else context,
                             "policy": policy, "metric": metric, "paired_difference": value.mean(),
                             "lower_95": low, "upper_95": high, "scenarios": len(value)})
    return pd.DataFrame(rows)


def save(pdf, output):
    audit_anchors(pdf)
    output.mkdir(parents=True, exist_ok=True)
    records, registry = [], []
    for phase_no, phase in enumerate(ANCHORS):
        for family_no, family in enumerate(FAMILIES):
            for case in range(48):
                label = 1000 * phase_no + 100 * family_no + case
                held_seed, plan_seed = 48000 + label, 96000 + label
                shelter, severity, signal, detail = scenario(phase, family, case, held_seed, plan_seed)
                registry.append((phase, family, case, held_seed, plan_seed, shelter, severity, signal))
                records.extend((phase, family, case, shelter, *row) for row in detail)
    register = pd.DataFrame(registry, columns=["phase", "family", "case", "held_seed", "planning_seed",
                                               "shelter_assumed", "severity_simulated", "planning_signal"])
    assert register.held_seed.is_unique and register.planning_seed.is_unique
    assert not set(register.held_seed) & set(register.planning_seed)
    frame = pd.DataFrame(records, columns=["phase", "family", "case", "shelter_assumed", "service", "policy",
                                             "zone", "need1", "need2", "served1", "served2", "supply1", "supply2"])
    frame["need"] = frame.need1 + frame.need2
    frame["served"] = frame.served1 + frame.served2
    group = ["phase", "family", "case", "shelter_assumed", "policy"]
    scores = pd.DataFrame([dict(zip(group, idx, strict=True), **summarized(part))
                           for idx, part in frame.groupby(group, sort=False)])
    metrics = [name for name in scores if name.endswith("_fraction")]
    summary = scores.groupby(["phase", "family", "policy"], as_index=False)[metrics].mean()
    service = frame.groupby(["phase", "family", "shelter_assumed", "service", "policy"], as_index=False)[["need", "served"]].sum()
    service["served_fraction"] = service.served / service.need
    paired_scores = paired(scores, metrics)
    for name, table in [("scenario_registry", register), ("scenario_metrics", scores), ("summary", summary),
                        ("service_metrics", service), ("paired_changes", paired_scores)]:
        table.to_csv(output / f"{name}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("casper_pdf", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    save(args.casper_pdf, args.output)
