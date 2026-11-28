#!/usr/bin/env python3
"""Matched fresh-draw simulation fixture. No participant or financial data."""
import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

PIN = "b0b6edb229d112dd6d11896a512bf1112414e4f6"
SURVEY_SHA = "9991799f25d8be0aa235048806561674ec97b29126af7b8ebb5a5923f7aa94ba"
PROTOCOL_SHA = "e72a1ae1b20d247163a9a0768804f052bdfd93340d8d2c2634200cd1d3c81976"
METRICS = ("overall", "high_vulnerability", "least_served")
ARMS = ("scripted_human_rule", "optimizer_only", "optimizer_with_scripted_veto")
SEED_OFFSETS = (748000, 796000, 848000, 896000)
STOCKS = (.55, .72, .90)
REGIMES = ("normal", "delay", "closure")
VETO_MARGIN = .10
RESAMPLES = 4000
BOOT_SEED = 20261128
SOURCES = ("analysis/thesis_transport_pareto.py", "analysis/thesis_service_scenarios.py",
           "analysis/thesis_precoverage_documentary.py", "analysis/thesis_water_recourse.py",
           "research/requirements.txt", "research/continuous/BATCH14_PROTOCOL.md",
           "research/continuous/transport_pareto/results/scenario_registry.csv")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(repo, survey):
    git = ["git", "-C", str(repo)]
    assert subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip() == PIN, "wrong source HEAD"
    subprocess.run(git + ["diff", "--exit-code", "HEAD", "--", "analysis", *SOURCES], check=True)
    assert sha(survey) == SURVEY_SHA, "wrong CASPER PDF"
    assert sha(Path(__file__).with_name("PROTOCOL.md")) == PROTOCOL_SHA, "protocol changed since frozen fixture design"
    return {name: sha(repo/name) for name in SOURCES}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def minimum_eligible(state, sent):
    active = (state.routes > 0) * (state.need * (state.inlet[:, None] > 0)).sum(axis=0)
    yes = active > 0
    covered = (state.arrival * sent).sum(axis=0)
    return float(np.min(covered[yes] / active[yes])) if yes.any() else None


def decision(transport, state, policy):
    if policy == ARMS[0]:
        return transport.proportional(state), False, None
    proposal = transport.dispatch(state, "vulnerability", 0.)
    if policy == ARMS[1]:
        return proposal, False, None
    rule = transport.proportional(state)
    p, h = minimum_eligible(state, proposal), minimum_eligible(state, rule)
    shortfall = None if (p is None or h is None) else h - p
    veto = shortfall is not None and shortfall > VETO_MARGIN + 1e-12
    pick = rule if veto else proposal
    np.testing.assert_array_equal(pick, rule if veto else proposal)
    return pick, veto, shortfall


def second_state(initial, second, first_sent):
    carry = np.maximum(0., initial.inlet - first_sent.sum(axis=1))
    return replace(second, inlet=second.inlet + carry)


def after_first(transport, initial, first, reveal_second):
    # The callback enforces an explicit reveal boundary: choice is fixed before the later state is received.
    sent = first[0]
    served1 = transport.verify(initial, sent, 0.)
    observed = reveal_second()
    return second_state(initial, observed, sent), served1


def measures(transport, both_need, both_served):
    zone_need, zone_served = both_need.sum(axis=0), both_served.sum(axis=0)
    assess = zone_need > 0
    high = transport.HIGH
    assert zone_need[high].sum() > 0 and assess.any()
    result = [float(both_served.sum()/both_need.sum()),
              float(zone_served[high].sum()/zone_need[high].sum()),
              float(np.min(zone_served[assess]/zone_need[assess]))]
    assert all(-1e-12 <= x <= 1+1e-12 for x in result)
    return dict(zip(METRICS, result, strict=True))


def allocation_digest(x):
    return hashlib.sha256(np.asarray(x, dtype="<f8").tobytes(order="C")).hexdigest()


def compare_scores(table):
    order = [p for p in table if p["arm"] == ARMS[0]]
    ids = [r["scenario_id"] for r in order]
    strata = np.array([(p["phase"] + "|" + p["family"]) for p in order])
    assert len(order) == len(set(ids)) == 48
    by_arm = {arm: {row["scenario_id"]: row for row in table if row["arm"] == arm} for arm in ARMS}
    groups = [np.flatnonzero(strata == tax) for tax in sorted(set(strata))]
    assert len(groups) == 4 and all(len(g) == 12 for g in groups)
    rng = np.random.default_rng(BOOT_SEED)
    draws = np.concatenate([rng.choice(g, size=(RESAMPLES, len(g)), replace=True) for g in groups], axis=1)
    pairs = []
    for left, right in ((ARMS[1], ARMS[0]), (ARMS[2], ARMS[0]), (ARMS[2], ARMS[1])):
        assert set(ids) == set(by_arm[left]) == set(by_arm[right])
        for metric in METRICS:
            a = np.array([by_arm[left][i][metric] for i in ids])
            b = np.array([by_arm[right][i][metric] for i in ids])
            d = a-b
            qlo, qhi = np.quantile(d[draws].mean(axis=1), (.025, .975))
            pairs.append(dict(left=left, right=right, metric=metric, n=len(ids), left_mean=float(a.mean()),
                              right_mean=float(b.mean()), difference=float(d.mean()), lower_95=float(qlo),
                              upper_95=float(qhi), positive=int((d > 1e-9).sum()),
                              tied=int((np.abs(d) <= 1e-9).sum()), negative=int((d < -1e-9).sum())))
    return pairs


def run(repo, survey, output):
    hashes = verify_sources(repo, survey)
    sys.path.insert(0, str(repo / "analysis"))
    import thesis_transport_pareto as tr
    import scipy
    import pandas
    tr.audit_anchors(survey)
    old_seeds = set()
    with (repo/SOURCES[-1]).open(newline="") as f:
        for row in csv.DictReader(f):
            old_seeds.update(int(row[k]) for k in ("held_demand", "plan_demand", "held_network", "plan_network"))
    registry, scores, actions = [], [], []
    seeds_seen = set()
    for p, phase in enumerate(tr.ANCHORS):
        for f, family in enumerate(tr.FAMILIES):
            for case in range(12):
                label = 1000*p + 100*f + case
                seeds = tuple(offset+label for offset in SEED_OFFSETS)
                assert not set(seeds).intersection(old_seeds | seeds_seen)
                seeds_seen.update(seeds)
                regime = REGIMES[(case // 3 + p + f) % 3]
                stock = STOCKS[(case + case // 3 + p + 2*f) % 3]
                shelter, initial, later, _ = tr.world(phase, family, case, stock, seeds, regime)
                key = dict(scenario_id=f"hgo-{label:04d}", phase=phase, family=family, case=case, regime=regime)
                registry.append({**key, "stock_ratio":stock, "shelter_anchor":shelter,
                                 **dict(zip(("held_demand","plan_demand","held_network","plan_network"), seeds, strict=True))})

                # Seal every first-stage choice before any policy sees its second stage.
                first = {policy:decision(tr, initial, policy) for policy in ARMS}
                for arm in ARMS:
                    x1, veto1, gap1 = first[arm]
                    before = allocation_digest(x1)
                    altered, _ = after_first(tr, initial, first[arm],
                                             lambda: replace(later, routes=np.zeros_like(later.routes)))
                    assert not altered.routes.any(), "noninterference control did not change the later world"
                    assert before == allocation_digest(x1), "first action changed when future was replaced"
                    state2, y1 = after_first(tr, initial, first[arm], lambda: later)
                    x2, veto2, gap2 = decision(tr, state2, arm)
                    y2 = tr.verify(state2, x2, 0.)
                    assert before == allocation_digest(x1)
                    values = measures(tr, initial.need+later.need, y1+y2)
                    scores.append(dict(**key, arm=arm, **values, stage1_veto=int(veto1), stage2_veto=int(veto2)))
                    actions.append(dict(scenario_id=key["scenario_id"], arm=arm, stage1_action_sha256=before,
                                        stage2_action_sha256=allocation_digest(x2), stage1_veto_shortfall=gap1,
                                        stage2_veto_shortfall=gap2))
    assert len(registry) == len({row["scenario_id"] for row in registry}) == 48
    assert len(seeds_seen) == 192
    assert len(scores) == len(actions) == 144
    contrasts = compare_scores(scores)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("scenario_registry",registry), ("case_metrics",scores),
                       ("action_digests",actions), ("paired_contrasts",contrasts)):
        write_csv(output/f"{name}.csv", rows)
    (output/"provenance.json").write_text(json.dumps(
        dict(commit=PIN, survey_sha256=SURVEY_SHA, protocol_sha256=PROTOCOL_SHA,
             script_sha256=sha(Path(__file__)), input_sha256=hashes,
             python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__,
             pandas=pandas.__version__, cases=48, action_pairs=144,
             disjoint_seeds=len(seeds_seen), stage_constraint_checks=288,
             stage1_future_replacement_checks=144), indent=2, sort_keys=True)+"\n")
    hybrid = [r for r in scores if r["arm"]==ARMS[2]]
    print("48 paired scenarios; 144 arm rows; 288 scored feasible dispatches; 144 future-replacement controls.")
    print("Vetoes by stage:", [sum(r[f"stage{i}_veto"] for r in hybrid) for i in (1,2)])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for field in ("repo", "survey", "output"):
        ap.add_argument(field, type=lambda value:Path(value).resolve())
    args = ap.parse_args()
    run(args.repo, args.survey, args.output)
