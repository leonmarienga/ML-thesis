"""Check the 10-point veto boundary, mutual constraints, and closed-route denominators."""
from pathlib import Path
import csv
import sys
import numpy as np

from replay import ARMS, decision, measures

sys.path.insert(0, str(Path(sys.argv[1]).resolve()/"analysis"))
import thesis_transport_pareto as tr
stage = tr.Stage(np.full((tr.SERVICES,tr.ZONES),100.),np.full(tr.SERVICES,25.),
                 np.full(tr.ZONES,1500.),np.ones((tr.SERVICES,tr.ZONES)),10000.)
for arm in ARMS:
    sent, veto, gap = decision(tr, stage, arm)
    tr.verify(stage, sent, 0.)
    assert not veto
hybrid, veto, gap = decision(tr, stage, ARMS[2])
assert 0 < gap < .10 and not veto
np.testing.assert_array_equal(hybrid, decision(tr, stage, ARMS[1])[0])
stage.inlet[:] = 100
hybrid, veto, gap = decision(tr, stage, ARMS[2])
assert veto and gap > .10
np.testing.assert_array_equal(hybrid, decision(tr, stage, ARMS[0])[0])
tr.verify(stage, hybrid, 0.)
stage.routes[0] = 0
for arm in ARMS:
    sent, _, _ = decision(tr, stage, arm)
    served = tr.verify(stage, sent, 0.)
    assert np.allclose(served[:,0], 0) and measures(tr, stage.need, served)["least_served"] == 0
stage.routes[:] = 0
hybrid, veto, gap = decision(tr, stage, ARMS[2])
assert not veto and gap is None and np.allclose(hybrid, 0)
with (Path(sys.argv[2])/"action_digests.csv").open(newline="") as f:
    groups = {}
    for r in csv.DictReader(f):
        groups.setdefault(r["scenario_id"],{})[r["arm"]] = r
assert len(groups) == 48
for g in groups.values():
    a=g[ARMS[2]]
    intended = g[ARMS[0] if float(a["stage1_veto_shortfall"]) > .10+1e-12 else ARMS[1]]
    assert a["stage1_action_sha256"] == intended["stage1_action_sha256"]
print("PASS: veto boundary sides, proposal/fallback, closed and no-eligible routes,")
print("      and 48 independently logged first-stage selections match the intended arm.")
