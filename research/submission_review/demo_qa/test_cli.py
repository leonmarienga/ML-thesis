"""Smoke the CLI against the exact historical research checkout from the report."""
import ast
import csv
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import json

PIN = "b0b6edb229d112dd6d11896a512bf1112414e4f6"
repo = Path(sys.argv[1]).resolve()
base = repo / "research/results"
out = Path(__file__).resolve().parent
script = out / "analysis/thesis_demo.py"
assert subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() == PIN


def load(filename, root=base):
    with (root / filename).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run(disaster, scenario, results=base, error=None):
    process = subprocess.run([sys.executable, "-I", "-S", str(script), str(disaster), "--scenario-event",
                              str(scenario), "--results", str(results)], capture_output=True, text=True)
    if error is not None:
        assert process.returncode == 2 and error in process.stderr, process.stderr
        assert not process.stdout and "Traceback" not in process.stderr
        return
    assert process.returncode == 0, process.stderr
    assert not process.stderr
    return json.loads(process.stdout)


def constants(code, keys):
    tree = ast.parse(code)
    return {assignment.targets[0].id: tuple(ast.literal_eval(assignment.value))
            for assignment in tree.body if isinstance(assignment, ast.Assign)
            and isinstance(assignment.targets[0], ast.Name) and assignment.targets[0].id in keys}


keys = {"BANDS", "PHYSICAL"}
archived_constants = constants((repo / "analysis/thesis_hazard_features.py").read_text(), keys)
assert constants(script.read_text(), keys) == archived_constants
with (base / "hazards/cohort_features.csv").open(encoding="utf-8") as handle:
    assert {r["fyDeclared"] for r in csv.DictReader(handle)} <= {str(y) for y in range(2010, 2025)}
events = load("hazards/cohort_features.csv")
model = load("forward/forward_predictions.csv")
simulation = load("allocation/synthetic_allocation_demonstration.csv")
intervals = load("uncertainty/forward_prediction_intervals.csv")
event = next(row for row in events if row["disasterNumber"] == "4722")
pred = next(r for r in model if r["disasterNumber"] == "4722" and r["variant"] == "nested_metadata")
independent_sim = [r for r in simulation if r["scenario_event"] == "1"]
relevant_intervals = [r for r in intervals if r["disasterNumber"] == "4722" and r["variant"] == "nested_metadata" and r["method"] == "dollar"]
a = run(4722, 1)
assert set(a) == {"historical_disaster", "retrospective_model_estimate", "simulated_humanitarian_allocation"}
actual, estimate, sim = a.values()
assert actual["historical_fema_obligations_usd"] == float(event["target"])
assert actual["evidence_fields_available"] == [key for key in archived_constants["PHYSICAL"] if event[key]]
assert estimate["historical_obligation_estimate_usd"] == float(pred["predicted_amount"])
assert estimate["band"] == archived_constants["BANDS"][int(pred["predicted_band"])]
assert list(estimate["band_probabilities"].values()) == [float(pred[f"p_band_{i}"]) for i in range(6)]
assert estimate["intervals"] == [{k: float(r[k]) for k in ("level", "lower_usd", "upper_usd")} for r in relevant_intervals]
assert sim["policy_comparison"] == [{k: r[k] if k in {"hazard", "policy"} else float(r[k])
                                     for k in ("hazard", "policy", "estimated_need_units", "simulated_need_units", "allocation_units", "vulnerability")}
                                    for r in independent_sim]
assert run(4722, 2)["historical_disaster"] == actual
assert run(4722, 2)["simulated_humanitarian_allocation"] != sim
modeled = {row["disasterNumber"] for row in model if row["variant"] == "nested_metadata"}
outside = next(int(row["disasterNumber"]) for row in events if row["disasterNumber"] not in modeled)
assert run(outside, 1)["retrospective_model_estimate"] == {"status": "outside evaluated forward cohort"}
run(-1, 1, error="disaster -1 is not in archived results")
run(4722, -1, error="scenario -1 is not in archived simulation results")
with tempfile.TemporaryDirectory() as temp:
    scratch = Path(temp)
    (scratch / "hazards").mkdir()
    shutil.copy(base / "hazards/cohort_features.csv", scratch / "hazards/cohort_features.csv")
    run(4722, 1, scratch, error="No such file")
    path = scratch / "hazards/cohort_features.csv"
    with path.open("a", encoding="utf-8") as handle:
        with (base / "hazards/cohort_features.csv").open() as source:
            handle.write(next(line for line in source if line.startswith("4722,")))
    run(4722, 1, scratch, error="disaster 4722 has duplicate rows")
    path.write_text("disasterNumber,unexpected\n4722,invalid\n", encoding="utf-8")
    run(4722, 1, scratch, error="missing field(s)")
    missing = dict(event, fyDeclared="2026", target="fixture_not_a_funding_label")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=missing.keys())
        writer.writeheader()
        writer.writerow(missing)
    run(4722, 1, scratch, error="later outcomes are embargoed")
print("PASS: pinned source definitions, historical scope, exact observed obligations vs separate simulation,")
print("      six archived probabilities, all archived intervals and selected policy rows, changing scenarios,")
print("      out-of-forward result, unknown event/scenario, missing and malformed inputs, duplicate event and date guard.")
