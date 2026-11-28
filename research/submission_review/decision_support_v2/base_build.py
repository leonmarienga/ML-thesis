"""Generate a standalone, offline demonstrator from the original Git objects."""
from argparse import ArgumentParser
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess

COMMIT = "b0b6edb229d112dd6d11896a512bf1112414e4f6"
DIR = "research/continuous/transport_pareto/results"
NAMES = ("pareto_points", "preference_choices", "disruption_sensitivity", "scenario_registry", "paired_floors")
METRICS = ("overall", "high_vulnerability", "least_served")
PREFS = {
    "total service": [100, 0, 0],
    "vulnerability priority": [25, 65, 10],
    "minimum coverage priority": [20, 20, 60],
    "balanced": [40, 30, 30],
}


def source(repo):
    payload = {"commit": COMMIT, "hashes": {}, "weights": PREFS}
    for name in NAMES:
        filename = f"{DIR}/{name}.csv"
        raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{COMMIT}:{filename}"])
        payload["hashes"][name + ".csv"] = hashlib.sha256(raw).hexdigest()
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
        if name == "scenario_registry":
            payload["counts"] = dict(Counter(r["regime"] for r in rows))
            continue
        for row in rows:
            for field in (*METRICS, "floor", "score", "difference", "lower_95", "upper_95"):
                if field in row:
                    row[field] = float(row[field])
            if "nondominated" in row:
                assert row["nondominated"] in ("True", "False")
                row["nondominated"] = row["nondominated"] == "True"
        payload[name] = rows

    points = payload["pareto_points"]
    ids = {(p["policy"], p["floor"]) for p in points}
    assert len(ids) == len(points) == 89
    assert sum(p["nondominated"] for p in points) == 46
    assert payload["counts"] == {"normal": 15, "delay": 18, "closure": 15}
    assert len(payload["disruption_sensitivity"]) == 3 * len(points)
    for regime in payload["counts"]:
        group = [p for p in payload["disruption_sensitivity"] if p["regime"] == regime]
        assert len(group) == len(points)
        assert {(p["policy"], p["floor"]) for p in group} == ids

    assert {r["preference"] for r in payload["preference_choices"]} == set(PREFS)
    for expected in payload["preference_choices"]:
        weights = PREFS[expected["preference"]]
        score = lambda row: sum(row[m] * w / 100 for m, w in zip(METRICS, weights))
        picked = max((p for p in points if p["nondominated"]), key=score)
        assert all(picked[field] == expected[field] for field in ("policy", "floor", *METRICS))
        assert abs(score(picked) - expected["score"]) < 1e-12
    return payload


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("pinned_repo", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("index.html"))
    args = parser.parse_args()
    base = Path(__file__).parent
    html = (base / "template.html").read_text()
    data = json.dumps(source(args.pinned_repo), separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    js = (base / "app.js").read_text()
    if "</script" in js.lower():
        raise ValueError("Unsafe script terminator")
    html = html.replace("<!-- DATA -->", data).replace("<!-- APP -->", js)
    args.output.write_text(html)


if __name__ == "__main__":
    main()
