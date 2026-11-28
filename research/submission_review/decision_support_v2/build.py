"""Build the stand-alone v2 research viewer from pinned historical summaries and replayed simulations."""
import argparse
import csv
import hashlib
import io
import json
from math import isclose, isfinite
from pathlib import Path
from statistics import mean
import subprocess
import zipfile
import base_build

HERE = Path(__file__).resolve().parent
METRICS = ("overall", "high_vulnerability", "least_served")
ARMS = ("scripted_human_rule", "optimizer_only", "optimizer_with_scripted_veto")
REGIMES = {"all": 48, "normal": 15, "delay": 18, "closure": 15}
FIXTURE_FILES = {
    "case_metrics.csv": "7f9defaff3ffd6750198f661a136a3bf31e49bf4c02c752b140e7d3380922c49",
    "paired_contrasts.csv": "9dcd9913c1ae62cd48390ed87409783d5afaedfe16f6209a26392d58937a8ce9",
    "scenario_registry.csv": "4fb27f1281c1b80c25e6f4629a5e17a14c84cdfa5ff63a0546478363d549f6af",
}
GATES = {
    "Severe Ice dominance": ("analysis/nonbio_825_severe_ice_operational_dominance.py", "ice"),
    "Fire duration": ("analysis/nonbio_826_fire_duration_floor.py", "fire"),
    "Hurricane zero-mission prerequisite": ("analysis/nonbio_827_hurricane_zero_mission_prereq.py", "hur"),
    "Hurricane amendment floor": ("analysis/nonbio_829_hurricane_amendment_floor.py", "hur"),
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def table(raw):
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


def verify_fixture(fixture):
    rows = {}
    for filename, expected in FIXTURE_FILES.items():
        raw = (fixture / "results" / filename).read_bytes()
        assert digest(raw) == expected, f"nonmatching fixture source: {filename}"
        rows[filename] = table(raw)
    registry = {r["scenario_id"]:r["regime"] for r in rows["scenario_registry.csv"]}
    assert len(registry) == len(rows["scenario_registry.csv"]) == 48
    sets = {r: {} for r in REGIMES}
    scored = {(r["scenario_id"],r["arm"]): r for r in rows["case_metrics.csv"]}
    assert len(scored) == len(rows["case_metrics.csv"]) == 144
    for regime, n in REGIMES.items():
        ids = [k for k,v in registry.items() if regime == "all" or v==regime]
        assert len(ids) == n
        for arm in ARMS:
            cases = [scored[idx,arm] for idx in ids]
            assert all(registry[r["scenario_id"]] == r["regime"] for r in cases)
            metrics = {m:[float(r[m]) for r in cases] for m in METRICS}
            assert all(0 <= v <= 1 for vals in metrics.values() for v in vals)
            sets[regime][arm] = {
                "scores": {m:mean(values) for m,values in metrics.items()},
                "vetoes": [sum(int(r[f"stage{i}_veto"]) for r in cases) for i in (1,2)],
            }

    contrasts = {}
    for r in rows["paired_contrasts.csv"]:
        left, right, m = r["left"], r["right"], r["metric"]
        a, b = sets["all"][left]["scores"][m], sets["all"][right]["scores"][m]
        assert int(r["n"]) == 48 and all(isclose(x,float(r[k]),abs_tol=1e-12) for x,k in
                                               ((a,"left_mean"),(b,"right_mean"),(a-b,"difference")))
        lo, hi = float(r["lower_95"]),float(r["upper_95"])
        assert isfinite(lo) and isfinite(hi) and lo <= hi
        contrasts.setdefault(left+" "+right,{})[m] = [lo,hi]
        contrasts.setdefault(right+" "+left,{})[m] = [-hi,-lo]
    assert len(contrasts) == 6 and all(len(value)==3 for value in contrasts.values())
    assert sets["all"][ARMS[2]]["vetoes"] == [45,34]
    return dict(groups=sets,counts=REGIMES,intervals=contrasts,file_hashes=FIXTURE_FILES)


def evidence(repo):
    pin = base_build.COMMIT
    hashes = {}
    def read(filename):
        raw = subprocess.check_output(["git","-C",str(repo),"show",f"{pin}:{filename}"])
        hashes[filename] = digest(raw)
        return raw

    archived_path = "research/inputs/actions/track_a.zip"
    archive = read(archived_path)
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        canonical = next(i for i in zf.namelist() if i.endswith("all_912_immutable_overlay.csv"))
        arch_rows = table(zf.read(canonical))
    assert len(arch_rows) == 912
    assert all(2010 <= int(r["fyDeclared"]) <= 2024 for r in arch_rows)
    fieldnames = list(arch_rows[0])
    predicted = next(k for k in ("router_pred","pred_band","predicted_band","routed_band") if k in fieldnames)
    actual = "true_band" if "true_band" in fieldnames else "actual_band"
    correct = sum(r[actual]==r[predicted] for r in arch_rows)
    assert correct == 830

    metricpath = "research/results/track_a/model_metrics.csv"
    versions = [r for r in table(read(metricpath)) if r["axis"]=="all"]
    versions = {r["variant"]:float(r["mae_usd"]) for r in versions}
    assert isclose(versions["track_a_reported_historical_50_200m_overlay"],5014819.011614494,abs_tol=1e-8)

    cohortpath = "research/results/hazards/cohort_features.csv"
    cohort = table(read(cohortpath))
    assert all(2010 <= int(r["fyDeclared"]) <= 2024 for r in cohort)
    sample, = [r for r in cohort if r["disasterNumber"]=="4722"]
    assert isclose(float(sample["target"]),1243478.26,abs_tol=.01)

    paths = {}
    for label,(file,var) in GATES.items():
        code = read(file).decode()
        lines = [i for i,line in enumerate(code.splitlines(),1) if
                 line.lstrip().startswith(var) and "target_clean" in line and "50_000_000" in line]
        assert len(lines)==1, label
        paths[label] = file+"#L"+str(lines[0])

    p = "research/results/uncertainty/interval_metrics.csv"
    intervals = {}
    for level in (.8,.9):
        match = [r for r in table(read(p))
                 if r["variant"]=="nested_metadata" and r["method"]=="dollar" and float(r["level"])==level]
        whole, = [r for r in match if r["axis"]=="all"]
        tail = [r for r in match if r["axis"]=="actual_band" and int(r["group"])>=3]
        assert len(tail)==3
        group = {"all":(round(int(whole["n"])*float(whole["coverage"])),int(whole["n"])),
                 "tail":(sum(round(int(r["n"])*float(r["coverage"])) for r in tail),sum(int(r["n"]) for r in tail))}
        group["below"] = tuple(a-b for a,b in zip(group["all"],group["tail"],strict=True))
        intervals[str(round(100*level))] = group
    assert intervals["90"] == {"all":(281,308),"below":(281,299),"tail":(0,9)}
    return dict(track_a=dict(correct=correct,n=len(arch_rows),mae=versions,gate_paths=paths),
                example=dict(id=4722,hazard=sample["incidentType"],state=sample["state"],
                             year=int(sample["fyDeclared"]),amount=float(sample["target"])),
                intervals=intervals,file_hashes=hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pinned_repo",type=Path)
    parser.add_argument("fixture",type=Path)
    parser.add_argument("--output",type=Path,default=HERE/"index.html")
    args = parser.parse_args()
    original = base_build.source(args.pinned_repo)
    original["evidence"] = evidence(args.pinned_repo)
    original["veto_fixture"] = verify_fixture(args.fixture)
    data = json.dumps(original,separators=(",",":"),allow_nan=False).replace("<","\\u003c")
    css = (HERE/"v2.css").read_text()
    panels = (HERE/"panels.html").read_text()
    html = (HERE/"base_template.html").read_text()
    app = (HERE/"base_app.js").read_text()+"\n"+(HERE/"v2_app.js").read_text()
    assert "</script" not in app.casefold()
    anchor = '<section aria-labelledby="select-title">'
    assert html.count(anchor)==1
    html = (html.replace("</style>",css+"\n</style>").replace(anchor,panels+"\n"+anchor)
            .replace("Relief allocation trade-offs | Batch 14 research prototype","Disaster relief decision support | Evidence viewer v2")
            .replace("Research prototype • Batch 14","Research prototype • Evidence viewer v2")
            .replace("<strong>Simulation only.</strong>","<strong>Research only.</strong>")
            .replace("<h1>Relief allocation trade-offs</h1>","<h1>Disaster relief decision support</h1>")
            .replace("Compare two settings on three simulated service outcomes. Change which outcomes matter most, then see what the selected policy did under different road conditions.","Inspect where the historical evidence holds and where it fails. Then compare two different simulations of relief allocation and their trade-offs.")
            .replace("No observed aid delivered, historical funding, or real-world effectiveness estimates appear in this interface.","Historical FEMA accounting appears only in the evidence section. The allocation panels contain simulations and no observed aid delivered or participant results.")
            .replace("<h2 id=\"select-title\">Choose how to compare</h2>","<h2 id=\"select-title\">Batch 14 frontier: choose how to compare</h2>")
            .replace("<!-- DATA -->",data).replace("<!-- APP -->",app))
    args.output.write_text(html)


if __name__ == "__main__":
    main()
