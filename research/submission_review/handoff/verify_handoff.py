#!/usr/bin/env python3
"""Run the verification handoff from a trusted freshly extracted package."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
NOTEBOOK_PIN = "fb314e0edb71faccf9b7cbb196b0a245459aa696"
VIEWER_PIN = "b951c7a4883400c0d69a04d71ec91290351267e3"
EVIDENCE_SHA = "006773bfcb065f426fd79384c7d275fe102eda2c0b1450d286a4a82a9510cb32"
NAMES = (
    "forward_bootstrap.csv", "forward_matrices.csv", "forward_t5.csv",
    "gauge_stage_pairs.csv", "gauge_stage_summary.json", "interval_t8.csv",
    "replay_open_b11.md", "replay_open_b5.md", "tabletop_dry.json",
    "target_blind_folds.csv", "target_blind_summary.json", "track_a_amounts.csv",
    "track_a_bands.csv", "transport_floors.csv", "uncertainty_review.csv",
    "veto_contrasts.csv", "weight_bootstrap.csv", "weight_metrics.csv",
)
VIEWER_FILES = ("index.html", "test_structure.py", "test_v1.js", "test_v2.js")
NOTEBOOK_FILES = ("thesis-review.ipynb", "thesis-review.executed.ipynb", "notebook_tool.py", "claims.json")
FILES = {"QUICKSTART.md", "verify_handoff.py", "notebook/evidence/manifest.json"}
FILES |= {f"viewer/{f}" for f in VIEWER_FILES} | {f"notebook/{f}" for f in NOTEBOOK_FILES}
FILES |= {f"notebook/evidence/{f}" for f in NAMES}
SOURCE_MANIFEST = "SOURCE_MANIFEST.json"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def hashes(b):
    return hashlib.sha256(b).hexdigest(), hashlib.sha1(f"blob {len(b)}\0".encode()+b).hexdigest()


def check_bytes(b, entry):
    digest, git_blob = hashes(b)
    require(len(b) == entry["size"] and digest == entry["sha256"] and git_blob == entry["git_blob"],
            f"Source digest mismatch: {entry['member']}")


def check_manifest(manifest):
    require(manifest["schema_version"] == 1 and manifest["repository"] == "leonmarienga/ML-thesis",
            "Unknown source manifest schema or repository")
    entries = manifest["files"]
    seen = set()
    for item in entries:
        m = item["member"]
        require(m in FILES and m not in seen, f"Extra or duplicate manifest member: {m}")
        seen.add(m)
        if m in {"QUICKSTART.md","verify_handoff.py"}:
            require(item["origin"] == "local handoff helper" and item["commit"] is None,
                    f"Unexpected origin for local helper {m}")
        else:
            p = PurePosixPath(item["path"])
            require(item["origin"] == "published Git blob" and re.fullmatch(r"[0-9a-f]{40}", item["commit"])
                    and p.parts[0] == "research" and ".." not in p.parts, f"Invalid source for {m}")
            if m.startswith("viewer/"):
                require(item["commit"] == VIEWER_PIN and item["path"] == "research/submission_review/decision_support_v2/"+p.name,
                        f"Wrong viewer provenance: {m}")
            if m in {f"notebook/{x}" for x in NOTEBOOK_FILES} | {"notebook/evidence/manifest.json"}:
                require(item["commit"] == NOTEBOOK_PIN and item["path"] == "research/submission_review/reviewer_notebook/"+p.name,
                        f"Wrong notebook provenance: {m}")
    require(seen == FILES, "Incomplete source manifest")
    return entries


def verify_and_read(root):
    m = root / SOURCE_MANIFEST
    require(m.resolve(strict=True) == m and m.is_file(), "Missing or symlink source manifest")
    entries = check_manifest(json.loads(m.read_bytes()))
    for entry in entries:
        f = root / entry["member"]
        require(f.resolve(strict=True) == f and f.is_file(), f"Missing or symlink member: {entry['member']}")
        check_bytes(f.read_bytes(), entry)
    ev = root / "notebook/evidence/manifest.json"
    ev_bytes = ev.read_bytes()
    require(hashes(ev_bytes)[0] == EVIDENCE_SHA, "Unrecognized notebook evidence pin")
    by_member = {r["member"]: r for r in entries}
    inner = json.loads(ev_bytes)["files"]
    require(sorted(r["name"] for r in inner) == sorted(NAMES), "Unexpected inner manifest allowlist")
    for r in inner:
        outer = by_member["notebook/evidence/"+r["name"]]
        require(all(r[k] == outer[k] for k in ("sha256","git_blob","size","commit","path")),
                f"Inconsistent source provenance for {r['name']}")
    return len(entries)


def run(cmd, cwd, env):
    res = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=cwd, env=env, timeout=120)
    return res.stdout.strip()


def main():
    verified = verify_and_read(HERE)
    env = {k:v for k,v in os.environ.items()
           if k not in {"PYTHONOPTIMIZE","THESIS_REVIEW_EVIDENCE","NODE_OPTIONS","NODE_PATH"}}
    py = [sys.executable, "-I", "-S"]
    n, v = HERE/"notebook", HERE/"viewer"
    with TemporaryDirectory(prefix="thesis-t11-") as temp:
        output = Path(temp)
        log = run(py+[str(n/"notebook_tool.py"),"run",str(n/"thesis-review.ipynb"),
                      "--executed",str(output/"executed.ipynb"),"--claims",str(output/"claims.json")], output, env)
        require("[08-claims]" in log and (output/"claims.json").read_bytes() == (n/"claims.json").read_bytes(),
                "Reproduced scope ledger did not match the archived byte sequence")
        scope_ledger = json.loads((output/"claims.json").read_bytes())
        require(len(scope_ledger) == 9 and scope_ledger["tabletop"]["participants"] == 0
                and scope_ledger["track_a"]["correct"] == 830, "Notebook result scopes changed")
        cells = json.loads((output/"executed.ipynb").read_bytes())["cells"]
        require([c["execution_count"] for c in cells if c["cell_type"] == "code"] == list(range(1,9)),
                "Notebook run did not finish all eight code cells")
    structure = run(py+["test_structure.py"],v,env)
    node = shutil.which("node")
    node_checks = [run([node,f],v,env) for f in ("test_v1.js","test_v2.js")] if node else "not run: Node unavailable"
    result = {"verified_manifest_files":verified,"notebook_cells":8,"scoped_claims":9,"claims_content_equal":True,
              "viewer_structure":structure,"node_interactions":node_checks,
              "scope":"Verification only. No models, full simulator, Word, restricted outcomes, participant work, or browser UI tested by this command."}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, KeyError, ValueError, subprocess.SubprocessError) as exc:
        print(f"HANDOFF FAILED: {exc}",file=sys.stderr)
        sys.exit(1)
