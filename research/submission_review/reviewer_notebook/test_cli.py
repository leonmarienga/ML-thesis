"""Exercise the reviewer's real CLI against temporary, disposable evidence copies."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("evidence", type=Path)
a = p.parse_args()
base = Path(__file__).resolve().parent
with TemporaryDirectory(dir=base, prefix="notebook-test-") as temp:
    d = Path(temp)
    working = d / "outside"
    working.mkdir()
    evidence = shutil.copytree(a.evidence, d / "evidence")
    notebook = shutil.copy2(base/"thesis-review.ipynb", d/"thesis-review.ipynb")
    env = dict(os.environ)
    env.pop("THESIS_REVIEW_EVIDENCE",None)

    def run(*args, **kw):
        return subprocess.run([sys.executable, "-I","-S",str(base/"notebook_tool.py"),
                               "run",str(notebook), *map(str,args)], cwd=working,
                              capture_output=True, text=True, env=kw.get("env",env), timeout=20)

    good = run("--claims",d/"claims.json")
    if good.returncode or json.loads((d/"claims.json").read_bytes())!=json.loads((base/"claims.json").read_bytes()):
        raise ValueError("Adjacent-directory resolver or isolated replay failed")
    if run(env={**env, "THESIS_REVIEW_EVIDENCE":str(evidence)}).returncode:
        raise ValueError("Environment override failed")
    refusal = run("--claims", evidence/"forbidden.json")
    if refusal.returncode != 1 or "Refusing output within evidence" not in refusal.stderr:
        raise ValueError("Evidence output isolation failed")
    for name, marker in (("manifest.json","Manifest SHA-256 mismatch"),("track_a_bands.csv","Evidence digest mismatch: track_a_bands.csv")):
        f = evidence/name
        data = f.read_bytes()
        f.write_bytes(data+b"\n")
        bad = run("--executed", d/"forbidden.ipynb", "--claims", d/"forbidden.json")
        if bad.returncode != 1 or marker not in bad.stderr or bad.stdout or any(d.glob("forbidden.*")):
            raise ValueError(f"Tamper test did not fail closed: {name}")
        f.write_bytes(data)
    print("PASS: notebook-adjacent inputs from different CWD and environment override; evidence writes refused; separately tampered manifest/member each refused before any output export.")
