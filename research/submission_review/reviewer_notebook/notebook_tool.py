#!/usr/bin/env python3
"""Build or execute the pinned thesis-review notebook using the standard library."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import re
import sys

TITLE = """# Disaster relief thesis review: reproduce selected claims

Run from a trusted notebook copy. Put the unmodified evidence directory beside this notebook
as `evidence/`, start the Jupyter kernel with this directory as its working directory, or set
`THESIS_REVIEW_EVIDENCE` to an absolute folder path before starting Jupyter. The evidence source
is an exact manifest-pinned selection from GitHub, and is verified in full before metric parsing.
No network, additional Python packages, model training, full baseline run, or FY2025 outcomes are
needed. Historical FEMA obligations are administrative records, not economic damage or humanitarian need.

**Scope.** Where possible this notebook recomputes arithmetic from aggregated observations or matched
case summaries. Reported statistics that cannot be recomputed from those inputs are labelled archived.
It does not certify model independence, show operational skill, or evaluate a human participant.
"""
SECTIONS = [
    ("01_bootstrap", "Authenticate the offline evidence"),
    ("02_track_a", "Track A historical archive (not a forecast)"),
    ("03_forward", "Matched forward comparison on examined historical cases"),
    ("04_uncertainty", "Uncertainty coverage and upper-tail failure"),
    ("05_policy", "Transportation and governance trade-offs (synthetic only)"),
    ("06_followup", "Independent-validation limits"),
    ("07_tabletop", "Prospective protocol dry-run record"),
    ("08_claims", "Machine-readable claim ledger"),
]


def dump(data):
    return json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)+"\n"


def write_new(path, data):
    with path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(data)


def markdown(id_, text):
    return {"cell_type":"markdown","id":id_,"metadata":{},"source":text.splitlines(keepends=True)}


def build(manifest_hash, output):
    if not re.fullmatch(r"[0-9a-f]{64}", manifest_hash):
        raise ValueError("Use the independently verified manifest SHA-256, not a value inferred at runtime")
    cells = [markdown("review-intro", TITLE)]
    for id_, title in SECTIONS:
        script = Path(__file__).resolve().parent / "cells" / f"{id_}.py"
        source = script.read_text(encoding="utf-8").replace("@MANIFEST_SHA256@", manifest_hash)
        compile(source, str(script), "exec")
        cell_id = id_.replace("_","-")
        cells += [
            markdown("heading-"+cell_id, "## "+title+"\n"),
            {"cell_type":"code","id":cell_id,"metadata":{}, "execution_count":None,"outputs":[],
             "source":source.splitlines(keepends=True)},
        ]
    meta = {"kernelspec":{"display_name":"Python 3", "language":"python", "name":"python3"},
            "language_info":{"name":"python","file_extension":".py","mimetype":"text/x-python"}}
    write_new(output, dump({"nbformat":4, "nbformat_minor":5, "metadata":meta, "cells":cells}))


def execute(path, evidence_override, executed_output, claims_output):
    # This executor runs Python from a trusted notebook; it is not a sandbox for unknown notebooks.
    path = path.resolve(strict=True)
    override = evidence_override if evidence_override else os.environ.get("THESIS_REVIEW_EVIDENCE")
    evidence = Path(override).resolve(strict=True) if override else path.parent / "evidence"
    protected = evidence.resolve()
    for out in (executed_output, claims_output):
        if out is not None:
            resolved = out.resolve()
            if resolved == protected or protected in resolved.parents or resolved.exists():
                raise ValueError(f"Refusing output within evidence or over an existing file: {out}")
    notebook = json.loads(path.read_bytes())
    expected_code_ids = [name.replace("_","-") for name, _ in SECTIONS]
    ids = [c["id"] for c in notebook["cells"] if c["cell_type"] == "code"]
    if notebook["nbformat"] != 4 or ids != expected_code_ids:
        raise ValueError("Not the expected reviewer notebook")
    namespace = {"__name__":"__main__"}
    os.environ["THESIS_REVIEW_EVIDENCE"] = str(evidence)
    for index, cell in enumerate([c for c in notebook["cells"] if c["cell_type"] == "code"], 1):
        body = cell["source"]
        body = "".join(body) if isinstance(body, list) else body
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(compile(body, f"{path.name}:{cell['id']}", "exec"), namespace)
        lines = out.getvalue()
        cell["execution_count"] = index
        cell["outputs"] = [{"output_type":"stream","name":"stdout","text":lines.splitlines(keepends=True)}] if lines else []
        if lines:
            print(f"[{cell['id']}]\n"+lines, end="")
    if claims_output:
        write_new(claims_output, dump(namespace["claims"]))
    if executed_output:
        notebook["metadata"]["reviewer_execution"] = {"engine":"Python standard library, in process (not a Jupyter kernel)"}
        write_new(executed_output, dump(notebook))


def main():
    a = argparse.ArgumentParser(description=__doc__)
    sub = a.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("build", help="maintainer: embed an independently verified evidence manifest hash")
    b.add_argument("manifest_sha256")
    b.add_argument("output", type=Path)
    e = sub.add_parser("run", help="execute trusted notebook cells offline, with optional export")
    e.add_argument("notebook", type=Path)
    e.add_argument("--evidence", type=Path)
    e.add_argument("--executed", type=Path, help="new filename for the executed ipynb")
    e.add_argument("--claims", type=Path, help="new filename for structured claims JSON")
    args = a.parse_args()
    if args.mode == "build":
        build(args.manifest_sha256, args.output)
    else:
        execute(args.notebook, args.evidence, args.executed, args.claims)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as exc:
        print(f"REVIEW CHECK FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
