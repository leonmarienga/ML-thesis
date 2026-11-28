"""Optional technical validation using the invoking environment's Jupyter Python kernel."""
import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("notebook", type=Path)
p.add_argument("evidence", type=Path)
p.add_argument("executed", type=Path)
p.add_argument("--stdlib-claims",type=Path, required=True)
p.add_argument("--stdlib-executed",type=Path, required=True)
a = p.parse_args()
evidence = a.evidence.resolve(strict=True)
out = a.executed.resolve()
if out == evidence or evidence in out.parents or out.exists():
    raise ValueError("Use a new output path outside the evidence directory")
notebook = nbformat.read(a.notebook, as_version=4)
nbformat.validate(notebook)
nbformat.validate(nbformat.read(a.stdlib_executed, as_version=4))
os.environ["THESIS_REVIEW_EVIDENCE"] = str(evidence)
# Register an ephemeral named kernelspec using this exact interpreter, never a global/default Python.
with TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix="jk-") as sandbox:
    d = Path(sandbox)
    spec = d / "kernels" / "reviewer-validation"
    spec.mkdir(parents=True)
    (spec / "kernel.json").write_text(json.dumps({"argv":[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"],
                                                 "display_name":"Temporary reviewer validation","language":"python"}))
    os.environ["JUPYTER_PATH"] = str(d)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(d / "run")
    os.environ["IPYTHONDIR"] = str(d / "ipython")
    manager = AsyncKernelManager(kernel_name="reviewer-validation", transport="ipc", ip=str(d / "socket"))
    runner = NotebookClient(notebook, km=manager, timeout=60,
                            resources={"metadata":{"path":str(a.notebook.resolve().parent)}})
    runner.execute()
nbformat.validate(notebook)
code = [c for c in notebook.cells if c.cell_type == "code"]
assert [c.execution_count for c in code] == list(range(1,len(code)+1))
assert not any(o.output_type=="error" for c in code for o in c.outputs)
ledger = json.loads("".join(o.text for o in code[-1].outputs if o.output_type=="stream" and o.name=="stdout"))
assert ledger == json.loads(a.stdlib_claims.read_bytes()), "Jupyter and standard-library claim ledgers differ"
notebook.metadata["reviewer_execution"] = {"engine":"Jupyter actual ipykernel, via nbclient",
                                            "python":sys.version.split()[0]}
with out.open("x",encoding="utf-8") as f:
    nbformat.write(notebook,f)
print(f"nbformat validated source and both outputs; actual ipykernel completed {len(code)} code cells; "
      f"Jupyter ledger equals stdlib ledger ({len(ledger)} scoped claims).")
