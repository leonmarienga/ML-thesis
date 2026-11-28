"""Execute through Jupyter's in-process kernel when process sockets are unavailable."""
import argparse
import atexit
import io
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import nbformat
from ipykernel.inprocess.manager import InProcessKernelManager

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
with TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix="inproc-") as scratch:
    os.environ["IPYTHONDIR"] = scratch
    manager = InProcessKernelManager()
    manager.start_kernel()
    client = manager.client()
    client.start_channels()
    code = [c for c in notebook.cells if c.cell_type == "code"]
    try:
        for count, c in enumerate(code, 1):
            manager.kernel.stdout = io.StringIO()
            manager.kernel.stderr = io.StringIO()
            msgid = client.execute(c.source)
            reply = client.get_shell_msg(timeout=10)
            if reply["parent_header"]["msg_id"] != msgid or reply["content"]["status"] != "ok":
                raise ValueError(f"Jupyter cell {c.id} failed: {reply['content']}")
            c.execution_count = reply["content"]["execution_count"]
            c.outputs = []
            while True:
                event = client.get_iopub_msg(timeout=10)
                if event["parent_header"].get("msg_id") != msgid:
                    continue
                if event["msg_type"] == "status" and event["content"]["execution_state"] == "idle":
                    break
                if event["msg_type"] == "stream":
                    c.outputs.append(nbformat.v4.new_output("stream", **event["content"]))
                elif event["msg_type"] not in ("execute_input","status"):
                    raise ValueError(f"Unhandled Jupyter output {event['msg_type']} in {c.id}")
            for stream in ("stdout", "stderr"):
                text = getattr(manager.kernel, stream).getvalue()
                if text:
                    c.outputs.append(nbformat.v4.new_output("stream", name=stream, text=text))
            if c.execution_count != count:
                raise ValueError("Unexpected Jupyter cell execution count")
    finally:
        client.stop_channels()
        atexit.unregister(manager.kernel.shell.atexit_operations)
        manager.kernel.shell.atexit_operations()
        manager.shutdown_kernel()
nbformat.validate(notebook)
ledger = json.loads("".join(o.text for o in code[-1].outputs if o.output_type=="stream" and o.name=="stdout"))
if ledger != json.loads(a.stdlib_claims.read_bytes()):
    raise ValueError("In-process Jupyter and standard-library claim ledgers differ")
notebook.metadata["reviewer_execution"] = {"engine":"Jupyter ipykernel InProcessKernelManager (no subprocess/transport)",
                                            "python":sys.version.split()[0]}
with out.open("x",encoding="utf-8") as f:
    nbformat.write(notebook,f)
print(f"nbformat validated source and both outputs; Jupyter in-process kernel completed {len(code)} code cells; "
      f"Jupyter ledger equals stdlib ledger ({len(ledger)} scoped claims).")
