"""Run every unmodified baseline notebook cell in one socket-free IPython process."""
from pathlib import Path
import sys
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

source, output = map(Path, sys.argv[1:])
nb = nbformat.read(source, as_version=4)
shell = InteractiveShell.instance()
cells = [c for c in nb.cells if c.cell_type == 'code']
for i, cell in enumerate(cells, 1):
    with capture_output() as cap:
        result = shell.run_cell(cell.source, store_history=False)
    if not result.success:
        print(f"Cell {i} failed.", file=sys.stderr)
        if cap.stdout:
            print(cap.stdout, file=sys.stderr)
        if cap.stderr:
            print(cap.stderr, file=sys.stderr)
        result.raise_error()
    cell.execution_count = i
    cell.outputs = []
    for stream in ("stdout", "stderr"):
        if log := getattr(cap, stream):
            cell.outputs.append(nbformat.v4.new_output("stream", name=stream, text=log))
    for rich in cap.outputs:
        cell.outputs.append(nbformat.v4.new_output("display_data", data=rich.data, metadata=rich.metadata))
nb.metadata["verified_execution_context"] = {"runner": "socket-free IPython replay", "unmodified_code_cells": len(cells)}
nbformat.validate(nb)
nbformat.write(nb, output)
print(f"{len(cells)} unmodified notebook code cells executed sequentially; outputs captured and validated")
