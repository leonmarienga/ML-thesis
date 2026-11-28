#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${1:?usage: run.sh <pinned-repo> <full-path-to-venv-python> <new-output-dir>}"
PY="${2:?full path to venv python}"
OUT="${3:?new output directory (must not exist)}"
(cd "$HERE" && sha256sum -c PROTOCOL.sha256)
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
"$PY" "$HERE/target_blind_router.py" --repo "$REPO" --output "$OUT"
