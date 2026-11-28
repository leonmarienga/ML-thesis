#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
sha256sum -c SHA256SUMS
python -B coordinates.py
python -B enc.py
sha256sum -c SHA256SUMS
