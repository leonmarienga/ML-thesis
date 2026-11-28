authenticated = False
claims = {}
evidence = {}

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
from statistics import mean

# Pin over the manifest bytes, not a trust-on-first-use list generated from the input.
MANIFEST_SHA256 = "@MANIFEST_SHA256@"
evidence_dir = Path(os.environ.get("THESIS_REVIEW_EVIDENCE", str(Path.cwd() / "evidence"))).resolve(strict=True)

def require(ok, message):
    if not ok:
        raise ValueError(message)

manifest_path = evidence_dir / "manifest.json"
require(not manifest_path.is_symlink(), "Manifest must be a regular file")
manifest_bytes = manifest_path.read_bytes()
require(hashlib.sha256(manifest_bytes).hexdigest() == MANIFEST_SHA256,
        "Manifest SHA-256 mismatch; no evidence was used")
manifest = json.loads(manifest_bytes)
require(manifest["schema_version"] == 1 and manifest["repository"] == "leonmarienga/ML-thesis",
        "Unexpected evidence schema or repository")
verified = {}
for member in manifest["files"]:
    name = member["name"]
    require(bool(re.fullmatch(r"[a-z][a-z_0-9]*\.(csv|json|md)", name)) and name not in verified,
            "Unsafe or duplicate evidence name")
    require(bool(re.fullmatch(r"[0-9a-f]{40}", member["commit"])) and
            member["path"].startswith("research/") and ".." not in member["path"].split("/"),
            f"Unpinned or unsafe source path for {name}")
    source = evidence_dir / name
    require(source.is_file() and not source.is_symlink(), f"Evidence unavailable or not a regular file: {name}")
    data = source.read_bytes()
    blob = hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()
    require(member["size"] == len(data) and member["sha256"] == hashlib.sha256(data).hexdigest() and
            member["git_blob"] == blob, f"Evidence digest mismatch: {name}; no metrics calculated")
    verified[name] = data

def rows(name):
    return list(csv.DictReader(io.StringIO(evidence[name].decode("utf-8-sig"), newline="")))

def one(items, **where):
    matched = [r for r in items if all(r[k] == v for k, v in where.items())]
    require(len(matched) == 1, f"Expected an unambiguous row for {where}")
    return matched[0]

def near(a, b, tolerance=1e-8):
    return abs(a-b) <= tolerance

evidence = verified
authenticated = True
print(f"Verified {len(evidence)} pinned files against SHA-256, byte counts and Git blob IDs before parsing")
