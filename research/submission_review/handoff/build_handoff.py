#!/usr/bin/env python3
"""Build a deterministic ZIP from explicit, local, immutable-source inputs only."""
import argparse
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZipInfo, ZIP_STORED
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_handoff import (
    EVIDENCE_SHA, FILES, HERE, NAMES, NOTEBOOK_PIN, SOURCE_MANIFEST, VIEWER_PIN,
    check_bytes, check_manifest, hashes, require,
)

# Independent pins for the exact previously published viewer and notebook blobs.
# Entries: input SHA256 and published Git blob ID (tested before creating output).
VIEWER = {
    "index.html": ("9ddf0fedf618e7e9e3df5a194890ecb1f88883d4a48a4917cba2d423f7e84203", "815513ee7cc50b9617212a7a60c9c05eb45667f2"),
    "test_structure.py": ("452e0fe1638f9c57d42e317330817004c14eed7b614638719bfac69845cca45f", "6a34c86bf7e05145de42dfc55e5a89be4ae93ca6"),
    "test_v1.js": ("cb7388b84f95986539a3f7be94e0681f9b6d04bf220885438c172db13b440984", "fd4c9fac3b826ebd346536abadf433e2bb9b50b2"),
    "test_v2.js": ("f55541a4ee6b6187e168fc24cebec64c219fb9c221fe80f651fb3593ebe86bb7", "8e1850ccfcb8e51610632b4239c7b78b17b98fb3"),
}
NOTEBOOK = {
    "thesis-review.ipynb": ("bae2b4698d2c001ee201ea5603fadf268a4449130e5b0eb1be75daf2bb032caa", "45f8a6d8687a423857e4e54c57a835d95375c196"),
    "thesis-review.executed.ipynb": ("ba1484449fc6d4c55e2d82bc682f2dc5aa6f8309cc1ac16d910bc4898ae4bcfe", "3873207df971498788036f387194349fff50299e"),
    "notebook_tool.py": ("be5955e9ee91b54507807651b49bb991677180ec49d6bac99ab4760dedbb27b6", "a46911c9744e90af67e1e245127286990285036c"),
    "claims.json": ("2004b613ef33e8c10ca67cfd24fa065bade2c0c0654702e5448ac5a5b554b9ee", "5a99f8c534143e29737bbf40af9362d247462bbf"),
}
EVIDENCE_BLOB = "9a023855a3e20e5bc52691a97da80d60b22bd23e"


def safe_input(folder, leaf):
    f = folder / leaf
    require(f.resolve(strict=True) == f and f.is_file(), f"Input is missing or is a symlink: {leaf}")
    return f.read_bytes()


def build(notebook, viewer, evidence, archive, manifest_out):
    notebook, viewer, evidence = [p.resolve(strict=True) for p in (notebook,viewer,evidence)]
    if manifest_out and archive.resolve() == manifest_out.resolve():
        raise ValueError("Archive and manifest must have different output names")
    for target in filter(None,(archive,manifest_out)):
        require(not target.exists() and target.resolve().parent not in (notebook,viewer,evidence),
                "Write new filenames outside the input folders")
    payload, entries = {}, []

    def add(member, b, origin, commit, path, h=None, size=None):
        require(member in FILES and member not in payload, f"Unexpected or duplicate output member: {member}")
        digest, blob = hashes(b)
        entry = {"member":member, "origin":origin, "commit":commit, "path":path,
                 "size":len(b) if size is None else size, "sha256":h[0] if h else digest,
                 "git_blob":h[1] if h else blob}
        check_bytes(b,entry)
        payload[member] = b
        entries.append(entry)

    for name in ("QUICKSTART.md","verify_handoff.py"):
        add(name, safe_input(HERE,name), "local handoff helper", None, name)
    origin = "published Git blob"
    for leaf, pin in VIEWER.items():
        add("viewer/"+leaf,safe_input(viewer,leaf),origin,VIEWER_PIN,
            "research/submission_review/decision_support_v2/"+leaf,pin)
    for leaf, pin in NOTEBOOK.items():
        add("notebook/"+leaf,safe_input(notebook,leaf),origin,NOTEBOOK_PIN,
            "research/submission_review/reviewer_notebook/"+leaf,pin)
    manifest_bytes = safe_input(evidence,"manifest.json")
    add("notebook/evidence/manifest.json",manifest_bytes,origin,NOTEBOOK_PIN,
        "research/submission_review/reviewer_notebook/manifest.json",(EVIDENCE_SHA,EVIDENCE_BLOB))
    manifest = json.loads(manifest_bytes)
    require(manifest["schema_version"] == 1 and manifest["repository"] == "leonmarienga/ML-thesis",
            "Wrong notebook evidence schema")
    rows = manifest["files"]
    require(sorted(i["name"] for i in rows) == sorted(NAMES), "Unexpected input allowlist")
    for row in rows:
        leaf = row["name"]
        add("notebook/evidence/"+leaf,safe_input(evidence,leaf),origin,row["commit"],row["path"],
            (row["sha256"],row["git_blob"]),row["size"])
    output_manifest = {"schema_version":1,"repository":"leonmarienga/ML-thesis",
                       "files":sorted(entries,key=lambda x:x["member"])}
    check_manifest(output_manifest)
    payload[SOURCE_MANIFEST] = (json.dumps(output_manifest,sort_keys=True,indent=2)+"\n").encode("utf-8")
    with ZipFile(archive,"x") as z:
        for member in sorted(payload):
            info = ZipInfo(member,date_time=(1980,1,1,0,0,0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = ZIP_STORED
            z.writestr(info,payload[member])
    if manifest_out:
        with manifest_out.open("xb") as f:
            f.write(payload[SOURCE_MANIFEST])
    print(f"{len(payload)} allowlisted ZIP members, {archive.stat().st_size} bytes, SHA-256 {hashes(archive.read_bytes())[0]}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--notebook",type=Path,required=True,help="directory of the four exact previously published notebook files")
    p.add_argument("--viewer",type=Path,required=True,help="directory of pinned index.html and three tests")
    p.add_argument("--evidence",type=Path,required=True,help="directory of exact manifest.json and 18 evidence inputs")
    p.add_argument("--archive",type=Path,required=True)
    p.add_argument("--manifest-out",type=Path,help="optional copy of portable manifest outside ZIP")
    a = p.parse_args()
    build(a.notebook,a.viewer,a.evidence,a.archive,a.manifest_out)


if __name__ == "__main__":
    try:
        main()
    except (OSError,KeyError,ValueError) as exc:
        print(f"BUILD FAILED: {exc}",file=sys.stderr)
        sys.exit(1)
