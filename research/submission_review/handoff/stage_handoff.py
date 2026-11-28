#!/usr/bin/env python3
"""Fetch exactly the T11 published blobs and build the offline reviewer ZIP."""
import argparse
import json
from pathlib import Path, PurePosixPath
import re
import sys
from tempfile import TemporaryDirectory
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_handoff import build
from verify_handoff import HERE, NOTEBOOK_PIN, SOURCE_MANIFEST, VIEWER_PIN, check_bytes, check_manifest, hashes, require

T11_MANIFEST_SHA = "2e0238ba08b9eb04066266e0c1013fabe958684734bba7a22102aec337fb7a49"
T11_ZIP_SHA = "f86d5d0b1aae4d9865c134f08c853d9689a3f02f415224011616754786f61db0"
PUBLISHED_COMMITS = {NOTEBOOK_PIN, VIEWER_PIN, "b0b6edb229d112dd6d11896a512bf1112414e4f6",
                     "fdba8ea4cff6dbd7c893076ef09391b2f96ae277",
                     "2038178997ca47ef0ea5fb591c4519a1f288409c"}
BASE = "https://raw.githubusercontent.com/leonmarienga/ML-thesis/"


class NoRedirect(HTTPRedirectHandler):
    """The manifest pins exact source paths; reject redirects rather than following them."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError(f"Unexpected HTTP {code} redirect from pinned artifact")


def source_rows():
    path = HERE / SOURCE_MANIFEST
    require(path.is_file() and not path.is_symlink(), "Missing regular T11 source manifest")
    data = path.read_bytes()
    require(hashes(data)[0] == T11_MANIFEST_SHA, "Unrecognized T11 source manifest SHA-256; no downloads made")
    entries = check_manifest(json.loads(data))
    for row in entries:
        if row["origin"] == "local handoff helper":
            check_bytes((HERE / row["member"]).read_bytes(), row)
    rows = [row for row in entries if row["origin"] == "published Git blob"]
    require(len(rows) == 27, "T11 requires exactly 27 published source members")
    for row in rows:
        p = PurePosixPath(row["path"])
        require(str(p) == row["path"] and p.parts[0] == "research" and
                all(re.fullmatch(r"[A-Za-z0-9_.-]+",x) and x not in (".","..") for x in p.parts)
                and not re.search(r"fy.?2025|embargo", str(p), re.IGNORECASE)
                and row["commit"] in PUBLISHED_COMMITS and 0 < row["size"] <= 100000,
                f"Unsafe or unexpected pinned source: {row['member']}")
    return rows


def raw_url(row):
    return BASE + row["commit"] + "/" + quote(row["path"],safe="/")


def fetch(row, opener):
    url = raw_url(row)
    req = Request(url,headers={"User-Agent":"T11-thesis-review-stager/1","Accept":"application/octet-stream"})
    with opener.open(req,timeout=20) as response:
        require(response.geturl() == url and response.status == 200,
                f"Unexpected response for {row['member']}")
        return response.read(row["size"]+1)


def stage(output, local=None, opener=None):
    rows = source_rows()  # Manifest/source checks occur before any network or output staging.
    output = output.absolute()
    require(output.parent.is_dir() and not output.exists() and not output.is_symlink(),
            "Use a new ZIP filename in an existing directory")
    if local is not None:
        local = local.resolve(strict=True)
        require(local != output.parent and local not in output.parents,
                "Write outside the offline source directory")
    elif opener is None:
        opener = build_opener(NoRedirect())
    with TemporaryDirectory(dir=output.parent,prefix=".t11-stage-") as scratch:
        root = Path(scratch)
        for row in rows:
            if local is not None:
                f = local / row["member"]
                require(f.resolve(strict=True) == f and f.is_file(),
                        f"Missing or symlink source: {row['member']}")
                with f.open("rb") as input_file:
                    b = input_file.read(row["size"]+1)
            else:
                b = fetch(row, opener)
            check_bytes(b,row)
            f = root / row["member"]
            f.parent.mkdir(parents=True,exist_ok=True)
            f.write_bytes(b)
        zipped = root / "verified.zip"
        build(root/"notebook",root/"viewer",root/"notebook/evidence",zipped,root/"generated-manifest.json")
        require(hashes((root / "generated-manifest.json").read_bytes())[0] == T11_MANIFEST_SHA,
                "Generated manifest did not match the pinned copy")
        b = zipped.read_bytes()
        require(hashes(b)[0] == T11_ZIP_SHA, "Rebuilt ZIP did not match the pinned release bytes")
        with output.open("xb") as f:
            f.write(b)
    print(f"Verified {len(rows)} {'local' if local is not None else 'retrieved'} exact published members; wrote {output}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive",type=Path,help="a new output filename")
    p.add_argument("--local-source",type=Path,help="use an extracted T11 layout instead, without network access")
    a = p.parse_args()
    stage(a.archive,a.local_source)


if __name__ == "__main__":
    try:
        main()
    except (URLError,OSError,KeyError,ValueError) as exc:
        print(f"STAGING FAILED: {exc}",file=sys.stderr)
        sys.exit(1)
