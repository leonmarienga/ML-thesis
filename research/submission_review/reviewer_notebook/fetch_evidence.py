#!/usr/bin/env python3
"""Prepare or verify the allowlisted reviewer notebook evidence, not full repo data."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from tempfile import NamedTemporaryFile
from urllib.parse import quote
from urllib.request import urlopen

MANIFEST_SHA256 = "006773bfcb065f426fd79384c7d275fe102eda2c0b1450d286a4a82a9510cb32"
REPOSITORY="leonmarienga/ML-thesis"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok,detail):
    if not ok: raise ValueError(detail)


def read_manifest(path):
    data=path.read_bytes()
    require(digest(data)==MANIFEST_SHA256,"Unrecognized manifest digest")
    m=json.loads(data)
    require(m["schema_version"]==1 and m["repository"]==REPOSITORY,"Wrong manifest identity")
    seen=set()
    for r in m["files"]:
        name=r["name"]
        components=PurePosixPath(r["path"]).parts
        require(bool(re.fullmatch(r"[a-z0-9_]+\.(?:csv|json|md)",name)) and name not in seen,"Invalid or duplicate evidence name")
        require(components and components[0]=="research" and ".." not in components,"Invalid repository path")
        require(bool(re.fullmatch("[a-f0-9]{40}",r["commit"])) and bool(re.fullmatch("[a-f0-9]{40}",r["git_blob"])) and bool(re.fullmatch("[a-f0-9]{64}",r["sha256"])) and 0<r["size"]<=100000,"Invalid pinned identity or size")
        seen.add(name)
    return data,m["files"]


def write_atomic(path,b):
    with NamedTemporaryFile(dir=path.parent,delete=False) as f:
        temporary=Path(f.name)
        try:
            f.write(b)
            f.flush()
            f.close()
            os.replace(temporary,path)
        finally:
            temporary.unlink(missing_ok=True)


def record_ok(row,raw):
    actual_git=hashlib.sha1(f"blob {len(raw)}\0".encode()+raw).hexdigest()
    return len(raw)==row["size"] and digest(raw)==row["sha256"] and actual_git==row["git_blob"]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("destination",type=Path)
    p.add_argument("--manifest",type=Path,default=Path(__file__).with_name("manifest.json"))
    p.add_argument("--check-only",action="store_true",help="verify local files without any network requests")
    a=p.parse_args()
    manifest,rows=read_manifest(a.manifest)
    path=a.destination
    require(not path.is_symlink(),"Destination must not be a symlink")
    if not a.check_only:
        path.mkdir(parents=True,exist_ok=True)
    require(path.is_dir(),"Missing evidence directory")
    stored=path/"manifest.json"
    require(not stored.is_symlink(),"Refusing a symlinked destination manifest")
    if stored.exists(): require(stored.is_file() and stored.read_bytes()==manifest,"Unexpected destination manifest")
    if a.check_only: require(stored.is_file(),"Missing staged manifest")
    for row in rows:
        leaf=path/row["name"]
        require(not leaf.is_symlink(),"Refusing symlink: "+leaf.name)
        if leaf.exists():
            require(leaf.is_file() and leaf.stat().st_size==row["size"] and record_ok(row,leaf.read_bytes()),"Existing copy differs: "+leaf.name)
        else:
            require(not a.check_only,"Missing file: "+leaf.name)
            url=f'https://raw.githubusercontent.com/{REPOSITORY}/{row["commit"]}/{quote(row["path"])}'
            with urlopen(url,timeout=45) as response:
                data=response.read(row["size"]+1)
            require(record_ok(row,data),"Refetched copy differs: "+leaf.name)
            write_atomic(leaf,data)
        print("PASS",leaf.name)
    if not a.check_only and not stored.exists():
        write_atomic(stored,manifest)
    print(f"PASS: {len(rows)} exact evidence files; no unlisted data requested.")


if __name__=="__main__":
    main()
