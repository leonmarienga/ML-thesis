#!/usr/bin/env python3
"""Offline artifact smoke test for one supervisor-review GitHub commit."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET
from zipfile import ZipFile

PIN = "b951c7a4883400c0d69a04d71ec91290351267e3"
REL = Path("research/submission_review")
APP = REL / "decision_support_v2"
WORD = REL / "review_draft"
TABLE = REL / "tabletop"
HTML = "9ddf0fedf618e7e9e3df5a194890ecb1f88883d4a48a4917cba2d423f7e84203"
DOCX = "Disaster Relief Thesis Supervisor Review Draft.docx"
DOC_HASH = "d86bf71f933ebf694a9d830ee90dd9007af251668aad8f7b49ee9444c4e7767d"
APP_FILES = set("QA.txt README.md base_app.js base_build.py base_template.html browser-probe.txt build.py index.html panels.html test_structure.py test_v1.js test_v2.js v2.css v2_app.js".split())
EXPECTED_DRY = {
    "balanced_exposures_per_scenario_condition": 2,
    "closed_zone_retained": True,
    "negative_tests": "invalid route, stock, quantity and acceptance rejected",
    "participants": 0, "scenario_ids": 6, "schedule_slots": 4,
    "status": "simulation_only",
}
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def require(condition, problem):
    if not condition:
        raise ValueError(problem)


def run(command, cwd, env=None):
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, timeout=120)
    if result.returncode:
        raise ValueError(f"{command[0]} exited {result.returncode}: {result.stderr.decode('utf-8', 'replace').strip()}")
    return result.stdout


class Pin:
    def __init__(self, root, tree_path):
        self.root = root.resolve(strict=True)
        d = json.loads(tree_path.read_bytes())
        require(d["sha"] == PIN and d["truncated"] is False, "Expected the complete GitHub tree at the specified review commit")
        self.entries = {row["path"]: row for row in d["tree"] if row["type"] == "blob"}

    def read(self, rel):
        path = self.root / rel
        require(path.resolve(strict=True) == path and path.is_file(), f"Not a regular file in the pinned layout: {rel}")
        b = path.read_bytes()
        obj = self.entries[rel.as_posix()]
        git = hashlib.sha1(f"blob {len(b)}\0".encode("ascii") + b).hexdigest()
        require(obj["size"] == len(b) and obj["sha"] == git, f"Git blob mismatch: {rel}")
        return b

    def manifest(self, folder, name, expected):
        found = {}
        for line in self.read(folder / name).decode("utf-8").splitlines():
            m = re.fullmatch(r"([0-9a-f]{64}) [ *](?:\./)?(.+)", line)
            require(m is not None, f"Invalid manifest line in {name}")
            checksum, leaf = m.groups()
            require(leaf in expected and leaf not in found, f"Unexpected or duplicate manifest item: {leaf}")
            require(hashlib.sha256(self.read(folder / leaf)).hexdigest() == checksum, f"SHA-256 mismatch: {leaf}")
            found[leaf] = checksum
        require(set(found) == expected, f"Incomplete manifest: {name}")
        return found


def word_structure(path):
    with ZipFile(path) as z:
        names = z.namelist()
        require(len(names) == len(set(names)) and z.testzip() is None, "Word ZIP duplicate or CRC failure")
        parsed = {name: ET.fromstring(z.read(name)) for name in names if name.endswith((".xml", ".rels"))}
        require({"[Content_Types].xml", "_rels/.rels", "word/document.xml", "word/styles.xml"}.issubset(parsed), "Missing Word XML parts")
        doc = parsed["word/document.xml"]
        require(doc.tag == f"{{{W}}}document" and doc.find(f"{{{W}}}body") is not None, "Missing Word document body")
        relationships = {n.attrib["Id"]: n.attrib for n in parsed["word/_rels/document.xml.rels"]}
        embeds = [n.attrib[f"{{{R}}}embed"] for n in doc.findall(f".//{{{A}}}blip")]
        require(len(embeds) == len(set(embeds)) == 3, "Expected three distinct embedded image relationships")
        media = {}
        for rid in embeds:
            r = relationships[rid]
            target = PurePosixPath(r["Target"])
            require(r["Type"] == R+"/image" and r.get("TargetMode", "Internal") == "Internal" and target.parts == ("media", target.name), f"Unexpected media relationship: {rid}")
            f = "word/" + target.as_posix()
            data = z.read(f)
            require(data.startswith(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"), f"Invalid PNG header: {f}")
            width, height = struct.unpack_from(">II", data, 16)
            require(width > 0 and height > 0, f"Invalid image dimensions: {f}")
            media[f] = [width, height]
        require(set(media) == {n for n in names if n.startswith("word/media/")}, "Word archive includes unexpected media")
        return {"zip_entries": len(names), "parsed_xml_parts": len(parsed), "embedded_png_pixels": media}


def browser_acceptance(pin, node, modules):
    source = pin.read(APP / "browser_qa/browser_qa.cjs")
    expected = json.loads(pin.read(APP / "browser_qa/package.json"))["dependencies"]
    for p, version in expected.items():
        installed = json.loads((modules / p / "package.json").read_bytes())["version"]
        require(installed == version, f"Browser dependency {p} must be {version}, got {installed}")
    with TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix="browser-qa-") as temporary:
        work = Path(temporary)
        (work / "node_modules").symlink_to(modules, target_is_directory=True)
        (work / "browser_qa.cjs").write_bytes(source)
        env = dict(os.environ, TMPDIR=temporary, XDG_CACHE_HOME=temporary, XDG_CONFIG_HOME=temporary,
                   HOME=temporary, PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD="1")
        result = json.loads(run([node, "browser_qa.cjs", str(pin.root / APP / "index.html")], work, env=env))
        require(result["hash"] == HTML, "Browser replay used wrong artifact")
        require([(r["viewport"]["width"], r["viewport"]["height"]) for r in result["checks"]] == [(1440, 1000), (390, 844)], "Browser replay omitted a required width")
        for r in result["checks"]:
            require(r["dims"]["viewport"] == r["dims"]["body"] == r["viewport"]["width"], "Browser showed horizontal overflow")
            require(all(r[k] is True for k in ("keyboardInterval", "keyboardRoadFilter", "keyboardComparator")) and not r["pageErrors"] and not r["httpRequests"], "Browser interaction, offline or error check failed")
            require((work / f"qa-{r['viewport']['width']}.png").is_file(), "Browser screenshot was not written")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_root", type=Path, help="offline repository-shaped tree with the pinned review artifacts")
    parser.add_argument("github_tree_json", type=Path, help="GitHub recursive Git tree JSON retrieved by commit (no model/data files needed)")
    parser.add_argument("--browser-node-modules", type=Path, help="optional preinstalled, exact local browser QA dependencies")
    a = parser.parse_args()
    pin = Pin(a.review_root, a.github_tree_json)
    app = pin.manifest(APP, "SHA256SUMS", APP_FILES)
    doc = pin.manifest(WORD, "REVIEW_SHA256SUMS", {DOCX})
    require((app["index.html"], doc[DOCX]) == (HTML, DOC_HASH), "Pinned deliverable hash drift")
    report = {"commit": PIN, "sha256": {"decision_support_v2/index.html": HTML, DOCX: DOC_HASH},
              "published_manifest_members_verified": {"app": len(app), "word": len(doc)},
              "word": word_structure(pin.root / WORD / DOCX)}
    p = pin.read(TABLE / "dry_run.json")
    pin.read(TABLE / "dry_run.py")
    require(json.loads(p) == EXPECTED_DRY, "Wrong tabletop dry-run contract")
    require(run([sys.executable, "-I", "-S", str(pin.root / TABLE / "dry_run.py")], pin.root / TABLE) == p,
            "Tabletop dry run differed from archived JSON (or its synthetic-only boundary)")
    report["tabletop"] = {"fresh_matches_archive_bytes": True, **EXPECTED_DRY}
    report["html_structure"] = run([sys.executable, "-I", "-S", "test_structure.py"], pin.root / APP).decode().strip()
    node = shutil.which("node")
    report["node"] = ({"version": run([node, "--version"], pin.root / APP).decode().strip(),
                       **{f: run([node, f], pin.root / APP).decode().strip() for f in ["test_v1.js", "test_v2.js"]}}
                      if node else {"status": "not run: Node missing"})
    if a.browser_node_modules:
        require(node is not None, "--browser-node-modules requires Node")
        report["real_browser"] = browser_acceptance(pin, node, a.browser_node_modules.resolve(strict=True))
    else:
        report["real_browser"] = {"status": "not run: optional preinstalled dependencies not supplied"}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, ET.ParseError, subprocess.SubprocessError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
