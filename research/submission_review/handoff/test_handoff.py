"""Inspect exact ZIP membership, extract safely, and run the reviewer entry points."""
import argparse
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_STORED

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_handoff import FILES, HERE, SOURCE_MANIFEST, hashes, require


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("--browser-runtime", type=Path, help="optional local F10 browser harness runtime for archived HTML")
    a = p.parse_args()
    expected_members = FILES | {SOURCE_MANIFEST}
    with TemporaryDirectory(dir=HERE,prefix="t11-test-") as d, ZipFile(a.archive) as z:
        root = Path(d)/"unzip"
        root.mkdir()
        names = z.namelist()
        require(set(names) == expected_members and len(names) == len(expected_members),
                "Duplicate, extra or missing ZIP members")
        require(z.testzip() is None, "ZIP CRC mismatch")
        for item in z.infolist():
            require(stat.S_ISREG(item.external_attr >> 16) and item.date_time == (1980,1,1,0,0,0)
                    and item.compress_type == ZIP_STORED, "Unexpected ZIP metadata")
        z.extractall(root)  # Paths above must match the exact allowlist.
        require((root/SOURCE_MANIFEST).read_bytes() == (HERE/SOURCE_MANIFEST).read_bytes(),
                "Archive and external manifests differ")
        command = [sys.executable,"-I","-S","verify_handoff.py"]
        def run(env=None):
            return subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,timeout=120)
        good = run()
        require(good.returncode == 0, good.stderr)
        checks = json.loads(good.stdout)
        require(checks["verified_manifest_files"] == len(FILES), "Incorrect verification file count")
        no_node = run(dict(os.environ,PATH=""))
        require(no_node.returncode == 0 and json.loads(no_node.stdout)["node_interactions"].startswith("not run:"),
                "Node-free path did not record skipped tests correctly")
        for target in ("viewer/index.html","notebook/evidence/forward_t5.csv"):
            bad_input = root/target
            exact = bad_input.read_bytes()
            bad_input.write_bytes(exact+b"\n")
            bad = run()
            require(bad.returncode == 1 and f"Source digest mismatch: {target}" in bad.stderr and not bad.stdout,
                    f"Verifier did not reject mutation of {target}")
            bad_input.write_bytes(exact)
        # Rebuild without a repository and with unlisted files deliberately present.
        sentinel = root/"notebook/evidence"/"PRIVATE_SENTINEL_DO_NOT_PACKAGE.txt"
        sentinel.write_text("This test creates no private or restricted data.\n")
        rebuild = Path(d)/"repeat.zip"
        build = subprocess.run(
            [sys.executable,"-I","-S",str(HERE/"build_handoff.py"),
             "--notebook","notebook","--viewer","viewer","--evidence","notebook/evidence",
             "--archive",str(rebuild)],cwd=root,capture_output=True,text=True,timeout=120)
        require(build.returncode == 0,build.stderr)
        require(rebuild.read_bytes() == a.archive.read_bytes(), "Archive not byte-identical on rebuild")
        output = {"zip_sha256":hashes(a.archive.read_bytes())[0],"zip_size":a.archive.stat().st_size,
                  "zip_members":len(names),"fresh_extraction":checks,"fallback_without_node":"pass",
                  "tamper_rejected":["viewer/index.html","notebook/evidence/forward_t5.csv"],
                  "deterministic_rebuild_with_extra_input_ignored":"pass"}
        if a.browser_runtime:
            report = Path(d)/"f10.json"
            harness = HERE.parent/"f10-offline-viewer"/"capture_f10.cjs"
            browser = subprocess.run(["node",str(harness),str(root/"viewer/index.html"),str(a.browser_runtime.resolve()),
                                      str(Path(d)/"f10.png"),str(report)],cwd=root,capture_output=True,text=True,timeout=120)
            require(browser.returncode == 0,browser.stderr)
            js = json.loads(report.read_bytes())
            output["supplemental_actual_chromium"] = {
                "artifact":js["artifact"],"state_proof":js["capture"]["proof_of_change"],
                "page_errors":js["capture"]["page_errors"],"http_requests":js["capture"]["http_requests"]}
        print(json.dumps(output,sort_keys=True,indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError,ValueError,KeyError,subprocess.SubprocessError) as exc:
        print(f"TEST FAILED: {exc}",file=sys.stderr)
        sys.exit(1)
