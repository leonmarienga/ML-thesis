"""Test T11's actual offline CLI and constrained HTTP transport with recorded exact blobs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage_handoff as stage
from verify_handoff import FILES, HERE, SOURCE_MANIFEST, require


class Response:
    status = 200
    def __init__(self, url, data):
        self.url = url
        self.data = data
    def geturl(self):
        return self.url
    def read(self, count):
        return self.data[:count]
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


class RecordedHttp:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []
    def open(self, request, timeout):
        url = request.full_url
        require(request.get_method() == "GET" and timeout == 20 and url in self.responses,
                f"Unexpected HTTP request: {url}")
        self.calls.append(url)
        return Response(url, self.responses[url])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive",type=Path)
    zip_path = p.parse_args().archive.resolve(strict=True)
    archive = zip_path.read_bytes()
    require(hashlib.sha256(archive).hexdigest() == stage.T11_ZIP_SHA, "Test ZIP is not the pinned version")
    with TemporaryDirectory(dir=HERE,prefix="test-stage-") as d:
        temp = Path(d)
        snapshot = temp / "extracted"
        with ZipFile(zip_path) as z:
            expected = FILES | {SOURCE_MANIFEST}
            require(len(z.namelist()) == len(expected) and set(z.namelist()) == expected,
                    "Unexpected test ZIP inputs")
            z.extractall(snapshot)
        offline = temp / "offline.zip"
        run = subprocess.run([sys.executable,"-I","-S",str(HERE/"stage_handoff.py"),str(offline),
                              "--local-source",str(snapshot)],capture_output=True,text=True,timeout=30)
        require(run.returncode == 0 and offline.read_bytes() == archive, run.stderr)
        manifest = json.loads((HERE/SOURCE_MANIFEST).read_bytes())
        responses = {
            f"https://raw.githubusercontent.com/leonmarienga/ML-thesis/{r['commit']}/{r['path']}":
            (snapshot/r["member"]).read_bytes()
            for r in manifest["files"] if r["origin"] == "published Git blob"
        }
        require(len(responses) == 27, "Expected exactly 27 unique URLs")
        http = RecordedHttp(responses)
        remote_fixture_zip = temp/"remote_fixture.zip"
        stage.stage(remote_fixture_zip,opener=http)
        require(remote_fixture_zip.read_bytes() == archive and http.calls == list(responses),
                "Wrong URL sequence or HTTP-path output")
        wrong = dict(responses)
        wrong[next(iter(wrong))] += b"\n"
        target = temp/"bad.zip"
        wrong_http = RecordedHttp(wrong)
        try:
            stage.stage(target,opener=wrong_http)
        except ValueError as e:
            require("Source digest mismatch" in str(e), "Wrong corrupt response error")
        else:
            raise ValueError("Modified HTTP response accepted")
        require(len(wrong_http.calls) == 1 and not target.exists(),"Corrupt stage did not stop")
        bad_manifest = temp/"bad-manifest"
        bad_manifest.mkdir()
        shutil.copy2(HERE/SOURCE_MANIFEST,bad_manifest/SOURCE_MANIFEST)
        path = bad_manifest/SOURCE_MANIFEST
        path.write_bytes(path.read_bytes().replace(b"ML-thesis",b"other-repo",1))
        stage.HERE = bad_manifest
        never = RecordedHttp({})
        try:
            stage.stage(target,opener=never)
        except ValueError as e:
            require("Unrecognized T11 source manifest" in str(e), "Wrong changed manifest error")
        else:
            raise ValueError("Modified manifest accepted")
        require(not never.calls and not target.exists(), "Corrupt manifest caused I/O")
        try:
            stage.NoRedirect().redirect_request(None,None,302,None,None,"https://other.invalid/")
        except ValueError:
            pass
        else:
            raise ValueError("Off-host redirect was accepted")
    print("PASS: isolated CLI rebuild from local extraction; exactly 27 raw-GitHub GET paths against recorded fixtures; corrupt response and corrupt manifest rejected; redirects rejected; identical archive bytes.")


if __name__ == "__main__":
    main()
