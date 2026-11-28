# Build the T11 portable verification ZIP in one command

Keep these six trusted companion files together: `stage_handoff.py`, `build_handoff.py`,
`verify_handoff.py`, `SOURCE_MANIFEST.json`, `QUICKSTART.md`, and this guide.
The scripts use the Python 3.10+ standard library.

```sh
python -I -S stage_handoff.py thesis-review-verification.zip
```

Only this staging command needs internet access. It uses 27 bounded, read-only GETs
to `raw.githubusercontent.com` for the single repository `leonmarienga/ML-thesis`.
The full source manifest is pinned by SHA-256, and the commands refuse redirects,
moving branches, unlisted paths, symlinks and mismatched bytes. There is no
general URL, repository, token, download-all or Git clone option. It checks the
SHA-256, Git blob ID, and size of each requested file before the offline ZIP
builder runs. Restricted FY2025 outcomes, Word files, model weights, training
data, private or participant records are not requested. The stage is temporary;
the ZIP is written only after all inputs and the complete ZIP pass verification.

The expected ZIP SHA-256 is
`f86d5d0b1aae4d9865c134f08c853d9689a3f02f415224011616754786f61db0`.
For example, `sha256sum thesis-review-verification.zip` checks the release bytes
on Linux. Unzip into a fresh directory, then run `python -I -S verify_handoff.py`
there and follow the embedded quickstart. The finished ZIP needs no network.

Already have a trusted extracted T11 layout? Rebuild entirely offline:

```sh
python -I -S stage_handoff.py rebuilt.zip --local-source /path/to/extracted-T11
```

**Verified here:** real isolated offline CLI and fixture-backed HTTP transport
each rebuilt the exact same ZIP, with all expected source paths requested in
order on the HTTP path. Independent changed-manifest, changed-response and
redirect tests refused to continue. The contaminated manifest triggered zero
requests and produced no output. The first live attempt timed out. An independent live invocation then retrieved all 27 exact GETs and rebuilt the same SHA-256-pinned ZIP. Its clean-extraction verification passed. No browser, model training, or simulator run was repeated.
