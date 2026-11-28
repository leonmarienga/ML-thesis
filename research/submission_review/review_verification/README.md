# Offline supervisor-review smoke check

The reviewer entry points are the packaged HTML, editable Word document and tabletop script in a checkout or repository-shaped export of commit `b951c7a4883400c0d69a04d71ec91290351267e3`:

- `research/submission_review/decision_support_v2/index.html` (and its published `SHA256SUMS`).
- `research/submission_review/review_draft/Disaster Relief Thesis Supervisor Review Draft.docx` (and `REVIEW_SHA256SUMS`).
- `research/submission_review/tabletop/dry_run.py` and `dry_run.json`.

Run from this directory with Python 3.10+ and the complete recursive GitHub tree JSON for that commit:

```bash
python -I -S reviewer_smoke.py /path/to/review-export /path/to/tree.json
```

The checker itself uses only the Python standard library. If Node is installed, the two published JavaScript tests run against the actual packaged HTML using their test DOM. To also replay the published real-browser checks, supply a preinstalled dependency directory with Playwright 1.62.1 and @sparticuz/chromium 158.0.0. The tool does not install anything or deploy the viewer:

```bash
python -I -S reviewer_smoke.py /path/to/review-export /path/to/tree.json --browser-node-modules /path/to/node_modules
```

The retained `smoke-results.json` passed all checks, including real Chromium keyboard and layout at desktop and narrow widths. Manifest members are authenticated against the supplied tree's Git blob IDs before checking their SHA-256. The CLI emits JSON on success and errors on missing or modified required files. Trust the tree JSON only if downloaded independently for the pinned commit. If Node or browser dependencies are not supplied, the report explicitly marks the applicable check not run.

Checks are artifact integrity and function smoke tests, not an appraisal of scientific claims. Word parsing checks container CRC, XML and three resolved embedded PNGs; it does not rerender the document or show Word-client fidelity. Headless Chromium tests do not establish accessibility conformance, cross-browser/touch parity or user outcomes. The tabletop is a test of synthetic schedule and example record validation, with 0 participants, not a live study. This workflow does not rerun models, download data or access FY2025 outcomes.
