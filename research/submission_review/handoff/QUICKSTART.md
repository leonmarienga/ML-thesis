# Offline thesis reviewer handoff

This review-only bundle contains the exact standalone evidence viewer, the scoped reproduction notebook and its 18 pinned evidence files. It does not contain a Word draft, model weights, training datasets, private records, or reserved FY2025 outcomes.

1. Unzip into a new directory. From that directory, run:
   ```sh
   python -I -S verify_handoff.py
   ```
   Python 3.10+ needs no extra packages. Node, if installed, also replays the viewer's embedded interaction tests against a test DOM. The verifier checks exact evidence, rebuilds the nine-claim ledger in a temporary folder and inspects the viewer's HTML structure. Nothing is fetched, trained, or submitted.
2. Open `viewer/index.html` as a local file in a desktop browser. Try the historical interval switch and the **Road conditions** selector. Historical FEMA obligations are accounting records, not disaster losses, unmet needs or aid delivered. The policy comparisons are simulations with scripted decision rules, not human or field results.
3. If you use Jupyter, open `notebook/thesis-review.ipynb` from its own directory and choose **Restart Kernel and Run All**. The `notebook/evidence/` folder already sits next to it. The archived executed notebook records in-process Jupyter execution only. The bundled `claims.json` is a small machine-readable result ledger.

`SOURCE_MANIFEST.json` records the exact repository path, commit, SHA-256, Git blob and size for each included published artifact; the two new handoff helper files are marked local. Verify a downloaded ZIP against the release's externally supplied SHA-256, if available, before trusting executable contents. The bundled manifest can detect corruption but cannot authenticate an archive whose manifest and verifier have themselves been replaced. Only execute a trusted copy.

This is not the original full training or simulation pipeline, an embargo release, a participant study, or approval of the thesis. Locally nonmatching archive comparisons in Batch 5 and Batch 11 remain visible in the pinned evidence notes.
