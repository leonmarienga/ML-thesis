# T11 Portable verification handoff

**Status:** The scripts, manifests and evidence record are maintained here. The assembled ZIP was built locally, extracted and verified; the executable ZIP itself is not distributed in this directory. This is a narrow review bundle with no repository checkout, no Word binary, model files, private records, FY2025 outcomes, or participant data.

| Component | Exact input and check | Verified outcome | Limit |
| --- | --- | --- | --- |
| Portable layout | Explicit member list, Git blob/SHA-256/size, fixed ZIP metadata | 30 entries: 29 inventoried files plus the source manifest. One local rebuild from freshly extracted inputs was byte-identical and ignored a deliberately unlisted sentinel. | Bundle integrity does not establish scientific validity. |
| Reviewer notebook | Four exact published notebook artifacts at `fb314e0edb71faccf9b7cbb196b0a245459aa696`, original manifest at SHA-256 `006773bfcb065f426fd79384c7d275fe102eda2c0b1450d286a4a82a9510cb32` | Python 3.12.14, isolated standard library only: all 8 code cells and nine-claim ledger exactly matched published result bytes. | Not a full historical training/data re-creation; the published executed notebook is the pre-existing in-process Jupyter result. Batch 5 and 11 replay limits remain. |
| Offline evidence viewer | Four exact published viewer files at `b951c7a4883400c0d69a04d71ec91290351267e3`; shipped index hash `9ddf0fedf618e7e9e3df5a194890ecb1f88883d4a48a4917cba2d423f7e84203` | HTML structure passed; Node 24.22.3 executed 1,424 frontier combinations, 24 arm/regime comparisons and both interval levels. The no-Node fallback passed and marked interaction tests not run. | DOM tests are not assistive technology, usability or browser acceptance. |
| Fresh extracted viewer (optional QA) | Opened the extracted file directly in headless Chromium 158.0.8127.0 with existing F10 harness | Keyboard selection switched 48 matched simulations to the 15 closure subset; no attempted HTTP or uncaught page errors. | One native keyboard workflow on one Linux engine. No new simulator/field or participant assessment. |
| Failure paths | Freshly extracted viewer HTML and forward evidence individually changed | Both checks failed before notebook/JS execution; no verified result emitted. | Trust the external release checksum and a trusted executable copy. A malicious repackager could replace its manifest or verifier. |

The outer source manifest lists paths and immutable commits for each original Git blob. The four viewer blob IDs were also matched against the previously verified review tree; notebook IDs matched the prior publication tree. The inner manifest remains the exact originally published notebook manifest. Data are existing published aggregates and the published stage-only pairs; the bytes are not fetched from a moving branch. See `TEST_RESULTS.json` and `SHA256SUMS`.

## Build again from staged named inputs

Create the two source directories with the named published files listed under `VIEWER` and `NOTEBOOK` in `build_handoff.py`, or reuse a verified extraction with the same structure. The third input directory must contain the original notebook `manifest.json` and 18 exact files. If not already staged, the existing [notebook staging guide](https://github.com/leonmarienga/ML-thesis/blob/fb314e0edb71faccf9b7cbb196b0a245459aa696/research/submission_review/reviewer_notebook/README.md) has a narrow fetcher for just those inputs. No full repository clone is needed. Building is offline:

```sh
python -I -S build_handoff.py \
  --notebook /path/to/staged/notebook \
  --viewer /path/to/staged/viewer \
  --evidence /path/to/staged/evidence \
  --archive thesis-review-verification-local.zip \
  --manifest-out SOURCE_MANIFEST-local.json
python -I -S test_handoff.py thesis-review-verification-local.zip
```

The optional browser check is development QA and needs the separate existing F10 harness and pinned browser runtime. Neither is needed or included in the ZIP. Reviewers start with the embedded `QUICKSTART.md` and `python -I -S verify_handoff.py` from an extracted directory. For a fresh one-command build from the exact public sources, see [staging](STAGING.md). An independent live invocation retrieved all 27 exact Git files, reproduced the ZIP digest and passed clean extraction without training or touching reserved outcomes.
