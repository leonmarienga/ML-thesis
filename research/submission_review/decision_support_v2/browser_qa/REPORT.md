# Offline decision-support viewer browser acceptance

Tested the exact standalone HTML committed at `6ada0ab4c7a7a45360d7697d2810a3e5bb0a2d48`, SHA-256 `9ddf0fedf618e7e9e3df5a194890ecb1f88883d4a48a4917cba2d423f7e84203`. The check loaded the file directly with Playwright 1.62.1 and headless Chromium 158.0.8127.0, packaged as `@sparticuz/chromium@158.0.0` on Linux. It did not deploy the app or send data to a live service.

| Acceptance check | Desktop 1440×1000 | Narrow 390×844 |
| --- | --- | --- |
| Page loads with archived obligation and interval evidence | Pass | Pass |
| Tab reaches the interval switch; arrow keys select 80% and restore 90% | Pass | Pass |
| Keyboard changes the road subset; full comparison restores 48 matched worlds | Pass | Pass |
| Subset does not display a pooled interval | Pass | Pass |
| Keyboard changes the original decision-frontier comparator | Pass | Pass |
| No document-level horizontal overflow, uncaught page errors or HTTP requests | Pass | Pass |

Screenshots were checked as research-viewer rendering evidence. This is functional and layout acceptance in a single headless engine, not accessibility conformance, native browser or touch parity, actual participant usability, or operational validation. No user data or FY2025 outcomes were accessed. The view remains read-only and the dispatch warning remains in place.

To replay on an install-compatible Linux host with Node 24: in this folder, run `npm ci --ignore-scripts` and `node browser_qa.cjs ../index.html`. The script verifies the accepted HTML hash before opening the page, blocks HTTP requests, emits JSON and writes screenshots in the working directory.
