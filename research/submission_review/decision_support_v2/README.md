# Disaster relief decision support: offline evidence viewer v2

Open **index.html** in a local web browser. No backend, credentials, network requests, or participant logging are needed. Source links require internet only if followed. It is a read-only thesis research demonstrator, not authorization or dispatch software.

## What is new

- **Historical evidence:** 830/912 archived Track A routes, specific overlay dollar MAE, and a DR-4722 historical obligation example. A visible caution explains why the archived result is not unbiased forward prediction; the four target-conditioned selection lines have pinned links.
- **Interval coverage:** switch between the two historical nominal levels and compare all, below $50 million and $50 million or above. At nominal 90%, actual archive coverage was 281/308 overall and 0/9 in the upper group. No predictive guarantee is attached.
- **Matched veto fixture:** choose an ordered pair of the three scripted/optimization arms and a regime. Compare overall, high-vulnerability and least-served service with signed percentage-point gaps. Only the full matched 48-world comparison displays the archived scenario-resampling intervals; no subgroup interval is generated. Reversing the pair reverses the interval and the gap.
- **Preserved original frontier:** the 89 designs, four published preferences, all comparators and three regime filters are carried forward from the v1 prototype. This is a separate 48-case simulation, not another claim about the veto study.
- **Participant limits:** the proportional rule and veto are scripted, no consent or participant data are collected, and the viewer reports neither observed aid delivered nor a tested human–AI effect. Accounting labels, historical-model diagnostics and synthetic coverage are explicitly different kinds of evidence.

## Rebuild

The build expects Python 3, Git and a checkout containing commit b0b6edb229d112dd6d11896a512bf1112414e4f6. It reads historical summaries and baseline sources from pinned Git objects and the three hash-pinned aggregate/case-result CSVs under a local copy of the **Thesis Policy Veto Fixture** archive. No FY2025 assets are read. Raw historical cases are not embedded; the one cited historical dollar example, published aggregates, and no episode action traces are displayed.

    python build.py /path/to/pinned/thesis /path/to/unpacked/policy-veto-fixture
    node test_v1.js
    node test_v2.js
    python test_structure.py

The report and paired interval provenance are in the separate fixture bundle (archive SHA-256 a4f6d6b451ddca083104f34c7200b70ecdefb5cc2d5da020330e120f205e79d9), whose inputs are checked at both fixture and viewer build steps.

## Verification scope

A fresh deterministic build matched the packaged HTML byte-for-byte. The Node harness executes the embedded JavaScript and event handlers against a test DOM: 1,424 original frontier interactions, 24 new ordered arm/regime combinations, and both interval levels. Structural checks cover native dropdown labels, duplicate IDs, navigation links, ARIA targets, scoped table headers and captions. The local Chromium launcher lacks its browser binary (see browser-probe.txt), so this does not establish browser-specific screen layout, keyboard focus behavior, accessibility with assistive technology, or usability. There is no public deployment.


A later [headless Chromium acceptance pass](browser_qa/REPORT.md) verified the exact same viewer at desktop and narrow mobile widths using a separately pinned browser binary; no screen-reader or participant testing was performed.
