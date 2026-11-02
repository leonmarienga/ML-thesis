# Phase 29 — NASA OPERA real raster access gate

Completed run: https://github.com/leonmarienga/ML-thesis/actions/runs/41422398790

Previous fixed 12-flood 2023–2024 cohort has 387 OPERA scene metadata records. Same-MGRS pre/post selection generated 2,341 candidate pairs. Example: FEMA 4718, tile T14TNQ, acquisitions 2023-04-07 and 2023-04-10.

The protected archive GeoTIFF probe returned HTTP 400 application/json without NASA Earthdata access. No actual GeoTIFF was downloaded; zero WTR pixel classes read; no inundation polygon, area or exposed housing measured. The Python script implements an optional bounded real-raster decoder if authorized access is later configured. The current workflow deliberately does not provide credentials. Never include access tokens in code, issue descriptions or logs.

A water classification map is not inherently a validated flood event. Proper analysis still requires pre-event seasonal water baseline, QA and cloud masking, geographic and event attribution, and separate exposure geometry. Model training was not conducted. Accepted six-band router 830/912 and $5.015M MAE remain unchanged.
