# Research continuation checkpoint

Starting commit: 041afba8a10da9c2a962222e0a2644284a84fb7b.
Branch: experiment/thesis-continuation-20261117; parent research branch verified remotely.
Last historical completed experiment: Phase49b (run 41581789674).
Phase50 implementation and execution now in progress. Track A and original sources are immutable.

Artifact download blocker: authenticated ZIP requests redirect to Azure blob storage and
return HTTP403, including one approved outside-sandbox retry. Stop retries; retain
metadata and use the connected GitHub decoded-job-log interface to recover evidence.
The Phase49 decoded log was recovered from job 125736337527. It contains all 187
site identifiers, event observation fractions, rounded event/following water gains,
and the pre-Alaska-correction summary. This is not the original CSV/GeoJSON archive.

Bounded fallback: reconstruct county/site polygons from the original committed
sampling algorithm and exact county IDs in that log, using static Census 2023
geometry and incident dates in the immutable workbook. Audit new reads against
the logged values. Report remaining lineage gaps; do not silently certify exact
archive recovery. The primary analysis uses 63 original-anchor site keys; four
Minnesota extra sites are a separately reported sensitivity.

Planned stopping rules: Phase50 seasonal and interannual placebos at 30/60/80%
coverage; no funding-based tuning. Then bounded authoritative-source availability
checks across represented hazards, leakage and as-of review, explicit simulated
allocation policy comparison, synthesis and reproducible notebook. No additional
funding model selected without verified inputs and a justified validation protocol.
