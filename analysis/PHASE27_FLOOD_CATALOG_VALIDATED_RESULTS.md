# Phase 27 — Validated flood-map source catalogs, negative availability finding

GitHub Action: https://github.com/leonmarienga/ML-thesis/actions/runs/41419794364

## Validated catalog extraction

- Copernicus activations: six paginated pages, **1,068/1,068** reported records retrieved. Of 552 activations dated FY2019–2024, **one** catalog record identified the United States; **none** met the flood/storm category/title filter. Therefore **no temporal/geographic candidate for the 92 US FEMA Flood cases was found in this *specific* Copernicus catalog**. This does not describe any other Copernicus/NASA/USGS source coverage.
- University of Alabama SDML USFIMR original HTML listing: **42** parsed historical map rows, years **1997–2017**. Thus the original archive table contains no 2019–2024 observation-year rows. The alternative waterserv.ua.edu site could not be reached from the runner, so its content and coverage remain unknown.
- Polygon positive-control attempted five 2017 archive ZIP URLs, all using `ftp://sdml-server.ua.edu/...`; Python requests cannot fetch FTP URLs. **No geometry has been successfully downloaded or validated**. The test therefore validates catalog parsing, *not polygon ingestion*.
- Observed 2019–2024 inundation polygons linked to FEMA disasters: **zero retrieved**; true mapping availability across all possible archives remains **unknown**. No flooded households or properties computed.
- No funding target used for matching, model training performed, or accepted 830/912 six-band router and $5.015M model altered.

## Decision and next source

Stop treating static FEMA flood-hazard maps as realized inundation. Do not claim footprint-based exposure based on a mere AOI or catalog centroid.

The next research candidate is NASA OPERA DSWx-HLS/DSWx-S1 for 2023–2024 floods, where temporal coverage exists. Before attributing event flooding, confirm successful granule access, pre-event baseline/seasonal surface-water mask, cloud/quality mask, geographic intersection with independent reported flood areas, and collection/version dates. A DSWx water pixel alone is not event-specific flood evidence.

Further NOAA-derived exposure can use population by affected county as a **proxy**, but never call those residents actually inundated.
