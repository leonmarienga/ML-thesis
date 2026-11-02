# Phase 28 — NASA OPERA DSWx-HLS catalog and raster-access feasibility

Successful GitHub Action: https://github.com/leonmarienga/ML-thesis/actions/runs/41421221883

## Design
A fixed outcome-blind pilot of 12 FEMA Flood disasters (six incident-year 2023 and six 2024), sampled by ascending disaster ID where NOAA report coordinates were available. Used NASA Common Metadata Repository collection `OPERA_L3_DSWX-HLS_V1` (`C2617126679-POCLOUD`). Each query covered NOAA point location, 45 days before incident, the capped incident window of <=21 days, and 14 days after incident. This is a **catalog feasibility** check only.

## Observed findings
- **12/12** completed CMR catalog queries.
- **12/12** had at least one before-incident granule and at least one during/after-incident granule, with advertised WTR-layer links on both sides.
- **387** granule metadata records in total; no query was truncated at 200.
- The single protected GeoTIFF bounded-range HTTP probe to PO.DAAC archive returned **400 application/json**. This is **not** enough to diagnose precisely whether Earthdata login, incorrect range support, a signed URL issue or some other problem caused access rejection. No raster decoded.
- **0** actual water-classification raster pixels read, **0** event-specific inundation polygons created, **0** exposed dwellings/people estimated. A catalog result does not imply usable cloud-free images.

## Scientific barriers
- DSWx maps observed surface water, not necessarily flooding. Detect temporary event water by comparing pre-event seasonal baseline, permanent water masks, quality classes, and common geo-grid.
- HLS optical observations can be obscured by clouds, shadows, snow/ice; camera date and processing dates must be distinguished.
- NOAA point coordinates are report locations, not spatially validated flood footprints or affected-home polygons.
- Source versions are retrospective; historical availability at response decision-time has not been established.
- Actual raster retrieval may require Earthdata credentials and permissions. Do not store users' Earthdata credentials in GitHub or public logs.
- The 12-case pilot is not representative coverage for all 92 FEMA Flood records.

## Research decision
The fixed pilot confirms promising **catalog availability**, but no observable flood extent or prediction-enrichment contribution yet. Next step is legitimate raster access under Earthdata authentication, then bounded GeoTIFF decode and pre/post quality masking. Accepted original six-band router 830/912, MAE $5.015M and R² .8868 stay unchanged.
