# Phase 30 — Public Sentinel-2 actual raster decoding, optical-quality limitation

Successful workflow: https://github.com/leonmarienga/ML-thesis/actions/runs/41430052004

**Source:** Element 84 Earth Search public Sentinel-2 Collection 1 L2A Cloud-Optimized GeoTIFFs, **not NASA OPERA DSWx**. The fixed 12-event FY2023–2024 FEMA Flood pilot from Phase 28 returned 283 Sentinel-2 STAC scenes across 12 successful queries and 387 same-tile pre/post candidate pairs.

**Successful raster positive control:** The first selected pair was for FEMA disaster 4718, MGRS tile 14TNR, acquired April 7 and April 10, 2023. Real SCL (20 m), green (B03), and near-infrared (B08) pixels were successfully read, reprojected into a common SCL grid, and evaluated in a ~2 km vicinity of the NOAA report point. GeoTIFF's projected CRS: EPSG:32614. Pixels decoded: 40,000 per scene.

**Critical image-quality finding:** The pre-event SCL image classified all 40,000 sampled pixels as code 11 (snow or ice); post-event classified 33,021 as snow/ice, 5,873 as nonvegetated, 597 as water and 509 as unclassified. Our specified paired clear-pixel classes were 4,5,6,7; therefore **ZERO paired valid/clear pixels** exist. Candidate water-change area in the code is numerically 0 km² solely because there are no valid comparison pixels: **THIS IS UNOBSERVABLE, NOT A NEGATIVE FLOOD DETECTION**. No event-specific inundation area was established.

**Workflow status:** Public COG reading has been demonstrated. NASA protected archive raster access is still blocked; this open-data experiment does not solve Earthdata authentication.

**Next gate:** Choose a scene pair with actual cloud/snow-free coverage at the same location and common tile, or select a different fixed flood case according to a predeclared quality threshold, preserving rejected cases in the audit. Establish seasonal permanent/background surface-water masks before inferring transient flooding; SCL plus exploratory NDWI is not validated inundation. Do not sample based on desired positive flood water gain, only image observability.

No flooded homes, damages, resource needs, or funding-model improvements have been measured. The frozen accepted router 830/912 and final MAE ~$5.015M remain unchanged.
