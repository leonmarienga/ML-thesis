# Phase 31 — Real Sentinel-2 clear-pixel observability screening

Successful run: https://github.com/leonmarienga/ML-thesis/actions/runs/41431485155

Fixed cohort: same 12 FEMA Flood disasters from 2023–24 examined in Phase 28–30. Previous public STAC search found 283 scene records. Predeclared bounded sample: <=2 MGRS tiles, <=4 pre and <=4 post images per tile for each case, 112 real local Scene Classification Layer rasters successfully decoded. **224** same-tile before/after pairs assessed using real local 4 x 4 km NOAA report-point clear-pixel coverage.

Quality gate: both scenes >=20% strictly clear, shared >=30% pixel footprint classified as SCL vegetation, bare land, or water (classes 4/5/6), excluding snow/ice, cloud/shadow, unclassified and nodata. Chosen pair ordered by highest shared observability and temporal proximity, NEVER by amount of detected water.

**6/12 disasters** had >=1 suitable pair and both B03 green and B08 near infrared bands read successfully. Remaining 6 cases are **unobservable under this bounded selection**, not zero flood severity.

| FEMA disaster | Valid paired pixels | Candidate new SCL water class (km²) | New SCL water with post NDWI>=0.20 (km²) |
|---|---:|---:|---:|
| 4709 | 88.56% | 0.0316 | 0.0100 |
| 4711 | 100% | 0.0044 | 0.0016 |
| 4758 | 99.89% | 0.0072 | 0.0000 |
| 4781 | 98.93% | 0.0420 | 0.0136 |
| 4787 | 99.45% | 0.0000 | 0.0000 |
| 4789 | 95.70% | 0.0000 | 0.0000 |
| **Total (separate sample windows)** | | **0.0852** | **0.0252** |

**These areas are candidate SCL water-class transitions, NOT confirmed new flood extent.** Each covers only a local 4km x 4km window around one NOAA report location, not an incident polygon. Independent disaster attribution, seasonal background/permanent-water baseline, land surface change quality checks, historical version availability and exposed infrastructure spatial intersection are missing. Two apparent zero transitions on quality-passing pairs are *observed zero candidate changes*, not proof no flood occurred elsewhere. Six rejected cases have no valid measurement and must NOT be coded as zero severity.

The selection is exploratory and uses the previously inspected same twelve cases, so results cannot be treated as independent confirmation.

No funding model was changed. Accepted router 830/912, overall MAE $5.015M and R² ~.8868 remain unchanged.

Full 112-scene quality audit, 224 pair acceptance/rejection records, six selected candidate scenes and case measurements are packaged in GitHub artifact `phase31-sentinel2-observability`.
