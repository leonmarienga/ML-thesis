# Operational tropical wind contours and later observed wind

Archived official 24- and 48-hour tropical wind forecasts provide a physically testable precursor to exposure estimates. The alternate NOAA ATCF and CO-OPS public endpoints supplied the forecast records and later archived station winds after initial individual advisory and buoy URLs returned HTTP 403. Three Gulf hurricane episodes were locked before station measurements were compared. [ATCF archive](https://ftp.nhc.noaa.gov/atcf/archive/), [CO-OPS historical wind access](https://api.tidesandcurrents.noaa.gov/api/prod/), [protocol](../BATCH4_PROTOCOL.md).

The analysis intended 1,128 storm–lead–site pairs and found 842 qualifying measurements across 27 distinct coastal stations. Eight of the 36 storm–station requests had no source records. The remaining missing pairs persisted within the 28 readable requests. Each match required compatible UTC valid and observed times. Matching offsets were zero to six minutes in the observed set. The equal-area comparison holds the forecast track, total sector area and station sample fixed. Its radius is the square root of the mean of the four squared quadrant radii. [Station audit](results/station_audit.csv), [pair-level evidence](results/wind_match_cases.csv), [input manifest](results/source_manifest.csv).

| Forecast geometry | Measured threshold matches | Positive signals below station threshold | Missed measured exceedances | Sensitivity | Precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Official quadrant contour | 31 | 148 | 3 | 91.2% | 17.3% |
| Equal-area circle | 32 | 175 | 2 | 94.1% | 15.5% |

The quadrant representation reduced below-threshold alerts by 27 and missed one additional measured threshold exceedance. Both geometries produced many station signals in which observed speed stayed below 34 kt. Publication times specified by the original forecasting procedure are distinct from nominal ATCF initialization times. The nominal 24- and 48-hour leads provide pre-observation ordering for this pilot. Wind height, averaging, siting, storm motion and the point-versus-area mismatch limit physical interpretation. Reused stations and pairs across lead times do not add independent hurricanes. Three episodes are insufficient to claim a generalized operational advantage or to calibrate an exposure model. [Contingencies by storm and lead](results/station_contingencies.csv), [NHC product definition](https://www.nhc.noaa.gov/aboutnhcprod.shtml).

The forecast-derived contours can be carried forward as a versioned physical representation for further validation. The measurements supply no validated damage, need or FEMA obligation labels.
