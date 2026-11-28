# Michigan Bar: public forecast clock and exact-time gauge-stage pilot

This extends the earlier discharge compatibility check without changing its QINE availability assumption. The new validation scores the official archived **stage bulletin itself**, against independent future approved gauge observations, for one already examined flood episode. The forecast product may have assimilated observations from the same gauge before issuance; these are not counted as forecast validation. No cross-site, event-independent or fiscal claim is made.

## Provenance and clock

- The [CNRFC archive legend and guide](https://www.cnrfc.noaa.gov/verified_rvf.php) distinguishes local forecast posting time from the later graphical rendering. The exact Single Plot results for the [Dec 28 morning](https://www.cnrfc.noaa.gov/archive2/2022/Dec/RVFverification/MHBC1.12282022.12_rvf.png) and [Dec 31 morning](https://www.cnrfc.noaa.gov/archive2/2022/Dec/RVFverification/MHBC1.12312022.12_rvf.png) both identify station MHBC1 and the 12.0 ft flood / 7.0 ft monitor lines. The archive printed posting times 08:45 PST and 08:57 PST, respectively, on the forecast dates. These convert to 16:45 UTC and 16:57 UTC, equal to the corresponding SHEF creation timestamps. The graphics were rendered after the event; only their identified posting fields are used, not pixels or retrospectively drawn data points.
- The contemporaneous National Weather Service RVFSJ text broadcasts are retained by the [Iowa Environmental Mesonet AFOS archive](https://mesonet.agron.iastate.edu/nws/text.php). The download restricts the product and period then filters MHBC1 locally; the query endpoint rejects five-character `matches` (HTTP 422). The replay selects exactly one WMO header FGUS56 KRSA 281713 and one FGUS56 KRSA 311700. It checks station, date, explicit UTC, units in feet, hourly forecast stage continuation structure, a match to the graphic posting timestamps, and absence of a correction suffix. See the [official SHEF code manual](https://www.weather.gov/media/mdl/SHEF_CodeManual_5July2012.pdf). Evaluation lead is measured from the later **WMO header** at 17:13 UTC and 17:00 UTC, not the earlier graphical posting or series creation field. This does not prove a specific user saw a forecast at that minute.
- Observations are the [USGS Water Data API](https://api.waterdata.usgs.gov/docs/ogcapi/) continuous gage-height parameter 00065, station USGS-11335000, instantaneous statistic 00011, one approved, unqualified series in feet. The response has 757 complete 15-minute observations and no next page. Use both exact valid-time matching and the first observed 15-minute record reaching the historical 12.0-ft threshold. The [2022–23 CNRFC storm analysis](https://www.cnrfc.noaa.gov/storm_summaries/dec2022Jan2023storms.php) identifies the station crosswalk and historical flood threshold. The [Oct 2026 NWS stage revision](https://www.weather.gov/sto/CosumnesRiverFloodStageChanges) does not apply retroactively.
- The older QINE zip contains flow, not the stage trace captured here. Matching station and cycle is not product equivalence. No rating conversion, QINE timestamp substitution, interpolation, imputation or label changes were performed.

## Result

| WMO issue (UTC) | Future hourly pairs | First forecast hour ≥12.0 ft (UTC) | MAE / bias (ft) | Forecast / observed hourly-grid peak (ft) |
| --- | ---: | --- | ---: | ---: |
| Dec 28 17:13 | 115 | Dec 31 23:00 | 1.258 / −1.244 | 13.0 / 16.6 |
| Dec 31 17:00 | 115 | Dec 31 18:00, already above on first hour | 0.639 / −0.221 | 17.5 / 16.6 |

USGS first reached the historical 12 ft line at Dec 31 09:00 UTC on the 15-minute grid. The original bulletin was issued 63 h 47 min before that recorded crossing, but predicted threshold crossing **14 h late** and underestimated the hourly-grid peak by 3.6 ft. The later bulletin was issued eight hours after crossing. Its lower MAE reflects a different origin and evaluation window and is not evidence of a better early forecast or estimated model skill across storms. Never pool the 230 pairs as independent flood events. This single case demonstrates availability and limits of an archived operational evidence channel; it has not validated a disaster-funding predictor, simulation transfer or relief decision.

## Reproduce

```sh
bash fetch_public_sources.sh inputs
python -I -S verify_stage.py inputs outputs --refetched-usgs
```

The original downloads remain retained for audit. The public refetch script records all exact URLs. PNGs and AFOS raw bulletin bytes are checked against frozen SHA-256 digests. The USGS response envelope includes retrieval metadata, so its refetch mode permits envelope differences but requires the same hash of all 757 canonical observational facts, exact series metadata and complete grid. Without that option the original USGS file digest is checked too. Exact repeated local outputs are provided; the calculation requires no model, FEMA label, FY2025 outcome or dependency beyond the Python standard library. The 2022 records are public physical observations and are not tied to any embargo release.
