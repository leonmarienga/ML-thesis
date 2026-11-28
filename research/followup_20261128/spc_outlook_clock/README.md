# SPC Day 1 outlook clocks: authentic-data availability pilot

**Finding:** The archive's valid start, product issue, WMO header and IEM canonical cycle are not interchangeable. A valid-window-only selector would use an outlook whose own issue timestamp was still in the future at **10 of the 501 hourly cutoffs** with a valid archived window. Requiring nominal issuance by cutoff selects an earlier archived eligible outlook all ten times. This is leakage evidence for the tested selector, not an estimate of frequency in a representative population or a model result.

| Calendar probe (UTC issuance May 1–7) | Polygon rows | Clock signatures | Distinct valid windows (with >1 signature) | Noncanonical signatures | Issued before / at / after valid start | Future issue selected by validity only | Availability-selected signature changed by canonical-only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2012 | 416 | 40 | 35 (5) | 5 | 32 / 4 / 4 | 3/167 | 3 |
| 2018 | 351 | 37 | 35 (2) | 2 | 31 / 3 / 3 | 2/167 | 1 |
| 2024 | 485 | 38 | 35 (3) | 3 | 26 / 3 / 9 | 5/167 | 2 |

Each seven-day run includes 168 hourly cutoffs. The first midnight had no valid products within the requested issue window; that artificial left boundary is not evidence no outlook existed. Noncanonical rows are not automatically separate forecasts, invalid data, or corrected editions. Likewise, the 115 distinct clock signatures are not independent disasters and the 1,252 polygons are not 1,252 observations of conditions. The official [NOAA current outlook service](https://mapservices.weather.noaa.gov/vector/rest/services/outlooks/SPC_wx_outlks/FeatureServer) is not time-enabled. The [IEM historical archive documentation](https://mesonet.agron.iastate.edu/request/gis/outlooks.phtml) distinguishes the clocks and explains why more than one issuance can share a cycle.

## Separate text clock cross-check

For May 1 only in each year, the [IEM archived NOAA text channel](https://mesonet.agron.iastate.edu/nws/text.php) returned 18 ACUS01 KWNS / SWODY1 bulletins, and the GIS archive produced 18 clock signatures. Exact UTC valid start, valid end and authored SPC AC timestamp matched 16. Their WMO headers were one or two minutes later. One 2012 GIS signature was one minute off the sole text match within the same validity window; it remains non-exact. A second was unpaired (the contemporary 02:14 text had a different validity window). This cross-check does not prove contemporaneous public access to the identical geometry: both representations were retrieved through IEM. Direct SPC HTML probes for the May 1 13Z outlook in 2012 and 2024 gave HTTP 403 in this environment; that indicates access failure, not archive absence.

For example, the 2024 May 1 20Z interval has GIS PRODISS 20:14 UTC, and authored/WMO text 20:14/20:16. A valid-start gate alone would use the new outlook at 20:00; the earlier product was the nominally eligible alternative. Separately, the 2018 May 1 01Z signature has GIS PRODISS 00:59, but the WMO text header is 01:01. An as-of gate using only GIS PRODISS can therefore admit a product before its corroborating header. Retain conservative `not-before = max(authored_or_GIS_issue, matched WMO time)` where exact linkage exists, and explicit unverified first public dissemination / geometry vintages. Do not use a fixed best-average buffer or assume corrections were available at the original cycle start.

## Scope and reproduction

This is a product-clock feasibility pilot, not a severe weather forecast evaluation or a FEMA model change. No polygon geometry was used for exposure, no observations or FEMA outcomes were read, no model was fit and the FY2025 embargo is unchanged. The [analysis scope](SCOPE.md) records the calendar rule and ranking. To test predictive value later, freeze an episode-disjoint geographic/temporal join, preserve effective population vintages and outlook corrections, and use a separate independent observed endpoint (for example, time-matched realized wind) before considering any distinct administrative-funding analysis. An observed hazard quantity is neither humanitarian need nor obligation.

```bash
bash fetch.sh /tmp/spc-clock-inputs
python -I -S audit.py /tmp/spc-clock-inputs /tmp/spc-clock-inputs/results
python -B compare_bulletins.py /tmp/spc-clock-inputs /tmp/spc-clock-inputs/results
python -I -S check_sql.py /tmp/spc-clock-inputs/results
```

Clean refetch reproduced all five analysis outputs exactly and all 12 unpacked ZIP members byte-for-byte. ZIP envelope checksums differed between retrievals; member timestamps changed. The ledger preserves both envelope hashes, all stable member hashes and exact text-response digests without copying raw data. Independent SQLite ranking reproduced both cutoff comparisons; see [verification record](VERIFY.txt).
