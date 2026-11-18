# Independent extension of tropical wind verification

The same fixed 24- and 48-hour forecast verification design was applied to six additional hurricanes across the Gulf, Caribbean and southeastern coast. There was no forecast-geometry tuning from the initial three storms. The new physical test is on distinct hurricane and station observations. It does not provide an untouched test of FEMA obligations. [Protocol](../BATCH5_PROTOCOL.md), [first three-storm trial](../winds/REPORT.md).

The extension found 1,704 valid forecast–observation pairs across 42 distinct station locations. The 2,620 planned pairs included missing records or winds outside the sample requirement. The site-level negative class denotes a measured speed below 34 kt. Spatial and measurement differences prevent treating it as a fully verified contour error.

| Extent | Matched exceedances | Positive signals below measured threshold | Missed measured exceedances | Sensitivity | Precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Four-quadrant forecast | 76 | 264 | 12 | 86.4% | 22.4% |
| Equal-area circle | 78 | 264 | 10 | 88.6% | 22.8% |

The new storm set did not reproduce the prior reduction in below-threshold alerts for quadrant geometry. The two methods had equal counts of below-threshold alerts, and the circle identified two more exceedance pairs. Effects varied across storms. The quadrant representation reduced these alerts for Harvey by 20 and increased them for Helene by 17. Measurement pairs within a storm are dependent and share station observations across forecast leads. No finding establishes a universal superior threshold geometry. [Storm-by-lead contingency counts](results/station_contingencies.csv), [source and observability ledger](results/station_audit.csv), [case comparisons](results/wind_match_cases.csv).

These data remain useful for geographically grounded exposure and alert stress tests. They do not establish observed area-wide exposure, economic loss, humanitarian need or FEMA obligations.
