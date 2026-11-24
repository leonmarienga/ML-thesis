# Historical wind geometry and county outage support

The four episodes overlap the already retained NHC wind archive and EAGLE-I county sample. The analysis used the published 2017 county internal points and reused the frozen geodesic quadrant and equal-area circle calculation. Each matched entry used a 24-hour nominal NHC forecast with a valid time from zero to less than six hours earlier. It measures presence of a forecast contour at a representative point, with no independent confirmation of the source's original public issue instant or individual customer exposure. [Protocol](../BATCH9_PROTOCOL.md), [2017 Census Gazetteer](https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.2017.html), [NHC archive](https://ftp.nhc.noaa.gov/atcf/archive/).

| Storm | Entries in the existing date window | Within 400 km | Complete within 400 km | Sustained below threshold within 400 km |
| --- | ---: | ---: | ---: | ---: |
| Michael 2018 | 28 | 3 | 2 | 0 |
| Laura 2020 | 24 | 1 | 1 | 0 |
| Ian 2022 | 20 | 3 | 3 | 2 |
| Idalia 2023 | 22 | 1 | 1 | 1 |

All 94 entries in the four windows had a sufficiently recent forecast valid time. Eight were within 400 km and seven had complete follow-up, with three sustained outcomes below threshold and four that missed the sustained criterion. Three regional cases with complete windows were inside both 34-knot contours. The contours agreed on membership for every matched entry in this small sample. That observed agreement cannot establish geometric equivalence outside the sample. Eighty-six entries fell beyond 400 km, of which 43 had complete windows. Their coincident timing shows how date-only matching could pull unrelated county events into the comparison. [Regional and state counts](results/storm_support.csv), [time and location matches](results/matched_entries.csv).

Ian was the only storm with both endpoint outcomes among regional complete cases. The matched evidence therefore falls short of the independent training and held storm requirements. An independently scored simulation would require more dated weather episodes and a geographically targeted county panel chosen from forecast footprints before reading outage outcomes. Measured outage trajectories retain their separate service interpretation. [County points](results/county_points.csv), [physical source manifest](results/source_manifest.csv).
