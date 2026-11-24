# Independently measured humanitarian need and service receipt

Three Census Household Pulse Survey waves, the published US Virgin Islands CASPER assessment, and the FEMA shelter service metadata were inspected separately. The audit checked survey units, timing, response populations and the scope of any observed service delivery. No funding outcome or individual record enters the analysis. [Protocol](../BATCH10_PROTOCOL.md), [source manifest](results/source_manifest.csv).

## Census displacement and resource deprivation

The official Census tables provide weighted estimates for adults aged 18 and older. Displacement refers to natural disasters in the prior year. The shortage tables condition on adults who reported such displacement and refer to food and drinkable water shortages one month after the disaster. The shortage columns are available across hazard types in the audited tables. They cannot be assigned to the hurricane-only count or to a specific landfall. The three waves cover overlapping recall periods. The reported standard errors are preserved for published counts and were not converted to ratio standard errors because the required covariance is unavailable. A missing or suppressed estimate remains missing. [Week 52](https://www.census.gov/data/tables/2023/demo/hhp/hhp52.html), [Week 61](https://www.census.gov/data/tables/2023/demo/hhp/hhp61.html), [Week 62](https://www.census.gov/data/tables/2023/demo/hhp/hhp62.html).

| Wave | Estimated U.S. adults displaced by any disaster | Estimated U.S. adults displaced by a hurricane | A lot of food shortage, share of all displaced adults | A lot of drinkable water shortage, share of all displaced adults |
| --- | ---: | ---: | ---: | ---: |
| 52 | 3,333,947 | 1,927,340 | 19.1% | 15.3% |
| 61 | 2,769,611 | 1,299,240 | 18.2% | 12.6% |
| 62 | 2,864,135 | 1,048,381 | 17.0% | 16.7% |

The tidy extract retains 1,560 published cells from the national and state-level tabs. Of these, 1,197 contain numeric estimates and 363 are omitted or suppressed. The companion summary distinguishes shares of all displaced adults from shares among people who answered the question. These experimental surveys inform contextual deprivation scenarios. The fields do not identify event-resolved demand, beneficiaries or allocations. [Counts and standard errors](results/hps_weighted_tables.csv), [share denominators](results/hps_summary.csv).

## Household needs and a separate service outcome

The USVI post-Irma-and-Maria cluster assessments report weighted household need for food of **21.0%** in November 2017 (95% CI 15.7–26.2) and **11.1%** in February 2018 (5.7–16.5). Unmet water need was **23.2%** (17.6–28.9) and **10.9%** (6.4–15.5). Surveyed households with utility electricity were **25.5%** (16.9–34.1) and **96.7%** (93.6–99.7). The later survey reported that **7.7%** (3.1–12.3) of households had obtained behavioral-health services and **4.3%** (0.3–8.3) could not obtain services they needed. The study also documents referrals and some public health follow-up. [CDC published CASPER assessment](https://stacks.cdc.gov/view/cdc/114255/cdc_114255_DS1.pdf), [verified estimates](results/casper_published_estimates.csv).

The surveys are repeated cross-sections after two interacting storms. They describe need, recovery and observed receipt with separate measures. They do not track comparable interventions and outcomes in the same households. The evidence can benchmark plausible scenario inputs for one combined event, with its reported uncertainty and sampling limitations. It cannot establish the effectiveness of an allocation rule.

## Shelter evidence and allocation feasibility

FEMA's shelter layer advertises opening and closing dates, shelter population and capacity. TEMPO advertises county FIPS, event and current shelter population fields. The inspected metadata do not advertise a usable time-series axis, historical snapshot queries or a versioned shelter history. Only the metadata were retrieved. Occupancy measures utilization under the actual opening and access rules. It cannot measure people unable to obtain shelter without a separate demand denominator. The three audited sources therefore add independent need or service observations and no valid allocation-policy comparison. [Shelter layer](https://gis.fema.gov/arcgis/rest/services/NSS/FEMA_NSS/FeatureServer/0), [TEMPO layer](https://gis.fema.gov/arcgis/rest/services/TEMPO/TEMPO/MapServer/0), [metadata screen](results/shelter_metadata_screen.csv).
