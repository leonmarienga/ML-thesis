# Original declaration filing and decision-time availability

The archived Federal Register supplies an authoritative release time for many original declarations. It provides inadequate early coverage for a T0 through T+7 decision-time predictor on this historical cohort.

The search covered nine official FEMA Notice index pages between October 2018 and September 2024. Exact historical DR keys matched 1,116 documents. This included 343 original notices and 773 related notices, such as amendments. An original notice with matching declaration day and parsed filing time was available for 341 out of 376 historical declarations. Where there were repeated originals, the earliest valid filing was retained. The archive was queried and analyzed without joining funding targets. [Official record manifest](results/source_manifest.csv), [original filings](results/original_notice_filings.csv), [related notices](results/related_notice_index.csv).

| Horizon | Confirmed public | Ambiguous timing | Confirmed late | Missing | Right censored |
| --- | ---: | ---: | ---: | ---: | ---: |
| Declaration time | 0 | 0 | 341 | 32 | 3 |
| +24 hours | 0 | 0 | 341 | 30 | 5 |
| +48 hours | 0 | 0 | 341 | 30 | 5 |
| +72 hours | 0 | 0 | 341 | 29 | 6 |
| +7 days | 0 | 0 | 341 | 24 | 11 |
| +14 days | 7 | 5 | 329 | 20 | 15 |
| +30 days | 98 | 9 | 234 | 18 | 17 |

A case is confirmed public when filing is within the horizon even at the earliest possible Presidential issuance time on the effective civil date. The earliest verified filing was more than 7.36 days after the latest possible time on the declaration day. The median lower lag bound was **40.41 days**. Statutory filing and the notice footer establish public inspection of these archived notices. None of the matched notices certifies the original Presidential signing time or the content of earlier releases. Missing matches and late filings support no universal conclusion about FEMA's other channels. The truncated year can disproportionately affect FY2024 coverage. [Horizon evidence](results/horizon_coverage.csv), [44 U.S.C. §§ 1503–1504](https://www.govinfo.gov/content/pkg/USCODE-2024-title44/html/USCODE-2024-title44-chap15.htm), [National Archives public inspection](https://www.archives.gov/federal-register/public-inspection).

The archive can retrospectively verify administrative wording and later disclosure. Early forecast candidates still require dated official FEMA communications, preserved releases from the event and the version actually available to the responder. The early-coverage gate precluded a model fit with Federal Register notice features. [Locked FY2025 protocol](../../follow_up/FY2025_LOCK.md).
