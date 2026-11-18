# Earlier document recovery and decision-time model support

The extension used the same 16 briefing sitemaps and added 26 newly inspected pages to the original 31. Forty-seven of 57 inspected pages matched a unique historical declaration through state, effective day and incident onset. Five could not supply enough identity detail and five did not match a unique historical record. The sample covered 57 of 106 candidate URLs. The later preserved site contained a nominal release claim within T+24 for 44 matched declarations. Exact program or jurisdiction text required an independently timestamped copy. [Locked protocol](../BATCH7_PROTOCOL.md), [sample frame](results/sample_frame.csv), [identity review](results/document_identity.csv).

The evidence contains 42 qualified original URLs queried for early copies. The checks accepted 114 snapshots across 31 declarations after validating the original replay URL, Memento time and body identity. Eleven of the 42 queried identities had no qualifying capture in the bounded window. Exact recovered body coverage rose from one to 21 at T+24, and from six to 31 at T+48 and T+72. The recovered T+72 corpus includes 12 severe-storm declarations in 12 episode groups, six hurricane declarations in three groups, and two fires in two groups. The previously recorded substantive wildfire edit and hurricane cross-reference remain the two earliest-body differences observed. The lexical sample and limited query window preclude a population-wide estimate of archive retention. [Version bodies and UTC bounds](results/capture_versions.csv), [hazard coverage](results/documentary_hazard_counts.csv).

## Joint maturity and feature support

For each held fiscal year, a training record must precede the earliest held incident or declaration, pass the shared episode purge and have 36 months plus 90 days elapsed by that origin. A held record must have passed the same interval by the historical action snapshot. The retrospective signed-action archive supports those elapsed amounts under its current available version. It does not independently establish earlier public action releases. The 2017–2020 replay bodies are direct evidence of earlier versions. The 2021–2024 inputs use archived edit metadata, which is a weaker source tier. The combined count is a ceiling for cases covered by either evidence tier.

| Held FY | Prior groups at T+72 | Held groups at T+72 | Prior groups at least $50M | Held groups at least $50M | Held groups at least $50M, maturity only |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2019 | 0 | 6 | 0 | 1 | 3 |
| 2020 | 0 | 4 | 0 | 0 | 1 |
| 2021 | 0 | 12 | 0 | 1 | 1 |
| 2022 | 5 | 12 | 1 | 2 | 3 |
| 2023 | 9 | 7 | 2 | 1 | 2 |
| 2024 | 17 | 0 | 2 | 0 | 0 |

Counts refer to groups under the historical episode definition and use the elapsed 36-month net obligation. No prior group with a T+72 document in either tier reaches $200M. The largest combined prior cohort contains 19 declarations across 17 groups. The largest held cohort contains 14 declarations across 12 groups. At the current historical snapshot, all FY2024 held cases lack an elapsed 36-month plus 90-day label or a published-action history meeting the label rule. [All thresholds and origin cutoffs](results/documentary_support_by_origin.csv).

None of the documentary tiers attained the 30-prior and 15-held group gate in a single origin. Perfect documentary coverage within the known mature population would pass the general count gate in five origins. It would still yield at most three $50M+ held groups in any origin against the required five. The rarer thresholds have at most one held group per origin even at that ceiling. A source-only expansion can benefit general decision-time models. A stringent same-origin rare-tail evaluation also requires additional independent, comparably defined mature episodes. This batch froze no new candidate model. [Gate summary](results/documentary_gate_summary.csv).
