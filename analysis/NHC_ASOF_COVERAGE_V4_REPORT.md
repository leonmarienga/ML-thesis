# NHC as-of advisory coverage update — 1 November 2026

## Verified outcome
The corrected legacy archive extractor `analysis/nhc_asof_highband_v4.py` was executed successfully in [GitHub Actions run 41331272968](https://github.com/leonmarienga/ML-thesis/actions/runs/41331272968).

- High-value tropical FEMA disaster records: **16**.
- Records with NHC archive-matched public advisories dated at/before the conservative declaration-day 00:00 UTC cutoff: **15/16**.
- Records with NHC advisory candidates for declaration +72h: **15/16**.
- Records with parsed wind speed at both cutoffs: **15/16**.
- Unmatched record: **4715, Typhoon Mawar, Guam, 2023**. NHC Atlantic/East Pacific archive is not the relevant operational warning archive.
- The 16 rows represent only **12 named cyclones** (11 in the NHC-match group). Sandy appears in NY/NJ, Maria PR/VI, Michael FL/GA, and Helene GA/NC; repeated storms must not count as independent meteorological examples.
- The **15 source-available hurricane declaration rows are not 15 independent storm severity observations**.

## Historical archive timestamp parsing corrections
Older NHC index pages (2012–2021) include advisory URLs ending in `.shtml?` and HTML timestamp comments in archive-local clock time. The anchor label identifies time zones, e.g. EDT or AST. The v4 parser explicitly converts that clock time to UTC using the zone identified in the label, rather than assuming the comment is UTC. Newer 2022–2024 pages display UTC labels. This improvement raised qualifying matches from 4 to 15 without inventing missing observations.

## Interpretation and outstanding safeguards
The extracted wind is **the latest public advisory's sustained wind** as of each cutoff. It is **not maximum wind through the event**, peak landfall intensity, county wind exposure, or funding need.

1. The current cutoff uses the *calendar date* of FEMA declaration, normalized to 00:00 UTC. This is conservative but not a verified actual proclamation timestamp; it may omit same-day observations legitimately available for funding decisions.
2. Storm naming uses a retrospective FEMA declaration title. It must be checked against first-published declaration versions to certify that cyclone identity existed at t0.
3. HTML archive timestamps must be spot-checked against the original advisory bulletin header and converted correctly around daylight time; source URL and archive-local time should remain in the provenance audit.
4. Guam's Typhoon Mawar requires original operational warning products from **JTWC** and/or **NWS Guam**, not the post-season JTWC best track. A post-storm JTWC annual report cannot stand in for an as-of forecast or analysis.
5. Coverage of all **912** non-Biological disasters and all hazard groups is not established. Only 16 high-value tropical cases were selected for this source feasibility test. Therefore this cannot justify selecting a better feature weight or claiming a global six-band improvement.
6. The frozen router, 830/912 (91.01%), and accepted MAE/R² stack remain untouched; **no numerical model acceptance decision** was made.

## Next experiment gates
- Build a full-event, outcome-blind advisory index covering all tropical disaster declarations, not just high-value cases; avoid target-conditioned enrichment for training purposes.
- Enrich using original declaration versions and exact timestamps, not retrospective names or date-only cutoffs.
- Publish a verified test subset with available source links and timestamp accuracy for each advisory.
- Join source observations with applicable event footprint / exposed population known as of t0; maintain clear missings.
- Only then run earlier-fiscal-year-only validation, including per-band recall, total MAE, confidence intervals and storm-level/grouped diagnostics.

**The research finding from this phase is source coverage, not an improvement in funded-amount prediction.**
