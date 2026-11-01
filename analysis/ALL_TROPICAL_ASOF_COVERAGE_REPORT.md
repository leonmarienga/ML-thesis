# All-tropical historical NHC as-of coverage — 2026-11-01

Successful GitHub Actions run: https://github.com/leonmarienga/ML-thesis/actions/runs/41332319839

## Target-blind source matching
The new extractor `analysis/nhc_all_tropical_asof.py` reads all 971 master FEMA cases, selects *all* 123 records of incidentType Hurricane, Tropical Storm, or Typhoon, and matches published NHC archive advisory issue times without checking funding amounts or funding bands. Funding bands are added only for post-extraction coverage auditing.

**Coverage:** 106/123 records (86.18%) have an archival NHC public advisory on or before both the conservative date-only T0 and T0+72h cutoff, representing 45 distinct named-storm/fiscal-year identifiers. T0 wind MPH numeric data are available for all 106. The 17 unmatched consist of 10 ambiguous/missing NHC archive matches, 6 western Pacific typhoons requiring a different warning provider, and 1 missing name/date.

| Funding band | Total tropical cases | T0 NHC archive matches |
|---|---:|---:|
| $0–100K | 68 | 57 |
| $100K–1M | 17 | 14 |
| $1M–50M | 21 | 20 |
| $50M–200M | 9 | 7 |
| $200M–500M | 5 | 5 |
| $500M+ | 3 | 3 |
| **Total** | **123** | **106** |

## What these results establish and do not
They establish **historical NHC archive coverage**, not prospective real-time prediction performance or humanitarian need.

- The FEMA declarationDate used here is normalized to **00:00 UTC**; precise contemporary declaration issuance timestamp remains unverified.
- Storm identity comes from the *current* FEMA declaration title, which may have been updated after t0. Identity is therefore *not yet prospectively validated*.
- Timestamp conversion uses archive HTML comments plus the zone shown in archive link labels, including historic EDT/AST. Spot-check each original public bulletin to verify clock interpretation.
- A single hurricane can create multiple FEMA disaster records; don't treat the 106 records as 106 independent storms.
- The latest advisory's wind speed is **not** peak event wind, localized impact or exposed population.
- No typhoon wind features were imputed from final best tracks; do not use late final storm records.
- No accepted router or specialist was modified.
- A model test needs full feature and missingness policy, as-of-valid storm IDs, and training restricted to prior fiscal years. Compare on the same cases and report uncertainty, without updating the frozen benchmark.

## Recommended following investigation
Review 10 ambiguous NHC cases with official declaration titles, verify the earliest official FEMA record timestamps and storm identity, and implement archived NWS Guam/JTWC warning data for six western Pacific cases. Then assess numerical wind/pressure feature quality by decision-date and source, and proceed to a *pre-registered* earlier-year-only ablation.
