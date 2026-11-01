# NHC Advisory Age and Bulletin Timestamp Audit — 2026-11-01

Research action: https://github.com/leonmarienga/ML-thesis/actions/runs/41336947649

## Population
All **123 tropical FEMA records** (971-record master), no funding-outcome filter in source matching. **113** link to historical NHC advisories issued at or before a conservative date-only FEMA declaration cutoff (00:00 UTC). The remaining ten are not imputed.

## Freshness findings
Of the 113 historical matches:
- **29** have the last archived advisory within **24 hours** of T0.
- **41** are within **72 hours**.
- **51** are within **7 days**.
- **62** are **older than seven days**.
- Median latest-advisory age: **453 hours (18.9 days)**.
- Maximum observed age: **4383 hours (182.6 days)**.
- A T+72h freshness check gives **27** advisories no more than 72h old, even though all 113 still have an earlier archived advisory.

These results show that `NHC latest advisory` is often a historic *storm observation*, not a contemporaneous current hazard estimate when FEMA declares assistance months later.

## Bulletin-header timestamp consistency
The audit fetched and parsed **25** original public advisory bulletin headers spanning 2009–2024. **25/25** have header clock time and timezone consistent with the historical archive's comment timestamp conversion to UTC. This supports the parser's timezone conversion but is a sampled check, not verification of every source record.

## Decision-time validity remains incomplete
- The FEMA `declarationDate` available in the original master has midnight UTC. This is a **date-based conservative cutoff**, not confirmed exact approval/announcement time. An original press release or declaration legal instrument may give date only; don't fabricate an hour.
- NHC archive bulletin issuance is time stamped; matching to *current* FEMA declarationTitle remains retrospective and not proven to have been available from the first issued version.
- NHC wind is storm-wide latest reported sustained wind, not county-specific impact or cumulative maximum through cutoff.
- If using wind as a severity predictor, record **advisory age** and either use an explicit maximum-age threshold or reinterpret old advisories as *historical event descriptors* rather than present conditions.
- Deduplicate/cluster hurricane identity (storm name, season and basin) during evaluation; several FEMA disaster cases share the same meteorological event.
- Avoid fitting a temporal threshold on the 13-case high funding band; predefine threshold on source characteristics and run an independent test.

## Conclusions
We have improved source provenance and identified a serious temporal-validity limitation. No six-band classifier, specialist, or deployed Django logic has been changed. We **cannot claim an increase in funding-model prediction accuracy** based on this audit.
