# Tropical incident-year correction and remaining Pacific archive audit — 2026-11-01

Run: https://github.com/leonmarienga/ML-thesis/actions/runs/41334493058

## Results
The earlier 123-record outcome-blind NHC extraction resolved 106/123 (86.18%). The incident-year fix, using `incidentBeginDate` to locate the archived named storm rather than the possibly much later `declarationDate`, resolves **113/123 (91.87%)** at both midnight UTC on declaration day and 72 hours afterward. This recovers **7 additional records** without changing the funding model.

Coverage among tropical FEMA cases by true funding band (audit after matching):
- 0–100K: 62/68
- 100K–1M: 16/17
- 1M–50M: 20/21
- 50–200M: 7/9
- 200–500M: 5/5
- 500M+: 3/3

## Ten remaining cases
- 4224 GU, Dolphin (2015) — western Pacific warning archive needed
- 4235 MP, Soudelor (2015) — western Pacific warning archive needed
- 4357 AS, Gita (2018) — south Pacific/Fiji regional warning archive needed
- 4396 MP and 4398 GU, Mangkhut (2018) — western Pacific warning archive needed
- 4404 MP, Yutu (2018) — western Pacific warning archive needed
- 4433 GU, Wutip (2019) — western Pacific warning archive needed
- 4715 GU and 4716 MP, Mawar (2023) — western Pacific warning archive needed
- 4818 NY, unnamed current FEMA declaration title for severe storm/flooding (2024) — no unique contemporaneous storm identity; no imputation

**Important correction:** The previous 17 unmatched included a number of historic Atlantic hurricanes for which federal declarations occurred the *following calendar year*. Searching their 2013/2021 declaration-year hurricane archives was incorrect; searching incident-year source archives now recovers them.

## West and South Pacific plan
Use original *operationally issued* bulletins, not post-storm best tracks:
- Guam/CNMI: NWS Guam / WMO RSMC Tokyo JMA or JTWC archived operational advisory editions, with verified issue timestamps.
- American Samoa / Gita: RSMC Nadi / Fiji Met Service and relevant operational NWS Pago Pago warnings.
- Do not infer a specific wind speed from storm-name lookup or time-shifted final best-track records. Retain missing where historical original warnings cannot be verified.

## Validity caveats
- FEMA `declarationDate` is treated as midnight UTC; not proven to equal the exact moment aid decisions were made.
- Current FEMA declaration titles remain retrospective identification aids; their presence at initial declaration is not yet validated.
- Latest available NHC advisories may be days/months old at a declaration: the source is **historically issued before t0**, but may be stale and not useful for real-time severity.
- The exact same 113 records have an eligible archive advisory at t0 and t+72h; this does not show that observations are distinct or that they add information 72 hours later.
- `incidentBeginDate` may be edited retroactively, not itself certified from initial declaration versions.
- No five-neighbour specialist, router, or Django app was changed; no new amount-prediction or routing accuracy claims are justified.
