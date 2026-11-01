# NHC historical advisory availability audit — 2026-11-01

Status: **completed source-timestamp extraction proof of concept**, **not** a funding-model improvement.

## Data

Research branch: `experiment/nhc-asof-advisories-20261101`.
Source: [GitHub Action 41330279996](https://github.com/leonmarienga/ML-thesis/actions/runs/41330279996).
Research-only output: `nhc_predeclaration_highband_features.csv`, `identity_source_audit.csv`, `summary.json`.

The high-value tropical pilot includes **16 FEMA cases**. Archived NHC storm index resolved for **15**; the western Pacific typhoon record did not match the Atlantic/East Pacific index. Of those 15, only **4** yielded parsable UTC advisory links under the strict HTML-comment parser (2022/2024 archive pages), and all 4 had an advisory issued at or before both T0 and T+72h. For the other 11, the index pages did not expose the expected HTML comment format: **no inference is warranted about whether earlier advisories existed**. They are parser failures or unverified availability, not negative observations.

| FEMA disaster | Event | T0 last archive issue (UTC) | T0 sustained wind (mph) | T0 pressure (mb) |
|---|---|---|---:|---:|
| 4673 | Hurricane Ian, Florida | 2022-09-29 00:00 | 115 | 960 |
| 4671 | Hurricane Fiona, Puerto Rico | 2022-09-21 00:00 | 125 | 951 |
| 4830 | Hurricane Helene, Georgia | 2024-09-28 15:00 | 15 | 998 |
| 4827 | Tropical Storm Helene, North Carolina | 2024-09-27 21:00 | 35 | 984 |

These are values parsed from the latest public advisory available before the cutoff as defined in the script, **not maximum storm intensity** and **not attributed state-level damage or affected-population estimates**. The T0 definition is conservatively the start of the FEMA declaration date (00:00 UTC), which may precede the actual issuance time. A better t0 requires exact issued_at timestamp from contemporaneous FEMA record history.

## Methodological implications

- With only four time-restricted tropical cases, there is not enough validated source coverage to claim a reliable general improvement to the `$50M–$200M` specialist or overall router.
- For 2012, 2017, 2018, 2020 and 2021 archival pages, implement layout-specific **verified** issue-time parsers; do not assign dates from filenames, revision date, or final HURDAT2.
- A distinct CM/CPHC archive is needed for Typhoon Mawar in Guam.
- Hurricane names derived from retrospective FEMA declaration titles are not yet identity-time validated, even where archived advisory issuance timestamps are sound.
- The frozen accepted six-band router at 830/912 remains unchanged. No accepted specialist or Django production code was modified.

## Validation gates before any model replacement

1. Audit every bulletin URL and parse its issued UTC timestamp from original archive metadata.
2. Verify storm identity by archived contemporaneous FEMA declaration title or bulletin, not future finalized naming.
3. Verify exact FEMA declaration issuance timestamps and run both t0 and t+72h conditions.
4. Extend valid severity coverage to a suitable non-tropical baseline and report missingness by hazard and year.
5. Train strictly on earlier fiscal years, retain immutable accepted model comparator, and report confidence intervals with 13 rare-band cases.
