# Rebuild Table T6 against its exact evidence revisions

`T6_SOURCE_PINS.csv` maps 14 exact commit/path pairs to SHA-256 digests; no outcome endpoints are queried. Montpelier is checked against the archived manuscript's reported counts, not a replay of an unavailable raw acquisition in this table. The fuller source outcomes cited elsewhere are unchanged.

If the pinned Git objects already exist in a local repository cache, run:

```sh
GIT_NO_LAZY_FETCH=1 python -I -B build_t6.py --repo /path/to/local/repo-cache --output /tmp/t6-rebuilt
```

The check uses named blobs only rather than checking out the source tree. The command was tested on an existing object cache, with network lazy retrieval disabled. Compare the regenerated CSV and Markdown files byte-for-byte to the checked-in tables. For an evidence-only review without Git replay, use the links and exact source digests in the source-pin table.

## Wheeling identity exclusion

The [2009 USGS catalog, printed p. 117](https://pubs.usgs.gov/of/2009/1199/pdf/OF2009-1199.pdf) describes Wheeling station 03112500 at river mile 86.7 and lists its datum as unknown. The [1991 water-year report, printed p. 85](https://pubs.usgs.gov/wdr/1991/wv-91-1/report.pdf) gives 610.80 ft NGVD29 for Martins Ferry 03111534 at river mile 87.5. Its note mentions 03112500 as a former identifier but does not transfer the Martins Ferry reference to today's separate Wheeling site. None of this establishes the April 2024 local reference or warrants shifting a stage reading. No held reading was requested.
