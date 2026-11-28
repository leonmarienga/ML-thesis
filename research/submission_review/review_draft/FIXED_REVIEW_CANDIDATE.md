# Fixed supervisor-review candidate

The [six-chapter manuscript](https://github.com/leonmarienga/ML-thesis/blob/7edb63624f35e6181d813f2abba956d4bc024cb6/research/submission_review/THESIS.md) is the fixed reading version for this candidate. Its cover, keywords, linked contents, lists of figures and tables, and four appendices are included. Three previously moving follow-up citations are now pinned to exact evidence files. Appendix A and the [reviewer claim map](../REVIEWER_START_HERE.md) distinguish archived reconstruction, new retrospective comparisons, simulated service, and prospective participant research.

## Review build checks

- The Word source builder was run with Pandoc 3.1.11.1 and Inkscape 1.4. The Word file contains 11 checked exact figure payloads and 19 Office display equations. Layout was rendered with LibreOffice into 33 A4 pages. All pages were checked for pagination, with full-size spot checks on the front matter, dense tables and formulas. Table T8 was widened to prevent a broken header; the other 32 page rasters were unchanged.
- The document relationship audit found 12 distinct valid bookmark destinations, 15 distinct exact targets from relative source links, and 3 links to pinned follow-up result files. The three follow-up targets were checked against the exact Git tree.
- The offline verification archive is built separately using the [source-bound one-command guide](../handoff/STAGING.md). It contains an executed eight-cell notebook, selected evidence, and the interactive viewer. It does not train historical models. The complete package nests this existing tested archive unmodified.
- The staged ZIP was independently assembled twice with identical bytes. Its five allowed members are a start guide, a package manifest, editable Word, PDF review copy and the offline verification archive. [Artifact identities](fixed-review-package-manifest.json) record bytes and SHA-256 values.

## Availability and academic review

This file records a locally staged candidate. It does not publish the new Word, PDF or combined ZIP, and does not change or revalidate the older Word snapshot. The source manuscript and standalone verification build are available at the repository links. The [academic review boundaries](../ACADEMIC_REVIEW_BOUNDARIES.md) distinguish supervisor review from formal opinion, human-subject approval, institutional formatting and academic submission.

Package SHA-256: `39770680ddba0d5ea1ba2df5b3d563ce0755f7aeb689170360d7898da6748e60` (4,981,077 bytes).
