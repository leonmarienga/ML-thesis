# Rebuild the editable supervisor review draft

This is a review draft, not an institutionally formatted submission. The source is `../THESIS.md` on the supervisor-review branch. Pandoc 3.1.11.1 and python-docx 1.2.0 produced the Word file; LibreOfficeDev 26.8.0.0 rendered it as 15 A4 pages for layout inspection. The script embeds three existing evidence figures and checks the Word archive and six chapter headings. PDF and page screenshots are QA intermediates, not required deliverables.

Generate the F6 PNG using `../figures/track_a_f6_t4.py` and the matching `research/results/track_a/all_912_immutable_overlay.csv` from the frozen baseline. Obtain the frontier PNG from `research/continuous/transport_pareto/results/pareto_frontiers.png` at the same baseline. Generate the F8 PNG with `../figures/uncertainty/build_f8_t8.py` against the pinned baseline. From this folder:

```bash
python build_review_docx.py ../THESIS.md /path/to/F6-track-a.png /path/to/pareto_frontiers.png 'Disaster Relief Thesis Supervisor Review Draft.docx' --review-ref thesis/supervisor-review-20261128 --uncertainty-figure /path/to/F8-matched-intervals.png
sha256sum -c REVIEW_SHA256SUMS
```

The manifest verifies the committed draft bytes. Rebuilding may change ZIP timestamps without changing content. Do not treat this Word artifact or the notebook replay as proof of a completed institutional submission or scientific deployment.
