# Build the integrated supervisor manuscript locally

The earlier published Word document stays linked to its original smoke-test checkpoint. This builder targets the **newer Markdown**, including Appendix D and Figures F1–F8 and B1. It does not use the old builder's optional F8-only render. It does not submit or transmit the generated manuscript.

From the repository's `research/submission_review` folder, with Python 3.12, python-docx 1.2.0, Pandoc 3.1.11.1 and Inkscape 1.2 or later, use:

```sh
python review_draft/build_integrated_docx.py THESIS.md \
  /path/to/pinned/research/continuous/transport_pareto/results/pareto_frontiers.png \
  "Integrated Supervisor Review Draft.docx" \
  --review-ref thesis/supervisor-review-20261128 --figure-manifest figure-manifest.json
```

Take the transport input from the [frozen baseline](https://github.com/leonmarienga/ML-thesis/blob/b0b6edb229d112dd6d11896a512bf1112414e4f6/research/continuous/transport_pareto/results/pareto_frontiers.png). `--review-ref` sets the resolver for linked repository assets; prefer an immutable source commit for a final snapshot. Figure SVGs are resolved adjacent to the given manuscript and explicitly rasterized to PNG. The builder rejects a missing figure; it also checks every generated media-part digest, relationship, alternate text and adjacent figure caption. The reviewed source has 19 Office Math display nodes. Amounts such as $5,014,819.01 remain plain currency text.

Render the generated DOCX with LibreOffice and inspect every affected page, including equations and heading/caption page breaks. The figure manifest contains the source manuscript hash, generated document hash, and an exact source-to-embedded-image crosswalk. The manifest is sufficient to identify a tested local build; an actual reviewed file must be checkpointed separately. Academic approval, institutional formatting and any release permissions remain separate.
