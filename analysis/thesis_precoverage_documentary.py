#!/usr/bin/env python3
"""Verify paper citations and apply the frozen precoverage obligation gate."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import pandas as pd

GATES = ["signed_amount_and_action", "complete_amendments_and_reversals", "all_agency_and_geographic_scope",
         "comparable_36_month_endpoint", "known_ledger_coverage"]


def text_pages(pdf):
    full = subprocess.check_output(["pdftotext", "-q", "-layout", str(pdf), "-"], encoding="utf-8")
    return [" ".join(page.split()) for page in full.split("\f")]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("raw", type=Path)
    cli.add_argument("curated", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    documents = pd.DataFrame(json.loads((args.raw / "source_manifest.json").read_text()))
    evidence = pd.read_csv(args.curated, dtype={"exact_dr_id": "Int64", "nominal_usd": "Int64", "count": "Int64"})
    assert evidence.evidence_id.is_unique and evidence[GATES].dtypes.eq("bool").all()
    roster = pd.read_csv("research/continuous/coverage/audit/declaration_roster.csv", usecols=[
        "disasterNumber", "state", "fyDeclared", "declarationType",
    ])
    published = documents[documents.status.eq("retained")]
    seen = {}
    for row in published.itertuples():
        filename = args.raw / row.file
        assert sha256(filename.read_bytes()).hexdigest() == row.sha256
        if row.source in set(evidence.source):
            seen[row.source] = text_pages(filename)
    exact = evidence[evidence.exact_dr_id.notna()]
    matched = exact.merge(roster, left_on="exact_dr_id", right_on="disasterNumber", how="left", validate="many_to_one")
    assert matched.declarationType.eq("DR").all() and matched.fyDeclared.le(2012).all()
    recognized = []
    for row in evidence.itertuples():
        page = seen[row.source][row.pdf_page - 1]
        assert row.exact_text_marker.casefold() in page.casefold(), (row.source, row.pdf_page, row.exact_text_marker)
        recognized.append(True)
    evidence["page_marker_verified"] = recognized
    evidence["comparable_label_eligible"] = evidence.exact_dr_id.notna() & evidence[GATES].all(axis=1)
    dated = evidence[evidence.value_type.eq("fema_agency_obligation_snapshot")].sort_values("source_as_of_date")
    assert dated.scope.nunique() == 1
    initial = dated.nominal_usd.iloc[0]
    dated = dated.assign(change_from_prior_usd=dated.nominal_usd.diff(), change_from_first_usd=dated.nominal_usd - initial)
    summary = pd.DataFrame([{"public_source_routes": len(documents), "successfully_retained_routes": len(published),
                             "distinct_retained_document_hashes": published.sha256.nunique(),
                             "verified_passages": len(evidence), "identified_exact_declarations": exact.exact_dr_id.nunique(),
                             "eligible_36_month_labels": evidence.comparable_label_eligible.sum()}])
    evidence.to_csv(args.output / "verified_evidence.csv", index=False)
    documents.to_csv(args.output / "source_manifest.csv", index=False)
    matched.to_csv(args.output / "exact_declaration_references.csv", index=False)
    dated.to_csv(args.output / "dod_revisions.csv", index=False)
    summary.to_csv(args.output / "gate_summary.csv", index=False)


if __name__ == "__main__":
    main()
