#!/usr/bin/env python3
"""Verify archived recipient-side audit passages and declarations."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from thesis_precoverage_documentary import GATES, text_pages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("curated", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.curated, dtype={"dr_id": "Int64", "usd": "Int64", "count": "Int64"})
    assert data.item.is_unique and data[GATES].dtypes.eq("bool").all()
    sources = json.loads((args.raw / "source_manifest.json").read_text())
    checksum = []
    reviewed = {}
    for source in sources:
        if not source["filename"]:
            continue
        path = args.raw / source["filename"]
        digest = sha256(path.read_bytes()).hexdigest()
        assert digest == source["sha256"]
        checksum.append(digest)
        if source["source"] not in set(data.source):
            continue
        if source.get("ocr_filename"):
            ocr = args.raw / source["ocr_filename"]
            assert sha256(ocr.read_bytes()).hexdigest() == source["ocr_sha256"]
            reviewed[source["source"]] = [" ".join(page.split()) for page in ocr.read_text().split("\f")]
        else:
            reviewed[source["source"]] = text_pages(path)
    for row in data.itertuples():
        page = reviewed[row.source][row.file_page - 1]
        assert row.marker.casefold() in page.casefold(), (row.source, row.file_page, row.marker)
    roster = pd.read_csv("research/continuous/coverage/audit/declaration_roster.csv", usecols=["disasterNumber", "state", "fyDeclared"])
    exact = data[data.dr_id.notna()].merge(roster, left_on="dr_id", right_on="disasterNumber", validate="many_to_one")
    assert len(exact) == data.dr_id.notna().sum() and exact.fyDeclared.le(2012).all()
    data["comparable_elapsed_label"] = data.dr_id.notna() & data[GATES].all(axis=1)
    stats = pd.DataFrame([{"source_routes": len(sources), "retained_pdf_hashes": len(set(checksum)),
                           "verified_passages": len(data), "exact_assignment_keys": data.exact_assignment_key.nunique(),
                           "eligible_elapsed_declaration_labels": int(data.comparable_elapsed_label.sum())}])
    for name, frame in [("verified_records", data), ("exact_assignment_keys", exact),
                        ("source_manifest", pd.DataFrame(sources)), ("gate_summary", stats)]:
        frame.to_csv(args.output / f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
