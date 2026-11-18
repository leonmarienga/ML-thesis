#!/usr/bin/env python3
"""Verify the unchanged FY2025 reservation and fetch declaration types only."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import requests

from thesis_retrieve_sources import FEMA, acquire

ROSTER_SHA = "b8ad33116b3fd12b5c410b3774bff687a35ee817652350088157e66c26b3a2ed"
FIELDS = ["id", "disasterNumber", "declarationType", "state", "fyDeclared", "incidentType", "declarationDate"]
ROSTER_FIELDS = ["disasterNumber", "state", "incidentType", "declarationDate", "incidentBeginDate",
                 "closeout_nonmissing_rows", "declaration_rows", "latest_refresh"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("roster", type=Path)
    p.add_argument("cache", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    assert sha256(args.roster.read_bytes()).hexdigest() == ROSTER_SHA
    roster = pd.read_csv(args.roster, parse_dates=["declarationDate", "incidentBeginDate"])
    assert list(roster)==ROSTER_FIELDS and len(roster)==roster.disasterNumber.nunique()==143
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    raw, manifest = [], []
    offset, count = 0, None
    while count is None or offset < count:
        query = {"$filter": "fyDeclared eq 2025", "$select": ",".join(FIELDS),
                 "$top": 1000, "$skip": offset, "$inlinecount": "allpages", "$orderby": "id"}
        prepared = requests.Request("GET", FEMA, params=query).prepare().url
        filename = f"fy2025_declaration_types_{offset}.json"
        row = acquire("FEMA_metadata_only", prepared, filename, args.cache)
        payload = json.loads((args.cache/filename).read_text())
        count = payload["metadata"]["count"] if count is None else count
        observations = payload["DisasterDeclarationsSummaries"]
        assert observations and all(set(x) == set(FIELDS) and x["fyDeclared"] == 2025 for x in observations)
        assert count < 6000 and payload["metadata"]["count"] == count
        raw.extend(observations)
        manifest.append(row)
        offset += len(observations)
    meta = pd.DataFrame(raw)
    assert len(meta) == count and meta.id.is_unique
    selected = meta[meta.disasterNumber.isin(roster.disasterNumber)].drop(columns="id").drop_duplicates()
    assert selected.disasterNumber.is_unique
    selected["declarationDate"] = pd.to_datetime(selected.declarationDate)
    report = roster.merge(selected,on="disasterNumber",how="left",suffixes=("","_recheck"),validate="one_to_one")
    report["metadata_agrees"] = (report.state.eq(report.state_recheck)&report.incidentType.eq(report.incidentType_recheck)
                                  &report.declarationDate.eq(report.declarationDate_recheck))
    report["candidate_36m_outcome_cutoff"] = report.declarationDate + pd.DateOffset(months=36)
    report["candidate_90d_review_date"] = report.candidate_36m_outcome_cutoff + pd.Timedelta(days=90)
    report["outcome_access_authorized"] = False
    report.to_csv(args.output/"fy2025_target_free_protocol_audit.csv",index=False)
    pd.DataFrame(manifest).to_csv(args.output/"fy2025_declaration_type_sources.csv",index=False)
    print(report.declarationType.value_counts(dropna=False).to_string())
    print("Identity metadata agrees:", report.metadata_agrees.sum(), "of 143")


if __name__ == "__main__":
    main()
