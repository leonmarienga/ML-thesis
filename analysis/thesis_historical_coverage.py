#!/usr/bin/env python3
"""Enumerate historical DR records and audit the public Mission Assignment publication boundary."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from thesis_retrieve_sources import acquire

API = "https://www.fema.gov/api/open/v2"
FIRST_YEAR, LAST_YEAR = 1995, 2024
START = pd.Timestamp("2012-10-01", tz="UTC")
PAGE_SIZE = 1000
DECLARATIONS = "DisasterDeclarationsSummaries"
MISSIONS = "MissionAssignments"
DR_FIELDS = "id,disasterNumber,fyDeclared,state,declarationType,incidentType,declarationDate,incidentBeginDate"
MA_FIELDS = "actionId,maId,maAmendNumber,disasterNumber,obligationAmount,dateObligated,dateReceived"


class HistoricalSource:
    def __init__(self, path):
        self.path = path
        self.manifest = []
        path.mkdir(parents=True, exist_ok=True)

    def retrieve(self, entity, label, filters, fields, sort, cap):
        rows = []
        for offset in range(0, PAGE_SIZE * cap, PAGE_SIZE):
            name = f"{label}_{offset}.json"
            params = {"$top": PAGE_SIZE, "$skip": offset, "$inlinecount": "allpages",
                      "$filter": filters, "$select": fields, "$orderby": sort}
            url = requests.Request("GET", f"{API}/{entity}", params=params).prepare().url
            details = acquire(entity, url, name, self.path)
            payload = json.loads((self.path / name).read_text())
            meta = payload["metadata"]
            assert meta["entityname"] == entity and meta["version"] == "v2"
            assert meta["filter"] == filters
            page = payload[entity]
            self.manifest.append({**details, "row_count": len(page), "full_count": int(meta["count"])})
            rows.extend(page)
            if len(rows) >= int(meta["count"]):
                assert len(rows) == int(meta["count"])
                return rows
        raise RuntimeError(f"{label} exceeded the request budget")


def declaration_roster(source):
    roster = []
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        remaining = 100 - len(source.manifest)
        filters = f"fyDeclared eq {year} and declarationType eq 'DR' and incidentType ne 'Biological'"
        rows = pd.DataFrame(source.retrieve(DECLARATIONS, f"dds_{year}", filters, DR_FIELDS, "id asc", remaining))
        assert not rows.empty and rows.fyDeclared.eq(year).all() and rows.id.is_unique
        assert rows.declarationType.eq("DR").all() and not rows.incidentType.eq("Biological").any()
        columns = ["fyDeclared", "state", "declarationType", "incidentType", "declarationDate", "incidentBeginDate"]
        unique = rows[["disasterNumber", *columns]].drop_duplicates()
        if not unique.disasterNumber.is_unique:
            raise ValueError(f"Inconsistent declaration-level fields in FY{year}")
        unique = unique.merge(rows.groupby("disasterNumber").size().rename("dds_rows"), on="disasterNumber")
        roster.append(unique)
    d = pd.concat(roster, ignore_index=True).sort_values("disasterNumber")
    assert d.disasterNumber.is_unique
    assert pd.to_datetime(d.declarationDate, utc=True).max() < pd.Timestamp("2024-10-01", tz="UTC")
    return d


def original_sample(d):
    counts = {"zero": 8, "funded_below_50m": 8, "50_to_200m": 4, "200_to_500m": 4, "500m_plus": 4}
    scope = d[pd.to_datetime(d.declaration, utc=True).ge(START)].copy()
    scope["stratum"] = pd.cut(scope.target, [-np.inf, .001, 50e6, 200e6, 500e6, np.inf],
                              labels=counts, right=False)
    scope["order"] = scope.disasterNumber.map(lambda key: sha256(f"historical-coverage-v1:{key}".encode()).hexdigest())
    return pd.concat([scope[scope.stratum.eq(key)].nsmallest(n, "disasterNumber") if key == "zero" else
                      scope[scope.stratum.eq(key)].sort_values("order").head(n) for key, n in counts.items()])


def get_audit(source, ids, forbidden):
    ids = sorted({int(value) for value in ids})
    assert not set(ids) & forbidden
    first = len(source.manifest)
    all_rows = []
    for start in range(0, len(ids), 28):
        block = ids[start:start + 28]
        filters = " or ".join(f"disasterNumber eq {key}" for key in block)
        remaining = 60 - (len(source.manifest) - first)
        rows = source.retrieve(MISSIONS, f"ma_{start}", filters, MA_FIELDS, "actionId asc", remaining)
        assert {int(row["disasterNumber"]) for row in rows} <= set(block)
        all_rows.extend(rows)
    return pd.DataFrame(all_rows, columns=MA_FIELDS.split(","))


def aggregate(actions, metadata):
    assert metadata.disasterNumber.is_unique
    a = actions.merge(metadata[["disasterNumber", "declaration_date"]], on="disasterNumber", validate="many_to_one")
    assert len(a) == len(actions)
    a["obligationAmount"] = pd.to_numeric(a.obligationAmount, errors="raise")
    assert a.obligationAmount.notna().all() and a.actionId.notna().all()
    a["unique_action"] = ~a.duplicated("actionId", keep=False)
    a["unique_amendment"] = ~a.duplicated(["disasterNumber", "maId", "maAmendNumber"], keep=False)
    a["obligated"] = pd.to_datetime(a.dateObligated, utc=True, errors="coerce")
    a["has_obligation_date"] = a.obligated.notna()
    a["source_structure_complete"] = a.maId.notna() & a.maAmendNumber.notna()
    for months in (12, 24, 36, 60):
        horizon = a.declaration_date + pd.DateOffset(months=months)
        a[f"net_through_{months}m_usd"] = a.obligationAmount.where(a.obligated.le(horizon), 0.)
    out = a.groupby("disasterNumber").agg(actions=("actionId", "size"), assignments=("maId", "nunique"),
        unique_actions=("unique_action", "all"), unique_amendments=("unique_amendment", "all"),
        all_dates=("has_obligation_date", "all"), keys_complete=("source_structure_complete", "all"),
        net_obligation_usd=("obligationAmount", "sum"),
        negative_actions=("obligationAmount", lambda x: int(x.lt(0).sum())),
        first_obligation_utc=("obligated", "min"), last_obligation_utc=("obligated", "max"),
        **{f"net_through_{m}m_usd": (f"net_through_{m}m_usd", "sum") for m in (12, 24, 36, 60)})
    d = metadata.merge(out, left_on="disasterNumber", right_index=True, how="left", validate="one_to_one")
    d["snapshot_status"] = np.select([d.actions.isna(), d.unique_actions.eq(False) | d.unique_amendments.eq(False) |
         d.keys_complete.eq(False), d.all_dates.eq(False)], ["no_public_rows", "ambiguous_action_structure",
         "missing_obligation_date"], default="published_actions")
    return d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = HistoricalSource(args.raw)
    frozen = pd.read_csv("research/results/hazards/cohort_features.csv", parse_dates=["declaration"])
    locked = set(pd.read_csv("research/results/follow_up/fy2025_target_free_roster.csv", usecols=["disasterNumber"]).disasterNumber)
    roster = declaration_roster(source)
    assert not set(roster.disasterNumber) & locked
    roster["in_workbook"] = roster.disasterNumber.isin(frozen.disasterNumber)
    roster.to_csv(args.output / "declaration_roster.csv", index=False)
    sample = original_sample(frozen[~frozen.incidentType.eq("Biological")])
    missing = roster[(~roster.in_workbook) & roster.fyDeclared.ge(2013)].copy()
    overlap = roster[roster.disasterNumber.isin(sample.disasterNumber)].copy()
    assert len(overlap) == len(sample)
    metadata = pd.concat([missing.assign(scope="omitted"), overlap.assign(scope="workbook_check")], ignore_index=True)
    metadata["declaration_date"] = pd.to_datetime(metadata.declarationDate, utc=True)
    metadata = metadata.merge(sample[["disasterNumber", "stratum", "target"]], on="disasterNumber", how="left")
    results = get_audit(source, metadata.disasterNumber, locked)
    results.to_csv(args.output / "mission_actions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    aggregate(results, metadata).to_csv(args.output / "mission_obligation_audit.csv", index=False)
    pd.DataFrame(source.manifest).to_csv(args.output / "source_manifest.csv", index=False)
    print(f"Enumerated {len(roster):,} historical declarations and audited {len(metadata):,} IDs with {len(results):,} actions.")


if __name__ == "__main__":
    main()
