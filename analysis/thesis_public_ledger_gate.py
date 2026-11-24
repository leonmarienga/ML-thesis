#!/usr/bin/env python3
"""Replay exact-ID OpenFEMA legacy queries without interpreting omissions as zeros."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from thesis_historical_coverage import MA_FIELDS, MISSIONS, START, aggregate

LEGACY = (1603, 1604, 1786, 1791, 4020)
CONTROLS = (4085, 4086)


def read_json(path):
    return json.loads(path.read_text())


def verify(raw, name, checksum):
    payload = (raw / name).read_bytes()
    assert sha256(payload).hexdigest() == checksum
    return payload


def read_docs(raw):
    documents = []
    for group in ("document_attempts", "catalogue_metadata", "policy_alternatives"):
        for item in read_json(raw / (group + ".json")):
            if item["filename"]:
                verify(raw, item["filename"], item["sha256"])
            documents.append({"route_category": group, **item})
    return pd.DataFrame(documents)


def result_tables(raw):
    prior = pd.read_csv("research/continuous/coverage/audit/declaration_roster.csv", usecols=[
        "disasterNumber", "state", "declarationDate", "fyDeclared", "declarationType",
    ])
    prior = prior[prior.disasterNumber.isin(LEGACY)]
    assert len(prior) == len(LEGACY) and prior.declarationType.eq("DR").all()
    assert pd.to_datetime(prior.declarationDate, utc=True).lt(START).all()
    attempts = read_json(raw / "legacy_attempts.json")
    assert tuple(item["id"] for item in attempts) == LEGACY
    outcomes = []
    for source in attempts:
        pages = source["pages"]
        actions, allowed_keys = [], set(MA_FIELDS.split(","))
        expected = None
        id = source["id"]
        for meta in pages:
            original = verify(raw, meta["filename"], meta["sha256"])
            data = json.loads(original)
            assert data["metadata"]["entityname"] == MISSIONS and data["metadata"]["version"] == "v2"
            assert data["metadata"]["filter"] == f"disasterNumber eq {id}"
            expected = int(data["metadata"]["count"])
            assert all(int(r["disasterNumber"]) == id and set(r) <= allowed_keys for r in data[MISSIONS])
            actions.extend(data[MISSIONS])
        if source["status"] == "read":
            assert expected == len(actions)
        summary = {"disasterNumber": id, "query_status": source["status"], "scoped_response_pages": len(pages),
                   "returned_actions": len(actions), "advertised_precoverage": True,
                   "signed_complete_legacy_ledger_found": False, "eligible_new_elapsed_label": False}
        if actions:
            keys = pd.DataFrame(actions, columns=MA_FIELDS.split(","))
            checks = aggregate(keys)
            summary["returned_action_audit_complete"] = bool(checks.iloc[0].get("all_dates", False))
        outcomes.append(summary)
    joined = prior.merge(pd.DataFrame(outcomes), on="disasterNumber", validate="one_to_one")
    reference = pd.read_csv("research/continuous/horizons/results/historical_horizons.csv", usecols=[
        "disasterNumber", "state", "event_group", "actions", "negative_actions", "unique_actions", "unique_amendments",
        "all_dates", "keys_complete", "snapshot_status",
    ])
    reference = reference[reference.disasterNumber.isin(CONTROLS)]
    assert len(reference) == len(CONTROLS) and reference.actions.gt(0).all() and reference.negative_actions.gt(0).all()
    gate = pd.DataFrame([{"legacy_declarations_checked": len(joined), "legacy_http_reads": joined.query_status.eq("read").sum(),
                          "legacy_actions": joined.returned_actions.sum(), "positive_route_control_declarations": len(reference),
                          "control_actions": reference.actions.sum(), "control_negative_actions": reference.negative_actions.sum(),
                          "eligible_new_elapsed_labels": joined.eligible_new_elapsed_label.sum()}])
    return {"legacy_queries": joined, "retained_control": reference, "documentary_routes": read_docs(raw), "gate_summary": gate}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("raw", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, table in result_tables(args.raw).items():
        table.to_csv(args.output / (name + ".csv"), index=False)


if __name__ == "__main__":
    main()
