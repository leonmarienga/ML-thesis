#!/usr/bin/env python3
"""Audit immutable Track A predictions and reserve a target-free follow-up cohort."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from thesis_forward_models import detailed_metrics
from thesis_hazard_features import BANDS, read_master
from thesis_retrieve_sources import acquire, FEMA


def audit_track_a(workbook, archive, rebuilt, output):
    archived = pd.read_csv(archive).sort_values("disasterNumber").reset_index(drop=True)
    reconstructed = pd.read_csv(rebuilt).sort_values("disasterNumber").reset_index(drop=True)
    pd.testing.assert_frame_equal(archived, reconstructed, check_exact=False, atol=2e-8, rtol=0)
    master = read_master(workbook).query("incidentType != 'Biological'")
    keys = archived.merge(master, on=["disasterNumber", "fyDeclared", "state", "incidentType"],
                          suffixes=("", "_workbook"), validate="one_to_one")
    assert len(keys) == 912 and np.allclose(keys.target, keys.target_workbook, rtol=0, atol=.005)
    assert keys.actual_band.tolist() == [BANDS[i] for i in keys.actual_band_workbook]
    assert int(archived.actual_band.eq(archived.router_pred).sum()) == 830
    names = {
        "frozen_stack_prediction": "track_a_original_stack",
        "accepted_low_band_prediction": "track_a_two_accepted_low_specialists",
        "challenger_stack_prediction": "track_a_reported_historical_50_200m_overlay",
    }
    results = []
    for column, name in names.items():
        d = archived.assign(variant=name, predicted_amount=archived[column])
        d["actual_band"] = d.actual_band.map(BANDS.index)
        d["predicted_band"] = d.router_pred.map(BANDS.index)
        # The archive has hard routes only. The one-hot representation permits confusion
        # summaries, but its Brier score cannot be interpreted as probability calibration.
        for i in range(6):
            d[f"p_band_{i}"] = d.predicted_band.eq(i).astype(float)
        results.append(d)
    detailed_metrics(pd.concat(results), output)
    path = output / "model_metrics.csv"
    metrics = pd.read_csv(path).drop(columns="multiclass_brier")
    metrics.to_csv(path, index=False)
    print(metrics.query("axis == 'all'").to_string(index=False))


def reserve_cohort(output):
    raw = output / "raw"
    raw.mkdir(exist_ok=True)
    rows, files = [], []
    expected, offset = None, 0
    while expected is None or offset < expected:
        params = {"$filter": "fyDeclared eq 2025", "$top": 1000, "$skip": offset,
                  "$inlinecount": "allpages", "$orderby": "id",
                  "$select": "id,disasterNumber,state,incidentType,fyDeclared,declarationDate,incidentBeginDate,incidentEndDate,disasterCloseoutDate,lastRefresh"}
        url = requests.Request("GET", FEMA, params=params).prepare().url
        filename = f"fema_target_free_fy2025_{offset}.json"
        files.append(acquire("openfema_target_free", url, filename, raw))
        data = json.loads((raw / filename).read_text())
        expected = data["metadata"]["count"] if expected is None else expected
        page = data["DisasterDeclarationsSummaries"]
        assert page and data["metadata"]["count"] == expected and expected < 6000
        rows.extend(page)
        offset += len(page)
    d = pd.DataFrame(rows)
    assert len(d) == expected and d.id.is_unique
    d = d.query("incidentType != 'Biological'")
    unique = d.groupby("disasterNumber").agg(state=("state", "first"), incidentType=("incidentType", "first"),
        declarationDate=("declarationDate", "min"), incidentBeginDate=("incidentBeginDate", "min"),
        closeout_nonmissing_rows=("disasterCloseoutDate", "count"), declaration_rows=("id", "size"),
        latest_refresh=("lastRefresh", "max")).reset_index()
    unique.to_csv(output/"fy2025_target_free_roster.csv", index=False)
    pd.DataFrame(files).to_csv(output/"fy2025_metadata_manifest.csv", index=False)
    print(f"FY2025 target-free non-Biological candidates: {len(unique)}; any closeout: {unique.closeout_nonmissing_rows.gt(0).sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-a", type=Path, required=True)
    parser.add_argument("--track-a-rebuilt", type=Path, required=True)
    parser.add_argument("--workbook", type=Path, default=Path("master_openfema_40plus.xlsx"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-free", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.target_free.mkdir(parents=True, exist_ok=True)
    audit_track_a(args.workbook, args.track_a, args.track_a_rebuilt, args.output)
    reserve_cohort(args.target_free)
