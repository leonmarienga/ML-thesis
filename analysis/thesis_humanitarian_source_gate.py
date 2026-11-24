#!/usr/bin/env python3
"""Reproduce independent published need and service observations with source gates."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

from openpyxl import load_workbook
import pandas as pd

from thesis_precoverage_documentary import text_pages

WEEKS = {52: ("2022-12-09", "2022-12-19"), 61: ("2023-08-23", "2023-09-04"), 62: ("2023-09-20", "2023-10-02")}
FIELDS = {
    "displacement": (9, {1: "adult_population", 2: "displaced_any_prior_year", 3: "displaced_hurricane_prior_year",
                        9: "not_displaced", 10: "displacement_did_not_report"}),
    "water_food": (8, {1: "displaced_adult_population", 5: "food_shortage_a_lot", 6: "food_shortage_did_not_report",
                       10: "water_shortage_a_lot", 11: "water_shortage_did_not_report"}),
}
META = ("fema_shelter_service.json", "fema_shelter_layer.json", "fema_tempo_layer.json")


def verify(raw, name):
    record = json.loads((raw / (name + ".json")).read_text())
    assert record["filename"] == name
    assert sha256((raw / name).read_bytes()).hexdigest() == record["sha256"]
    return record


def finite_number(value):
    return value if isinstance(value, (int, float)) else None


def hps_tables(raw):
    records, manifest = [], []
    for week, (start, end) in WEEKS.items():
        manifest.append(verify(raw, f"census_hps_{week}.html"))
        for table, (row, columns) in FIELDS.items():
            books = []
            for kind in ("est", "se"):
                file = f"hps_{week}_{table}_{kind}.xlsx"
                books.append(load_workbook(raw / file, read_only=True, data_only=True))
                manifest.append(verify(raw, file))
            estimates, errors = books
            assert estimates.sheetnames == errors.sheetnames
            for region in estimates.sheetnames:
                if not re.fullmatch(r"[A-Z]{2}", region):
                    continue
                vectors = [list(sheet[region].values)[:row] for sheet in books]
                if table == "displacement":
                    assert all(v[6][3] == "Hurricane" for v in vectors)
                else:
                    assert all(v[5][5] == v[5][10] == "A lot" for v in vectors)
                assert all(v[row - 1][0] == "Total" and f"Week {week}" in v[1][0] for v in vectors)
                for column, variable in columns.items():
                    estimate, standard_error = [finite_number(v[row - 1][column]) for v in vectors]
                    records.append({"week": week, "collection_start": start, "collection_end": end,
                                    "region": region, "table": table, "measure": variable, "adults_estimate": estimate,
                                    "adults_standard_error": standard_error, "published_numeric_estimate": estimate is not None})
            for book in books:
                book.close()
    values = pd.DataFrame(records)
    wide = values.pivot(index=["week", "region"], columns="measure", values="adults_estimate").reset_index()
    assert (wide.displaced_adult_population - wide.displaced_any_prior_year).abs().dropna().le(1).all()
    for resource in ("food", "water"):
        label = resource + "_shortage"
        wide[label + "_share_of_displaced"] = wide[label + "_a_lot"] / wide.displaced_adult_population
        answered = wide.displaced_adult_population - wide[label + "_did_not_report"]
        wide[label + "_share_of_reported"] = wide[label + "_a_lot"] / answered.where(answered.gt(0))
    return values, wide, manifest


def survey_audit(raw, extracts):
    source = verify(raw, "casper_usvi.pdf")
    rows = pd.read_csv(extracts)
    pages = text_pages(raw / "casper_usvi.pdf")
    for item in rows.itertuples():
        page = pages[item.pdf_page - 1]
        tokens = item.marker.split("|")
        assert all(token.casefold() in page.casefold() for token in tokens), item.measure
    return rows, source


def occupancy_metadata(raw):
    summaries, details = [], []
    for file in META:
        details.append(verify(raw, file))
        block = json.loads((raw / file).read_text())
        keys = set(x["name"].lower() for x in block.get("fields", []))
        capabilities = block.get("advancedQueryCapabilities", {})
        supports_historical = block.get("supportsQueryWithHistoricMoment", capabilities.get("supportsQueryWithHistoricMoment"))
        summaries.append({"metadata_file": file, "name": block.get("name") or "FEMA_NSS Service",
                          "versioned_data_advertised": block.get("hasVersionedData"),
                          "historic_moment_advertised": supports_historical, "time_series_metadata": bool(block.get("timeInfo")),
                          "observed_population_field": bool(keys & {"total_population", "popinopenshelters"}),
                          "capacity_field": bool(keys & {"evacuation_capacity", "post_impact_capacity", "sheltertotalcapacity"}),
                          "county_field": bool(keys & {"county_parish", "counties_cnty_fips"}),
                          "event_field": "event" in keys, "published_field_names": "|".join(sorted(keys))})
    return pd.DataFrame(summaries), details


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("raw", type=Path)
    cli.add_argument("casper_extract", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    values, summary, docs = hps_tables(args.raw)
    surveys, c = survey_audit(args.raw, args.casper_extract)
    shelter, d = occupancy_metadata(args.raw)
    for name, table in [
        ("hps_weighted_tables", values), ("hps_summary", summary), ("casper_published_estimates", surveys),
        ("shelter_metadata_screen", shelter), ("source_manifest", pd.DataFrame([*docs, c, *d])),
    ]:
        table.to_csv(args.output / (name + ".csv"), index=False)


if __name__ == "__main__":
    main()
