#!/usr/bin/env python3
"""Verify original Federal Register declaration notices and availability bounds."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import json
from pathlib import Path
import re
from xml.etree import ElementTree
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from thesis_retrieve_sources import acquire

API = "https://www.federalregister.gov/api/v1/documents.json"
FIRST = "2018-10-01"
END = "2024-09-30"
EASTERN = ZoneInfo("America/New_York")
HORIZONS = (0, 24, 48, 72, 168, 336, 720)
FEMA_ID = re.compile(r"\bFEMA[-–—\s]+(\d{4})[-–—\s]+DR\b", re.IGNORECASE)
DATES = re.compile(r"([A-Z][a-z]+ \d{1,2}, \d{4})")
FILED = re.compile(r"Filed\s+(\d{1,2}-\d{1,2}-\d{2,4});\s*(\d{1,2}:\d{2})\s*([ap])\.?m\.?", re.I)
FIELDS = ["title", "abstract", "dates", "docket_ids", "document_number", "publication_date", "full_text_xml_url"]


def ledgers(raw):
    details, docs = [], []
    for year in range(2018, 2025):
        low, high = max(FIRST, f"{year}-01-01"), min(END, f"{year}-12-31")
        filters = [("conditions[agencies][]", "federal-emergency-management-agency"),
                   ("conditions[type][]", "NOTICE"), ("conditions[publication_date][gte]", low),
                   ("conditions[publication_date][lte]", high), ("per_page", 500), ("order", "oldest")]
        params = [*filters, *(("fields[]", field) for field in FIELDS)]
        url = requests.Request("GET", API, params=params).prepare().url
        current = []
        page = 0
        while url:
            if len(details) >= 30:
                raise ValueError("Federal Register index page cap reached")
            page += 1
            name = f"notice_index_{year}_{page}.json"
            details.append(acquire("federal_register_fema_index", url, name, raw))
            data = json.loads((raw / name).read_text())
            current.extend(data["results"])
            url = data.get("next_page_url")
            if not url:
                assert len(current) == data["count"]
        assert all(low <= row["publication_date"] <= high for row in current)
        docs.extend(current)
    assert len({row["document_number"] for row in docs}) == len(docs)
    return docs, details


def docket_ids(row):
    text = " ".join([*(row.get("docket_ids") or []), row.get("abstract") or ""])
    return sorted({int(number) for number in FEMA_ID.findall(text)})


def original(row):
    title = row["title"].lower()
    return "major disaster and related determinations" in title and not any(
           element in title for element in ("amendment", "correction"))


def retrieve_time(row, raw):
    detail = acquire("federal_register_original_notice", row["full_text_xml_url"],
                     f"notice_{row['document_number']}.xml", raw)
    xml = ElementTree.fromstring((raw / detail["filename"]).read_bytes())
    notes = [node.text or "" for node in xml.iter() if node.tag.split("}")[-1] == "FRDOC"]
    assert len(notes) <= 1
    notice = notes[0].replace("–", "-").replace("−", "-") if notes else ""
    pattern = FILED.search(notice)
    filing = None
    if pattern:
        day, clock, period = pattern.groups()
        fmt = "%m-%d-%Y" if len(day.rsplit("-", 1)[1]) == 4 else "%m-%d-%y"
        filing = datetime.strptime(f"{day} {clock} {period}m", f"{fmt} %I:%M %p").replace(tzinfo=EASTERN)
        assert filing.date().isoformat() <= row["publication_date"]
    dates = DATES.findall(row.get("dates") or "")
    date = pd.Timestamp(dates[0]).date().isoformat() if len(set(dates)) == 1 else ""
    return detail, {"disasterNumber": row["disasterNumber"], "document_number": row["document_number"],
                    "publication_date": row["publication_date"], "document_title": row["title"],
                    "declaration_date_in_notice": date, "filing_notation": notice,
                    "filed_utc": pd.Timestamp(filing).tz_convert("UTC").isoformat() if filing else ""}


def compute(cases, notices):
    result = []
    stop = pd.Timestamp(END, tz=EASTERN).tz_convert("UTC")
    for row in cases.itertuples(index=False):
        start = pd.Timestamp(row.declaration).tz_localize(EASTERN)
        next_day = start + pd.DateOffset(days=1)
        window = notices[notices.disasterNumber.eq(row.disasterNumber)].copy()
        if len(window):
            window = window[window.declaration_date_in_notice.eq(row.declaration)]
            window = window[window.filed_utc.ne("")].sort_values("filed_utc")
        has_filing = not window.empty
        first = window.iloc[0] if has_filing else None
        filing = pd.Timestamp(first.filed_utc) if has_filing else None
        assert filing is None or filing >= start - timedelta(days=1)
        lower = (filing-next_day).total_seconds()/3600 if has_filing else None
        upper = (filing-start).total_seconds()/3600 if has_filing else None
        for hour in HORIZONS:
            if has_filing:
                status = "covered" if upper <= hour else "late" if lower >= hour else "ambiguous"
            else:
                status = "right_censored" if next_day + timedelta(hours=hour) >= stop else "missing"
            result.append({"disasterNumber": row.disasterNumber, "fyDeclared": row.fyDeclared,
                 "declared_on": row.declaration, "horizon_hours": hour, "status": status,
                 "notice_id": first.document_number if has_filing else "", "filing_utc": first.filed_utc if has_filing else "",
                 "elapsed_lower_hours": lower, "elapsed_upper_hours": upper})
    return pd.DataFrame(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.raw.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    ids = set(pd.read_csv("research/results/forward/forward_predictions.csv", usecols=["disasterNumber"]).disasterNumber)
    cases = pd.read_csv("research/results/hazards/cohort_features.csv", usecols=["disasterNumber", "fyDeclared", "declaration"])
    cases = cases[cases.disasterNumber.isin(ids)]
    assert len(ids) == len(cases) == 376 and cases.fyDeclared.max() == 2024
    assert max(cases.declaration) <= END
    docs, source = ledgers(args.raw)
    candidates = []
    related = []
    for doc in docs:
        key = docket_ids(doc)
        match = set(key) & ids
        if match:
            related.append({"document_number": doc["document_number"], "publication_date": doc["publication_date"],
                            "title": doc["title"], "recognised_drs": "|".join(map(str, key)),
                            "original": original(doc), "unique_id": len(key) == 1})
            if original(doc) and len(key) == 1 and doc.get("full_text_xml_url"):
                candidates.append({**doc, "disasterNumber": key[0]})
    if len(candidates) > 400:
        raise ValueError("Federal Register original document cap reached")
    with ThreadPoolExecutor(max_workers=8) as pool:
        parsed = list(pool.map(lambda doc: retrieve_time(doc, args.raw), candidates))
    source.extend(item[0] for item in parsed)
    notices = pd.DataFrame([item[1] for item in parsed])
    rows = compute(cases, notices)
    summary = rows.groupby(["horizon_hours", "status"]).size().rename("declarations").reset_index()
    paths = {"horizon_coverage.csv": rows, "coverage_summary.csv": summary, "original_notice_filings.csv": notices,
             "related_notice_index.csv": pd.DataFrame(related), "source_manifest.csv": pd.DataFrame(source)}
    for name, frame in paths.items():
        frame.to_csv(args.output / name, index=False)
    print(f"Verified notice candidates for {notices.disasterNumber.nunique()} of 376 historical declarations.")


if __name__ == "__main__":
    main()
