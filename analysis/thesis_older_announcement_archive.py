#!/usr/bin/env python3
"""Check earlier announcement identity separately from the vintage of its text."""
import argparse
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from html import unescape
import json
from pathlib import Path
import re
from time import sleep
from urllib.parse import urlencode, urlparse

import pandas as pd
import requests

from thesis_forward_models import YEARS, split
from thesis_whitehouse_announcements import EASTERN, MONTH, STATES, match_article, xml_urls

HOST = "trumpwhitehouse.archives.gov"
CAPTURE_HOST = "web.archive.org"
ROOT = f"https://{HOST}/sitemap/sitemap.xml"
SAMPLE = re.compile(r"/sitemap-pt-briefing-statement-(201[7-9]|2020)-(01|04|07|10)\.xml$")
APPROVAL = re.compile(r"\bdeclared (?:that )?a major disaster exists\b", re.I)
BODY_STATE = re.compile(r"\bexists in (?:the )?(?:(?:state|territory|commonwealth) of )?(.+?) and ordered\b", re.I)
ONSET = re.compile(rf"\b(?:from|beginning(?: on)?|on|during the period of)\s+({MONTH}\s+\d{{1,2}})\b", re.I)
DISPLAY_DATE = re.compile(rf"^{MONTH} \d{{1,2}}, 20\d{{2}}$", re.I)
SOURCE_END = "2020-12-31"
HOURS = (0, 24, 48, 72, 168, 336, 720)
CAPTURE_LIMIT = 12
BODY_STATE_CODES = {name.replace("-", " "): key.upper() for key, name in STATES.items()}


class Evidence:
    def __init__(self, raw, fetch):
        self.raw = raw
        self.fetch = fetch
        self.manifest_path = raw / "source_manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}

    def get(self, url, suffix):
        host = urlparse(url).hostname
        assert host in {HOST, CAPTURE_HOST}
        replay = host == CAPTURE_HOST and urlparse(url).path.startswith("/web/")
        if url in self.manifest and (not replay or "memento_datetime" in self.manifest[url]):
            record = self.manifest[url]
            if not record["filename"]:
                return None
            path = self.raw / record["filename"]
            content = path.read_bytes()
            assert len(content) == record["bytes"] and sha256(content).hexdigest() == record["sha256"]
            return path
        if not self.fetch:
            raise ValueError(f"Frozen evidence has no entry for {url}")
        past = [datetime.fromisoformat(row["retrieved_utc"]) for key, row in self.manifest.items()
                if urlparse(key).hostname == host]
        if past:
            spacing = 10 if host == HOST else 2
            sleep(max(0, spacing - (datetime.now(timezone.utc) - max(past)).total_seconds()))
        resolved, memento = "", ""
        try:
            response = requests.get(url, timeout=25)
            assert urlparse(response.url).hostname == host and len(response.content) < 2_000_000
            status, content = response.status_code, response.content
            resolved = response.url
            memento = response.headers.get("Memento-Datetime", "")
        except requests.exceptions.RequestException:
            status, content = 0, b""
        name = sha256(url.encode()).hexdigest()[:20] + suffix if status == 200 else ""
        record = {"url": url, "status": status, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                  "filename": name, "bytes": len(content), "sha256": sha256(content).hexdigest()}
        if replay:
            record.update(resolved_url=resolved, memento_datetime=memento)
        if name:
            (self.raw / name).write_bytes(content)
        self.manifest[url] = record
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2) + "\n")
        print(status, url, flush=True)
        return self.raw / name if name else None


def strip_markup(text):
    return " ".join(unescape(re.sub(r"<[^>]+>", " ", text)).split())


def notice(path):
    html = path.read_text(encoding="utf-8", errors="replace") if path else ""
    times = {strip_markup(value) for value in re.findall(r"<time\b[^>]*>(.*?)</time>", html, re.I | re.S)}
    dates = sorted(value for value in times if DISPLAY_DATE.fullmatch(value))
    release = pd.Timestamp(dates[0]) if len(dates) == 1 else pd.NaT
    main = re.search(r"<main\b[^>]*>(.*?)</main>", html, re.I | re.S)
    paragraphs = [strip_markup(value) for value in re.findall(r"<p\b[^>]*>(.*?)</p>",
                  main[1] if main else html, re.I | re.S)]
    position = next((i for i, para in enumerate(paragraphs) if APPROVAL.search(para)), None)
    opening = paragraphs[position] if position is not None else ""
    body_state = BODY_STATE.search(opening)
    state = BODY_STATE_CODES.get(re.sub(r"^the ", "", body_state[1].lower())) if body_state else None
    onset = ONSET.search(opening.split("affected by", 1)[-1]) if opening else None
    available_years = set(re.findall(r"\b(?:19|20)\d{2}\b", onset.string[onset.start():])) if onset else set()
    relation = ("yesterday" if re.search(r"\byesterday\b", opening, re.I)
                else "today" if re.search(r"\btoday\b", opening, re.I) else "unmarked")
    event_day = release - timedelta(days=int(relation == "yesterday")) if relation != "unmarked" else pd.NaT
    return {"state": state, "inferred_declaration_day": event_day, "release_civil_day": release,
            "incidence_relation": relation, "incident_start_month_day_reference": onset[1] if onset else "",
            "incident_start_year_reference": next(iter(available_years)) if len(available_years) == 1 else "",
            "opening": opening, "body": paragraphs[position:] if position is not None else [],
            "modification_metadata_tags": len(re.findall(r"article:modified_time|dateModified", html, re.I))}


def captures(store, link, start, release, body, expected, cohort):
    original = "https://www.whitehouse.gov" + urlparse(link).path
    deadline = start + timedelta(hours=72)
    query = urlencode({"url": original, "from": start.strftime("%Y%m%d"), "to": deadline.strftime("%Y%m%d"),
                      "output": "json", "limit": 10, "filter": "statuscode:200"})
    resource = f"https://{CAPTURE_HOST}/cdx/search/cdx?{query}"
    saved = store.get(resource, ".json")
    if saved is None:
        return "index_unavailable", resource, []
    try:
        metadata = json.loads(saved.read_text())
        stamps = [dict(zip(metadata[0], row))["timestamp"] for row in metadata[1:]] if metadata else []
    except (ValueError, KeyError, TypeError):
        return "index_unreadable", resource, []
    details = []
    for stamp in sorted(set(stamps)):
        when = pd.to_datetime(stamp, format="%Y%m%d%H%M%S", utc=True)
        if not release <= when <= deadline:
            continue
        url = f"https://{CAPTURE_HOST}/web/{stamp}id_/{original}"
        downloaded = store.get(url, ".html")
        evidence = store.manifest[url]
        parsed = notice(downloaded)
        body_text = parsed.pop("body")
        nr, _ = match_article(parsed, cohort)
        clock = pd.to_datetime(evidence["memento_datetime"], format="%a, %d %b %Y %H:%M:%S GMT",
                               errors="coerce", utc=True)
        capture_verified = evidence["resolved_url"] == url and clock == when and nr == expected
        details.append({"article": link, "disasterNumber": expected, "capture_url": url, "capture_utc": when,
                        "elapsed_upper_hours": (when - start).total_seconds() / 3600,
                        "content_identity_verified": capture_verified, "same_body_as_nara": bool(body and body == body_text),
                        "source_sha256": evidence["sha256"], "body": " | ".join(body_text)})
    if any(row["content_identity_verified"] for row in details):
        result = "verified_early_version"
    else:
        result = "early_metadata_without_verified_body" if details else "no_capture_in_window"
    return result, resource, details


def support(cohort, records):
    ledger = pd.read_csv("research/continuous/horizons/results/historical_horizons.csv",
                         usecols=["disasterNumber", "release_review_date", "snapshot_status"])
    ready = pd.to_datetime(ledger.release_review_date, utc=True)
    population = set(cohort.loc[cohort.declaration.between("2017-01-01", SOURCE_END), "disasterNumber"])
    matched = records[records.status.eq("matched")]
    rows = []
    for year in YEARS:
        prior, held = split(cohort, year)
        cutoff = min(held.start.min(), held.declaration.min()).tz_localize("UTC")
        mature = set(ledger.loc[ready.lt(cutoff) & ledger.snapshot_status.eq("published_actions"), "disasterNumber"])
        for hour in HOURS:
            with_claim = set(matched.loc[matched.release_claim_upper_hours.le(hour), "disasterNumber"])
            verified = set(matched.loc[matched.capture_elapsed_upper_hours.le(hour), "disasterNumber"])
            rows.append({"held_fy": year, "horizon_hours": hour,
                         "mature_eligible_population_ceiling": prior.disasterNumber.isin(population & mature).sum(),
                         "mature_sample_claim_upper_bound": prior.disasterNumber.isin(with_claim & mature).sum(),
                         "mature_sample_verified_lower_bound": prior.disasterNumber.isin(verified & mature).sum(),
                         "held_sample_claim_upper_bound": held.disasterNumber.isin(with_claim).sum(),
                         "held_sample_verified_lower_bound": held.disasterNumber.isin(verified).sum()})
    return pd.DataFrame(rows)


def extended_capture_selection(matched_bodies, legacy_urls):
    ordered = sorted(matched_bodies, key=lambda link: (matched_bodies[link][0], link))
    retained = [link for link in ordered if link in legacy_urls]
    remaining = [link for link in ordered if link not in legacy_urls]
    for year in range(2017, 2021):
        retained.extend([link for link in remaining if matched_bodies[link][1].year == year][:4])
    return set(retained)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--extend-quarterly-sample", action="store_true")
    args = parser.parse_args()
    args.raw.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    store = Evidence(args.raw, args.fetch)
    dates = ["start", "end", "declaration"]
    columns = ["disasterNumber", "state", "incidentType", "fyDeclared", "event_group", *dates]
    cohort = pd.read_csv("research/results/hazards/cohort_features.csv", usecols=columns, parse_dates=dates)
    cohort = cohort[cohort.incidentType.ne("Biological")]
    assert len(cohort) == 912 and cohort.disasterNumber.is_unique and cohort.fyDeclared.max() == 2024
    identity = cohort[cohort.declaration.between("2017-01-01", SOURCE_END)]
    maps = sorted(url for url in xml_urls(store.get(ROOT, ".xml")) if SAMPLE.search(url))
    assert len(maps) == 16
    candidates, records, matched_bodies, versions = [], [], {}, []
    legacy_urls = set()
    articles_per_month = 4 if args.extend_quarterly_sample else 2
    for sitemap in maps:
        urls = sorted({url for url in xml_urls(store.get(sitemap, ".xml"))
                       if re.search(r"president-.*(?:approves|signs|declares).*disaster-declaration", url)
                       and "emergency-declaration" not in url})
        for i, url in enumerate(urls):
            candidates.append({"sitemap": sitemap, "article": url, "sampled": i < articles_per_month})
            if i >= articles_per_month:
                continue
            if i < 2:
                legacy_urls.add(url)
            data = notice(store.get(url, ".html"))
            body = data.pop("body")
            keyed = {**data, "inferred_declaration_day": data["inferred_declaration_day"].strftime("%Y-%m-%d")
                     if pd.notna(data["inferred_declaration_day"]) else ""}
            nr, status = match_article(keyed, identity)
            day = data["inferred_declaration_day"]
            release = data["release_civil_day"]
            claim = ((release + pd.DateOffset(days=1)).tz_localize(EASTERN) - day.tz_localize(EASTERN)
                     ).total_seconds() / 3600 if nr is not None else None
            records.append({"article": url, "disasterNumber": nr, "status": status, **keyed,
                            "release_claim_upper_hours": claim, "capture_status": "not_queried",
                            "verified_capture_utc": None, "capture_query": "", "capture_elapsed_upper_hours": None,
                            "earliest_capture_text_matches_later_copy": None})
            if nr is not None:
                matched_bodies[url] = (day.tz_localize(EASTERN), release.tz_localize(EASTERN), body)
    selected = (extended_capture_selection(matched_bodies, legacy_urls) if args.extend_quarterly_sample
                else set(sorted(matched_bodies, key=lambda link: (matched_bodies[link][0], link))[:CAPTURE_LIMIT]))
    for row in records:
        link = row["article"]
        if link in selected:
            day, release, body = matched_bodies[link]
            state, source, snapshots = captures(store, link, day, release, body, row["disasterNumber"], identity)
            versions.extend(snapshots)
            row.update(capture_status=state, capture_query=source)
            first = next((item for item in snapshots if item["content_identity_verified"]), None)
            if first:
                row.update(verified_capture_utc=first["capture_utc"], capture_elapsed_upper_hours=first["elapsed_upper_hours"],
                           earliest_capture_text_matches_later_copy=first["same_body_as_nara"])
    records = pd.DataFrame(records)
    for name, table in [("sample_frame", pd.DataFrame(candidates)), ("document_identity", records),
                        ("capture_versions", pd.DataFrame(versions)),
                        ("source_manifest", pd.DataFrame(store.manifest.values())), ("mature_support", support(cohort, records))]:
        table.to_csv(args.output / f"{name}.csv", index=False)
    print(f"Found {records.disasterNumber.nunique()} matched identities across {len(records)} notices.")


if __name__ == "__main__":
    main()
