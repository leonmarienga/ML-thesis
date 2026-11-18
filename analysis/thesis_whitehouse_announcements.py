#!/usr/bin/env python3
"""Audit archived original White House public announcement timestamps without outcomes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from html import unescape
import json
from pathlib import Path
import re
from urllib.parse import urlparse
from xml.etree import ElementTree
from zoneinfo import ZoneInfo

import pandas as pd

from thesis_declaration_publication import HORIZONS
from thesis_retrieve_sources import acquire, public_get

HOST = "bidenwhitehouse.archives.gov"
INDEX = f"https://{HOST}/wp-sitemap.xml"
END = "2024-09-30"
EASTERN = ZoneInfo("America/New_York")
MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
SOURCE_DATE = re.compile(rf"\b(?:from|beginning(?:\s+on)?|on)\s+({MONTH}\s+\d{{1,2}})(?:,\s*(\d{{4}}))?\b", re.I)
APPROVAL = re.compile(r"\b(today|yesterday)\s*,\s*president\s+.{0,140}?\s+declared that a major disaster", re.I)
URL_DATE = re.compile(r"/(20\d\d)/(\d\d)/(\d\d)/")
STATES = dict(zip(
    ["al","ak","az","ar","ca","co","ct","de","dc","fl","ga","hi","id","il","in","ia","ks","ky","la","me",
     "md","ma","mi","mn","ms","mo","mt","ne","nv","nh","nj","nm","ny","nc","nd","oh","ok","or","pa","ri",
     "sc","sd","tn","tx","ut","vt","va","wa","wv","wi","wy","as","gu","mp","pr","vi"],
    "alabama|alaska|arizona|arkansas|california|colorado|connecticut|delaware|district-of-columbia|florida|georgia|hawaii|idaho|illinois|indiana|iowa|kansas|kentucky|louisiana|maine|maryland|massachusetts|michigan|minnesota|mississippi|missouri|montana|nebraska|nevada|new-hampshire|new-jersey|new-mexico|new-york|north-carolina|north-dakota|ohio|oklahoma|oregon|pennsylvania|rhode-island|south-carolina|south-dakota|tennessee|texas|utah|vermont|virginia|washington|west-virginia|wisconsin|wyoming|american-samoa|guam|northern-mariana-islands|puerto-rico|virgin-islands".split("|"),
    strict=True))


def xml_urls(path):
    return [tag.text for tag in ElementTree.fromstring(path.read_bytes()).iter() if tag.tag.endswith("}loc")]


def archive_index(raw):
    detail = acquire("nara_white_house_sitemap", INDEX, "sitemap_index.xml", raw)
    maps = [url for url in xml_urls(raw / detail["filename"]) if re.search(r"/post-sitemap\d*\.xml$", url)]
    assert 1 <= len(maps) <= 30
    source, urls = [detail], set()
    for url in maps:
        name = urlparse(url).path.rsplit("/",1)[-1]
        source.append(acquire("nara_white_house_post_sitemap",url,name,raw))
        for article in xml_urls(raw / name):
            stamp = URL_DATE.search(article)
            if stamp and "-".join(stamp.groups()) <= END and "-approves-" in article and "disaster-declaration" in article:
                urls.add(article)
    date_check = sorted(urls, key=lambda url: URL_DATE.search(url).group())
    if len(date_check) > 400:
        raise ValueError("Archived article cap reached")
    return date_check, source


def html_snapshot(link, raw):
    assert urlparse(link).hostname == HOST
    name = f"notice_{sha256(link.encode()).hexdigest()[:16]}.html"
    file = raw / name
    info = file.with_suffix(".html.json")
    if file.exists() and info.exists():
        detail = json.loads(info.read_text())
        assert detail["download_url"] == link and detail["sha256"] == sha256(file.read_bytes()).hexdigest()
        return detail, file.read_text()
    response = public_get(link)
    assert urlparse(response.url).hostname == HOST
    assert "text/html" in response.headers.get("Content-Type", "") and len(response.content) < 2_000_000
    html = response.content.decode("utf-8")
    detail = {"source_id": "nara_white_house_original_approval", "download_url": link,
              "resolved_url": response.url, "filename": name, "sha256": sha256(response.content).hexdigest(),
              "bytes": len(response.content), "retrieved_utc": datetime.now(timezone.utc).isoformat()}
    file.write_bytes(response.content)
    info.write_text(json.dumps(detail, indent=2)+"\n")
    return detail, html


def extract_timestamp(values):
    times = {pd.Timestamp(value).tz_convert("UTC") for value in values}
    return next(iter(times)) if len(times)==1 else None


def read_article(link, raw):
    detail, html = html_snapshot(link, raw)
    slug = urlparse(link).path.strip("/").split("/")[-1]
    state = [key for key, name in sorted(STATES.items(), key=lambda item: -len(item[1]))
             if re.search(rf"-approves-(?:a-)?(?:major-)?(?:u-s-)?{name}-(?:major-)?disaster-declaration(?:-\d+)?$", slug)]
    tag = extract_timestamp(re.findall(r'<meta property="article:published_time" content="([^"]+)"',html))
    ld = extract_timestamp(re.findall(r'"datePublished"\s*:\s*"([^"]+)"',html))
    body_candidates = re.findall(r"<p(?:\s[^>]*)?>(.*?)</p>",html,re.I|re.S)
    paragraphs = [" ".join(unescape(re.sub(r"<[^>]+>"," ",para)).split()) for para in body_candidates]
    paragraph = next((para for para in paragraphs if APPROVAL.search(para)), "")
    phrase = APPROVAL.search(paragraph)
    period = SOURCE_DATE.search(paragraph[phrase.end():]) if phrase else None
    modified = extract_timestamp(re.findall(r'"dateModified"\s*:\s*"([^"]+)"',html))
    release = tag if tag is not None and tag == ld else None
    local = release.tz_convert(EASTERN) if release is not None else None
    explicit_day = local.date() if local is not None else None
    if explicit_day is not None and phrase and phrase.group(1).lower()=="yesterday":
        explicit_day = local.date() - timedelta(days=1)
    return detail, {"link": link, "state": state[0].upper() if len(state)==1 else "",
        "archived_published_utc": release.isoformat() if release is not None else "",
        "archived_modified_utc": modified.isoformat() if modified is not None else "",
        "inferred_declaration_day": explicit_day.isoformat() if explicit_day is not None and phrase else "",
        "incident_start_month_day_reference": period.group(1) if period else "",
        "incident_start_year_reference": period.group(2) if period and period.group(2) else "",
        "article_clock_status":"agree" if release is not None else "no_both_timestamp_match",
        "original_major_statement_recognized": bool(phrase), "opening_statement": paragraph}


def match_article(notice, cases):
    if not all(notice[name] for name in ("state","inferred_declaration_day","incident_start_month_day_reference")):
        return None, "unresolved_document"
    a = cases[cases.state.eq(notice["state"]) & cases.declaration.eq(notice["inferred_declaration_day"])]
    wanted = pd.Timestamp(f"{notice['incident_start_month_day_reference']} 2000")
    starts = pd.to_datetime(a.start)
    ok = starts.dt.month.eq(wanted.month) & starts.dt.day.eq(wanted.day)
    if notice["incident_start_year_reference"]:
        ok &= starts.dt.year.eq(int(notice["incident_start_year_reference"]))
    matched = a[ok]
    if len(matched)!=1:
        return None, "no_unique_state_date_incident_match"
    return int(matched.iloc[0].disasterNumber), "matched"


def estimate_coverage(cases, notices):
    records = []
    matched = notices[notices.status.eq("matched")]
    for row in cases.itertuples():
        start = pd.Timestamp(row.declaration, tz=EASTERN)
        next_day = start + pd.DateOffset(days=1)
        alternatives = matched[matched.disasterNumber.eq(row.disasterNumber)].sort_values("archived_published_utc")
        initial = alternatives.iloc[0] if len(alternatives) else None
        for clock in ("archived_published_utc","archived_modified_utc"):
            when = pd.Timestamp(initial[clock]) if initial is not None and initial[clock] else None
            min_delay = (when-next_day).total_seconds()/3600 if when is not None else None
            max_delay = (when-start).total_seconds()/3600 if when is not None else None
            for duration in HORIZONS:
                state = "missing" if when is None else "covered" if max_delay<=duration else "late" if min_delay>=duration else "ambiguous"
                records.append({"disasterNumber": row.disasterNumber, "fyDeclared": row.fyDeclared,
                         "clock": clock, "hours": duration, "status": state, "elapsed_lower_hours": min_delay,
                         "elapsed_upper_hours": max_delay})
    return pd.DataFrame(records)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    args.raw.mkdir(parents=True,exist_ok=True)
    args.output.mkdir(parents=True,exist_ok=True)
    cols=["disasterNumber","fyDeclared","state","start","declaration"]
    cases=pd.read_csv("research/results/hazards/cohort_features.csv", usecols=cols)
    ids=set(pd.read_csv("research/results/forward/forward_predictions.csv",usecols=["disasterNumber"]).disasterNumber)
    cases=cases[cases.disasterNumber.isin(ids) & cases.declaration.ge("2021-01-20")]
    assert cases.declaration.max() <= END and cases.fyDeclared.max()==2024
    links, manifest=archive_index(args.raw)
    records=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for detail, notice in pool.map(lambda link: read_article(link,args.raw),links):
            key, status=match_article(notice,cases)
            records.append({**notice,"disasterNumber":key,"status":status})
            manifest.append(detail)
    a=pd.DataFrame(records)
    avail=estimate_coverage(cases,a)
    cuts=avail.groupby(["clock","hours","status"]).size().rename("declarations").reset_index()
    for name,table in [("announcement_identity_audit",a),("case_availability",avail),("availability_summary",cuts),
                       ("source_manifest",pd.DataFrame(manifest))]:
        table.to_csv(args.output/f"{name}.csv",index=False)
    print(f"Matched {a.query('status==\"matched\"').disasterNumber.nunique()} original announcements across {len(cases)} applicable historical cases.")


if __name__=="__main__":
    main()
