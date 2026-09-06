#!/usr/bin/env python3
"""
Target-blind audit linking FEMA hurricane declarations in master_openfema_40plus.xlsx
to official NHC HURDAT2 storm identities.

Important:
- totalObligatedFunding is NEVER used in candidate generation or matching.
- Funding is attached only after a match decision, for downstream audit reporting.
- HURDAT2 is retrospective best-track data. Fields ending in _final_besttrack or
  _incident_besttrack are diagnostic/retrospective and must not be treated as
  declaration-time operational features.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import sys
import time
import urllib.parse
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

MASTER = Path("master_openfema_40plus.xlsx")
OUT = Path("nhc_audit")
OUT.mkdir(exist_ok=True)

USER_AGENT = "Mozilla/5.0 FEMA-NHC-thesis-audit/1.0"
FEMA_API = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"
NHC_HURDAT_DIR = "https://www.nhc.noaa.gov/data/hurdat/"

DATE_COLS = {"declarationDate", "incidentBeginDate", "incidentEndDate", "lastIAFilingDate", "minDateObligated", "maxDateObligated"}


def http_get(url: str, timeout: int = 45, retries: int = 4) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:
            last = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} attempts: {url}: {last}")


def col_index(ref: str) -> int:
    letters = re.match(r"[A-Z]+", ref.upper())
    if not letters:
        raise ValueError(ref)
    out = 0
    for ch in letters.group(0):
        out = out * 26 + (ord(ch) - 64)
    return out - 1


def excel_date(value):
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        # Excel 1900 date system; openpyxl-compatible epoch handling.
        return datetime(1899, 12, 30) + timedelta(days=float(value))
    s = str(value).strip()
    if not s:
        return None
    # Common ISO forms.
    s2 = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s2)
        return dt.replace(tzinfo=None)
    except Exception:
        pass
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def read_first_sheet_xlsx(path: Path):
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    pkg_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall(f"{{{ns}}}si"):
                shared.append("".join(n.text or "" for n in si.iter(f"{{{ns}}}t")))

        wb = ET.fromstring(z.read("xl/workbook.xml"))
        sheet = wb.find(f".//{{{ns}}}sheet")
        rid = sheet.attrib[f"{{{rel_ns}}}id"]
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        target = None
        for rel in rels.findall(f"{{{pkg_ns}}}Relationship"):
            if rel.attrib.get("Id") == rid:
                target = rel.attrib["Target"]
                break
        if target is None:
            raise RuntimeError("Could not resolve first worksheet")
        if target.startswith("/"):
            sheet_path = target.lstrip("/")
        else:
            sheet_path = "xl/" + target.lstrip("./")
        root = ET.fromstring(z.read(sheet_path))

        rows = []
        max_col = 0
        for row in root.findall(f".//{{{ns}}}sheetData/{{{ns}}}row"):
            vals = {}
            for c in row.findall(f"{{{ns}}}c"):
                ref = c.attrib.get("r", "")
                idx = col_index(ref)
                max_col = max(max_col, idx)
                typ = c.attrib.get("t")
                v = c.find(f"{{{ns}}}v")
                if typ == "inlineStr":
                    val = "".join(n.text or "" for n in c.iter(f"{{{ns}}}t"))
                elif v is None:
                    val = None
                else:
                    raw = v.text or ""
                    if typ == "s":
                        val = shared[int(raw)] if raw else ""
                    elif typ == "b":
                        val = raw == "1"
                    elif typ in ("str", "e"):
                        val = raw
                    else:
                        try:
                            num = float(raw)
                            val = int(num) if num.is_integer() else num
                        except ValueError:
                            val = raw
                vals[idx] = val
            rows.append(vals)

    if not rows:
        return []
    headers = [rows[0].get(i) for i in range(max_col + 1)]
    records = []
    for row in rows[1:]:
        rec = {str(headers[i]): row.get(i) for i in range(len(headers)) if headers[i] not in (None, "")}
        for c in DATE_COLS:
            if c in rec:
                rec[c] = excel_date(rec[c])
        records.append(rec)
    return records


def to_int(v):
    try:
        return int(float(v))
    except Exception:
        return None


def norm_text(s):
    return re.sub(r"[^A-Z0-9]+", " ", str(s or "").upper()).strip()


def extract_title_name(title: str):
    t = norm_text(title)
    # Drop common declaration descriptors and keep the substantive storm token(s).
    patterns = [
        r"^HURRICANE\s+",
        r"^TROPICAL STORM\s+",
        r"^TROPICAL CYCLONE\s+",
    ]
    for p in patterns:
        t2 = re.sub(p, "", t)
        if t2 != t:
            t = t2
            break
    # Remove generic suffixes.
    t = re.sub(r"\bAND ASSOCIATED (FLOODING|STORMS|DAMAGE)\b.*$", "", t).strip()
    # A normal named storm is typically one token, occasionally hyphenated in source;
    # HURDAT names are normalized to alphanumerics, so the first non-generic token is enough.
    generic = {"HURRICANE", "STORM", "FLOODING", "SEVERE", "WINDS", "RAIN", "RAINFALL", "TROPICAL"}
    toks = [x for x in t.split() if x not in generic]
    return toks[0] if toks else ""


def fema_lookup(disaster_number: int):
    query = urllib.parse.urlencode({"$filter": f"disasterNumber eq {disaster_number}", "$top": "1"})
    url = FEMA_API + "?" + query
    try:
        data = json.loads(http_get(url).decode("utf-8"))
        records = None
        for v in data.values():
            if isinstance(v, list):
                records = v
                break
        if not records:
            return disaster_number, {}, url, "no_record"
        r = records[0]
        return disaster_number, {
            "declarationTitle": r.get("declarationTitle"),
            "state": r.get("state"),
            "declarationDate": r.get("declarationDate"),
            "incidentBeginDate": r.get("incidentBeginDate"),
            "incidentEndDate": r.get("incidentEndDate"),
            "incidentType": r.get("incidentType"),
        }, url, "ok"
    except Exception as exc:
        return disaster_number, {}, url, f"error:{exc}"


def discover_hurdat_urls():
    candidates = []
    try:
        html = http_get(NHC_HURDAT_DIR).decode("utf-8", errors="replace")
        hrefs = re.findall(r'href=["\']([^"\']*hurdat2[^"\']*\.txt)["\']', html, flags=re.I)
        candidates = [urllib.parse.urljoin(NHC_HURDAT_DIR, h) for h in hrefs]
    except Exception:
        candidates = []

    # Known 2024 fallbacks are sufficient for a dataset ending in FY2024.
    fallbacks = [
        "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2024-040425.txt",
        "https://www.nhc.noaa.gov/data/hurdat/hurdat2-nepac-1949-2024-031725.txt",
    ]
    for u in fallbacks:
        if u not in candidates:
            candidates.append(u)

    atl = [u for u in candidates if "nepac" not in u.lower() and re.search(r"hurdat2-1851-", u)]
    pac = [u for u in candidates if "nepac" in u.lower()]
    # Prefer the lexicographically latest discovered filename.
    chosen = []
    if atl:
        chosen.append(sorted(atl)[-1])
    if pac:
        chosen.append(sorted(pac)[-1])
    if len(chosen) < 2:
        raise RuntimeError(f"Could not identify both HURDAT2 files from {candidates}")
    return chosen


def parse_hurdat(text: str, source_url: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    storms = []
    i = 0
    while i < len(lines):
        parts = [p.strip() for p in lines[i].split(",")]
        if len(parts) >= 3 and re.match(r"^(AL|EP|CP)\d{6}$", parts[0]):
            sid, name, n = parts[0], parts[1], int(parts[2])
            obs = []
            for ln in lines[i + 1:i + 1 + n]:
                p = [x.strip() for x in ln.split(",")]
                if len(p) < 8:
                    continue
                try:
                    dt = datetime.strptime(p[0] + p[1].zfill(4), "%Y%m%d%H%M")
                except Exception:
                    continue
                def num(x):
                    try:
                        v = float(x)
                        return None if v <= -900 else v
                    except Exception:
                        return None
                radii = [num(x) for x in p[8:20]]
                obs.append({
                    "dt": dt,
                    "record_id": p[2],
                    "status": p[3],
                    "lat": p[4],
                    "lon": p[5],
                    "wind": num(p[6]),
                    "pressure": num(p[7]),
                    "radii": radii,
                })
            if obs:
                storms.append({
                    "id": sid,
                    "name": norm_text(name).replace(" ", ""),
                    "display_name": name.strip(),
                    "basin": sid[:2],
                    "year": int(sid[4:8]),
                    "start": obs[0]["dt"],
                    "end": obs[-1]["dt"],
                    "obs": obs,
                    "source_url": source_url,
                })
            i += n + 1
        else:
            i += 1
    return storms


def overlap(a0, a1, b0, b1, pad_days=2):
    if not all([a0, a1, b0, b1]):
        return False
    return b1 >= a0 - timedelta(days=pad_days) and b0 <= a1 + timedelta(days=pad_days)


def storm_summary(storm, incident_begin, incident_end):
    obs = storm["obs"]
    winds = [o["wind"] for o in obs if o["wind"] is not None]
    pressures = [o["pressure"] for o in obs if o["pressure"] is not None]
    all_r = [r for o in obs for r in o["radii"] if r is not None]
    subset = []
    if incident_begin and incident_end:
        subset = [o for o in obs if incident_begin - timedelta(days=1) <= o["dt"] <= incident_end + timedelta(days=1)]
    sw = [o["wind"] for o in subset if o["wind"] is not None]
    sp = [o["pressure"] for o in subset if o["pressure"] is not None]
    sr = [r for o in subset for r in o["radii"] if r is not None]
    return {
        "nhc_storm_start": storm["start"].isoformat(),
        "nhc_storm_end": storm["end"].isoformat(),
        "nhc_peak_wind_kt_final_besttrack": max(winds) if winds else None,
        "nhc_min_pressure_mb_final_besttrack": min(pressures) if pressures else None,
        "nhc_max_wind_radius_nm_final_besttrack": max(all_r) if all_r else None,
        "nhc_max_wind_kt_incident_besttrack": max(sw) if sw else None,
        "nhc_min_pressure_mb_incident_besttrack": min(sp) if sp else None,
        "nhc_max_wind_radius_nm_incident_besttrack": max(sr) if sr else None,
        "nhc_incident_obs_count": len(subset),
    }


def funding_band(v):
    try:
        x = float(v)
    except Exception:
        return "missing"
    if x <= 0:
        return "zero"
    if x < 50_000_000:
        return "<50M"
    if x < 200_000_000:
        return "50M-200M"
    if x < 500_000_000:
        return "200M-500M"
    return "500M+"


def fmt_date(dt):
    return dt.date().isoformat() if isinstance(dt, datetime) else ""


def main():
    master = read_first_sheet_xlsx(MASTER)
    if len(master) != 971:
        print(f"WARNING: expected 971 master rows, found {len(master)}", file=sys.stderr)

    hurricanes = []
    for r in master:
        if norm_text(r.get("incidentType")) == "HURRICANE":
            rr = dict(r)
            rr["_dn"] = to_int(r.get("disasterNumber"))
            hurricanes.append(rr)

    dns = sorted({r["_dn"] for r in hurricanes if r["_dn"] is not None})
    print(f"Master rows: {len(master)}")
    print(f"Hurricane rows: {len(hurricanes)}")
    print(f"Unique hurricane disaster numbers: {len(dns)}")

    # FEMA title/date lookup, keyed only by disaster number.
    fema = {}
    statuses = Counter()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(fema_lookup, dn) for dn in dns]
        for fut in as_completed(futs):
            dn, rec, url, status = fut.result()
            fema[dn] = {"record": rec, "url": url, "status": status}
            statuses[status.split(":")[0]] += 1
    print("FEMA lookup status:", dict(statuses))

    hurdat_urls = discover_hurdat_urls()
    print("HURDAT sources:")
    for u in hurdat_urls:
        print("  ", u)
    storms = []
    for u in hurdat_urls:
        txt = http_get(u, timeout=90).decode("utf-8", errors="replace")
        storms.extend(parse_hurdat(txt, u))
    storms = [s for s in storms if 2008 <= s["year"] <= 2025]
    print(f"HURDAT storms retained: {len(storms)}")

    audit_rows = []
    for r in hurricanes:
        dn = r["_dn"]
        api = fema.get(dn, {}).get("record", {})
        title = api.get("declarationTitle") or ""
        state = r.get("state") or api.get("state") or ""
        begin = r.get("incidentBeginDate") or excel_date(api.get("incidentBeginDate"))
        end = r.get("incidentEndDate") or excel_date(api.get("incidentEndDate")) or begin
        decl = r.get("declarationDate") or excel_date(api.get("declarationDate"))
        if begin and not end:
            end = begin
        name_token = extract_title_name(title)
        normalized_token = norm_text(name_token).replace(" ", "")

        # Matching uses title/date/state only. It never receives funding.
        year_candidates = set()
        for dt in (begin, end, decl):
            if isinstance(dt, datetime):
                year_candidates.update({dt.year - 1, dt.year, dt.year + 1})
        pool = [s for s in storms if not year_candidates or s["year"] in year_candidates]
        name_matches = [s for s in pool if normalized_token and s["name"] == normalized_token]
        date_matches = [s for s in pool if begin and end and overlap(begin, end, s["start"], s["end"], 2)]
        both = [s for s in name_matches if s in date_matches]

        chosen = None
        confidence = "D"
        method = "unmatched"
        candidate_count = 0
        if len(both) == 1:
            chosen = both[0]
            confidence = "A"
            method = "declaration_title_name + incident_date_overlap"
            candidate_count = 1
        elif len(name_matches) == 1:
            chosen = name_matches[0]
            confidence = "B"
            method = "unique_declaration_title_name"
            candidate_count = 1
        elif len(date_matches) == 1:
            chosen = date_matches[0]
            confidence = "B"
            method = "unique_incident_date_overlap"
            candidate_count = 1
        else:
            # Do not force an ambiguous match.
            candidate_count = max(len(both), len(name_matches), len(date_matches))
            if candidate_count > 1:
                confidence = "C"
                method = "ambiguous_candidates"

        out = {
            "disasterNumber": dn,
            "state": state,
            "fyDeclared": r.get("fyDeclared"),
            "declarationTitle": title,
            "incidentBeginDate": fmt_date(begin),
            "incidentEndDate": fmt_date(end),
            "declarationDate": fmt_date(decl),
            "title_name_token": name_token,
            "match_confidence": confidence,
            "match_method": method,
            "candidate_count": candidate_count,
            "nhc_storm_id": chosen["id"] if chosen else "",
            "nhc_storm_name": chosen["display_name"] if chosen else "",
            "nhc_basin": chosen["basin"] if chosen else "",
            "nhc_source_url": chosen["source_url"] if chosen else "",
        }
        if chosen:
            out.update(storm_summary(chosen, begin, end))

        # Target is attached only AFTER the match decision above.
        out["audit_totalObligatedFunding"] = r.get("totalObligatedFunding")
        out["audit_funding_band"] = funding_band(r.get("totalObligatedFunding"))
        audit_rows.append(out)

    # Stable field order.
    base_fields = [
        "disasterNumber","state","fyDeclared","declarationTitle","incidentBeginDate","incidentEndDate","declarationDate",
        "title_name_token","match_confidence","match_method","candidate_count","nhc_storm_id","nhc_storm_name","nhc_basin",
        "nhc_storm_start","nhc_storm_end","nhc_peak_wind_kt_final_besttrack","nhc_min_pressure_mb_final_besttrack",
        "nhc_max_wind_radius_nm_final_besttrack","nhc_max_wind_kt_incident_besttrack","nhc_min_pressure_mb_incident_besttrack",
        "nhc_max_wind_radius_nm_incident_besttrack","nhc_incident_obs_count","nhc_source_url",
        "audit_totalObligatedFunding","audit_funding_band"
    ]
    with (OUT / "nhc_hurricane_match_audit.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=base_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(audit_rows)

    conf = Counter(r["match_confidence"] for r in audit_rows)
    high = [r for r in audit_rows if r["audit_funding_band"] in {"50M-200M","200M-500M","500M+"}]
    high_conf = Counter(r["match_confidence"] for r in high)
    matched = [r for r in audit_rows if r["match_confidence"] in {"A","B"}]

    key_dns = {4339, 4671}
    key = [r for r in audit_rows if r["disasterNumber"] in key_dns]
    with (OUT / "nhc_key_cases.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=base_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(key)

    unresolved = [r for r in audit_rows if r["match_confidence"] in {"C","D"}]
    with (OUT / "nhc_unresolved.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=base_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(unresolved)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "master_rows": len(master),
        "hurricane_rows": len(hurricanes),
        "hurricane_unique_disaster_numbers": len(dns),
        "confidence_counts": dict(conf),
        "matched_A_or_B": len(matched),
        "matched_A_or_B_pct": round(100 * len(matched) / len(audit_rows), 2) if audit_rows else 0,
        "high_band_hurricane_rows": len(high),
        "high_band_confidence_counts": dict(high_conf),
        "HURDAT_sources": hurdat_urls,
        "methodology": {
            "target_used_in_matching": False,
            "funding_attached_after_match": True,
            "HURDAT2_is_retrospective_best_track": True,
            "A": "unique declaration-title storm name plus incident-date overlap",
            "B": "unique title-name OR unique incident-date candidate",
            "C": "multiple plausible candidates; not auto-merged",
            "D": "no match"
        },
        "key_cases": key,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n=== MATCH AUDIT SUMMARY ===")
    print(json.dumps({k:v for k,v in summary.items() if k != "key_cases"}, indent=2))
    print("\n=== KEY CASES ===")
    for r in key:
        print(json.dumps(r, default=str))

    # Fail only on structural issues, not on imperfect coverage.
    if len(hurricanes) < 50:
        raise SystemExit("Unexpectedly few hurricane rows; audit likely read the workbook incorrectly.")


if __name__ == "__main__":
    main()
