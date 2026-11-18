#!/usr/bin/env python3
"""Audit documentary assistance fields and named administrative scope at early horizons."""
import argparse
from collections import defaultdict
from hashlib import sha256
from html import unescape
from pathlib import Path
import re
import unicodedata

import pandas as pd

from thesis_forward_models import YEARS, read_table, split
from thesis_whitehouse_announcements import APPROVAL

CONTINUOUS = Path("research/continuous")
FIPS_SOURCE = Path("research/results/hazards/county_keys.csv")
FULL_DDS = CONTINUOUS / "complete_geography/results/declaration_geography_audit.csv.gz"
BEFORE = re.compile(r"\b(?:counties|parishes|municipalities|boroughs|county|parish|municipality) of ([^.;:]+)")
AFTER = re.compile(r"\bin (?:the )?([A-Z][A-Za-zÀ-ž ,’'\-]{1,600}?) (?:Counties|Parishes|Boroughs|Municipalities|County)\b")


def normalize(name):
    ascii_characters = unicodedata.normalize("NFKD", name).encode("ascii",errors="ignore").decode()
    return re.sub("[^a-z0-9]", "", ascii_characters.lower())


def county_reference(county, cohort):
    keyed = county.merge(cohort[["disasterNumber","state"]], on="disasterNumber", validate="many_to_one")
    names = defaultdict(lambda: defaultdict(set))
    for row in keyed.itertuples():
        name = re.sub(r"\s*\([^)]*\)\s*$", "", row.designatedArea).strip()
        names[row.state][normalize(name)].add(row.county_fips)
    return names


def fields(body, names):
    text = " ".join(body.split())
    clauses = re.split(r"(?<=[.!?])\s+",text)
    explicit={"individuals":0,"governments":0,"mitigation":0}
    found,unknown, evidence=set(),set(),[]
    for clause in clauses:
        line=clause.lower()
        grant=("funding" in line or "assistance" in line or "aid" in line)
        available=("available" in line or "authorized" in line or "provides" in line)
        scope=[]
        if grant and available:
            if "affected individuals" in line or "individuals and households" in line:
                scope.append("individuals")
            if "government" in line and ("eligible" in line or "cost-sharing" in line):
                scope.append("governments")
            if "hazard mitigation" in line:
                scope.append("mitigation")
        if not scope:
            continue
        for code in scope:
            explicit[code]+=1
        for span in [*BEFORE.findall(clause),*AFTER.findall(clause)]:
            span=re.split(r"\s+and\s+(?:emergency|all|statewide|for|the entire|local governments)",
                          span,maxsplit=1,flags=re.I)[0]
            tokens=re.split(r",\s*(?:and\s+)?|\s+and\s+",span)
            for value in tokens:
                clean=value.strip(" .")
                name=normalize(re.sub(r"^(?:the )|(?: County| Parish)$","", clean))
                keys=names.get(name,set())
                if len(keys)==1:
                    found |= keys
                elif clean:
                    unknown.add(clean)
        evidence.append(clause)
    return {**{key+"_evidence_clauses":value for key,value in explicit.items()},
            "named_county_keys":"|".join(sorted(found)), "unresolved_place_tokens":"|".join(sorted(unknown)),
            "named_county_n":len(found),"evidence": " | ".join(evidence)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--full-geography-audit", type=Path, default=FULL_DDS)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    coast=read_table(Path("research/results/hazards/cohort_features.csv"))
    original=pd.read_csv(CONTINUOUS/"announcements/results/announcement_identity_audit.csv").query('status=="matched"')
    manifest=pd.read_csv(CONTINUOUS/"announcements/results/source_manifest.csv").set_index("download_url")
    availability=pd.read_csv(CONTINUOUS/"announcements/results/case_availability.csv")
    time=availability.query('clock=="archived_modified_utc" and status=="covered"')
    earlier=set(time.query("hours==24").disasterNumber)
    valid=set(time.query("hours==72").disasterNumber)
    assert len(valid)==91 and earlier<=valid
    source_cohort=coast[coast.disasterNumber.isin(valid)]
    strict=pd.read_csv(FIPS_SOURCE,dtype={"county_fips":str})
    archive=pd.read_csv(args.full_geography_audit,usecols=["disasterNumber","county_fips","designatedArea","state_agrees"],
                        dtype={"county_fips":str})
    archive=archive[archive.disasterNumber.isin(coast.disasterNumber)&archive.state_agrees&archive.county_fips.notna()]
    known=county_reference(archive,coast)
    eventual=archive.groupby("disasterNumber").county_fips.agg(set)
    date_locked=strict.groupby("disasterNumber").county_fips.agg(set)
    observations=[]
    for row in original[original.disasterNumber.isin(valid)].itertuples():
        case_id=int(row.disasterNumber)
        evidence=manifest.loc[row.link]
        raw=(args.raw/evidence.filename).read_bytes()
        assert sha256(raw).hexdigest()==evidence.sha256
        paragraphs=[" ".join(unescape(re.sub(r"<[^>]+>"," ",p)).split()) for p in
                    re.findall(r"<p(?:\s[^>]*)?>(.*?)</p>",raw.decode("utf-8"),re.I|re.S)]
        start=next(i for i,para in enumerate(paragraphs) if APPROVAL.search(para))
        document=" ".join(paragraphs[start:start+16])
        closing=re.split(r"FOR FURTHER INFORMATION",document,maxsplit=1,flags=re.I)[0]
        extracted=fields(closing,known[row.state])
        early_keys=set(filter(None,extracted["named_county_keys"].split("|")))
        late_keys=eventual.get(case_id,set())
        strict_keys=date_locked.get(case_id,set())
        observations.append({"disasterNumber":case_id,"state":row.state,"document":row.link,"source_sha256":evidence.sha256,
            "within_24h_archived_modification":case_id in earlier,"within_72h_archived_modification":True,
            "eventual_named_county_n":len(late_keys),"early_and_eventual_n":len(early_keys & late_keys),
            "prior_scope_county_n":len(strict_keys),"early_and_prior_scope_n":len(early_keys & strict_keys),
            "early_archived_matches_absent_in_prior_scope_n":len((early_keys & late_keys)-strict_keys),
            "eventual_not_extracted_in_early_text_n":len(late_keys - early_keys),
            "extracted_early_outside_eventual_n":len(early_keys-late_keys),**extracted})
    extracted=pd.DataFrame(observations)
    assert extracted.disasterNumber.is_unique and len(extracted)==len(source_cohort)==91
    ledger=pd.read_csv(CONTINUOUS/"horizons/results/historical_horizons.csv",usecols=["disasterNumber",
         "release_review_date","snapshot_status"])
    reviewed=pd.to_datetime(ledger.release_review_date,utc=True)
    support=[]
    for year in YEARS:
        prior, held=split(coast,year)
        cutoff=min(held.start.min(),held.declaration.min()).tz_localize("UTC")
        eligible_maturity=set(ledger[reviewed.lt(cutoff)&ledger.snapshot_status.eq("published_actions")].disasterNumber)
        for horizon, scope in [(24,earlier),(72,valid)]:
            support.append({"held_fy":year,"horizon":horizon, "training_rows_with_document":prior.disasterNumber.isin(scope).sum(),
                            "held_rows_with_document":held.disasterNumber.isin(scope).sum(),
                            "training_rows_with_field_and_elapsed_label":prior.disasterNumber.isin(scope&eligible_maturity).sum()})
    x=pd.DataFrame(support)
    extracted.sort_values("disasterNumber").to_csv(args.output/"field_evidence.csv",index=False)
    x.to_csv(args.output/"forward_joint_support.csv",index=False)
    print(f"Saved documentary fields for {len(extracted)} records across archived early versions.")


if __name__=="__main__":
    main()
