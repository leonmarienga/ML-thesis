#!/usr/bin/env python3
"""Reassemble the geographic input from all byte-verified FEMA JSON pages."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import pandas as pd

from thesis_hazard_features import fema_geography, read_master

PAGE = re.compile(r"fema_(\d{4})_(\d+)\.json")
MANIFEST = Path("research/inputs/RAW_SOURCE_MANIFEST.csv")


def verify(raw):
    manifest = pd.read_csv(MANIFEST).query('source_id == "openfema" and status == "retrieved"').copy()
    counts = {}
    for row in manifest.itertuples(index=False):
        m = PAGE.fullmatch(row.filename)
        assert m
        year, offset = map(int, m.groups())
        assert 2010<=year<=2024 and offset%1000==0
        content = (raw/row.filename).read_bytes()
        assert sha256(content).hexdigest()==row.sha256
        source = json.loads(content)
        meta=source["metadata"]
        count = len(source["DisasterDeclarationsSummaries"])
        counts.setdefault(year,[]).append((offset,count,int(meta["count"])))
        assert int(meta["skip"])==offset
        assert all(int(document["fyDeclared"])==year for document in source["DisasterDeclarationsSummaries"])
    assert len(manifest)==36 and len(counts)==15
    support=[]
    for year,pages in sorted(counts.items()):
        pages.sort()
        total=sum(row[1] for row in pages)
        assert sorted(row[0] for row in pages)==list(range(0,total,1000))
        assert {row[2] for row in pages}=={total}
        support.append({"fyDeclared":year,"verified_pages":len(pages),"archived_rows":total,
                        "first_page_rows":pages[0][1],"beyond_first_page":total-pages[0][1]})
    return manifest,pd.DataFrame(support)


def build(raw, output, artifacts):
    manifest,support=verify(raw)
    output.mkdir(parents=True,exist_ok=True)
    support.to_csv(output/"pagination_audit.csv",index=False)
    with tempfile.TemporaryDirectory(dir=output.parent,prefix="full-pages-") as root:
        compatibility=Path(root)
        # Preserve the original assembly routine and map exact page offsets to one-digit aliases.
        for f in raw.iterdir():
            match=PAGE.fullmatch(f.name)
            if not match:
                if artifacts and f.is_file():
                    (compatibility/f.name).symlink_to(f.resolve())
                continue
            year,offset=map(int,match.groups())
            assert offset//1000<=9
            alias=f"fema_{year}_{offset//1000}.json"
            (compatibility/alias).symlink_to(f.resolve())
        d=read_master(Path("master_openfema_40plus.xlsx"))
        full_audit, county, _=fema_geography(compatibility,d)
        county=county[["disasterNumber","county_fips","designatedArea","id"]]
        subset = county[county.disasterNumber.isin(d.loc[~d.incidentType.eq("Biological"),"disasterNumber"])]
        previous=pd.read_csv("research/results/hazards/county_keys.csv")
        cases=d.loc[~d.incidentType.eq("Biological"),["disasterNumber","fyDeclared","state","incidentType"]].copy()
        new_numbers=subset.groupby("disasterNumber").county_fips.nunique()
        old_numbers=previous.groupby("disasterNumber").county_fips.nunique()
        cases["prior_counties"]=cases.disasterNumber.map(old_numbers).fillna(0).astype(int)
        cases["recovered_counties"]=cases.disasterNumber.map(new_numbers).fillna(0).astype(int)
        assert cases.recovered_counties.ge(cases.prior_counties).all()
        old_ids=set(zip(previous.disasterNumber,previous.county_fips))
        new_ids=set(zip(county.disasterNumber,county.county_fips))
        assert old_ids<=new_ids
        cases.to_csv(output/"county_coverage_by_declaration.csv",index=False)
        county.to_csv(output/"county_keys.csv",index=False)
        useful=full_audit[["disasterNumber","state","county_fips","designatedArea","id","dates_agree","state_agrees"]]
        useful.to_csv(output/"declaration_geography_audit.csv.gz",index=False,compression={"method":"gzip","mtime":0})
        if artifacts:
            executable=Path(__file__).resolve().with_name("thesis_hazard_features.py")
            subprocess.run([sys.executable,str(executable),"--raw",str(compatibility),
                            "--artifacts",str(artifacts),"--output",str(output/"hazard_replay")],check=True)
    print(f"Verified {len(manifest)} pages and preserved every original date-locked county key.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--artifacts",type=Path,help="Enable the original full hazard assembly replay")
    args=p.parse_args()
    build(args.raw,args.output,args.artifacts)


if __name__=="__main__":
    main()
