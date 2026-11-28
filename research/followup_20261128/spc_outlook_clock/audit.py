#!/usr/bin/env python3
"""Bounded Day 1 outlook product-clock feasibility; never reads impact or funding."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from struct import unpack_from
from zipfile import ZipFile

YEARS=(2012,2018,2024)
DAY=timedelta(days=1)
HOUR=timedelta(hours=1)
MAX_BYTES=12_000_000
SCHEMA=("ISSUE","EXPIRE","PRODISS","TYPE","DAY","THRESHOLD","CATEGORY","CYCLE")

def utc(s):
    return datetime.strptime(s,"%Y%m%d%H%M").replace(tzinfo=timezone.utc)

def iso(d):
    return d.strftime("%Y-%m-%dT%H:%MZ")

def fields(zpath):
    assert zpath.stat().st_size < MAX_BYTES
    with ZipFile(zpath) as z:
        dbf=[p for p in z.infolist() if p.filename.endswith(".dbf")]
        assert len(dbf)==1 and dbf[0].file_size<MAX_BYTES
        data=z.read(dbf[0])
    n,h,r=unpack_from("<IHH",data,4)
    assert 32 < h < len(data) and r > 1
    assert len(data) in (h+n*r,h+n*r+1)
    assert data[h-1]==13
    f=[]
    for j in range(32,h-1,32):
        f.append((data[j:j+11].rstrip(b"\0").decode("ascii"),data[j+16]))
    assert tuple(k for k,_ in f)==SCHEMA and r==1+sum(i for _,i in f)
    for j in range(n):
        off=h+j*r
        assert data[off]==32
        values={}
        for k,sz in f:
            off+=1
            values[k]=data[off:off+sz].decode("ascii").strip()
            off+=sz-1
        yield values

def summary(zpath,year):
    origin=datetime(year,5,1,tzinfo=timezone.utc)
    end=origin+7*DAY
    periods=defaultdict(set)
    rows=Counter()
    category=defaultdict(set)
    for row in fields(zpath):
        s,e,i=(utc(row[k]) for k in ("ISSUE","EXPIRE","PRODISS"))
        c=int(float(row["CYCLE"]))
        assert row["TYPE"]=="C" and float(row["DAY"])==1 and s<e and -1<=c<=23
        if not origin<=i<end:
            rows["out_of_window"]+=1
            continue
        k=(s,e,i,c)
        rows[k]+=1
        category[k].add((row["CATEGORY"],row["THRESHOLD"]))
        periods[s,e].add((i,c))
    signatures=sorted(k for k in rows if isinstance(k,tuple))
    assert signatures
    lag=[int((i-s).total_seconds()/60) for s,e,i,c in signatures]
    # The tuple key is fixed: start, then issue, then end and cycle to break ties.
    choose=lambda x:max(x,key=lambda t:(t[0],t[2],t[1],t[3]),default=None)
    hourly=Counter()
    mismatches=[]
    for j in range(168):
        cut=origin+j*HOUR
        valid=[k for k in signatures if k[0]<=cut<k[1]]
        unaware=choose(valid)
        available=choose([k for k in valid if k[2]<=cut])
        canonical_only=choose([k for k in valid if k[2]<=cut and k[3]!=-1])
        hourly["valid_window_present"]+=int(unaware is not None)
        hourly["nominal_issue_eligible_present"]+=int(available is not None)
        if unaware and unaware[2]>cut:
            hourly["valid_only_picks_future_issuance"]+=1
            mismatches.append([year,iso(cut),iso(unaware[2]),iso(available[2]) if available else "",unaware[3]])
        if available!=canonical_only:
            hourly["canonical_only_differs_from_all_issued"]+=1
    rec=dict(year=year,polygon_rows_in_issue_window=sum(rows[k] for k in signatures),
             polygon_rows_removed_outside_issue_window=rows["out_of_window"],
             distinct_clock_signatures=len(signatures),distinct_valid_windows=len(periods),
             repeated_valid_windows=sum(len(p)>1 for p in periods.values()),
             noncanonical_signatures=sum(k[3]==-1 for k in signatures),
             issued_before_valid=sum(t<0 for t in lag),issued_at_valid=sum(t==0 for t in lag),
             issued_after_valid=sum(t>0 for t in lag),min_issue_minus_valid_min=min(lag),
             max_issue_minus_valid_min=max(lag),hourly_grid=168,
             **{k:hourly[k] for k in ("valid_window_present","nominal_issue_eligible_present",
             "valid_only_picks_future_issuance","canonical_only_differs_from_all_issued")})
    ledger=[[year,iso(s),iso(e),iso(i),c,rows[s,e,i,c],len(category[s,e,i,c])] for s,e,i,c in signatures]
    return rec,ledger,mismatches

def main():
    p=argparse.ArgumentParser()
    p.add_argument("downloads",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    summary_rows,ledger,mismatches=[],[],[]
    raw=[]
    for y in YEARS:
        z=a.downloads/f"{y}.zip"
        r,l,m=summary(z,y)
        summary_rows.append(r); ledger.extend(l); mismatches.extend(m)
        raw.append(dict(year=y,zip_sha256=hashlib.sha256(z.read_bytes()).hexdigest(),bytes=z.stat().st_size))
    a.out.mkdir(parents=True,exist_ok=False)
    for file,header,data in (
      ("issue-signatures.csv",("year","valid_start","valid_end","product_issue","cycle","polygon_rows","category_threshold_pairs"),ledger),
      ("future-issue-cutoffs.csv",("year","decision_cutoff","valid_only_product_issue","issue_eligible_product_issue","valid_only_cycle"),mismatches),
    ):
        with (a.out/file).open("w",newline="") as f:
            w=csv.writer(f,lineterminator="\n"); w.writerow(header); w.writerows(data)
    (a.out/"summary.json").write_text(json.dumps(summary_rows,indent=2)+"\n")
    (a.out/"archive-digests.json").write_text(json.dumps(raw,indent=2)+"\n")
    print(json.dumps(summary_rows,indent=2))
if __name__=="__main__":
    main()
