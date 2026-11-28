#!/usr/bin/env python3
"""Independent SQL cross-check of ranking counts on emitted metadata only."""
import csv
import json
import sqlite3
import sys
from pathlib import Path

d=Path(sys.argv[1])
con=sqlite3.connect(":memory:")
con.execute("CREATE TABLE sig(y INTEGER, v0 TEXT, v1 TEXT, issued TEXT, cycle INTEGER)")
with (d/"issue-signatures.csv").open(newline="") as f:
    con.executemany("INSERT INTO sig VALUES(?,?,?,?,?)",
        ((int(r["year"]),r["valid_start"],r["valid_end"],r["product_issue"],int(r["cycle"])) for r in csv.DictReader(f)))
query="""
WITH RECURSIVE
step(n) AS (SELECT 0 UNION ALL SELECT n+1 FROM step WHERE n<167),
yr AS (SELECT DISTINCT y FROM sig),
orig AS (SELECT y, strftime('%Y-%m-%dT%H:%MZ',y||'-05-01',n||' hours') AS t FROM yr CROSS JOIN step),
mode(k) AS (VALUES(0),(1),(2)),
eligible AS (
 SELECT o.y, o.t, s.issued, s.cycle, mode.k,
   ROW_NUMBER() OVER(PARTITION BY o.y,o.t,mode.k ORDER BY s.v0 DESC,s.issued DESC,s.v1 DESC,s.cycle DESC) AS rank
 FROM orig o JOIN sig s ON s.y=o.y AND s.v0<=o.t AND o.t<s.v1
 CROSS JOIN mode WHERE ((mode.k=0 OR s.issued<=o.t) AND (mode.k!=2 OR cycle!=-1))
),
pair AS (
 SELECT y,t,
   MAX(CASE WHEN k=0 THEN issued END) AS n,
   MAX(CASE WHEN k=1 THEN issued||'/'||cycle END) AS i,
   MAX(CASE WHEN k=2 THEN issued||'/'||cycle END) AS c
 FROM eligible WHERE rank=1 GROUP BY y,t
)
SELECT y,COUNT(n),COUNT(i),SUM(n>t),SUM(i IS NOT c) FROM pair GROUP BY y ORDER BY y
"""
fields=["year","valid_window_present","nominal_issue_eligible_present",
        "valid_only_picks_future_issuance","canonical_only_differs_from_all_issued"]
expected=[tuple(r[k] for k in fields) for r in json.loads((d/"summary.json").read_bytes())]
actual=list(con.execute(query))
assert expected==actual,(expected,actual)
print("Independent SQLite ranking counts:",actual)
