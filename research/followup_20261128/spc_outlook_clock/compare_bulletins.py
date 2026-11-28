#!/usr/bin/env python3
"""Cross-representation comparison of archived Day 1 forecasts, no observed outcomes."""
import csv
import hashlib
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from audit import YEARS, iso

base,out=map(Path,sys.argv[1:3])
data=[]
metadata=[]
def stamp(y,ddhhmm):
    return datetime.strptime(f"{y}05"+ddhhmm,"%Y%m%d%H%M").replace(tzinfo=timezone.utc)
with (base/"results/issue-signatures.csv").open(newline="") as f:
    for r in csv.DictReader(f):
        if r["product_issue"][5:10]!="05-01":
            continue
        data.append(r)
fields=[*data[0],"bulletin_spc_ac","bulletin_wmo","match_class","wmo_minus_dbf_minutes"]
for y in YEARS:
    raw=(base/f"{y}.txt").read_bytes()
    assert len(raw)<1_000_000
    s=raw.decode("ascii")
    # Pair header + authored issue and validity within the same ACUS01 bulletin.
    starts=list(re.finditer(r"(?m)^ACUS01 KWNS (\d{6})(?: [A-Z]{3})?\s*$",s))
    assert 0<len(starts)<25
    bullets=[]
    for j,m in enumerate(starts):
        block=s[m.end():starts[j+1].start() if j+1<len(starts) else len(s)]
        i=re.search(r"(?m)^SPC AC (\d{6})\s*$",block)
        v=re.search(r"(?im)^VALID (\d{6})Z\s*-\s*(\d{6})Z",block)
        assert i and v and re.search(r"(?im)^SWODY1\s*$",block[:180])
        w=stamp(y,m[1]); a=stamp(y,i[1])
        b=stamp(y,v[1]); e=stamp(y,v[2])
        assert w.day==a.day==1 and 0<=(w-a).total_seconds()<=600 and b<e
        bullets.append((b,e,a,w))
    assert len(bullets)==len(set(bullets))
    metadata.append({"year":y,"text_sha256":hashlib.sha256(raw).hexdigest(),"text_bulletins":len(bullets)})
    for r in data:
        if int(r["year"])!=y:
            continue
        s,e,i=(datetime.fromisoformat(r[k].replace("Z","+00:00")) for k in ("valid_start","valid_end","product_issue"))
        exact=[(a,w) for b,f,a,w in bullets if (b,f,a)==(s,e,i)]
        near=[(a,w) for b,f,a,w in bullets if (b,f)==(s,e) and abs((i-a).total_seconds())==60]
        assert len(exact)<=1
        found=exact or near
        klass="exact" if exact else "single_same_window_one_minute_offset_not_exact" if len(near)==1 else "unpaired"
        if len(found)==1:
            a,w=found[0]
            r.update(bulletin_spc_ac=iso(a),bulletin_wmo=iso(w),match_class=klass,wmo_minus_dbf_minutes=int((w-i).total_seconds()/60))
        else:
            r.update(match_class=klass)
with (out/"first_day_text_clock_crosscheck.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fields,lineterminator="\n");w.writeheader();w.writerows(data)
import json
digests={"input_text_samples":metadata,"match_counts":dict(Counter(r["match_class"] for r in data))}
(out/"text_clock_summary.json").write_text(json.dumps(digests,sort_keys=True,indent=2)+"\n")
print(digests)
