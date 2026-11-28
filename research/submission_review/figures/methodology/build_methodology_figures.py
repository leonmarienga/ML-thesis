#!/usr/bin/env python3
"""Build the archived six-band lineage and the audited forward split figure."""
import argparse
import csv
from datetime import date
from hashlib import sha256
from html import escape
import io
from pathlib import Path

AUDIT_SHA="b76a3cb36dc0844adaa5f7958b3c94e92ba15fea661815dc0b38a252f006460e"
INK="#182a3d"
MUTED="#4a5d70"


class Svg:
    def __init__(self, h, title, desc):
        self.body=[f'<svg xmlns="http://www.w3.org/2000/svg" width="940" height="{h}" viewBox="0 0 940 {h}" role="img" aria-labelledby="title desc">',
                   f'<title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc>',
                   '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" orient="auto" markerWidth="7" markerHeight="7"><path d="M0 0L10 5L0 10Z" fill="#52687e"/></marker></defs>',
                   f'<rect width="940" height="{h}" fill="#fff"/>']
    def text(self,x,y,v,s=16,weight=400,fill=INK,anchor="start"):
        self.body.append(f'<text x="{x}" y="{y}" font-size="{s}" font-family="DejaVu Sans, sans-serif" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(str(v))}</text>')
    def box(self,x,y,w,h,fill="#f0f4f8",stroke="#bac8d5",radius=8):
        self.body.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>')
    def line(self,x,y,X,Y,arrow=False):
        extra=' marker-end="url(#arrow)"' if arrow else ''
        self.body.append(f'<path d="M{x} {y}L{X} {Y}" stroke="#52687e" stroke-width="1.7" fill="none"{extra}/>')
    def write(self,p): p.write_text("\n".join(self.body+["</svg>",""]),encoding="utf-8")


def f2():
    a=Svg(610,"Frozen six-band routing and amount-stack lineage",
          "Archived outputs pass from a shared historical six-band router to band-specific dollar estimators. Accepted low-band overlays and the research-only mid-band overlay do not change routing.")
    a.text(26,34,"Track A: one frozen router, three amount-stack versions",24,700)
    a.text(26,61,"912 FY2010–24 non-Biological declarations. Historical reconstruction, not a prospective test.",15,400,MUTED)
    for x,w,t,s in [(26,224,"Historical inputs","Declaration evidence"),
                    (306,288,"Six-band router","Archived predicted band"),
                    (650,264,"Dollar stack","Estimator for predicted band")]:
        a.box(x,79,w,63);a.text(x+12,103,t,18,700);a.text(x+12,125,s,15)
    a.line(253,111,296,111,True);a.line(597,111,640,111,True)
    basis=[
        ("$0–<$100K","Outer-training median","Accepted: two-stage zero / positive"),
        ("$100K–<$1M","Outer-training median","Accepted: dedicated regressor"),
        ("$1M–<$50M","900-tree log-dollar ExtraTrees","Unchanged"),
        ("$50M–<$200M","Outer-training median","Research only: hybrid / shrinkage"),
        ("$200M–<$500M","Two sub-band specialists","Unchanged; donor caveat"),
        ("$500M+","Analogue selection via inner LFYO","Unchanged"),
    ]
    a.text(32,177,"PREDICTED BAND",14,700)
    a.text(243,177,"ORIGINAL AMOUNT ESTIMATOR",14,700)
    a.text(613,177,"LATER AMOUNT OVERLAY",14,700)
    for i,(label,base,more) in enumerate(basis):
        y=190+49*i
        a.box(26,y,888,42,("#edf3f9" if i in (0,1) else "#fbf0e7" if i==3 else "#f6f8fa"),"#d6dfe8",5)
        for x,t in [(32,label),(243,base),(613,more)]:
            a.text(x,y+26,t,15,700 if x==32 else 400)
    a.box(26,495,888,91,"#faf0ed","#d2b9b3")
    a.text(41,521,"Historical boundary",17,700,"#862d29")
    a.text(41,546,"LFYO can train on later years; four late rules used the true held amount to decide eligibility.",15,400,"#752c2a")
    a.text(41,569,"Amount overlays leave the router unchanged. 830/912 is archive agreement, not forecast skill.",15,400,"#752c2a")
    return a


def inspect(path):
    raw=path.read_bytes()
    if sha256(raw).hexdigest()!=AUDIT_SHA: raise ValueError("Forward fold audit does not match frozen digest")
    rows=list(csv.DictReader(io.StringIO(raw.decode("ascii"))))
    assert [int(r["test_year"]) for r in rows]==list(range(2019,2025))
    assert sum(int(r["n_test"]) for r in rows)==376
    for r in rows:
        r["gap_days"]=(date.fromisoformat(r["min_test_incident_start"])-date.fromisoformat(r["max_train_end"])).days
        assert r["gap_days"]>0 and int(r["max_train_fy"])<int(r["test_year"])
    return rows


def f3(rows):
    a=Svg(560,"Purged rolling-origin split and six held fiscal years",
          "The forward model uses earlier eligible years only. It excludes shared episode groups and records that end at or after the earliest held incident-start or declaration date. Six audited origins hold 376 declarations in total.")
    a.text(26,34,"Forward evaluation: what can enter each training origin?",24,700)
    a.text(26,60,"The flow applies to the archived Track B ablations, not every thesis experiment.",15,400,MUTED)
    nodes=[(26,218,"Earlier fiscal years","Candidate training roster"),
           (283,354,"Time and episode purge","Strict cutoff; no shared episode groups"),
           (677,237,"Held fiscal year","No fitting on held outcomes")]
    for x,w,t,c in nodes:
        a.box(x,82,w,65,"#eff4f9" if x!=283 else "#ecf5ee","#b9c9d6")
        a.text(x+10,108,t,17,700);a.text(x+10,131,c,14)
    a.line(248,115,274,115,True);a.line(640,115,667,115,True)
    a.text(26,173,"Cutoff is the earlier of the first held incident start and the first held declaration date.",15)
    a.text(26,197,"Nested selection repeats the split on earlier origins; chronology does not prove feature availability.",14,400,MUTED)
    columns=[(40,"HELD FY"),(194,"FIT CASES"),(322,"EARLIER EXCLUDED"),(528,"HELD CASES"),(671,"END-to-START GAP")]
    for x,t in columns:a.text(x,237,t,14,700)
    for i,r in enumerate(rows):
        y=249+39*i
        a.box(26,y,888,34,"#f7f9fa" if i%2==0 else "#fff","#e0e6ec",4)
        rec=[r["test_year"],r["n_train"],r["rejected_earlier_records"],r["n_test"],"1 day" if r["gap_days"] == 1 else f'{r["gap_days"]} days']
        for (x,_),v in zip(columns,rec):a.text(x,y+23,v,15,700 if x==40 else 400)
    a.text(26,516,"Six historical test origins  ·  376 held declarations  ·  previously examined, not independent confirmation",14,700)
    a.text(26,540,"End-to-start gap: earliest held incident start minus latest eligible training incident end.",14,400,MUTED)
    return a


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("fold_audit",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args();rows=inspect(a.fold_audit)
    a.out.mkdir(parents=True,exist_ok=True)
    f2().write(a.out/"F2-frozen-router-lineage.svg")
    f3(rows).write(a.out/"F3-forward-split-audit.svg")
