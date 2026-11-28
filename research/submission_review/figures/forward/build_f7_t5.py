#!/usr/bin/env python3
"""Build aggregate F7/T5 from two published summary tables, not row-level outcomes."""
import argparse
import csv
import hashlib
from html import escape
from pathlib import Path

PINNED = {
    "confusion_matrices.csv": "eef78bc63876eef6bf5127ac2862de9eb0cf651870463355131aaa8d61d44509",
    "paired_event_bootstrap.csv": "127ae8cbc1a44823eaabb9ce0f25be00b783a4c595fe5c31c474f2281a4acc85",
}
BANDS = ("0-100K", "100K-1M", "1M-50M", "50-200M", "200-500M", "500M+")
LABELS = ("$0–<$100K", "$100K–<$1M", "$1M–<$50M", "$50M–<$200M", "$200M–<$500M", "$500M+")
LEFT = "ablation_retrospective_metadata"
RIGHT = "ablation_verified_link_only"


def read_data(root):
    result = {}
    for name, digest in PINNED.items():
        raw = (root / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"Input digest differs from frozen checkpoint: {name}")
        result[name] = list(csv.DictReader(raw.decode().splitlines()))
    return result


def summarize(data, mode):
    rows = [r for r in data["confusion_matrices.csv"] if r["variant"] == mode]
    mat = {(r["actual_band"], r["predicted_band"]): int(r["n"]) for r in rows}
    if len(rows) != 36 or len(mat) != 36 or not all((a, b) in mat for a in BANDS for b in BANDS):
        raise ValueError("Expected one complete six-by-six matrix")
    n = [sum(mat[a, p] for p in BANDS) for a in BANDS]
    h = [mat[a, a] for a in BANDS]
    if n != [266, 50, 47, 9, 1, 3]:
        raise ValueError("Unexpected cohort composition")
    return n, h, sum(mat[a,p] for a in BANDS[3:] for p in BANDS[3:]), sum(mat[a,p] for a in BANDS[:3] for p in BANDS[3:])


def svg(n, a, b, values):
    ink, muted, left, right = "#142438", "#46566A", "#31567F", "#B14D24"
    x0, x1, y_end = 286, 772, 335
    s = ['<svg xmlns="http://www.w3.org/2000/svg" width="940" height="546" viewBox="0 0 940 546" role="img" aria-labelledby="title description">',
         '<title id="title">Matched forward ablations: recall by band and dollar MAE</title>',
         '<desc id="description">Exact-band recall is paired within six observed bands. The only 200 to 500 million case changes from a hit to a miss. The dollar MAE interval crosses zero.</desc>',
         '<rect width="940" height="546" fill="white"/>']
    def text(x,y,label,size=16,color=ink,bold=False,anchor="start"):
        weight = ' font-weight="700"' if bold else ""
        s.append(f'<text x="{x}" y="{y}" fill="{color}" text-anchor="{anchor}" font-family="DejaVu Sans, sans-serif" font-size="{size}"{weight}>{escape(label)}</text>')
    def line(x,y,xx,yy,color,w=1):
        s.append(f'<path d="M {x} {y} L {xx} {yy}" fill="none" stroke="{color}" stroke-width="{w}"/>')
    def dot(x,y,color,square=False):
        if square: s.append(f'<rect x="{x-5}" y="{y-5}" width="10" height="10" fill="{color}"/>')
        else: s.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{color}"/>')
    text(29,33,"Matched forward ablations: 376 declarations, 325 event groups",24,bold=True)
    text(29,60,"FY2019–24 historical rolling-origin results. Separate from frozen Track A.",15,muted)
    dot(32,87,left); text(45,92,"Retrospective metadata",14)
    dot(248,87,right,True); text(260,92,"Verified-link ablation",14)
    text(29,129,"A   Exact-band recall",18,bold=True); text(891,129,"Actual n",14,muted,False,"end")
    s.append('<rect x="20" y="269" width="891" height="30" fill="#FFF3EB"/>')
    for percent in range(0,101,25):
        px=x0+(x1-x0)*percent/100
        line(px,143,px,y_end,"#DBE2E9")
        text(px,355,f"{percent}%",13,muted,False,"middle")
    for i,(count,hit_a,hit_b) in enumerate(zip(n,a,b)):
        y=158+31*i
        text(29,y+5,LABELS[i],15,bold=i==4)
        text(891,y+5,str(count),15,bold=i==4,anchor="end")
        xa=x0+(x1-x0)*hit_a/count
        xb=x0+(x1-x0)*hit_b/count
        line(xa,y-4,xb,y+4,"#A0AEBB",2)
        dot(xa,y-4,left)
        dot(xb,y+4,right,True)
    line(29,377,911,377,"#D4DCE4")
    text(29,411,"B   Change in mean absolute dollar error",18,bold=True)
    text(29,436,"Verified-link minus retrospective metadata. Positive means higher error.",15,muted)
    delta, lo, hi = values
    px=lambda v: x0+(x1-x0)*(v+20000)/70000
    line(x0,490,x1,490,"#BECAD3")
    line(px(0),448,px(0),490,"#73869A",1.5)
    line(px(lo),466,px(hi),466,right,3)
    for v in (lo,hi): line(px(v),459,px(v),473,right,2)
    dot(px(delta),466,right,True)
    for v in range(-20000,50001,10000):
        line(px(v),490,px(v),495,"#BECAD3")
        text(px(v),512,f"{v/1000:g}",13,muted,False,"middle")
    text(29,513,"USD thousands",14,muted)
    def signed(v):
        return ("−" if v < 0 else "+") + f"${abs(v):,.0f}"
    text(29,535,f"Point estimate {signed(delta)}; 95% episode-bootstrap interval: {signed(lo)} to {signed(hi)}.",14,muted)
    return "\n".join(s+["</svg>",""])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forward_dir",type=Path)
    parser.add_argument("out_dir",type=Path)
    args=parser.parse_args()
    d=read_data(args.forward_dir)
    n,a,at,af=summarize(d,LEFT)
    _,b,bt,bf=summarize(d,RIGHT)
    cash=next(r for r in d["paired_event_bootstrap.csv"] if r["variant"]==RIGHT)
    delta=tuple(float(cash[k]) for k in ("mae_difference_usd","mae_ci_low","mae_ci_high"))
    if [sum(a),sum(b),at,af,bt,bf]!=[192,224,8,36,6,32] or cash["clusters"]!="325" or not delta[1]<0<delta[2]:
        raise ValueError("Forward comparison differs from audited result")
    args.out_dir.mkdir(exist_ok=True,parents=True)
    with (args.out_dir/"T5-matched-forward.csv").open("w",newline="") as f:
        w=csv.writer(f,lineterminator="\n")
        w.writerow(["actual_band","actual_n","retrospective_metadata_hits","verified_link_hits","retrospective_metadata_recall","verified_link_recall"])
        for i,k in enumerate(BANDS): w.writerow([k,n[i],a[i],b[i],f"{a[i]/n[i]:.9f}",f"{b[i]/n[i]:.9f}"])
    (args.out_dir/"F7-matched-forward.svg").write_text(svg(n,a,b,delta),encoding="utf-8")


if __name__=="__main__": main()
