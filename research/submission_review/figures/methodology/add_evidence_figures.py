#!/usr/bin/env python3
"""Build two documentation diagrams from established evidence paths."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))  # trusted adjacent source in isolated mode
from build_methodology_figures import Svg, MUTED


def source_matrix():
    a=Svg(562,"Source linkage and decision-time evidence",
          "Physical evidence must pass identity, measurement, timing and evaluation support checks. Example source outcomes preserve the distinct stage and discharge clocks and the unqualified coastal collocation.")
    a.text(26,34,"A timestamp in an archive does not certify availability",24,700)
    a.text(26,60,"Four checks determine which source comparisons can be interpreted.",15,400,MUTED)
    top=[(26,"1   Identity","Same site, event or footprint"),
         (255,"2   Quantity","Unit, datum or valid area"),
         (484,"3   Clock","Issue and valid time"),
         (713,"4   Support","Independent episode checks")]
    for x,t,q in top:
        a.box(x,79,202,62,"#edf4f9","#aec2d1")
        a.text(x+9,103,t,17,700)
        a.text(x+9,125,q,11.8)
    for x in (232,461,690):a.line(x,110,x+17,110,True)
    a.text(36,181,"SOURCE",14,700)
    a.text(282,181,"SUPPORTED COMPARISON",14,700)
    a.text(606,181,"WHAT REMAINS UNVERIFIED",14,700)
    rows=[
        ("Satellite flood windows",["Event and prior-year change","Only commonly observed pixels"],["Full-footprint coverage","and effects on obligations"]),
        ("Montpelier stage",["Exact future gauge pairs","within one flood episode"],["Public dissemination clock","and cross-event skill"]),
        ("Michigan Bar stage",["Two archived RVFSJ clocks","and approved same-gauge stage"],["Separate QINE discharge clock","and cross-event skill"]),
        ("Coastal source screen",["Candidate locations and","local geometry checked"],["Representative wet-mesh match","and transfer skill"]),
    ]
    for i,(name,good,bad) in enumerate(rows):
        y=195+i*77
        a.box(26,y,889,67,"#f7f9fb" if i%2==0 else "#fff","#d2dce6",7)
        a.text(37,y+37,name,16,700)
        for x,v in [(283,good),(607,bad)]:
            a.text(x,y+25,v[0],15);a.text(x,y+47,v[1],15)
    a.box(26,511,889,37,"#faf0ed","#d2b9b3",6)
    a.text(38,535,"Failures stay visible. A physical match alone supplies neither a signed funding label nor evidence of need.",14,700,"#752c2a")
    return a


def replay_paths():
    a=Svg(554,"Offline application and reviewer replay are separate",
          "The offline decision support viewer and reviewer notebook are separate outputs from pinned research evidence. Neither requires live inference, FY2025 outcomes or measurement of real dispatch.")
    a.text(26,34,"Two reviewer paths from pinned evidence",24,700)
    a.text(26,60,"The application visualizes scenarios. The notebook verifies selected reported claims.",15,400,MUTED)
    a.box(27,78,886,68,"#edf4f9","#bacbd9")
    a.text(42,106,"Versioned source artifacts",18,700)
    a.text(42,130,"Historical predictions and checks, declared physical studies, and explicitly labeled simulated outputs",15)
    a.line(252,148,252,178,True);a.line(687,148,687,178,True)
    for x,title,sub,color,stroke in [
          (28,"Offline decision-support viewer","Deterministic HTML built from archived fixtures","#edf5ef","#aac4b2"),
          (492,"Scoped reviewer notebook","Manifest pins 18 public review inputs","#f5f1e9","#cdbca2")]:
        a.box(x,181,421,76,color,stroke)
        a.text(x+14,213,title,18,700)
        a.text(x+14,238,sub,15)
    for x in (238,702): a.line(x,261,x,290,True)
    boxes=[
        (28,["Historical obligation and uncertainty views","Transport and scripted review comparators","Inspect assumptions and compare road regimes"]),
        (492,["Reject altered or missing evidence","Recompute supported aggregate summaries","Export executed cells and scoped claim ledger"])
    ]
    for x,txt in boxes:
        a.box(x,294,421,117,"#f7f9fa","#c3ced5")
        for i,t in enumerate(txt):a.text(x+14,325+i*31,t,15)
    a.box(27,441,886,83,"#faf0ed","#d2b9b3")
    a.text(42,470,"Boundary for both paths",17,700,"#862d29")
    a.text(42,499,"No live funding forecast, automatic dispatch, measured relief delivery or complete model retraining.",15,400,"#752c2a")
    return a


if __name__=="__main__":
    output=Path(sys.argv[1])
    output.mkdir(exist_ok=True,parents=True)
    source_matrix().write(output/"F4-hazard-source-gates.svg")
    replay_paths().write(output/"F5-reviewer-replay-paths.svg")
