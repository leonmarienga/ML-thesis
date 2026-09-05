from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score, balanced_accuracy_score, f1_score

import external_physical_severity_high_router_experiment as base
import hurricane_local_impact_signal_audit as hlocal

OUT = Path("nested_hurricane_local_impact_rescue_results")
OUT.mkdir(exist_ok=True)

SAFE_CATS = [c for c in base.BASE_CAT if c != "eventScale"]
HUR_TYPES = {"HURRICANE", "TROPICAL STORM"}

CANDIDATES = [
    "wind_at_closest_kt",
    "hours_within_100km",
    "hours_within_200km",
    "hours_within_300km",
    "max_wind_within_100km",
    "local_hurricane_hours_300km",
    "local_impact_integral",
    "distance_weighted_wind",
    "local_peak_proxy",
]


def num(x):
    return pd.to_numeric(x, errors="coerce").replace([np.inf,-np.inf], np.nan)


def build_hurricane_features(d):
    source = d[
        d["incidentType"].astype(str).str.upper().isin(HUR_TYPES)
        & d["band"].isin([3,4,5])
    ].copy()

    titles = hlocal.fetch_titles(source["disasterNumber"].astype(int).tolist())
    cents = hlocal.fetch_state_centroids()
    storms, points = hlocal.parse_hurdat_points()

    source = source.merge(
        titles[["disasterNumber","declarationTitle"]],
        on="disasterNumber", how="left"
    )
    source = source.merge(
        cents[["state","centroid_lat","centroid_lon"]],
        on="state", how="left"
    )

    rows=[]
    for _, r in source.iterrows():
        s = hlocal.exact_named_match(r, storms)
        q = {"disasterNumber":int(r.disasterNumber)}
        if s is None:
            q["hurricaneMatched"] = 0
        else:
            f = hlocal.local_features(
                str(s.storm_id), points,
                float(r.centroid_lat) if pd.notna(r.centroid_lat) else np.nan,
                float(r.centroid_lon) if pd.notna(r.centroid_lon) else np.nan,
                float(base.POP.get(str(r.state), np.nan)),
            )
            q["hurricaneMatched"] = 1 if f is not None else 0
            if f:
                for c in CANDIDATES:
                    q[c] = f.get(c, np.nan)
        rows.append(q)

    out = pd.DataFrame(rows)
    for c in CANDIDATES:
        if c not in out:
            out[c] = np.nan
    out.to_csv(OUT/"hurricane_local_features.csv", index=False)
    return out


def best_threshold(y, x):
    y=np.asarray(y,int)
    x=np.asarray(x,float)
    ok=np.isfinite(x)
    y=y[ok]; x=x[ok]
    if len(y)<4 or len(np.unique(y))<2 or len(np.unique(x))<2:
        return None
    best=None
    for sign in [1,-1]:
        s=sign*x
        vals=np.unique(s)
        thrs=np.r_[vals.min()-1e-12,(vals[:-1]+vals[1:])/2,vals.max()+1e-12]
        for t in thrs:
            pred=(s>=t).astype(int)
            r0=float((pred[y==0]==0).mean())
            r1=float((pred[y==1]==1).mean())
            key=(min(r0,r1), r0+r1)
            if best is None or key>best[0]:
                best=(key,sign,float(t),r0,r1)
    return {
        "sign":int(best[1]),
        "threshold":float(best[2]),
        "lower_recall":float(best[3]),
        "upper_recall":float(best[4]),
        "min_recall":float(min(best[3],best[4]))
    }


def select_gate(train):
    h = train[
        train["incidentType"].astype(str).str.upper().isin(HUR_TYPES)
        & train["hurricaneMatched"].eq(1)
        & train["band"].isin([3,4,5])
    ].copy()
    if h.empty:
        return None

    y=(h["band"]==5).astype(int).to_numpy()
    best=None
    for c in CANDIDATES:
        if c not in h: continue
        op=best_threshold(y, num(h[c]).to_numpy(float))
        if not op: continue
        # Prefer both-side balance first, then lower-side specificity because
        # rescue must not steal lower/middle cases.
        key=(op["min_recall"], op["lower_recall"], op["upper_recall"])
        if best is None or key>best[0]:
            best=(key,c,op)
    if best is None:
        return None
    return {"feature":best[1], **best[2]}


def metrics(y,p):
    rec=recall_score(y,p,labels=[3,4,5],average=None,zero_division=0)
    cm=confusion_matrix(y,p,labels=[3,4,5])
    return {
        "r3":float(rec[0]),"r4":float(rec[1]),"r5":float(rec[2]),
        "correct3":int(cm[0,0]),"correct4":int(cm[1,1]),"correct5":int(cm[2,2]),
        "n3":int((y==3).sum()),"n4":int((y==4).sum()),"n5":int((y==5).sum()),
        "min_recall":float(rec.min()),
        "balanced_accuracy":float(balanced_accuracy_score(y,p)),
        "macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),
        "confusion_matrix":cm.tolist(),
        "pass80":bool(np.all(rec>=.8)),
    }


def main():
    d=base.load_master()
    hf=build_hurricane_features(d)
    d=d.merge(hf,on="disasterNumber",how="left")
    d["hurricaneMatched"]=num(d["hurricaneMatched"]).fillna(0).astype(int)

    high=d[
        d["band"].isin([3,4,5])
        & ~d["incidentType"].astype(str).str.upper().eq("BIOLOGICAL")
    ].copy().reset_index(drop=True)

    nums=[c for c in base.BASE_NUM if c in high.columns]
    cats=[c for c in SAFE_CATS if c in high.columns]

    years=sorted(high["fyDeclared"].astype(int).unique())
    baseline=np.full(len(high),-99,int)
    rescued=np.full(len(high),-99,int)
    selections=[]
    rows=[]

    for outer in years:
        tr=high["fyDeclared"].astype(int)!=outer
        te=~tr

        Xtr=base.prep(high.loc[tr],nums,cats)
        Xte=base.prep(high.loc[te],nums,cats)
        m=base.fit_cat(Xtr,high.loc[tr,"band"].astype(int),Xte,cats)
        bp=np.asarray(m.predict(Xte)).reshape(-1).astype(int)

        test=high.loc[te].copy()
        rp=bp.copy()

        gate=select_gate(high.loc[tr].copy())
        if gate is not None:
            eligible=(
                test["incidentType"].astype(str).str.upper().isin(HUR_TYPES)
                & test["hurricaneMatched"].eq(1)
            ).to_numpy()
            x=num(test[gate["feature"]]).to_numpy(float)
            flag=eligible & np.isfinite(x) & ((gate["sign"]*x)>=gate["threshold"])
            # Rescue-only: never demote an existing prediction. Only promote
            # hurricane-like cases to $500M+ when outer-training data supports it.
            rp[flag]=5

        baseline[te.to_numpy()]=bp
        rescued[te.to_numpy()]=rp

        selections.append({
            "outer_year":int(outer),
            **(gate if gate is not None else {"feature":None})
        })

        idx=np.flatnonzero(te.to_numpy())
        for j,ii in enumerate(idx):
            r=high.iloc[ii]
            rows.append({
                "outer_year":int(outer),
                "disasterNumber":int(r.disasterNumber),"state":str(r.state),
                "incidentType":str(r.incidentType),"target":float(r.target),"band":int(r.band),
                "baseline_prediction":int(bp[j]),"rescued_prediction":int(rp[j]),
                "hurricaneMatched":int(r.hurricaneMatched),
                "selected_feature":gate["feature"] if gate else "",
                "selected_threshold":gate["threshold"] if gate else np.nan,
                "selected_sign":gate["sign"] if gate else np.nan,
            })

    y=high["band"].astype(int).to_numpy()
    bm=metrics(y,baseline)
    rm=metrics(y,rescued)

    pred=pd.DataFrame(rows)
    pred.to_csv(OUT/"nested_oof_predictions.csv",index=False)
    pd.DataFrame(selections).to_csv(OUT/"outer_fold_hurricane_gate_selection.csv",index=False)

    changed=pred[pred.baseline_prediction!=pred.rescued_prediction].copy()
    changed.to_csv(OUT/"hurricane_rescue_changes.csv",index=False)

    summary={
        "purpose":"Fully outer-held-year diagnostic of a hurricane local-impact rescue specialist on the non-Biological high router.",
        "non_biological_high_counts":{str(b):int((high.band==b).sum()) for b in [3,4,5]},
        "baseline":bm,
        "hurricane_rescue":rm,
        "changed_cases":changed.to_dict("records"),
        "outer_fold_selections":selections,
        "guardrails":[
            "Biological disasters are excluded from both training and evaluation in this diagnostic.",
            "The base router remains strict leave-fiscal-year-out and excludes funding-derived eventScale.",
            "Hurricane local-impact features are target-free HURDAT2/Census measurements.",
            "Feature and threshold selection use only the outer training years; the held fiscal year is never used in rescue selection.",
            "The rescue is promotion-only to $500M+ to avoid using a specialist to demote unrelated lower/middle cases.",
            "This is a nested development diagnostic for the non-Biological branch; the final six-band requirement remains unchanged."
        ]
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str),flush=True)


if __name__=="__main__":
    main()
