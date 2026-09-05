from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, confusion_matrix, balanced_accuracy_score, f1_score

import external_physical_severity_high_router_experiment as base
import generic_external_severity_high_router_experiment as generic
import generic_external_trajectory_high_router_experiment as trajmod

OUT = Path("non_biological_high_router_diagnostic_results")
OUT.mkdir(exist_ok=True)

GENERIC = list(generic.GENERIC_NUM)
TRAJ = list(trajmod.TRAJ)
SAFE_CATS = [c for c in base.BASE_CAT if c != "eventScale"]


def num(x):
    return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)


def metrics(y, pred):
    rec = recall_score(y, pred, labels=[3,4,5], average=None, zero_division=0)
    cm = confusion_matrix(y, pred, labels=[3,4,5])
    return {
        "r3": float(rec[0]), "r4": float(rec[1]), "r5": float(rec[2]),
        "correct3": int(cm[0,0]), "correct4": int(cm[1,1]), "correct5": int(cm[2,2]),
        "n3": int((y==3).sum()), "n4": int((y==4).sum()), "n5": int((y==5).sum()),
        "min_recall": float(rec.min()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "pass_80_each": bool(np.all(rec >= 0.80)),
    }


def pair_oof(high, lo, hi, nums, cats):
    mask = high["band"].isin([lo,hi]).to_numpy()
    p = np.full(len(high), np.nan)
    for year in sorted(high.loc[mask, "fyDeclared"].astype(int).unique()):
        tr = mask & (high["fyDeclared"].astype(int).to_numpy() != year)
        te = mask & (high["fyDeclared"].astype(int).to_numpy() == year)
        if te.sum() == 0:
            continue
        ytr = (high.loc[tr, "band"] == hi).astype(int)
        if ytr.nunique() < 2:
            continue
        Xtr = base.prep(high.loc[tr], nums, cats)
        Xte = base.prep(high.loc[te], nums, cats)
        m = base.fit_cat(Xtr, ytr, Xte, cats)
        cls = list(m.classes_)
        p[te] = m.predict_proba(Xte)[:, cls.index(1)]

    valid = mask & np.isfinite(p)
    yy = (high.loc[valid, "band"] == hi).astype(int).to_numpy()
    pp = p[valid]
    op = base.op_point(yy, pp) if len(np.unique(yy)) == 2 else {
        "auc": np.nan, "threshold": np.nan, "lower_recall": np.nan,
        "upper_recall": np.nan, "min_recall": np.nan, "pass80": False
    }
    return p, op


def main():
    d = base.load_master()

    source = d[["disasterNumber","state","incidentType","effectiveDeclarationDate"]].copy()

    # Build the same target-independent external features used in prior clean screens.
    generic.OUT = OUT
    gen = generic.build_generic_external(source)
    trajmod.OUT = OUT
    trj = trajmod.build_trajectory(source)

    d = d.merge(gen[["disasterNumber"] + GENERIC], on="disasterNumber", how="left")
    d = d.merge(trj[["disasterNumber"] + TRAJ], on="disasterNumber", how="left")
    for c in GENERIC + TRAJ:
        d[c] = num(d[c]).fillna(0.5)

    all_high = d[d["band"].isin([3,4,5])].copy()
    removed = all_high[all_high["incidentType"].astype(str).str.upper().str.strip() == "BIOLOGICAL"].copy()
    high = all_high[all_high["incidentType"].astype(str).str.upper().str.strip() != "BIOLOGICAL"].copy().reset_index(drop=True)

    safe_num = [c for c in base.BASE_NUM if c in high.columns]
    cats = [c for c in SAFE_CATS if c in high.columns]

    variants = {
        "baseline_safe_no_biological": (safe_num, cats),
        "baseline_plus_generic_severity_no_biological": (safe_num + GENERIC, cats),
        "baseline_plus_trajectory_no_biological": (safe_num + TRAJ, cats),
        "baseline_plus_generic_and_trajectory_no_biological": (safe_num + GENERIC + TRAJ, cats),
    }

    years = sorted(high["fyDeclared"].astype(int).unique())
    rows = []
    preds = []
    pair_rows = []
    pair_preds = []

    for name, (nums, cats_) in variants.items():
        pred = np.full(len(high), -99, dtype=int)

        for year in years:
            tr = high["fyDeclared"].astype(int) != year
            te = ~tr
            if te.sum() == 0:
                continue
            ytr = high.loc[tr, "band"].astype(int)
            Xtr = base.prep(high.loc[tr], nums, cats_)
            Xte = base.prep(high.loc[te], nums, cats_)
            m = base.fit_cat(Xtr, ytr, Xte, cats_)
            pred[te] = np.asarray(m.predict(Xte)).reshape(-1).astype(int)

        met = metrics(high["band"].astype(int).to_numpy(), pred)
        rows.append({
            "variant": name,
            **{k:v for k,v in met.items() if k != "confusion_matrix"},
            "confusion_matrix": json.dumps(met["confusion_matrix"]),
        })

        q = high[["disasterNumber","state","fyDeclared","incidentType","target","band"]].copy()
        q["variant"] = name
        q["predicted_band"] = pred
        preds.append(q)

        for lo, hi in [(3,4),(4,5)]:
            p, op = pair_oof(high, lo, hi, nums, cats_)
            pair_rows.append({"variant": name, "pair": f"{lo}vs{hi}", **op})
            mask = high["band"].isin([lo,hi])
            qq = high.loc[mask, ["disasterNumber","state","fyDeclared","incidentType","target","band"]].copy()
            qq["variant"] = name
            qq["pair"] = f"{lo}vs{hi}"
            qq["prob_upper"] = p[mask.to_numpy()]
            pair_preds.append(qq)

    results = pd.DataFrame(rows).sort_values(
        ["pass_80_each","min_recall","balanced_accuracy"],
        ascending=[False,False,False]
    )
    results.to_csv(OUT/"multiclass_results.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(OUT/"multiclass_predictions.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(OUT/"pairwise_results.csv", index=False)
    pd.concat(pair_preds, ignore_index=True).to_csv(OUT/"pairwise_predictions.csv", index=False)

    removed[["disasterNumber","state","fyDeclared","incidentType","target","band"]].to_csv(
        OUT/"removed_biological_high_cases.csv", index=False
    )
    high[["disasterNumber","state","fyDeclared","incidentType","target","band"]].to_csv(
        OUT/"retained_non_biological_high_cases.csv", index=False
    )

    counts = high.groupby(["incidentType","band"]).size().reset_index(name="n")

    summary = {
        "purpose": "Diagnostic: rerun the high three-band router after completely removing Biological disasters from both training and evaluation.",
        "original_high_count": int(len(all_high)),
        "removed_biological_count": int(len(removed)),
        "remaining_non_biological_high_count": int(len(high)),
        "remaining_band_counts": {str(b): int((high.band==b).sum()) for b in [3,4,5]},
        "integer_80_floors_for_remaining_data": {
            "3": int(np.ceil(0.8 * (high.band==3).sum())),
            "4": int(np.ceil(0.8 * (high.band==4).sum())),
            "5": int(np.ceil(0.8 * (high.band==5).sum())),
        },
        "incident_band_counts": counts.to_dict("records"),
        "multiclass": rows,
        "pairwise": pair_rows,
        "guardrails": [
            "Biological disasters are excluded from both training and test folds.",
            "Validation remains strict leave-fiscal-year-out on the remaining non-Biological high-value cases.",
            "No funding-derived feature, obligation amount, or eventScale is used as a predictor.",
            "External severity/trajectory builders remain target-independent.",
            "This is a diagnostic of whether Biological cases are the dominant structural obstacle; it does not replace the thesis requirement to route all six operational bands."
        ]
    }
    (OUT/"summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
