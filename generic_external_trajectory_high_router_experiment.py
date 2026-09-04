from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, confusion_matrix, balanced_accuracy_score, f1_score

import external_physical_severity_high_router_experiment as base
import cdc_biological_severity_signal_audit as cdc_audit
import generic_external_severity_high_router_experiment as generic

OUT = Path("generic_external_trajectory_high_router_results")
OUT.mkdir(exist_ok=True)

TRAJ = [
    "trajGrowth7Pct",
    "trajGrowth14Pct",
    "trajRecentBurdenPct",
    "trajRecentFatalityPct",
    "trajOnsetPct",
]
GENERIC = list(generic.GENERIC_NUM)
FOCUS = [4480, 4482, 4485, 4486, 4489, 4515]
SAFE_BASE_CAT = [c for c in base.BASE_CAT if c != "eventScale"]


def num(x):
    return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)


def rank_pct(df: pd.DataFrame, col: str):
    x = num(df[col]).fillna(0.0)
    return x.rank(pct=True, method="average")


def window_sum(g: pd.DataFrame, col: str, lo, hi) -> float:
    if col not in g.columns:
        return 0.0
    z = g[(g["date"] > lo) & (g["date"] <= hi)]
    v = num(z[col]).fillna(0).clip(lower=0)
    return float(v.sum())


def cdc_trajectory_snapshot(cdc: pd.DataFrame, decl, target_state: str):
    if pd.isna(decl):
        return None
    eligible = cdc[cdc["date"] <= decl].copy()
    if eligible.empty:
        return None

    rows = []
    for st, g in eligible.groupby("state_abbr"):
        pop = float(base.POP.get(str(st), np.nan))
        if not np.isfinite(pop) or pop <= 0:
            continue
        g = g.sort_values("date")
        r7 = window_sum(g, "new_case", decl - pd.Timedelta(days=7), decl)
        p7 = window_sum(g, "new_case", decl - pd.Timedelta(days=14), decl - pd.Timedelta(days=7))
        r14 = window_sum(g, "new_case", decl - pd.Timedelta(days=14), decl)
        p14 = window_sum(g, "new_case", decl - pd.Timedelta(days=28), decl - pd.Timedelta(days=14))
        d7 = window_sum(g, "new_death", decl - pd.Timedelta(days=7), decl)
        first = g[num(g["tot_cases"]).fillna(0) > 0]
        onset = float((decl - first["date"].min()).days) if len(first) else 0.0
        rows.append({
            "state": str(st),
            "growth7": (r7 + 1.0) / (p7 + 1.0),
            "growth14": (r14 + 1.0) / (p14 + 1.0),
            "recent_burden": r7 / pop * 1e5,
            "recent_fatality": d7 / pop * 1e5,
            "onset": onset,
        })
    ref = pd.DataFrame(rows)
    if ref.empty or target_state not in set(ref["state"]):
        return None
    for c in ["growth7", "growth14", "recent_burden", "recent_fatality", "onset"]:
        ref[c + "_pct"] = rank_pct(ref, c)
    z = ref[ref["state"] == target_state].iloc[-1]
    return {
        "trajGrowth7Pct": float(z["growth7_pct"]),
        "trajGrowth14Pct": float(z["growth14_pct"]),
        "trajRecentBurdenPct": float(z["recent_burden_pct"]),
        "trajRecentFatalityPct": float(z["recent_fatality_pct"]),
        "trajOnsetPct": float(z["onset_pct"]),
        "trajRawGrowth7": float(z["growth7"]),
        "trajRawGrowth14": float(z["growth14"]),
        "trajSourceAudit": "CDC",
    }


def noaa_trajectory_snapshot(noaa: pd.DataFrame, decl, incident: str, target_state: str):
    if pd.isna(decl):
        return None
    types = base.noaa_types(incident)
    if not types:
        return None

    n = noaa[
        noaa["EVENT_TYPE"].isin(types)
        & (noaa["END_DT"] <= decl)
        & (noaa["END_DT"] > decl - pd.Timedelta(days=28))
    ].copy()

    states = []
    for abbr, pop in base.POP.items():
        nm = base.STATE_NAME.get(abbr)
        if nm:
            states.append({"state": abbr, "STATE": nm, "population": float(pop)})
    ref = pd.DataFrame(states)
    if target_state not in set(ref["state"]):
        return None

    if len(n):
        n["deaths"] = (
            num(n["DEATHS_DIRECT"]).fillna(0) + num(n["DEATHS_INDIRECT"]).fillna(0)
        ).clip(lower=0)
        n["recent7"] = ((n["END_DT"] > decl - pd.Timedelta(days=7)) & (n["END_DT"] <= decl)).astype(int)
        n["prev7"] = ((n["END_DT"] > decl - pd.Timedelta(days=14)) & (n["END_DT"] <= decl - pd.Timedelta(days=7))).astype(int)
        n["recent14"] = ((n["END_DT"] > decl - pd.Timedelta(days=14)) & (n["END_DT"] <= decl)).astype(int)
        n["prev14"] = ((n["END_DT"] > decl - pd.Timedelta(days=28)) & (n["END_DT"] <= decl - pd.Timedelta(days=14))).astype(int)
        n["recent_deaths"] = n["deaths"] * n["recent7"]
        agg = n.groupby("STATE", as_index=False).agg(
            recent7=("recent7", "sum"),
            prev7=("prev7", "sum"),
            recent14=("recent14", "sum"),
            prev14=("prev14", "sum"),
            recent_deaths=("recent_deaths", "sum"),
        )
    else:
        agg = pd.DataFrame(columns=["STATE","recent7","prev7","recent14","prev14","recent_deaths"])

    ref = ref.merge(agg, on="STATE", how="left")
    for c in ["recent7","prev7","recent14","prev14","recent_deaths"]:
        ref[c] = num(ref[c]).fillna(0).clip(lower=0)
    ref["growth7"] = (ref["recent7"] + 1.0) / (ref["prev7"] + 1.0)
    ref["growth14"] = (ref["recent14"] + 1.0) / (ref["prev14"] + 1.0)
    ref["recent_burden"] = ref["recent7"] / ref["population"] * 1e6
    ref["recent_fatality"] = ref["recent_deaths"] / ref["population"] * 1e6

    # For short physical events, "onset" is the age of the earliest qualifying
    # event record in the preceding 28 days. Zero-event states get 0.
    if len(n):
        first = n.groupby("STATE")["BEGIN_DT"].min()
        ref["onset"] = ref["STATE"].map(
            lambda s: float((decl - first[s]).days) if s in first.index and pd.notna(first[s]) else 0.0
        ).clip(lower=0)
    else:
        ref["onset"] = 0.0

    for c in ["growth7","growth14","recent_burden","recent_fatality","onset"]:
        ref[c + "_pct"] = rank_pct(ref, c)
    z = ref[ref["state"] == target_state].iloc[-1]
    return {
        "trajGrowth7Pct": float(z["growth7_pct"]),
        "trajGrowth14Pct": float(z["growth14_pct"]),
        "trajRecentBurdenPct": float(z["recent_burden_pct"]),
        "trajRecentFatalityPct": float(z["recent_fatality_pct"]),
        "trajOnsetPct": float(z["onset_pct"]),
        "trajRawGrowth7": float(z["growth7"]),
        "trajRawGrowth14": float(z["growth14"]),
        "trajSourceAudit": "NOAA",
    }


def build_trajectory(source: pd.DataFrame):
    forbidden = {
        "target","band","totalObligatedFunding",
        "fundingPerMission","fundingPerAgency","fundingPerDay","eventScale"
    }
    bad = forbidden.intersection(source.columns)
    if bad:
        raise RuntimeError(f"Forbidden columns reached trajectory builder: {sorted(bad)}")

    years = sorted(
        pd.to_datetime(source["effectiveDeclarationDate"], errors="coerce")
        .dropna().dt.year.astype(int).unique().tolist()
    )
    years = [y for y in years if 2010 <= y <= 2024]
    noaa = base.fetch_noaa(years)
    cdc = cdc_audit.fetch_cdc()

    rows = []
    cache = {}
    for _, r in source.iterrows():
        dn = int(r["disasterNumber"])
        st = str(r["state"])
        inc = str(r["incidentType"])
        decl = pd.to_datetime(r["effectiveDeclarationDate"], errors="coerce")
        key = (str(decl), st, inc)
        if key in cache:
            feat = dict(cache[key]) if cache[key] is not None else None
        else:
            if inc.upper().strip() == "BIOLOGICAL":
                feat = cdc_trajectory_snapshot(cdc, decl, st)
            else:
                feat = noaa_trajectory_snapshot(noaa, decl, inc, st)
            cache[key] = dict(feat) if feat is not None else None
        if feat is None:
            feat = {c: 0.5 for c in TRAJ}
            feat.update({
                "trajRawGrowth7": np.nan,
                "trajRawGrowth14": np.nan,
                "trajSourceAudit": "NEUTRAL_UNSUPPORTED",
            })
        rows.append({"disasterNumber": dn, **feat})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "generic_trajectory_all_disasters.csv", index=False)
    return out


def metrics(y, pred):
    rec = recall_score(y, pred, labels=[3,4,5], average=None, zero_division=0)
    cm = confusion_matrix(y, pred, labels=[3,4,5])
    return {
        "r3": float(rec[0]), "r4": float(rec[1]), "r5": float(rec[2]),
        "correct3": int(cm[0,0]), "correct4": int(cm[1,1]), "correct5": int(cm[2,2]),
        "min_recall": float(rec.min()),
        "balanced_accuracy": float(balanced_accuracy_score(y,pred)),
        "macro_f1": float(f1_score(y,pred,average="macro",zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "pass_counts": bool(cm[0,0]>=32 and cm[1,1]>=8 and cm[2,2]>=6),
    }


def pair_oof(high, lo, hi, nums, cats):
    mask = high["band"].isin([lo,hi]).to_numpy()
    p = np.full(len(high), np.nan)
    for year in sorted(high.loc[mask,"fyDeclared"].astype(int).unique()):
        tr = mask & (high["fyDeclared"].astype(int).to_numpy()!=year)
        te = mask & (high["fyDeclared"].astype(int).to_numpy()==year)
        Xtr = base.prep(high.loc[tr], nums, cats)
        Xte = base.prep(high.loc[te], nums, cats)
        ytr = (high.loc[tr,"band"]==hi).astype(int)
        m = base.fit_cat(Xtr,ytr,Xte,cats)
        cls = list(m.classes_)
        p[te] = m.predict_proba(Xte)[:,cls.index(1)]
    yy = (high.loc[mask,"band"]==hi).astype(int).to_numpy()
    op = base.op_point(yy,p[mask])
    return p, op


def main():
    d = base.load_master()

    source = d[["disasterNumber","state","incidentType","effectiveDeclarationDate"]].copy()
    traj = build_trajectory(source)

    # Reuse the already-clean target-independent generic severity builder.
    generic.OUT = OUT
    gen = generic.build_generic_external(source)

    d = d.merge(traj,on="disasterNumber",how="left").merge(
        gen[["disasterNumber"]+GENERIC], on="disasterNumber", how="left"
    )
    for c in TRAJ + GENERIC:
        d[c] = num(d[c]).fillna(0.5).clip(0,1)

    high = d[d["band"].isin([3,4,5])].copy().reset_index(drop=True)
    safe_num = [c for c in base.BASE_NUM if c in high.columns]
    safe_cat = [c for c in SAFE_BASE_CAT if c in high.columns]

    variants = {
        "baseline_safe": (safe_num,safe_cat),
        "baseline_plus_generic_severity": (safe_num+GENERIC,safe_cat),
        "baseline_plus_trajectory": (safe_num+TRAJ,safe_cat),
        "baseline_plus_generic_severity_and_trajectory": (safe_num+GENERIC+TRAJ,safe_cat),
    }

    multi_rows=[]
    pair_rows=[]
    preds=[]
    pair_preds=[]

    years=sorted(high["fyDeclared"].astype(int).unique())
    for name,(nums,cats) in variants.items():
        pred=np.full(len(high),-99,dtype=int)
        for year in years:
            tr=high["fyDeclared"].astype(int)!=year
            te=~tr
            Xtr=base.prep(high.loc[tr],nums,cats)
            Xte=base.prep(high.loc[te],nums,cats)
            m=base.fit_cat(Xtr,high.loc[tr,"band"].astype(int),Xte,cats)
            pred[te]=np.asarray(m.predict(Xte)).reshape(-1).astype(int)
        met=metrics(high["band"].astype(int).to_numpy(),pred)
        multi_rows.append({"variant":name,**{k:v for k,v in met.items() if k!="confusion_matrix"},"confusion_matrix":json.dumps(met["confusion_matrix"])})
        q=high[["disasterNumber","state","fyDeclared","incidentType","target","band","trajSourceAudit"]].copy()
        q["variant"]=name
        q["predicted_band"]=pred
        preds.append(q)

        for lo,hi in [(3,4),(4,5)]:
            p,op=pair_oof(high,lo,hi,nums,cats)
            pair_rows.append({"variant":name,"pair":f"{lo}vs{hi}",**op})
            mask=high["band"].isin([lo,hi])
            qq=high.loc[mask,["disasterNumber","state","fyDeclared","incidentType","target","band","trajSourceAudit"]].copy()
            qq["variant"]=name
            qq["pair"]=f"{lo}vs{hi}"
            qq["prob_upper"]=p[mask.to_numpy()]
            pair_preds.append(qq)

    mdf=pd.DataFrame(multi_rows).sort_values(["min_recall","balanced_accuracy"],ascending=False)
    pdf=pd.DataFrame(pair_rows).sort_values(["pair","min_recall"],ascending=[True,False])
    mdf.to_csv(OUT/"multiclass_results.csv",index=False)
    pdf.to_csv(OUT/"pairwise_results.csv",index=False)
    pd.concat(preds,ignore_index=True).to_csv(OUT/"multiclass_predictions.csv",index=False)
    pd.concat(pair_preds,ignore_index=True).to_csv(OUT/"pairwise_predictions.csv",index=False)

    focus=high[high.disasterNumber.isin(FOCUS)][
        ["disasterNumber","state","fyDeclared","incidentType","target","band","trajSourceAudit"]+TRAJ+
        ["trajRawGrowth7","trajRawGrowth14"]
    ].sort_values("disasterNumber")
    focus.to_csv(OUT/"focus_2020_biological_trajectory.csv",index=False)

    summary={
        "purpose":"Test a cross-hazard, target-free recent-severity-trajectory representation motivated by the 2020 Biological signal audit.",
        "high_counts":{str(b):int((high.band==b).sum()) for b in [3,4,5]},
        "multiclass":multi_rows,
        "pairwise":pair_rows,
        "focus_cases":focus.to_dict("records"),
        "guardrails":[
            "Trajectory features are constructed for all master disasters without target, funding band, obligation amount, funding-per-resource fields, or eventScale.",
            "Biological uses only CDC observations on or before declaration; NOAA hazards use only event records ending on or before declaration.",
            "HHS hospital variables are deliberately excluded from this predictive experiment because contemporaneous availability has not yet been established.",
            "Original funding-derived eventScale is excluded from the model.",
            "Predictions are strict leave-fiscal-year-out. Pairwise operating thresholds are development-selected from combined OOF predictions and are not a final nested-validation claim.",
            "If any trajectory configuration clears the 80-percent integer floors in development, it must be rerun with nested outer/inner fiscal-year selection before claiming success."
        ],
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str),flush=True)


if __name__=="__main__":
    main()
