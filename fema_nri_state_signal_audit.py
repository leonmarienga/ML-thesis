from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score

import external_physical_severity_high_router_experiment as base

OUT = Path("fema_nri_state_signal_audit_results")
OUT.mkdir(exist_ok=True)

NRI_URL = "https://services.arcgis.com/XG15cJAlne2vxtgt/arcgis/rest/services/National_Risk_Index_Counties/FeatureServer/0/query"

# Current NRI is deliberately used only for a signal audit. It is not temporally
# valid for historical declarations before the corresponding NRI vintage existed.
NRI_FIELDS = [
    "STATEABBRV", "POPULATION",
    "RISK_SCORE", "EAL_SCORE", "SOVI_SCORE", "RESL_SCORE", "CRF_VALUE",
    "HRCN_RISKS", "WFIR_RISKS", "ERQK_RISKS",
    "CFLD_RISKS", "RFLD_RISKS", "SWND_RISKS", "TRND_RISKS",
    "HAIL_RISKS", "ISTM_RISKS", "WNTW_RISKS", "LTNG_RISKS",
]

STRUCTURAL = [
    "nriRiskMean", "nriRiskMax",
    "nriEalMean", "nriEalMax",
    "nriSoviMean", "nriSoviMax",
    "nriReslMean", "nriReslMin",
    "nriCrfMean",
]

MAPPED = [
    "nriHazardRiskMean", "nriHazardRiskMax",
]

BASE_NUM = [
    c for c in base.BASE_NUM
    if c not in {
        "fundingPerMission", "fundingPerAgency", "fundingPerDay",
        "totalObligatedFunding", "target"
    }
]
BASE_CAT = [c for c in base.BASE_CAT if c != "eventScale"]


def as_num(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def fetch_nri_counties() -> pd.DataFrame:
    rows = []
    offset = 0
    while True:
        params = {
            "where": "1=1",
            "outFields": ",".join(NRI_FIELDS),
            "returnGeometry": "false",
            "f": "json",
            "resultOffset": offset,
            "resultRecordCount": 2000,
        }
        r = requests.get(NRI_URL, params=params, timeout=180, headers={"User-Agent": "ML-thesis-research/1.0"})
        r.raise_for_status()
        payload = r.json()
        if "error" in payload:
            raise RuntimeError(payload["error"])
        features = payload.get("features", [])
        rows.extend([f.get("attributes", {}) for f in features])
        if len(features) < 2000:
            break
        offset += 2000
    if not rows:
        raise RuntimeError("NRI ArcGIS endpoint returned no county rows.")
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "nri_current_counties_raw_selected_fields.csv", index=False)
    return d


def weighted_mean(g: pd.DataFrame, col: str) -> float:
    x = as_num(g[col])
    w = as_num(g["POPULATION"]).fillna(0).clip(lower=0)
    ok = x.notna()
    if not ok.any():
        return np.nan
    if w.loc[ok].sum() <= 0:
        return float(x.loc[ok].mean())
    return float(np.average(x.loc[ok], weights=w.loc[ok]))


def aggregate_state_nri(nri: pd.DataFrame) -> pd.DataFrame:
    out = []
    for state, g in nri.groupby("STATEABBRV"):
        row = {"state": str(state)}
        for src, mean_name, extreme_name, extreme_fn in [
            ("RISK_SCORE", "nriRiskMean", "nriRiskMax", "max"),
            ("EAL_SCORE", "nriEalMean", "nriEalMax", "max"),
            ("SOVI_SCORE", "nriSoviMean", "nriSoviMax", "max"),
            ("RESL_SCORE", "nriReslMean", "nriReslMin", "min"),
            ("CRF_VALUE", "nriCrfMean", None, None),
        ]:
            row[mean_name] = weighted_mean(g, src)
            if extreme_name:
                vals = as_num(g[src]).dropna()
                row[extreme_name] = float(getattr(vals, extreme_fn)()) if len(vals) else np.nan

        # Population-weighted and maximum hazard-specific state profiles.
        for c in NRI_FIELDS:
            if c.endswith("_RISKS"):
                row[c + "_MEAN"] = weighted_mean(g, c)
                vals = as_num(g[c]).dropna()
                row[c + "_MAX"] = float(vals.max()) if len(vals) else np.nan
        out.append(row)

    s = pd.DataFrame(out)
    s.to_csv(OUT / "nri_current_state_aggregates.csv", index=False)
    return s


def hazard_prefix(incident: str):
    x = str(incident).strip().upper()
    if "HURRICANE" in x or "TROPICAL" in x:
        return ["HRCN"]
    if "FIRE" in x:
        return ["WFIR"]
    if "EARTHQUAKE" in x:
        return ["ERQK"]
    if "FLOOD" in x:
        return ["RFLD", "CFLD"]
    if "SEVERE STORM" in x or x == "STORM":
        return ["SWND", "TRND", "HAIL", "LTNG", "RFLD"]
    if "TORNADO" in x:
        return ["TRND"]
    if "WINTER" in x or "SNOW" in x:
        return ["WNTW", "ISTM"]
    if "ICE" in x:
        return ["ISTM", "WNTW"]
    if "WIND" in x:
        return ["SWND"]
    # Biological and other non-NRI hazards intentionally get no mapped hazard score.
    return []


def attach_nri(master: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    d = master.merge(state, on="state", how="left")
    mean_vals = []
    max_vals = []
    for _, r in d.iterrows():
        prefixes = hazard_prefix(r["incidentType"])
        means = [r.get(f"{p}_RISKS_MEAN", np.nan) for p in prefixes]
        maxes = [r.get(f"{p}_RISKS_MAX", np.nan) for p in prefixes]
        means = [float(v) for v in means if pd.notna(v)]
        maxes = [float(v) for v in maxes if pd.notna(v)]
        mean_vals.append(float(np.mean(means)) if means else 50.0)
        max_vals.append(float(np.max(maxes)) if maxes else 50.0)
    d["nriHazardRiskMean"] = mean_vals
    d["nriHazardRiskMax"] = max_vals
    for c in STRUCTURAL + MAPPED:
        d[c] = as_num(d[c]).fillna(50.0)
    return d


def oof_multiclass(d: pd.DataFrame, nums: list[str], cats: list[str]) -> np.ndarray:
    high = d[d["band"].isin([3, 4, 5])].copy().reset_index(drop=True)
    pred = np.full(len(high), -99, dtype=int)
    for year in sorted(high["fyDeclared"].astype(int).unique()):
        tr = high["fyDeclared"].astype(int) != year
        te = ~tr
        Xtr = base.prep(high.loc[tr], nums, cats)
        Xte = base.prep(high.loc[te], nums, cats)
        model = base.fit_cat(Xtr, high.loc[tr, "band"].astype(int), Xte, cats)
        pred[te] = np.asarray(model.predict(Xte)).reshape(-1).astype(int)
    return high, pred


def multi_metrics(y, p):
    rec = recall_score(y, p, labels=[3,4,5], average=None, zero_division=0)
    cm = confusion_matrix(y, p, labels=[3,4,5])
    return {
        "r3": float(rec[0]), "r4": float(rec[1]), "r5": float(rec[2]),
        "correct3": int(cm[0,0]), "correct4": int(cm[1,1]), "correct5": int(cm[2,2]),
        "min_recall": float(rec.min()),
        "confusion_matrix": cm.tolist(),
        "pass_counts": bool(cm[0,0] >= 32 and cm[1,1] >= 8 and cm[2,2] >= 6),
    }


def oof_pair(d: pd.DataFrame, lo: int, hi: int, nums: list[str], cats: list[str]):
    z = d[d["band"].isin([lo, hi])].copy().reset_index(drop=True)
    prob = np.full(len(z), np.nan)
    for year in sorted(z["fyDeclared"].astype(int).unique()):
        tr = z["fyDeclared"].astype(int) != year
        te = ~tr
        Xtr = base.prep(z.loc[tr], nums, cats)
        Xte = base.prep(z.loc[te], nums, cats)
        ytr = (z.loc[tr, "band"] == hi).astype(int)
        model = base.fit_cat(Xtr, ytr, Xte, cats)
        classes = list(model.classes_)
        prob[te] = model.predict_proba(Xte)[:, classes.index(1)]
    y = (z["band"] == hi).astype(int).to_numpy()
    op = base.op_point(y, prob)
    op["auc"] = float(roc_auc_score(y, prob))
    return z, prob, op


def main():
    master = base.load_master()
    nri = fetch_nri_counties()
    state = aggregate_state_nri(nri)
    d = attach_nri(master, state)

    variants = {
        "baseline": (BASE_NUM, BASE_CAT),
        "baseline_plus_nri_structural_CURRENT_LOOKAHEAD_DIAGNOSTIC": (
            BASE_NUM + STRUCTURAL, BASE_CAT
        ),
        "baseline_plus_nri_structural_and_hazard_CURRENT_LOOKAHEAD_DIAGNOSTIC": (
            BASE_NUM + STRUCTURAL + MAPPED, BASE_CAT
        ),
        "nri_only_CURRENT_LOOKAHEAD_DIAGNOSTIC": (
            STRUCTURAL + MAPPED, ["incidentType"]
        ),
    }

    multi_rows = []
    pair_rows = []
    pred_rows = []
    pair_pred_rows = []

    for name, (nums, cats) in variants.items():
        nums = [c for c in nums if c in d.columns]
        cats = [c for c in cats if c in d.columns]

        high, pred = oof_multiclass(d, nums, cats)
        met = multi_metrics(high["band"].astype(int).to_numpy(), pred)
        multi_rows.append({"variant": name, **met})
        q = high[["disasterNumber","state","fyDeclared","incidentType","target","band"]].copy()
        q["variant"] = name
        q["predicted_band"] = pred
        pred_rows.append(q)

        for lo, hi in [(3,4),(4,5)]:
            z, prob, op = oof_pair(d, lo, hi, nums, cats)
            pair_rows.append({"variant": name, "pair": f"{lo}vs{hi}", **op})
            qq = z[["disasterNumber","state","fyDeclared","incidentType","target","band"]].copy()
            qq["variant"] = name
            qq["pair"] = f"{lo}vs{hi}"
            qq["prob_upper"] = prob
            pair_pred_rows.append(qq)

    mdf = pd.DataFrame(multi_rows).sort_values(["min_recall"], ascending=False)
    pdf = pd.DataFrame(pair_rows).sort_values(["pair","min_recall"], ascending=[True,False])
    mdf.to_csv(OUT / "multiclass_signal_audit.csv", index=False)
    pdf.to_csv(OUT / "pairwise_signal_audit.csv", index=False)
    pd.concat(pred_rows, ignore_index=True).to_csv(OUT / "multiclass_oof_predictions.csv", index=False)
    pd.concat(pair_pred_rows, ignore_index=True).to_csv(OUT / "pairwise_oof_predictions.csv", index=False)

    focus = d[d["disasterNumber"].isin([4482,4486,4489,4515,4480,4485])][
        ["disasterNumber","state","fyDeclared","incidentType","target","band"] + STRUCTURAL + MAPPED
    ].sort_values("disasterNumber")
    focus.to_csv(OUT / "focus_2020_biological_nri_state_profiles.csv", index=False)

    summary = {
        "purpose": "Signal audit of official FEMA National Risk Index state profiles before deciding whether a temporally safe historical reconstruction is justified.",
        "nri_source": NRI_URL,
        "nri_current_service_warning": "CURRENT NRI IS TEMPORAL LOOKAHEAD FOR HISTORICAL FEMA DECLARATIONS AND IS NOT AN ACCEPTABLE FINAL ROUTER FEATURE.",
        "official_nri_first_release": "October 2020 v1.17.0",
        "multiclass": multi_rows,
        "pairwise": pair_rows,
        "focus_2020_biological": focus.to_dict("records"),
        "decision_rule": "Only if current NRI structural/hazard scores add material separability will we reconstruct pre-event equivalents from historically available source vintages. Current-NRI results can never be claimed as final validation.",
        "guardrails": [
            "No FEMA obligation/funding target or funding-derived master feature is used as an NRI predictor.",
            "Original funding-derived eventScale is excluded.",
            "Leave-fiscal-year-out model fitting is still enforced, but that alone does not cure temporal lookahead in the current NRI covariates.",
            "Biological has no NRI natural-hazard score; its mapped hazard values are neutral 50 and only composite vulnerability/resilience structure can contribute.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
