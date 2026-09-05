from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr
from sklearn.metrics import (
    recall_score, confusion_matrix, balanced_accuracy_score,
    f1_score, roc_auc_score
)

import external_physical_severity_high_router_experiment as base

OUT = Path("political_context_high_router_results")
OUT.mkdir(exist_ok=True)

PRES_URL = "https://raw.githubusercontent.com/fivethirtyeight/election-results/refs/heads/main/election_results_presidential.csv"
GOV_URL = "https://raw.githubusercontent.com/fivethirtyeight/election-results/refs/heads/main/election_results_gubernatorial.csv"
RACES_URL = "https://raw.githubusercontent.com/fivethirtyeight/election-results/refs/heads/main/races.csv"

POL_NUM = [
    "polDaysToNextPresElectionLog",
    "polDaysSinceLastPresElectionLog",
    "polPresElectionYear",
    "polPrevPresTwoPartyMarginAbs",
    "polPrevPresSwingness",
    "polPrevPresWinnerAlignedWithCurrentPresident",
    "polPrevGovMarginAbs",
    "polPrevGovSwingness",
    "polPrevGovWinnerAlignedWithCurrentPresident",
    "polDaysSinceLastGovElectionLog",
    "polTerritoryNoPresidentialVote",
    "polPresElectionProximity180",
]
POL_CAT = [
    "polCurrentPresidentParty",
    "polPrevPresWinnerParty",
    "polPrevGovWinnerParty",
]

SAFE_BASE_CAT = [c for c in base.BASE_CAT if c != "eventScale"]


def num(x):
    return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)


def current_president_party(dt: pd.Timestamp) -> str:
    # Study period is 2010-2024. Party is an observable fact at declaration time.
    if pd.isna(dt):
        return "UNK"
    if dt < pd.Timestamp("2017-01-20"):
        return "DEM"
    if dt < pd.Timestamp("2021-01-20"):
        return "REP"
    if dt < pd.Timestamp("2025-01-20"):
        return "DEM"
    return "UNK"


def presidential_election_day(year: int) -> pd.Timestamp:
    # Federal statute: Tuesday after the first Monday in November.
    d = pd.Timestamp(year=year, month=11, day=1)
    while d.weekday() != 0:  # Monday
        d += pd.Timedelta(days=1)
    return d + pd.Timedelta(days=1)


def next_presidential_election(dt: pd.Timestamp) -> pd.Timestamp:
    y = int(dt.year)
    cycle = y + ((4 - (y % 4)) % 4)
    candidate = presidential_election_day(cycle)
    if candidate < dt:
        candidate = presidential_election_day(cycle + 4)
    return candidate


def fetch_election_data():
    headers = {"User-Agent": "ML-thesis-research/1.0"}
    pres = pd.read_csv(PRES_URL)
    gov = pd.read_csv(GOV_URL)
    races = pd.read_csv(RACES_URL)

    for d in [pres, gov]:
        d["race_id"] = num(d["race_id"]).astype("Int64")
        d["votes"] = num(d["votes"])
        d["percent"] = num(d["percent"])
        d["state_abbrev"] = d["state_abbrev"].astype(str).str.upper().str.strip()
        d["ballot_party"] = d["ballot_party"].astype(str).str.upper().str.strip()
        d["stage"] = d["stage"].astype(str).str.lower().str.strip()
        d["winner"] = d["winner"].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)

    races["id"] = num(races["id"]).astype("Int64")
    races["date"] = pd.to_datetime(races["date"], errors="coerce").dt.tz_localize(None)
    races["stage"] = races["stage"].astype(str).str.lower().str.strip()
    races["state_abbrev"] = races["state_abbrev"].astype(str).str.upper().str.strip()

    date_map = races[["id", "date"]].drop_duplicates("id").rename(columns={"id": "race_id"})
    pres = pres.merge(date_map, on="race_id", how="left")
    gov = gov.merge(date_map, on="race_id", how="left")

    # Persist source snapshots for reproducibility.
    pres.to_csv(OUT / "fivethirtyeight_presidential_results_snapshot.csv", index=False)
    gov.to_csv(OUT / "fivethirtyeight_gubernatorial_results_snapshot.csv", index=False)
    races.to_csv(OUT / "fivethirtyeight_races_snapshot.csv", index=False)

    return pres, gov, races


def two_party_election_summary(results: pd.DataFrame, state: str, declaration_date: pd.Timestamp, office: str):
    z = results[
        (results["state_abbrev"] == state)
        & (results["date"].notna())
        & (results["date"] < declaration_date)
    ].copy()

    if office == "pres":
        z = z[z["stage"].eq("general")]
    else:
        # Some states elect governors after a runoff; use the latest completed
        # statewide governor election stage available before declaration.
        z = z[z["stage"].isin(["general", "runoff", "jungle primary"])]

    if z.empty:
        return None

    # Pick latest completed election date, then aggregate DEM/REP ballot lines.
    latest_date = z["date"].max()
    q = z[z["date"] == latest_date].copy()

    # Some candidates can appear on multiple ballot-party lines.
    by_party = q.groupby("ballot_party", as_index=False)["votes"].sum()
    dem = float(by_party.loc[by_party["ballot_party"].eq("DEM"), "votes"].sum())
    rep = float(by_party.loc[by_party["ballot_party"].eq("REP"), "votes"].sum())
    denom = dem + rep

    winner_rows = q[q["winner"] == True]
    if len(winner_rows):
        winner_party = str(winner_rows.iloc[0]["ballot_party"]).upper()
        if winner_party not in {"DEM", "REP"}:
            # If winner line is nonstandard/fusion, infer from two-party totals.
            winner_party = "DEM" if dem > rep else "REP" if rep > dem else "OTHER"
    else:
        winner_party = "DEM" if dem > rep else "REP" if rep > dem else "OTHER"

    if denom <= 0:
        return {
            "date": latest_date,
            "margin_abs": np.nan,
            "swingness": np.nan,
            "winner_party": winner_party,
        }

    margin = abs(dem - rep) / denom
    return {
        "date": latest_date,
        "margin_abs": float(margin),
        "swingness": float(1.0 - margin),
        "winner_party": winner_party,
    }


def build_political_features(source: pd.DataFrame) -> pd.DataFrame:
    forbidden = {
        "target", "band", "totalObligatedFunding", "eventScale",
        "fundingPerMission", "fundingPerAgency", "fundingPerDay",
    }
    bad = forbidden.intersection(source.columns)
    if bad:
        raise RuntimeError(f"Forbidden fields reached political builder: {sorted(bad)}")

    pres, gov, _ = fetch_election_data()

    rows = []
    no_pres_vote = {"PR", "VI", "GU", "AS", "MP"}

    for _, r in source.iterrows():
        dn = int(r["disasterNumber"])
        st = str(r["state"]).upper().strip()
        dt = pd.to_datetime(r["effectiveDeclarationDate"], errors="coerce")
        president_party = current_president_party(dt)

        if pd.isna(dt):
            rows.append({
                "disasterNumber": dn,
                **{c: np.nan for c in POL_NUM},
                "polCurrentPresidentParty": president_party,
                "polPrevPresWinnerParty": "UNK",
                "polPrevGovWinnerParty": "UNK",
            })
            continue

        next_pres = next_presidential_election(dt)
        days_to_pres = max((next_pres - dt).days, 0)
        # Last *calendar* presidential election date, independent of result availability.
        prev_cycle = int(next_pres.year) - 4 if next_pres > dt else int(next_pres.year)
        prev_pres_day = presidential_election_day(prev_cycle)
        if prev_pres_day >= dt:
            prev_pres_day = presidential_election_day(prev_cycle - 4)
        days_since_pres = max((dt - prev_pres_day).days, 0)

        p = None if st in no_pres_vote else two_party_election_summary(pres, st, dt, "pres")
        g = two_party_election_summary(gov, st, dt, "gov")

        pres_margin = float(p["margin_abs"]) if p and pd.notna(p["margin_abs"]) else 0.5
        pres_swing = float(p["swingness"]) if p and pd.notna(p["swingness"]) else 0.5
        pres_winner = str(p["winner_party"]) if p else ("NO_VOTE" if st in no_pres_vote else "UNK")

        gov_margin = float(g["margin_abs"]) if g and pd.notna(g["margin_abs"]) else 0.5
        gov_swing = float(g["swingness"]) if g and pd.notna(g["swingness"]) else 0.5
        gov_winner = str(g["winner_party"]) if g else "UNK"
        gov_days = max((dt - g["date"]).days, 0) if g else 1460

        rows.append({
            "disasterNumber": dn,
            "polDaysToNextPresElectionLog": float(np.log1p(days_to_pres)),
            "polDaysSinceLastPresElectionLog": float(np.log1p(days_since_pres)),
            "polPresElectionYear": int(dt.year % 4 == 0),
            "polPrevPresTwoPartyMarginAbs": pres_margin,
            "polPrevPresSwingness": pres_swing,
            "polPrevPresWinnerAlignedWithCurrentPresident": int(pres_winner == president_party),
            "polPrevGovMarginAbs": gov_margin,
            "polPrevGovSwingness": gov_swing,
            "polPrevGovWinnerAlignedWithCurrentPresident": int(gov_winner == president_party),
            "polDaysSinceLastGovElectionLog": float(np.log1p(gov_days)),
            "polTerritoryNoPresidentialVote": int(st in no_pres_vote),
            "polPresElectionProximity180": float(np.exp(-days_to_pres / 180.0)),
            "polCurrentPresidentParty": president_party,
            "polPrevPresWinnerParty": pres_winner,
            "polPrevGovWinnerParty": gov_winner,
            "polPrevPresElectionDateAudit": p["date"] if p else pd.NaT,
            "polPrevGovElectionDateAudit": g["date"] if g else pd.NaT,
            "polDeclarationDateAudit": dt,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "political_context_all_disasters.csv", index=False)
    return out


def model_metrics(y, pred):
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
        classes = list(m.classes_)
        p[te] = m.predict_proba(Xte)[:, classes.index(1)]

    valid = mask & np.isfinite(p)
    yy = (high.loc[valid, "band"] == hi).astype(int).to_numpy()
    pp = p[valid]
    op = base.op_point(yy, pp) if len(np.unique(yy)) == 2 else {}
    return p, op


def run_subset(d: pd.DataFrame, subset_name: str, mask: pd.Series):
    high = d[mask & d["band"].isin([3,4,5])].copy().reset_index(drop=True)

    base_num = [c for c in base.BASE_NUM if c in high.columns]
    base_cat = [c for c in SAFE_BASE_CAT if c in high.columns]
    pol_num = [c for c in POL_NUM if c in high.columns]
    pol_cat = [c for c in POL_CAT if c in high.columns]

    variants = {
        "baseline_safe": (base_num, base_cat),
        "baseline_plus_political": (base_num + pol_num, base_cat + pol_cat),
        "political_only": (pol_num, pol_cat + ["incidentType"]),
    }

    rows = []
    pred_rows = []
    pair_rows = []

    years = sorted(high["fyDeclared"].astype(int).unique())

    for name, (nums, cats) in variants.items():
        pred = np.full(len(high), -99, dtype=int)

        for year in years:
            tr = high["fyDeclared"].astype(int) != year
            te = ~tr
            if te.sum() == 0:
                continue
            Xtr = base.prep(high.loc[tr], nums, cats)
            Xte = base.prep(high.loc[te], nums, cats)
            ytr = high.loc[tr, "band"].astype(int)
            m = base.fit_cat(Xtr, ytr, Xte, cats)
            pred[te] = np.asarray(m.predict(Xte)).reshape(-1).astype(int)

        met = model_metrics(high["band"].astype(int).to_numpy(), pred)
        rows.append({
            "subset": subset_name,
            "variant": name,
            **{k:v for k,v in met.items() if k != "confusion_matrix"},
            "confusion_matrix": json.dumps(met["confusion_matrix"]),
        })

        q = high[
            ["disasterNumber","state","fyDeclared","incidentType","target","band"]
            + POL_NUM + POL_CAT
        ].copy()
        q["subset"] = subset_name
        q["variant"] = name
        q["predicted_band"] = pred
        pred_rows.append(q)

        for lo, hi in [(3,4),(4,5)]:
            p, op = pair_oof(high, lo, hi, nums, cats)
            pair_rows.append({
                "subset": subset_name,
                "variant": name,
                "pair": f"{lo}vs{hi}",
                **op,
            })

    return high, rows, pred_rows, pair_rows


def signal_audit(high):
    rows = []
    ylog = np.log1p(num(high["target"]).to_numpy(float))
    for c in POL_NUM:
        x = num(high[c])
        m = x.notna() & np.isfinite(ylog)
        if m.sum() >= 6 and x[m].nunique() > 1:
            rho, p = spearmanr(x[m], ylog[m])
            rows.append({
                "feature": c,
                "n": int(m.sum()),
                "spearman_rho_log_funding": float(rho),
                "p_value": float(p),
            })
    return pd.DataFrame(rows).sort_values(
        "spearman_rho_log_funding",
        key=lambda s: s.abs(),
        ascending=False
    ) if rows else pd.DataFrame()


def main():
    d = base.load_master()

    source = d[[
        "disasterNumber","state","incidentType","effectiveDeclarationDate"
    ]].copy()

    pol = build_political_features(source)
    d = d.merge(pol, on="disasterNumber", how="left")

    for c in POL_NUM:
        d[c] = num(d[c]).fillna(0.5)
    for c in POL_CAT:
        d[c] = d[c].astype(str).fillna("UNK")

    all_mask = pd.Series(True, index=d.index)
    nonbio_mask = ~d["incidentType"].astype(str).str.upper().str.strip().eq("BIOLOGICAL")

    all_high, rows1, preds1, pairs1 = run_subset(d, "all_high", all_mask)
    nonbio_high, rows2, preds2, pairs2 = run_subset(d, "non_biological_high", nonbio_mask)

    results = pd.DataFrame(rows1 + rows2)
    results.to_csv(OUT / "multiclass_results.csv", index=False)
    pd.concat(preds1 + preds2, ignore_index=True).to_csv(
        OUT / "multiclass_predictions.csv", index=False
    )
    pd.DataFrame(pairs1 + pairs2).to_csv(OUT / "pairwise_results.csv", index=False)

    assoc_all = signal_audit(all_high)
    assoc_nonbio = signal_audit(nonbio_high)
    assoc_all.to_csv(OUT / "political_signal_all_high.csv", index=False)
    assoc_nonbio.to_csv(OUT / "political_signal_non_biological_high.csv", index=False)

    focus_ids = [4339,4344,4671,4086,4277,4673,4611,4480,4482,4485,4486,4489,4515]
    focus_cols = [
        "disasterNumber","state","fyDeclared","incidentType","target","band",
        "effectiveDeclarationDate"
    ] + POL_NUM + POL_CAT
    focus = d[d["disasterNumber"].isin(focus_ids)][focus_cols].sort_values("disasterNumber")
    focus.to_csv(OUT / "focus_political_context_cases.csv", index=False)

    summary = {
        "purpose": "Test declaration-time political context as a target-free high-router feature family.",
        "sources": {
            "presidential_results": PRES_URL,
            "gubernatorial_results": GOV_URL,
            "race_dates": RACES_URL,
        },
        "all_high_counts": {str(b): int((all_high.band==b).sum()) for b in [3,4,5]},
        "non_biological_high_counts": {str(b): int((nonbio_high.band==b).sum()) for b in [3,4,5]},
        "multiclass": rows1 + rows2,
        "pairwise": pairs1 + pairs2,
        "best_political_correlations_all_high": assoc_all.head(12).to_dict("records") if len(assoc_all) else [],
        "best_political_correlations_non_biological": assoc_nonbio.head(12).to_dict("records") if len(assoc_nonbio) else [],
        "focus_cases": focus.to_dict("records"),
        "guardrails": [
            "Every political feature is determined using only election outcomes dated before the FEMA declaration, plus statutory election-calendar timing known in advance.",
            "No funding amount, funding band, obligation-derived feature, or eventScale enters the political feature builder.",
            "Current-president party is determined by inauguration date, not inferred from the disaster outcome.",
            "For presidential competitiveness, only the last completed presidential election before declaration is used.",
            "For gubernatorial context, the feature is explicitly the last completed gubernatorial election winner/competitiveness, not an unverified claim about the sitting governor in succession cases.",
            "Puerto Rico and other territories receive an explicit no-presidential-vote indicator rather than fabricated presidential election margins.",
            "Model evaluation is strict leave-fiscal-year-out.",
            "Pairwise thresholds are development OOF diagnostics only. Any apparent pass must still undergo proper nested outer/inner fiscal-year selection."
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
