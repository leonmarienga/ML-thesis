from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, confusion_matrix, balanced_accuracy_score, f1_score

import external_physical_severity_high_router_experiment as base
import cdc_biological_severity_signal_audit as cdc_audit

OUT = Path("generic_external_severity_high_router_results")
OUT.mkdir(exist_ok=True)

GENERIC_NUM = [
    "genericOverallPct",
    "genericFatalityPct",
    "genericAcutePct",
    "genericBurdenPct",
    "genericMagnitudePct",
]

FOCUS_IDS = [4486, 4515, 4480, 4485]


def pct_rank_value(values: pd.Series, target_idx) -> float:
    x = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if len(x) == 0:
        return 0.5
    ranks = x.rank(pct=True, method="average")
    v = ranks.loc[target_idx]
    return float(v) if pd.notna(v) else 0.5


def cdc_snapshot_percentiles(cdc: pd.DataFrame, declaration_date, target_state: str):
    if pd.isna(declaration_date):
        return None

    eligible = cdc[cdc["date"] <= declaration_date].copy()
    if eligible.empty:
        return None

    snap = (
        eligible.sort_values(["state_abbr", "date"])
        .groupby("state_abbr", as_index=False)
        .tail(1)
        .copy()
    )
    snap["population"] = snap["state_abbr"].map(base.POP)
    snap = snap[pd.to_numeric(snap["population"], errors="coerce").fillna(0) > 0].copy()
    if snap.empty or target_state not in set(snap["state_abbr"]):
        return None

    pop = pd.to_numeric(snap["population"], errors="coerce")
    cases = pd.to_numeric(snap["tot_cases"], errors="coerce").fillna(0).clip(lower=0)
    deaths = pd.to_numeric(snap["tot_death"], errors="coerce").fillna(0).clip(lower=0)
    new_cases = pd.to_numeric(snap["new_case"], errors="coerce").fillna(0).clip(lower=0)
    new_deaths = pd.to_numeric(snap["new_death"], errors="coerce").fillna(0).clip(lower=0)

    snap["case_rate"] = cases / pop * 1e5
    snap["death_rate"] = deaths / pop * 1e5
    snap["new_case_rate"] = new_cases / pop * 1e5
    snap["new_death_rate"] = new_deaths / pop * 1e5
    snap["severity_raw"] = (
        np.log1p(snap["case_rate"])
        + 1.5 * np.log1p(snap["death_rate"])
        + 0.75 * np.log1p(snap["new_case_rate"])
        + 0.75 * np.log1p(snap["new_death_rate"])
    )

    for c in ["case_rate", "death_rate", "new_case_rate", "new_death_rate", "severity_raw"]:
        snap[c + "_pct"] = snap[c].rank(pct=True, method="average")

    z = snap[snap["state_abbr"] == target_state].iloc[-1]
    return {
        "genericOverallPct": float(z["severity_raw_pct"]),
        "genericFatalityPct": float(z["death_rate_pct"]),
        "genericAcutePct": float((z["new_case_rate_pct"] + z["new_death_rate_pct"]) / 2.0),
        "genericBurdenPct": float(z["case_rate_pct"]),
        "genericMagnitudePct": float(z["case_rate_pct"]),
        "source": "CDC",
        "rawRecordCount": float(cases.loc[z.name]),
        "rawFatalityRate": float(z["death_rate"]),
        "rawAcuteRate": float((z["new_case_rate"] + z["new_death_rate"]) / 2.0),
        "rawMagnitude": float(z["case_rate"]),
        "referenceStates": int(len(snap)),
    }


def noaa_snapshot_percentiles(
    noaa: pd.DataFrame, declaration_date, incident_type: str, target_state: str
):
    if pd.isna(declaration_date):
        return None

    event_types = base.noaa_types(incident_type)
    if not event_types:
        return None

    lo = declaration_date - pd.Timedelta(days=21)
    n = noaa[
        noaa["EVENT_TYPE"].isin(event_types)
        & (noaa["END_DT"] <= declaration_date)
        & (noaa["END_DT"] >= lo)
    ].copy()

    # Build a complete state reference frame. A state with no qualifying event
    # receives zero burden rather than "missing"; therefore data-source
    # availability is never itself a predictor.
    states = []
    for abbr, pop in base.POP.items():
        state_name = base.STATE_NAME.get(abbr)
        if state_name:
            states.append({"state_abbr": abbr, "STATE": state_name, "population": float(pop)})
    ref = pd.DataFrame(states)
    if target_state not in set(ref["state_abbr"]):
        return None

    if n.empty:
        agg = pd.DataFrame(columns=["STATE", "record_count", "deaths", "injuries", "magnitude"])
    else:
        n["deaths_total"] = pd.to_numeric(
            n["DEATHS_DIRECT"], errors="coerce"
        ).fillna(0) + pd.to_numeric(n["DEATHS_INDIRECT"], errors="coerce").fillna(0)
        n["injuries_total"] = pd.to_numeric(
            n["INJURIES_DIRECT"], errors="coerce"
        ).fillna(0) + pd.to_numeric(n["INJURIES_INDIRECT"], errors="coerce").fillna(0)
        agg = (
            n.groupby("STATE", as_index=False)
            .agg(
                record_count=("EVENT_TYPE", "size"),
                deaths=("deaths_total", "sum"),
                injuries=("injuries_total", "sum"),
                magnitude=("MAGNITUDE", "max"),
            )
        )

    ref = ref.merge(agg, on="STATE", how="left")
    for c in ["record_count", "deaths", "injuries", "magnitude"]:
        ref[c] = pd.to_numeric(ref[c], errors="coerce").fillna(0).clip(lower=0)

    ref["fatality_rate"] = ref["deaths"] / ref["population"] * 1e6
    ref["acute_rate"] = ref["injuries"] / ref["population"] * 1e6
    ref["severity_raw"] = (
        np.log1p(ref["record_count"])
        + 1.25 * np.log1p(ref["acute_rate"])
        + 2.0 * np.log1p(ref["fatality_rate"])
        + 0.25 * np.log1p(ref["magnitude"])
    )

    for c in ["record_count", "fatality_rate", "acute_rate", "magnitude", "severity_raw"]:
        ref[c + "_pct"] = ref[c].rank(pct=True, method="average")

    z = ref[ref["state_abbr"] == target_state].iloc[-1]
    return {
        "genericOverallPct": float(z["severity_raw_pct"]),
        "genericFatalityPct": float(z["fatality_rate_pct"]),
        "genericAcutePct": float(z["acute_rate_pct"]),
        "genericBurdenPct": float(z["record_count_pct"]),
        "genericMagnitudePct": float(z["magnitude_pct"]),
        "source": "NOAA",
        "rawRecordCount": float(z["record_count"]),
        "rawFatalityRate": float(z["fatality_rate"]),
        "rawAcuteRate": float(z["acute_rate"]),
        "rawMagnitude": float(z["magnitude"]),
        "referenceStates": int(len(ref)),
    }


def build_generic_external(source: pd.DataFrame) -> pd.DataFrame:
    forbidden = {"target", "band", "totalObligatedFunding"}
    overlap = forbidden.intersection(source.columns)
    if overlap:
        raise RuntimeError(f"Target-derived columns reached external feature builder: {sorted(overlap)}")

    years = sorted(
        pd.to_datetime(source["effectiveDeclarationDate"], errors="coerce")
        .dropna()
        .dt.year.astype(int)
        .unique()
        .tolist()
    )
    years = [y for y in years if 2010 <= y <= 2024]
    noaa = base.fetch_noaa(years)
    cdc = cdc_audit.fetch_cdc()

    rows = []
    for _, r in source.iterrows():
        state = str(r["state"])
        incident = str(r["incidentType"])
        decl = pd.to_datetime(r["effectiveDeclarationDate"], errors="coerce")

        feat = None
        if incident.upper().strip() == "BIOLOGICAL":
            feat = cdc_snapshot_percentiles(cdc, decl, state)
        else:
            feat = noaa_snapshot_percentiles(noaa, decl, incident, state)

        if feat is None:
            feat = {
                "genericOverallPct": 0.5,
                "genericFatalityPct": 0.5,
                "genericAcutePct": 0.5,
                "genericBurdenPct": 0.5,
                "genericMagnitudePct": 0.5,
                "source": "NEUTRAL_UNSUPPORTED",
                "rawRecordCount": np.nan,
                "rawFatalityRate": np.nan,
                "rawAcuteRate": np.nan,
                "rawMagnitude": np.nan,
                "referenceStates": 0,
            }

        rows.append({"disasterNumber": int(r["disasterNumber"]), **feat})

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "generic_external_severity_all_disasters.csv", index=False)
    return out


def high_metrics(y, pred):
    rec = recall_score(y, pred, labels=[3, 4, 5], average=None, zero_division=0)
    cm = confusion_matrix(y, pred, labels=[3, 4, 5])
    return {
        "r3": float(rec[0]),
        "r4": float(rec[1]),
        "r5": float(rec[2]),
        "correct3": int(cm[0, 0]),
        "correct4": int(cm[1, 1]),
        "correct5": int(cm[2, 2]),
        "min_recall": float(rec.min()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "pass_all80": bool(
            cm[0, 0] >= 32 and
            cm[1, 1] >= 8 and
            cm[2, 2] >= 6
        ),
    }


def evaluate(d: pd.DataFrame, ext: pd.DataFrame):
    d = d.merge(ext, on="disasterNumber", how="left")
    for c in GENERIC_NUM:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.5).clip(0, 1)

    high = d[d["band"].isin([3, 4, 5])].copy().reset_index(drop=True)
    years = sorted(high["fyDeclared"].astype(int).unique())

    sets = {
        "baseline": ([c for c in base.BASE_NUM if c in high.columns],
                     [c for c in base.BASE_CAT if c in high.columns]),
        "baseline_plus_generic_external": (
            [c for c in base.BASE_NUM if c in high.columns] + GENERIC_NUM,
            [c for c in base.BASE_CAT if c in high.columns],
        ),
    }

    result_rows = []
    prediction_rows = []
    pair_rows = []

    for feature_set, (nums, cats) in sets.items():
        for family in ["cat", "et"]:
            pred = np.full(len(high), -99, dtype=int)
            for year in years:
                tr = high["fyDeclared"].astype(int) != year
                te = ~tr
                Xtr = base.prep(high.loc[tr], nums, cats)
                Xte = base.prep(high.loc[te], nums, cats)
                ytr = high.loc[tr, "band"].astype(int)

                if family == "cat":
                    m = base.fit_cat(Xtr, ytr, Xte, cats)
                    pred[te] = np.asarray(m.predict(Xte)).reshape(-1).astype(int)
                else:
                    m, A, B = base.fit_et(Xtr, ytr, Xte, cats)
                    pred[te] = m.predict(B).astype(int)

            met = high_metrics(high["band"].astype(int).to_numpy(), pred)
            result_rows.append({
                "feature_set": feature_set,
                "model": family,
                **{k: v for k, v in met.items() if k != "confusion_matrix"},
                "confusion_matrix": json.dumps(met["confusion_matrix"]),
            })
            q = high[
                ["disasterNumber", "state", "fyDeclared", "incidentType", "target", "band", "source"]
            ].copy()
            q["feature_set"] = feature_set
            q["model"] = family
            q["predicted_band"] = pred
            prediction_rows.append(q)

        # Pairwise local separability diagnostic. Operating thresholds are
        # development-selected on combined OOF predictions and are not a final claim.
        for lo, hi in [(3, 4), (4, 5), (3, 5)]:
            mask = high["band"].isin([lo, hi]).to_numpy()
            p = np.full(len(high), np.nan)
            for year in sorted(high.loc[mask, "fyDeclared"].astype(int).unique()):
                tr = mask & (high["fyDeclared"].astype(int).to_numpy() != year)
                te = mask & (high["fyDeclared"].astype(int).to_numpy() == year)
                Xtr = base.prep(high.loc[tr], nums, cats)
                Xte = base.prep(high.loc[te], nums, cats)
                ytr = (high.loc[tr, "band"] == hi).astype(int)
                m = base.fit_cat(Xtr, ytr, Xte, cats)
                cls = list(m.classes_)
                p[te] = m.predict_proba(Xte)[:, cls.index(1)]

            yy = (high.loc[mask, "band"] == hi).astype(int).to_numpy()
            met = base.op_point(yy, p[mask])
            pair_rows.append({
                "feature_set": feature_set,
                "pair": f"{lo}vs{hi}",
                "model": "cat",
                **met,
            })

    result_df = pd.DataFrame(result_rows).sort_values(
        ["min_recall", "balanced_accuracy"], ascending=False
    )
    result_df.to_csv(OUT / "multiclass_results.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        OUT / "multiclass_predictions.csv", index=False
    )
    pair_df = pd.DataFrame(pair_rows).sort_values(
        ["pair", "min_recall"], ascending=[True, False]
    )
    pair_df.to_csv(OUT / "pairwise_results.csv", index=False)

    focus_cols = [
        "disasterNumber", "state", "fyDeclared", "incidentType", "target", "band", "source"
    ] + GENERIC_NUM + [
        "rawRecordCount", "rawFatalityRate", "rawAcuteRate", "rawMagnitude", "referenceStates"
    ]
    focus = high[high["disasterNumber"].isin(FOCUS_IDS)][focus_cols].copy()
    focus.to_csv(OUT / "focus_2020_biological_generic_features.csv", index=False)

    coverage = (
        d.groupby(["incidentType", "source"])
        .size().reset_index(name="n")
        .sort_values(["incidentType", "source"])
    )
    coverage.to_csv(OUT / "coverage_all_disasters.csv", index=False)

    high_coverage = (
        high.groupby(["incidentType", "source"])
        .size().reset_index(name="n")
        .sort_values(["incidentType", "source"])
    )
    high_coverage.to_csv(OUT / "coverage_high_disasters.csv", index=False)

    summary = {
        "high_counts": {str(b): int((high["band"] == b).sum()) for b in [3, 4, 5]},
        "acceptance_counts": {"3": 32, "4": 8, "5": 6},
        "all_disaster_external_source_counts": d["source"].value_counts().to_dict(),
        "high_external_source_counts": high["source"].value_counts().to_dict(),
        "multiclass": result_rows,
        "pairwise": pair_rows,
        "focus_cases": focus.to_dict("records"),
        "guardrails": [
            "The external feature builder receives no funding target, funding band, obligation amount, or funding-derived field.",
            "External severity is generated for every master disaster, not only disasters/events selected by a high funding target.",
            "External source/coverage identity is saved only for audit and is excluded from model predictors.",
            "Unsupported hazards receive neutral 0.5 generic percentiles; missingness/availability is not a predictor.",
            "CDC observations are restricted to dates on or before each FEMA declaration and are normalized against contemporaneous states.",
            "NOAA events are restricted to the 21 days ending on each FEMA declaration and are normalized against a complete state reference frame including zero-event states.",
            "No NOAA property-damage, crop-damage, FEMA funding, or obligation-dollar field is loaded as a predictor.",
            "Outer predictions use strict leave-fiscal-year-out training. Pairwise operating thresholds remain development diagnostics only.",
            "Any configuration that clears all integer recall floors must be rerun with nested outer/inner fiscal-year model/threshold selection before a final thesis claim."
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str), flush=True)


def main():
    d = base.load_master()

    # Critical target-independence boundary: the external builder receives only
    # deployment-time descriptors and cannot inspect the true funding outcome.
    source = d[
        ["disasterNumber", "state", "incidentType", "population2010", "effectiveDeclarationDate"]
    ].copy()

    ext = build_generic_external(source)
    evaluate(d, ext)


if __name__ == "__main__":
    main()
