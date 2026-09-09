#!/usr/bin/env python3
"""
Strict LFYO Flood-specific specialist for the $1M lower boundary.

Motivation
----------
Under the frozen low router, 25 true $1M-$50M declarations fall down into
$100K-$1M. Thirteen of those 25 are Floods. In the relevant Flood range there
are 30 true $100K-$1M Floods and 28 true $1M-$50M Floods, so a Flood-specific
boundary model is sufficiently balanced to test without inventing sub-bands.

Architecture
------------
- Stage 1 remains frozen: semantics RF for <100K vs >=100K, threshold selected
  by inner LFYO macro-F1.
- Non-Flood rows keep the frozen semantics ExtraTrees log-dollar regression.
- Only Flood rows that Stage 1 already routes to >=100K are passed to a Flood
  specialist for $100K-$1M vs $1M-$50M.
- Specialist threshold is selected only from inner-LFYO Flood predictions.
- No >=$50M router component is touched.
- Biological remains excluded/frozen.

Variants
--------
- base
- flood_current_log
- flood_current_rf
- flood_semantics_log
- flood_semantics_rf
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from mission_semantic_audit import (
    CURRENT_19, build_semantic_rollup, fetch_all_mission_assignments,
    normalize_master, normalize_model_frame, prep_pipeline,
)
from nonbio_all_ranges import funding_band, valid_cols
from nonbio_low_thresholds import (
    inner_binary_oof, choose_threshold, fit_binary,
    positive_proba, low_metrics,
)
from nonbio_cross_1m_rescue import (
    fit_base_reg, predict_reg_dollars,
)

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "nonbio_flood_1m_specialist"
OUT.mkdir(parents=True, exist_ok=True)

LOW_BANDS = ["0-100K", "100K-1M", "1M-50M"]


def fit_flood_binary(train, features, kind, seed):
    t = train[
        (train["incidentType"] == "Flood")
        & (train["target_clean"] >= 100_000)
        & (train["target_clean"] < 50_000_000)
    ].copy()
    y = (t["target_clean"] >= 1_000_000).astype(int)
    X = normalize_model_frame(t[features])

    if kind == "rf":
        model = RandomForestClassifier(
            n_estimators=900,
            random_state=seed,
            class_weight="balanced_subsample",
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=1,
        )
    else:
        model = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            C=0.5,
        )

    pipe = prep_pipeline(X, model)
    pipe.fit(X, y)
    return pipe


def flood_oof(outer_train, features, kind, seedbase):
    d = outer_train[
        (outer_train["incidentType"] == "Flood")
        & (outer_train["target_clean"] >= 100_000)
        & (outer_train["target_clean"] < 50_000_000)
    ].copy()

    rows = []

    for fy in sorted(d["fyDeclared"].astype(int).unique()):
        tr = d[d["fyDeclared"].astype(int) != fy].copy()
        te = d[d["fyDeclared"].astype(int) == fy].copy()

        ytr = (tr["target_clean"] >= 1_000_000).astype(int)
        yte = (te["target_clean"] >= 1_000_000).astype(int)

        if te.empty or ytr.nunique() < 2:
            continue

        model = fit_flood_binary(
            tr, features, kind, seedbase + int(fy)
        )
        pp = model.predict_proba(
            normalize_model_frame(te[features])
        )[:, 1]

        for (_, r), y, p in zip(te.iterrows(), yte, pp):
            rows.append({
                "disasterNumber": int(r["disasterNumber"]),
                "fy": int(fy),
                "y": int(y),
                "p": float(p),
            })

    return pd.DataFrame(rows)


def fit_outer_base(train, semfeat, outer_fy):
    low_train = train[
        train["target_clean"] < 50_000_000
    ].copy()

    s1_oof = inner_binary_oof(
        train, semfeat, stage=1, kind="rf",
        seedbase=110000 + outer_fy * 10,
    )
    s1_th, _ = choose_threshold(s1_oof, "macro_f1")

    y1 = (low_train["target_clean"] >= 100_000).astype(int)
    s1 = fit_binary(
        low_train, semfeat, y1, "rf",
        120000 + outer_fy,
    )

    reg = fit_base_reg(
        train, semfeat, 130000 + outer_fy
    )

    return s1, float(s1_th), reg


def base_predict(s1, s1_th, reg, dd, semfeat):
    if dd.empty:
        return (
            np.array([], dtype=object),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    p1 = positive_proba(s1, dd, semfeat)
    dollars = predict_reg_dollars(reg, dd, semfeat)

    pred = np.full(len(dd), "0-100K", dtype=object)
    idx = np.flatnonzero(p1 >= s1_th)
    if len(idx):
        pred[idx] = np.where(
            dollars[idx] >= 1_000_000,
            "1M-50M",
            "100K-1M",
        )

    return pred.astype(str), p1, dollars


def main():
    master = normalize_master(pd.read_excel(MASTER))
    master["target_clean"] = pd.to_numeric(
        master["totalObligatedFunding"], errors="coerce"
    ).fillna(0).clip(lower=0)
    master["actual_band"] = master["target_clean"].map(funding_band)

    ma = fetch_all_mission_assignments()
    sem, _ = build_semantic_rollup(master, ma)

    df = master.merge(
        sem, on="disasterNumber", how="left"
    )
    df = df[
        df["incidentType"] != "Biological"
    ].copy().reset_index(drop=True)

    current = valid_cols(df, CURRENT_19)
    semcols = valid_cols(
        df,
        [
            c for c in df.columns
            if (c.startswith("sem_") or c.startswith("ma_"))
            and not any(
                bad in c.lower()
                for bad in ["oblig", "fund", "cost", "amount", "dollar"]
            )
        ],
    )
    semfeat = list(dict.fromkeys(current + semcols))

    low = df[df["target_clean"] < 50_000_000].copy()

    variants = {
        "base": None,
        "flood_current_log": (current, "log"),
        "flood_current_rf": (current, "rf"),
        "flood_semantics_log": (semfeat, "log"),
        "flood_semantics_rf": (semfeat, "rf"),
    }

    rows = {k: [] for k in variants}
    threshold_rows = []

    for outer_fy in sorted(low["fyDeclared"].astype(int).unique()):
        train = low[
            low["fyDeclared"].astype(int) != outer_fy
        ].copy()
        test = low[
            low["fyDeclared"].astype(int) == outer_fy
        ].copy()

        s1, s1_th, reg = fit_outer_base(
            train, semfeat, int(outer_fy)
        )
        base, p1, dollars = base_predict(
            s1, s1_th, reg, test, semfeat
        )

        for (_, r), pred, pp, d in zip(
            test.iterrows(), base, p1, dollars
        ):
            rows["base"].append({
                "disasterNumber": int(r["disasterNumber"]),
                "fyDeclared": int(r["fyDeclared"]),
                "state": r["state"],
                "incidentType": r["incidentType"],
                "actual_band": r["actual_band"],
                "base_pred": str(pred),
                "final_pred": str(pred),
                "stage1_prob": float(pp),
                "base_reg_dollars": float(d),
            })

        for name, spec in variants.items():
            if name == "base":
                continue

            features, kind = spec
            oo = flood_oof(
                train, features, kind,
                210000 if kind == "log" else 220000,
            )
            th, diag = choose_threshold(
                oo, "macro_f1"
            )

            flood_train = train[
                (train["incidentType"] == "Flood")
                & (train["target_clean"] >= 100_000)
                & (train["target_clean"] < 50_000_000)
            ].copy()
            ytrain = (
                flood_train["target_clean"] >= 1_000_000
            ).astype(int)

            model = None
            if len(flood_train) >= 12 and ytrain.nunique() >= 2:
                model = fit_flood_binary(
                    flood_train, features, kind,
                    230000 + int(outer_fy),
                )

            final = base.copy()
            specialist_prob = np.full(
                len(test), np.nan, dtype=float
            )

            idx = np.flatnonzero(
                (p1 >= s1_th)
                & (test["incidentType"].to_numpy() == "Flood")
            )

            if len(idx) and model is not None:
                pp = model.predict_proba(
                    normalize_model_frame(
                        test.iloc[idx][features]
                    )
                )[:, 1]
                specialist_prob[idx] = pp
                final[idx] = np.where(
                    pp >= th,
                    "1M-50M",
                    "100K-1M",
                )

            for (_, r), bp, fp, sprob, d in zip(
                test.iterrows(),
                base,
                final,
                specialist_prob,
                dollars,
            ):
                rows[name].append({
                    "disasterNumber": int(r["disasterNumber"]),
                    "fyDeclared": int(r["fyDeclared"]),
                    "state": r["state"],
                    "incidentType": r["incidentType"],
                    "actual_band": r["actual_band"],
                    "base_pred": str(bp),
                    "final_pred": str(fp),
                    "specialist_prob": (
                        None
                        if not np.isfinite(sprob)
                        else float(sprob)
                    ),
                    "base_reg_dollars": float(d),
                })

            threshold_rows.append({
                "outer_fy": int(outer_fy),
                "variant": name,
                "threshold": float(th),
                "inner_sensitivity": diag.get("sensitivity"),
                "inner_specificity": diag.get("specificity"),
                "inner_score": diag.get("score"),
            })

    results = {}

    for name in variants:
        p = pd.DataFrame(rows[name])
        p.to_csv(
            OUT / f"{name}_predictions.csv",
            index=False,
        )
        q = p.rename(columns={"final_pred": "pred"})
        lm = low_metrics(q, "pred")

        flood = p[
            p["incidentType"] == "Flood"
        ].copy()
        fb = {}
        for b in ["100K-1M", "1M-50M"]:
            m = flood["actual_band"] == b
            n = int(m.sum())
            c = int(
                (flood.loc[m, "final_pred"] == b).sum()
            )
            fb[b] = {
                "correct": c,
                "total": n,
                "recall": c / n if n else None,
            }

        mid = p[p["actual_band"] == "1M-50M"]
        lowb = p[p["actual_band"] == "100K-1M"]

        results[name] = {
            "low_metrics": lm,
            "flood_boundary": fb,
            "mid_down": int(
                (mid["final_pred"] == "100K-1M").sum()
            ),
            "low_up": int(
                (lowb["final_pred"] == "1M-50M").sum()
            ),
        }

    pd.DataFrame(threshold_rows).to_csv(
        OUT / "thresholds.csv", index=False
    )

    (OUT / "summary.json").write_text(
        json.dumps({"results": results}, indent=2),
        encoding="utf-8",
    )

    md = [
        "# Flood-specific $1M boundary specialist audit",
        "",
        "- Strict outer LFYO.",
        "- Specialist applies only to Flood rows already routed >=$100K.",
        "- Threshold selected from inner-LFYO Flood predictions.",
        "",
        "| Variant | Low acc | Macro | 100K-1M | 1M-50M | Flood 100K-1M | Flood 1M-50M | Mid->low | Low->mid |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for name, r in results.items():
        lm = r["low_metrics"]
        pb = lm["per_band"]
        fb = r["flood_boundary"]

        md.append(
            f"| {name} | "
            f"{lm['accuracy']:.1%} | "
            f"{lm['macro_recall']:.1%} | "
            f"{pb['100K-1M']['correct']}/{pb['100K-1M']['total']} "
            f"({pb['100K-1M']['recall']:.1%}) | "
            f"{pb['1M-50M']['correct']}/{pb['1M-50M']['total']} "
            f"({pb['1M-50M']['recall']:.1%}) | "
            f"{fb['100K-1M']['correct']}/{fb['100K-1M']['total']} "
            f"({fb['100K-1M']['recall']:.1%}) | "
            f"{fb['1M-50M']['correct']}/{fb['1M-50M']['total']} "
            f"({fb['1M-50M']['recall']:.1%}) | "
            f"{r['mid_down']} | {r['low_up']} |"
        )

    (OUT / "summary.md").write_text(
        "\n".join(md), encoding="utf-8"
    )
    print("\n".join(md))


if __name__ == "__main__":
    main()
