#!/usr/bin/env python3
"""
Strict nested-LFYO one-way Flood rescue at the $1M boundary.

The prior Flood specialist had useful signal but replaced the boundary in both
directions, gaining 7 Flood mid cases while losing 7 Flood low cases.

This audit is conservative:
- frozen base low router remains authoritative;
- specialist is invoked ONLY for Floods base predicts $100K-$1M;
- it may only promote to $1M-$50M;
- it can never demote a base $1M-$50M prediction.

Thresholds are selected from inner-LFYO held-out predictions.
Variants:
- base
- rescue_macro: maximize 3-band low macro recall
- rescue_guard70: same, but preserve >=70% Flood $100K-$1M recall in inner OOF
- rescue_guard75
- rescue_guard80

No >=$50M component is touched. Biological remains excluded/frozen.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
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
OUT = ROOT / "audit_outputs" / "nonbio_flood_1m_rescue"
OUT.mkdir(parents=True, exist_ok=True)

LOW_BANDS = ["0-100K", "100K-1M", "1M-50M"]


def fit_flood_log(train, features, seed):
    t = train[
        (train["incidentType"] == "Flood")
        & (train["target_clean"] >= 100_000)
        & (train["target_clean"] < 50_000_000)
    ].copy()
    y = (t["target_clean"] >= 1_000_000).astype(int)
    X = normalize_model_frame(t[features])
    model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        C=0.5,
    )
    pipe = prep_pipeline(X, model)
    pipe.fit(X, y)
    return pipe


def fit_outer_base(train, semfeat, outer_fy):
    low_train = train[train["target_clean"] < 50_000_000].copy()

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


def inner_base_and_flood_scores(outer_train, semfeat, flood_features, outer_fy):
    rows = []
    years = sorted(outer_train["fyDeclared"].astype(int).unique())

    for fy in years:
        tr = outer_train[
            outer_train["fyDeclared"].astype(int) != fy
        ].copy()
        te = outer_train[
            (outer_train["fyDeclared"].astype(int) == fy)
            & (outer_train["target_clean"] < 50_000_000)
        ].copy()
        if te.empty:
            continue

        s1, s1_th, reg = fit_outer_base(
            tr, semfeat, 300000 + int(outer_fy) * 10 + int(fy)
        )
        base, _, dollars = base_predict(
            s1, s1_th, reg, te, semfeat
        )

        flood_train = tr[
            (tr["incidentType"] == "Flood")
            & (tr["target_clean"] >= 100_000)
            & (tr["target_clean"] < 50_000_000)
        ].copy()
        ytr = (flood_train["target_clean"] >= 1_000_000).astype(int)

        fmodel = None
        if len(flood_train) >= 12 and ytr.nunique() >= 2:
            fmodel = fit_flood_log(
                flood_train,
                flood_features,
                400000 + int(outer_fy) * 10 + int(fy),
            )

        probs = np.full(len(te), np.nan, dtype=float)
        idx = np.flatnonzero(
            (base == "100K-1M")
            & (te["incidentType"].to_numpy() == "Flood")
        )
        if len(idx) and fmodel is not None:
            probs[idx] = fmodel.predict_proba(
                normalize_model_frame(
                    te.iloc[idx][flood_features]
                )
            )[:, 1]

        for (_, r), bp, p, d in zip(
            te.iterrows(), base, probs, dollars
        ):
            rows.append({
                "disasterNumber": int(r["disasterNumber"]),
                "fy": int(fy),
                "incidentType": r["incidentType"],
                "actual_band": r["actual_band"],
                "base_pred": str(bp),
                "flood_prob": None if not np.isfinite(p) else float(p),
                "base_reg_dollars": float(d),
            })

    return pd.DataFrame(rows)


def score_variant(inner, threshold):
    d = inner.copy()
    d["pred"] = d["base_pred"]
    m = (
        (d["base_pred"] == "100K-1M")
        & (d["incidentType"] == "Flood")
        & d["flood_prob"].notna()
        & (d["flood_prob"] >= threshold)
    )
    d.loc[m, "pred"] = "1M-50M"

    recs = []
    for b in LOW_BANDS:
        z = d["actual_band"] == b
        if z.any():
            recs.append(float((d.loc[z, "pred"] == b).mean()))
    macro = float(np.mean(recs))
    accuracy = float((d["pred"] == d["actual_band"]).mean())

    flood_low = d[
        (d["incidentType"] == "Flood")
        & (d["actual_band"] == "100K-1M")
    ]
    flood_mid = d[
        (d["incidentType"] == "Flood")
        & (d["actual_band"] == "1M-50M")
    ]
    flr = (
        float((flood_low["pred"] == "100K-1M").mean())
        if len(flood_low) else np.nan
    )
    fmr = (
        float((flood_mid["pred"] == "1M-50M").mean())
        if len(flood_mid) else np.nan
    )

    return {
        "macro": macro,
        "accuracy": accuracy,
        "flood_low_recall": flr,
        "flood_mid_recall": fmr,
        "promotions": int(m.sum()),
    }


def choose_threshold(inner, guard=None):
    probs = pd.to_numeric(
        inner["flood_prob"], errors="coerce"
    ).dropna().to_numpy(float)

    if len(probs) == 0:
        return 1.0, score_variant(inner, 1.0)

    grid = np.unique(np.r_[
        0.05,
        np.arange(0.10, 0.96, 0.02),
        0.99,
        probs,
    ])
    best = None

    for th in grid:
        m = score_variant(inner, float(th))
        if (
            guard is not None
            and np.isfinite(m["flood_low_recall"])
            and m["flood_low_recall"] < guard
        ):
            continue

        key = (
            m["macro"],
            m["accuracy"],
            float(th),  # conservative tie-break
        )
        if best is None or key > best[0]:
            best = (key, float(th), m)

    if best is None:
        return 1.0, score_variant(inner, 1.0)
    return best[1], best[2]


def main():
    master = normalize_master(pd.read_excel(MASTER))
    master["target_clean"] = pd.to_numeric(
        master["totalObligatedFunding"], errors="coerce"
    ).fillna(0).clip(lower=0)
    master["actual_band"] = master["target_clean"].map(funding_band)

    ma = fetch_all_mission_assignments()
    sem, _ = build_semantic_rollup(master, ma)

    df = master.merge(sem, on="disasterNumber", how="left")
    df = df[df["incidentType"] != "Biological"].copy().reset_index(drop=True)

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
    flood_features = current

    low = df[df["target_clean"] < 50_000_000].copy()

    variants = {
        "base": None,
        "rescue_macro": None,
        "rescue_guard70": 0.70,
        "rescue_guard75": 0.75,
        "rescue_guard80": 0.80,
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

        inner = inner_base_and_flood_scores(
            train, semfeat, flood_features, int(outer_fy)
        )

        selected = {"base": (1.0, score_variant(inner, 1.0))}
        for name, guard in variants.items():
            if name == "base":
                continue
            selected[name] = choose_threshold(inner, guard)

        s1, s1_th, reg = fit_outer_base(
            train, semfeat, int(outer_fy)
        )
        base, _, dollars = base_predict(
            s1, s1_th, reg, test, semfeat
        )

        flood_train = train[
            (train["incidentType"] == "Flood")
            & (train["target_clean"] >= 100_000)
            & (train["target_clean"] < 50_000_000)
        ].copy()
        ytr = (flood_train["target_clean"] >= 1_000_000).astype(int)

        fmodel = None
        if len(flood_train) >= 12 and ytr.nunique() >= 2:
            fmodel = fit_flood_log(
                flood_train,
                flood_features,
                500000 + int(outer_fy),
            )

        probs = np.full(len(test), np.nan, dtype=float)
        idx = np.flatnonzero(
            (base == "100K-1M")
            & (test["incidentType"].to_numpy() == "Flood")
        )
        if len(idx) and fmodel is not None:
            probs[idx] = fmodel.predict_proba(
                normalize_model_frame(
                    test.iloc[idx][flood_features]
                )
            )[:, 1]

        for name in variants:
            th, diag = selected[name]
            final = base.copy()

            if name != "base":
                promote = (
                    (base == "100K-1M")
                    & (test["incidentType"].to_numpy() == "Flood")
                    & np.isfinite(probs)
                    & (probs >= th)
                )
                final[promote] = "1M-50M"

            for (_, r), bp, fp, p, d in zip(
                test.iterrows(), base, final, probs, dollars
            ):
                rows[name].append({
                    "disasterNumber": int(r["disasterNumber"]),
                    "fyDeclared": int(r["fyDeclared"]),
                    "state": r["state"],
                    "incidentType": r["incidentType"],
                    "actual_band": r["actual_band"],
                    "base_pred": str(bp),
                    "final_pred": str(fp),
                    "flood_prob": None if not np.isfinite(p) else float(p),
                    "base_reg_dollars": float(d),
                    "threshold": float(th),
                })

            threshold_rows.append({
                "outer_fy": int(outer_fy),
                "variant": name,
                "threshold": float(th),
                "inner_macro": diag["macro"],
                "inner_accuracy": diag["accuracy"],
                "inner_flood_low_recall": diag["flood_low_recall"],
                "inner_flood_mid_recall": diag["flood_mid_recall"],
                "inner_promotions": diag["promotions"],
            })

    results = {}
    for name in variants:
        p = pd.DataFrame(rows[name])
        p.to_csv(OUT / f"{name}_predictions.csv", index=False)

        q = p.rename(columns={"final_pred": "pred"})
        lm = low_metrics(q, "pred")

        flood = p[p["incidentType"] == "Flood"]
        fb = {}
        for b in ["100K-1M", "1M-50M"]:
            m = flood["actual_band"] == b
            n = int(m.sum())
            c = int((flood.loc[m, "final_pred"] == b).sum())
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
            "mid_down": int((mid["final_pred"] == "100K-1M").sum()),
            "low_up": int((lowb["final_pred"] == "1M-50M").sum()),
            "promotions": int(
                (
                    (p["base_pred"] == "100K-1M")
                    & (p["final_pred"] == "1M-50M")
                ).sum()
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
        "# One-way Flood $1M rescue audit",
        "",
        "| Variant | Low acc | Macro | 100K-1M | 1M-50M | Flood low | Flood mid | Mid->low | Low->mid | Promotions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
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
            f"{r['mid_down']} | "
            f"{r['low_up']} | "
            f"{r['promotions']} |"
        )

    (OUT / "summary.md").write_text(
        "\n".join(md), encoding="utf-8"
    )
    print("\n".join(md))


if __name__ == "__main__":
    main()
