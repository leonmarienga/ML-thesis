#!/usr/bin/env python3
"""
Standard nested-LFYO one-way Flood rescue at the $1M boundary.

This uses the accepted low router's standard inner OOF predictions:
- Stage 1 OOF probabilities and threshold from outer-training years
- ExtraTrees OOF dollar predictions from held-out inner fiscal years
- Flood logistic rescue trained on inner-train Floods and scored only on
  held-out Floods the base OOF router predicts $100K-$1M

The rescue is one-way only: it may promote $100K-$1M -> $1M-$50M.
It never demotes a base $1M-$50M prediction.

Variants:
- base
- rescue_macro
- rescue_guard70
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
from nonbio_low_thresholds import low_metrics
from nonbio_cross_1m_rescue import (
    base_oof_predictions, fit_outer_base,
)

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "nonbio_flood_1m_rescue_standard"
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


def flood_rescue_oof_scores(train, base_oof, features, outer_fy):
    rows = []
    for fy in sorted(train["fyDeclared"].astype(int).unique()):
        tr = train[train["fyDeclared"].astype(int) != fy].copy()

        cand = base_oof[
            (base_oof["fy"].astype(int) == fy)
            & (base_oof["base_pred"] == "100K-1M")
        ].copy()

        if cand.empty:
            continue

        te = train[
            train["disasterNumber"].astype(int).isin(
                cand["disasterNumber"].astype(int)
            )
        ].copy()
        te = cand[["disasterNumber"]].merge(
            te, on="disasterNumber", how="left"
        )
        te = te[te["incidentType"] == "Flood"].copy()
        if te.empty:
            continue

        flood_train = tr[
            (tr["incidentType"] == "Flood")
            & (tr["target_clean"] >= 100_000)
            & (tr["target_clean"] < 50_000_000)
        ].copy()
        ytr = (flood_train["target_clean"] >= 1_000_000).astype(int)
        if len(flood_train) < 12 or ytr.nunique() < 2:
            continue

        model = fit_flood_log(
            flood_train,
            features,
            300000 + int(outer_fy) * 100 + int(fy),
        )
        pp = model.predict_proba(
            normalize_model_frame(te[features])
        )[:, 1]

        for dn, p in zip(
            te["disasterNumber"].astype(int), pp
        ):
            rows.append({
                "disasterNumber": int(dn),
                "prob_mid": float(p),
            })

    return pd.DataFrame(rows)


def apply_rescue(base_oof, scores, threshold):
    d = base_oof.copy()
    smap = dict(
        zip(
            scores["disasterNumber"].astype(int),
            scores["prob_mid"].astype(float),
        )
    )
    d["rescue_prob"] = d["disasterNumber"].astype(int).map(smap)
    d["pred"] = d["base_pred"]

    m = (
        (d["base_pred"] == "100K-1M")
        & d["rescue_prob"].notna()
        & (d["rescue_prob"] >= threshold)
    )
    d.loc[m, "pred"] = "1M-50M"
    return d


def score(d):
    recalls = {}
    for b in LOW_BANDS:
        m = d["actual_band"] == b
        recalls[b] = (
            float((d.loc[m, "pred"] == b).mean())
            if m.any() else np.nan
        )

    macro = float(np.nanmean(list(recalls.values())))
    acc = float((d["pred"] == d["actual_band"]).mean())

    # Flood-specific preservation / recovery.
    flood_low = d[
        (d["actual_band"] == "100K-1M")
        & d["disasterNumber"].isin(
            d.loc[
                d["rescue_prob"].notna(),
                "disasterNumber"
            ]
        )
    ]
    # More robust: actual Flood identity comes from scores membership, since only
    # Flood candidates receive rescue_prob.
    flr = (
        float((flood_low["pred"] == "100K-1M").mean())
        if len(flood_low) else np.nan
    )

    promoted = int(
        (
            (d["base_pred"] == "100K-1M")
            & (d["pred"] == "1M-50M")
        ).sum()
    )

    return {
        "macro": macro,
        "accuracy": acc,
        "flood_low_candidate_recall": flr,
        "promotions": promoted,
    }


def choose_threshold(base_oof, scores, guard=None):
    if scores.empty:
        d = apply_rescue(base_oof, scores, 1.0)
        return 1.0, score(d)

    probs = scores["prob_mid"].to_numpy(float)
    grid = np.unique(np.r_[
        0.05,
        np.arange(0.10, 0.96, 0.02),
        0.99,
        probs,
    ])

    best = None
    for th in grid:
        d = apply_rescue(base_oof, scores, float(th))
        m = score(d)

        if (
            guard is not None
            and np.isfinite(m["flood_low_candidate_recall"])
            and m["flood_low_candidate_recall"] < guard
        ):
            continue

        key = (
            m["macro"],
            m["accuracy"],
            float(th),
        )
        if best is None or key > best[0]:
            best = (key, float(th), m)

    if best is None:
        d = apply_rescue(base_oof, scores, 1.0)
        return 1.0, score(d)
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

        base_oof, _ = base_oof_predictions(
            train, semfeat, int(outer_fy)
        )
        scores = flood_rescue_oof_scores(
            train, base_oof, flood_features, int(outer_fy)
        )

        selected = {"base": (1.0, score(apply_rescue(base_oof, scores, 1.0)))}
        for name, guard in variants.items():
            if name == "base":
                continue
            selected[name] = choose_threshold(
                base_oof, scores, guard
            )

        outer_predict = fit_outer_base(
            train, semfeat, int(outer_fy)
        )
        base_pred, base_dollars = outer_predict(test)

        flood_train = train[
            (train["incidentType"] == "Flood")
            & (train["target_clean"] >= 100_000)
            & (train["target_clean"] < 50_000_000)
        ].copy()
        ytr = (flood_train["target_clean"] >= 1_000_000).astype(int)

        model = None
        if len(flood_train) >= 12 and ytr.nunique() >= 2:
            model = fit_flood_log(
                flood_train, flood_features,
                500000 + int(outer_fy),
            )

        probs = np.full(len(test), np.nan, dtype=float)
        idx = np.flatnonzero(
            (base_pred == "100K-1M")
            & (test["incidentType"].to_numpy() == "Flood")
        )
        if len(idx) and model is not None:
            probs[idx] = model.predict_proba(
                normalize_model_frame(
                    test.iloc[idx][flood_features]
                )
            )[:, 1]

        for name in variants:
            th, diag = selected[name]
            final = base_pred.copy()

            if name != "base":
                promote = (
                    (base_pred == "100K-1M")
                    & (test["incidentType"].to_numpy() == "Flood")
                    & np.isfinite(probs)
                    & (probs >= th)
                )
                final[promote] = "1M-50M"

            for (_, r), bp, fp, p, d in zip(
                test.iterrows(),
                base_pred,
                final,
                probs,
                base_dollars,
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
                "inner_flood_low_candidate_recall": diag["flood_low_candidate_recall"],
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
        "# Standard nested one-way Flood $1M rescue audit",
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
            f"{r['mid_down']} | {r['low_up']} | {r['promotions']} |"
        )

    (OUT / "summary.md").write_text(
        "\n".join(md), encoding="utf-8"
    )
    print("\n".join(md))


if __name__ == "__main__":
    main()
