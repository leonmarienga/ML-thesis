#!/usr/bin/env python3
"""
Strict LFYO one-way Hurricane rescue at the $1M lower boundary.

This audit is completely separate from the frozen Hurricane >=$50M branch.
It touches only Hurricane rows that the frozen LOW router predicts as
$100K-$1M. A specialist may promote them to $1M-$50M, but can never demote an
existing $1M-$50M prediction or alter a row routed to >=$50M.

Relevant low-range balance:
- Hurricane $100K-$1M: 15
- Hurricane $1M-$50M: 19
- Accepted 814 router still has 3 downward Hurricane errors.

Thresholds are selected only from inner-LFYO held-out predictions.
Primary objective: 3-low-band macro recall.
Guard: preserve at least 75% inner recall on genuine Hurricane $100K-$1M.

Variants:
- base
- hurr_current_log
- hurr_current_rf
- hurr_semantics_log
- hurr_semantics_rf

No >=$50M component is touched. Biological remains excluded/frozen.
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
from nonbio_low_thresholds import low_metrics
from nonbio_cross_1m_rescue import (
    base_oof_predictions, fit_outer_base,
)

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "nonbio_hurricane_1m_rescue"
OUT.mkdir(parents=True, exist_ok=True)

LOW_BANDS = ["0-100K", "100K-1M", "1M-50M"]
HAZARD = "Hurricane"


def fit_specialist(train, features, kind, seed):
    t = train[
        (train["incidentType"] == HAZARD)
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


def rescue_oof_scores(train, base_oof, features, kind, outer_fy):
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
        te = te[te["incidentType"] == HAZARD].copy()
        if te.empty:
            continue

        st = tr[
            (tr["incidentType"] == HAZARD)
            & (tr["target_clean"] >= 100_000)
            & (tr["target_clean"] < 50_000_000)
        ].copy()
        ytr = (st["target_clean"] >= 1_000_000).astype(int)
        if len(st) < 10 or ytr.nunique() < 2:
            continue

        model = fit_specialist(
            st, features, kind,
            (300000 if kind == "log" else 310000)
            + int(outer_fy) * 100 + int(fy),
        )
        pp = model.predict_proba(
            normalize_model_frame(te[features])
        )[:, 1]

        for dn, p in zip(te["disasterNumber"].astype(int), pp):
            rows.append({
                "disasterNumber": int(dn),
                "prob_mid": float(p),
            })

    return pd.DataFrame(rows)


def apply_rescue(base_oof, scores, threshold):
    d = base_oof.copy()
    smap = dict(zip(
        scores["disasterNumber"].astype(int),
        scores["prob_mid"].astype(float),
    ))
    d["rescue_prob"] = d["disasterNumber"].astype(int).map(smap)
    d["pred"] = d["base_pred"]

    m = (
        (d["base_pred"] == "100K-1M")
        & d["rescue_prob"].notna()
        & (d["rescue_prob"] >= threshold)
    )
    d.loc[m, "pred"] = "1M-50M"
    return d


def score(inner):
    recs = {}
    for band in LOW_BANDS:
        m = inner["actual_band"] == band
        recs[band] = (
            float((inner.loc[m, "pred"] == band).mean())
            if m.any() else np.nan
        )

    macro = float(np.nanmean(list(recs.values())))
    accuracy = float((inner["pred"] == inner["actual_band"]).mean())

    hurr_low = inner[
        inner["rescue_prob"].notna()
        & (inner["actual_band"] == "100K-1M")
    ]
    hurr_mid = inner[
        inner["rescue_prob"].notna()
        & (inner["actual_band"] == "1M-50M")
    ]

    low_recall = (
        float((hurr_low["pred"] == "100K-1M").mean())
        if len(hurr_low) else np.nan
    )
    mid_recall = (
        float((hurr_mid["pred"] == "1M-50M").mean())
        if len(hurr_mid) else np.nan
    )

    promotions = int(
        ((inner["base_pred"] == "100K-1M") & (inner["pred"] == "1M-50M")).sum()
    )

    return {
        "macro": macro,
        "accuracy": accuracy,
        "hurr_low_candidate_recall": low_recall,
        "hurr_mid_candidate_recall": mid_recall,
        "promotions": promotions,
    }


def choose_threshold(base_oof, scores, guard=0.75):
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
            np.isfinite(m["hurr_low_candidate_recall"])
            and m["hurr_low_candidate_recall"] < guard
        ):
            continue
        key = (m["macro"], m["accuracy"], float(th))
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
    low = df[df["target_clean"] < 50_000_000].copy()

    variants = {
        "base": None,
        "hurr_current_log": (current, "log"),
        "hurr_current_rf": (current, "rf"),
        "hurr_semantics_log": (semfeat, "log"),
        "hurr_semantics_rf": (semfeat, "rf"),
    }

    rows = {k: [] for k in variants}
    threshold_rows = []

    for outer_fy in sorted(low["fyDeclared"].astype(int).unique()):
        train = low[low["fyDeclared"].astype(int) != outer_fy].copy()
        test = low[low["fyDeclared"].astype(int) == outer_fy].copy()

        base_oof, _ = base_oof_predictions(train, semfeat, int(outer_fy))
        outer_predict = fit_outer_base(train, semfeat, int(outer_fy))
        base_pred, base_dollars = outer_predict(test)

        selected = {"base": (1.0, None, None)}
        for name, spec in variants.items():
            if name == "base":
                continue
            features, kind = spec
            scores = rescue_oof_scores(
                train, base_oof, features, kind, int(outer_fy)
            )
            th, diag = choose_threshold(base_oof, scores, guard=0.75)
            selected[name] = (th, diag, spec)

        models = {}
        for name, spec in variants.items():
            if name == "base":
                continue
            features, kind = spec
            st = train[
                (train["incidentType"] == HAZARD)
                & (train["target_clean"] >= 100_000)
                & (train["target_clean"] < 50_000_000)
            ].copy()
            ytr = (st["target_clean"] >= 1_000_000).astype(int)
            models[name] = (
                fit_specialist(
                    st, features, kind,
                    (500000 if kind == "log" else 510000) + int(outer_fy),
                )
                if len(st) >= 10 and ytr.nunique() >= 2
                else None
            )

        for name in variants:
            final = base_pred.copy()
            th, diag, spec = selected[name]
            probs = np.full(len(test), np.nan, dtype=float)

            if name != "base":
                features, kind = spec
                model = models[name]
                idx = np.flatnonzero(
                    (base_pred == "100K-1M")
                    & (test["incidentType"].to_numpy() == HAZARD)
                )
                if len(idx) and model is not None:
                    pp = model.predict_proba(
                        normalize_model_frame(test.iloc[idx][features])
                    )[:, 1]
                    probs[idx] = pp
                    final[idx[pp >= th]] = "1M-50M"

            for (_, r), bp, fp, p, dollars in zip(
                test.iterrows(), base_pred, final, probs, base_dollars
            ):
                rows[name].append({
                    "disasterNumber": int(r["disasterNumber"]),
                    "fyDeclared": int(r["fyDeclared"]),
                    "state": r["state"],
                    "incidentType": r["incidentType"],
                    "actual_band": r["actual_band"],
                    "base_pred": str(bp),
                    "final_pred": str(fp),
                    "specialist_prob": None if not np.isfinite(p) else float(p),
                    "base_reg_dollars": float(dollars),
                    "threshold": float(th),
                })

            if name != "base":
                threshold_rows.append({
                    "outer_fy": int(outer_fy),
                    "variant": name,
                    "threshold": float(th),
                    "inner_macro": diag["macro"],
                    "inner_accuracy": diag["accuracy"],
                    "inner_hurr_low_candidate_recall": diag["hurr_low_candidate_recall"],
                    "inner_hurr_mid_candidate_recall": diag["hurr_mid_candidate_recall"],
                    "inner_promotions": diag["promotions"],
                })

    results = {}
    for name in variants:
        p = pd.DataFrame(rows[name])
        p.to_csv(OUT / f"{name}_predictions.csv", index=False)
        q = p.rename(columns={"final_pred": "pred"})
        lm = low_metrics(q, "pred")

        hz = p[p["incidentType"] == HAZARD]
        hb = {}
        for band in ["100K-1M", "1M-50M"]:
            m = hz["actual_band"] == band
            n = int(m.sum())
            c = int((hz.loc[m, "final_pred"] == band).sum())
            hb[band] = {"correct": c, "total": n, "recall": c / n if n else None}

        mid = p[p["actual_band"] == "1M-50M"]
        lowb = p[p["actual_band"] == "100K-1M"]
        results[name] = {
            "low_metrics": lm,
            "hurricane_boundary": hb,
            "mid_down": int((mid["final_pred"] == "100K-1M").sum()),
            "low_up": int((lowb["final_pred"] == "1M-50M").sum()),
            "promotions": int(
                ((p["base_pred"] == "100K-1M") & (p["final_pred"] == "1M-50M")).sum()
            ),
        }

    pd.DataFrame(threshold_rows).to_csv(OUT / "thresholds.csv", index=False)
    (OUT / "summary.json").write_text(
        json.dumps({"results": results}, indent=2), encoding="utf-8"
    )

    md = [
        "# Hurricane one-way $1M rescue audit",
        "",
        "| Variant | Low acc | Macro | 100K-1M | 1M-50M | Hurr low | Hurr mid | Mid->low | Low->mid | Promotions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in results.items():
        lm = r["low_metrics"]
        pb = lm["per_band"]
        hb = r["hurricane_boundary"]
        md.append(
            f"| {name} | {lm['accuracy']:.1%} | {lm['macro_recall']:.1%} | "
            f"{pb['100K-1M']['correct']}/{pb['100K-1M']['total']} ({pb['100K-1M']['recall']:.1%}) | "
            f"{pb['1M-50M']['correct']}/{pb['1M-50M']['total']} ({pb['1M-50M']['recall']:.1%}) | "
            f"{hb['100K-1M']['correct']}/{hb['100K-1M']['total']} ({hb['100K-1M']['recall']:.1%}) | "
            f"{hb['1M-50M']['correct']}/{hb['1M-50M']['total']} ({hb['1M-50M']['recall']:.1%}) | "
            f"{r['mid_down']} | {r['low_up']} | {r['promotions']} |"
        )

    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
