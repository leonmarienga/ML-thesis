#!/usr/bin/env python3
"""
Strict nested-LFYO cross-$1M rescue audit (low branch only).

Purpose
-------
The accepted 812/912 router has largely solved upward leakage from $1M-$50M.
The dominant remaining error is now downward leakage:
true $1M-$50M -> predicted $100K-$1M.

This audit freezes the accepted lower router:
1. semantics RF for <100K vs >=100K, threshold from inner-LFYO macro-F1
2. semantics ExtraTrees log-dollar regression for the $1M boundary

A one-way rescue verifier is invoked ONLY when the frozen base low router
predicts $100K-$1M. The verifier may promote that row to $1M-$50M.

Threshold selection
-------------------
For each outer LFYO fold:
- generate inner-LFYO base predictions and rescue probabilities
- evaluate only using inner held-out predictions
- choose the rescue threshold maximizing THREE-low-band macro recall
  (0-100K, 100K-1M, 1M-50M)
- tie-break by low-branch accuracy, then by higher threshold (conservative)

Models
------
- compact operational logistic
- compact operational RF
- full semantics logistic
- full semantics RF

No >=$50M component is changed here.
Biological remains excluded/frozen.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score

from mission_semantic_audit import (
    CURRENT_19,
    build_semantic_rollup,
    fetch_all_mission_assignments,
    normalize_master,
    normalize_model_frame,
    prep_pipeline,
)
from nonbio_all_ranges import funding_band, valid_cols
from nonbio_low_thresholds import (
    inner_binary_oof,
    choose_threshold,
    fit_binary,
    positive_proba,
    low_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "nonbio_cross_1m_rescue"
OUT.mkdir(parents=True, exist_ok=True)

LOW_BANDS = ["0-100K", "100K-1M", "1M-50M"]

COMPACT_FEATURES = [
    "durationDays",
    "declarationDelayDays",
    "expectedResourceScore",
    "ma_mean_amendment",
    "ma_max_amendment",
    "ma_amendment_share",
    "sem_duration_mean",
    "sem_duration_median",
    "sem_duration_p90",
    "sem_duration_max",
    "sem_long_90d_count",
    "sem_topic_transportation_count",
    "sem_topic_transportation_share",
    "sem_topic_housing_shelter_count",
    "sem_topic_housing_shelter_share",
    "sem_agency_coe_mvd_count",
    "sem_agency_coe_mvd_share",
]


def fit_base_reg(train: pd.DataFrame, features, seed: int):
    t = train[
        (train["target_clean"] >= 100_000)
        & (train["target_clean"] < 50_000_000)
    ].copy()
    X = normalize_model_frame(t[features])
    y = np.log1p(t["target_clean"].to_numpy(float))
    model = ExtraTreesRegressor(
        n_estimators=900,
        random_state=seed,
        max_features="sqrt",
        min_samples_leaf=1,
        n_jobs=1,
    )
    pipe = prep_pipeline(X, model)
    pipe.fit(X, y)
    return pipe


def predict_reg_dollars(model, df: pd.DataFrame, features):
    if df.empty:
        return np.array([], dtype=float)
    return np.expm1(
        model.predict(normalize_model_frame(df[features]))
    )


def fit_rescue(train: pd.DataFrame, features, kind: str, seed: int):
    t = train[
        (train["target_clean"] >= 100_000)
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


def base_oof_predictions(
    train: pd.DataFrame,
    semfeat,
    outer_fy: int,
):
    """
    Inner-LFYO held-out predictions for the frozen base low router.
    Stage-1 threshold is selected from inner OOF scores on outer-training data,
    exactly as in the accepted lower-boundary architecture.
    """
    s1_oof = inner_binary_oof(
        train,
        semfeat,
        stage=1,
        kind="rf",
        seedbase=110000 + outer_fy * 10,
    )
    s1_th, _ = choose_threshold(s1_oof, "macro_f1")

    p1_map = dict(
        zip(
            s1_oof["disasterNumber"].astype(int),
            s1_oof["p"].astype(float),
        )
    )

    rows = []
    years = sorted(train["fyDeclared"].astype(int).unique())

    for fy in years:
        tr = train[train["fyDeclared"].astype(int) != fy].copy()
        te = train[
            (train["fyDeclared"].astype(int) == fy)
            & (train["target_clean"] < 50_000_000)
        ].copy()
        if te.empty:
            continue

        reg = fit_base_reg(
            tr,
            semfeat,
            210000 + outer_fy * 100 + int(fy),
        )
        dollars = predict_reg_dollars(reg, te, semfeat)

        for (_, r), d in zip(te.iterrows(), dollars):
            dn = int(r["disasterNumber"])
            p1 = p1_map.get(dn, np.nan)

            if not np.isfinite(p1):
                base_pred = None
            elif p1 < s1_th:
                base_pred = "0-100K"
            else:
                base_pred = (
                    "1M-50M"
                    if float(d) >= 1_000_000
                    else "100K-1M"
                )

            rows.append({
                "disasterNumber": dn,
                "fy": int(fy),
                "actual_band": r["actual_band"],
                "base_pred": base_pred,
                "base_reg_dollars": float(d),
                "stage1_prob": float(p1) if np.isfinite(p1) else None,
            })

    return pd.DataFrame(rows), float(s1_th)


def rescue_oof_scores(
    train: pd.DataFrame,
    base_oof: pd.DataFrame,
    features,
    kind: str,
    outer_fy: int,
):
    """
    Train rescue models on inner-train rows, score only held-out rows that
    the frozen base OOF router predicted as $100K-$1M.
    """
    rows = []
    years = sorted(train["fyDeclared"].astype(int).unique())

    for fy in years:
        tr = train[train["fyDeclared"].astype(int) != fy].copy()
        cand = base_oof[
            (base_oof["fy"].astype(int) == fy)
            & (base_oof["base_pred"] == "100K-1M")
        ].copy()

        if cand.empty:
            continue

        rescue_train = tr[
            (tr["target_clean"] >= 100_000)
            & (tr["target_clean"] < 50_000_000)
        ].copy()
        ytr = (rescue_train["target_clean"] >= 1_000_000).astype(int)
        if ytr.nunique() < 2:
            continue

        model = fit_rescue(
            rescue_train,
            features,
            kind,
            (
                310000
                if kind == "log"
                else 320000
            ) + outer_fy * 100 + int(fy),
        )

        te = train[
            train["disasterNumber"].astype(int).isin(
                cand["disasterNumber"].astype(int)
            )
        ].copy()

        # Preserve candidate order from base_oof.
        te = cand[["disasterNumber"]].merge(
            te,
            on="disasterNumber",
            how="left",
        )

        pp = model.predict_proba(
            normalize_model_frame(te[features])
        )[:, 1]

        for dn, p in zip(
            te["disasterNumber"].astype(int),
            pp,
        ):
            rows.append({
                "disasterNumber": int(dn),
                "prob_mid": float(p),
            })

    return pd.DataFrame(rows)


def apply_rescue_to_oof(
    base_oof: pd.DataFrame,
    rescue_scores: pd.DataFrame,
    threshold: float,
):
    d = base_oof.copy()
    smap = dict(
        zip(
            rescue_scores["disasterNumber"].astype(int),
            rescue_scores["prob_mid"].astype(float),
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


def low_score(d: pd.DataFrame, pred_col: str):
    recs = []
    correct = 0
    total = 0
    for band in LOW_BANDS:
        m = d["actual_band"] == band
        n = int(m.sum())
        c = int((d.loc[m, pred_col] == band).sum())
        if n:
            recs.append(c / n)
            correct += c
            total += n
    return (
        float(np.mean(recs)) if recs else 0.0,
        float(correct / total) if total else 0.0,
    )


def choose_rescue_threshold(
    base_oof: pd.DataFrame,
    rescue_scores: pd.DataFrame,
):
    if rescue_scores.empty:
        return 1.0, {
            "macro_recall": None,
            "accuracy": None,
            "promotions": 0,
        }

    probs = rescue_scores["prob_mid"].to_numpy(float)
    candidates = np.unique(
        np.r_[
            0.05,
            np.arange(0.10, 0.96, 0.025),
            0.99,
            probs,
        ]
    )
    candidates = candidates[
        (candidates >= 0.02)
        & (candidates <= 0.995)
    ]

    best = None

    for th in candidates:
        d = apply_rescue_to_oof(
            base_oof,
            rescue_scores,
            float(th),
        )
        macro, acc = low_score(d, "pred")
        promotions = int(
            (
                (d["base_pred"] == "100K-1M")
                & (d["pred"] == "1M-50M")
            ).sum()
        )

        # Primary: macro recall.
        # Secondary: accuracy.
        # Tertiary: higher threshold, i.e. conservative promotion.
        key = (
            float(macro),
            float(acc),
            float(th),
        )
        if best is None or key > best[0]:
            best = (
                key,
                float(th),
                float(macro),
                float(acc),
                promotions,
            )

    _, th, macro, acc, promotions = best
    return th, {
        "macro_recall": macro,
        "accuracy": acc,
        "promotions": promotions,
    }


def fit_outer_base(
    train: pd.DataFrame,
    semfeat,
    outer_fy: int,
):
    low_train = train[
        train["target_clean"] < 50_000_000
    ].copy()

    s1_oof = inner_binary_oof(
        train,
        semfeat,
        stage=1,
        kind="rf",
        seedbase=410000 + outer_fy * 10,
    )
    s1_th, _ = choose_threshold(
        s1_oof,
        "macro_f1",
    )

    y1 = (
        low_train["target_clean"] >= 100_000
    ).astype(int)
    s1 = fit_binary(
        low_train,
        semfeat,
        y1,
        "rf",
        420000 + outer_fy,
    )

    reg = fit_base_reg(
        train,
        semfeat,
        430000 + outer_fy,
    )

    def predict(dd: pd.DataFrame):
        if dd.empty:
            return (
                np.array([], dtype=object),
                np.array([], dtype=float),
            )

        p1 = positive_proba(
            s1,
            dd,
            semfeat,
        )
        dollars = predict_reg_dollars(
            reg,
            dd,
            semfeat,
        )

        pred = np.full(
            len(dd),
            "0-100K",
            dtype=object,
        )

        idx = np.flatnonzero(
            p1 >= s1_th
        )
        if len(idx):
            pred[idx] = np.where(
                dollars[idx] >= 1_000_000,
                "1M-50M",
                "100K-1M",
            )

        return pred.astype(str), dollars

    return predict


def main():
    master = normalize_master(
        pd.read_excel(MASTER)
    )
    master["target_clean"] = pd.to_numeric(
        master["totalObligatedFunding"],
        errors="coerce",
    ).fillna(0).clip(lower=0)
    master["actual_band"] = master[
        "target_clean"
    ].map(funding_band)

    ma = fetch_all_mission_assignments()
    sem, _ = build_semantic_rollup(
        master,
        ma,
    )

    nonbio = master.merge(
        sem,
        on="disasterNumber",
        how="left",
    )
    nonbio = nonbio[
        nonbio["incidentType"] != "Biological"
    ].copy().reset_index(drop=True)

    current = valid_cols(
        nonbio,
        CURRENT_19,
    )
    semcols = valid_cols(
        nonbio,
        [
            c for c in nonbio.columns
            if (
                c.startswith("sem_")
                or c.startswith("ma_")
            )
            and not any(
                bad in c.lower()
                for bad in [
                    "oblig",
                    "fund",
                    "cost",
                    "amount",
                    "dollar",
                ]
            )
        ],
    )
    semfeat = list(
        dict.fromkeys(
            current + semcols
        )
    )

    compact = valid_cols(
        nonbio,
        COMPACT_FEATURES,
    )

    variants = {
        "base_no_rescue": None,
        "compact_log": (compact, "log"),
        "compact_rf": (compact, "rf"),
        "semantics_log": (semfeat, "log"),
        "semantics_rf": (semfeat, "rf"),
    }

    rows = {k: [] for k in variants}
    threshold_rows = []

    low_all = nonbio[
        nonbio["target_clean"] < 50_000_000
    ].copy()

    for outer_fy in sorted(
        low_all["fyDeclared"]
        .astype(int)
        .unique()
    ):
        train = low_all[
            low_all["fyDeclared"].astype(int)
            != outer_fy
        ].copy()

        test = low_all[
            low_all["fyDeclared"].astype(int)
            == outer_fy
        ].copy()

        base_oof, _ = base_oof_predictions(
            train,
            semfeat,
            int(outer_fy),
        )

        outer_predict = fit_outer_base(
            train,
            semfeat,
            int(outer_fy),
        )

        base_pred, base_dollars = (
            outer_predict(test)
        )

        # Base result first.
        for (_, r), pred, dollars in zip(
            test.iterrows(),
            base_pred,
            base_dollars,
        ):
            rows["base_no_rescue"].append({
                "disasterNumber": int(
                    r["disasterNumber"]
                ),
                "fyDeclared": int(
                    r["fyDeclared"]
                ),
                "state": r["state"],
                "incidentType": (
                    r["incidentType"]
                ),
                "actual_band": (
                    r["actual_band"]
                ),
                "base_pred": str(pred),
                "final_pred": str(pred),
                "base_reg_dollars": float(
                    dollars
                ),
                "rescue_prob": None,
            })

        for name, spec in variants.items():
            if name == "base_no_rescue":
                continue

            features, kind = spec

            rescue_scores = (
                rescue_oof_scores(
                    train,
                    base_oof,
                    features,
                    kind,
                    int(outer_fy),
                )
            )

            th, diag = (
                choose_rescue_threshold(
                    base_oof,
                    rescue_scores,
                )
            )

            rescue_train = train[
                (train["target_clean"] >= 100_000)
                & (train["target_clean"] < 50_000_000)
            ].copy()

            rescue_model = fit_rescue(
                rescue_train,
                features,
                kind,
                (
                    510000
                    if kind == "log"
                    else 520000
                ) + int(outer_fy),
            )

            final_pred = base_pred.copy()
            rescue_prob = np.full(
                len(test),
                np.nan,
                dtype=float,
            )

            idx = np.flatnonzero(
                base_pred == "100K-1M"
            )
            if len(idx):
                pp = rescue_model.predict_proba(
                    normalize_model_frame(
                        test.iloc[idx][features]
                    )
                )[:, 1]
                rescue_prob[idx] = pp
                final_pred[
                    idx[pp >= th]
                ] = "1M-50M"

            for (_, r), bp, fp, dollars, rp in zip(
                test.iterrows(),
                base_pred,
                final_pred,
                base_dollars,
                rescue_prob,
            ):
                rows[name].append({
                    "disasterNumber": int(
                        r["disasterNumber"]
                    ),
                    "fyDeclared": int(
                        r["fyDeclared"]
                    ),
                    "state": r["state"],
                    "incidentType": (
                        r["incidentType"]
                    ),
                    "actual_band": (
                        r["actual_band"]
                    ),
                    "base_pred": str(bp),
                    "final_pred": str(fp),
                    "base_reg_dollars": float(
                        dollars
                    ),
                    "rescue_prob": (
                        None
                        if not np.isfinite(rp)
                        else float(rp)
                    ),
                })

            threshold_rows.append({
                "outer_fy": int(
                    outer_fy
                ),
                "variant": name,
                "threshold": float(th),
                "inner_macro_recall": (
                    diag.get("macro_recall")
                ),
                "inner_accuracy": (
                    diag.get("accuracy")
                ),
                "inner_promotions": (
                    diag.get("promotions")
                ),
            })

    results = {}

    for name in variants:
        p = pd.DataFrame(rows[name])
        p.to_csv(
            OUT / f"{name}_predictions.csv",
            index=False,
        )

        metrics = low_metrics(
            p.rename(
                columns={
                    "final_pred": "pred"
                }
            ),
            "pred",
        )

        true_mid = p[
            p["actual_band"] == "1M-50M"
        ]
        true_low = p[
            p["actual_band"] == "100K-1M"
        ]

        results[name] = {
            "low_metrics": metrics,
            "mid_down_to_100K_1M": int(
                (
                    true_mid["final_pred"]
                    == "100K-1M"
                ).sum()
            ),
            "mid_down_to_0_100K": int(
                (
                    true_mid["final_pred"]
                    == "0-100K"
                ).sum()
            ),
            "low_up_to_1M_50M": int(
                (
                    true_low["final_pred"]
                    == "1M-50M"
                ).sum()
            ),
            "promotions": int(
                (
                    (p["base_pred"] == "100K-1M")
                    & (
                        p["final_pred"]
                        == "1M-50M"
                    )
                ).sum()
            ),
        }

    pd.DataFrame(
        threshold_rows
    ).to_csv(
        OUT / "thresholds.csv",
        index=False,
    )

    (
        OUT / "summary.json"
    ).write_text(
        json.dumps(
            {
                "compact_features": compact,
                "semantic_feature_count": len(
                    semfeat
                ),
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    md = [
        "# Cross-$1M rescue audit",
        "",
        "- Strict outer LFYO.",
        "- Rescue invoked only when frozen base predicts $100K-$1M.",
        "- Rescue threshold selected from inner held-out predictions.",
        "- Biological excluded.",
        "",
        "| Variant | Low acc | Macro | 0-100K | 100K-1M | 1M-50M | Mid->100K-1M | 100K-1M->Mid | Promotions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for name, r in results.items():
        lm = r["low_metrics"]
        pb = lm["per_band"]
        md.append(
            f"| {name} | "
            f"{lm['accuracy']:.1%} | "
            f"{lm['macro_recall']:.1%} | "
            f"{pb['0-100K']['correct']}/{pb['0-100K']['total']} "
            f"({pb['0-100K']['recall']:.1%}) | "
            f"{pb['100K-1M']['correct']}/{pb['100K-1M']['total']} "
            f"({pb['100K-1M']['recall']:.1%}) | "
            f"{pb['1M-50M']['correct']}/{pb['1M-50M']['total']} "
            f"({pb['1M-50M']['recall']:.1%}) | "
            f"{r['mid_down_to_100K_1M']} | "
            f"{r['low_up_to_1M_50M']} | "
            f"{r['promotions']} |"
        )

    (
        OUT / "summary.md"
    ).write_text(
        "\n".join(md),
        encoding="utf-8",
    )
    print("\n".join(md))


if __name__ == "__main__":
    main()
