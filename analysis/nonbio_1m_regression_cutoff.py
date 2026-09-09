#!/usr/bin/env python3
"""Strict nested-LFYO calibration of the predicted-dollar cutoff at $1M."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from mission_semantic_audit import (
    CURRENT_19, build_semantic_rollup, fetch_all_mission_assignments,
    normalize_master,
)
from nonbio_all_ranges import funding_band, valid_cols
from nonbio_low_thresholds import (
    inner_binary_oof, choose_threshold, low_metrics,
)
from nonbio_cross_1m_rescue import (
    fit_base_reg, predict_reg_dollars, fit_outer_base,
)

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "nonbio_1m_regression_cutoff"
OUT.mkdir(parents=True, exist_ok=True)

LOW_BANDS = ["0-100K", "100K-1M", "1M-50M"]


def regression_oof(train, features, seedbase):
    rows = []
    for fy in sorted(train["fyDeclared"].astype(int).unique()):
        tr = train[train["fyDeclared"].astype(int) != fy].copy()
        te = train[
            (train["fyDeclared"].astype(int) == fy)
            & (train["target_clean"] < 50_000_000)
        ].copy()
        if te.empty:
            continue
        reg = fit_base_reg(tr, features, seedbase + int(fy))
        dollars = predict_reg_dollars(reg, te, features)
        for (_, r), d in zip(te.iterrows(), dollars):
            rows.append({
                "disasterNumber": int(r["disasterNumber"]),
                "actual_band": r["actual_band"],
                "pred_dollars": float(d),
            })
    return pd.DataFrame(rows)


def inner_table(train, semfeat, outer_fy):
    s1 = inner_binary_oof(
        train, semfeat, stage=1, kind="rf",
        seedbase=110000 + outer_fy * 10,
    )
    s1_th, _ = choose_threshold(s1, "macro_f1")
    reg = regression_oof(
        train, semfeat, 210000 + outer_fy * 100,
    )
    d = reg.merge(
        s1[["disasterNumber", "p"]],
        on="disasterNumber", how="inner",
    ).rename(columns={"p": "stage1_prob"})
    return d, float(s1_th)


def score_cutoff(d, s1_th, cutoff):
    pred = np.full(len(d), "0-100K", dtype=object)
    idx = np.flatnonzero(
        d["stage1_prob"].to_numpy(float) >= s1_th
    )
    if len(idx):
        dollars = d.iloc[idx]["pred_dollars"].to_numpy(float)
        pred[idx] = np.where(
            dollars >= cutoff, "1M-50M", "100K-1M"
        )
    actual = d["actual_band"].astype(str).to_numpy()
    rec = {}
    for b in LOW_BANDS:
        m = actual == b
        rec[b] = float((pred[m] == b).mean()) if m.any() else np.nan
    return {
        "macro": float(np.nanmean(list(rec.values()))),
        "accuracy": float((pred == actual).mean()),
        "r_low": rec["100K-1M"],
        "r_mid": rec["1M-50M"],
    }


def choose_cutoff(d, s1_th, guard=None):
    vals = d["pred_dollars"].to_numpy(float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    grid = np.unique(np.r_[
        np.arange(250_000, 2_025_000, 25_000),
        2_250_000, 2_500_000, 1_000_000,
        np.quantile(vals, np.linspace(0.05, 0.95, 31)) if len(vals) else [],
    ])
    best = None
    for c in grid:
        m = score_cutoff(d, s1_th, float(c))
        if guard is not None and m["r_low"] < guard:
            continue
        key = (
            m["macro"], m["accuracy"],
            -abs(float(c) - 1_000_000.0),
        )
        if best is None or key > best[0]:
            best = (key, float(c), m)
    if best is None:
        return 1_000_000.0, score_cutoff(d, s1_th, 1_000_000.0)
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
        "fixed_1m": None,
        "calibrated_macro": None,
        "calibrated_guard75": 0.75,
        "calibrated_guard80": 0.80,
    }
    rows = {k: [] for k in variants}
    cuts = []

    for outer_fy in sorted(low["fyDeclared"].astype(int).unique()):
        train = low[low["fyDeclared"].astype(int) != outer_fy].copy()
        test = low[low["fyDeclared"].astype(int) == outer_fy].copy()

        inner, inner_s1_th = inner_table(train, semfeat, int(outer_fy))
        selected = {
            "fixed_1m": (
                1_000_000.0,
                score_cutoff(inner, inner_s1_th, 1_000_000.0),
            ),
            "calibrated_macro": choose_cutoff(inner, inner_s1_th, None),
            "calibrated_guard75": choose_cutoff(inner, inner_s1_th, 0.75),
            "calibrated_guard80": choose_cutoff(inner, inner_s1_th, 0.80),
        }

        base_predict = fit_outer_base(train, semfeat, int(outer_fy))
        _, dollars = base_predict(test)

        # Obtain the outer stage-1 prediction once from the same base router.
        base_pred, _ = base_predict(test)
        stage1_upper = base_pred != "0-100K"

        for name in variants:
            cutoff, diag = selected[name]
            pred = np.full(len(test), "0-100K", dtype=object)
            idx = np.flatnonzero(stage1_upper)
            if len(idx):
                pred[idx] = np.where(
                    dollars[idx] >= cutoff,
                    "1M-50M",
                    "100K-1M",
                )

            for (_, r), p, d in zip(test.iterrows(), pred, dollars):
                rows[name].append({
                    "disasterNumber": int(r["disasterNumber"]),
                    "fyDeclared": int(r["fyDeclared"]),
                    "state": r["state"],
                    "incidentType": r["incidentType"],
                    "actual_band": r["actual_band"],
                    "final_pred": str(p),
                    "pred_dollars": float(d),
                    "cutoff": float(cutoff),
                })

            cuts.append({
                "outer_fy": int(outer_fy),
                "variant": name,
                "cutoff": float(cutoff),
                "inner_macro": diag["macro"],
                "inner_accuracy": diag["accuracy"],
                "inner_low_recall": diag["r_low"],
                "inner_mid_recall": diag["r_mid"],
            })

    results = {}
    for name in variants:
        p = pd.DataFrame(rows[name])
        p.to_csv(OUT / f"{name}_predictions.csv", index=False)
        q = p.rename(columns={"final_pred": "pred"})
        lm = low_metrics(q, "pred")
        mid = p[p["actual_band"] == "1M-50M"]
        lowb = p[p["actual_band"] == "100K-1M"]
        byfy = p.groupby("fyDeclared")["cutoff"].first()

        results[name] = {
            "low_metrics": lm,
            "mid_down": int((mid["final_pred"] == "100K-1M").sum()),
            "low_up": int((lowb["final_pred"] == "1M-50M").sum()),
            "cutoff_median": float(byfy.median()),
            "cutoff_min": float(byfy.min()),
            "cutoff_max": float(byfy.max()),
        }

    pd.DataFrame(cuts).to_csv(OUT / "cutoffs.csv", index=False)
    (OUT / "summary.json").write_text(
        json.dumps({"results": results}, indent=2),
        encoding="utf-8",
    )

    md = [
        "# $1M predicted-dollar cutoff calibration",
        "",
        "| Variant | Low acc | Macro | 0-100K | 100K-1M | 1M-50M | Mid->low | Low->mid | Median cutoff |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in results.items():
        lm = r["low_metrics"]
        pb = lm["per_band"]
        cutoff_text = "$" + format(r["cutoff_median"], ",.0f")
        md.append(
            f"| {name} | {lm['accuracy']:.1%} | {lm['macro_recall']:.1%} | "
            f"{pb['0-100K']['correct']}/{pb['0-100K']['total']} ({pb['0-100K']['recall']:.1%}) | "
            f"{pb['100K-1M']['correct']}/{pb['100K-1M']['total']} ({pb['100K-1M']['recall']:.1%}) | "
            f"{pb['1M-50M']['correct']}/{pb['1M-50M']['total']} ({pb['1M-50M']['recall']:.1%}) | "
            f"{r['mid_down']} | {r['low_up']} | {cutoff_text} |"
        )

    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
