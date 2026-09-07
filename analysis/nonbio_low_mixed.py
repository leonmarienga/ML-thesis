#!/usr/bin/env python3
"""
Strict-LFYO mixed low-range specialist audit.

Motivation
----------
Previous audit showed complementary strengths:
- semantics RF: excellent 0-100K recognition / best raw low accuracy
- current19 logistic: strongest 100K-1M and 1M-50M balance

This audit mixes specialists by decision stage:
Stage 1: 0-100K vs >=100K
Stage 2: 100K-1M vs 1M-50M

The >=50M root remains fixed to the accepted nested-recall OR + logistic
candidate verifier calibrated to 95% inner recall.
The high-value hierarchy remains frozen at 22/23 conditional accuracy.
Biological remains excluded/frozen.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score

from mission_semantic_audit import (
    CURRENT_19, build_semantic_rollup, fetch_all_mission_assignments,
    normalize_master, normalize_model_frame, prep_pipeline,
)
from external_severity_ablation import build_external
from nonbio_hazard_hierarchy import initial_mechanism_counts
from nonbio_outage_rescue import build_eaglei_all
from nonbio_all_ranges import BANDS, funding_band, high22_predict, six_band_metrics, valid_cols
from nonbio_root_recall import fit_log, proba, inner_oof_scores, recall_first_threshold
from nonbio_root_cascade import verifier_oof, threshold_for_recall, fit_verifier

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "nonbio_low_mixed"
OUT.mkdir(parents=True, exist_ok=True)
LOW_BANDS = ["0-100K", "100K-1M", "1M-50M"]


def fit_binary(train: pd.DataFrame, features: List[str], y: pd.Series, kind: str, seed: int):
    X = normalize_model_frame(train[features])
    if kind == "rf":
        model = RandomForestClassifier(
            n_estimators=900,
            random_state=seed,
            class_weight="balanced_subsample",
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=-1,
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


class MixedLow:
    def __init__(self, s1, f1, s2, f2):
        self.s1 = s1
        self.f1 = f1
        self.s2 = s2
        self.f2 = f2

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        p1 = self.s1.predict(normalize_model_frame(df[self.f1])).astype(int)
        out = np.full(len(df), "0-100K", dtype=object)
        idx = np.flatnonzero(p1 == 1)
        if len(idx):
            p2 = self.s2.predict(
                normalize_model_frame(df.iloc[idx][self.f2])
            ).astype(int)
            out[idx] = np.where(p2 == 1, "1M-50M", "100K-1M")
        return out.astype(str)


def fit_mixed_low(
    train: pd.DataFrame,
    f1: List[str], k1: str,
    f2: List[str], k2: str,
    seed: int,
):
    low = train[train["target_clean"] < 50_000_000].copy()

    y1 = (low["target_clean"] >= 100_000).astype(int)
    s1 = fit_binary(low, f1, y1, k1, seed + 1000)

    upper = low[low["target_clean"] >= 100_000].copy()
    y2 = (upper["target_clean"] >= 1_000_000).astype(int)
    s2 = fit_binary(upper, f2, y2, k2, seed + 2000)

    return MixedLow(s1, f1, s2, f2)


def low_metrics(df: pd.DataFrame, col: str) -> Dict:
    pb = {}
    rs = []
    for b in LOW_BANDS:
        m = df["actual_band"] == b
        n = int(m.sum())
        c = int((df.loc[m, col] == b).sum())
        r = c / n if n else None
        pb[b] = {"correct": c, "total": n, "recall": r}
        if r is not None:
            rs.append(r)
    return {
        "accuracy": float((df[col] == df["actual_band"]).mean()),
        "macro_recall": float(np.mean(rs)),
        "per_band": pb,
    }


def main():
    master = normalize_master(pd.read_excel(MASTER))
    master["target_clean"] = pd.to_numeric(
        master["totalObligatedFunding"], errors="coerce"
    ).fillna(0).clip(lower=0)
    master["actual_band"] = master["target_clean"].map(funding_band)

    ma = fetch_all_mission_assignments()
    sem, _ = build_semantic_rollup(master, ma)
    ext, ext_audit = build_external(master)
    mech = initial_mechanism_counts(master, ma)
    print("Building EAGLE-I Hurricane features...", flush=True)
    eag = build_eaglei_all(master)

    df = (
        master.merge(sem, on="disasterNumber", how="left")
        .merge(ext, on="disasterNumber", how="left")
        .merge(mech, on="disasterNumber", how="left")
        .merge(eag, on="disasterNumber", how="left")
    )
    df["initial_usace_esf3_dfa_count"] = (
        df["initial_usace_esf3_dfa_count"].fillna(0).astype(int)
    )
    nonbio = df[df["incidentType"] != "Biological"].copy().reset_index(drop=True)

    current = valid_cols(nonbio, CURRENT_19)
    semcols = valid_cols(
        nonbio,
        [c for c in nonbio.columns if c.startswith("sem_") or c.startswith("ma_")],
    )
    semfeat = list(dict.fromkeys(current + semcols))
    extcols = valid_cols(
        nonbio,
        [c for c in nonbio.columns
         if c.startswith("nhc_") or c.startswith("noaa_") or c.startswith("calfire_")],
    )
    semext = list(dict.fromkeys(semfeat + extcols))

    high_all = nonbio[nonbio["target_clean"] >= 50_000_000].copy()
    hc = valid_cols(high_all, CURRENT_19)
    hs = valid_cols(
        high_all,
        [c for c in high_all.columns if c.startswith("sem_") or c.startswith("ma_")],
    )
    he = valid_cols(
        high_all,
        [c for c in high_all.columns
         if c.startswith("nhc_") or c.startswith("noaa_") or c.startswith("calfire_")],
    )
    high_gate_features = hc + hs + he
    high_lower_features = hc + hs

    variants = {
        "semRF_to_curLOG": (semfeat, "rf", current, "log"),
        "curRF_to_curLOG": (current, "rf", current, "log"),
        "semRF_to_semLOG": (semfeat, "rf", semfeat, "log"),
        "curRF_to_semLOG": (current, "rf", semfeat, "log"),
        "semRF_to_curRF": (semfeat, "rf", current, "rf"),
        "semLOG_to_curLOG": (semfeat, "log", current, "log"),
    }

    rows = {k: [] for k in variants}
    oracle = {k: [] for k in variants}
    root_rows = []

    for outer_fy in sorted(nonbio["fyDeclared"].astype(int).unique()):
        train = nonbio[nonbio["fyDeclared"].astype(int) != outer_fy].copy()
        test = nonbio[nonbio["fyDeclared"].astype(int) == outer_fy].copy()
        train_high = train[train["target_clean"] >= 50_000_000].copy()

        # Fixed accepted root candidate.
        oof_cur = inner_oof_scores(train, current).rename(columns={"prob": "pcur"})
        oof_sem = inner_oof_scores(train, semext).rename(columns={"prob": "psem"})
        th_cur, _ = recall_first_threshold(oof_cur.rename(columns={"pcur": "prob"}))
        th_sem, _ = recall_first_threshold(oof_sem.rename(columns={"psem": "prob"}))

        oo = oof_cur[["disasterNumber", "pcur"]].merge(
            oof_sem[["disasterNumber", "psem"]],
            on="disasterNumber",
            how="inner",
        )
        tm = train.merge(oo, on="disasterNumber", how="left")
        tm["candidate"] = (tm["pcur"] >= th_cur) | (tm["psem"] >= th_sem)
        cand_train = tm[tm["candidate"]].copy()

        voof = verifier_oof(cand_train, semext, "log", 60000)
        vth, vdiag = threshold_for_recall(voof, 0.95)
        verifier = fit_verifier(
            cand_train, semext, "log", 80000 + int(outer_fy)
        )

        cur_root = fit_log(train, current)
        sem_root = fit_log(train, semext)
        pcur = proba(cur_root, test, current)
        psem = proba(sem_root, test, semext)
        candidate = (pcur >= th_cur) | (psem >= th_sem)

        root_pred = np.zeros(len(test), dtype=int)
        idx = np.flatnonzero(candidate)
        if len(idx):
            vp = verifier.predict_proba(
                normalize_model_frame(test.iloc[idx][semext])
            )[:, 1]
            root_pred[idx] = (vp >= vth).astype(int)

        root_rows.extend([
            {
                "outer_fy": int(outer_fy),
                "disasterNumber": int(r["disasterNumber"]),
                "actual_high": int(r["target_clean"] >= 50_000_000),
                "pred_high": int(root_pred[j]),
            }
            for j, (_, r) in enumerate(test.iterrows())
        ])

        pred_high_rows = test.loc[root_pred == 1].copy()
        high_map = high22_predict(
            train_high,
            pred_high_rows,
            high_gate_features,
            high_lower_features,
        ) if not pred_high_rows.empty else {}

        for name, (f1, k1, f2, k2) in variants.items():
            low_model = fit_mixed_low(
                train, f1, k1, f2, k2,
                seed=120000 + int(outer_fy),
            )

            true_low = test[test["target_clean"] < 50_000_000].copy()
            if not true_low.empty:
                pp = low_model.predict(true_low)
                for (_, r), p in zip(true_low.iterrows(), pp):
                    oracle[name].append({
                        "disasterNumber": int(r["disasterNumber"]),
                        "actual_band": r["actual_band"],
                        "pred": str(p),
                    })

            pred_low_rows = test.loc[root_pred == 0].copy()
            low_map = {}
            if not pred_low_rows.empty:
                lp = low_model.predict(pred_low_rows)
                low_map = {
                    int(dn): str(p)
                    for dn, p in zip(pred_low_rows["disasterNumber"], lp)
                }

            for j, (_, r) in enumerate(test.iterrows()):
                dn = int(r["disasterNumber"])
                rp = int(root_pred[j])
                final = high_map[dn] if rp else low_map[dn]
                rows[name].append({
                    "disasterNumber": dn,
                    "state": r["state"],
                    "incidentType": r["incidentType"],
                    "fyDeclared": int(r["fyDeclared"]),
                    "actual_band": r["actual_band"],
                    "root_actual_high": int(r["target_clean"] >= 50_000_000),
                    "root_pred_high": rp,
                    "final_pred": final,
                })

    root_df = pd.DataFrame(root_rows)
    root_high_recall = float(recall_score(
        root_df["actual_high"], root_df["pred_high"],
        pos_label=1, zero_division=0
    ))

    results = {}
    for name in variants:
        p = pd.DataFrame(rows[name])
        o = pd.DataFrame(oracle[name])
        p.to_csv(OUT / f"{name}_predictions.csv", index=False)
        o.to_csv(OUT / f"{name}_oracle_low.csv", index=False)
        results[name] = {
            "oracle_low": low_metrics(o, "pred"),
            "end_to_end": six_band_metrics(p, "final_pred"),
        }

    root_df.to_csv(OUT / "root_predictions.csv", index=False)
    ext_audit.to_csv(OUT / "external_match_audit.csv", index=False)

    summary = {
        "scope": "912 non-Biological declarations; Biological excluded",
        "root_high_recall": root_high_recall,
        "root": "nested-recall OR + logistic candidate verifier @ 95% inner recall",
        "high_branch": "frozen 22/23",
        "results": results,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Mixed low-range specialist audit",
        "",
        f"- Root >=$50M recall: **{root_high_recall:.1%}**",
        "- High branch frozen at **22/23**.",
        "",
        "| Variant | Oracle low acc | Oracle macro | 0-100K | 100K-1M | 1M-50M | End-to-end | Overall macro | 50-200M | 200-500M | 500M+ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in results.items():
        o = r["oracle_low"]
        e = r["end_to_end"]
        md.append(
            f"| {name} | {o['accuracy']:.1%} | {o['macro_recall']:.1%} | "
            f"{o['per_band']['0-100K']['correct']}/{o['per_band']['0-100K']['total']} ({o['per_band']['0-100K']['recall']:.1%}) | "
            f"{o['per_band']['100K-1M']['correct']}/{o['per_band']['100K-1M']['total']} ({o['per_band']['100K-1M']['recall']:.1%}) | "
            f"{o['per_band']['1M-50M']['correct']}/{o['per_band']['1M-50M']['total']} ({o['per_band']['1M-50M']['recall']:.1%}) | "
            f"{e['overall_correct']}/{e['overall_total']} ({e['overall_accuracy']:.1%}) | "
            f"{e['macro_recall']:.1%} | "
            f"{e['per_band']['50-200M']['correct']}/{e['per_band']['50-200M']['total']} ({e['per_band']['50-200M']['recall']:.1%}) | "
            f"{e['per_band']['200-500M']['correct']}/{e['per_band']['200-500M']['total']} ({e['per_band']['200-500M']['recall']:.1%}) | "
            f"{e['per_band']['500M+']['correct']}/{e['per_band']['500M+']['total']} ({e['per_band']['500M+']['recall']:.1%}) |"
        )

    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
