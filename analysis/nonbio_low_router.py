#!/usr/bin/env python3
"""
Strict-LFYO low-range specialist audit using the accepted recall-first root candidate.

Root is fixed to:
- Stage A nested-recall OR candidate generator
- Stage B logistic candidate verifier calibrated to 95% inner-LFYO high recall
This was the best high-tail-preserving compromise in the prior audit.

Low branch variants:
1. current19 multiclass logistic
2. semantics multiclass logistic
3. current19 multiclass RF
4. semantics multiclass RF
5. current19 ordinal logistic
6. semantics ordinal logistic
7. current19 ordinal RF
8. semantics ordinal RF

Low bands:
0-100K
100K-1M
1M-50M

Biological remains excluded/frozen.
High branch remains frozen at 22/23 conditional accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score

from mission_semantic_audit import (
    CURRENT_19, build_semantic_rollup, fetch_all_mission_assignments,
    normalize_master, normalize_model_frame, prep_pipeline,
)
from external_severity_ablation import build_external
from nonbio_hazard_hierarchy import initial_mechanism_counts
from nonbio_outage_rescue import build_eaglei_all
from nonbio_all_ranges import (
    BANDS, funding_band, high22_predict, six_band_metrics, valid_cols,
)
from nonbio_root_recall import fit_log, proba, inner_oof_scores, recall_first_threshold
from nonbio_root_cascade import verifier_oof, threshold_for_recall, fit_verifier

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "nonbio_low_router"
OUT.mkdir(parents=True, exist_ok=True)

LOW_BANDS = ["0-100K", "100K-1M", "1M-50M"]


def fit_clf(train: pd.DataFrame, features: List[str], y: pd.Series, kind: str, seed: int):
    X = normalize_model_frame(train[features])
    if kind == "rf":
        model = RandomForestClassifier(
            n_estimators=800,
            random_state=seed,
            class_weight="balanced_subsample",
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=-1,
        )
    else:
        model = LogisticRegression(
            max_iter=5000, class_weight="balanced", C=0.5
        )
    p = prep_pipeline(X, model)
    p.fit(X, y)
    return p


class MulticlassLow:
    def __init__(self, model, features):
        self.model = model
        self.features = features
    def predict(self, df):
        return self.model.predict(normalize_model_frame(df[self.features])).astype(str)


class OrdinalLow:
    def __init__(self, stage1, stage2, features):
        self.stage1 = stage1
        self.stage2 = stage2
        self.features = features
    def predict(self, df):
        X = normalize_model_frame(df[self.features])
        p1 = self.stage1.predict(X).astype(int)
        out = np.full(len(df), "0-100K", dtype=object)
        idx = np.flatnonzero(p1 == 1)
        if len(idx):
            X2 = normalize_model_frame(df.iloc[idx][self.features])
            p2 = self.stage2.predict(X2).astype(int)
            out[idx] = np.where(p2 == 1, "1M-50M", "100K-1M")
        return out.astype(str)


def fit_low_model(train: pd.DataFrame, features: List[str], kind: str, architecture: str, seed: int):
    t = train[train["target_clean"] < 50_000_000].copy()
    if architecture == "multiclass":
        y = t["actual_band"].astype(str)
        m = fit_clf(t, features, y, kind, seed)
        return MulticlassLow(m, features)

    # Ordinal stage 1: <100K vs >=100K
    y1 = (t["target_clean"] >= 100_000).astype(int)
    m1 = fit_clf(t, features, y1, kind, seed + 1000)

    # Ordinal stage 2: 100K-1M vs 1M-50M
    t2 = t[t["target_clean"] >= 100_000].copy()
    y2 = (t2["target_clean"] >= 1_000_000).astype(int)
    m2 = fit_clf(t2, features, y2, kind, seed + 2000)
    return OrdinalLow(m1, m2, features)


def low_metrics(df: pd.DataFrame, pred_col: str) -> Dict:
    out = {}
    recs = []
    for b in LOW_BANDS:
        m = df["actual_band"] == b
        n = int(m.sum())
        c = int((df.loc[m, pred_col] == b).sum())
        r = c / n if n else None
        out[b] = {"correct": c, "total": n, "recall": r}
        if r is not None:
            recs.append(r)
    return {
        "accuracy": float((df[pred_col] == df["actual_band"]).mean()),
        "macro_recall": float(np.mean(recs)),
        "per_band": out,
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
        nonbio, [c for c in nonbio.columns if c.startswith("sem_") or c.startswith("ma_")]
    )
    semfeat = list(dict.fromkeys(current + semcols))
    extcols = valid_cols(
        nonbio, [c for c in nonbio.columns
                 if c.startswith("nhc_") or c.startswith("noaa_") or c.startswith("calfire_")]
    )
    semext = list(dict.fromkeys(semfeat + extcols))

    high_all = nonbio[nonbio["target_clean"] >= 50_000_000].copy()
    hc = valid_cols(high_all, CURRENT_19)
    hs = valid_cols(high_all, [c for c in high_all.columns if c.startswith("sem_") or c.startswith("ma_")])
    he = valid_cols(high_all, [c for c in high_all.columns
                               if c.startswith("nhc_") or c.startswith("noaa_") or c.startswith("calfire_")])
    high_gate_features = hc + hs + he
    high_lower_features = hc + hs

    variants = {
        "current19_multiclass_log": (current, "log", "multiclass"),
        "semantics_multiclass_log": (semfeat, "log", "multiclass"),
        "current19_multiclass_rf": (current, "rf", "multiclass"),
        "semantics_multiclass_rf": (semfeat, "rf", "multiclass"),
        "current19_ordinal_log": (current, "log", "ordinal"),
        "semantics_ordinal_log": (semfeat, "log", "ordinal"),
        "current19_ordinal_rf": (current, "rf", "ordinal"),
        "semantics_ordinal_rf": (semfeat, "rf", "ordinal"),
    }

    rows = {k: [] for k in variants}
    oracle_low = {k: [] for k in variants}
    root_diag = []

    for outer_fy in sorted(nonbio["fyDeclared"].astype(int).unique()):
        train = nonbio[nonbio["fyDeclared"].astype(int) != outer_fy].copy()
        test = nonbio[nonbio["fyDeclared"].astype(int) == outer_fy].copy()
        train_high = train[train["target_clean"] >= 50_000_000].copy()

        # Accepted root candidate: nested OR + logistic verifier at 95% inner recall.
        oof_cur = inner_oof_scores(train, current).rename(columns={"prob": "pcur"})
        oof_sem = inner_oof_scores(train, semext).rename(columns={"prob": "psem"})
        th_cur, _ = recall_first_threshold(oof_cur.rename(columns={"pcur": "prob"}))
        th_sem, _ = recall_first_threshold(oof_sem.rename(columns={"psem": "prob"}))

        oo = oof_cur[["disasterNumber", "pcur"]].merge(
            oof_sem[["disasterNumber", "psem"]], on="disasterNumber", how="inner"
        )
        train_meta = train.merge(oo, on="disasterNumber", how="left")
        train_meta["candidate"] = (
            (train_meta["pcur"] >= th_cur) | (train_meta["psem"] >= th_sem)
        )
        cand_train = train_meta[train_meta["candidate"]].copy()

        voof = verifier_oof(cand_train, semext, "log", 60000)
        vth, vdiag = threshold_for_recall(voof, 0.95)
        verifier = fit_verifier(cand_train, semext, "log", 80000 + int(outer_fy))

        cur = fit_log(train, current)
        semm = fit_log(train, semext)
        pcur_test = proba(cur, test, current)
        psem_test = proba(semm, test, semext)
        candidate = (pcur_test >= th_cur) | (psem_test >= th_sem)

        root_pred = np.zeros(len(test), dtype=int)
        cand_idx = np.flatnonzero(candidate)
        if len(cand_idx):
            vp = verifier.predict_proba(
                normalize_model_frame(test.iloc[cand_idx][semext])
            )[:, 1]
            root_pred[cand_idx] = (vp >= vth).astype(int)

        root_diag.append({
            "outer_fy": int(outer_fy),
            "candidate_n": int(candidate.sum()),
            "root_high_n": int(root_pred.sum()),
            "verifier_threshold": float(vth),
            "verifier_oof_recall": vdiag.get("recall"),
            "verifier_oof_precision": vdiag.get("precision"),
        })

        pred_high_rows = test.loc[root_pred == 1].copy()
        high_map = high22_predict(
            train_high, pred_high_rows, high_gate_features, high_lower_features
        ) if not pred_high_rows.empty else {}

        for name, (features, kind, architecture) in variants.items():
            low_model = fit_low_model(
                train, features, kind, architecture,
                seed=100000 + int(outer_fy)
            )

            # Oracle low-branch diagnostic on all true <50M cases.
            true_low = test[test["target_clean"] < 50_000_000].copy()
            if not true_low.empty:
                op = low_model.predict(true_low)
                for (_, r), p in zip(true_low.iterrows(), op):
                    oracle_low[name].append({
                        "disasterNumber": int(r["disasterNumber"]),
                        "actual_band": r["actual_band"],
                        "oracle_low_pred": str(p),
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

    results = {}
    for name in variants:
        p = pd.DataFrame(rows[name])
        op = pd.DataFrame(oracle_low[name])
        p.to_csv(OUT / f"{name}_predictions.csv", index=False)
        op.to_csv(OUT / f"{name}_oracle_low.csv", index=False)
        results[name] = {
            "oracle_low": low_metrics(op.rename(columns={"oracle_low_pred": "pred"}), "pred"),
            "end_to_end": six_band_metrics(p, "final_pred"),
            "root_high_recall": float(recall_score(
                p["root_actual_high"], p["root_pred_high"],
                pos_label=1, zero_division=0
            )),
        }

    pd.DataFrame(root_diag).to_csv(OUT / "root_fold_diagnostics.csv", index=False)
    ext_audit.to_csv(OUT / "external_match_audit.csv", index=False)

    summary = {
        "scope": "912 non-Biological declarations; Biological excluded",
        "root": "nested-recall OR + logistic candidate verifier calibrated to 95% inner recall",
        "high_branch": "frozen 22/23 conditional hierarchy",
        "results": results,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Low-range specialist audit",
        "",
        "- Root fixed to the best high-tail-preserving cascade candidate.",
        "- High branch frozen at **22/23**.",
        "- Low branch evaluated both conditionally (oracle <50M) and end-to-end.",
        "",
        "| Variant | Oracle low acc | Oracle low macro | Oracle 0-100K | Oracle 100K-1M | Oracle 1M-50M | End-to-end | Overall macro | 50-200M | 200-500M | 500M+ |",
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
