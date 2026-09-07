#!/usr/bin/env python3
"""
Two-stage >=$50M entry cascade under strict outer LFYO.

Stage A: nested-recall OR candidate generator
  - current19 logistic
  - current19 + mission semantics + external severity logistic
  - thresholds chosen from outer-training inner-LFYO predictions to include all
    known >=$50M outer-training cases

Stage B: candidate verifier
  - trained ONLY on Stage-A candidate rows from the outer-training set
  - verifier threshold chosen from leave-fiscal-year-out verifier predictions
    on those candidate rows to preserve 100% high-value recall in training
  - compares logistic and random-forest verifiers

The outer held-out fiscal year is untouched until final scoring.
Biological remains excluded/frozen.
The downstream high-value hierarchy is frozen at 22/23 conditional accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    precision_score, recall_score,
)

from mission_semantic_audit import (
    CURRENT_19, build_semantic_rollup, fetch_all_mission_assignments,
    normalize_master, normalize_model_frame, prep_pipeline,
)
from external_severity_ablation import build_external
from nonbio_hazard_hierarchy import initial_mechanism_counts
from nonbio_outage_rescue import build_eaglei_all
from nonbio_all_ranges import (
    BANDS, funding_band, high22_predict, fit_low_multiclass,
    six_band_metrics, valid_cols,
)
from nonbio_root_recall import fit_log, proba, inner_oof_scores, recall_first_threshold

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "nonbio_root_cascade"
OUT.mkdir(parents=True, exist_ok=True)


def fit_verifier(train: pd.DataFrame, features: List[str], kind: str, seed: int):
    y = (train["target_clean"] >= 50_000_000).astype(int)
    X = normalize_model_frame(train[features])
    if kind == "rf":
        model = RandomForestClassifier(
            n_estimators=700,
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
    pipe = prep_pipeline(X, model)
    pipe.fit(X, y)
    return pipe


def verifier_oof(
    candidate_train: pd.DataFrame,
    features: List[str],
    kind: str,
    seed_base: int,
) -> pd.DataFrame:
    rows = []
    years = sorted(candidate_train["fyDeclared"].astype(int).unique())
    for fy in years:
        tr = candidate_train[candidate_train["fyDeclared"].astype(int) != fy].copy()
        te = candidate_train[candidate_train["fyDeclared"].astype(int) == fy].copy()
        ytr = (tr["target_clean"] >= 50_000_000).astype(int)
        if te.empty or ytr.nunique() < 2:
            continue
        model = fit_verifier(tr, features, kind, seed_base + int(fy))
        pp = model.predict_proba(
            normalize_model_frame(te[features])
        )[:, 1]
        for (_, r), p in zip(te.iterrows(), pp):
            rows.append({
                "disasterNumber": int(r["disasterNumber"]),
                "fy": int(fy),
                "actual_high": int(r["target_clean"] >= 50_000_000),
                "prob": float(p),
            })
    return pd.DataFrame(rows)


def threshold_for_recall(oof: pd.DataFrame, target_recall: float) -> Tuple[float, Dict]:
    if oof.empty or int(oof["actual_high"].sum()) == 0:
        return 0.5, {"recall": None, "precision": None, "fp": None}

    pos_probs = np.sort(oof.loc[oof["actual_high"] == 1, "prob"].to_numpy())
    thresholds = sorted(set(np.r_[0.0, pos_probs, 0.5]), reverse=True)
    feasible = []
    for th in thresholds:
        pred = (oof["prob"] >= th).astype(int)
        rec = recall_score(oof["actual_high"], pred, pos_label=1, zero_division=0)
        prec = precision_score(oof["actual_high"], pred, pos_label=1, zero_division=0)
        fp = int(((oof["actual_high"] == 0) & (pred == 1)).sum())
        if rec + 1e-12 >= target_recall:
            feasible.append((prec, -fp, th, rec))
    if not feasible:
        return 0.0, {"recall": 1.0, "precision": float(oof["actual_high"].mean()), "fp": int((oof["actual_high"] == 0).sum())}
    feasible.sort(reverse=True)
    prec, negfp, th, rec = feasible[0]
    return float(th), {
        "recall": float(rec),
        "precision": float(prec),
        "fp": int(-negfp),
    }


def root_metrics(df: pd.DataFrame) -> Dict:
    y = df["root_actual_high"].to_numpy(int)
    yp = df["root_pred_high"].to_numpy(int)
    return {
        "accuracy": float(accuracy_score(y, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(y, yp)),
        "high_recall": float(recall_score(y, yp, pos_label=1, zero_division=0)),
        "high_precision": float(precision_score(y, yp, pos_label=1, zero_division=0)),
        "low_recall": float(recall_score(y, yp, pos_label=0, zero_division=0)),
        "false_positive_count": int(((y == 0) & (yp == 1)).sum()),
        "confusion_matrix": confusion_matrix(y, yp, labels=[0, 1]).tolist(),
        "false_negative_high": df.loc[
            (df["root_actual_high"] == 1) & (df["root_pred_high"] == 0),
            ["disasterNumber", "state", "incidentType", "actual_band", "root_score"],
        ].to_dict(orient="records"),
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
    extcols = valid_cols(
        nonbio, [c for c in nonbio.columns
                 if c.startswith("nhc_") or c.startswith("noaa_") or c.startswith("calfire_")]
    )
    semext = list(dict.fromkeys(current + semcols + extcols))

    high_all = nonbio[nonbio["target_clean"] >= 50_000_000].copy()
    hc = valid_cols(high_all, CURRENT_19)
    hs = valid_cols(high_all, [c for c in high_all.columns if c.startswith("sem_") or c.startswith("ma_")])
    he = valid_cols(high_all, [c for c in high_all.columns
                               if c.startswith("nhc_") or c.startswith("noaa_") or c.startswith("calfire_")])
    high_gate_features = hc + hs + he
    high_lower_features = hc + hs

    variants = {
        "cascade_log_100recall": ("log", 1.00),
        "cascade_rf_100recall": ("rf", 1.00),
        "cascade_log_95recall": ("log", 0.95),
        "cascade_rf_95recall": ("rf", 0.95),
    }
    rows = {k: [] for k in variants}
    fold_diag = []

    for outer_fy in sorted(nonbio["fyDeclared"].astype(int).unique()):
        train = nonbio[nonbio["fyDeclared"].astype(int) != outer_fy].copy()
        test = nonbio[nonbio["fyDeclared"].astype(int) == outer_fy].copy()
        train_high = train[train["target_clean"] >= 50_000_000].copy()

        # Candidate generator thresholds from inner LFYO.
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

        # Fit global candidate-generator models for outer test.
        cur = fit_log(train, current)
        semm = fit_log(train, semext)
        pcur_test = proba(cur, test, current)
        psem_test = proba(semm, test, semext)
        test_candidate = (pcur_test >= th_cur) | (psem_test >= th_sem)

        # Fixed low specialist for non-high final routing.
        low_model = fit_low_multiclass(
            train, current, "log", seed=50000 + int(outer_fy)
        )

        for name, (kind, target_recall) in variants.items():
            # Cross-validated verifier threshold using candidate outer-training rows only.
            voof = verifier_oof(
                cand_train, semext, kind,
                seed_base=(60000 if kind == "log" else 70000)
            )
            vth, vdiag = threshold_for_recall(voof, target_recall)

            verifier = fit_verifier(
                cand_train, semext, kind,
                seed=(80000 if kind == "log" else 90000) + int(outer_fy)
            )

            final_root = np.zeros(len(test), dtype=int)
            root_score = np.zeros(len(test), dtype=float)
            cand_idx = np.flatnonzero(test_candidate)
            if len(cand_idx):
                cand_test = test.iloc[cand_idx]
                vp = verifier.predict_proba(
                    normalize_model_frame(cand_test[semext])
                )[:, 1]
                root_score[cand_idx] = vp
                final_root[cand_idx] = (vp >= vth).astype(int)

            # Non-candidates remain low.
            pred_high_rows = test.loc[final_root == 1].copy()
            high_map = high22_predict(
                train_high, pred_high_rows, high_gate_features, high_lower_features
            ) if not pred_high_rows.empty else {}

            pred_low_rows = test.loc[final_root == 0].copy()
            low_map = {}
            if not pred_low_rows.empty:
                lp = low_model.predict(
                    normalize_model_frame(pred_low_rows[current])
                )
                low_map = {
                    int(dn): str(p)
                    for dn, p in zip(pred_low_rows["disasterNumber"], lp)
                }

            for j, (_, r) in enumerate(test.iterrows()):
                dn = int(r["disasterNumber"])
                rp = int(final_root[j])
                final = high_map[dn] if rp else low_map[dn]
                rows[name].append({
                    "disasterNumber": dn,
                    "state": r["state"],
                    "incidentType": r["incidentType"],
                    "fyDeclared": int(r["fyDeclared"]),
                    "actual_band": r["actual_band"],
                    "root_actual_high": int(r["target_clean"] >= 50_000_000),
                    "candidate_generated": bool(test_candidate[j]),
                    "root_pred_high": rp,
                    "root_score": float(root_score[j]),
                    "p_candidate_current": float(pcur_test[j]),
                    "p_candidate_semext": float(psem_test[j]),
                    "final_pred": final,
                })

            fold_diag.append({
                "outer_fy": int(outer_fy),
                "variant": name,
                "candidate_train_n": int(len(cand_train)),
                "candidate_train_high_n": int((cand_train["target_clean"] >= 50_000_000).sum()),
                "candidate_test_n": int(test_candidate.sum()),
                "candidate_threshold_current": float(th_cur),
                "candidate_threshold_semext": float(th_sem),
                "verifier_threshold": float(vth),
                "verifier_oof_recall": vdiag.get("recall"),
                "verifier_oof_precision": vdiag.get("precision"),
                "verifier_oof_fp": vdiag.get("fp"),
            })

    results = {}
    for name in variants:
        p = pd.DataFrame(rows[name])
        p.to_csv(OUT / f"{name}_predictions.csv", index=False)
        results[name] = {
            "root": root_metrics(p),
            "end_to_end": six_band_metrics(p, "final_pred"),
            "candidate_generator": {
                "high_recall": float(recall_score(
                    p["root_actual_high"], p["candidate_generated"].astype(int),
                    pos_label=1, zero_division=0
                )),
                "candidate_count": int(p["candidate_generated"].sum()),
                "false_positive_candidates": int(
                    ((p["root_actual_high"] == 0) & p["candidate_generated"]).sum()
                ),
            },
        }

    pd.DataFrame(fold_diag).to_csv(OUT / "fold_diagnostics.csv", index=False)
    ext_audit.to_csv(OUT / "external_match_audit.csv", index=False)

    summary = {
        "scope": "912 non-Biological declarations; Biological excluded",
        "frozen_high_branch": "22/23",
        "results": results,
        "caution": (
            "Candidate verifier selection remains developmental. Outer LFYO is untouched, "
            "but this is not an external validation cohort."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Two-stage recall-first >=$50M root cascade",
        "",
        "- Stage A: nested-recall OR candidate generator.",
        "- Stage B: candidate verifier.",
        "- High branch frozen at **22/23**.",
        "",
        "| Variant | Candidate high recall | Candidate FP | Final high recall | Final high precision | Root FP | End-to-end | Macro recall | 0-100K | 100K-1M | 1M-50M | 50-200M | 200-500M | 500M+ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in results.items():
        c = r["candidate_generator"]
        g = r["root"]
        e = r["end_to_end"]
        pb = e["per_band"]
        md.append(
            f"| {name} | {c['high_recall']:.1%} | {c['false_positive_candidates']} | "
            f"{g['high_recall']:.1%} | {g['high_precision']:.1%} | {g['false_positive_count']} | "
            f"{e['overall_correct']}/{e['overall_total']} ({e['overall_accuracy']:.1%}) | "
            f"{e['macro_recall']:.1%} | "
            + " | ".join(
                f"{pb[b]['correct']}/{pb[b]['total']} ({pb[b]['recall']:.1%})"
                for b in BANDS
            )
            + " |"
        )

    md += ["", "## Remaining final root false negatives"]
    for name, r in results.items():
        md.append(f"### {name}")
        errs = r["root"]["false_negative_high"]
        if not errs:
            md.append("- **None**")
        else:
            for x in errs:
                md.append(
                    f"- FEMA {x['disasterNumber']} {x['state']} {x['incidentType']} "
                    f"{x['actual_band']} score={x['root_score']:.4f}"
                )

    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
