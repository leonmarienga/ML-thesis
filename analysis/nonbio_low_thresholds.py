#!/usr/bin/env python3
"""
Strict nested-LFYO threshold calibration for the mixed low-range router.

Frozen architecture
-------------------
Root >=50M:
  nested-recall OR candidate generator + logistic verifier calibrated to 95%
  inner-LFYO recall (same accepted root candidate).

High branch:
  frozen 22/23 conditional hierarchy.

Low branch:
  Stage 1: semantics Random Forest, 0-100K vs >=100K
  Stage 2: current19 Random Forest OR logistic, 100K-1M vs 1M-50M

New element
-----------
Each low-stage probability threshold is selected ONLY from inner LFYO
predictions on the outer-training set. We compare:
- fixed 0.50
- threshold maximizing balanced accuracy
- threshold maximizing macro F1
- threshold maximizing Youden J (TPR + TNR - 1)

No outer held-out fiscal-year labels are used for threshold selection.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, recall_score
)

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
OUT = ROOT / "audit_outputs" / "nonbio_low_thresholds"
OUT.mkdir(parents=True, exist_ok=True)

LOW_BANDS = ["0-100K", "100K-1M", "1M-50M"]


def fit_binary(train, features, y, kind, seed):
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
        model = LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5)
    pipe = prep_pipeline(X, model)
    pipe.fit(X, y)
    return pipe


def positive_proba(model, df, features):
    return model.predict_proba(normalize_model_frame(df[features]))[:, 1]


def inner_binary_oof(train, features, stage, kind, seedbase):
    rows = []
    years = sorted(train["fyDeclared"].astype(int).unique())
    for fy in years:
        tr0 = train[train["fyDeclared"].astype(int) != fy].copy()
        te0 = train[train["fyDeclared"].astype(int) == fy].copy()

        if stage == 1:
            tr = tr0[tr0["target_clean"] < 50_000_000].copy()
            te = te0[te0["target_clean"] < 50_000_000].copy()
            ytr = (tr["target_clean"] >= 100_000).astype(int)
            yte = (te["target_clean"] >= 100_000).astype(int)
        else:
            tr = tr0[(tr0["target_clean"] >= 100_000) & (tr0["target_clean"] < 50_000_000)].copy()
            te = te0[(te0["target_clean"] >= 100_000) & (te0["target_clean"] < 50_000_000)].copy()
            ytr = (tr["target_clean"] >= 1_000_000).astype(int)
            yte = (te["target_clean"] >= 1_000_000).astype(int)

        if te.empty or ytr.nunique() < 2:
            continue
        model = fit_binary(tr, features, ytr, kind, seedbase + int(fy))
        pp = positive_proba(model, te, features)
        for (_, r), y, p in zip(te.iterrows(), yte, pp):
            rows.append({
                "disasterNumber": int(r["disasterNumber"]),
                "fy": int(fy),
                "y": int(y),
                "p": float(p),
            })
    return pd.DataFrame(rows)


def choose_threshold(oof, objective):
    if oof.empty or oof["y"].nunique() < 2:
        return 0.5, {"score": None, "sensitivity": None, "specificity": None}

    candidates = np.unique(np.r_[0.05, np.arange(0.10, 0.91, 0.02), 0.95, oof["p"].to_numpy()])
    candidates = candidates[(candidates >= 0.02) & (candidates <= 0.98)]

    best = None
    for th in candidates:
        yp = (oof["p"].to_numpy() >= th).astype(int)
        y = oof["y"].to_numpy()
        sens = recall_score(y, yp, pos_label=1, zero_division=0)
        spec = recall_score(y, yp, pos_label=0, zero_division=0)
        if objective == "balanced":
            score = balanced_accuracy_score(y, yp)
        elif objective == "macro_f1":
            score = f1_score(y, yp, average="macro", zero_division=0)
        elif objective == "youden":
            score = sens + spec - 1.0
        else:
            score = -abs(th - 0.5)

        # tie-break toward threshold nearer 0.5, then higher specificity
        key = (float(score), -abs(float(th)-0.5), float(spec))
        if best is None or key > best[0]:
            best = (key, float(th), float(score), float(sens), float(spec))

    _, th, score, sens, spec = best
    return th, {
        "score": score,
        "sensitivity": sens,
        "specificity": spec,
    }


def low_metrics(df, col):
    pb = {}
    recs = []
    for b in LOW_BANDS:
        m = df["actual_band"] == b
        n = int(m.sum())
        c = int((df.loc[m, col] == b).sum())
        r = c/n if n else None
        pb[b] = {"correct": c, "total": n, "recall": r}
        if r is not None:
            recs.append(r)
    return {
        "accuracy": float((df[col] == df["actual_band"]).mean()),
        "macro_recall": float(np.mean(recs)),
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
    df["initial_usace_esf3_dfa_count"] = df["initial_usace_esf3_dfa_count"].fillna(0).astype(int)
    nonbio = df[df["incidentType"] != "Biological"].copy().reset_index(drop=True)

    current = valid_cols(nonbio, CURRENT_19)
    semcols = valid_cols(nonbio, [c for c in nonbio.columns if c.startswith("sem_") or c.startswith("ma_")])
    semfeat = list(dict.fromkeys(current + semcols))
    extcols = valid_cols(nonbio, [c for c in nonbio.columns
                                  if c.startswith("nhc_") or c.startswith("noaa_") or c.startswith("calfire_")])
    semext = list(dict.fromkeys(semfeat + extcols))

    high_all = nonbio[nonbio["target_clean"] >= 50_000_000].copy()
    hc = valid_cols(high_all, CURRENT_19)
    hs = valid_cols(high_all, [c for c in high_all.columns if c.startswith("sem_") or c.startswith("ma_")])
    he = valid_cols(high_all, [c for c in high_all.columns
                               if c.startswith("nhc_") or c.startswith("noaa_") or c.startswith("calfire_")])
    high_gate_features = hc + hs + he
    high_lower_features = hc + hs

    variants = {}
    for stage2_kind in ["rf", "log"]:
        for objective in ["fixed", "balanced", "macro_f1", "youden"]:
            variants[f"semRF_cur{stage2_kind.upper()}_{objective}"] = (stage2_kind, objective)

    rows = {k: [] for k in variants}
    oracle = {k: [] for k in variants}
    threshold_rows = []

    for outer_fy in sorted(nonbio["fyDeclared"].astype(int).unique()):
        train = nonbio[nonbio["fyDeclared"].astype(int) != outer_fy].copy()
        test = nonbio[nonbio["fyDeclared"].astype(int) == outer_fy].copy()
        train_high = train[train["target_clean"] >= 50_000_000].copy()

        # Frozen accepted root candidate.
        oof_cur = inner_oof_scores(train, current).rename(columns={"prob":"pcur"})
        oof_sem = inner_oof_scores(train, semext).rename(columns={"prob":"psem"})
        th_cur, _ = recall_first_threshold(oof_cur.rename(columns={"pcur":"prob"}))
        th_sem, _ = recall_first_threshold(oof_sem.rename(columns={"psem":"prob"}))
        oo = oof_cur[["disasterNumber","pcur"]].merge(
            oof_sem[["disasterNumber","psem"]], on="disasterNumber", how="inner"
        )
        tm = train.merge(oo, on="disasterNumber", how="left")
        tm["candidate"] = (tm["pcur"] >= th_cur) | (tm["psem"] >= th_sem)
        cand_train = tm[tm["candidate"]].copy()

        voof = verifier_oof(cand_train, semext, "log", 60000)
        vth, _ = threshold_for_recall(voof, 0.95)
        verifier = fit_verifier(cand_train, semext, "log", 80000 + int(outer_fy))

        cur_root = fit_log(train, current)
        sem_root = fit_log(train, semext)
        pcur = proba(cur_root, test, current)
        psem = proba(sem_root, test, semext)
        candidate = (pcur >= th_cur) | (psem >= th_sem)
        root_pred = np.zeros(len(test), dtype=int)
        idx = np.flatnonzero(candidate)
        if len(idx):
            vp = verifier.predict_proba(normalize_model_frame(test.iloc[idx][semext]))[:,1]
            root_pred[idx] = (vp >= vth).astype(int)

        pred_high_rows = test.loc[root_pred == 1].copy()
        high_map = high22_predict(
            train_high, pred_high_rows, high_gate_features, high_lower_features
        ) if not pred_high_rows.empty else {}

        # Stage 1 inner OOF once: semantics RF.
        s1_oof = inner_binary_oof(train, semfeat, stage=1, kind="rf", seedbase=110000)

        for name, (stage2_kind, objective) in variants.items():
            if objective == "fixed":
                s1_th = 0.5
                s1_diag = {"score":None,"sensitivity":None,"specificity":None}
            else:
                s1_th, s1_diag = choose_threshold(s1_oof, objective)

            s2_oof = inner_binary_oof(train, current, stage=2, kind=stage2_kind, seedbase=120000)
            if objective == "fixed":
                s2_th = 0.5
                s2_diag = {"score":None,"sensitivity":None,"specificity":None}
            else:
                s2_th, s2_diag = choose_threshold(s2_oof, objective)

            low_train = train[train["target_clean"] < 50_000_000].copy()
            y1 = (low_train["target_clean"] >= 100_000).astype(int)
            s1 = fit_binary(low_train, semfeat, y1, "rf", 130000 + int(outer_fy))

            upper_train = low_train[low_train["target_clean"] >= 100_000].copy()
            y2 = (upper_train["target_clean"] >= 1_000_000).astype(int)
            s2 = fit_binary(upper_train, current, y2, stage2_kind, 140000 + int(outer_fy))

            def predict_low(dd):
                if dd.empty:
                    return np.array([], dtype=object)
                p1 = positive_proba(s1, dd, semfeat)
                out = np.full(len(dd), "0-100K", dtype=object)
                ii = np.flatnonzero(p1 >= s1_th)
                if len(ii):
                    p2 = positive_proba(s2, dd.iloc[ii], current)
                    out[ii] = np.where(p2 >= s2_th, "1M-50M", "100K-1M")
                return out.astype(str)

            true_low = test[test["target_clean"] < 50_000_000].copy()
            op = predict_low(true_low)
            for (_, r), p in zip(true_low.iterrows(), op):
                oracle[name].append({
                    "disasterNumber": int(r["disasterNumber"]),
                    "actual_band": r["actual_band"],
                    "pred": str(p),
                })

            pred_low_rows = test.loc[root_pred == 0].copy()
            lp = predict_low(pred_low_rows)
            low_map = {int(dn):str(p) for dn,p in zip(pred_low_rows["disasterNumber"],lp)}

            for j,(_,r) in enumerate(test.iterrows()):
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

            threshold_rows.append({
                "outer_fy": int(outer_fy),
                "variant": name,
                "stage1_threshold": float(s1_th),
                "stage1_inner_sensitivity": s1_diag.get("sensitivity"),
                "stage1_inner_specificity": s1_diag.get("specificity"),
                "stage2_threshold": float(s2_th),
                "stage2_inner_sensitivity": s2_diag.get("sensitivity"),
                "stage2_inner_specificity": s2_diag.get("specificity"),
            })

    results = {}
    for name in variants:
        p = pd.DataFrame(rows[name])
        o = pd.DataFrame(oracle[name])
        p.to_csv(OUT/f"{name}_predictions.csv", index=False)
        o.to_csv(OUT/f"{name}_oracle_low.csv", index=False)
        results[name] = {
            "oracle_low": low_metrics(o,"pred"),
            "end_to_end": six_band_metrics(p,"final_pred"),
            "root_high_recall": float(recall_score(
                p["root_actual_high"], p["root_pred_high"],
                pos_label=1, zero_division=0
            )),
        }

    pd.DataFrame(threshold_rows).to_csv(OUT/"thresholds.csv", index=False)
    ext_audit.to_csv(OUT/"external_match_audit.csv", index=False)

    summary = {
        "scope":"912 non-Biological declarations; Biological excluded",
        "root":"nested-recall OR + logistic verifier @95% inner recall",
        "high_branch":"frozen 22/23",
        "results":results,
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

    md = [
        "# Nested threshold-calibrated low router",
        "",
        "- Stage 1: semantics RF, 0-100K vs >=100K.",
        "- Stage 2: current19 RF/log, 100K-1M vs 1M-50M.",
        "- Thresholds selected by inner LFYO only.",
        "",
        "| Variant | Oracle low acc | Oracle macro | 0-100K | 100K-1M | 1M-50M | End-to-end | Overall macro | 50-200M | 200-500M | 500M+ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name,r in results.items():
        o=r["oracle_low"]; e=r["end_to_end"]
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
    (OUT/"summary.md").write_text("\n".join(md),encoding="utf-8")
    print("\n".join(md))

if __name__=="__main__":
    main()
