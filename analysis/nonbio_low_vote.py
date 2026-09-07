#!/usr/bin/env python3
"""
Strict-LFYO low-router voting audit.

Three pre-specified low specialists vote:
1. current19 multiclass logistic
2. semantics multiclass random forest
3. nested threshold-calibrated mixed hierarchy:
   - semantics RF: <100K vs >=100K
   - current19 RF: 100K-1M vs 1M-50M
   - both thresholds selected by inner-LFYO macro-F1

Voting is target-blind at outer test time:
- majority vote when two or more agree
- if all three disagree, use the ordinal median band
A second diagnostic uses the mixed hierarchy as tie-breaker.

The >=50M root remains the accepted nested-recall OR + logistic candidate
verifier calibrated to 95% inner recall.
The high branch remains frozen at 22/23 conditional accuracy.
Biological remains excluded/frozen.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter
from typing import List

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
from nonbio_low_thresholds import inner_binary_oof, choose_threshold

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "nonbio_low_vote"
OUT.mkdir(parents=True, exist_ok=True)

LOW_BANDS = ["0-100K", "100K-1M", "1M-50M"]
BAND_IDX = {b: i for i, b in enumerate(LOW_BANDS)}


def fit_clf(train, features, y, kind, seed):
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


def majority_median(a, b, c):
    vals = [str(a), str(b), str(c)]
    cnt = Counter(vals)
    best, n = cnt.most_common(1)[0]
    if n >= 2:
        return best
    idx = sorted(BAND_IDX[x] for x in vals)
    return LOW_BANDS[idx[1]]


def majority_mixed_tiebreak(a, b, c):
    vals = [str(a), str(b), str(c)]
    cnt = Counter(vals)
    best, n = cnt.most_common(1)[0]
    if n >= 2:
        return best
    return str(c)


def low_metrics(df, col):
    pb = {}
    recs = []
    for b in LOW_BANDS:
        m = df["actual_band"] == b
        n = int(m.sum())
        c = int((df.loc[m, col] == b).sum())
        r = c / n if n else None
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

    variant_names = [
        "base_current19_log",
        "base_semantics_rf",
        "base_mixed_macro_f1",
        "vote_majority_median",
        "vote_majority_mixed_tiebreak",
    ]
    rows = {k: [] for k in variant_names}
    oracle = {k: [] for k in variant_names}
    agreement_rows = []

    for outer_fy in sorted(nonbio["fyDeclared"].astype(int).unique()):
        train = nonbio[nonbio["fyDeclared"].astype(int) != outer_fy].copy()
        test = nonbio[nonbio["fyDeclared"].astype(int) == outer_fy].copy()
        train_high = train[train["target_clean"] >= 50_000_000].copy()

        # Frozen accepted root candidate.
        oof_cur = inner_oof_scores(train, current).rename(columns={"prob": "pcur"})
        oof_sem = inner_oof_scores(train, semext).rename(columns={"prob": "psem"})
        th_cur, _ = recall_first_threshold(oof_cur.rename(columns={"pcur": "prob"}))
        th_sem, _ = recall_first_threshold(oof_sem.rename(columns={"psem": "prob"}))
        oo = oof_cur[["disasterNumber", "pcur"]].merge(
            oof_sem[["disasterNumber", "psem"]], on="disasterNumber", how="inner"
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
        ci = np.flatnonzero(candidate)
        if len(ci):
            vp = verifier.predict_proba(
                normalize_model_frame(test.iloc[ci][semext])
            )[:, 1]
            root_pred[ci] = (vp >= vth).astype(int)

        pred_high_rows = test.loc[root_pred == 1].copy()
        high_map = high22_predict(
            train_high, pred_high_rows, high_gate_features, high_lower_features
        ) if not pred_high_rows.empty else {}

        # Base low model 1: current19 multiclass logistic.
        low_train = train[train["target_clean"] < 50_000_000].copy()
        y_multi = low_train["actual_band"].astype(str)
        m_cur_log = fit_clf(
            low_train, current, y_multi, "log", 150000 + int(outer_fy)
        )

        # Base low model 2: semantics multiclass RF.
        m_sem_rf = fit_clf(
            low_train, semfeat, y_multi, "rf", 160000 + int(outer_fy)
        )

        # Base low model 3: mixed threshold-calibrated hierarchy.
        s1_oof = inner_binary_oof(train, semfeat, stage=1, kind="rf", seedbase=110000)
        s1_th, _ = choose_threshold(s1_oof, "macro_f1")
        s2_oof = inner_binary_oof(train, current, stage=2, kind="rf", seedbase=120000)
        s2_th, _ = choose_threshold(s2_oof, "macro_f1")

        y1 = (low_train["target_clean"] >= 100_000).astype(int)
        s1 = fit_clf(low_train, semfeat, y1, "rf", 170000 + int(outer_fy))
        upper_train = low_train[low_train["target_clean"] >= 100_000].copy()
        y2 = (upper_train["target_clean"] >= 1_000_000).astype(int)
        s2 = fit_clf(upper_train, current, y2, "rf", 180000 + int(outer_fy))

        def base_preds(dd):
            if dd.empty:
                z = np.array([], dtype=object)
                return z, z, z

            p1 = m_cur_log.predict(normalize_model_frame(dd[current])).astype(str)
            p2 = m_sem_rf.predict(normalize_model_frame(dd[semfeat])).astype(str)

            q1 = positive_proba(s1, dd, semfeat)
            p3 = np.full(len(dd), "0-100K", dtype=object)
            ii = np.flatnonzero(q1 >= s1_th)
            if len(ii):
                q2 = positive_proba(s2, dd.iloc[ii], current)
                p3[ii] = np.where(q2 >= s2_th, "1M-50M", "100K-1M")
            return p1, p2, p3.astype(str)

        true_low = test[test["target_clean"] < 50_000_000].copy()
        a, b, c = base_preds(true_low)
        vote_med = np.array([majority_median(x,y,z) for x,y,z in zip(a,b,c)], dtype=object)
        vote_mix = np.array([majority_mixed_tiebreak(x,y,z) for x,y,z in zip(a,b,c)], dtype=object)

        oracle_preds = {
            "base_current19_log": a,
            "base_semantics_rf": b,
            "base_mixed_macro_f1": c,
            "vote_majority_median": vote_med,
            "vote_majority_mixed_tiebreak": vote_mix,
        }
        for name, predarr in oracle_preds.items():
            for (_, r), p in zip(true_low.iterrows(), predarr):
                oracle[name].append({
                    "disasterNumber": int(r["disasterNumber"]),
                    "actual_band": r["actual_band"],
                    "pred": str(p),
                })

        for (_, r), x, y, z, vm, vx in zip(
            true_low.iterrows(), a, b, c, vote_med, vote_mix
        ):
            agreement_rows.append({
                "disasterNumber": int(r["disasterNumber"]),
                "fyDeclared": int(r["fyDeclared"]),
                "actual_band": r["actual_band"],
                "current19_log": str(x),
                "semantics_rf": str(y),
                "mixed_macro_f1": str(z),
                "unanimous": bool(x == y == z),
                "all_different": bool(len({x,y,z}) == 3),
                "vote_median": str(vm),
                "vote_mixed_tiebreak": str(vx),
            })

        pred_low_rows = test.loc[root_pred == 0].copy()
        aa, bb, cc = base_preds(pred_low_rows)
        vv_med = np.array([majority_median(x,y,z) for x,y,z in zip(aa,bb,cc)], dtype=object)
        vv_mix = np.array([majority_mixed_tiebreak(x,y,z) for x,y,z in zip(aa,bb,cc)], dtype=object)
        final_low_preds = {
            "base_current19_log": aa,
            "base_semantics_rf": bb,
            "base_mixed_macro_f1": cc,
            "vote_majority_median": vv_med,
            "vote_majority_mixed_tiebreak": vv_mix,
        }

        low_maps = {
            name: {int(dn): str(p) for dn, p in zip(pred_low_rows["disasterNumber"], parr)}
            for name, parr in final_low_preds.items()
        }

        for name in variant_names:
            for j, (_, r) in enumerate(test.iterrows()):
                dn = int(r["disasterNumber"])
                rp = int(root_pred[j])
                final = high_map[dn] if rp else low_maps[name][dn]
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
    for name in variant_names:
        p = pd.DataFrame(rows[name])
        o = pd.DataFrame(oracle[name])
        p.to_csv(OUT / f"{name}_predictions.csv", index=False)
        o.to_csv(OUT / f"{name}_oracle_low.csv", index=False)
        results[name] = {
            "oracle_low": low_metrics(o, "pred"),
            "end_to_end": six_band_metrics(p, "final_pred"),
            "root_high_recall": float(recall_score(
                p["root_actual_high"], p["root_pred_high"],
                pos_label=1, zero_division=0
            )),
        }

    ag = pd.DataFrame(agreement_rows)
    ag.to_csv(OUT / "agreement_audit.csv", index=False)
    agreement_summary = {
        "unanimous_n": int(ag["unanimous"].sum()),
        "unanimous_accuracy": float(
            (ag.loc[ag["unanimous"], "current19_log"] ==
             ag.loc[ag["unanimous"], "actual_band"]).mean()
        ) if ag["unanimous"].any() else None,
        "all_different_n": int(ag["all_different"].sum()),
        "all_different_median_accuracy": float(
            (ag.loc[ag["all_different"], "vote_median"] ==
             ag.loc[ag["all_different"], "actual_band"]).mean()
        ) if ag["all_different"].any() else None,
    }

    ext_audit.to_csv(OUT / "external_match_audit.csv", index=False)
    summary = {
        "scope": "912 non-Biological declarations; Biological excluded",
        "root": "nested-recall OR + logistic verifier @95% inner recall",
        "high_branch": "frozen 22/23",
        "voting_rule": "3-router majority; ordinal median if all three disagree",
        "agreement": agreement_summary,
        "results": results,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Low-router voting audit",
        "",
        "- Three pre-specified low routers vote.",
        "- Majority wins; all-different vote uses ordinal median.",
        f"- Unanimous low cases: **{agreement_summary['unanimous_n']}**",
        f"- All-different low cases: **{agreement_summary['all_different_n']}**",
        "",
        "| Variant | Oracle low acc | Oracle macro | 0-100K | 100K-1M | 1M-50M | End-to-end | Overall macro | 50-200M | 200-500M | 500M+ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in results.items():
        o = r["oracle_low"]; e = r["end_to_end"]
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
