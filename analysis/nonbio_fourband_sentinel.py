#!/usr/bin/env python3
"""
Strict nested-LFYO four-band sentinel veto audit.

Upstream root is frozen:
  nested recall-first OR candidate generator
  + logistic candidate verifier calibrated to 95% inner-LFYO high recall

Sentinel model is trained ONLY on >=$1M outer-training rows with four classes:
  1M-50M, 50-200M, 200-500M, 500M+

It can only veto an upstream >=$50M candidate back to the low branch.

Veto threshold:
  derived from inner-LFYO predictions on outer-training data only.
  Let p_low = P(class == 1M-50M).
  The veto threshold is the maximum p_low observed among true >=$50M
  inner-LFYO training cases (plus a tiny epsilon). Therefore no high-value
  inner-training case is vetoed.

Models/features:
  current19 logistic / RF
  semantics logistic / RF
  semantics+external logistic / RF

Low branch:
  semantics RF -> current19 RF with inner macro-F1 thresholds.

High branch:
  frozen 22/23 conditional hierarchy.

Biological excluded/frozen.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score

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
from nonbio_root_recall import (
    fit_log, proba, inner_oof_scores, recall_first_threshold,
)
from nonbio_root_cascade import (
    verifier_oof, threshold_for_recall, fit_verifier,
)
from nonbio_low_thresholds import inner_binary_oof, choose_threshold

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "nonbio_fourband_sentinel"
OUT.mkdir(parents=True, exist_ok=True)

SENTINEL_LABELS = ["1M-50M", "50-200M", "200-500M", "500M+"]


def fit_multi(train: pd.DataFrame, features: List[str], kind: str, seed: int):
    y = train["actual_band"].astype(str)
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


def low_probability(model, df: pd.DataFrame, features: List[str]) -> np.ndarray:
    if df.empty:
        return np.array([], dtype=float)
    probs = model.predict_proba(normalize_model_frame(df[features]))
    classes = list(model.classes_)
    if "1M-50M" not in classes:
        return np.zeros(len(df), dtype=float)
    return probs[:, classes.index("1M-50M")]


def sentinel_oof(
    outer_train: pd.DataFrame,
    features: List[str],
    kind: str,
    seedbase: int,
) -> pd.DataFrame:
    rows = []
    d = outer_train[outer_train["target_clean"] >= 1_000_000].copy()
    for fy in sorted(d["fyDeclared"].astype(int).unique()):
        tr = d[d["fyDeclared"].astype(int) != fy].copy()
        te = d[d["fyDeclared"].astype(int) == fy].copy()
        if te.empty or tr["actual_band"].nunique() < 2:
            continue
        m = fit_multi(tr, features, kind, seedbase + int(fy))
        pp = low_probability(m, te, features)
        for (_, r), p in zip(te.iterrows(), pp):
            rows.append({
                "disasterNumber": int(r["disasterNumber"]),
                "fy": int(fy),
                "actual_band": r["actual_band"],
                "actual_high": int(r["target_clean"] >= 50_000_000),
                "p_low_sentinel": float(p),
            })
    return pd.DataFrame(rows)


def perfect_high_veto_threshold(oof: pd.DataFrame) -> Tuple[float, Dict]:
    high = oof[oof["actual_high"] == 1]
    if high.empty:
        return 1.0, {"inner_high_recall": None, "inner_low_reject": None}
    th = min(1.0, float(high["p_low_sentinel"].max()) + 1e-12)
    keep = oof["p_low_sentinel"] < th
    inner_high_recall = float(
        keep[oof["actual_high"] == 1].mean()
    )
    lows = oof["actual_high"] == 0
    low_reject = float((~keep[lows]).mean()) if lows.any() else None
    return th, {
        "inner_high_recall": inner_high_recall,
        "inner_low_reject": low_reject,
        "inner_high_max_p_low": float(high["p_low_sentinel"].max()),
    }


def fit_low_stage(train, features, y, seed):
    X = normalize_model_frame(train[features])
    model = RandomForestClassifier(
        n_estimators=900,
        random_state=seed,
        class_weight="balanced_subsample",
        max_features="sqrt",
        min_samples_leaf=1,
        n_jobs=-1,
    )
    p = prep_pipeline(X, model)
    p.fit(X, y)
    return p


def positive_proba(model, df, features):
    return model.predict_proba(normalize_model_frame(df[features]))[:, 1]


def build_low_predictor(train, semfeat, current, outer_fy):
    s1_oof = inner_binary_oof(
        train, semfeat, stage=1, kind="rf", seedbase=110000
    )
    s1_th, _ = choose_threshold(s1_oof, "macro_f1")
    s2_oof = inner_binary_oof(
        train, current, stage=2, kind="rf", seedbase=120000
    )
    s2_th, _ = choose_threshold(s2_oof, "macro_f1")

    low = train[train["target_clean"] < 50_000_000].copy()
    y1 = (low["target_clean"] >= 100_000).astype(int)
    s1 = fit_low_stage(low, semfeat, y1, 130000 + int(outer_fy))

    upper = low[low["target_clean"] >= 100_000].copy()
    y2 = (upper["target_clean"] >= 1_000_000).astype(int)
    s2 = fit_low_stage(upper, current, y2, 140000 + int(outer_fy))

    def predict(dd):
        if dd.empty:
            return np.array([], dtype=object)
        p1 = positive_proba(s1, dd, semfeat)
        out = np.full(len(dd), "0-100K", dtype=object)
        ii = np.flatnonzero(p1 >= s1_th)
        if len(ii):
            p2 = positive_proba(s2, dd.iloc[ii], current)
            out[ii] = np.where(
                p2 >= s2_th, "1M-50M", "100K-1M"
            )
        return out.astype(str)

    return predict


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

    feature_sets = {
        "current19": current,
        "semantics": semfeat,
        "semext": semext,
    }
    variants = {
        f"{fname}_{kind}": (feats, kind)
        for fname, feats in feature_sets.items()
        for kind in ["log", "rf"]
    }
    variants["no_sentinel"] = (None, None)

    rows = {k: [] for k in variants}
    diag = []

    for outer_fy in sorted(nonbio["fyDeclared"].astype(int).unique()):
        train = nonbio[nonbio["fyDeclared"].astype(int) != outer_fy].copy()
        test = nonbio[nonbio["fyDeclared"].astype(int) == outer_fy].copy()
        train_high = train[train["target_clean"] >= 50_000_000].copy()

        # Frozen upstream root.
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

        upstream = np.zeros(len(test), dtype=int)
        ci = np.flatnonzero(candidate)
        if len(ci):
            vp = verifier.predict_proba(
                normalize_model_frame(test.iloc[ci][semext])
            )[:,1]
            upstream[ci] = (vp >= vth).astype(int)

        low_predict = build_low_predictor(train, semfeat, current, outer_fy)

        for name, (features, kind) in variants.items():
            final_root = upstream.copy()
            sent_score = np.full(len(test), np.nan)
            sth = np.nan
            sdiag = {}

            if name != "no_sentinel":
                soof = sentinel_oof(
                    train,
                    features,
                    kind,
                    200000 if kind == "log" else 210000,
                )
                sth, sdiag = perfect_high_veto_threshold(soof)
                strain = train[train["target_clean"] >= 1_000_000].copy()
                smodel = fit_multi(
                    strain,
                    features,
                    kind,
                    220000 + int(outer_fy),
                )

                si = np.flatnonzero(upstream == 1)
                if len(si):
                    pp = low_probability(
                        smodel, test.iloc[si], features
                    )
                    sent_score[si] = pp
                    veto = pp >= sth
                    final_root[si[veto]] = 0

            pred_high_rows = test.loc[final_root == 1].copy()
            high_map = high22_predict(
                train_high,
                pred_high_rows,
                high_gate_features,
                high_lower_features,
            ) if not pred_high_rows.empty else {}

            pred_low_rows = test.loc[final_root == 0].copy()
            lp = low_predict(pred_low_rows)
            low_map = {
                int(dn): str(p)
                for dn,p in zip(pred_low_rows["disasterNumber"],lp)
            }

            for j,(_,r) in enumerate(test.iterrows()):
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
                    "upstream_root_high": int(upstream[j]),
                    "root_pred_high": rp,
                    "sentinel_p_low": (
                        float(sent_score[j])
                        if np.isfinite(sent_score[j]) else np.nan
                    ),
                    "final_pred": final,
                })

            diag.append({
                "outer_fy": int(outer_fy),
                "variant": name,
                "sentinel_threshold": (
                    float(sth) if np.isfinite(sth) else np.nan
                ),
                "inner_high_recall": sdiag.get("inner_high_recall"),
                "inner_low_reject": sdiag.get("inner_low_reject"),
                "inner_high_max_p_low": sdiag.get("inner_high_max_p_low"),
            })

    results = {}
    for name in variants:
        p = pd.DataFrame(rows[name])
        p.to_csv(OUT/f"{name}_predictions.csv", index=False)
        y = p["root_actual_high"].to_numpy(int)
        yp = p["root_pred_high"].to_numpy(int)
        results[name] = {
            "root_high_recall": float(recall_score(
                y, yp, pos_label=1, zero_division=0
            )),
            "root_high_precision": float(precision_score(
                y, yp, pos_label=1, zero_division=0
            )),
            "root_fp": int(((y==0)&(yp==1)).sum()),
            "root_fp_1M50M": int(
                ((p["actual_band"]=="1M-50M")
                 &(p["root_pred_high"]==1)).sum()
            ),
            "end_to_end": six_band_metrics(p,"final_pred"),
            "remaining_high_fn": p.loc[
                (p["root_actual_high"]==1)&(p["root_pred_high"]==0),
                ["disasterNumber","state","incidentType","actual_band"],
            ].to_dict(orient="records"),
        }

    pd.DataFrame(diag).to_csv(OUT/"sentinel_fold_diagnostics.csv",index=False)
    ext_audit.to_csv(OUT/"external_match_audit.csv",index=False)

    summary = {
        "scope":"912 non-Biological declarations; Biological excluded",
        "upstream_root":"nested-recall OR + logistic verifier @95% inner recall",
        "sentinel":"four-class >=1M model; one-way low-sentinel veto",
        "low_branch":"semRF -> current19 RF, inner macro-F1 thresholds",
        "high_branch":"frozen 22/23",
        "results":results,
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

    md = [
        "# Four-band sentinel veto audit",
        "",
        "- Sentinel classes: 1M-50M / 50-200M / 200-500M / 500M+.",
        "- One-way veto only.",
        "- Veto threshold preserves all inner-LFYO high training cases.",
        "",
        "| Variant | Root high recall | Root precision | Root FP | FP 1M-50M | End-to-end | Macro recall | 0-100K | 100K-1M | 1M-50M | 50-200M | 200-500M | 500M+ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name,r in results.items():
        e=r["end_to_end"]; pb=e["per_band"]
        md.append(
            f"| {name} | {r['root_high_recall']:.1%} | "
            f"{r['root_high_precision']:.1%} | {r['root_fp']} | "
            f"{r['root_fp_1M50M']} | "
            f"{e['overall_correct']}/{e['overall_total']} ({e['overall_accuracy']:.1%}) | "
            f"{e['macro_recall']:.1%} | "
            + " | ".join(
                f"{pb[b]['correct']}/{pb[b]['total']} ({pb[b]['recall']:.1%})"
                for b in BANDS
            )
            + " |"
        )
    md += ["","## Remaining high-value root false negatives"]
    for name,r in results.items():
        md.append(f"### {name}")
        if not r["remaining_high_fn"]:
            md.append("- **None**")
        else:
            for x in r["remaining_high_fn"]:
                md.append(
                    f"- FEMA {x['disasterNumber']} {x['state']} "
                    f"{x['incidentType']} {x['actual_band']}"
                )

    (OUT/"summary.md").write_text("\n".join(md),encoding="utf-8")
    print("\n".join(md))

if __name__=="__main__":
    main()
