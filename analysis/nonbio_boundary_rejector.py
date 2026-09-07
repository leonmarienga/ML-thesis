#!/usr/bin/env python3
"""
Strict nested-LFYO $1M-$50M vs >=$50M boundary rejector audit.

Frozen upstream root:
- nested recall-first OR candidate generator
- logistic candidate verifier calibrated to 95% inner-LFYO high recall

New boundary rejector:
- applied ONLY to rows already selected by the frozen root as >=$50M
- may only reject a candidate back to the low branch
- trained on outer-training rows in either:
    A) >=$1M only
    B) >=$100K only
- target: >=$50M vs below-$50M
- threshold chosen from inner-LFYO OOF predictions as the highest threshold
  that preserves 100% high-value recall in the outer-training data.

Models/features compared:
- current19 logistic / RF
- semantics logistic / RF
- semantics+external logistic / RF

Low branch:
- frozen best balanced mixed hierarchy:
  Stage 1 semantics RF
  Stage 2 current19 RF
  inner-LFYO macro-F1 thresholds

High branch:
- frozen 22/23 conditional hierarchy

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
from sklearn.metrics import (
    recall_score, precision_score, confusion_matrix,
)

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
from nonbio_low_thresholds import (
    inner_binary_oof, choose_threshold,
)

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "nonbio_boundary_rejector"
OUT.mkdir(parents=True, exist_ok=True)

LOW_BANDS = ["0-100K", "100K-1M", "1M-50M"]


def fit_boundary(train, features, kind, seed):
    y = (train["target_clean"] >= 50_000_000).astype(int)
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


def boundary_oof(
    outer_train: pd.DataFrame,
    features: List[str],
    kind: str,
    min_funding: float,
    seedbase: int,
) -> pd.DataFrame:
    rows = []
    for fy in sorted(outer_train["fyDeclared"].astype(int).unique()):
        tr0 = outer_train[outer_train["fyDeclared"].astype(int) != fy].copy()
        te0 = outer_train[outer_train["fyDeclared"].astype(int) == fy].copy()

        tr = tr0[tr0["target_clean"] >= min_funding].copy()
        te = te0[te0["target_clean"] >= min_funding].copy()
        ytr = (tr["target_clean"] >= 50_000_000).astype(int)
        yte = (te["target_clean"] >= 50_000_000).astype(int)

        if te.empty or ytr.nunique() < 2:
            continue

        m = fit_boundary(tr, features, kind, seedbase + int(fy))
        pp = m.predict_proba(
            normalize_model_frame(te[features])
        )[:, 1]

        for (_, r), y, p in zip(te.iterrows(), yte, pp):
            rows.append({
                "disasterNumber": int(r["disasterNumber"]),
                "fy": int(fy),
                "actual_high": int(y),
                "prob": float(p),
            })
    return pd.DataFrame(rows)


def perfect_recall_threshold(oof: pd.DataFrame) -> Tuple[float, Dict]:
    if oof.empty or int(oof["actual_high"].sum()) == 0:
        return 0.0, {"recall": None, "precision": None, "fp": None}

    threshold = float(oof.loc[oof["actual_high"] == 1, "prob"].min())
    yp = (oof["prob"] >= threshold).astype(int)

    return threshold, {
        "recall": float(recall_score(
            oof["actual_high"], yp, pos_label=1, zero_division=0
        )),
        "precision": float(precision_score(
            oof["actual_high"], yp, pos_label=1, zero_division=0
        )),
        "fp": int(((oof["actual_high"] == 0) & (yp == 1)).sum()),
        "tn": int(((oof["actual_high"] == 0) & (yp == 0)).sum()),
    }


def fit_low_stage(train, features, y, kind, seed):
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
    p = prep_pipeline(X, model)
    p.fit(X, y)
    return p


def positive_proba(model, df, features):
    return model.predict_proba(normalize_model_frame(df[features]))[:, 1]


def build_low_predictor(train, semfeat, current, outer_fy):
    # Frozen best-balanced low hierarchy: sem RF -> current RF, macro-F1 thresholds.
    s1_oof = inner_binary_oof(
        train, semfeat, stage=1, kind="rf", seedbase=110000
    )
    s1_th, _ = choose_threshold(s1_oof, "macro_f1")

    s2_oof = inner_binary_oof(
        train, current, stage=2, kind="rf", seedbase=120000
    )
    s2_th, _ = choose_threshold(s2_oof, "macro_f1")

    low_train = train[train["target_clean"] < 50_000_000].copy()
    y1 = (low_train["target_clean"] >= 100_000).astype(int)
    s1 = fit_low_stage(
        low_train, semfeat, y1, "rf", 130000 + int(outer_fy)
    )

    upper = low_train[low_train["target_clean"] >= 100_000].copy()
    y2 = (upper["target_clean"] >= 1_000_000).astype(int)
    s2 = fit_low_stage(
        upper, current, y2, "rf", 140000 + int(outer_fy)
    )

    def predict(dd):
        if dd.empty:
            return np.array([], dtype=object)
        p1 = positive_proba(s1, dd, semfeat)
        out = np.full(len(dd), "0-100K", dtype=object)
        idx = np.flatnonzero(p1 >= s1_th)
        if len(idx):
            p2 = positive_proba(s2, dd.iloc[idx], current)
            out[idx] = np.where(
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

    variants = {}
    for scope_name, min_funding in [("1Mplus", 1_000_000), ("100Kplus", 100_000)]:
        for feat_name in ["current19", "semantics", "semext"]:
            for kind in ["log", "rf"]:
                variants[f"{scope_name}_{feat_name}_{kind}"] = (
                    min_funding, feature_sets[feat_name], kind
                )

    # Also retain no-reject baseline.
    variants["no_boundary_rejector"] = (None, None, None)

    rows = {name: [] for name in variants}
    diagnostics = []

    for outer_fy in sorted(nonbio["fyDeclared"].astype(int).unique()):
        train = nonbio[nonbio["fyDeclared"].astype(int) != outer_fy].copy()
        test = nonbio[nonbio["fyDeclared"].astype(int) == outer_fy].copy()
        train_high = train[train["target_clean"] >= 50_000_000].copy()

        # Frozen upstream accepted root.
        oof_cur = inner_oof_scores(train, current).rename(columns={"prob": "pcur"})
        oof_sem = inner_oof_scores(train, semext).rename(columns={"prob": "psem"})
        th_cur, _ = recall_first_threshold(
            oof_cur.rename(columns={"pcur": "prob"})
        )
        th_sem, _ = recall_first_threshold(
            oof_sem.rename(columns={"psem": "prob"})
        )

        oo = oof_cur[["disasterNumber", "pcur"]].merge(
            oof_sem[["disasterNumber", "psem"]],
            on="disasterNumber",
            how="inner",
        )
        tm = train.merge(oo, on="disasterNumber", how="left")
        tm["candidate"] = (
            (tm["pcur"] >= th_cur) | (tm["psem"] >= th_sem)
        )
        cand_train = tm[tm["candidate"]].copy()

        voof = verifier_oof(cand_train, semext, "log", 60000)
        vth, _ = threshold_for_recall(voof, 0.95)
        verifier = fit_verifier(
            cand_train, semext, "log", 80000 + int(outer_fy)
        )

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

        # Frozen low predictor fitted once for this outer fold.
        low_predict = build_low_predictor(
            train, semfeat, current, outer_fy
        )

        for name, (min_funding, features, kind) in variants.items():
            final_root = root_pred.copy()
            boundary_score = np.full(len(test), np.nan)
            bth = np.nan
            bdiag = {}

            if name != "no_boundary_rejector":
                boof = boundary_oof(
                    train,
                    features,
                    kind,
                    min_funding,
                    seedbase=200000 if kind == "log" else 210000,
                )
                bth, bdiag = perfect_recall_threshold(boof)

                btrain = train[train["target_clean"] >= min_funding].copy()
                bmodel = fit_boundary(
                    btrain,
                    features,
                    kind,
                    220000 + int(outer_fy),
                )

                selected_idx = np.flatnonzero(root_pred == 1)
                if len(selected_idx):
                    bp = bmodel.predict_proba(
                        normalize_model_frame(
                            test.iloc[selected_idx][features]
                        )
                    )[:, 1]
                    boundary_score[selected_idx] = bp
                    keep = bp >= bth
                    final_root[selected_idx] = keep.astype(int)

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
                    "upstream_root_high": int(root_pred[j]),
                    "root_pred_high": rp,
                    "boundary_score": (
                        float(boundary_score[j])
                        if np.isfinite(boundary_score[j])
                        else np.nan
                    ),
                    "final_pred": final,
                })

            diagnostics.append({
                "outer_fy": int(outer_fy),
                "variant": name,
                "boundary_threshold": (
                    float(bth) if np.isfinite(bth) else np.nan
                ),
                "boundary_oof_recall": bdiag.get("recall"),
                "boundary_oof_precision": bdiag.get("precision"),
                "boundary_oof_fp": bdiag.get("fp"),
                "boundary_oof_tn": bdiag.get("tn"),
            })

    results = {}
    for name in variants:
        p = pd.DataFrame(rows[name])
        p.to_csv(OUT / f"{name}_predictions.csv", index=False)

        y = p["root_actual_high"].to_numpy(int)
        yp = p["root_pred_high"].to_numpy(int)
        fp = int(((y == 0) & (yp == 1)).sum())
        fn = int(((y == 1) & (yp == 0)).sum())

        results[name] = {
            "root_high_recall": float(recall_score(
                y, yp, pos_label=1, zero_division=0
            )),
            "root_high_precision": float(precision_score(
                y, yp, pos_label=1, zero_division=0
            )),
            "root_fp": fp,
            "root_fn": fn,
            "root_fp_by_band": {
                b: int(
                    ((p["actual_band"] == b)
                     & (p["root_actual_high"] == 0)
                     & (p["root_pred_high"] == 1)).sum()
                )
                for b in LOW_BANDS
            },
            "end_to_end": six_band_metrics(p, "final_pred"),
            "remaining_root_false_negatives": p.loc[
                (p["root_actual_high"] == 1)
                & (p["root_pred_high"] == 0),
                ["disasterNumber", "state", "incidentType", "actual_band"],
            ].to_dict(orient="records"),
        }

    pd.DataFrame(diagnostics).to_csv(
        OUT / "boundary_fold_diagnostics.csv", index=False
    )
    ext_audit.to_csv(OUT / "external_match_audit.csv", index=False)

    summary = {
        "scope": "912 non-Biological declarations; Biological excluded",
        "upstream_root": (
            "nested-recall OR + logistic candidate verifier @95% inner recall"
        ),
        "boundary_rule": (
            "one-way rejection of upstream >=50M candidates; "
            "inner-LFYO threshold preserves 100% high training recall"
        ),
        "low_branch": (
            "semantics RF -> current19 RF with inner macro-F1 thresholds"
        ),
        "high_branch": "frozen 22/23",
        "results": results,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    md = [
        "# $1M-$50M vs >=$50M boundary rejector audit",
        "",
        "- Upstream root frozen.",
        "- Rejector is one-way only.",
        "- Rejector threshold preserves 100% inner-LFYO high recall.",
        "- High branch frozen at **22/23**.",
        "",
        "| Variant | Root high recall | Root precision | Root FP | FP 1M-50M | End-to-end | Macro recall | 0-100K | 100K-1M | 1M-50M | 50-200M | 200-500M | 500M+ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for name, r in results.items():
        e = r["end_to_end"]
        pb = e["per_band"]
        md.append(
            f"| {name} | {r['root_high_recall']:.1%} | "
            f"{r['root_high_precision']:.1%} | {r['root_fp']} | "
            f"{r['root_fp_by_band']['1M-50M']} | "
            f"{e['overall_correct']}/{e['overall_total']} "
            f"({e['overall_accuracy']:.1%}) | "
            f"{e['macro_recall']:.1%} | "
            + " | ".join(
                f"{pb[b]['correct']}/{pb[b]['total']} "
                f"({pb[b]['recall']:.1%})"
                for b in BANDS
            )
            + " |"
        )

    md += ["", "## Remaining root false negatives"]
    for name, r in results.items():
        md.append(f"### {name}")
        if not r["remaining_root_false_negatives"]:
            md.append("- **None**")
        else:
            for x in r["remaining_root_false_negatives"]:
                md.append(
                    f"- FEMA {x['disasterNumber']} {x['state']} "
                    f"{x['incidentType']} {x['actual_band']}"
                )

    (OUT / "summary.md").write_text(
        "\n".join(md), encoding="utf-8"
    )
    print("\n".join(md))


if __name__ == "__main__":
    main()
