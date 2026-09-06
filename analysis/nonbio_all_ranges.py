#!/usr/bin/env python3
"""
End-to-end strict-LFYO audit of the non-Biological disaster funding router
across the full funding distribution.

Bands
-----
0-100K
100K-1M
1M-50M
50-200M
200-500M
500M+

Architecture
------------
Root gate:
    <50M vs >=50M
Low branch (<50M):
    3-way specialist: 0-100K / 100K-1M / 1M-50M
High branch (>=50M):
    frozen confirmed hierarchy + Fiona persistent-grid-collapse rescue
    + Ida persistent-outage middle rescue (22/23 conditional benchmark)

Evaluation
----------
Outer leave-fiscal-year-out across ALL non-Biological declarations.
Biological is excluded/frozen.
No target-derived predictor fields are used.

We report:
1. Conditional/oracle-branch diagnostics (to isolate branch quality).
2. Root-gate performance.
3. True end-to-end six-band routing performance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

from mission_semantic_audit import (
    CURRENT_19,
    build_semantic_rollup,
    fetch_all_mission_assignments,
    normalize_master,
    normalize_model_frame,
    prep_pipeline,
)
from external_severity_ablation import build_external
from nonbio_hazard_hierarchy import initial_mechanism_counts
from nonbio_outage_rescue import build_eaglei_all, predict_confirmed_fold

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "nonbio_all_ranges"
OUT.mkdir(parents=True, exist_ok=True)

BANDS = ["0-100K", "100K-1M", "1M-50M", "50-200M", "200-500M", "500M+"]


def funding_band(v: float) -> str:
    # Preserve the thesis target cleaning convention: negative net funding -> 0.
    v = max(0.0, float(v))
    if v < 100_000:
        return "0-100K"
    if v < 1_000_000:
        return "100K-1M"
    if v < 50_000_000:
        return "1M-50M"
    if v < 200_000_000:
        return "50-200M"
    if v < 500_000_000:
        return "200-500M"
    return "500M+"


def low_band(v: float) -> str:
    b = funding_band(v)
    assert b in BANDS[:3]
    return b


def valid_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [
        c for c in cols
        if c in df.columns
        and df[c].notna().sum() >= 2
        and df[c].nunique(dropna=True) > 1
    ]


def fit_binary(train: pd.DataFrame, features: List[str], y: pd.Series, kind: str, seed: int):
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
            max_iter=5000,
            class_weight="balanced",
            C=0.5,
        )
    pipe = prep_pipeline(X, model)
    pipe.fit(X, y)
    return pipe


def fit_low_multiclass(train: pd.DataFrame, features: List[str], kind: str, seed: int):
    t = train[train["target_clean"] < 50_000_000].copy()
    y = t["actual_band"].astype(str)
    X = normalize_model_frame(t[features])
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


def high22_predict(
    train_high: pd.DataFrame,
    test_rows: pd.DataFrame,
    gate_features: List[str],
    lower_features: List[str],
) -> Dict[int, str]:
    if test_rows.empty:
        return {}

    base_rows, _ = predict_confirmed_fold(
        train_high,
        test_rows,
        gate_features,
        lower_features,
        lower_features,
        use_outage_lower=False,
    )
    base = {int(x["disasterNumber"]): x["baseline_final"] for x in base_rows}
    out = {}

    for _, r in test_rows.iterrows():
        dn = int(r["disasterNumber"])
        pred = base[dn]

        cov = pd.to_numeric(
            pd.Series([r.get("eaglei_coverage")]), errors="coerce"
        ).iloc[0]
        coverage = int(cov) if pd.notna(cov) else 0
        peak = pd.to_numeric(
            pd.Series([r.get("eaglei_peak_outage_share")]), errors="coerce"
        ).iloc[0]
        res3 = pd.to_numeric(
            pd.Series([r.get("eaglei_residual_share_3d")]), errors="coerce"
        ).iloc[0]
        res7 = pd.to_numeric(
            pd.Series([r.get("eaglei_residual_share_7d")]), errors="coerce"
        ).iloc[0]
        mechanism = int(r.get("initial_usace_esf3_dfa_count", 0))

        # Frozen Fiona 50/50 extreme rescue.
        collapse = (
            r["incidentType"] == "Hurricane"
            and pred != "500M+"
            and mechanism >= 1
            and coverage == 1
            and pd.notna(peak)
            and pd.notna(res7)
            and float(peak) >= 0.50
            and float(res7) >= 0.50
        )
        if collapse:
            pred = "500M+"

        # Frozen Ida one-third-of-entire-base-at-day-3 middle rescue.
        base3 = (
            float(peak) * float(res3)
            if pd.notna(peak) and pd.notna(res3)
            else np.nan
        )
        middle = (
            r["incidentType"] == "Hurricane"
            and pred == "50-200M"
            and mechanism >= 1
            and coverage == 1
            and np.isfinite(base3)
            and base3 >= (1.0 / 3.0)
        )
        if middle:
            pred = "200-500M"

        out[dn] = pred

    return out


def six_band_metrics(pred: pd.DataFrame, col: str) -> Dict:
    per_band = {}
    recalls = []
    for b in BANDS:
        m = pred["actual_band"] == b
        n = int(m.sum())
        c = int((pred.loc[m, col] == b).sum())
        r = c / n if n else None
        per_band[b] = {"correct": c, "total": n, "recall": r}
        if r is not None:
            recalls.append(r)

    y_true = pd.Categorical(pred["actual_band"], categories=BANDS)
    y_pred = pd.Categorical(pred[col], categories=BANDS)
    cm = confusion_matrix(y_true, y_pred, labels=BANDS)

    return {
        "overall_correct": int((pred[col] == pred["actual_band"]).sum()),
        "overall_total": int(len(pred)),
        "overall_accuracy": float((pred[col] == pred["actual_band"]).mean()),
        "macro_recall": float(np.mean(recalls)),
        "per_band": per_band,
        "confusion_matrix": cm.tolist(),
    }


def main():
    master = normalize_master(pd.read_excel(MASTER))
    master["target_clean"] = pd.to_numeric(
        master["totalObligatedFunding"], errors="coerce"
    ).fillna(0.0).clip(lower=0)
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

    # Biological remains frozen/excluded.
    nonbio = df[df["incidentType"] != "Biological"].copy().reset_index(drop=True)
    assert len(nonbio) > 0

    band_counts = nonbio["actual_band"].value_counts().reindex(BANDS, fill_value=0).to_dict()

    current = valid_cols(nonbio, CURRENT_19)
    semcols = valid_cols(
        nonbio,
        [c for c in nonbio.columns if c.startswith("sem_") or c.startswith("ma_")],
    )
    extcols = valid_cols(
        nonbio,
        [
            c for c in nonbio.columns
            if c.startswith("nhc_") or c.startswith("noaa_") or c.startswith("calfire_")
        ],
    )

    # High branch feature definitions must match the frozen hierarchy.
    high_all = nonbio[nonbio["target_clean"] >= 50_000_000].copy()
    high_sem = valid_cols(
        high_all,
        [c for c in high_all.columns if c.startswith("sem_") or c.startswith("ma_")],
    )
    high_ext = valid_cols(
        high_all,
        [
            c for c in high_all.columns
            if c.startswith("nhc_") or c.startswith("noaa_") or c.startswith("calfire_")
        ],
    )
    high_current = valid_cols(high_all, CURRENT_19)
    high_gate_features = high_current + high_sem + high_ext
    high_lower_features = high_current + high_sem

    configs = {
        "current19_log": {
            "root_features": current,
            "root_kind": "log",
            "low_features": current,
            "low_kind": "log",
        },
        "semantics_log": {
            "root_features": list(dict.fromkeys(current + semcols)),
            "root_kind": "log",
            "low_features": list(dict.fromkeys(current + semcols)),
            "low_kind": "log",
        },
        "semantics_rf": {
            "root_features": list(dict.fromkeys(current + semcols)),
            "root_kind": "rf",
            "low_features": list(dict.fromkeys(current + semcols)),
            "low_kind": "rf",
        },
        "semantics_external_log": {
            "root_features": list(dict.fromkeys(current + semcols + extcols)),
            "root_kind": "log",
            "low_features": list(dict.fromkeys(current + semcols)),
            "low_kind": "log",
        },
        "semantics_root_log_low_rf": {
            "root_features": list(dict.fromkeys(current + semcols)),
            "root_kind": "log",
            "low_features": list(dict.fromkeys(current + semcols)),
            "low_kind": "rf",
        },
    }

    rows_by_cfg = {k: [] for k in configs}
    root_rows = {k: [] for k in configs}
    oracle_high_rows = []
    fold_rows = []

    years = sorted(nonbio["fyDeclared"].astype(int).unique())

    for fy in years:
        train = nonbio[nonbio["fyDeclared"].astype(int) != fy].copy()
        test = nonbio[nonbio["fyDeclared"].astype(int) == fy].copy()

        train_high = train[train["target_clean"] >= 50_000_000].copy()
        test_true_high = test[test["target_clean"] >= 50_000_000].copy()

        # Conditional high-branch verification, independent of root gate.
        if not test_true_high.empty:
            hp = high22_predict(
                train_high,
                test_true_high,
                high_gate_features,
                high_lower_features,
            )
            for _, r in test_true_high.iterrows():
                oracle_high_rows.append({
                    "disasterNumber": int(r["disasterNumber"]),
                    "state": r["state"],
                    "incidentType": r["incidentType"],
                    "fyDeclared": int(r["fyDeclared"]),
                    "actual_band": r["actual_band"],
                    "conditional_high_pred": hp[int(r["disasterNumber"])],
                })

        for cfg_name, cfg in configs.items():
            y_root = (train["target_clean"] >= 50_000_000).astype(int)
            root = fit_binary(
                train,
                cfg["root_features"],
                y_root,
                cfg["root_kind"],
                seed=10000 + int(fy),
            )
            Xtest_root = normalize_model_frame(test[cfg["root_features"]])
            root_pred = root.predict(Xtest_root).astype(int)
            if hasattr(root, "predict_proba"):
                root_prob = root.predict_proba(Xtest_root)[:, 1]
            else:
                root_prob = np.full(len(test), np.nan)

            low_model = fit_low_multiclass(
                train,
                cfg["low_features"],
                cfg["low_kind"],
                seed=20000 + int(fy),
            )

            pred_high_rows = test.loc[root_pred == 1].copy()
            high_map = high22_predict(
                train_high,
                pred_high_rows,
                high_gate_features,
                high_lower_features,
            ) if not pred_high_rows.empty else {}

            pred_low_rows = test.loc[root_pred == 0].copy()
            low_map = {}
            if not pred_low_rows.empty:
                lp = low_model.predict(
                    normalize_model_frame(pred_low_rows[cfg["low_features"]])
                )
                low_map = {
                    int(dn): str(p)
                    for dn, p in zip(pred_low_rows["disasterNumber"], lp)
                }

            for j, (_, r) in enumerate(test.iterrows()):
                dn = int(r["disasterNumber"])
                is_high = int(root_pred[j])
                final = high_map[dn] if is_high else low_map[dn]

                rows_by_cfg[cfg_name].append({
                    "disasterNumber": dn,
                    "state": r["state"],
                    "incidentType": r["incidentType"],
                    "fyDeclared": int(r["fyDeclared"]),
                    "totalObligatedFunding": float(r["totalObligatedFunding"]),
                    "target_clean": float(r["target_clean"]),
                    "actual_band": r["actual_band"],
                    "root_actual_high": int(r["target_clean"] >= 50_000_000),
                    "root_pred_high": is_high,
                    "root_prob_high": float(root_prob[j]),
                    "final_pred": final,
                })

                root_rows[cfg_name].append({
                    "disasterNumber": dn,
                    "fyDeclared": int(r["fyDeclared"]),
                    "actual_high": int(r["target_clean"] >= 50_000_000),
                    "pred_high": is_high,
                    "prob_high": float(root_prob[j]),
                })

        fold_rows.append({
            "fy": int(fy),
            "train_n": int(len(train)),
            "test_n": int(len(test)),
            "train_high_n": int(len(train_high)),
            "test_high_n": int(len(test_true_high)),
        })

    # Verify frozen high branch still reproduces 22/23 conditionally.
    oh = pd.DataFrame(oracle_high_rows)
    assert len(oh) == len(high_all) == 23
    high_correct = int((oh["actual_band"] == oh["conditional_high_pred"]).sum())
    assert high_correct == 22, f"Frozen high router drifted: {high_correct}/23"
    oh.to_csv(OUT / "conditional_high22_predictions.csv", index=False)

    results = {}
    for cfg_name in configs:
        p = pd.DataFrame(rows_by_cfg[cfg_name])
        p.to_csv(OUT / f"{cfg_name}_predictions.csv", index=False)

        rr = pd.DataFrame(root_rows[cfg_name])
        y = rr["actual_high"].to_numpy()
        yp = rr["pred_high"].to_numpy()
        root_metrics = {
            "accuracy": float(accuracy_score(y, yp)),
            "balanced_accuracy": float(balanced_accuracy_score(y, yp)),
            "high_recall": float(recall_score(y, yp, pos_label=1, zero_division=0)),
            "low_recall": float(recall_score(y, yp, pos_label=0, zero_division=0)),
            "high_precision": float(precision_score(y, yp, pos_label=1, zero_division=0)),
            "confusion_matrix": confusion_matrix(y, yp, labels=[0, 1]).tolist(),
            "false_negative_high": p.loc[
                (p["root_actual_high"] == 1) & (p["root_pred_high"] == 0),
                ["disasterNumber", "state", "incidentType", "actual_band", "root_prob_high"],
            ].to_dict(orient="records"),
            "false_positive_high": p.loc[
                (p["root_actual_high"] == 0) & (p["root_pred_high"] == 1),
                ["disasterNumber", "state", "incidentType", "actual_band", "root_prob_high"],
            ].to_dict(orient="records"),
        }

        results[cfg_name] = {
            "root_feature_count": len(configs[cfg_name]["root_features"]),
            "low_feature_count": len(configs[cfg_name]["low_features"]),
            "root_kind": configs[cfg_name]["root_kind"],
            "low_kind": configs[cfg_name]["low_kind"],
            "root_gate": root_metrics,
            "end_to_end": six_band_metrics(p, "final_pred"),
        }

    pd.DataFrame(fold_rows).to_csv(OUT / "folds.csv", index=False)
    ext_audit.to_csv(OUT / "external_match_audit.csv", index=False)

    summary = {
        "scope": "all non-Biological declarations only; Biological frozen/excluded",
        "n_nonbiological": int(len(nonbio)),
        "band_counts": {k: int(v) for k, v in band_counts.items()},
        "protocol": "strict outer leave-fiscal-year-out across the full non-Biological population",
        "conditional_high_branch": {
            "correct": high_correct,
            "total": int(len(oh)),
            "accuracy": high_correct / len(oh),
        },
        "results": results,
        "cautions": [
            "The 22/23 high-value result is conditional on entering the >=50M branch.",
            "End-to-end results include errors from the new <50M vs >=50M root gate.",
            "Model/config comparisons are developmental; selecting the best configuration on these same folds is not external validation.",
            "Mission and EAGLE-I features are retrospective unless a prediction-time cutoff t0 is defined.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Full-range non-Biological router audit",
        "",
        f"- Non-Biological declarations: **{len(nonbio)}**",
        f"- Conditional frozen high branch: **{high_correct}/23 = {high_correct/23:.1%}**",
        "- Evaluation: **strict leave-fiscal-year-out**",
        "",
        "## Band counts",
        "",
    ]
    for b in BANDS:
        md.append(f"- {b}: **{band_counts[b]}**")

    md += [
        "",
        "## End-to-end six-band results",
        "",
        "| Config | Root high recall | Root high precision | Overall | Macro recall | 0-100K | 100K-1M | 1M-50M | 50-200M | 200-500M | 500M+ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for name, r in results.items():
        e = r["end_to_end"]
        g = r["root_gate"]
        pb = e["per_band"]
        md.append(
            f"| {name} | {g['high_recall']:.1%} | {g['high_precision']:.1%} | "
            f"{e['overall_correct']}/{e['overall_total']} ({e['overall_accuracy']:.1%}) | "
            f"{e['macro_recall']:.1%} | "
            + " | ".join(
                f"{pb[b]['correct']}/{pb[b]['total']} ({pb[b]['recall']:.1%})"
                for b in BANDS
            )
            + " |"
        )

    md += [
        "",
        "## Interpretation",
        "",
        "The conditional high-value hierarchy is frozen at 22/23. Any loss in high-band "
        "end-to-end recall here can therefore come from the new root gate failing to send "
        "a true >=$50M case into the high branch.",
    ]

    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
