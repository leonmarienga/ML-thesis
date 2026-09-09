#!/usr/bin/env python3
"""
Strict-LFYO focused Hurricane operational-depth verifier.

This is a development experiment motivated by the diagnostic mechanism audit,
but it does NOT copy thresholds from that audit.

Baseline:
- deterministic independent $1M lower boundary
- global semext $50M sentinel
- confirmed binary Hurricane semext veto
- frozen high-value hierarchy

New second Hurricane veto:
uses only a small, interpretable response-depth feature set:
- ma_max_amendment
- sem_topic_power_count
- sem_long_90d_count
- initial_usace_esf3_dfa_count
- sem_type_dfa_count
- missionAssignmentCount

Models tested:
- logistic
- random forest

For each outer LFYO fold, the verifier threshold is selected only from
inner-LFYO predictions on outer-training Hurricanes >=$1M and must preserve
100% of true >=$50M Hurricane training cases.

The verifier is one-way veto only and is applied AFTER the already-confirmed
binary Hurricane semext veto.

Biological remains excluded/frozen.
"""

from __future__ import annotations
import json
from pathlib import Path

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
from nonbio_all_ranges import funding_band, high22_predict, six_band_metrics, valid_cols
from nonbio_root_recall import fit_log, proba, inner_oof_scores, recall_first_threshold
from nonbio_root_cascade import verifier_oof, threshold_for_recall, fit_verifier
from nonbio_fourband_sentinel import sentinel_oof, perfect_high_veto_threshold, fit_multi, low_probability
from nonbio_hurricane_boundary_confirm import (
    data_hash, build_low_predictor,
    fit_hurricane_binary, hurricane_oof, safe_threshold,
)

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "nonbio_hurricane_depth_verifier"
OUT.mkdir(parents=True, exist_ok=True)

DEPTH_FEATURES = [
    "ma_max_amendment",
    "sem_topic_power_count",
    "sem_long_90d_count",
    "initial_usace_esf3_dfa_count",
    "sem_type_dfa_count",
    "missionAssignmentCount",
]


def fit_depth_binary(train: pd.DataFrame, features, kind: str, seed: int):
    y = (train["target_clean"] >= 50_000_000).astype(int)
    X = normalize_model_frame(train[features])
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


def depth_oof(outer_train: pd.DataFrame, features, kind: str, seedbase: int):
    d = outer_train[
        (outer_train["incidentType"] == "Hurricane")
        & (outer_train["target_clean"] >= 1_000_000)
    ].copy()
    rows = []
    for fy in sorted(d["fyDeclared"].astype(int).unique()):
        tr = d[d["fyDeclared"].astype(int) != fy].copy()
        te = d[d["fyDeclared"].astype(int) == fy].copy()
        ytr = (tr["target_clean"] >= 50_000_000).astype(int)
        if te.empty or ytr.nunique() < 2:
            continue
        model = fit_depth_binary(tr, features, kind, seedbase + fy)
        probs = model.predict_proba(
            normalize_model_frame(te[features])
        )[:, 1]
        for (_, r), p in zip(te.iterrows(), probs):
            rows.append({
                "disasterNumber": int(r["disasterNumber"]),
                "fy": int(fy),
                "actual_high": int(r["target_clean"] >= 50_000_000),
                "prob_high": float(p),
            })
    return pd.DataFrame(rows)


def safe_keep_threshold(oof: pd.DataFrame):
    if oof.empty or int(oof["actual_high"].sum()) == 0:
        return 0.0, {"recall": None, "precision": None, "fp": None}
    th = float(oof.loc[oof["actual_high"] == 1, "prob_high"].min())
    pred = (oof["prob_high"] >= th).astype(int)
    return th, {
        "recall": float(recall_score(
            oof["actual_high"], pred, pos_label=1, zero_division=0
        )),
        "precision": float(precision_score(
            oof["actual_high"], pred, pos_label=1, zero_division=0
        )),
        "fp": int(((oof["actual_high"] == 0) & (pred == 1)).sum()),
    }


def main():
    master = normalize_master(pd.read_excel(MASTER))
    master["target_clean"] = pd.to_numeric(
        master["totalObligatedFunding"], errors="coerce"
    ).fillna(0).clip(lower=0)
    master["actual_band"] = master["target_clean"].map(funding_band)

    ma = fetch_all_mission_assignments()
    ma_hash = data_hash(ma)
    sem, _ = build_semantic_rollup(master, ma)
    ext, ext_audit = build_external(master)
    mech = initial_mechanism_counts(master, ma)
    print("Mission nonfinancial snapshot SHA256:", ma_hash, flush=True)
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

    depth_features = [c for c in DEPTH_FEATURES if c in nonbio.columns]
    missing = [c for c in DEPTH_FEATURES if c not in nonbio.columns]
    if missing:
        raise RuntimeError(f"Missing depth features: {missing}")

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

    variants = ["binary_only", "binary_plus_depth_log", "binary_plus_depth_rf"]
    rows = {v: [] for v in variants}
    diagnostics = []

    for fy in sorted(nonbio["fyDeclared"].astype(int).unique()):
        train = nonbio[nonbio["fyDeclared"].astype(int) != fy].copy()
        test = nonbio[nonbio["fyDeclared"].astype(int) == fy].copy()
        train_high = train[train["target_clean"] >= 50_000_000].copy()

        # Shared upstream root.
        oof_cur = inner_oof_scores(train, current).rename(columns={"prob": "pcur"})
        oof_sem = inner_oof_scores(train, semext).rename(columns={"prob": "psem"})
        th_cur, _ = recall_first_threshold(oof_cur.rename(columns={"pcur": "prob"}))
        th_sem, _ = recall_first_threshold(oof_sem.rename(columns={"psem": "prob"}))

        oo = oof_cur[["disasterNumber", "pcur"]].merge(
            oof_sem[["disasterNumber", "psem"]],
            on="disasterNumber", how="inner",
        )
        tm = train.merge(oo, on="disasterNumber", how="left")
        tm["candidate"] = (tm["pcur"] >= th_cur) | (tm["psem"] >= th_sem)
        cand_train = tm[tm["candidate"]].copy()

        voof = verifier_oof(cand_train, semext, "log", 60000)
        vth, _ = threshold_for_recall(voof, 0.95)
        verifier = fit_verifier(cand_train, semext, "log", 80000 + fy)

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
            )[:, 1]
            upstream[ci] = (vp >= vth).astype(int)

        # Global sentinel.
        soof = sentinel_oof(train, semext, "log", 200000)
        sth, _ = perfect_high_veto_threshold(soof)
        strain = train[train["target_clean"] >= 1_000_000].copy()
        smodel = fit_multi(strain, semext, "log", 220000 + fy)
        global_root = upstream.copy()
        si = np.flatnonzero(upstream == 1)
        if len(si):
            lowp = low_probability(smodel, test.iloc[si], semext)
            global_root[si[lowp >= sth]] = 0

        # Confirmed binary Hurricane veto.
        bin_oof = hurricane_oof(train, semext)
        bin_th, _ = safe_threshold(bin_oof)
        htrain = train[
            (train["incidentType"] == "Hurricane")
            & (train["target_clean"] >= 1_000_000)
        ].copy()
        hy = (htrain["target_clean"] >= 50_000_000).astype(int)
        bin_model = (
            fit_hurricane_binary(htrain, semext, 420000 + fy)
            if len(htrain) >= 6 and hy.nunique() >= 2
            else None
        )

        binary_root = global_root.copy()
        hidx = np.flatnonzero(
            (binary_root == 1)
            & (test["incidentType"].to_numpy() == "Hurricane")
        )
        if len(hidx) and bin_model is not None:
            bp = bin_model.predict_proba(
                normalize_model_frame(test.iloc[hidx][semext])
            )[:, 1]
            binary_root[hidx[bp < bin_th]] = 0

        low_predict = build_low_predictor(train, semfeat, fy)

        roots = {"binary_only": binary_root.copy()}

        for kind in ["log", "rf"]:
            oof = depth_oof(
                train, depth_features, kind,
                500000 if kind == "log" else 510000,
            )
            dth, ddiag = safe_keep_threshold(oof)
            dtrain = htrain.copy()
            dy = (dtrain["target_clean"] >= 50_000_000).astype(int)
            dmodel = (
                fit_depth_binary(
                    dtrain, depth_features, kind,
                    520000 + fy if kind == "log" else 530000 + fy,
                )
                if len(dtrain) >= 6 and dy.nunique() >= 2
                else None
            )

            root = binary_root.copy()
            idx = np.flatnonzero(
                (root == 1)
                & (test["incidentType"].to_numpy() == "Hurricane")
            )
            if len(idx) and dmodel is not None:
                dp = dmodel.predict_proba(
                    normalize_model_frame(test.iloc[idx][depth_features])
                )[:, 1]
                root[idx[dp < dth]] = 0

            name = f"binary_plus_depth_{kind}"
            roots[name] = root

            diagnostics.append({
                "outer_fy": int(fy),
                "kind": kind,
                "threshold": float(dth),
                "inner_recall": ddiag.get("recall"),
                "inner_precision": ddiag.get("precision"),
                "inner_fp": ddiag.get("fp"),
            })

        for variant in variants:
            final_root = roots[variant]

            pred_high_rows = test.loc[final_root == 1].copy()
            high_map = high22_predict(
                train_high, pred_high_rows,
                high_gate_features, high_lower_features,
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
                rows[variant].append({
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
    for variant in variants:
        p = pd.DataFrame(rows[variant])
        p.to_csv(OUT / f"{variant}_predictions.csv", index=False)

        y = p["root_actual_high"].to_numpy(int)
        yp = p["root_pred_high"].to_numpy(int)
        fn = p.loc[
            (p["root_actual_high"] == 1)
            & (p["root_pred_high"] == 0),
            ["disasterNumber", "state", "incidentType", "actual_band"],
        ].to_dict(orient="records")

        results[variant] = {
            "root_high_recall": float(recall_score(
                y, yp, pos_label=1, zero_division=0
            )),
            "root_high_precision": float(precision_score(
                y, yp, pos_label=1, zero_division=0
            )),
            "root_fp": int(((y == 0) & (yp == 1)).sum()),
            "end_to_end": six_band_metrics(p, "final_pred"),
            "high_false_negatives": fn,
        }

    base_fn = {
        x["disasterNumber"]
        for x in results["binary_only"]["high_false_negatives"]
    }
    for variant in variants[1:]:
        vfn = {
            x["disasterNumber"]
            for x in results[variant]["high_false_negatives"]
        }
        results[variant]["safe_same_high_fn_set"] = (vfn == base_fn)

    pd.DataFrame(diagnostics).to_csv(
        OUT / "fold_diagnostics.csv", index=False
    )
    ext_audit.to_csv(
        OUT / "external_match_audit.csv", index=False
    )

    summary = {
        "mission_nonfinancial_sha256": ma_hash,
        "depth_features": depth_features,
        "results": results,
        "development_note": (
            "The compact depth feature family was motivated by a prior exploratory "
            "mechanism audit. This remains development validation, not external validation."
        ),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    md = [
        "# Hurricane operational-depth verifier audit",
        "",
        "- Baseline is the confirmed binary Hurricane semext veto.",
        "- Second veto uses only six interpretable response-depth features.",
        "- Thresholds are selected by inner LFYO to preserve all high Hurricane training cases.",
        "",
        "| Variant | Safe FN set | High recall | Root FP | Overall | Macro | 1M-50M | 50-200M | 200-500M | 500M+ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for variant, r in results.items():
        e = r["end_to_end"]
        pb = e["per_band"]
        safe = (
            True if variant == "binary_only"
            else r.get("safe_same_high_fn_set", False)
        )
        md.append(
            f"| {variant} | {safe} | "
            f"{r['root_high_recall']:.1%} | {r['root_fp']} | "
            f"{e['overall_correct']}/{e['overall_total']} ({e['overall_accuracy']:.1%}) | "
            f"{e['macro_recall']:.1%} | "
            f"{pb['1M-50M']['correct']}/{pb['1M-50M']['total']} ({pb['1M-50M']['recall']:.1%}) | "
            f"{pb['50-200M']['correct']}/{pb['50-200M']['total']} ({pb['50-200M']['recall']:.1%}) | "
            f"{pb['200-500M']['correct']}/{pb['200-500M']['total']} ({pb['200-500M']['recall']:.1%}) | "
            f"{pb['500M+']['correct']}/{pb['500M+']['total']} ({pb['500M+']['recall']:.1%}) |"
        )

    (OUT / "summary.md").write_text(
        "\n".join(md), encoding="utf-8"
    )
    print("\n".join(md))


if __name__ == "__main__":
    main()
