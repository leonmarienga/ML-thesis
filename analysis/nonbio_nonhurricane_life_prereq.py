#!/usr/bin/env python3
"""
Strict nested-LFYO pooled non-Hurricane response-intensity verifier.

Baseline:
- accepted full non-Biological router including Hurricane two-key veto
- current safe best: 802/912 on the current deterministic snapshot
- frozen high-value hierarchy

Candidate second boundary verifier applies ONLY to non-Hurricane rows already
selected as >=$50M after the accepted router.

Interpretable response-intensity features:
- sem_priority_life_sustaining_count
- sem_priority_life_sustaining_share
- sem_topic_logistics_commodities_share
- sem_priority_high_count
- sem_topic_debris_count
- sem_topic_transportation_count
- sem_long_30d_count
- missionAssignmentCount

Models:
- logistic
- random forest

Each threshold is selected only from inner-LFYO predictions on outer-training
non-Hurricane rows >=$1M and is calibrated to preserve 100% of true >=$50M
non-Hurricane training cases.

Any variant that changes the accepted baseline high-value false-negative set is
unsafe and must not be adopted.

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
from nonbio_hurricane_depth_verifier import (
    DEPTH_FEATURES, fit_depth_binary, depth_oof, safe_keep_threshold,
)
from nonbio_hurricane_two_key_veto import (
    DFA_FEATURES, fit_dfa_binary, dfa_oof,
)

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "nonbio_nonhurricane_life_prereq"
OUT.mkdir(parents=True, exist_ok=True)

RESPONSE_FEATURES = [
    "sem_priority_life_sustaining_count",
    "sem_priority_life_sustaining_share",
    "sem_topic_logistics_commodities_share",
    "sem_priority_high_count",
    "sem_topic_debris_count",
    "sem_topic_transportation_count",
    "sem_long_30d_count",
    "missionAssignmentCount",
]


def fit_response_binary(train, features, kind, seed):
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


def response_oof(outer_train, features, kind, seedbase):
    d = outer_train[
        (outer_train["incidentType"] != "Hurricane")
        & (outer_train["incidentType"] != "Biological")
        & (outer_train["target_clean"] >= 1_000_000)
    ].copy()
    rows = []
    for fy in sorted(d["fyDeclared"].astype(int).unique()):
        tr = d[d["fyDeclared"].astype(int) != fy].copy()
        te = d[d["fyDeclared"].astype(int) == fy].copy()
        ytr = (tr["target_clean"] >= 50_000_000).astype(int)
        if te.empty or ytr.nunique() < 2:
            continue
        model = fit_response_binary(tr, features, kind, seedbase + fy)
        pp = model.predict_proba(
            normalize_model_frame(te[features])
        )[:, 1]
        for (_, r), p in zip(te.iterrows(), pp):
            rows.append({
                "disasterNumber": int(r["disasterNumber"]),
                "fy": int(fy),
                "actual_high": int(r["target_clean"] >= 50_000_000),
                "prob_high": float(p),
            })
    return pd.DataFrame(rows)


def safe_keep_threshold(oof):
    if oof.empty or int(oof["actual_high"].sum()) == 0:
        return 0.0, {"recall": None, "precision": None, "fp": None}
    th = float(
        oof.loc[oof["actual_high"] == 1, "prob_high"].min()
    )
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


def monotonic_prereq_threshold(outer_train, feature):
    d = outer_train[
        (outer_train["incidentType"] != "Hurricane")
        & (outer_train["incidentType"] != "Biological")
        & (outer_train["target_clean"] >= 1_000_000)
    ].copy()

    inner_rows = []
    for fy in sorted(d["fyDeclared"].astype(int).unique()):
        tr = d[d["fyDeclared"].astype(int) != fy].copy()
        te = d[d["fyDeclared"].astype(int) == fy].copy()
        high_vals = pd.to_numeric(
            tr.loc[tr["target_clean"] >= 50_000_000, feature],
            errors="coerce",
        ).dropna()
        if te.empty or high_vals.empty:
            continue
        th = float(high_vals.min())
        vals = pd.to_numeric(te[feature], errors="coerce")
        veto = vals.notna() & (vals < th)
        for (_, r), v in zip(te.iterrows(), veto):
            inner_rows.append({
                "actual_high": int(r["target_clean"] >= 50_000_000),
                "veto": bool(v),
            })

    inner = pd.DataFrame(inner_rows)
    if inner.empty:
        return float("-inf"), {
            "enabled": False,
            "inner_high_recall": None,
            "inner_veto_count": 0,
        }

    high = inner[inner["actual_high"] == 1]
    inner_high_recall = (
        float((~high["veto"]).mean())
        if not high.empty else None
    )
    enabled = (
        inner_high_recall is not None
        and inner_high_recall >= 1.0 - 1e-12
    )

    final_high_vals = pd.to_numeric(
        outer_train.loc[
            (outer_train["incidentType"] != "Hurricane")
            & (outer_train["incidentType"] != "Biological")
            & (outer_train["target_clean"] >= 50_000_000),
            feature,
        ],
        errors="coerce",
    ).dropna()

    if not enabled or final_high_vals.empty:
        return float("-inf"), {
            "enabled": False,
            "inner_high_recall": inner_high_recall,
            "inner_veto_count": int(inner["veto"].sum()),
        }

    return float(final_high_vals.min()), {
        "enabled": True,
        "inner_high_recall": inner_high_recall,
        "inner_veto_count": int(inner["veto"].sum()),
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

    missing = [c for c in RESPONSE_FEATURES if c not in nonbio.columns]
    if missing:
        raise RuntimeError(f"Missing response features: {missing}")
    response_features = list(RESPONSE_FEATURES)

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

    variants = ["accepted_807", "plus_life_count", "plus_life_share", "plus_life_both"]
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
        boof = hurricane_oof(train, semext)
        bth, _ = safe_threshold(boof)
        htrain = train[
            (train["incidentType"] == "Hurricane")
            & (train["target_clean"] >= 1_000_000)
        ].copy()
        hy = (htrain["target_clean"] >= 50_000_000).astype(int)
        bmodel = (
            fit_hurricane_binary(htrain, semext, 420000 + fy)
            if len(htrain) >= 6 and hy.nunique() >= 2 else None
        )

        hurricane_root = global_root.copy()
        hidx = np.flatnonzero(
            (hurricane_root == 1)
            & (test["incidentType"].to_numpy() == "Hurricane")
        )
        if len(hidx) and bmodel is not None:
            bp = bmodel.predict_proba(
                normalize_model_frame(test.iloc[hidx][semext])
            )[:, 1]
            hurricane_root[hidx[bp < bth]] = 0

        # Accepted Hurricane two-key depth RF + DFA RF veto.
        doof = depth_oof(train, DEPTH_FEATURES, "rf", 510000)
        dth, _ = safe_keep_threshold(doof)
        dmodel = fit_depth_binary(
            htrain, DEPTH_FEATURES, "rf", 530000 + fy
        )

        dfaoo = dfa_oof(train, DFA_FEATURES, "rf", 610000)
        dfath, _ = safe_threshold(dfaoo)
        dfamodel = fit_dfa_binary(
            htrain, DFA_FEATURES, "rf", 630000 + fy
        )

        accepted_root = hurricane_root.copy()
        idx = np.flatnonzero(
            (accepted_root == 1)
            & (test["incidentType"].to_numpy() == "Hurricane")
        )
        if len(idx):
            dp = dmodel.predict_proba(
                normalize_model_frame(test.iloc[idx][DEPTH_FEATURES])
            )[:, 1]
            dfap = dfamodel.predict_proba(
                normalize_model_frame(test.iloc[idx][DFA_FEATURES])
            )[:, 1]
            veto = (dp < dth) & (dfap < dfath)
            accepted_root[idx[veto]] = 0

        low_predict = build_low_predictor(train, semfeat, fy)

        roots = {}

        # Pooled non-Hurricane response verifiers.
        nhtrain = train[
            (train["incidentType"] != "Hurricane")
            & (train["incidentType"] != "Biological")
            & (train["target_clean"] >= 1_000_000)
        ].copy()

        for kind in ["log", "rf"]:
            oo = response_oof(
                train, response_features, kind,
                700000 if kind == "log" else 710000,
            )
            th, dg = safe_keep_threshold(oo)
            yy = (nhtrain["target_clean"] >= 50_000_000).astype(int)
            model = (
                fit_response_binary(
                    nhtrain, response_features, kind,
                    720000 + fy if kind == "log" else 730000 + fy,
                )
                if len(nhtrain) >= 8 and yy.nunique() >= 2
                else None
            )

            root = accepted_root.copy()
            nidx = np.flatnonzero(
                (root == 1)
                & (test["incidentType"].to_numpy() != "Hurricane")
            )
            if len(nidx) and model is not None:
                pp = model.predict_proba(
                    normalize_model_frame(test.iloc[nidx][response_features])
                )[:, 1]
                root[nidx[pp < th]] = 0

            roots[f"plus_response_{kind}"] = root
            diagnostics.append({
                "outer_fy": int(fy),
                "kind": kind,
                "threshold": float(th),
                "inner_recall": dg.get("recall"),
                "inner_precision": dg.get("precision"),
                "inner_fp": dg.get("fp"),
            })


        # Accepted pooled RF verifier is the 807/912 baseline.
        accepted_807 = roots["plus_response_rf"].copy()
        roots["accepted_807"] = accepted_807.copy()

        count_feature = "sem_priority_life_sustaining_count"
        share_feature = "sem_priority_life_sustaining_share"

        count_th, count_diag = monotonic_prereq_threshold(
            train, count_feature
        )
        share_th, share_diag = monotonic_prereq_threshold(
            train, share_feature
        )

        nidx = np.flatnonzero(
            (accepted_807 == 1)
            & (test["incidentType"].to_numpy() != "Hurricane")
        )

        count_veto = np.zeros(len(test), dtype=bool)
        share_veto = np.zeros(len(test), dtype=bool)

        if len(nidx):
            count_vals = pd.to_numeric(
                test.iloc[nidx][count_feature],
                errors="coerce",
            ).to_numpy(float)
            share_vals = pd.to_numeric(
                test.iloc[nidx][share_feature],
                errors="coerce",
            ).to_numpy(float)

            count_veto[nidx] = (
                np.isfinite(count_vals)
                & (count_vals < count_th)
            )
            share_veto[nidx] = (
                np.isfinite(share_vals)
                & (share_vals < share_th)
            )

        roots["plus_life_count"] = accepted_807.copy()
        roots["plus_life_count"][count_veto] = 0

        roots["plus_life_share"] = accepted_807.copy()
        roots["plus_life_share"][share_veto] = 0

        roots["plus_life_both"] = accepted_807.copy()
        roots["plus_life_both"][count_veto & share_veto] = 0

        diagnostics.append({
            "outer_fy": int(fy),
            "kind": "life_prereq",
            "count_threshold": (
                None if not np.isfinite(count_th) else float(count_th)
            ),
            "count_enabled": count_diag.get("enabled"),
            "count_inner_high_recall": count_diag.get("inner_high_recall"),
            "share_threshold": (
                None if not np.isfinite(share_th) else float(share_th)
            ),
            "share_enabled": share_diag.get("enabled"),
            "share_inner_high_recall": share_diag.get("inner_high_recall"),
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
        for x in results["accepted_802"]["high_false_negatives"]
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
        "response_features": response_features,\n        "prerequisite_features": [\n            "sem_priority_life_sustaining_count",\n            "sem_priority_life_sustaining_share",\n        ],
        "results": results,
        "development_note": (
            "The life-sustaining prerequisite was motivated by a prior exploratory "
            "exploratory mechanism audit. This remains development validation."
        ),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    md = [
        "# Non-Hurricane life-sustaining prerequisite audit",
        "",
        "- Baseline: accepted 807/912 router including pooled non-Hurricane response RF.",
        "- Candidate prerequisite applies only after the accepted pooled RF verifier.",
        "- Monotonic thresholds are learned from outer-training high cases and enabled only when inner LFYO preserves all high cases.",
        "",
        "| Variant | Safe FN set | High recall | Root FP | Overall | Macro | 1M-50M | 50-200M | 200-500M | 500M+ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for variant, r in results.items():
        e = r["end_to_end"]
        pb = e["per_band"]
        safe = (
            True if variant == "accepted_807"
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
