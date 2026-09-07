#!/usr/bin/env python3
"""
Strict nested-LFYO consensus audit for the four-band sentinel.

Upstream root is frozen:
  nested recall-first OR candidate generator
  + logistic candidate verifier calibrated to 95% inner-LFYO high recall.

Three logistic four-band sentinels:
  current19 / semantics / semantics+external.
Each threshold is learned only from inner-LFYO outer-training predictions and
is set above the maximum P(1M-50M) observed for true >=50M training cases.

Rules:
  no_sentinel
  semext_only
  consensus_2of3
  unanimous_3of3
  any_1of3

Sentinel is one-way veto only. Biological remains excluded/frozen.
"""

from __future__ import annotations
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score

from mission_semantic_audit import (
    CURRENT_19, build_semantic_rollup, fetch_all_mission_assignments,
    normalize_master, normalize_model_frame,
)
from external_severity_ablation import build_external
from nonbio_hazard_hierarchy import initial_mechanism_counts
from nonbio_outage_rescue import build_eaglei_all
from nonbio_all_ranges import BANDS, funding_band, high22_predict, six_band_metrics, valid_cols
from nonbio_root_recall import fit_log, proba, inner_oof_scores, recall_first_threshold
from nonbio_root_cascade import verifier_oof, threshold_for_recall, fit_verifier
from nonbio_fourband_sentinel import (
    sentinel_oof, perfect_high_veto_threshold, fit_multi, low_probability,
    build_low_predictor,
)

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "nonbio_sentinel_consensus"
OUT.mkdir(parents=True, exist_ok=True)
LOW_BANDS = ["0-100K", "100K-1M", "1M-50M"]


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
    sentinel_sets = {"current19": current, "semantics": semfeat, "semext": semext}

    high_all = nonbio[nonbio["target_clean"] >= 50_000_000].copy()
    hc = valid_cols(high_all, CURRENT_19)
    hs = valid_cols(high_all, [c for c in high_all.columns if c.startswith("sem_") or c.startswith("ma_")])
    he = valid_cols(high_all, [c for c in high_all.columns
                               if c.startswith("nhc_") or c.startswith("noaa_") or c.startswith("calfire_")])
    high_gate_features = hc + hs + he
    high_lower_features = hc + hs

    rules = ["no_sentinel", "semext_only", "consensus_2of3", "unanimous_3of3", "any_1of3"]
    rows = {r: [] for r in rules}
    diag = []

    for outer_fy in sorted(nonbio["fyDeclared"].astype(int).unique()):
        train = nonbio[nonbio["fyDeclared"].astype(int) != outer_fy].copy()
        test = nonbio[nonbio["fyDeclared"].astype(int) == outer_fy].copy()
        train_high = train[train["target_clean"] >= 50_000_000].copy()

        # Frozen accepted upstream root.
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

        upstream = np.zeros(len(test), dtype=int)
        ci = np.flatnonzero(candidate)
        if len(ci):
            vp = verifier.predict_proba(
                normalize_model_frame(test.iloc[ci][semext])
            )[:, 1]
            upstream[ci] = (vp >= vth).astype(int)

        low_predict = build_low_predictor(train, semfeat, current, outer_fy)

        selected = np.flatnonzero(upstream == 1)
        veto_matrix = np.zeros((len(test), 3), dtype=bool)
        set_names = ["current19", "semantics", "semext"]

        for k, sname in enumerate(set_names):
            feats = sentinel_sets[sname]
            soof = sentinel_oof(train, feats, "log", 200000 + 1000 * k)
            sth, sdiag = perfect_high_veto_threshold(soof)
            strain = train[train["target_clean"] >= 1_000_000].copy()
            smodel = fit_multi(
                strain, feats, "log", 220000 + 1000 * k + int(outer_fy)
            )
            if len(selected):
                pp = low_probability(smodel, test.iloc[selected], feats)
                veto_matrix[selected, k] = pp >= sth
            diag.append({
                "outer_fy": int(outer_fy),
                "sentinel": sname,
                "threshold": float(sth),
                "inner_high_recall": sdiag.get("inner_high_recall"),
                "inner_low_reject": sdiag.get("inner_low_reject"),
            })

        vote_count = veto_matrix.sum(axis=1)
        veto_rules = {
            "no_sentinel": np.zeros(len(test), dtype=bool),
            "semext_only": veto_matrix[:, 2],
            "consensus_2of3": vote_count >= 2,
            "unanimous_3of3": vote_count == 3,
            "any_1of3": vote_count >= 1,
        }

        for rule in rules:
            final_root = upstream.copy()
            final_root[veto_rules[rule]] = 0

            pred_high_rows = test.loc[final_root == 1].copy()
            high_map = high22_predict(
                train_high, pred_high_rows, high_gate_features, high_lower_features
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
                rows[rule].append({
                    "disasterNumber": dn,
                    "state": r["state"],
                    "incidentType": r["incidentType"],
                    "fyDeclared": int(r["fyDeclared"]),
                    "actual_band": r["actual_band"],
                    "root_actual_high": int(r["target_clean"] >= 50_000_000),
                    "upstream_root_high": int(upstream[j]),
                    "root_pred_high": rp,
                    "veto_current19": bool(veto_matrix[j, 0]),
                    "veto_semantics": bool(veto_matrix[j, 1]),
                    "veto_semext": bool(veto_matrix[j, 2]),
                    "veto_votes": int(vote_count[j]),
                    "final_pred": final,
                })

    results = {}
    for rule in rules:
        p = pd.DataFrame(rows[rule])
        p.to_csv(OUT / f"{rule}_predictions.csv", index=False)
        y = p["root_actual_high"].to_numpy(int)
        yp = p["root_pred_high"].to_numpy(int)
        results[rule] = {
            "root_high_recall": float(recall_score(y, yp, pos_label=1, zero_division=0)),
            "root_high_precision": float(precision_score(y, yp, pos_label=1, zero_division=0)),
            "root_fp": int(((y == 0) & (yp == 1)).sum()),
            "root_fn": int(((y == 1) & (yp == 0)).sum()),
            "root_fp_by_band": {
                b: int(((p["actual_band"] == b) & (p["root_actual_high"] == 0) & (p["root_pred_high"] == 1)).sum())
                for b in LOW_BANDS
            },
            "end_to_end": six_band_metrics(p, "final_pred"),
            "remaining_high_false_negatives": p.loc[
                (p["root_actual_high"] == 1) & (p["root_pred_high"] == 0),
                ["disasterNumber", "state", "incidentType", "actual_band"],
            ].to_dict(orient="records"),
        }

    pd.DataFrame(diag).to_csv(OUT / "sentinel_fold_diagnostics.csv", index=False)
    ext_audit.to_csv(OUT / "external_match_audit.csv", index=False)

    summary = {
        "scope": "912 non-Biological declarations; Biological excluded",
        "upstream_root": "nested recall OR + logistic verifier @95% inner recall",
        "sentinel": "three target-blind four-band logistic sentinels; one-way veto",
        "high_branch": "frozen 22/23 conditional hierarchy",
        "results": results,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Four-band sentinel consensus audit",
        "",
        "- Three logistic sentinels: current19 / semantics / semext.",
        "- Every threshold preserves all inner-LFYO high training cases.",
        "- One-way veto only.",
        "",
        "| Rule | High recall | Precision | Root FP | FP 1M-50M | End-to-end | Macro recall | 0-100K | 100K-1M | 1M-50M | 50-200M | 200-500M | 500M+ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rule, r in results.items():
        e = r["end_to_end"]
        pb = e["per_band"]
        md.append(
            f"| {rule} | {r['root_high_recall']:.1%} | {r['root_high_precision']:.1%} | "
            f"{r['root_fp']} | {r['root_fp_by_band']['1M-50M']} | "
            f"{e['overall_correct']}/{e['overall_total']} ({e['overall_accuracy']:.1%}) | "
            f"{e['macro_recall']:.1%} | "
            + " | ".join(
                f"{pb[b]['correct']}/{pb[b]['total']} ({pb[b]['recall']:.1%})"
                for b in BANDS
            ) + " |"
        )
    md += ["", "## Remaining high-value root false negatives"]
    for rule, r in results.items():
        md.append(f"### {rule}")
        if not r["remaining_high_false_negatives"]:
            md.append("- **None**")
        else:
            for x in r["remaining_high_false_negatives"]:
                md.append(
                    f"- FEMA {x['disasterNumber']} {x['state']} {x['incidentType']} {x['actual_band']}"
                )

    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
