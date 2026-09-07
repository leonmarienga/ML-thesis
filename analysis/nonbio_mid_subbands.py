#!/usr/bin/env python3
"""
Strict-LFYO internal sub-band audit for the broad $1M-$50M regime.

Hypothesis
----------
The final thesis band $1M-$50M is internally heterogeneous and is being
squeezed from both sides:
- lower edge confused with $100K-$1M
- upper edge confused with >=$50M

We keep the FINAL reported six bands unchanged. Internal sub-bands are used
only as specialist routing states and are collapsed back to $1M-$50M.

Fixed internal schemes:
A) 1-3M / 3-10M / 10-50M
B) 1-5M / 5-15M / 15-50M
C) 1-5M / 5-10M / 10-25M / 25-50M

For each scheme:
1. Low stage 1 remains semantics RF: <100K vs >=100K, with inner-LFYO
   macro-F1 threshold.
2. Low stage 2 becomes a multiclass current19 RF:
   100K-1M + the internal $1M-$50M sub-bands.
   Internal sub-band predictions collapse to final $1M-$50M.
3. High-entry sentinel becomes a semext logistic multiclass sentinel:
   the same internal low sub-bands + 50-200M / 200-500M / 500M+.
   Its low probability is the SUM of all internal <50M sub-band
   probabilities. The veto threshold is set above the maximum summed-low
   probability observed among true >=50M inner-LFYO training cases.

Upstream >=50M root is frozen.
High hierarchy is frozen.
Biological remains excluded/frozen.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

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
from nonbio_all_ranges import BANDS, funding_band, high22_predict, six_band_metrics, valid_cols
from nonbio_root_recall import fit_log, proba, inner_oof_scores, recall_first_threshold
from nonbio_root_cascade import verifier_oof, threshold_for_recall, fit_verifier
from nonbio_low_thresholds import inner_binary_oof, choose_threshold
from nonbio_fourband_sentinel import (
    sentinel_oof, perfect_high_veto_threshold, fit_multi, low_probability,
    build_low_predictor,
)

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "nonbio_mid_subbands"
OUT.mkdir(parents=True, exist_ok=True)

LOW_BANDS = ["0-100K", "100K-1M", "1M-50M"]
SCHEMES = {
    "split_3_10": [1_000_000, 3_000_000, 10_000_000, 50_000_000],
    "split_5_15": [1_000_000, 5_000_000, 15_000_000, 50_000_000],
    "split_5_10_25": [1_000_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000],
}


def internal_mid_label(v: float, edges: List[float]) -> str:
    v = float(v)
    for i in range(len(edges) - 1):
        if edges[i] <= v < edges[i + 1]:
            return f"MID_{i}_{int(edges[i])}_{int(edges[i+1])}"
    raise ValueError(f"value outside internal mid band: {v}")


def train_internal_label(v: float, edges: List[float]) -> str:
    v = float(v)
    if 100_000 <= v < 1_000_000:
        return "LOW_100K_1M"
    if 1_000_000 <= v < 50_000_000:
        return internal_mid_label(v, edges)
    if 50_000_000 <= v < 200_000_000:
        return "HIGH_50_200"
    if 200_000_000 <= v < 500_000_000:
        return "HIGH_200_500"
    if v >= 500_000_000:
        return "HIGH_500P"
    return "LOW_0_100K"


def fit_rf(train: pd.DataFrame, features: List[str], y: pd.Series, seed: int):
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


def fit_log_multi(train: pd.DataFrame, features: List[str], y: pd.Series):
    X = normalize_model_frame(train[features])
    model = LogisticRegression(
        max_iter=5000, class_weight="balanced", C=0.5
    )
    p = prep_pipeline(X, model)
    p.fit(X, y)
    return p


def positive_proba(model, df, features):
    return model.predict_proba(normalize_model_frame(df[features]))[:, 1]


def summed_mid_probability(model, df: pd.DataFrame, features: List[str]) -> np.ndarray:
    if df.empty:
        return np.array([], dtype=float)
    probs = model.predict_proba(normalize_model_frame(df[features]))
    classes = [str(c) for c in model.classes_]
    idx = [i for i, c in enumerate(classes) if c.startswith("MID_")]
    if not idx:
        return np.zeros(len(df), dtype=float)
    return probs[:, idx].sum(axis=1)


def sub_sentinel_oof(
    outer_train: pd.DataFrame,
    features: List[str],
    edges: List[float],
) -> pd.DataFrame:
    d = outer_train[outer_train["target_clean"] >= 1_000_000].copy()
    rows = []
    for fy in sorted(d["fyDeclared"].astype(int).unique()):
        tr = d[d["fyDeclared"].astype(int) != fy].copy()
        te = d[d["fyDeclared"].astype(int) == fy].copy()
        if te.empty:
            continue
        ytr = tr["target_clean"].map(lambda v: train_internal_label(v, edges))
        if ytr.nunique() < 2:
            continue
        m = fit_log_multi(tr, features, ytr)
        pp = summed_mid_probability(m, te, features)
        for (_, r), p in zip(te.iterrows(), pp):
            rows.append({
                "disasterNumber": int(r["disasterNumber"]),
                "fy": int(fy),
                "actual_high": int(r["target_clean"] >= 50_000_000),
                "p_mid_sum": float(p),
            })
    return pd.DataFrame(rows)


def safe_veto_threshold(oof: pd.DataFrame) -> Tuple[float, dict]:
    high = oof[oof["actual_high"] == 1]
    if high.empty:
        return 1.0, {"inner_high_recall": None, "inner_mid_reject": None}
    th = min(1.0, float(high["p_mid_sum"].max()) + 1e-12)
    keep = oof["p_mid_sum"] < th
    high_recall = float(keep[oof["actual_high"] == 1].mean())
    lows = oof["actual_high"] == 0
    low_reject = float((~keep[lows]).mean()) if lows.any() else None
    return th, {
        "inner_high_recall": high_recall,
        "inner_mid_reject": low_reject,
        "inner_high_max_p_mid": float(high["p_mid_sum"].max()),
    }


def build_split_low_predictor(
    train: pd.DataFrame,
    semfeat: List[str],
    current: List[str],
    outer_fy: int,
    edges: List[float],
):
    # Stage 1 stays frozen: <100K vs >=100K.
    s1_oof = inner_binary_oof(
        train, semfeat, stage=1, kind="rf", seedbase=110000
    )
    s1_th, _ = choose_threshold(s1_oof, "macro_f1")

    low = train[train["target_clean"] < 50_000_000].copy()
    y1 = (low["target_clean"] >= 100_000).astype(int)
    s1 = fit_rf(low, semfeat, y1, 130000 + int(outer_fy))

    # Stage 2 is now multiclass: 100K-1M + internal mid sub-bands.
    upper = low[low["target_clean"] >= 100_000].copy()
    y2 = upper["target_clean"].map(lambda v: train_internal_label(v, edges))
    s2 = fit_rf(upper, current, y2, 150000 + int(outer_fy))

    def predict(dd: pd.DataFrame) -> np.ndarray:
        if dd.empty:
            return np.array([], dtype=object)
        p1 = positive_proba(s1, dd, semfeat)
        out = np.full(len(dd), "0-100K", dtype=object)
        ii = np.flatnonzero(p1 >= s1_th)
        if len(ii):
            raw = s2.predict(
                normalize_model_frame(dd.iloc[ii][current])
            ).astype(str)
            collapsed = np.where(
                np.char.startswith(raw.astype(str), "MID_"),
                "1M-50M",
                "100K-1M",
            )
            out[ii] = collapsed
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

    variants = ["current_best"] + list(SCHEMES)
    rows = {v: [] for v in variants}
    diag = []

    # Report internal counts before any model fitting.
    count_rows = []
    mids = nonbio[
        (nonbio["target_clean"] >= 1_000_000)
        & (nonbio["target_clean"] < 50_000_000)
    ].copy()
    for name, edges in SCHEMES.items():
        labs = mids["target_clean"].map(lambda v: internal_mid_label(v, edges))
        vc = labs.value_counts()
        for lab, n in vc.items():
            count_rows.append({"scheme": name, "internal_band": lab, "n": int(n)})
    pd.DataFrame(count_rows).to_csv(OUT / "internal_band_counts.csv", index=False)

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

        # Current best frozen low predictor and four-band semext sentinel.
        base_low_predict = build_low_predictor(train, semfeat, current, outer_fy)
        base_soof = sentinel_oof(train, semext, "log", 200000)
        base_sth, _ = perfect_high_veto_threshold(base_soof)
        base_strain = train[train["target_clean"] >= 1_000_000].copy()
        base_smodel = fit_multi(
            base_strain, semext, "log", 220000 + int(outer_fy)
        )

        for variant in variants:
            final_root = upstream.copy()

            if variant == "current_best":
                low_predict = base_low_predict
                si = np.flatnonzero(upstream == 1)
                if len(si):
                    pp = low_probability(base_smodel, test.iloc[si], semext)
                    final_root[si[pp >= base_sth]] = 0
                diag.append({
                    "outer_fy": int(outer_fy),
                    "variant": variant,
                    "sentinel_threshold": float(base_sth),
                    "inner_high_recall": 1.0,
                })
            else:
                edges = SCHEMES[variant]
                low_predict = build_split_low_predictor(
                    train, semfeat, current, outer_fy, edges
                )
                soof = sub_sentinel_oof(train, semext, edges)
                sth, sdiag = safe_veto_threshold(soof)
                strain = train[train["target_clean"] >= 1_000_000].copy()
                sy = strain["target_clean"].map(lambda v: train_internal_label(v, edges))
                smodel = fit_log_multi(strain, semext, sy)
                si = np.flatnonzero(upstream == 1)
                if len(si):
                    pp = summed_mid_probability(smodel, test.iloc[si], semext)
                    final_root[si[pp >= sth]] = 0
                diag.append({
                    "outer_fy": int(outer_fy),
                    "variant": variant,
                    "sentinel_threshold": float(sth),
                    "inner_high_recall": sdiag.get("inner_high_recall"),
                    "inner_mid_reject": sdiag.get("inner_mid_reject"),
                })

            pred_high_rows = test.loc[final_root == 1].copy()
            high_map = high22_predict(
                train_high, pred_high_rows, high_gate_features, high_lower_features
            ) if not pred_high_rows.empty else {}

            pred_low_rows = test.loc[final_root == 0].copy()
            lp = low_predict(pred_low_rows)
            low_map = {
                int(dn): str(pp)
                for dn, pp in zip(pred_low_rows["disasterNumber"], lp)
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
                    "target_clean": float(r["target_clean"]),
                    "actual_band": r["actual_band"],
                    "root_actual_high": int(r["target_clean"] >= 50_000_000),
                    "upstream_root_high": int(upstream[j]),
                    "root_pred_high": rp,
                    "final_pred": final,
                })

    results = {}
    for variant in variants:
        p = pd.DataFrame(rows[variant])
        p.to_csv(OUT / f"{variant}_predictions.csv", index=False)
        y = p["root_actual_high"].to_numpy(int)
        yp = p["root_pred_high"].to_numpy(int)

        mid = p[p["actual_band"] == "1M-50M"].copy()
        results[variant] = {
            "root_high_recall": float(recall_score(y, yp, pos_label=1, zero_division=0)),
            "root_high_precision": float(precision_score(y, yp, pos_label=1, zero_division=0)),
            "root_fp": int(((y == 0) & (yp == 1)).sum()),
            "end_to_end": six_band_metrics(p, "final_pred"),
            "mid_interference": {
                "correct": int((mid["final_pred"] == "1M-50M").sum()),
                "down_to_100K_1M": int((mid["final_pred"] == "100K-1M").sum()),
                "up_to_high": int(mid["final_pred"].isin(["50-200M", "200-500M", "500M+"]).sum()),
                "other": int((~mid["final_pred"].isin(["100K-1M", "1M-50M", "50-200M", "200-500M", "500M+"])).sum()),
            },
            "remaining_high_false_negatives": p.loc[
                (p["root_actual_high"] == 1) & (p["root_pred_high"] == 0),
                ["disasterNumber", "state", "incidentType", "actual_band"],
            ].to_dict(orient="records"),
        }

    pd.DataFrame(diag).to_csv(OUT / "fold_diagnostics.csv", index=False)
    ext_audit.to_csv(OUT / "external_match_audit.csv", index=False)

    summary = {
        "scope": "912 non-Biological declarations; Biological excluded",
        "final_reported_bands": BANDS,
        "internal_schemes": SCHEMES,
        "note": "Internal sub-bands are specialist states only and collapse back to final 1M-50M.",
        "results": results,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Internal $1M-$50M sub-band interference audit",
        "",
        "- Final reported six bands are unchanged.",
        "- Internal sub-bands are used only to reduce neighboring-band interference.",
        "",
        "## Internal counts",
    ]
    counts = pd.DataFrame(count_rows)
    for name in SCHEMES:
        md.append(f"### {name}")
        for _, rr in counts[counts["scheme"] == name].sort_values("internal_band").iterrows():
            md.append(f"- {rr['internal_band']}: **{int(rr['n'])}**")

    md += [
        "",
        "## Strict LFYO end-to-end results",
        "",
        "| Variant | Overall | Macro recall | 100K-1M | 1M-50M | mid down | mid up | 50-200M | 200-500M | 500M+ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in results.items():
        e = r["end_to_end"]
        pb = e["per_band"]
        mi = r["mid_interference"]
        md.append(
            f"| {name} | {e['overall_correct']}/{e['overall_total']} ({e['overall_accuracy']:.1%}) | "
            f"{e['macro_recall']:.1%} | "
            f"{pb['100K-1M']['correct']}/{pb['100K-1M']['total']} ({pb['100K-1M']['recall']:.1%}) | "
            f"{pb['1M-50M']['correct']}/{pb['1M-50M']['total']} ({pb['1M-50M']['recall']:.1%}) | "
            f"{mi['down_to_100K_1M']} | {mi['up_to_high']} | "
            f"{pb['50-200M']['correct']}/{pb['50-200M']['total']} ({pb['50-200M']['recall']:.1%}) | "
            f"{pb['200-500M']['correct']}/{pb['200-500M']['total']} ({pb['200-500M']['recall']:.1%}) | "
            f"{pb['500M+']['correct']}/{pb['500M+']['total']} ({pb['500M+']['recall']:.1%}) |"
        )

    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
