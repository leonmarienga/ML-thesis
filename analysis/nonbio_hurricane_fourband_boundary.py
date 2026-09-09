#!/usr/bin/env python3
"""
Strict-LFYO Hurricane-specific four-band $50M boundary audit.

Same deterministic baseline architecture as nonbio_hurricane_boundary_confirm:
- recall-first upstream >=50M root
- global semext four-band sentinel
- semantics ExtraTrees continuous $1M boundary
- frozen high-value hierarchy

Additional Hurricane-only one-way veto variants:
1. confirmed binary semext logistic
2. Hurricane four-band semext logistic
3. Hurricane four-band semext + EAGLE-I logistic
4. union of binary + four-band semext vetoes
5. consensus of binary + four-band semext vetoes

Every Hurricane-specific threshold is selected only from inner-LFYO
outer-training predictions so that all true >=50M Hurricane training cases are
kept. Any outer variant that changes the baseline high-value false-negative set
is marked unsafe and must not be adopted.

Biological remains excluded/frozen.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
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
from nonbio_fourband_sentinel import (
    sentinel_oof, perfect_high_veto_threshold, fit_multi, low_probability,
)
from nonbio_hurricane_boundary_confirm import (
    data_hash, build_low_predictor,
    fit_hurricane_binary, hurricane_oof, safe_threshold,
)

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "nonbio_hurricane_fourband_boundary"
OUT.mkdir(parents=True, exist_ok=True)


def hurricane_label(v: float) -> str:
    v = float(v)
    if v < 50_000_000:
        return "1M-50M"
    if v < 200_000_000:
        return "50-200M"
    if v < 500_000_000:
        return "200-500M"
    return "500M+"


def fit_hurricane_fourband(train: pd.DataFrame, features, seed: int):
    t = train[
        (train["incidentType"] == "Hurricane")
        & (train["target_clean"] >= 1_000_000)
    ].copy()
    y = t["target_clean"].map(hurricane_label)
    X = normalize_model_frame(t[features])
    model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        C=0.5,
    )
    pipe = prep_pipeline(X, model)
    pipe.fit(X, y)
    return pipe


def hurricane_low_probability(model, df: pd.DataFrame, features) -> np.ndarray:
    if df.empty:
        return np.array([], dtype=float)
    probs = model.predict_proba(
        normalize_model_frame(df[features])
    )
    classes = [str(c) for c in model.classes_]
    if "1M-50M" not in classes:
        return np.zeros(len(df), dtype=float)
    idx = classes.index("1M-50M")
    return probs[:, idx]


def hurricane_fourband_oof(outer_train: pd.DataFrame, features, seedbase: int):
    d = outer_train[
        (outer_train["incidentType"] == "Hurricane")
        & (outer_train["target_clean"] >= 1_000_000)
    ].copy()
    rows = []

    for fy in sorted(d["fyDeclared"].astype(int).unique()):
        tr = d[d["fyDeclared"].astype(int) != fy].copy()
        te = d[d["fyDeclared"].astype(int) == fy].copy()
        if te.empty:
            continue

        ytr = tr["target_clean"].map(hurricane_label)
        if ytr.nunique() < 2:
            continue

        X = normalize_model_frame(tr[features])
        model = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            C=0.5,
        )
        pipe = prep_pipeline(X, model)
        pipe.fit(X, ytr)

        p_low = hurricane_low_probability(
            pipe, te, features
        )

        for (_, r), p in zip(te.iterrows(), p_low):
            rows.append({
                "disasterNumber": int(r["disasterNumber"]),
                "fy": int(fy),
                "actual_high": int(
                    r["target_clean"] >= 50_000_000
                ),
                "p_low": float(p),
            })

    return pd.DataFrame(rows)


def safe_low_veto_threshold(oof: pd.DataFrame):
    high = oof[oof["actual_high"] == 1]
    if high.empty:
        return 1.0, {
            "high_recall": None,
            "low_reject": None,
        }

    threshold = min(
        1.0,
        float(high["p_low"].max()) + 1e-12,
    )
    keep = oof["p_low"] < threshold

    high_recall = float(
        keep[oof["actual_high"] == 1].mean()
    )
    low_mask = oof["actual_high"] == 0
    low_reject = (
        float((~keep[low_mask]).mean())
        if low_mask.any()
        else None
    )

    return threshold, {
        "high_recall": high_recall,
        "low_reject": low_reject,
        "max_high_p_low": float(
            high["p_low"].max()
        ),
    }


def main():
    master = pd.read_excel(MASTER)
    from mission_semantic_audit import normalize_master
    master = normalize_master(master)
    master["target_clean"] = pd.to_numeric(
        master["totalObligatedFunding"],
        errors="coerce",
    ).fillna(0).clip(lower=0)
    master["actual_band"] = master[
        "target_clean"
    ].map(funding_band)

    ma = fetch_all_mission_assignments()
    ma_hash = data_hash(ma)
    sem, _ = build_semantic_rollup(master, ma)
    ext, ext_audit = build_external(master)
    mech = initial_mechanism_counts(master, ma)

    print(
        "Mission nonfinancial snapshot SHA256:",
        ma_hash,
        flush=True,
    )
    print(
        "Building EAGLE-I Hurricane features...",
        flush=True,
    )
    eag = build_eaglei_all(master)

    df = (
        master.merge(sem, on="disasterNumber", how="left")
        .merge(ext, on="disasterNumber", how="left")
        .merge(mech, on="disasterNumber", how="left")
        .merge(eag, on="disasterNumber", how="left")
    )
    df["initial_usace_esf3_dfa_count"] = (
        df["initial_usace_esf3_dfa_count"]
        .fillna(0)
        .astype(int)
    )
    nonbio = df[
        df["incidentType"] != "Biological"
    ].copy().reset_index(drop=True)

    current = valid_cols(nonbio, CURRENT_19)
    semcols = valid_cols(
        nonbio,
        [
            c for c in nonbio.columns
            if c.startswith("sem_")
            or c.startswith("ma_")
        ],
    )
    semfeat = list(
        dict.fromkeys(current + semcols)
    )
    extcols = valid_cols(
        nonbio,
        [
            c for c in nonbio.columns
            if c.startswith("nhc_")
            or c.startswith("noaa_")
            or c.startswith("calfire_")
        ],
    )
    semext = list(
        dict.fromkeys(semfeat + extcols)
    )
    eagcols = valid_cols(
        nonbio,
        [
            c for c in nonbio.columns
            if c.startswith("eaglei_")
        ],
    )
    hurr_eag = list(
        dict.fromkeys(semext + eagcols)
    )

    high_all = nonbio[
        nonbio["target_clean"] >= 50_000_000
    ].copy()
    hc = valid_cols(high_all, CURRENT_19)
    hs = valid_cols(
        high_all,
        [
            c for c in high_all.columns
            if c.startswith("sem_")
            or c.startswith("ma_")
        ],
    )
    he = valid_cols(
        high_all,
        [
            c for c in high_all.columns
            if c.startswith("nhc_")
            or c.startswith("noaa_")
            or c.startswith("calfire_")
        ],
    )
    high_gate_features = hc + hs + he
    high_lower_features = hc + hs

    variants = [
        "baseline",
        "binary_semext",
        "fourband_semext",
        "fourband_semext_eaglei",
        "binary_or_fourband",
        "binary_and_fourband",
    ]
    rows = {v: [] for v in variants}
    diagnostics = []

    for fy in sorted(
        nonbio["fyDeclared"].astype(int).unique()
    ):
        train = nonbio[
            nonbio["fyDeclared"].astype(int) != fy
        ].copy()
        test = nonbio[
            nonbio["fyDeclared"].astype(int) == fy
        ].copy()
        train_high = train[
            train["target_clean"] >= 50_000_000
        ].copy()

        # Shared upstream root.
        oof_cur = inner_oof_scores(
            train, current
        ).rename(columns={"prob": "pcur"})
        oof_sem = inner_oof_scores(
            train, semext
        ).rename(columns={"prob": "psem"})
        th_cur, _ = recall_first_threshold(
            oof_cur.rename(
                columns={"pcur": "prob"}
            )
        )
        th_sem, _ = recall_first_threshold(
            oof_sem.rename(
                columns={"psem": "prob"}
            )
        )

        oo = oof_cur[
            ["disasterNumber", "pcur"]
        ].merge(
            oof_sem[
                ["disasterNumber", "psem"]
            ],
            on="disasterNumber",
            how="inner",
        )
        tm = train.merge(
            oo,
            on="disasterNumber",
            how="left",
        )
        tm["candidate"] = (
            (tm["pcur"] >= th_cur)
            | (tm["psem"] >= th_sem)
        )
        cand_train = tm[
            tm["candidate"]
        ].copy()

        voof = verifier_oof(
            cand_train,
            semext,
            "log",
            60000,
        )
        vth, _ = threshold_for_recall(
            voof, 0.95
        )
        verifier = fit_verifier(
            cand_train,
            semext,
            "log",
            80000 + fy,
        )

        cur_root = fit_log(train, current)
        sem_root = fit_log(train, semext)
        pcur = proba(
            cur_root, test, current
        )
        psem = proba(
            sem_root, test, semext
        )
        candidate = (
            (pcur >= th_cur)
            | (psem >= th_sem)
        )

        upstream = np.zeros(
            len(test), dtype=int
        )
        ci = np.flatnonzero(candidate)
        if len(ci):
            vp = verifier.predict_proba(
                normalize_model_frame(
                    test.iloc[ci][semext]
                )
            )[:, 1]
            upstream[ci] = (
                vp >= vth
            ).astype(int)

        # Shared global semext sentinel.
        soof = sentinel_oof(
            train,
            semext,
            "log",
            200000,
        )
        sth, _ = (
            perfect_high_veto_threshold(
                soof
            )
        )
        strain = train[
            train["target_clean"]
            >= 1_000_000
        ].copy()
        smodel = fit_multi(
            strain,
            semext,
            "log",
            220000 + fy,
        )

        global_root = upstream.copy()
        si = np.flatnonzero(
            upstream == 1
        )
        if len(si):
            low_probs = low_probability(
                smodel,
                test.iloc[si],
                semext,
            )
            global_root[
                si[low_probs >= sth]
            ] = 0

        low_predict = build_low_predictor(
            train,
            semfeat,
            fy,
        )

        # Binary Hurricane verifier.
        bin_oof = hurricane_oof(
            train, semext
        )
        bin_th, bin_diag = safe_threshold(
            bin_oof
        )
        htrain = train[
            (train["incidentType"] == "Hurricane")
            & (
                train["target_clean"]
                >= 1_000_000
            )
        ].copy()
        hy = (
            htrain["target_clean"]
            >= 50_000_000
        ).astype(int)

        bin_model = None
        if (
            len(htrain) >= 6
            and hy.nunique() >= 2
        ):
            bin_model = (
                fit_hurricane_binary(
                    htrain,
                    semext,
                    420000 + fy,
                )
            )

        # Four-band semext.
        fb_oof = hurricane_fourband_oof(
            train,
            semext,
            500000,
        )
        fb_th, fb_diag = (
            safe_low_veto_threshold(
                fb_oof
            )
        )
        fb_model = fit_hurricane_fourband(
            train,
            semext,
            520000 + fy,
        )

        # Four-band semext + EAGLE-I.
        fbe_oof = hurricane_fourband_oof(
            train,
            hurr_eag,
            600000,
        )
        fbe_th, fbe_diag = (
            safe_low_veto_threshold(
                fbe_oof
            )
        )
        fbe_model = fit_hurricane_fourband(
            train,
            hurr_eag,
            620000 + fy,
        )

        diagnostics.append({
            "outer_fy": int(fy),
            "binary_threshold": float(
                bin_th
            ),
            "binary_inner_recall": (
                bin_diag.get("recall")
            ),
            "fourband_threshold": float(
                fb_th
            ),
            "fourband_inner_recall": (
                fb_diag.get(
                    "high_recall"
                )
            ),
            "fourband_inner_low_reject": (
                fb_diag.get(
                    "low_reject"
                )
            ),
            "fourband_eaglei_threshold": float(
                fbe_th
            ),
            "fourband_eaglei_inner_recall": (
                fbe_diag.get(
                    "high_recall"
                )
            ),
            "fourband_eaglei_inner_low_reject": (
                fbe_diag.get(
                    "low_reject"
                )
            ),
        })

        selected = np.flatnonzero(
            (global_root == 1)
            & (
                test[
                    "incidentType"
                ].to_numpy()
                == "Hurricane"
            )
        )

        binary_veto = np.zeros(
            len(test), dtype=bool
        )
        fb_veto = np.zeros(
            len(test), dtype=bool
        )
        fbe_veto = np.zeros(
            len(test), dtype=bool
        )

        if len(selected):
            if bin_model is not None:
                bp = (
                    bin_model.predict_proba(
                        normalize_model_frame(
                            test.iloc[
                                selected
                            ][semext]
                        )
                    )[:, 1]
                )
                binary_veto[
                    selected
                ] = bp < bin_th

            fp = hurricane_low_probability(
                fb_model,
                test.iloc[selected],
                semext,
            )
            fb_veto[
                selected
            ] = fp >= fb_th

            fep = hurricane_low_probability(
                fbe_model,
                test.iloc[selected],
                hurr_eag,
            )
            fbe_veto[
                selected
            ] = fep >= fbe_th

        roots = {
            "baseline": global_root.copy(),
            "binary_semext": global_root.copy(),
            "fourband_semext": global_root.copy(),
            "fourband_semext_eaglei": global_root.copy(),
            "binary_or_fourband": global_root.copy(),
            "binary_and_fourband": global_root.copy(),
        }

        roots["binary_semext"][
            binary_veto
        ] = 0
        roots["fourband_semext"][
            fb_veto
        ] = 0
        roots[
            "fourband_semext_eaglei"
        ][fbe_veto] = 0
        roots[
            "binary_or_fourband"
        ][binary_veto | fb_veto] = 0
        roots[
            "binary_and_fourband"
        ][binary_veto & fb_veto] = 0

        for variant in variants:
            final_root = roots[variant]

            pred_high_rows = test.loc[
                final_root == 1
            ].copy()
            high_map = high22_predict(
                train_high,
                pred_high_rows,
                high_gate_features,
                high_lower_features,
            ) if not pred_high_rows.empty else {}

            pred_low_rows = test.loc[
                final_root == 0
            ].copy()
            lp = low_predict(
                pred_low_rows
            )
            low_map = {
                int(dn): str(p)
                for dn, p in zip(
                    pred_low_rows[
                        "disasterNumber"
                    ],
                    lp,
                )
            }

            for j, (_, r) in enumerate(
                test.iterrows()
            ):
                dn = int(
                    r["disasterNumber"]
                )
                rp = int(final_root[j])
                final = (
                    high_map[dn]
                    if rp
                    else low_map[dn]
                )
                rows[variant].append({
                    "disasterNumber": dn,
                    "state": r["state"],
                    "incidentType": (
                        r["incidentType"]
                    ),
                    "fyDeclared": int(
                        r["fyDeclared"]
                    ),
                    "actual_band": (
                        r["actual_band"]
                    ),
                    "root_actual_high": int(
                        r["target_clean"]
                        >= 50_000_000
                    ),
                    "root_pred_high": rp,
                    "final_pred": final,
                })

    results = {}
    for variant in variants:
        p = pd.DataFrame(
            rows[variant]
        )
        p.to_csv(
            OUT
            / f"{variant}_predictions.csv",
            index=False,
        )
        y = p[
            "root_actual_high"
        ].to_numpy(int)
        yp = p[
            "root_pred_high"
        ].to_numpy(int)

        high_fn = p.loc[
            (p["root_actual_high"] == 1)
            & (p["root_pred_high"] == 0),
            [
                "disasterNumber",
                "state",
                "incidentType",
                "actual_band",
            ],
        ].to_dict(orient="records")

        results[variant] = {
            "root_high_recall": float(
                recall_score(
                    y,
                    yp,
                    pos_label=1,
                    zero_division=0,
                )
            ),
            "root_high_precision": float(
                precision_score(
                    y,
                    yp,
                    pos_label=1,
                    zero_division=0,
                )
            ),
            "root_fp": int(
                (
                    (y == 0)
                    & (yp == 1)
                ).sum()
            ),
            "end_to_end": (
                six_band_metrics(
                    p, "final_pred"
                )
            ),
            "high_false_negatives": high_fn,
        }

    base_fn = {
        x["disasterNumber"]
        for x in results[
            "baseline"
        ]["high_false_negatives"]
    }
    for variant in variants[1:]:
        variant_fn = {
            x["disasterNumber"]
            for x in results[
                variant
            ]["high_false_negatives"]
        }
        results[variant][
            "safe_same_high_fn_set"
        ] = variant_fn == base_fn

    pd.DataFrame(
        diagnostics
    ).to_csv(
        OUT / "fold_diagnostics.csv",
        index=False,
    )
    ext_audit.to_csv(
        OUT / "external_match_audit.csv",
        index=False,
    )

    summary = {
        "mission_nonfinancial_sha256": ma_hash,
        "eaglei_feature_count": len(
            eagcols
        ),
        "results": results,
    }
    (
        OUT / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    md = [
        "# Hurricane four-band boundary audit",
        "",
        "- Same deterministic baseline architecture.",
        "- All Hurricane veto thresholds selected by inner LFYO.",
        "- EAGLE-I variant remains retrospective external_final.",
        "",
        "| Variant | Safe FN set | High recall | Root FP | Overall | Macro | 1M-50M | 50-200M | 200-500M | 500M+ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for variant, r in results.items():
        e = r["end_to_end"]
        pb = e["per_band"]
        safe = (
            True
            if variant == "baseline"
            else r.get(
                "safe_same_high_fn_set",
                False,
            )
        )
        md.append(
            f"| {variant} | "
            f"{safe} | "
            f"{r['root_high_recall']:.1%} | "
            f"{r['root_fp']} | "
            f"{e['overall_correct']}/{e['overall_total']} "
            f"({e['overall_accuracy']:.1%}) | "
            f"{e['macro_recall']:.1%} | "
            f"{pb['1M-50M']['correct']}/{pb['1M-50M']['total']} "
            f"({pb['1M-50M']['recall']:.1%}) | "
            f"{pb['50-200M']['correct']}/{pb['50-200M']['total']} "
            f"({pb['50-200M']['recall']:.1%}) | "
            f"{pb['200-500M']['correct']}/{pb['200-500M']['total']} "
            f"({pb['200-500M']['recall']:.1%}) | "
            f"{pb['500M+']['correct']}/{pb['500M+']['total']} "
            f"({pb['500M+']['recall']:.1%}) |"
        )

    (
        OUT / "summary.md"
    ).write_text(
        "\n".join(md),
        encoding="utf-8",
    )
    print("\n".join(md))


if __name__ == "__main__":
    main()
