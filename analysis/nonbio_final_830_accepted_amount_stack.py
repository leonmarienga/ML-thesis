#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

import legacy_tempv2 as v2
from mission_semantic_audit import CURRENT_19, build_semantic_rollup, fetch_all_mission_assignments, normalize_master
from nonbio_all_ranges import valid_cols
from nonbio_cross_1m_rescue import fit_base_reg, predict_reg_dollars
from nonbio_830_router_amount_specialists import predict_200_500
from nonbio_500mplus_hybrid_audit import ANALOG_K, ANALOG_SCALE_EXP, ANALOG_SUPPORTS, analogue_predict

ROOT = Path(__file__).resolve().parents[1]
ACCEPTED = ROOT / "audit_inputs" / "final_accepted_amount_stack" / "accepted" / "candidate_predictions.csv"
MASTER = ROOT / "master_openfema_40plus.xlsx"
OUT = ROOT / "audit_outputs" / "nonbio_final_830_accepted_amount_stack"
OUT.mkdir(parents=True, exist_ok=True)

BANDS = ["0-100K", "100K-1M", "1M-50M", "50-200M", "200-500M", "500M+"]
BOUNDS = {
    "0-100K": (0.0, 1e5),
    "100K-1M": (1e5, 1e6),
    "1M-50M": (1e6, 50e6),
    "50-200M": (50e6, 200e6),
    "200-500M": (200e6, 500e6),
    "500M+": (500e6, 6e9),
}


def metrics(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    ly = np.log1p(np.maximum(y, 0))
    lp = np.log1p(np.maximum(p, 0))
    abs_err = np.abs(p - y)
    # Percentage tolerance is undefined/unstable at zero. Evaluate it only for positive targets.
    pos = y > 0
    out = {
        "n": int(len(y)),
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(mean_squared_error(y, p) ** 0.5),
        "MedAE": float(median_absolute_error(y, p)),
        "log_MAE": float(mean_absolute_error(ly, lp)),
        "log_RMSE": float(mean_squared_error(ly, lp) ** 0.5),
        "R2": float(r2_score(y, p)) if len(y) >= 2 and np.var(y) > 0 else None,
        "log_R2": float(r2_score(ly, lp)) if len(y) >= 2 and np.var(ly) > 0 else None,
        "positive_n": int(pos.sum()),
    }
    if pos.any():
        rel = abs_err[pos] / y[pos]
        ratio = np.maximum(p[pos] / y[pos], y[pos] / np.maximum(p[pos], 1.0))
        out.update({
            "within_20pct_positive": float(np.mean(rel <= 0.20)),
            "within_30pct_positive": float(np.mean(rel <= 0.30)),
            "within_50pct_positive": float(np.mean(rel <= 0.50)),
            "within_factor2_positive": float(np.mean(ratio <= 2.0)),
            "within_factor3_positive": float(np.mean(ratio <= 3.0)),
        })
    return out


def band_median(train, band):
    vals = train.loc[train.actual_band.astype(str) == band, "target"].to_numpy(float)
    if len(vals) == 0:
        lo, hi = BOUNDS[band]
        return float(lo if not np.isfinite(hi) else (lo + hi) / 2.0)
    return float(np.median(vals))


def select_analogue_cfg(outer_train):
    extremes = outer_train[outer_train.target > 500e6].copy()
    configs = [(s, k, e) for s in ANALOG_SUPPORTS for k in ANALOG_K for e in ANALOG_SCALE_EXP]
    store = {cfg: [] for cfg in configs}
    yy = []
    for yr in sorted(extremes.fyDeclared.astype(int).unique()):
        tr = outer_train[outer_train.fyDeclared.astype(int) != yr].copy()
        te = extremes[extremes.fyDeclared.astype(int) == yr].copy()
        if te.empty:
            continue
        yy.extend(te.target.to_numpy(float).tolist())
        for cfg in configs:
            s, k, e = cfg
            store[cfg].extend(analogue_predict(tr, te, s, k, e).tolist())
    y = np.asarray(yy, float)
    if len(y) < 2:
        return (500e6, 2, 0.5), {}
    scores = {}
    for cfg, pred in store.items():
        pp = np.asarray(pred, float)
        scores[cfg] = float(mean_absolute_error(np.log1p(y), np.log1p(pp)))
    best = min(scores, key=scores.get)
    return best, {str(k): float(v) for k, v in scores.items()}


def prepare_semantic_frame(accepted_ids):
    master = normalize_master(pd.read_excel(MASTER))
    master["target_clean"] = pd.to_numeric(master["totalObligatedFunding"], errors="coerce").fillna(0).clip(lower=0)
    ma = fetch_all_mission_assignments()
    sem, _ = build_semantic_rollup(master, ma)
    df = master.merge(sem, on="disasterNumber", how="left")
    df["disasterNumber"] = df.disasterNumber.astype(int)
    df = df[df.disasterNumber.isin(accepted_ids)].copy()
    current = valid_cols(df, CURRENT_19)
    semcols = valid_cols(
        df,
        [
            c for c in df.columns
            if (c.startswith("sem_") or c.startswith("ma_"))
            and not any(b in c.lower() for b in ["oblig", "fund", "cost", "amount", "dollar"])
        ],
    )
    features = list(dict.fromkeys(current + semcols))
    return df, features


def main():
    accepted = pd.read_csv(ACCEPTED)
    accepted["disasterNumber"] = accepted.disasterNumber.astype(int)
    pred_col = "candidate_pred_new" if "candidate_pred_new" in accepted.columns else "candidate_pred"
    assert len(accepted) == 912
    assert int((accepted[pred_col].astype(str) == accepted.actual_band.astype(str)).sum()) == 830

    d, high_features = v2.load_data()
    d["disasterNumber"] = d.disasterNumber.astype(int)
    d = d[d.disasterNumber.isin(set(accepted.disasterNumber))].copy()
    d = d.merge(
        accepted[["disasterNumber", "actual_band", pred_col]],
        on="disasterNumber",
        how="inner",
        validate="one_to_one",
    ).rename(columns={pred_col: "router_pred"})
    d["actual_band"] = d.actual_band.astype(str)
    d["router_pred"] = d.router_pred.astype(str)
    d["fyDeclared"] = pd.to_numeric(d.fyDeclared, errors="coerce").astype(int)
    assert len(d) == 912
    assert not (d.incidentType.astype(str).str.lower() == "biological").any()

    semdf, sem_features = prepare_semantic_frame(set(accepted.disasterNumber))
    semdf["fyDeclared"] = pd.to_numeric(semdf.fyDeclared, errors="coerce").astype(int)
    semdf = semdf.merge(
        accepted[["disasterNumber", "actual_band", pred_col]],
        on="disasterNumber",
        how="inner",
        validate="one_to_one",
    ).rename(columns={pred_col: "router_pred"})

    final_pred = pd.Series(index=d.index, dtype=float)
    router_median_pred = pd.Series(index=d.index, dtype=float)
    oracle_pred = pd.Series(index=d.index, dtype=float)
    method = pd.Series(index=d.index, dtype=object)
    oracle_method = pd.Series(index=d.index, dtype=object)
    fold_rows = []

    for year in sorted(d.fyDeclared.unique()):
        train = d[d.fyDeclared != year].copy()
        test = d[d.fyDeclared == year].copy()
        train_sem = semdf[semdf.fyDeclared != year].copy()
        test_sem = semdf[semdf.fyDeclared == year].copy()
        sem_by_dn = test_sem.set_index("disasterNumber", drop=False)

        medians = {b: band_median(train, b) for b in BANDS}
        analogue_cfg, analogue_scores = select_analogue_cfg(train)

        # Fit the accepted $1M-$50M ExtraTrees amount model once per outer year.
        reg = fit_base_reg(train_sem, sem_features, 700000 + int(year))

        # Precompute expensive specialist predictions for any row that could need them
        # under either deployable routing or the oracle-band diagnostic.
        need_mid = test[(test.router_pred == "200-500M") | (test.actual_band == "200-500M")].copy()
        mid_map = {}
        if len(need_mid):
            pp, mm = predict_200_500(d, high_features, need_mid, int(year))
            mid_map = {int(dn): (float(p), str(m)) for dn, p, m in zip(need_mid.disasterNumber, pp, mm)}

        need_ext = test[(test.router_pred == "500M+") | (test.actual_band == "500M+")].copy()
        ext_map = {}
        if len(need_ext):
            s, k, e = analogue_cfg
            pp = analogue_predict(train, need_ext, s, k, e)
            ext_map = {int(dn): float(p) for dn, p in zip(need_ext.disasterNumber, pp)}

        def amount_for(row, chosen_band, oracle=False):
            dn = int(row.disasterNumber)
            if chosen_band in ("0-100K", "100K-1M", "50-200M"):
                return medians[chosen_band], f"{chosen_band}_training_median"
            if chosen_band == "1M-50M":
                sr = sem_by_dn.loc[[dn]]
                p = float(predict_reg_dollars(reg, sr, sem_features)[0])
                p = float(np.clip(p, 1e6, 50e6))
                return p, "1M-50M_preserved_ExtraTrees_clipped"
            if chosen_band == "200-500M":
                p, m = mid_map[dn]
                return float(np.clip(p, 200e6, 500e6)), m
            if chosen_band == "500M+":
                return ext_map[dn], f"500Mplus_nested_analogue_{analogue_cfg}"
            raise ValueError(chosen_band)

        for idx, r in test.iterrows():
            deploy, meth = amount_for(r, str(r.router_pred), oracle=False)
            oracle, ometh = amount_for(r, str(r.actual_band), oracle=True)
            final_pred.loc[idx] = deploy
            oracle_pred.loc[idx] = oracle
            router_median_pred.loc[idx] = medians[str(r.router_pred)]
            method.loc[idx] = meth
            oracle_method.loc[idx] = ometh

        fold_rows.append({
            "fyDeclared": int(year),
            "n": int(len(test)),
            "router_correct": int((test.router_pred == test.actual_band).sum()),
            "analogue_cfg": str(analogue_cfg),
            "analogue_inner_scores": json.dumps(analogue_scores),
        })

    out = d[["disasterNumber", "state", "fyDeclared", "incidentType", "target", "actual_band", "router_pred"]].copy()
    out["router_correct"] = out.actual_band == out.router_pred
    out["router_band_median_prediction"] = router_median_pred
    out["accepted_stack_prediction"] = final_pred
    out["accepted_stack_method"] = method
    out["oracle_band_accepted_prediction"] = oracle_pred
    out["oracle_band_method"] = oracle_method
    out = out.sort_values("disasterNumber").reset_index(drop=True)
    out.to_csv(OUT / "predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "folds.csv", index=False)

    y = out.target.to_numpy(float)
    baseline = out.router_band_median_prediction.to_numpy(float)
    final = out.accepted_stack_prediction.to_numpy(float)
    oracle = out.oracle_band_accepted_prediction.to_numpy(float)

    by_actual_band = {}
    for b in BANDS:
        z = out[out.actual_band == b]
        by_actual_band[b] = {
            "router_band_median": metrics(z.target, z.router_band_median_prediction),
            "accepted_stack": metrics(z.target, z.accepted_stack_prediction),
            "oracle_band_accepted": metrics(z.target, z.oracle_band_accepted_prediction),
            "router_accuracy_within_actual_band": float(z.router_correct.mean()),
        }

    correct = out[out.router_correct]
    wrong = out[~out.router_correct]
    summary = {
        "scope": "912 non-Biological disasters behind frozen 830/912 six-band router",
        "router": {
            "correct": int(out.router_correct.sum()),
            "n": int(len(out)),
            "accuracy": float(out.router_correct.mean()),
        },
        "overall": {
            "router_band_median_baseline": metrics(y, baseline),
            "accepted_amount_stack": metrics(y, final),
            "oracle_band_accepted_stack": metrics(y, oracle),
        },
        "by_actual_band": by_actual_band,
        "correctly_routed_subset": metrics(correct.target, correct.accepted_stack_prediction),
        "misrouted_subset": metrics(wrong.target, wrong.accepted_stack_prediction),
        "comparisons": {
            "MAE_reduction_vs_router_band_median_pct": float(100 * (mean_absolute_error(y, baseline) - mean_absolute_error(y, final)) / mean_absolute_error(y, baseline)),
            "RMSE_reduction_vs_router_band_median_pct": float(100 * ((mean_squared_error(y, baseline) ** .5) - (mean_squared_error(y, final) ** .5)) / (mean_squared_error(y, baseline) ** .5)),
            "R2_gain_vs_router_band_median": float(r2_score(y, final) - r2_score(y, baseline)),
            "router_cost_MAE_vs_oracle_pct": float(100 * (mean_absolute_error(y, final) - mean_absolute_error(y, oracle)) / mean_absolute_error(y, oracle)),
        },
        "accepted_band_choices": {
            "0-100K": "outer-training actual-band median",
            "100K-1M": "outer-training actual-band median",
            "1M-50M": "preserved 900-tree ExtraTrees log-dollar regressor, outer-year excluded, prediction clipped to routed band",
            "50-200M": "outer-training actual-band median; split ML specialists rejected",
            "200-500M": "preserved target-free 200-300/300-500 specialist stack",
            "500M+": "nested training-selected historical analogue",
        },
        "protocol": [
            "Biological/COVID is excluded.",
            "The frozen broad router predictions are reproduced from accepted run 34701792537 and are not retuned here.",
            "Every supervised amount fit excludes the entire held fiscal year.",
            "Deployable amount choice is based on the frozen router prediction, never the held-out row's true funding band.",
            "Oracle-band accepted-stack metrics are diagnostic only and quantify the residual cost of broad-band routing.",
            "Simple band medians are calculated only from outer-training rows in that actual band.",
            "The 500M+ analogue configuration is selected by inner leave-fiscal-year-out log-MAE on outer-training extremes only.",
            "The 200M-500M stack retains its existing same-event target-free context caveat and small-sample development-validation caveat.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# Final non-Biological frozen-router + accepted-amount-stack evaluation",
        "",
        f"Router: {summary['router']['correct']}/{summary['router']['n']} = {summary['router']['accuracy']:.4%}",
        "",
    ]
    for name, m in summary["overall"].items():
        lines += [
            f"## {name}",
            f"- R2: {m['R2']}",
            f"- MAE: ${m['MAE']:,.0f}",
            f"- RMSE: ${m['RMSE']:,.0f}",
            f"- MedAE: ${m['MedAE']:,.0f}",
            f"- log MAE: {m['log_MAE']}",
            f"- log R2: {m['log_R2']}",
            "",
        ]
    lines.append("## By actual band")
    for b, block in by_actual_band.items():
        m = block["accepted_stack"]
        lines.append(f"- {b}: n={m['n']}, R2={m['R2']}, MAE=${m['MAE']:,.0f}, RMSE=${m['RMSE']:,.0f}, log_MAE={m['log_MAE']:.4f}")
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
