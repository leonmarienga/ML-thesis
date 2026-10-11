#!/usr/bin/env python3
"""Research-only all-training-data refit; DO NOT modify the frozen six-band router.

Preserve one genuinely nested outer-LFYO evaluation separately. This refit has
NO unseen-test metrics: all 13 labeled band cases are available for selecting
the deployed configuration by fiscal-year OOF and fitting its parameters.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import nonbio_50_200m_fast_hybrid as study

OUT = study.ROOT / "audit_outputs" / "nonbio_50_200m_full_training_refit"


def trained_expert(name, full_train):
    kind, low, high, target, param = study.SPECS[name]
    support = full_train[(full_train.target_m >= low) & (full_train.target_m < high)].copy()
    if kind == "analogue":
        return {"kind": "analogue", "k": int(param), "support": support[
            study.CATS + study.NUMS + ["target_m"]
        ].copy(), "name": name}
    if kind == "ridge":
        est = Ridge(alpha=param, solver="lsqr")
    elif kind == "extra":
        est = ExtraTreesRegressor(n_estimators=140, min_samples_leaf=param,
                                  max_features=0.8, n_jobs=-1, random_state=study.SEED)
    elif kind == "forest":
        est = RandomForestRegressor(n_estimators=140, min_samples_leaf=param,
                                    max_features=0.8, n_jobs=-1, random_state=study.SEED)
    elif kind == "ordinal":
        # Threshold family supported via archive to reuse the exact research implementation.
        return {"kind": "archive", "support": support[
            study.CATS + study.NUMS + ["target_m"]
        ].copy(), "name": name}
    else:
        raise ValueError(kind)
    y = np.clip(support.target_m.to_numpy(float), study.LO, study.HI-1e-6)
    pipeline = Pipeline([("pre", study.preprocessor()), ("reg", est)])
    pipeline.fit(support[study.CATS + study.NUMS], np.log(y) if target == "log" else y)
    return {"kind": "pipeline", "pipeline": pipeline, "target": target, "name": name,
            "support_n": len(support)}


def base_predict(expert, test):
    if expert["kind"] == "analogue":
        return study.analogue(expert["support"], test, expert["k"])
    if expert["kind"] == "archive":
        return study.predict_expert(expert["name"], expert["support"], test)
    value = expert["pipeline"].predict(test[study.CATS + study.NUMS])
    if expert["target"] == "log":
        value = np.exp(np.clip(value, np.log(study.LO), np.log(study.HI)))
    return np.clip(value, study.LO, study.HI-1e-6)


def fit_bundle(data):
    selected, ranking = study.select(data)
    family = selected["candidate"]
    members = study.MIXES.get(family, (family,))
    experts = [trained_expert(name, data) for name in members]
    median = float(data.loc[(data.target_m >= study.LO) &
                            (data.target_m < study.HI), "target_m"].median())
    return {
        "version": "research-only-final-refit-20261011",
        "selection": {"family": family, "alpha": float(selected["alpha"]),
                      "inner_MAE_M": float(selected["inner_MAE_M"]),
                      "inner_n": int(selected["inner_n"])},
        "experts": experts, "median_m": median,
        "required_fields": study.CATS + study.NUMS,
        "train_nonbio_n": int(len(data)),
        "train_true_band_n": int(((data.target_m >= study.LO) &
                                   (data.target_m < study.HI)).sum()),
    }, ranking


def predict_usd(bundle, records):
    absent = set(bundle["required_fields"]) - set(records.columns)
    if absent:
        raise ValueError(f"Missing required input features: {sorted(absent)}")
    masked = records.drop(columns=["totalObligatedFunding", "target_m"], errors="ignore").copy()
    # Critical: only the stored historical training support, never a test target.
    base = np.mean([base_predict(expert, masked) for expert in bundle["experts"]], axis=0)
    m = bundle["median_m"]; a = bundle["selection"]["alpha"]
    return np.clip(m + a * (base - m), study.LO, study.HI - 1e-6) * 1e6


def main():
    d = study.load()
    assert len(d) == 912 and ((d.target_m >= study.LO) &
                                (d.target_m < study.HI)).sum() == 13
    bundle, ranking = fit_bundle(d)
    # Verify exported/refitted model reproduces the exact in-memory research family.
    # Training rows are used ONLY for a functional smoke test, NOT accuracy claims.
    smoke = d.iloc[:7].copy()
    masked = smoke.drop(columns=["target_m", "totalObligatedFunding"], errors="ignore")
    actual_exported = predict_usd(bundle, masked)/1e6
    family = bundle["selection"]["family"]
    # candidate_preds computes all candidate families, so use the selected one.
    expected_base = study.candidate_preds(d, masked)[family]
    expected = np.clip(bundle["median_m"] + bundle["selection"]["alpha"] *
                       (expected_base - bundle["median_m"]), study.LO, study.HI-1e-6)
    assert np.allclose(actual_exported, expected, rtol=1e-9, atol=1e-5)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "research_only_amount_bundle.joblib"
    joblib.dump(bundle, path)
    reloaded = joblib.load(path)
    assert np.allclose(predict_usd(reloaded, masked), actual_exported*1e6,
                       rtol=1e-9, atol=1e-3)
    m_path = study.ROOT / "master_openfema_40plus.xlsx"
    report = {
        "scope": "Full-training refit of winner selected by all-data inner LFYO",
        "status": "Research only; not integrated into accepted six-band production stack",
        "selected": bundle["selection"], "median_USD": bundle["median_m"]*1e6,
        "training_n": bundle["train_nonbio_n"], "training_true_band_n": bundle["train_true_band_n"],
        "input_schema": bundle["required_fields"], "joblib_file": path.name,
        "source_workbook_sha256": hashlib.sha256(m_path.read_bytes()).hexdigest(),
        "smoke_test": "Training-row functional parity ONLY. Not a held-out accuracy estimate.",
        "caution": "Model-family choice after exploratory outer scores has research-level selection bias. "
                   "Validate prospectively on additional fiscal years before operational acceptance."
    }
    (OUT/"manifest.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    pd.DataFrame(ranking).to_csv(OUT/"all_training_inner_candidate_scores.csv", index=False)
    print("BEGIN_REFIT_MANIFEST", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    print("END_REFIT_MANIFEST", flush=True)

if __name__ == "__main__":
    main()
