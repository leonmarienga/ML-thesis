#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "master_openfema_40plus.xlsx"
ACCEPTED = ROOT / "audit_inputs" / "post830_six_band_specialists" / "accepted" / "candidate_predictions.csv"
OUT = ROOT / "audit_outputs" / "nonbio_830_six_band_specialist_stack"
OUT.mkdir(parents=True, exist_ok=True)

BANDS = ["0-100K", "100K-1M", "1M-50M", "50-200M", "200-500M", "500M+"]
LOW = BANDS[:3]
HIGH = BANDS[3:]
BONUS_GRID = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]

FEATURES = [
    "state", "incidentType", "expectedResourceLevel", "disasterCategory", "durationClass",
    "durationDays", "declarationDelayDays", "fyDeclared", "ihProgramDeclared", "paProgramDeclared",
    "hmProgramDeclared", "expectedResourceScore", "missionAssignmentCount", "uniqueAgencyCount",
    "uniqueMaTypeCount", "uniquePriorityCount", "responseComplexityScore", "missionDensity", "agencyDensity",
]


def build_preprocessor(df: pd.DataFrame):
    cats = [c for c in FEATURES if c in df.columns and (df[c].dtype == "object" or str(df[c].dtype).startswith("category") or c in {"state","incidentType","expectedResourceLevel","disasterCategory","durationClass"})]
    nums = [c for c in FEATURES if c in df.columns and c not in cats]
    prep = ColumnTransformer([
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cats),
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
        ]), nums),
    ], remainder="drop")
    return prep, cats, nums


def specialist_probabilities(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    prep, _, _ = build_preprocessor(train)
    Xtr = prep.fit_transform(train[FEATURES])
    Xte = prep.transform(test[FEATURES])
    probs = np.zeros((len(test), len(BANDS)), dtype=float)
    y = train["actual_band"].astype(str).values
    for j, band in enumerate(BANDS):
        yy = (y == band).astype(int)
        if yy.sum() == 0:
            probs[:, j] = 0.0
            continue
        if yy.sum() == len(yy):
            probs[:, j] = 1.0
            continue
        model = ExtraTreesClassifier(
            n_estimators=160,
            max_depth=8,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42 + j,
            n_jobs=-1,
        )
        model.fit(Xtr, yy)
        probs[:, j] = model.predict_proba(Xte)[:, 1]
    return probs


def combine(router_preds, probs, bonus: float):
    out = []
    for r, p in zip(router_preds, probs):
        allowed = LOW if r in LOW else HIGH
        idxs = [BANDS.index(b) for b in allowed]
        scores = p.copy()
        scores[BANDS.index(r)] += bonus
        best = max(idxs, key=lambda j: (scores[j], -j))
        out.append(BANDS[best])
    return np.asarray(out, dtype=object)


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    correct = int((y_true == y_pred).sum())
    recalls = recall_score(y_true, y_pred, labels=BANDS, average=None, zero_division=0)
    per_band = {}
    for i, b in enumerate(BANDS):
        n = int((y_true == b).sum())
        c = int(((y_true == b) & (y_pred == b)).sum())
        per_band[b] = {"correct": c, "n": n, "recall": float(recalls[i])}
    return {
        "correct": correct,
        "n": int(len(y_true)),
        "accuracy": float(correct / len(y_true)),
        "macro_recall": float(np.mean(recalls)),
        "per_band": per_band,
        "confusion": confusion_matrix(y_true, y_pred, labels=BANDS).tolist(),
    }


def main():
    accepted = pd.read_csv(ACCEPTED)
    accepted["disasterNumber"] = accepted["disasterNumber"].astype(int)
    pred_col = "candidate_pred_new" if "candidate_pred_new" in accepted.columns else "candidate_pred"
    base_correct = int((accepted[pred_col].astype(str) == accepted["actual_band"].astype(str)).sum())
    if len(accepted) != 912 or base_correct != 830:
        raise AssertionError(f"Expected accepted 830/912 baseline, got {base_correct}/{len(accepted)}")

    master = pd.read_excel(MASTER)
    master["disasterNumber"] = master["disasterNumber"].astype(int)
    if master["disasterNumber"].duplicated().any():
        master = master.sort_values("disasterNumber").drop_duplicates("disasterNumber", keep="first")

    need = ["disasterNumber"] + FEATURES
    missing = [c for c in need if c not in master.columns]
    if missing:
        raise KeyError(f"Missing master columns: {missing}")

    df = accepted.merge(master[need], on="disasterNumber", how="left", validate="one_to_one")
    df["router_pred"] = df[pred_col].astype(str)
    df["actual_band"] = df["actual_band"].astype(str)
    df["fyDeclared"] = pd.to_numeric(df["fyDeclared"], errors="raise").astype(int)

    # Ensure all feature rows joined.
    if df[FEATURES].isna().all(axis=1).any():
        raise AssertionError("Some accepted rows failed master-feature join")

    years = sorted(df["fyDeclared"].unique().tolist())
    outer_predictions = []
    fold_rows = []
    ml_only_predictions = []

    for outer_fy in years:
        print(f"OUTER FY {outer_fy}", flush=True)
        tr = df[df.fyDeclared != outer_fy].copy()
        te = df[df.fyDeclared == outer_fy].copy()

        # Inner LFYO specialist probabilities on outer-training only.
        inner_parts = []
        for inner_fy in sorted(tr.fyDeclared.unique().tolist()):
            tr2 = tr[tr.fyDeclared != inner_fy].copy()
            va = tr[tr.fyDeclared == inner_fy].copy()
            if len(tr2) == 0 or len(va) == 0:
                continue
            p = specialist_probabilities(tr2, va)
            part = va[["disasterNumber", "actual_band", "router_pred", "fyDeclared"]].copy()
            for j, b in enumerate(BANDS):
                part[f"p_{j}"] = p[:, j]
            inner_parts.append(part)
        inner = pd.concat(inner_parts, ignore_index=True)
        Pinner = inner[[f"p_{j}" for j in range(len(BANDS))]].to_numpy(float)

        candidates = []
        for bonus in BONUS_GRID:
            ip = combine(inner.router_pred.astype(str).values, Pinner, bonus)
            m = metrics(inner.actual_band.astype(str).values, ip)
            changes = int((ip != inner.router_pred.astype(str).values).sum())
            candidates.append((m["correct"], m["macro_recall"], -changes, bonus, m, changes))
        # maximize accuracy, then macro recall, then prefer fewer changes; if still tied prefer larger router bonus.
        candidates.sort(key=lambda z: (z[0], z[1], z[2], z[3]), reverse=True)
        _, _, _, selected_bonus, inner_metric, inner_changes = candidates[0]

        pte = specialist_probabilities(tr, te)
        pred = combine(te.router_pred.astype(str).values, pte, selected_bonus)
        ml_only = combine(te.router_pred.astype(str).values, pte, 0.0)

        out = te[["disasterNumber", "state", "incidentType", "fyDeclared", "target_clean", "actual_band", "router_pred"]].copy() if "target_clean" in te.columns else te[["disasterNumber", "state", "incidentType", "fyDeclared", "actual_band", "router_pred"]].copy()
        out["specialist_stack_pred"] = pred
        out["specialist_ml_only_pred"] = ml_only
        out["selected_router_bonus"] = selected_bonus
        for j, b in enumerate(BANDS):
            out[f"specialist_p_{b}"] = pte[:, j]
        outer_predictions.append(out)
        ml_only_predictions.extend(ml_only.tolist())

        fold_rows.append({
            "outer_fy": outer_fy,
            "outer_n": int(len(te)),
            "selected_router_bonus": float(selected_bonus),
            "inner_correct": int(inner_metric["correct"]),
            "inner_n": int(inner_metric["n"]),
            "inner_accuracy": float(inner_metric["accuracy"]),
            "inner_macro_recall": float(inner_metric["macro_recall"]),
            "inner_changes": int(inner_changes),
        })

    pred_df = pd.concat(outer_predictions, ignore_index=True).sort_values("disasterNumber").reset_index(drop=True)
    base = accepted[["disasterNumber", "actual_band", pred_col]].copy().sort_values("disasterNumber").reset_index(drop=True)
    if not np.array_equal(pred_df.disasterNumber.values, base.disasterNumber.values):
        raise AssertionError("Prediction row order mismatch")

    y = pred_df.actual_band.astype(str).values
    router = pred_df.router_pred.astype(str).values
    stack = pred_df.specialist_stack_pred.astype(str).values
    mlonly = pred_df.specialist_ml_only_pred.astype(str).values

    base_m = metrics(y, router)
    stack_m = metrics(y, stack)
    ml_m = metrics(y, mlonly)

    changed = pred_df[stack != router].copy()
    changed["router_correct"] = changed.router_pred == changed.actual_band
    changed["stack_correct"] = changed.specialist_stack_pred == changed.actual_band
    changed["change_effect"] = np.where(~changed.router_correct & changed.stack_correct, "fixed", np.where(changed.router_correct & ~changed.stack_correct, "broken", "sideways"))

    # Root safety diagnostics.
    actual_high = np.isin(y, HIGH)
    router_high = np.isin(router, HIGH)
    stack_high = np.isin(stack, HIGH)
    root = {
        "true_high_n": int(actual_high.sum()),
        "router_high_recall": float((router_high & actual_high).sum() / actual_high.sum()),
        "stack_high_recall": float((stack_high & actual_high).sum() / actual_high.sum()),
        "router_root_fp": int((router_high & ~actual_high).sum()),
        "stack_root_fp": int((stack_high & ~actual_high).sum()),
        "root_membership_changes": int((router_high != stack_high).sum()),
    }

    summary = {
        "baseline": base_m,
        "specialist_ml_only": ml_m,
        "router_plus_six_specialists": stack_m,
        "delta_correct_vs_router": int(stack_m["correct"] - base_m["correct"]),
        "changed_rows": int(len(changed)),
        "fixed_changes": int((changed.change_effect == "fixed").sum()),
        "broken_changes": int((changed.change_effect == "broken").sum()),
        "sideways_changes": int((changed.change_effect == "sideways").sum()),
        "root_safety": root,
        "model": "six one-vs-rest ExtraTrees band specialists + nested-LFYO selected router bonus; root-preserving",
        "features": FEATURES,
        "bonus_grid": BONUS_GRID,
    }

    pred_df.to_csv(OUT / "candidate_predictions.csv", index=False)
    changed.to_csv(OUT / "changed_rows.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "fold_diagnostics.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Six-band specialist stack on accepted 830 router",
        "",
        f"- Router baseline: **{base_m['correct']}/{base_m['n']} = {base_m['accuracy']:.4%}**",
        f"- Specialist ML only (root-preserving): **{ml_m['correct']}/{ml_m['n']} = {ml_m['accuracy']:.4%}**",
        f"- Router + six specialist ML models: **{stack_m['correct']}/{stack_m['n']} = {stack_m['accuracy']:.4%}**",
        f"- Delta vs router: **{stack_m['correct']-base_m['correct']:+d} rows**",
        f"- Macro recall: **{stack_m['macro_recall']:.4%}**",
        f"- Changed rows: **{len(changed)}** | fixed **{summary['fixed_changes']}** | broken **{summary['broken_changes']}** | sideways **{summary['sideways_changes']}**",
        f"- High-value root recall: **{root['stack_high_recall']:.4%} ({root['true_high_n']} true high cases)**",
        f"- Root false positives: **{root['stack_root_fp']}**",
        f"- Root membership changes vs router: **{root['root_membership_changes']}**",
        "",
        "## Per-band recall",
    ]
    for b in BANDS:
        q = stack_m["per_band"][b]
        qb = base_m["per_band"][b]
        lines.append(f"- {b}: **{q['correct']}/{q['n']} = {q['recall']:.2%}** (router {qb['correct']}/{qb['n']} = {qb['recall']:.2%})")
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
