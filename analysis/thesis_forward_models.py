#!/usr/bin/env python3
"""Matched retrospective ablations with nested, purged forward validation."""
import argparse
from itertools import product
from pathlib import Path

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error,
                             median_absolute_error, precision_recall_fscore_support, r2_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from thesis_hazard_features import BANDS, EXPOSURE, PHYSICAL, PROXY

SEED = 20261117
LABELS = np.arange(len(BANDS))
YEARS = range(2019, 2025)
CATS = ["incidentType", "state", "declarationType"]
META = ["incident_month", "declaration_delay_days", "ihProgramDeclared", "iaProgramDeclared",
        "paProgramDeclared", "hmProgramDeclared"]
RETRO = [*META, "incident_duration_days"]
VERIFIED_PHYSICAL = [c for c in PHYSICAL if c.startswith(("cyclone_", "mtbs_"))]
ARMS = {
    "metadata": META,
    "retrospective_metadata": RETRO,
    "geography": RETRO + EXPOSURE,
    "report_proxy": RETRO + PROXY,
    "physical": RETRO + PHYSICAL,
    "geography_physical": RETRO + EXPOSURE + PHYSICAL,
    "combined": RETRO + EXPOSURE + PROXY + PHYSICAL,
    "verified_link_only": RETRO + EXPOSURE + VERIFIED_PHYSICAL,
}
GRID = {"linear": (0.5, 5.0), "extra_trees": (2, 6), "random_forest": (2, 6), "hist_boost": (7, 15)}


def read_table(path):
    rows = pd.read_csv(path, parse_dates=["start", "end", "declaration"])
    d = rows.query("incidentType != 'Biological'").copy()
    assert len(d) == 912 and d.disasterNumber.is_unique
    d[META[2:]] = d[META[2:]].astype(float)
    return d


def split(d, year):
    test = d[d.fyDeclared.eq(year)].copy()
    latest_safe = min(test.start.min(), test.declaration.min())
    end = d.end.fillna(d.start)
    train = d[d.fyDeclared.lt(year) & end.lt(latest_safe) & ~d.event_group.isin(test.event_group)].copy()
    assert train.fyDeclared.max() < year and len(train) > 40
    assert not set(train.event_group) & set(test.event_group)
    return train, test


def estimators(family, hp):
    if family == "linear":
        return (LogisticRegression(C=1 / hp, max_iter=2000, class_weight="balanced", random_state=SEED),
                Ridge(alpha=hp))
    if family == "hist_boost":
        return (HistGradientBoostingClassifier(max_leaf_nodes=hp, max_iter=70, min_samples_leaf=12,
                                              random_state=SEED, class_weight="balanced"),
                HistGradientBoostingRegressor(max_leaf_nodes=hp, max_iter=70, min_samples_leaf=12,
                                             random_state=SEED, loss="absolute_error"))
    cls, reg = (ExtraTreesClassifier, ExtraTreesRegressor) if family == "extra_trees" else (RandomForestClassifier, RandomForestRegressor)
    kwargs = dict(n_estimators=80, min_samples_leaf=hp, max_features=.8, random_state=SEED, n_jobs=1)
    return cls(**kwargs, class_weight="balanced"), reg(**kwargs)


def fit_predict(train, test, fields, family, hp):
    classification, regression = estimators(family, hp)
    numeric = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), StandardScaler())
    transformer = ColumnTransformer([("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATS),
                                     ("numeric", numeric, fields)])
    a, b = transformer.fit_transform(train), transformer.transform(test)
    classification.fit(a, train.actual_band)
    regression.fit(a, np.log1p(train.target))
    probabilities = np.zeros((len(test), len(LABELS)))
    probabilities[:, classification.classes_] = classification.predict_proba(b)
    amount = np.expm1(regression.predict(b)).clip(0, max(train.target))
    return probabilities, amount


def score(actual, pred, probabilities, labels=LABELS):
    cls = probabilities.argmax(axis=1)
    y = actual.target.to_numpy()
    actual_labels = actual.actual_band.to_numpy(int)
    precision, recall, f1, support = precision_recall_fscore_support(actual_labels, cls, labels=labels, zero_division=0)
    indices = support > 0
    result = {
        "n": len(actual), "accuracy": accuracy_score(actual_labels, cls),
        "balanced_accuracy": recall[indices].mean(), "macro_precision": precision.mean(),
        "macro_recall": recall.mean(), "macro_f1": f1.mean(),
        "mae_usd": mean_absolute_error(y, pred), "rmse_usd": mean_squared_error(y, pred) ** .5,
        "median_ae_usd": median_absolute_error(y, pred), "r2": r2_score(y, pred) if len(y) > 1 and np.var(y) else np.nan,
        "log_mae": mean_absolute_error(np.log1p(y), np.log1p(pred)),
        "multiclass_brier": np.mean(((np.eye(len(LABELS))[actual_labels] - probabilities)**2).sum(axis=1)),
    }
    return result


def weighted_amount(prob, amounts):
    return (prob * amounts).sum(axis=1)


def specialists(train, test, prob):
    overall = train.target.median()
    basis = train.groupby("actual_band").target.median().reindex(LABELS, fill_value=overall).to_numpy()
    hazard = train.groupby(["incidentType", "actual_band"]).target.agg(["size", "median"])
    qualified = hazard[hazard["size"] >= 8]["median"]
    hybrid = np.array([[qualified.get((incident, band), basis[band]) for band in LABELS] for incident in test.incidentType])
    return {
        "hard_funding_band": basis[prob.argmax(axis=1)],
        "soft_funding_band": weighted_amount(prob, basis),
        "hybrid_hazard_band": weighted_amount(prob, hybrid),
    }


def hazard_experts(train, test, fields, probabilities, amount):
    p, a = probabilities.copy(), amount.copy()
    for hazard, group in train.groupby("incidentType"):
        present = test.incidentType.eq(hazard).to_numpy()
        if len(group) >= 75 and group.actual_band.nunique() >= 3 and present.any():
            q, y = fit_predict(group, test.loc[present], fields, "extra_trees", 6)
            p[present], a[present] = q, y
    return p, a


def choose_blend(splits, inner_fit, fields):
    results = []
    for i, (_train, test) in enumerate(splits):
        p0, a0 = inner_fit(i, fields, "linear", 5.0)
        p1, a1 = inner_fit(i, fields, "extra_trees", 6)
        results.append((test, p0, a0, p1, a1))
    trials = []
    for weight in (0, .25, .5, .75, 1):
        vals = [score(y, weight*a0+(1-weight)*a1, weight*p0+(1-weight)*p1) for y, p0, a0, p1, a1 in results]
        trials.append((weight, np.mean([v["macro_f1"] for v in vals]), np.mean([v["mae_usd"] for v in vals])))
    cls = max(trials, key=lambda x: (x[1], -abs(x[0]-.5)))[0]
    reg = min(trials, key=lambda x: (x[2], abs(x[0]-.5)))[0]
    return cls, reg


def fit_year(d, year):
    train, test = split(d, year)
    # Every inner validation record comes from the already-purged outer training set.
    splits = [split(train, y) for y in (year - 2, year - 1)]
    cached = {}
    def inner_fit(i, fields, family, hp):
        key = (i, tuple(fields), family, hp)
        if key not in cached:
            a, b = splits[i]
            cached[key] = fit_predict(a, b, fields, family, hp)
        return cached[key]

    outer = {}
    def outer_fit(fields, family, hp):
        key = (tuple(fields), family, hp)
        if key not in outer:
            outer[key] = fit_predict(train, test, fields, family, hp)
        return outer[key]

    base = test[["disasterNumber", "fyDeclared", "incidentType", "state", "event_group", "actual_band", "target"]]
    predictions, tuning = [], []
    def save(name, p, q):
        output = base.assign(variant=name, predicted_band=p.argmax(axis=1), predicted_amount=q,
                             top_band_probability=p.max(axis=1))
        for i in LABELS:
            output[f"p_band_{i}"] = p[:, i]
        predictions.append(output)

    for name, fields in ARMS.items():
        save(f"ablation_{name}", *outer_fit(fields, "extra_trees", 6))

    for arm in ("metadata", "combined", "verified_link_only"):
        fields = ARMS[arm]
        choices = []
        for family, params in GRID.items():
            for hp in params:
                result = pd.DataFrame([score(valid, fit[1], fit[0])
                                       for i, (_a, valid) in enumerate(splits)
                                       for fit in [inner_fit(i, fields, family, hp)]]).mean()
                choices.append({"family": family, "parameter": hp, "inner_macro_f1": result.macro_f1,
                                "inner_mae_usd": result.mae_usd})
        best_p = max(choices, key=lambda s: s["inner_macro_f1"])
        best_a = min(choices, key=lambda s: s["inner_mae_usd"])
        for setting in choices:
            tuning.append({"year": year, "arm": arm, **setting,
                           "selected_band_model": setting is best_p, "selected_amount_model": setting is best_a})
        p = outer_fit(fields, best_p["family"], best_p["parameter"])[0]
        a = outer_fit(fields, best_a["family"], best_a["parameter"])[1]
        save(f"nested_{arm}", p, a)

    fields = ARMS["verified_link_only"]
    p0, a0 = outer_fit(fields, "linear", 5)
    p1, a1 = outer_fit(fields, "extra_trees", 6)
    save("equal_blend_verified", (p0 + p1)/2, (a0 + a1)/2)
    cw, rw = choose_blend(splits, inner_fit, fields)
    save("oof_weighted_blend_verified", cw*p0+(1-cw)*p1, rw*a0+(1-rw)*a1)
    save("hazard_experts_verified", *hazard_experts(train, test, fields, p1, a1))
    for variant, amounts in specialists(train, test, p1).items():
        save(variant, p1, amounts)
    return pd.concat(predictions), tuning, {
        "test_year": year, "n_train": len(train), "n_test": len(test), "max_train_fy": train.fyDeclared.max(),
        "max_train_end": train.end.fillna(train.start).max(), "min_test_incident_start": test.start.min(),
        "rejected_earlier_records": int(d.fyDeclared.lt(year).sum() - len(train)),
        "inner_years": f"{year-2},{year-1}", "linear_probability_weight": cw, "linear_amount_weight": rw,
    }, train.disasterNumber.tolist()


def detailed_metrics(predictions, output):
    metrics, bands, matrices = [], [], []
    pcols = [f"p_band_{i}" for i in LABELS]
    for variant, d in predictions.groupby("variant"):
        for axis, values in [("all", [("all", d)]), ("year", d.groupby("fyDeclared")), ("hazard", d.groupby("incidentType"))]:
            for group, data in values:
                metrics.append({"variant": variant, "axis": axis, "group": group,
                                **score(data, data.predicted_amount.to_numpy(), data[pcols].to_numpy())})
        p, r, f, n = precision_recall_fscore_support(d.actual_band, d.predicted_band, labels=LABELS, zero_division=0)
        for i in LABELS:
            subset = d[d.actual_band.eq(i)]
            ae = (subset.target - subset.predicted_amount).abs()
            bands.append({"variant": variant, "band": BANDS[i], "n": n[i], "precision": p[i], "recall": r[i], "f1": f[i],
                          "mae_usd": ae.mean(), "rmse_usd": np.sqrt((ae**2).mean()), "median_ae_usd": ae.median()})
        mat = confusion_matrix(d.actual_band, d.predicted_band, labels=LABELS)
        matrices.extend({"variant": variant, "actual_band": BANDS[i], "predicted_band": BANDS[j], "n": mat[i, j]}
                        for i, j in product(LABELS, repeat=2))
    for file, rows in [("model_metrics", metrics), ("model_band_metrics", bands), ("confusion_matrices", matrices)]:
        pd.DataFrame(rows).to_csv(output/f"{file}.csv", index=False)


def paired_differences(predictions, output):
    base = predictions.query("variant == 'ablation_retrospective_metadata'").set_index("disasterNumber")
    rng = np.random.default_rng(SEED)
    rows = []
    for variant, frame in predictions.groupby("variant"):
        d = frame.set_index("disasterNumber").join(base[["predicted_amount", "predicted_band"]], rsuffix="_base")
        d["delta_ae"] = abs(d.predicted_amount - d.target) - abs(d.predicted_amount_base - d.target)
        d["delta_accuracy"] = d.predicted_band.eq(d.actual_band).astype(float) - d.predicted_band_base.eq(d.actual_band)
        grouped = d.groupby("event_group").agg(n=("target", "size"), delta_ae=("delta_ae", "sum"),
                                                delta_accuracy=("delta_accuracy", "sum")).to_numpy()
        draw = rng.integers(0, len(grouped), size=(3000, len(grouped)))
        sample = grouped[draw].sum(axis=1)
        lo, hi = np.quantile(sample[:, 1:] / sample[:, :1], [.025, .975], axis=0)
        rows.append({"variant": variant, "mae_difference_usd": d.delta_ae.mean(), "mae_ci_low": lo[0], "mae_ci_high": hi[0],
                     "accuracy_difference": d.delta_accuracy.mean(), "accuracy_ci_low": lo[1], "accuracy_ci_high": hi[1],
                     "clusters": len(grouped), "draws": 3000})
    pd.DataFrame(rows).to_csv(output/"paired_event_bootstrap.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = read_table(args.cohort)
    results = Parallel(n_jobs=args.workers)(delayed(fit_year)(data, year) for year in YEARS)
    predictions = pd.concat([r[0] for r in results], ignore_index=True)
    assert not predictions.duplicated(["disasterNumber", "variant"]).any()
    assert predictions.groupby("variant").size().eq(376).all()
    predictions.to_csv(args.output/"forward_predictions.csv", index=False)
    tuning = [row for result in results for row in result[1]]
    pd.DataFrame(tuning).to_csv(args.output/"nested_hyperparameters.csv", index=False)
    pd.DataFrame([r[2] for r in results]).to_csv(args.output/"forward_fold_audit.csv", index=False)
    pd.DataFrame([{"test_year": r[2]["test_year"], "disasterNumber": number}
                  for r in results for number in r[3]]).to_csv(args.output/"forward_training_roster.csv", index=False)
    detailed_metrics(predictions, args.output)
    paired_differences(predictions, args.output)
    print(pd.read_csv(args.output/"model_metrics.csv").query("axis == 'all'").to_string(index=False))


if __name__ == "__main__":
    main()
