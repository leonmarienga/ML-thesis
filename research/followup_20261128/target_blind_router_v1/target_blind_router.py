#!/usr/bin/env python3
"""Fixed target-blind v1, with split, perturbation, sealing and historical scoring."""
import argparse
import hashlib
import itertools
import json
import platform
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

SHA = "b0b6edb229d112dd6d11896a512bf1112414e4f6"
DIGEST = "319a45fc486dbf1dc07e3b7190256575cca9031144ecdac294d86b008f8b0e92"
SOURCE = "research/results/hazards/cohort_features.csv"
YEARS = tuple(range(2019, 2025))
ID = "disasterNumber"
YEAR = "fyDeclared"
EVENT = "event_group"
CATS = ["incidentType", "state", "declarationType"]
DATES = ["start", "end", "declaration"]
META = [ID, YEAR, EVENT, *CATS, *DATES]
Y = "actual_band"
HAZARDS = ("Severe Ice Storm", "Fire", "Hurricane")
MODELS = ("baseline", "candidate")
LABELS = np.arange(6)
BANDS = ("<100K", "100K-<1M", "1M-<50M", "50M-<200M", "200M-<500M", ">=500M")
SEED = 20261128
N_BOOT = 10000


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_path(repo, protocol):
    assert sha(protocol) == protocol.with_name("PROTOCOL.sha256").read_text().split()[0]
    ref = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    assert ref == SHA
    clean = subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--", SOURCE], check=False)
    assert clean.returncode == 0
    p = repo / SOURCE
    assert sha(p) == DIGEST
    return p


def input_tables(p):
    meta = pd.read_csv(p, usecols=META, parse_dates=DATES,
                       dtype={ID: int, YEAR: int, EVENT: int})
    assert len(meta) == 971 and meta[ID].is_unique
    assert set(meta[YEAR]) == set(range(2010, 2025))
    assert meta[["start", "declaration"]].notna().all().all()
    meta = meta.loc[meta.incidentType.ne("Biological")].set_index(ID).sort_index()
    assert len(meta) == 912 and set(HAZARDS) <= set(meta.incidentType)
    assert set(meta.columns) == (set(META) - {ID})
    labels = pd.read_csv(p, usecols=[ID, Y], dtype={ID: int, Y: int}).set_index(ID).loc[meta.index, Y]
    assert labels.notna().all() and set(labels) <= set(LABELS)
    return meta, labels


def split(m, year):
    test = m.loc[m[YEAR].eq(year)].copy()
    cutoff = min(test.start.min(), test.declaration.min())
    earlier = m[YEAR].lt(year)
    past = m.end.fillna(m.start).lt(cutoff)
    separate = ~m[EVENT].isin(test[EVENT])
    train = m.loc[earlier & past & separate].copy()
    assert len(test) and len(train) > 40 and train[YEAR].max() < year
    assert train.end.fillna(train.start).max() < cutoff
    assert set(train[EVENT]).isdisjoint(test[EVENT])
    return train, test, {
        "origin_fy": year, "n_train": len(train), "n_held": len(test),
        "held_group_count": test[EVENT].nunique(), "max_train_fy": train[YEAR].max(),
        "min_held_start_or_declaration": cutoff.date().isoformat(),
        "max_train_end_or_start": train.end.fillna(train.start).max().date().isoformat(),
        "removed_by_time": int((earlier & ~past).sum()),
        "removed_by_groups_after_time": int((earlier & past & ~separate).sum()),
        "held_groups_in_training": len(set(train[EVENT]) & set(test[EVENT])),
    }


def cats(m):
    return m[CATS].fillna("MISSING").astype(str)


def numeric(m):
    month = m.start.dt.month.to_numpy(dtype=float)
    theta = 2 * np.pi * (month - 1) / 12
    days = (m.declaration - m.start).dt.days.to_numpy(dtype=float)
    logdays = np.log1p(np.where(days >= 0, days, np.nan))
    return np.sin(theta), np.cos(theta), logdays


def transform(train, held):
    # Only these fields, not full split metadata, enter the transformer.
    allowed = [*CATS, "start", "declaration"]
    t, h = train[allowed].copy(), held[allowed].copy()
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    z = enc.fit_transform(cats(t))
    q = enc.transform(cats(h))
    first = sum(len(x) for x in enc.categories_[:-1])
    assert first < z.shape[1]
    ts, tc, td = numeric(t)
    hs, hc, hd = numeric(h)
    median = np.nanmedian(td)
    assert np.isfinite(median)
    timputed = np.where(np.isnan(td), median, td)
    mean, sd = timputed.mean(), timputed.std()
    if sd == 0:
        sd = 1.
    def design(m, oh, sine, cosine, delay_imputed, missing):
        delay = (delay_imputed - mean) / sd
        base = np.column_stack((oh, sine, cosine, delay, missing))
        ice, fire, hurricane = (m.incidentType.eq(k).to_numpy().astype(float) for k in HAZARDS)
        gap = np.column_stack((oh[:, first:] * ice[:, None], delay * fire,
                               oh[:, first:] * hurricane[:, None], delay * hurricane))
        expanded = np.column_stack((base, gap))
        assert np.isfinite(expanded).all()
        return base, expanded
    return (design(t, z, ts, tc, timputed, np.isnan(td)),
            design(h, q, hs, hc, np.where(np.isnan(hd), median, hd), np.isnan(hd)))


def predict_fold(meta, source_labels, year):
    train, held, audit = split(meta, year)
    train_y = source_labels.loc[train.index].to_numpy(dtype=int)
    assert not held.index.isin(train.index).any()
    (xt, xa), (ht, ha) = transform(train, held)
    probs = []
    for x, h in ((xt, ht), (xa, ha)):
        model = LogisticRegression(C=1., solver="lbfgs", max_iter=3000)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(x, train_y)
        p = np.zeros((len(held), len(LABELS)))
        p[:, model.classes_] = model.predict_proba(h)
        assert np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1.)
        probs.append(p)
    raw_base, raw_plus = probs
    base_class = raw_base.argmax(axis=1)
    eligible = held.incidentType.isin(HAZARDS).to_numpy() & (base_class < 3)
    candidate = np.where(eligible[:, None], raw_plus, raw_base)
    p = held[[YEAR, EVENT, "incidentType"]].copy()
    p["valid_elapsed"] = held.declaration.ge(held.start)
    p["gate_eligible"] = eligible
    for name, m in zip(MODELS, (raw_base, candidate)):
        p[f"{name}_prediction"] = m.argmax(axis=1)
        for j in LABELS:
            p[f"{name}_p{j}"] = m[:, j]
    assert all((p.baseline_prediction == p.candidate_prediction) | eligible)
    audit.update({
        "baseline_columns": xt.shape[1], "candidate_columns": xa.shape[1],
        "train_held_declaration_id_intersection": len(set(train.index) & set(held.index)),
    })
    roster = train[[YEAR, EVENT]].assign(origin_fy=year)
    return p, audit, roster


def encode_output(pred):
    return pred.to_csv(float_format="%.17g", lineterminator="\n").encode("utf-8")


def stress_test(meta, labels):
    tables, checks, rosters = [], [], []
    for year in YEARS:
        expected, audit, roster = predict_fold(meta, labels, year)
        hi = expected.index
        original = labels.loc[hi].to_numpy()
        rng = np.random.default_rng(SEED + year)
        alterations = [
            ("permutation", rng.permutation(original)),
            ("reassignment", (np.arange(len(hi), dtype=int) + year) % 6),
        ]
        audit["prediction_digest"] = hashlib.sha256(encode_output(expected)).hexdigest()
        for name, replacement in alterations:
            modified = labels.copy()
            modified.loc[hi] = replacement
            changed = int(np.sum(original != replacement))
            assert changed > 0
            other, other_audit, other_roster = predict_fold(meta, modified, year)
            pd.testing.assert_frame_equal(expected, other, check_exact=True)
            pd.testing.assert_frame_equal(roster, other_roster, check_exact=True)
            assert encode_output(expected) == encode_output(other)
            assert all(v == other_audit[k] for k, v in audit.items() if k in other_audit)
            audit.update({f"{name}_held_labels_changed": changed, f"{name}_invariance": True})
        tables.append(expected)
        checks.append(audit)
        rosters.append(roster)
    pred = pd.concat(tables)
    assert pred.index.is_unique and len(pred) == meta[YEAR].isin(YEARS).sum()
    assert Y not in pred.columns
    return pred, pd.DataFrame(checks), pd.concat(rosters)


def ratio(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)


def ctab(y, p):
    return np.bincount(6 * y + p, minlength=36).reshape(6, 6)


def mf1(m):
    diag = np.diagonal(m, axis1=-2, axis2=-1)
    return ratio(2 * diag, m.sum(-1) + m.sum(-2)).mean(-1)


def metrics(d, model):
    y = d[Y].to_numpy()
    p = d[f"{model}_prediction"].to_numpy()
    q = d[[f"{model}_p{i}" for i in LABELS]].to_numpy()
    c = ctab(y, p)
    summary = {
        "model": model, "n": len(d), "episode_groups": d[EVENT].nunique(),
        "correct": int(np.sum(y == p)), "accuracy": float(np.mean(y == p)),
        "macro_f1": float(mf1(c)), "mean_ordinal_absolute_band_error": float(np.mean(abs(y - p))),
        "multiclass_brier": float(np.mean(((np.eye(6)[y] - q) ** 2).sum(1))),
    }
    for k, name in ((3, "ge_50M"), (4, "ge_200M")):
        support = int(np.sum(y >= k))
        summary[f"{name}_support"] = support
        summary[f"{name}_recall"] = float(np.mean(p[y >= k] >= k)) if support else None
    return summary, c


def paired_bootstrap(d):
    matrices = np.stack([np.stack([ctab(g[Y].to_numpy(), g[f"{model}_prediction"].to_numpy()) for model in MODELS])
                         for _, g in d.groupby(EVENT, sort=True)])
    groups = len(matrices)
    rng = np.random.default_rng(SEED)
    draws = []
    for i in range(0, N_BOOT, 500):
        counts = rng.multinomial(groups, np.full(groups, 1 / groups), size=min(500, N_BOOT-i))
        totals = (counts @ matrices.reshape(groups, -1)).reshape(-1, 2, 6, 6)
        scores = mf1(totals)
        draws.append(scores[:, 1] - scores[:, 0])
    return np.concatenate(draws)


def save_results(d, pred, audit, training, out, repo):
    def save_table(name, records):
        pd.DataFrame(records).to_csv(out / name, index=False)
    results, bands, confusion, coverage = [], [], [], []
    strata = [("overall", "all", d), ("gate", "eligible", d.loc[d.gate_eligible])]
    strata += [("year", str(year), part) for year, part in d.groupby(YEAR)]
    strata += [("hazard", h, part) for h, part in d.groupby("incidentType") if h in HAZARDS]
    for dimension, group, part in strata:
        if part.empty:
            continue
        for model in MODELS:
            row, c = metrics(part, model)
            results.append({"dimension": dimension, "group": group, **row})
            if dimension == "overall":
                for j in LABELS:
                    support = int(c[j, :].sum())
                    pred_n = int(c[:, j].sum())
                    tn = int(c[j, j])
                    bands.append({"model": model, "band": BANDS[j], "actual_n": support,
                                  "predicted_n": pred_n, "correct_n": tn,
                                  "recall": tn / support if support else None,
                                  "precision": tn / pred_n if pred_n else None})
                confusion += [{"model": model, "actual_band": BANDS[i], "predicted_band": BANDS[j], "count": int(c[i, j])}
                              for i, j in itertools.product(LABELS, repeat=2)]
    for hz in ("All", *HAZARDS):
        for yr in ("All", *YEARS):
            part = d if hz == "All" else d.loc[d.incidentType.eq(hz)]
            part = part if yr == "All" else part.loc[part[YEAR].eq(yr)]
            change = part.baseline_prediction.ne(part.candidate_prediction)
            coverage.append({"hazard": hz, "year": yr, "held_n": len(part),
                             "valid_elapsed_n": int(part.valid_elapsed.sum()),
                             "gate_eligible_n": int(part.gate_eligible.sum()),
                             "changed_n": int(change.sum()),
                             "changed_group_n": part.loc[change, EVENT].nunique(),
                             "changed_corrected_n": int((change & part.baseline_prediction.ne(part[Y]) & part.candidate_prediction.eq(part[Y])).sum()),
                             "changed_new_error_n": int((change & part.baseline_prediction.eq(part[Y]) & part.candidate_prediction.ne(part[Y])).sum()),
                             "actual_ge_50m_n": int(part[Y].ge(3).sum()),
                             "actual_ge_50m_eligible_n": int((part[Y].ge(3) & part.gate_eligible).sum()),
                             "actual_ge_50m_changed_n": int((part[Y].ge(3) & change).sum())})
    boot = paired_bootstrap(d)
    low, high = np.quantile(boot, [.025, .975])
    result_base, result_candidate = results[:2]
    delta = result_candidate["macro_f1"] - result_base["macro_f1"]
    ge_recall_guard = all(result_candidate[f"{k}_recall"] is not None and result_base[f"{k}_recall"] is not None and
                        result_candidate[f"{k}_recall"] >= result_base[f"{k}_recall"] for k in ("ge_50M", "ge_200M"))
    change_groups = coverage[0]["changed_group_n"]
    summary = {"status": "exploratory_development_only", "source_commit": SHA, "source_path": SOURCE, "source_sha256": DIGEST,
               "protocol_sha256": sha(Path(__file__).resolve().with_name("PROTOCOL.md")),
               "code_sha256": sha(Path(__file__).resolve()), "python": platform.python_version(), "numpy": np.__version__,
               "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
               "fy_2025_accessed": False, "frozen_router_predictions_read": False,
               "prediction_csv_sha256": sha(out / "predictions_without_labels.csv"),
               "baseline": result_base, "candidate": result_candidate, "all_coverage": coverage[0],
               "paired_episode_macro_f1_delta": delta, "paired_episode_ci_95": [float(low), float(high)],
               "paired_episode_bootstrap_draws": N_BOOT, "upper_threshold_recall_guard_met": ge_recall_guard,
               "min_10_changed_episode_groups_met": change_groups >= 10,
               "independent_replication_priority_gate_met": bool(low > 0 and ge_recall_guard and change_groups >= 10),
               "held_outcome_stress_origins_passed": int((audit.permutation_invariance & audit.reassignment_invariance).sum())}
    save_table("metrics.csv", results)
    save_table("band_metrics.csv", bands)
    save_table("confusion.csv", confusion)
    save_table("eligibility_and_changes.csv", coverage)
    save_table("bootstrap_episode_delta.csv", {"macro_f1_delta": boot})
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args()
    p = check_path(a.repo, Path(__file__).resolve().with_name("PROTOCOL.md"))
    out = a.output
    out.mkdir(parents=True, exist_ok=False)
    meta, labels = input_tables(p)
    pred, audit, training = stress_test(meta, labels)
    (out / "predictions_without_labels.csv").write_bytes(encode_output(pred))
    audit.to_csv(out / "forward_fold_checks.csv", index=False)
    training.to_csv(out / "training_roster.csv")
    # This join is deliberately below all fits, predictions, paired perturbations and sealing.
    d = pred.join(labels, how="left", validate="one_to_one")
    assert d[Y].notna().all()
    save_results(d, pred, audit, training, out, a.repo)


if __name__ == "__main__":
    main()
