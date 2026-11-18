#!/usr/bin/env python3
"""Replay matched forward specialists and measure sparse fiscal-obligation tails."""
import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import genpareto
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from thesis_forward_models import ARMS, CATS, LABELS, META, SEED, YEARS, fit_predict, read_table, split
from thesis_uncertainty import conformal_quantile

TAIL = 50_000_000
ALERT = .10
THRESHOLDS = (5_000_000, 10_000_000, 20_000_000)
PRIOR_EPISODES = 8
MIN_GPD_EPISODES = 25
LIMITS = np.array([0, 1e5, 1e6, 50e6, 200e6, 500e6, np.inf])
EXTRA = [
    "clim_max_1day_mm", "clim_max_county_total_mm", "wind_county_overlap_km2",
    "wind_blockgroup_population_estimate", "fire_mapped_union_acres",
    "fire_blockgroup_housing_estimate",
]
FAMILIES = {
    "Hurricane": "cyclone", "Tropical Storm": "cyclone", "Typhoon": "cyclone", "Coastal Storm": "cyclone",
    "Severe Storm": "convective", "Tornado": "convective", "Straight-Line Winds": "convective",
    "Fire": "wildfire", "Flood": "flood",
}


def design(train, test, fields, cats=CATS):
    prep = ColumnTransformer([
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
        ("numeric", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True,
                         add_indicator=True), StandardScaler()), fields),
    ])
    return prep.fit_transform(train), prep.transform(test)


def specialist_amounts(train, test, fields, geography=False):
    """Re-estimate the historical architecture using admissible inputs for this fold."""
    a, b = design(train, test, fields)
    y = train.target.to_numpy()
    train_band = train.actual_band.to_numpy()
    medians = np.zeros((len(test), 6))
    point = medians.copy()
    support = []
    for band in LABELS:
        select = train_band == band
        observed = y[select]
        default = LIMITS[band]
        medians[:, band] = np.median(observed) if len(observed) else default
        point[:, band] = medians[:, band]
        support.append({"band": band, "training_cases": len(observed),
                        "training_groups": train.loc[select, "event_group"].nunique(),
                        "has_dedicated_fit": False, "feature_scope": "retrospective_physical" if geography else "metadata"})
        if band >= 3:
            continue
        positive = select & (y > 0)
        minimum = 12 if band == 0 else 15
        if positive.sum() < minimum:
            continue
        model = ExtraTreesRegressor(n_estimators=160, min_samples_leaf=3, random_state=SEED, n_jobs=1)
        model.fit(a[positive], np.log1p(y[positive]))
        estimate = np.expm1(model.predict(b))
        if band == 0 and np.any(select & (y == 0)):
            hurdle = LogisticRegression(C=.25, max_iter=1000)
            hurdle.fit(a[select], (y[select] > 0).astype(int))
            estimate *= hurdle.predict_proba(b)[:, 1] >= .5
        point[:, band] = np.clip(estimate, LIMITS[band], np.nextafter(LIMITS[band+1], 0))
        support[-1]["has_dedicated_fit"] = True
    return point, medians, support


def episode_maxima(data):
    return data.loc[data.groupby("event_group").target.idxmax()]


def hazard_family(s):
    return s.map(FAMILIES).fillna("other")


def rare_predictions(train, test, fields):
    unique = episode_maxima(train)
    tail = unique[unique.target >= TAIL]
    global_rate = (len(tail) + .5) / (len(unique) + 1)
    fallback_median = float(tail.target.median()) if len(tail) else TAIL
    rate_by_hazard, size_by_hazard = {}, {}
    for family, group in unique.groupby(hazard_family(unique.incidentType)):
        count = group.target.ge(TAIL).sum()
        rate_by_hazard[family] = (count + PRIOR_EPISODES*global_rate)/(len(group)+PRIOR_EPISODES)
        if count:
            family_median_log = np.median(np.log(group.loc[group.target.ge(TAIL), "target"]))
            weight = count / (count+PRIOR_EPISODES)
            size_by_hazard[family] = np.exp(weight*family_median_log+(1-weight)*np.log(fallback_median))
    family = hazard_family(test.incidentType)
    rate = family.map(rate_by_hazard).fillna(global_rate).to_numpy()
    size = family.map(size_by_hazard).fillna(fallback_median).to_numpy()
    if len(tail) >= PRIOR_EPISODES:
        x, q = design(unique, test, fields)
        model = LogisticRegression(C=.2, max_iter=1000)
        model.fit(x, unique.target.ge(TAIL))
        risk = model.predict_proba(q)[:, 1]
    else:
        risk = rate.copy()
    return {
        "prevalence_probability": np.repeat(global_rate, len(test)),
        "pooled_hazard_probability": rate, "regularized_tail_probability": risk,
        "pooled_tail_size": size, "global_tail_size": np.repeat(fallback_median, len(test)),
    }


def fit_gpd(values, u):
    excess = (values[values > u]-u)/1e6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        shape, _, scale = genpareto.fit(excess, floc=0)
    if not np.isfinite([shape, scale]).all() or scale <= 0:
        raise ValueError("Nonfinite GPD estimate")
    return shape, scale


def tail_audit(train, year):
    episodes = episode_maxima(train).target.to_numpy()
    tail = episodes[episodes >= TAIL]
    empirical = {}
    if len(tail) >= PRIOR_EPISODES:
        empirical = {level: np.quantile(tail, level, method="higher") for level in (.8, .9)}
    rng = np.random.default_rng(SEED+year)
    diagnostics, bounds = [], {}
    for u in THRESHOLDS:
        available = episodes[episodes > u]
        row = {"year": year, "threshold_usd": u, "training_groups": len(episodes),
               "exceedance_groups": len(available), "groups_above_50m": len(tail)}
        if len(available) < MIN_GPD_EPISODES:
            diagnostics.append({**row, "status": "insufficient_episode_support"})
            continue
        try:
            xi, scale = fit_gpd(available, u)
            boot = np.array([fit_gpd(rng.choice(available, size=len(available), replace=True), u)[0]
                             for _ in range(80)])
            ci = np.quantile(boot, [.05, .95])
            exceed_50 = genpareto.sf((TAIL-u)/1e6, xi, scale=scale)
            valid = bool(exceed_50 > 0 and ci[1] < 1 and ci[1]-ci[0] <= 1)
            summary = {**row, "status": "exploratory_interval_available" if valid else "unstable_shape",
                       "xi": xi, "scale_million_usd": scale, "xi_bootstrap_p05": ci[0], "xi_bootstrap_p95": ci[1],
                       "model_episode_probability_50m": exceed_50 * len(available)/len(episodes),
                       "multiplier_p90_given_50m": (u + 1e6*genpareto.isf(.1*exceed_50, xi, scale=scale))/TAIL if exceed_50 > 0 else np.nan}
            diagnostics.append(summary)
            if valid and u == THRESHOLDS[0]:
                bounds = {level: u + 1e6*genpareto.isf((1-level)*exceed_50, xi, scale=scale) for level in (.8, .9)}
        except (ValueError, FloatingPointError, RuntimeError) as exc:
            diagnostics.append({**row, "status": type(exc).__name__})
    return empirical, bounds, diagnostics


def evaluate_point(predictions):
    rows = []
    for variant, frame in predictions.groupby("variant"):
        for axis, subgroups in [
            ("obligations", [("all", frame), ("at_least_50m", frame[frame.target >= TAIL]),
                             ("below_50m", frame[frame.target < TAIL])]),
            ("hazard", list(frame.groupby("incidentType"))),
        ]:
            for subset, g in subgroups:
                error = g.amount_usd - g.target
                rows.append({"variant": variant, "axis": axis, "subset": subset, "n": len(g),
                             "groups": g.event_group.nunique(), "mae_usd": error.abs().mean(),
                             "median_ae_usd": error.abs().median(), "rmse_usd": np.sqrt(np.mean(error**2)),
                             "mean_signed_error_usd": error.mean(), "underpredicted_n": (error < 0).sum()})
    return pd.DataFrame(rows)


def risk_score(frame):
    rows = []
    for variant in ("prevalence_probability", "pooled_hazard_probability", "regularized_tail_probability",
                    "forward_router_sum", "physical_regularized_tail_probability"):
        if variant not in frame:
            continue
        p = frame[variant]
        y = frame.target.ge(TAIL)
        alert = p.ge(ALERT)
        rows.append({"variant": variant, "n": len(y), "positives": int(y.sum()),
                     "positive_groups": frame.loc[y, "event_group"].nunique(),
                     "average_precision": average_precision_score(y, p), "auroc": roc_auc_score(y, p),
                     "brier": brier_score_loss(y, p), "mean_probability": p.mean(),
                     "alerts": int(alert.sum()), "true_alerts": int((alert & y).sum()),
                     "alerts_below_50m": int((alert & ~y).sum()), "recall": (alert & y).sum()/y.sum(),
                     "precision": (alert & y).sum()/alert.sum() if alert.any() else np.nan})
    return pd.DataFrame(rows)


def interval_score(frame):
    rows = []
    for (variant, level), subset in frame.groupby(["variant", "level"]):
        for group, q in (("all", subset), ("at_least_50m", subset[subset.target.ge(TAIL)])):
            tail_cover = q.target.between(q.lower_usd, q.upper_usd)
            width = q.upper_usd-q.lower_usd
            rows.append({"variant": variant, "level": level, "subset": group, "n": len(q),
                         "covered": int(tail_cover.sum()), "coverage": tail_cover.mean(),
                         "mean_width_usd": width.mean(), "median_width_usd": width.median(),
                         "p90_width_usd": width.quantile(.9), "max_width_usd": width.max(),
                         "envelope_extensions": int(q.extended.sum())})
    return pd.DataFrame(rows)


def held_episode_influence(d):
    base = d.query("variant == 'nested_metadata'").set_index("disasterNumber")
    a = d[d.target.ge(TAIL)].copy()
    a["delta_ae"] = (a.amount_usd-a.target).abs() - a.disasterNumber.map((base.amount_usd-base.target).abs())
    result = []
    for variant, frame in a.groupby("variant"):
        for episode in sorted(frame.event_group.unique()):
            kept = frame[frame.event_group.ne(episode)]
            result.append({"variant": variant, "omitted_event_group": episode,
                           "remaining_cases": len(kept), "remaining_delta_mae_usd": kept.delta_ae.mean()})
    return pd.DataFrame(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--research", type=Path, default=Path("research"))
    p.add_argument("--physical", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    r = args.research / "results"
    d = read_table(r/"hazards/cohort_features.csv")
    physical = []
    if args.physical:
        features = pd.read_csv(args.physical)
        physical = [c for c in EXTRA if c in features and features[c].notna().any()]
        d = d.merge(features[["disasterNumber", *physical]], on="disasterNumber", how="left", validate="one_to_one")
    old = pd.read_csv(r/"forward/forward_predictions.csv").query("variant == 'nested_metadata'")
    old_intervals = pd.read_csv(r/"uncertainty/forward_prediction_intervals.csv").query(
        "variant == 'nested_metadata' and method == 'dollar'")
    point, risks, supports, gpd, intervals = [], [], [], [], []
    for year in YEARS:
        train, test = split(d, year)
        identity = test[["disasterNumber", "target", "event_group", "fyDeclared", "incidentType", "actual_band"]].reset_index(drop=True)
        locator = old.set_index("disasterNumber").loc[test.disasterNumber]
        order = np.arange(len(test))
        assert len(test) == len(locator) and np.allclose(test.target, locator.target)
        probs = locator[[f"p_band_{i}" for i in LABELS]].to_numpy()
        band, oracle_band = probs.argmax(axis=1), test.actual_band.to_numpy(int)
        base = locator.predicted_amount.to_numpy()
        expert, medians, support = specialist_amounts(train, test, META)
        supports.extend({"year": year, **s} for s in support)
        z = rare_predictions(train, test, META)
        risk = z["regularized_tail_probability"]
        risk_row = identity.assign(**z, forward_router_sum=probs[:, 3:].sum(axis=1))
        estimands = {
            "nested_metadata": base, "same_router_band_medians": medians[order, band],
            "forward_hurdle_and_band_specialists": expert[order, band],
            "oracle_forward_band_specialists": expert[order, oracle_band],
            "global_tail_oracle": np.where(test.target.ge(TAIL), z["global_tail_size"], base),
            "pooled_hazard_tail_oracle": np.where(test.target.ge(TAIL), z["pooled_tail_size"], base),
            "rare_10pct_review_override": np.where(risk >= ALERT, z["pooled_tail_size"], base),
            "rare_probability_weighted_amount": (1-risk)*np.minimum(base, TAIL) + risk*z["pooled_tail_size"],
        }
        if physical:
            e, _, support = specialist_amounts(train, test, META+physical, geography=True)
            supports.extend({"year": year, **s} for s in support)
            q = rare_predictions(train, test, META+physical)
            p_risk = q["regularized_tail_probability"]
            risk_row["physical_regularized_tail_probability"] = p_risk
            estimands["physical_hurdle_and_band_specialists"] = e[order, band]
            estimands["physical_rare_10pct_review_override"] = np.where(p_risk >= ALERT, q["pooled_tail_size"], base)
        point.extend(identity.assign(variant=k, amount_usd=v) for k, v in estimands.items())
        risks.append(risk_row)
        empirical, extremes, audit = tail_audit(train, year)
        gpd.extend(audit)
        for level, archived in old_intervals[old_intervals.fyDeclared.eq(year)].groupby("level"):
            q = archived.set_index("disasterNumber").loc[test.disasterNumber]
            lower, upper = q.lower_usd.to_numpy(), q.upper_usd.to_numpy()
            earlier = pd.concat([f for f in point if f.variant.iloc[0] == "forward_hurdle_and_band_specialists"])
            earlier = earlier[earlier.fyDeclared.lt(year) & ~earlier.event_group.isin(test.event_group)]
            assert len(earlier) >= 30
            delta = conformal_quantile((earlier.amount_usd-earlier.target).abs(), level)
            specialist = expert[order, band]
            for name, width_limit in (
                ("archived_dollar_residual", None), ("conditional_empirical_alert", empirical.get(level)),
                ("conditional_gpd_alert", extremes.get(level)),
            ):
                higher = upper if width_limit is None else np.where(risk >= ALERT, np.maximum(upper, width_limit), upper)
                intervals.append(identity.assign(variant=name, level=level, lower_usd=lower,
                                                 upper_usd=higher, extended=higher>upper,
                                                 tail_bound_available=width_limit is not None))
            intervals.append(identity.assign(variant="specialist_dollar_residual", level=level,
                lower_usd=np.maximum(specialist-delta, 0), upper_usd=specialist+delta,
                extended=False, tail_bound_available=False))
    all_point = pd.concat(point, ignore_index=True)
    all_risk = pd.concat(risks, ignore_index=True)
    all_intervals = pd.concat(intervals, ignore_index=True)
    assert all_point.groupby("variant").size().eq(376).all()
    assert all_risk.target.ge(TAIL).sum() == 13
    assert all_intervals.groupby(["variant", "level"]).size().eq(308).all()
    for name, values in {
        "tail_point_predictions": all_point, "tail_point_metrics": evaluate_point(all_point),
        "tail_risk_predictions": all_risk, "tail_risk_metrics": risk_score(all_risk),
        "tail_interval_predictions": all_intervals, "tail_interval_metrics": interval_score(all_intervals),
        "tail_gpd_support": pd.DataFrame(gpd), "forward_specialist_support": pd.DataFrame(supports),
        "tail_leave_one_group_out": held_episode_influence(all_point),
    }.items():
        values.to_csv(args.output/f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
