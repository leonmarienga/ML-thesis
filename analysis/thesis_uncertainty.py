#!/usr/bin/env python3
"""Backtest residual prediction intervals with earlier forward predictions only."""
import argparse
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

VARIANTS = ("ablation_metadata", "ablation_retrospective_metadata", "nested_verified_link_only", "nested_metadata")
LEVELS = (.8, .9)


def conformal_quantile(scores, level):
    """Order statistic with finite-sample correction."""
    rank = ceil((len(scores) + 1) * level)
    return np.partition(scores, rank - 1)[rank - 1] if rank <= len(scores) else np.inf


def intervals(predictions):
    rows = []
    for variant in VARIANTS:
        d = predictions[predictions.variant.eq(variant)]
        for year, test in d.groupby("fyDeclared"):
            # Repeated earlier episodes are excluded conservatively.
            calibration = d[d.fyDeclared.lt(year) & ~d.event_group.isin(test.event_group)]
            if len(calibration) < 30:
                continue
            for method in ("dollar", "log1p"):
                scale, inverse = (np.log1p, np.expm1) if method == "log1p" else (lambda x:x, lambda x:x)
                errors = (scale(calibration.target)-scale(calibration.predicted_amount)).abs().to_numpy()
                center = scale(test.predicted_amount)
                for level in LEVELS:
                    q = conformal_quantile(errors, level)
                    low, high = inverse((center-q).clip(lower=0)), inverse(center+q)
                    result = test[["disasterNumber", "fyDeclared", "incidentType", "actual_band",
                                   "predicted_band", "target", "predicted_amount", "top_band_probability"]].copy()
                    result = result.assign(variant=variant, method=method, level=level, calibration_n=len(errors),
                                           last_calibration_year=calibration.fyDeclared.max(), score_quantile=q,
                                           lower_usd=low, upper_usd=high)
                    result["width_usd"] = result.upper_usd - result.lower_usd
                    result["covered"] = result.target.between(result.lower_usd, result.upper_usd)
                    result["review"] = (result.top_band_probability < .6) | (result.predicted_band >= 3) | (result.width_usd >= 50e6)
                    rows.append(result)
    return pd.concat(rows, ignore_index=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("predictions", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(args.predictions)
    results = intervals(predictions)
    assert (results.fyDeclared > results.last_calibration_year).all()
    assert np.isfinite(results[["lower_usd", "upper_usd"]]).all().all()
    results.to_csv(args.output/"forward_prediction_intervals.csv", index=False)
    groups = ["variant", "method", "level"]
    summaries = []
    for axis in ("all", "fyDeclared", "incidentType", "actual_band"):
        d = results.assign(all="all")
        metric = d.groupby(groups + [axis]).agg(n=("covered", "size"), coverage=("covered", "mean"),
              median_width_usd=("width_usd", "median"), mean_width_usd=("width_usd", "mean"), review_rate=("review", "mean")).reset_index()
        summaries.append(metric.rename(columns={axis: "group"}).assign(axis=axis))
    pd.concat(summaries).to_csv(args.output/"interval_metrics.csv", index=False)
    pcols = [f"p_band_{i}" for i in range(6)]
    predictions["probability_sum"] = predictions[pcols].sum(axis=1)
    assert np.allclose(predictions.probability_sum, 1)
    predictions["correct"] = predictions.actual_band.eq(predictions.predicted_band)
    predictions["confidence_bin"] = pd.cut(predictions.top_band_probability, [0, .4, .6, .8, 1.01])
    predictions.groupby(["variant", "confidence_bin"], observed=True).agg(
        n=("correct", "size"), mean_confidence=("top_band_probability", "mean"), accuracy=("correct", "mean")
    ).to_csv(args.output/"router_reliability.csv")


if __name__ == "__main__":
    main()
