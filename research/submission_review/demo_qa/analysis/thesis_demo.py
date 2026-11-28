#!/usr/bin/env python3
"""Show retrospective evidence and a separate simulated allocation briefing."""
import argparse
import csv
import json
import math
from pathlib import Path

# Display order matches the results generator. Verified against the pinned definitions in QA.
BANDS = ("0-100K", "100K-1M", "1M-50M", "50-200M", "200-500M", "500M+")
PHYSICAL = (
    "measured_gust_kt", "estimated_gust_kt", "hail_in", "tornado_ef", "tornado_length_mi",
    "tornado_width_yd", "cyclone_max_wind_kt", "cyclone_min_pressure_mb",
    "cyclone_max_34kt_radius_nm", "cyclone_landfall_records", "mtbs_burned_acres",
    "quake_magnitude", "quake_depth_km", "quake_shakemap_max_mmi",
    "daymet_point_max_24h_mm", "daymet_point_incident_total_mm", "usgs_discharge_anomaly",
)
SIM_NUMERIC = ("estimated_need_units", "simulated_need_units", "allocation_units", "vulnerability")
PROBABILITY = tuple(f"p_band_{i}" for i in range(len(BANDS)))


def rows(root, filename, fields):
    with (root / filename).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = sorted(set(fields).difference(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{filename}: missing field(s) {', '.join(missing)}")
        for n, row in enumerate(reader, start=2):
            if None in row or any(row[field] is None for field in fields):
                raise ValueError(f"{filename}:{n}: malformed row")
            yield row


def number(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"field {key} must be finite")
    return value


def single(records, description, required=False):
    if len(records) > 1:
        raise ValueError(f"{description} has duplicate rows")
    if not records and required:
        raise ValueError(f"{description} is not in archived results")
    return records[0] if records else None


def build(root, disaster_number, scenario_event):
    identifier = str(disaster_number)
    event_fields = ("disasterNumber", "fyDeclared", "incidentType", "state", "start", "end", "target", *PHYSICAL)
    event = single(
        [r for r in rows(root, "hazards/cohort_features.csv", event_fields) if r["disasterNumber"] == identifier],
        f"disaster {identifier}", required=True,
    )
    year = int(event["fyDeclared"])
    if not 2010 <= year <= 2024:
        raise ValueError("briefing supports only the historical FY2010–FY2024 results; later outcomes are embargoed")
    available = [field for field in PHYSICAL if event[field].strip()]
    result = {
        "historical_disaster": {
            "id": disaster_number, "hazard": event["incidentType"], "state": event["state"],
            "recorded_dates": {"start": event["start"], "end": event["end"] or None},
            "historical_fema_obligations_usd": number(event, "target"),
            "evidence_fields_available": available,
            "evidence_fields_missing": [field for field in PHYSICAL if field not in available],
            "evidence_provenance": "research/SOURCE_AND_FEATURE_MANIFEST.csv",
        },
        "retrospective_model_estimate": {"status": "outside evaluated forward cohort"},
        "simulated_humanitarian_allocation": {
            "status": "independent synthetic scenario, expressed in service units",
            "scenario_event": scenario_event,
            "scope": "not observed aid, a funding estimate, or an operational recommendation",
        },
    }
    predictions = "forward/forward_predictions.csv"
    pred_fields = ("variant", "disasterNumber", "predicted_amount", "predicted_band", *PROBABILITY)
    pred = single(
        [r for r in rows(root, predictions, pred_fields)
         if r["variant"] == "nested_metadata" and r["disasterNumber"] == identifier],
        f"nested_metadata prediction for disaster {identifier}",
    )
    if pred is not None:
        interval_records = rows(root, "uncertainty/forward_prediction_intervals.csv",
                                ("variant", "method", "disasterNumber", "level", "lower_usd", "upper_usd"))
        intervals = [r for r in interval_records
                     if r["variant"] == "nested_metadata" and r["method"] == "dollar" and r["disasterNumber"] == identifier]
        levels = [{"level": number(r, "level"), "lower_usd": number(r, "lower_usd"),
                   "upper_usd": number(r, "upper_usd")} for r in intervals]
        p = [number(pred, key) for key in PROBABILITY]
        band = int(pred["predicted_band"])
        if not 0 <= band < len(BANDS) or any(x < 0 or x > 1 for x in p) or not math.isclose(sum(p), 1, abs_tol=1e-8):
            raise ValueError(f"disaster {identifier}: invalid band or band probabilities")
        if len({r["level"] for r in levels}) != len(levels) or any(
                not (0 < r["level"] < 1 and 0 <= r["lower_usd"] <= r["upper_usd"]) for r in levels):
            raise ValueError(f"disaster {identifier}: invalid or duplicate interval")
        result["retrospective_model_estimate"] = {
            "historical_obligation_estimate_usd": number(pred, "predicted_amount"),
            "band": BANDS[band],
            "band_note": "classifier output; the separately archived regression estimate may be outside this band",
            "band_probabilities": dict(zip(BANDS, p)),
            "intervals": levels,
            "human_review": True,
            "availability_limit": "later-year retrospective test, exact original decision-time versions unverified",
        }
    simulation = [
        {"hazard": r["hazard"], **{key: number(r, key) for key in SIM_NUMERIC}, "policy": r["policy"]}
        for r in rows(root, "allocation/synthetic_allocation_demonstration.csv",
                      ("scenario_event", "hazard", "policy", *SIM_NUMERIC))
        if r["scenario_event"] == str(scenario_event)
    ]
    if not simulation:
        raise ValueError(f"scenario {scenario_event} is not in archived simulation results")
    result["simulated_humanitarian_allocation"]["policy_comparison"] = simulation
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("disaster_number", type=int)
    parser.add_argument("--results", type=Path, default=Path("research/results"))
    parser.add_argument("--scenario-event", type=int, default=1)
    args = parser.parse_args()
    try:
        briefing = json.dumps(build(args.results, args.disaster_number, args.scenario_event), indent=2, allow_nan=False)
    except (ValueError, OSError, csv.Error) as exc:
        parser.error(str(exc))
    print(briefing)


if __name__ == "__main__":
    main()
