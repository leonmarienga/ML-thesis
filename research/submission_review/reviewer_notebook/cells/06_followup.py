require(authenticated, "Evidence is not verified; run the bootstrap in a fresh kernel")
from datetime import datetime, timedelta

folds = rows("target_blind_folds.csv")
require([int(r["origin_fy"]) for r in folds] == list(range(2019, 2025)), "Unexpected origins")
require(sum(int(r["n_held"]) for r in folds) == 376, "Target-blind held support changed")
for r in folds:
    require(int(r["max_train_fy"]) < int(r["origin_fy"]) and
            r["max_train_end_or_start"] < r["min_held_start_or_declaration"] and
            int(r["held_groups_in_training"]) == int(r["train_held_declaration_id_intersection"]) == 0,
            "Archived fold chronology/group check failed")
    require(r["permutation_invariance"] == r["reassignment_invariance"] == "True" and
            min(int(r["permutation_held_labels_changed"]), int(r["reassignment_held_labels_changed"])) > 0,
            "Archived held-outcome stress check failed")
tb = json.loads(evidence["target_blind_summary.json"])
require(tb["status"] == "exploratory_development_only" and tb["fy_2025_accessed"] is False and
        tb["held_outcome_stress_origins_passed"] == len(folds) and
        tb["independent_replication_priority_gate_met"] is False, "Target-blind gate changed")
gain = tb["candidate"]["correct"]-tb["baseline"]["correct"]
changed = tb["all_coverage"]["changed_n"]
groups = tb["all_coverage"]["changed_group_n"]
require((gain,changed,groups) == (1,3,2), "Target-blind qualification changed")
claims["target_blind"] = {"scope":"archived noninterference check records; examined cohort, not replication",
                          "origins":len(folds), "net_correct":gain, "changed":changed,
                          "changed_groups":groups, "promotion_gate_met":False}
print("Target-blind stress records pass six earlier-vs-held folds; one net correction, two changed groups. Not promoted.")

weights = rows("weight_metrics.csv")
a,b = [one(weights, arm=arm, rank="0") for arm in ("real_only","weighted_20x")]
require(a["n"] == b["n"] == "135" and a["tail_n"] == b["tail_n"] == "11", "Unexpected signed-action support")
tails = rows("weight_bootstrap.csv")
r = one(tails, endpoint="signed_elapsed", arm="weighted_20x", metric="tail_mae")
delta = float(b["tail_mae"])-float(a["tail_mae"])
require(near(delta, float(r["change"]), 0.01) and float(r["lower_95"]) < 0 < float(r["upper_95"]),
        "20× tail comparison or confidence interval changed")
alerts = [int(x["alerts_false_10pct"]) for x in (a,b)]
require(alerts == [13,39], "20× false alert counts changed")
claims["weight_20x"] = {"scope":"exploratory matched elapsed signed-action cohort; not independent",
                        "n":135, "tail_n":11, "tail_mae_change_usd":delta,
                        "archived_ci95_usd":[float(r["lower_95"]),float(r["upper_95"])],
                        "false_alerts_baseline_then_weighted":alerts}
print(f"20× signed-action candidate: tail MAE difference ${delta:,.0f}; CI crosses zero; false alerts 13 to 39.")

gauge_archive = json.loads(evidence["gauge_stage_summary.json"])
gauge_pairs = rows("gauge_stage_pairs.csv")
crossing = datetime.fromisoformat(gauge_archive["first_actual_15m_crossing_12ft_utc"])
require(gauge_archive["episodes"] == 1 and len(gauge_pairs) == 230 and crossing.isoformat() == "2022-12-31T09:00:00+00:00",
        "Gauge cohort changed")
gauge = []
for a in gauge_archive["cases"]:
    issue = datetime.fromisoformat(a["bulletin_wmo_issue_utc"])
    pairs = [r for r in gauge_pairs if r["wmo_issue_utc"] == issue.isoformat()]
    valid = [datetime.fromisoformat(r["valid_utc"]) for r in pairs]
    require(len(valid) == len(set(valid)) == 115 and valid == sorted(valid) and
            all(t > issue and (t-issue).total_seconds() == int(r["lead_minutes"])*60 for t,r in zip(valid,pairs)) and
            all(t-s == timedelta(hours=1) for s,t in zip(valid,valid[1:])),
            "Gauge issue/time support failed")
    fc = [float(r["forecast_stage_ft"]) for r in pairs]
    obs = [float(r["approved_gage_height_ft"]) for r in pairs]
    first = next(t for t,h in zip(valid, fc) if h >= 12)
    err = mean(abs(x-y) for x,y in zip(fc,obs))
    require(near(err,a["mae_ft"]) and max(fc) == a["forecast_peak_ft"] and max(obs) == a["observed_peak_at_forecast_hours_ft"] and
            first.isoformat() == a["first_forecast_hour_at_or_above_12_ft_utc"], "Gauge summary did not reproduce")
    precedes = issue < crossing
    gauge.append({"issue_utc":issue.isoformat(), "issue_preceded_observed_crossing":precedes,
                  "forecast_delay_hours_if_pre_crossing":(first-crossing).total_seconds()/3600 if precedes else None,
                  "future_hours":len(valid), "mae_ft":err,"forecast_peak_ft":max(fc),"observed_peak_ft":max(obs)})
require(len(gauge) == 2 and gauge[0]["forecast_delay_hours_if_pre_crossing"] == 14 and
        gauge[0]["forecast_peak_ft"] == 13 and gauge[0]["observed_peak_ft"] == 16.6 and
        gauge[1]["issue_preceded_observed_crossing"] is False, "Gauge crossing scope changed")
claims["gauge_stage"] = {"scope":"two issues, one historical event; no independent skill or obligation inference",
                         "episodes":1,"archived_actual_crossing_utc":crossing.isoformat(),"issues":gauge}
print("Gauge record: 115 future hours per issue, one event. Early crossing forecast 14 h late (peak 13.0 vs 16.6 ft); late issue was after crossing.")
