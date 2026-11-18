#!/usr/bin/env python3
"""Quantify controlled vintage and training-window changes on matched earlier outcomes."""
import argparse
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from thesis_forward_models import META, YEARS, fit_predict, read_table, score, split
from thesis_hazard_features import BANDS
from thesis_rare_event_followup import TAIL

COUNTS = ["mission_count", "agency_count", "esf_count", "ma_type_count", "priority_count"]
POST = ["missionAssignmentCount", "uniqueAgencyCount", "uniqueMaTypeCount", "uniquePriorityCount"]
ALLOWED_META = ["incident_month", "declaration_delay_days"]


def archive_csv(archive, entry):
    with ZipFile(archive) as zipped:
        return pd.read_csv(zipped.open(entry))


def summarize(variant, scope, sample, amount, probabilities, cutoff_row):
    values = sample.target.ge(TAIL)
    alt = amount[values]
    y = sample.loc[values,"target"]
    return {"variant": variant, "scope": scope, **cutoff_row, **score(sample, amount, probabilities),
            "tail_n": len(y), "tail_mae_usd": np.abs(alt-y).mean(),
            "tail_ap": average_precision_score(values, probabilities[:,3:].sum(axis=1)) if values.any() else np.nan}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("--research",type=Path,default=Path("research"))
    p.add_argument("--workbook",type=Path,default=Path("master_openfema_40plus.xlsx"))
    p.add_argument("--mission-archive",type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    root=args.research/"results"
    d=read_table(root/"hazards/cohort_features.csv")
    original=pd.read_excel(args.workbook,usecols=["disasterNumber",*POST])
    d=d.merge(original,on="disasterNumber",how="left",validate="one_to_one")
    mission=archive_csv(args.mission_archive,"mission_actions_t0_t72_all971.csv")
    names=archive_csv(args.research/"inputs/actions/nhc.zip","storm_matching_and_timestamp_audit.csv")
    nhc=archive_csv(args.research/"inputs/actions/nhc.zip","nhc_tropical_features_target_blind.csv")
    d=d.merge(mission,on="disasterNumber",how="left",validate="one_to_one").merge(nhc,on="disasterNumber",how="left",validate="one_to_one")
    scope_fields = {}
    for t in ("t0","t72"):
        age=pd.to_numeric(d[f"nhc_{t}_hours_before_cutoff"],errors="coerce")
        cols=[f"nhc_{t}_{q}" for q in ("wind_mph","pressure_mb")]
        d.loc[~age.between(0,72),cols]=np.nan
        scope_fields[t]=([f"ma_{t}_{c}" for c in COUNTS],cols)
    d["ever_mission"] = d.ma_total_action_rows.fillna(0).gt(0)
    d["original_release_certified"] = False
    counts=[]
    for by, frames in [("hazard",d.groupby("incidentType")),("all",[("all",d)]),("tail",[("at_least_50m",d[d.target.ge(TAIL)])])]:
        for label, q in frames:
            counts.append({"axis":by,"group":label,"cohort_count":len(q),
                           "event_groups":q.event_group.nunique(),
                           "any_final_missions":q.ever_mission.sum(),
                           "any_t0_action_proxy":q.ma_t0_mission_count.gt(0).sum(),
                           "any_t72_action_proxy":q.ma_t72_mission_count.gt(0).sum(),
                           "any_recent_nhc_t0":q.nhc_t0_wind_mph.notna().sum(),
                           "any_recent_nhc_t72":q.nhc_t72_wind_mph.notna().sum(),
                           "final_count_rows_after_t0": q.ma_refresh_after_t0.sum(),
                           "aliases_with_verified_original_fema_title":names.loc[names.disasterNumber.isin(q.disasterNumber),"fema_title_contemporaneously_verified"].eq(True).sum(),
                           "exact_release_certified":q.original_release_certified.sum()})
    pd.DataFrame(counts).to_csv(args.output/"decision_time_coverage.csv",index=False)
    summary, predictions, provenance = [],[],[]
    for fy in YEARS:
        earlier,test=split(d,fy)
        test_dates=min(test.start.min(),test.declaration.min())
        retrospective=d[d.fyDeclared.ne(fy)&~d.event_group.isin(test.event_group)].copy()
        stale=earlier[(earlier.declaration+pd.DateOffset(months=36)).lt(test_dates)].copy()
        all_branches={
            "fixed_metadata_purged_forward":(earlier,META),
            "fixed_metadata_purged_LFYO_descriptive":(retrospective,META),
            "fixed_metadata_lag36_proxy":(stale,META),
            "metadata_without_program_flags":(earlier,ALLOWED_META),
            "T0_action_proxy":(earlier,META+scope_fields["t0"][0]),
            "T72_action_proxy":(earlier,META+scope_fields["t72"][0]),
            "T72_action_and_recent_advisory_proxy":(earlier,META+scope_fields["t72"][0]+scope_fields["t72"][1]),
            "final_mission_counts_retrospective_only":(earlier,META+POST),
        }
        for name,(train,cols) in all_branches.items():
            prob,amount=fit_predict(train,test,cols,"extra_trees",6)
            provenance.append({"variant":name,"fy":fy,"train_cases":len(train),"train_groups":train.event_group.nunique(),
                               "future_fy_train_cases":train.fyDeclared.gt(fy).sum(),"train_tail_cases":train.target.ge(TAIL).sum(),
                               "train_tail_groups":train.loc[train.target.ge(TAIL),"event_group"].nunique(),
                               "earliest_held_event_date":test_dates,"max_train_fy":train.fyDeclared.max()})
            predictions.append(test[["disasterNumber","fyDeclared","incidentType","event_group","target","actual_band"]].assign(
                               variant=name,amount_usd=amount,**{f"p_{i}":prob[:,i] for i in range(6)}))
    q=pd.concat(predictions,ignore_index=True)
    q.to_csv(args.output/"decision_time_matched_predictions.csv",index=False)
    pd.DataFrame(provenance).to_csv(args.output/"decision_time_folds.csv",index=False)
    for name, a in q.groupby("variant"):
        assert len(a)==376
        summary.append(summarize(name,"same_forward_376",a,a.amount_usd.to_numpy(),a[[f"p_{n}" for n in range(6)]].to_numpy(),{}))
    frozen=pd.read_csv(root/"track_a/all_912_immutable_overlay.csv")
    matched_ids=set(q.disasterNumber)
    for label, f in [("all_historical_912",frozen),("same_forward_376",frozen[frozen.disasterNumber.isin(matched_ids)])]:
        f=f.copy()
        f["actual_band"]=f.actual_band.map(BANDS.index)
        p=np.eye(6)[f.router_pred.map(BANDS.index)]
        row=summarize("archived_track_A_descriptive",label,f,f.challenger_stack_prediction.to_numpy(),p,{})
        # Track A archive contains classes only. A hard-route pseudo-probability has no calibration interpretation.
        row["multiclass_brier"]=np.nan
        summary.append(row)
    pd.DataFrame(summary).to_csv(args.output/"decision_time_metrics.csv",index=False)

    # Inspect routing and amount components separately, preserving the original predictions.
    matched=q[q.variant.eq("fixed_metadata_purged_forward")][["disasterNumber","event_group"]]
    heavy=frozen.merge(matched,on="disasterNumber",how="inner",validate="one_to_one").query("target >= @TAIL")
    heavy.to_csv(args.output/"track_a_tail_matched_descriptive.csv",index=False)


if __name__=="__main__":
    main()
