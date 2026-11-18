#!/usr/bin/env python3
"""Measure coverage and matched value added by archived precipitation and wind exposure."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from thesis_forward_models import ARMS, YEARS, fit_predict, read_table, score, split
from thesis_rare_event_followup import TAIL

RAIN = ["clim_max_1day_mm", "clim_max_county_total_mm"]
WIND = ["wind_county_overlap_km2", "wind_blockgroup_population_estimate"]
BASELINE = ARMS["verified_link_only"]


def summarize(d):
    p = d[[f"p_band_{i}" for i in range(6)]].to_numpy()
    result = score(d, d.predicted_amount.to_numpy(), p)
    tail = d[d.target.ge(TAIL)]
    result.update(tail_n=len(tail), tail_mae_usd=(tail.target-tail.predicted_amount).abs().mean(),
                  tail_routed_below_50m=int((tail.predicted_band < 3).sum()))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--research", type=Path, default=Path("research"))
    p.add_argument("--weather", type=Path, required=True)
    p.add_argument("--spatial", type=Path, required=True)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    d=read_table(args.research/"results/hazards/cohort_features.csv")
    rain=pd.read_csv(args.weather/"precipitation_features.csv").drop(columns="incidentType")
    wind=pd.read_csv(args.spatial/"spatial_features.csv")
    audit=pd.read_csv(args.spatial/"cyclone_link_and_radii_audit.csv")
    fire_audit=pd.read_csv(args.spatial/"wildfire_perimeter_coverage.csv")
    cov=d.merge(rain,on="disasterNumber",how="left",validate="one_to_one").merge(
        wind,on="disasterNumber",how="left",validate="one_to_one")
    cov=cov.merge(audit[["disasterNumber","source_event_id","status","max_nearby_wind_kt"]].rename(columns={"status":"wind_status"}),
                  on="disasterNumber",how="left",validate="one_to_one")
    cov=cov.merge(fire_audit[["disasterNumber","status","names_from_supplemental_documents"]].rename(columns={"status":"fire_status"}),
                  on="disasterNumber",how="left",validate="one_to_one")

    slices = (("nonbiological_912",cov),("test_FY2019_24",cov[cov.fyDeclared.ge(2019)]),
              ("all_tail_cases",cov[cov.target.ge(TAIL)]),
              ("test_tail_cases",cov[cov.fyDeclared.ge(2019)&cov.target.ge(TAIL)]))
    coverage = []
    for cohort, rows in slices:
        for hazard, g in rows.groupby("incidentType"):
            coverage.append({"cohort":cohort,"hazard":hazard,"cases":len(g),
                "old_county_population":g.designated_2020_population.notna().sum(),
                "old_NHC_best_track":g.cyclone_max_wind_kt.notna().sum(),
                "new_named_track_with_nearby_wind":g.max_nearby_wind_kt.notna().sum(),
                "new_wind_footprint_overlap":g.wind_status.eq("modeled_wind_overlap").sum(),
                "new_wind_footprint_population_proxy":g.wind_blockgroup_population_estimate.notna().sum(),
                "old_MTBS_single_named_fire":g.mtbs_burned_acres.notna().sum(),
                "new_has_mapped_fire_area":g.fire_mapped_union_acres.notna().sum(),
                "new_all_reviewed_fire_names_mapped":g.fire_status.eq("all_reviewed_names_mapped").sum(),
                "new_fire_housing_proxy":g.fire_blockgroup_housing_estimate.notna().sum(),
                "old_NOAA_county_report":g.noaa_event_count.notna().sum(),
                "old_NOAA_measured_gust":g.measured_gust_kt.notna().sum(),
                "old_NOAA_estimated_gust":g.estimated_gust_kt.notna().sum(),
                "old_NOAA_hail":g.hail_in.notna().sum(), "old_NOAA_tornado_EF":g.tornado_ef.notna().sum(),
                "new_county_rain":g.clim_max_1day_mm.notna().sum(),
                "new_all_declared_counties_rain":g.clim_all_declared_counties.eq(True).sum(),
                "both_NOAA_report_and_new_rain":(g.noaa_event_count.notna()&g.clim_max_1day_mm.notna()).sum(),
                "rain_without_NOAA_report":(g.noaa_event_count.isna()&g.clim_max_1day_mm.notna()).sum()})
    pd.DataFrame(coverage).to_csv(args.output/"hazard_feature_coverage_audit.csv",index=False)

    # This targeted historical case received additional bespoke evidence search.
    # Exclude fire perimeter/housing fields from model comparisons at every fold.
    # Their sourcing was not uniform across wildfires, particularly the large cases.
    cov.loc[cov.disasterNumber.eq(4715),WIND]=np.nan
    exposure_features=cov[["disasterNumber",*RAIN,*WIND]]
    exposure_features.to_csv(args.output/"physical_model_inputs.csv",index=False)

    modes={
        "previous_verified_retrospective":BASELINE,
        "add_county_rain":BASELINE+RAIN,
        "add_wind_area_and_population":BASELINE+WIND,
        "add_rain_and_wind_exposure":BASELINE+RAIN+WIND,
    }
    clusters=[]
    paired=[]
    for y in YEARS:
        train,test=split(cov,y)
        assert not set(train.source_event_id.dropna())&set(test.source_event_id.dropna())
        clusters.append({"year":y,"n_train":len(train),"n_test":len(test),
                         "newly_discovered_cyclone_source_overlap":0})
        for mode, columns in modes.items():
            prob,amount=fit_predict(train,test,columns,"extra_trees",6)
            paired.append(test[["disasterNumber","incidentType","fyDeclared","target","actual_band","event_group"]].assign(
                variant=mode,predicted_amount=amount,predicted_band=prob.argmax(axis=1),
                rainfall_available=test.clim_max_1day_mm.notna(),wind_population_available=test.wind_blockgroup_population_estimate.notna(),
                **{f"p_band_{i}":prob[:,i] for i in range(6)}))
    pred=pd.concat(paired,ignore_index=True)
    pred.to_csv(args.output/"physical_matched_predictions.csv",index=False)
    pd.DataFrame(clusters).to_csv(args.output/"physical_fold_audit.csv",index=False)
    compare=[]
    for mode,g in pred.groupby("variant"):
        for subset, f in {"all_matched":g,"rain_covered":g[g.rainfall_available],
                          "wind_population_covered":g[g.wind_population_available],
                          "severe_storm":g[g.incidentType.eq("Severe Storm")],
                          "hurricane":g[g.incidentType.eq("Hurricane")]}.items():
            if len(f):
                compare.append({"variant":mode,"subset":subset,**summarize(f)})
    pd.DataFrame(compare).to_csv(args.output/"physical_matched_metrics.csv",index=False)

    right=pred.query("variant == 'previous_verified_retrospective'").set_index("disasterNumber")
    differences=[]
    rng=np.random.default_rng(20261118)
    for variant,g in pred.groupby("variant"):
        g=g.copy()
        g["old_amount"]=g.disasterNumber.map(right.predicted_amount)
        g["old_band"]=g.disasterNumber.map(right.predicted_band)
        for subset,d in (("all_matched",g),("rain_covered",g[g.rainfall_available]),
                         ("wind_population_covered",g[g.wind_population_available]),("tail_13",g[g.target.ge(TAIL)])):
            if not len(d):
                continue
            d=d.assign(dollar=(d.target-d.predicted_amount).abs()-(d.target-d.old_amount).abs(),
                       correct=d.actual_band.eq(d.predicted_band).astype(int)-d.actual_band.eq(d.old_band).astype(int))
            groups=d.groupby("event_group").agg(count=("disasterNumber","size"),dollar=("dollar","sum"),correct=("correct","sum")).to_numpy()
            summary=groups[rng.integers(0,len(groups),size=(2500,len(groups)))].sum(axis=1)
            q=np.quantile(summary[:,1:]/summary[:,:1],[.025,.975],axis=0)
            differences.append({"variant":variant,"subset":subset,"n":len(d),"groups":len(groups),
                "mae_delta_usd":d.dollar.mean(),"mae_bootstrap_low":q[0,0],"mae_bootstrap_high":q[1,0],
                "accuracy_delta":d.correct.mean(),"accuracy_bootstrap_low":q[0,1],"accuracy_bootstrap_high":q[1,1]})
    pd.DataFrame(differences).to_csv(args.output/"physical_paired_group_bootstrap.csv",index=False)


if __name__ == "__main__":
    main()
