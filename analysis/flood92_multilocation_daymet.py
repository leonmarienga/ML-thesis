#!/usr/bin/env python3
"""Phase 23: multi-location NASA Daymet rainfall profiles for 92 FEMA Flood cases.

Cohort: all 92 2019-2024 Flood declarations; 87 have NOAA event-linked
geographic points. Per disaster select at most THREE *actual NOAA report
coordinates*, independent of FEMA funding or gauge availability:
 (1) the previously used report point nearest median coordinate (central);
 (2) farthest candidate at least 10 km from selected points;
 (3) maximin point at least 10 km from both selected points.
Never synthesize a geographic point. Repeat at most one actual point per case.

Retain exact prior Daymet 21-day incident cap, >=3 observed event days and >=80%
completeness, up to ten prior calendar-year seasonal 3-day comparator.
Daymet = station-informed gridded precipitation, not GPM IMERG.

The maximum across 3 points is a *sample maximum*, NOT rainfall integrated
across a FEMA footprint. No spatial polygon or exposure weights are used.
Retrospective source versions are not verified available at T0/T+72.
No target funding or accepted ML predictions are used anywhere.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

import flood_87_geospatial_gauge_screen as geo
import flood87_daymet_independent_rainfall as original
import flood24_daymet_precipitation_enrichment as daymet

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/"audit_inputs"/"daymet_multilocation"
OUT=ROOT/"audit_outputs"/"daymet_multilocation"
OUT.mkdir(parents=True,exist_ok=True)
MAX_POINTS=3
MIN_SEPARATION_KM=10.0
MAX_WORKERS=6
METRICS=["event_precip_total_mm","event_max_1day_mm",
         "event_max_3day_mm","event_max_7day_mm",
         "antecedent_7day_total_mm",
         "event_peak_3day_prior_season_percentile"]

def locations(master,audit,county):
    center,valid=original.representative_points(master,audit,county)
    assert len(center)==87 and center.disasterNumber.nunique()==87
    records=[]
    for _,mid in center.iterrows():
        disaster=int(mid.disasterNumber)
        subset=valid[valid.disasterNumber==disaster].copy()
        subset["lat"]=pd.to_numeric(subset.lat,errors="raise")
        subset["lon"]=pd.to_numeric(subset.lon,errors="raise")
        # Source de-duplication: reports at same rounded coordinates are one
        # sampling site; unique NOAA EVENT_ID still included in audit.
        subset["lat4"]=subset.lat.round(4)
        subset["lon4"]=subset.lon.round(4)
        subset=subset.sort_values(["EVENT_ID","fips"]).drop_duplicates(
            ["lat4","lon4"]).reset_index(drop=True)
        lat0,lon0=float(mid.noaa_point_lat),float(mid.noaa_point_lon)
        central=geo.distance_km(lat0,lon0,subset.lat.to_numpy(float),
                                 subset.lon.to_numpy(float))
        chosen=[int(np.argmin(central))]
        # Preserve the original center point as the first coordinate.
        assert float(central[chosen[0]])<.05,(disaster,central[chosen[0]])
        while len(chosen)<MAX_POINTS:
            unused=[i for i in range(len(subset)) if i not in chosen]
            if not unused:break
            distances=np.array([
                 min(float(geo.distance_km(
                         float(subset.iloc[i].lat),float(subset.iloc[i].lon),
                         np.array([float(subset.iloc[j].lat)]),
                         np.array([float(subset.iloc[j].lon)]))[0])
                     for j in chosen)
                 for i in unused])
            keep=distances>=MIN_SEPARATION_KM
            if not keep.any():break
            eligible=[(unused[i],float(distances[i])) for i in range(len(unused)) if keep[i]]
            # Stable maximin distance; ties resolved by deterministic event ID.
            idx,score=sorted(eligible,key=lambda x:(
                       -x[1],int(subset.iloc[x[0]].EVENT_ID),
                       float(subset.iloc[x[0]].lat),float(subset.iloc[x[0]].lon)))[0]
            chosen.append(idx)
        for rank,i in enumerate(chosen,1):
            row=subset.iloc[i]
            records.append({
             "disasterNumber":disaster,"fyDeclared":int(master.loc[
                  master.disasterNumber==disaster,"fyDeclared"].iloc[0]),
             "point_rank":rank,"point_type":"central" if rank==1 else "spatial_diversity",
             "noaa_event_id":int(row.EVENT_ID),"county_fips":str(row.fips).zfill(5),
             "noaa_point_lat":float(row.lat),"noaa_point_lon":float(row.lon),
             "distinct_noaa_coordinate_count":len(subset),
             "original_event_linked_report_count":int(mid.noaa_report_location_count),
             "original_noaa_location_spread_p90_km":float(mid.location_spread_p90_km),
             "min_distance_to_prior_selected_km":0.0 if rank==1 else float(
                 min(geo.distance_km(float(row.lat),float(row.lon),
                         np.array([float(subset.iloc[j].lat)]),
                         np.array([float(subset.iloc[j].lon)]))[0]
                     for j in chosen[:rank-1])),
             "selection_rule":"Rank1 original NOAA report closest to median NOAA locations; ranks2/3 maximin >=10km actual NOAA report points, funding and USGS gauge independent"})
    points=pd.DataFrame(records)
    assert points.disasterNumber.nunique()==87
    assert (points.groupby("disasterNumber").size()<=MAX_POINTS).all()
    assert (points.groupby("disasterNumber").point_rank.min()==1).all()
    assert len(points[points.point_rank==1])==87
    assert not points.duplicated(["disasterNumber","noaa_point_lat","noaa_point_lon"]).any()
    return points

def one_point(fema,point):
    dummy=fema.copy()
    dummy["normalized_discharge_available"]=False
    proxy=point.copy()
    proxy["nearest_noaa_EVENT_ID"]=int(point.noaa_event_id)
    proxy["candidate_station_id"]="NOT_APPLICABLE"
    proxy["station_to_nearest_noaa_report_km"]=float("nan")
    result=daymet.evaluate_case(dummy,proxy)
    result.pop("original_flow_candidate",None)
    result.pop("reference_station_id",None)
    result.pop("noaa_point_to_station_km",None)
    return result

def as_float_values(values):
    return pd.to_numeric(values,errors="coerce").dropna().to_numpy(float)

def main():
    master=pd.read_excel(ROOT/"master_openfema_40plus.xlsx")
    assert len(master)==971
    flood=master[(master.incidentType=="Flood") &
                 master.fyDeclared.between(2019,2024)].copy()
    assert len(flood)==92
    audit=pd.read_csv(INPUT/"noaa"/"external_match_audit.csv")
    county=pd.read_csv(INPUT/"county"/"event_county_census_matches.csv",
                       dtype={"fips":str})
    single=pd.read_csv(INPUT/"single"/"all_92_flood_daymet_rainfall.csv")
    assert len(single)==92 and single.disasterNumber.nunique()==92
    points=locations(flood,audit,county)
    print("SELECTED_POINTS",len(points),"DIVERSITY",points.groupby(
         "disasterNumber").size().value_counts().to_dict(),flush=True)
    point_records=[]
    merged=points.merge(flood[["disasterNumber","incidentBeginDate","incidentEndDate"]],
                        on="disasterNumber",validate="many_to_one")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures={executor.submit(one_point,row,row):row for _,row in merged.iterrows()}
        for fut in as_completed(futures):
            point=futures[fut]
            try:
                calc=fut.result()
            except Exception as ex:
                calc={"disasterNumber":int(point.disasterNumber),
                      "rainfall_available":False,
                      "rainfall_status":"unhandled_exception:"+str(ex)[:120]}
            record=point[["disasterNumber","fyDeclared","point_rank","point_type",
                 "noaa_event_id","noaa_point_lat","noaa_point_lon",
                 "county_fips","original_event_linked_report_count",
                 "original_noaa_location_spread_p90_km",
                 "min_distance_to_prior_selected_km"]].to_dict()
            # Explicitly prefer original selected-point provenance values.
            calc={k:v for k,v in calc.items() if k not in record}
            record.update(calc)
            point_records.append(record)
            print("DAYMET_POINT",point.disasterNumber,point.point_rank,
                   record.get("rainfall_status"),flush=True)
    rows=pd.DataFrame(point_records)
    assert len(rows)==len(points) and not rows.duplicated(
        ["disasterNumber","point_rank"]).any()
    rows["rainfall_available"]=rows.rainfall_available.fillna(False).astype(bool)
    rows=rows.sort_values(["disasterNumber","point_rank"])
    rows.to_csv(OUT/"all_noaa_point_daymet_measurements.csv",index=False)
    # Paired audit: point 1 must match EXACT earlier gauge-independent sample
    # under original feature rules. Fail loudly rather than report spurious gains.
    center=rows[rows.point_rank==1][["disasterNumber","rainfall_available",
                 "event_precip_total_mm","event_max_3day_mm"]]
    prior=single[["disasterNumber","rainfall_available",
                 "event_precip_total_mm","event_max_3day_mm"]]
    joined=center.merge(prior,on="disasterNumber",suffixes=("_new","_old"),
                 validate="one_to_one")
    assert len(joined)==87
    assert joined.rainfall_available_new.eq(joined.rainfall_available_old).all(),(
           "Point-1 availability shifted from prior study")
    for col in ["event_precip_total_mm","event_max_3day_mm"]:
        same=joined[col+"_new"].notna()&joined[col+"_old"].notna()
        assert np.allclose(joined.loc[same,col+"_new"],
                           joined.loc[same,col+"_old"],rtol=0,atol=.011),(
                        "Point-1 Daymet values not reproduced",col)
    records=[]
    for _,f in flood.iterrows():
        disaster=int(f.disasterNumber)
        group=rows[rows.disasterNumber==disaster]
        valid=group[group.rainfall_available]
        record={"disasterNumber":disaster,"fyDeclared":int(f.fyDeclared),
          "n_requested_locations":len(group),
          "n_usable_locations":len(valid),
          "any_rainfall":len(valid)>=1,
          "at_least_two_locations":len(valid)>=2,
          "at_least_three_locations":len(valid)>=3,
          "central_available":bool(group.iloc[0].rainfall_available) if len(group) else False,
          "source_is_nasa_daymet_not_imerg":True,
          "flood_footprint_verified":False,
          "asof_available_verified":False}
        if len(group):
            record["selected_coordinate_spread_p90_km"]=float(
                group.iloc[0].original_noaa_location_spread_p90_km)
            record["available_point_ranks"]="|".join(
                 map(str,valid.point_rank.to_numpy(int)))
        for metric in METRICS:
            arr=as_float_values(valid[metric]) if metric in valid else np.array([])
            record["n_"+metric]=len(arr)
            if len(arr):
                record["multi_"+metric+"_median"]=float(np.median(arr))
                record["multi_"+metric+"_max"]=float(np.max(arr))
                record["multi_"+metric+"_min"]=float(np.min(arr))
                record["multi_"+metric+"_range"]=float(np.max(arr)-np.min(arr)) if len(arr)>=2 else np.nan
                record["multi_"+metric+"_cv"]=float(np.std(arr)/np.mean(arr)) if len(arr)>=2 and np.mean(arr)>0 else np.nan
        records.append(record)
    cases=pd.DataFrame(records).sort_values("disasterNumber")
    assert len(cases)==92 and cases.disasterNumber.nunique()==92
    cases.to_csv(OUT/"all_92_multilocation_rainfall_profiles.csv",index=False)
    byyear=cases.groupby("fyDeclared").agg(
       total=("disasterNumber","count"),any_rainfall=("any_rainfall","sum"),
       two_or_more=("at_least_two_locations","sum"),
       three_or_more=("at_least_three_locations","sum")).reset_index()
    byyear.to_csv(OUT/"coverage_by_year.csv",index=False)
    valid_at_least_two=cases[cases.at_least_two_locations]
    valid_any=cases[cases.any_rainfall]
    single_count=int(single.rainfall_available.sum())
    report={
     "original_fema_flood_cohort":92,
     "noaa_geolocated_floods":87,
     "sampled_report_points":len(rows),
     "floods_with_one_selected_point":int((cases.n_requested_locations==1).sum()),
     "floods_with_two_selected_points":int((cases.n_requested_locations==2).sum()),
     "floods_with_three_selected_points":int((cases.n_requested_locations==3).sum()),
     "original_single_point_rainfall_available":single_count,
     "multilocation_any_rainfall_available":int(cases.any_rainfall.sum()),
     "rainfall_coverage_gain_in_disasters":int(cases.any_rainfall.sum())-single_count,
     "with_at_least_two_usable_locations":int(cases.at_least_two_locations.sum()),
     "with_at_least_three_usable_locations":int(cases.at_least_three_locations.sum()),
     "no_noaa_geolocation":int((cases.n_requested_locations==0).sum()),
     "point_level_status_counts":{str(k):int(v) for k,v in rows.rainfall_status.value_counts().items()},
     "at_least_two_location_event_total_range_median_mm":
         float(valid_at_least_two.multi_event_precip_total_mm_range.median()) if len(valid_at_least_two) else None,
     "at_least_two_location_event_total_range_p90_mm":
         float(valid_at_least_two.multi_event_precip_total_mm_range.quantile(.9)) if len(valid_at_least_two) else None,
     "any_rainfall_multi_event_total_median_mm":
         float(valid_any.multi_event_precip_total_mm_median.median()) if len(valid_any) else None,
     "by_year":byyear.to_dict("records"),
     "original_single_point_results_reproduced":True,
     "selected_point_rule":"NOAA medoid-based existing point, then up to 2 maximin points >=10km apart, always actual reported points; no target or gauge used",
     "rainfall_product":"NASA ORNL DAAC Daymet single-pixel, station-informed gridded daily precipitation, not IMERG",
     "funding_models_modified":False,
     "model_accuracy_compared":False,
     "caveats":[
       "Selected NOAA report coordinates are not flood inundation polygons or exposure-weighted samples.",
       "Spreading points to geographic extremes may increase observed maxima without implying event severity increased; max is sample maximum.",
       "Within-disaster rainfall variation is only evaluated for >=2 measured locations; one-location ranges are NaN not zero.",
       "Retrospective Daymet and finalized NOAA data are not authenticated decision-time versions.",
       "Truncated <=21-day incident window and >=3 day minimum exclude short events; eligibility unchanged from earlier study.",
       "NOAA matching is retrospective and some geographic reports may belong to different reaches of a large disaster.",
       "This is not a strictly held-out funding model ablation; original accepted router/specialists unchanged."
     ]}
    (OUT/"summary.json").write_text(json.dumps(report,indent=2))
    print("BEGIN_MULTILOC_DAYMET")
    print(json.dumps(report,indent=2))
    print("END_MULTILOC_DAYMET")
if __name__=="__main__":main()
