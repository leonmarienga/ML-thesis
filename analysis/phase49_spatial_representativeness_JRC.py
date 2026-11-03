#!/usr/bin/env python3
"""Phase49: Outcome-blind spatial expansion of Phase48 FEMA county samples.

37 2023-24-incident FEMA floods from FY2023/24; 36 with strict county designations. Three geographically separated fixed windows per chosen county when feasible.
Deterministic county selection: smallest FIPS and farthest other centroid, max 2.
One 4km x 4km physical observation window at each selected county interior
point, clipped to the county, independent of prior NOAA/radar candidate points
and any FEMA dollar amount.

Independent Landsat JRC Global Surface Water v1.5 monthly C2 records 2023-24
plus prior-year same-calendar-month control and following month. The 0 class
means UNOBSERVED; it must NEVER be counted as dry. Pairwise changes only where
both temporal scenes have observable water/dry classifications (1/2).
Results are locally sampled surface water change, NOT total inundation.
No funding models are read or trained. Track A remains unchanged.
"""
from __future__ import annotations
from pathlib import Path
import io,os,json,math,hashlib,requests
import numpy as np,pandas as pd,geopandas as gpd,rasterio
from shapely.geometry import box,mapping,Point
from shapely.ops import transform as geo_transform
from rasterio.windows import from_bounds,Window
from rasterio.features import geometry_mask
from pyproj import Transformer,Geod

ROOT=Path(__file__).resolve().parents[1]
IN=ROOT/"audit_inputs"/"phase49"
OUT=ROOT/"audit_outputs"/"phase49";OUT.mkdir(parents=True,exist_ok=True)
SOURCE="https://data.source.coop/epoch/jrc-global-surface-water-monthly/v1_5"
COUNTY_URL="https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip"
PILOT_IDS={4708,4709,4711,4717,4718,4719,4758,4781,4787,4789,4793,4797}
MAX_COUNTIES=2; RADIUS_METERS=2000
GEOD=Geod(ellps="WGS84")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN","EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_TIMEOUT","55")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY","2")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS",".tif,.tiff")

census_response=requests.get(COUNTY_URL,timeout=(30,130),
headers={"User-Agent":"academic disaster hazard-enrichment systematic geospatial audit/1.0"})
census_response.raise_for_status()
census=OUT/"cb_2023_us_county_500k.zip"; census.write_bytes(census_response.content)
source_hash=hashlib.sha256(census_response.content).hexdigest()
COUNTIES=gpd.read_file("zip://"+str(census))[["GEOID","NAME","geometry"]].to_crs("EPSG:4326")
COUNTIES["county_fips"]=pd.to_numeric(COUNTIES.GEOID)
SHAPES={int(c.county_fips):c.geometry for _,c in COUNTIES.iterrows()}
assert len(SHAPES)>3000
census.unlink()

FEMA=pd.read_csv(IN/"phase34"/"fema_official_designated_areas.csv")
FEMA["county_fips"]=pd.to_numeric(FEMA.county_fips,errors="coerce")
FEMA["incident_year"]=pd.to_datetime(FEMA.incident_start).dt.year
COHORT=FEMA[FEMA.fyDeclared.isin([2023,2024])].copy()
assert COHORT.disasterNumber.nunique()==41
eligible=COHORT[COHORT.incident_year.isin([2023,2024])].copy()
assert eligible.disasterNumber.nunique()==37
strict=eligible[(eligible.is_strict_county==True)&eligible.county_fips.notna()].copy()
assert strict.disasterNumber.nunique()==36

def haversine(a,b):
    _,_,d=GEOD.inv(float(a.x),float(a.y),float(b.x),float(b.y))
    return float(d/1000)

roster=[];excluded=[]
for dn,d in COHORT.groupby("disasterNumber"):
    if int(d.incident_year.iloc[0]) not in [2023,2024]:
        excluded.append({"disasterNumber":int(dn),"reason":"incident_before_2023",
                         "incident_start":str(d.incident_start.iloc[0])})
        continue
    g=strict[strict.disasterNumber==dn].drop_duplicates("county_fips")
    assigned=sorted(set(g.county_fips.astype(int)))
    available=[x for x in assigned if x in SHAPES]
    if not assigned:
        excluded.append({"disasterNumber":int(dn),"reason":"no_strict_county_designation",
                         "incident_start":str(d.incident_start.iloc[0])});continue
    if not available:
        excluded.append({"disasterNumber":int(dn),"reason":"census_county_polygon_missing",
                         "incident_start":str(d.incident_start.iloc[0])});continue
    chosen=[available[0]]
    if len(available)>1:
        pivot=SHAPES[chosen[0]].representative_point()
        second=max(available[1:],key=lambda x:(haversine(pivot,SHAPES[x].representative_point()),-x))
        chosen.append(second)
    start=pd.to_datetime(d.incident_start.iloc[0],utc=True)
    curr=start.strftime("%Y-%m")
    previous=(start-pd.DateOffset(years=1)).strftime("%Y-%m")
    following=(start+pd.DateOffset(months=1)).strftime("%Y-%m")
    for j,fips in enumerate(chosen):
        county=SHAPES[fips]
        original=county.representative_point()
        # Site selection uses ONLY pre-existing county geometry; never water/funding.
        local=f"+proj=aeqd +lat_0={original.y} +lon_0={original.x} +datum=WGS84 +units=m +no_defs"
        forward=Transformer.from_crs("EPSG:4326",local,always_xy=True).transform
        back=Transformer.from_crs(local,"EPSG:4326",always_xy=True).transform
        county_xy=geo_transform(forward,county)
        ox,oy=forward(original.x,original.y)
        selected=[(ox,oy,"phase49_original_anchor")]
        xmin,ymin,xmax,ymax=county_xy.bounds
        # Deterministic 11x11 county-interior grid; require >=6.5km spacing
        # (larger than 4km square diagonal) and >=50% window county coverage.
        candidates=[]
        for xx in np.linspace(xmin,xmax,11):
            for yy in np.linspace(ymin,ymax,11):
                if not county_xy.covers(Point(xx,yy)):continue
                overlap_area=county_xy.intersection(box(xx-2000,yy-2000,xx+2000,yy+2000)).area
                if overlap_area < 8_000_000:continue
                candidates.append((float(xx),float(yy),float(overlap_area/16_000_000)))
        for _slot in (2,3):
            eligible=[]
            for xx,yy,coverage in candidates:
                min_distance=min(math.hypot(xx-sx,yy-sy) for sx,sy,_ in selected)
                if min_distance<6500:continue
                eligible.append((-min_distance,-coverage,xx,yy))
            if not eligible:break
            _,_,xx,yy=min(eligible)
            selected.append((xx,yy,"maximin_county_interior_grid"))
        for site_index,(sx,sy,site_method) in enumerate(selected,1):
            anchor=geo_transform(back,Point(sx,sy))
            square=geo_transform(back,box(sx-RADIUS_METERS,sy-RADIUS_METERS,
                                          sx+RADIUS_METERS,sy+RADIUS_METERS))
            clipped=square.intersection(county)
            if clipped.is_empty:raise RuntimeError(f"Empty county window {dn}/{fips}/{site_index}")
            area_m2=abs(GEOD.geometry_area_perimeter(clipped)[0])
            roster.append({"disasterNumber":int(dn),"fyDeclared":int(d.fyDeclared.iloc[0]),
              "incident_start":str(start.date()),"incident_year":int(start.year),
              "state":str(d.state.iloc[0]),"county_fips":int(fips),
              "county_label":str(COUNTIES.loc[COUNTIES.county_fips==fips,"NAME"].iloc[0]),
              "county_n_official":len(assigned),"county_n_polygon_available":len(available),
              "county_choice_rule":"min_FIPS_then_farthest_other_centroid",
              "choice_order":j+1,"site_index":site_index,"site_method":site_method,
              "original_optical_pilot_disaster":int(dn) in PILOT_IDS,
              "anchor_lon":float(anchor.x),"anchor_lat":float(anchor.y),
              "window_area_km2":float(area_m2/1e6),
              "event_month":curr,"following_month":following,"prior_year_same_month":previous,
              "window_geometry":clipped})

assert len(set(r["disasterNumber"] for r in roster))==36
assert sum(r["site_index"]==1 for r in roster)==63
assert len(roster)>=63
# Hold original Phase48 anchor positions/dates fixed exactly; do not reselect.
old=pd.read_csv(IN/"phase48"/"phase48_predeclared_36_disaster_county_sampling_roster.csv")
new_anchor=pd.DataFrame([{k:v for k,v in r.items() if k!="window_geometry"} for r in roster if r["site_index"]==1])
anchor_check=old.merge(new_anchor,on=["disasterNumber","county_fips"],validate="one_to_one",suffixes=("_old","_new"))
assert len(anchor_check)==63
assert float((anchor_check.anchor_lon_old-anchor_check.anchor_lon_new).abs().max())<1e-9
assert float((anchor_check.anchor_lat_old-anchor_check.anchor_lat_new).abs().max())<1e-9
assert all(anchor_check.event_month_old==anchor_check.event_month_new)
assert all(r["window_area_km2"]>0 for r in roster)
ra=pd.DataFrame([{k:v for k,v in r.items() if k!="window_geometry"} for r in roster])
ra.to_csv(OUT/"phase49_predeclared_36_disaster_county_sampling_roster.csv",index=False)
pd.DataFrame(excluded).to_csv(OUT/"phase49_exclusions_and_reasons.csv",index=False)
(OUT/"phase49_county_windows_2023_2024.geojson").write_text(json.dumps({
 "type":"FeatureCollection",
 "features":[{"type":"Feature","geometry":mapping(r["window_geometry"]),
              "properties":{k:r[k] for k in ("disasterNumber","county_fips","incident_start",
                            "event_month","window_area_km2","choice_order","site_index")}} for r in roster]}))

def tile(lon,lat):
    a=math.floor(lon/10)*10
    b=math.ceil(lat/10)*10
    return f"{abs(a)}{'W' if a<0 else 'E'}_{abs(b)}{'S' if b<0 else 'N'}"

def href(t,ym):
    y,m=ym.split("-")
    return f"{SOURCE}/{y}/{y}_{m}/monthlyhistory_{t}_v1_5_{y}_{m}.tif"

# A county's clipped window can straddle a JRC 10-degree tile seam; those
# few cases are explicitly marked, not silently clipped.
observations=[];pairwise=[]
for i,r in enumerate(roster):
    geom=r["window_geometry"]
    dates={"baseline":r["prior_year_same_month"],"event":r["event_month"],
           "following":r["following_month"]}
    sample_id=f"{r['disasterNumber']}_{r['county_fips']}_{r['site_index']}"
    t=tile(r["anchor_lon"],r["anchor_lat"])
    result={};tf_reference=None;shape_reference=None;mask_reference=None
    for phase,ym in dates.items():
        url=href(t,ym)
        entry={"sample_id":sample_id,"disasterNumber":r["disasterNumber"],
            "county_fips":r["county_fips"],"site_index":r["site_index"],"pilot_disaster":r["original_optical_pilot_disaster"],
            "source_url":url,"jrc_month":ym,"phase":phase,"status":"unattempted",
            "site_area_km2":r["window_area_km2"]}
        try:
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                GDAL_HTTP_TIMEOUT="55",GDAL_HTTP_MAX_RETRY="2",
                GDAL_HTTP_MULTIRANGE="YES"):
                with rasterio.open(url) as src:
                    assert src.crs is not None
                    if str(src.crs)!="EPSG:4326":raise RuntimeError("unexpected_projection")
                    win=from_bounds(*geom.bounds,transform=src.transform).round_offsets().round_lengths()
                    win=win.intersection(Window(0,0,src.width,src.height))
                    if win.width>260 or win.height>260:raise RuntimeError(f"unexpected_window_size {win}")
                    matrix=src.read(1,window=win)
                    tf=src.window_transform(win)
                    county_mask=geometry_mask([mapping(geom)],out_shape=matrix.shape,
                                  transform=tf,invert=True,all_touched=False)
                    if not county_mask.any():raise RuntimeError("window_has_no_center_in_polygon")
                    if tf_reference is not None and (
                         tf!=tf_reference or matrix.shape!=shape_reference
                         or not np.array_equal(county_mask,mask_reference)):
                        raise RuntimeError("source_raster_grid_not_aligned")
                    tf_reference=tf;shape_reference=matrix.shape;mask_reference=county_mask
                    arr=matrix[county_mask]
                    assert np.isin(arr,[0,1,2]).all()
                    present=(arr>0);water=(arr==2);dry=(arr==1)
                    entry.update({"status":"read_ok","sample_pixels":int(len(arr)),
                        "observed_pixels":int(present.sum()),"unobserved_pixels":int((arr==0).sum()),
                        "water_pixels":int(water.sum()),"dry_pixels":int(dry.sum()),
                        "observation_fraction":float(present.mean()),
                        "water_fraction_of_observed":float(water.sum()/present.sum()) if present.sum() else None})
                    result[phase]=arr.copy()
                    if "pixel_area_km2" not in entry:
                        lon0,lat0=r["anchor_lon"],r["anchor_lat"]
                        dx,dy=abs(src.res[0]),abs(src.res[1])
                        pxbox=box(lon0-dx/2,lat0-dy/2,lon0+dx/2,lat0+dy/2)
                        entry["pixel_area_km2"]=float(abs(GEOD.geometry_area_perimeter(pxbox)[0])/1e6)
                        result["area"]=entry["pixel_area_km2"]
        except Exception as e:
            entry.update({"status":"read_failed","error":type(e).__name__+": "+str(e)[:230]})
        observations.append(entry)
    for after in ("event","following"):
        cmp={"sample_id":sample_id,"disasterNumber":r["disasterNumber"],
         "county_fips":r["county_fips"],"site_index":r["site_index"],"event_month":r["event_month"],
         "comparison_phase":after,"status":"unattempted",
         "site_area_km2":r["window_area_km2"],
         "measured_independently_from_radar_NOAA_or_dollar_targets":True,
         "observed_water_change_is_not_verified_flood":True}
        if "baseline" in result and after in result:
            a,b=result["baseline"],result[after]
            eligible=(a>0)&(b>0)
            gain=(a==1)&(b==2)
            loss=(a==2)&(b==1)
            cmp.update({"status":"paired_read_ok","pixel_area_km2_approx":result["area"],
                "both_observed_n":int(eligible.sum()),
                "both_observed_frac_of_site":float(eligible.mean()),
                "new_water_cells_prior_dry_event_water":int(gain.sum()),
                "disappeared_water_cells_prior_water_event_dry":int(loss.sum()),
                "new_water_sampled_km2_approx":float(gain.sum()*result["area"]),
                "water_loss_sampled_km2_approx":float(loss.sum()*result["area"]),
                "only_unobserved_is_NOT_dry":True})
        else:cmp.update({"status":"missing_source_month"})
        pairwise.append(cmp)
    print("COUNTY_SAMPLE",i+1,"of",len(roster),sample_id,"event_observed",
          [q.get("observation_fraction") for q in observations[-3:] if q.get("phase")=="event"],
          "new_water",[round(x["new_water_sampled_km2_approx"],4) for x in pairwise[-2:]
                       if x.get("status")=="paired_read_ok"],flush=True)
obs=pd.DataFrame(observations)
pairs=pd.DataFrame(pairwise)
obs.to_csv(OUT/"phase49_real_JRC_monthly_2022_24_county_samples.csv",index=False)
pairs.to_csv(OUT/"phase49_paired_prior_year_surface_water_transitions.csv",index=False)
case_agg=pairs[pairs.status=="paired_read_ok"].groupby(
 ["disasterNumber","comparison_phase"],as_index=False).agg(
    sampled_counties_with_paired_data=("county_fips","nunique"),
    both_observed_pixels=("both_observed_n","sum"),
    sample_candidate_water_gain_km2=("new_water_sampled_km2_approx","sum"),
    sample_candidate_water_loss_km2=("water_loss_sampled_km2_approx","sum"))
case_agg.to_csv(OUT/"phase49_36_disaster_sampled_water_comparison.csv",index=False)

evt=obs[obs.phase=="event"]
event_pairs=pairs[pairs.comparison_phase=="event"]
valid_event=event_pairs[event_pairs.status=="paired_read_ok"]
good_cover=valid_event[valid_event.both_observed_frac_of_site>=.30]
aggregate={
 "fema_FY2023_24_flood_declarations":41,
 "incident_2023_24":37,
 "eligible_with_strict_county":36,
 "declared_exclusions":excluded,
 "fixed_county_samples":len(roster),
 "unique_counties_sampled":int(ra.county_fips.nunique()),
 "pilot_disasters_included":int(ra.loc[ra.original_optical_pilot_disaster,"disasterNumber"].nunique()),
 "new_nonpilot_disasters_included":int(ra.loc[~ra.original_optical_pilot_disaster,"disasterNumber"].nunique()),
 "real_monthly_raster_reads_attempted":len(obs),
 "real_monthly_raster_reads_successful":int((obs.status=="read_ok").sum()),
 "event_month_readable_sites":int((evt.status=="read_ok").sum()),
 "event_month_sites_30pct_observed":int((evt.observation_fraction>=.30).sum()),
 "paired_event_baseline_sites":len(valid_event),
 "paired_event_baseline_sites_both_observed_30pct":len(good_cover),
 "cases_with_any_paired_event_baseline_site":int(valid_event.disasterNumber.nunique()),
 "cases_with_any_new_water_cells":int(valid_event[valid_event.new_water_sampled_km2_approx>0].disasterNumber.nunique()),
 "observed_geographic_window_km2_before_cloud_censor":float(ra.window_area_km2.sum()),
 "matched_pair_new_water_within_sample_windows_km2_NOT_DISASTER_EXTENT":float(valid_event.new_water_sampled_km2_approx.sum()),
 "verified_disaster_specific_inundation_footprints":0,
 "population_or_building_exposure_estimates":0,
 "funding_models_fitted":0,
 "frozen_accepted_Track_A_unchanged":True,
 "source_admin":"Official FEMA designation joined to US Census 2023 county polygons",
 "source_satellite":"JRC Global Surface Water v1.5 monthly Landsat C2 via public source.coop",
 "restrictions":[
  "Sampling is deterministic and outcome-blind, but only up to two counties per disaster and three sample windows per selected county; not a statistically weighted disaster-wide extent estimate.",
  "The sampled 4km squares may include tributaries or permanent lakes. Prior-year changes can reflect seasonality, management or drought unrelated to FEMA event.",
  "The month in which the incident began is not an event-day flood map; short-lived flooding may be missed entirely.",
  "A JRC class 0 pixel is UNOBSERVED and must not enter the dry denominator or paired water transition.",
  "Spatial JRC pixels are approximately 25-30m; 4km windows are clipped using generalized 2023 county geometry.",
  "A change from prior-year dry to event-month water is candidate seasonal surface-water difference, not a validated flood polygon.",
  "Post-event monthly imagery and final county designations are retrospective, not available as T0/T+72 features.",
  "This is an eligibility and coverage study; it does NOT produce a full disaster footprint, loss estimate, or an operational AI decision-support feature.",
  "No FEMA funding targets accessed and accepted Track A not modified."]}

# Compare only quality-gated >=30% paired observable area; normalize all
# water changes by jointly observed area, not number of extra sample windows.
main=pairs[(pairs.comparison_phase=="event")&(pairs.status=="paired_read_ok")].copy()
main["coverage_pass"]=main.both_observed_frac_of_site>=.30
main["observed_area_km2"]=main.both_observed_n*main.pixel_area_km2_approx
main["water_gain_rate"]=main.new_water_sampled_km2_approx/main.observed_area_km2.replace(0,np.nan)
main.to_csv(OUT/"phase49_observation_gated_site_comparisons.csv",index=False)
quality=main[main.coverage_pass].copy()
groups=[]
for dn in sorted(ra.disasterNumber.unique()):
    record={"disasterNumber":int(dn)}
    for name,sites in [("anchor",quality[quality.site_index==1]),("expanded",quality)]:
        d=sites[sites.disasterNumber==dn]
        narea=float(d.observed_area_km2.sum())
        area=float(d.new_water_sampled_km2_approx.sum())
        record[name+"_quality_sites"]=len(d)
        record[name+"_observed_area_km2"]=narea
        record[name+"_new_water_km2"]=area if len(d) else np.nan
        record[name+"_rate_per_100_observed_km2"]=100*area/narea if narea>0 else np.nan
        record[name+"_any_detected"]=bool(area>0) if len(d) else None
    groups.append(record)
comparison=pd.DataFrame(groups)
comparison["expanded_minus_anchor_rate"]=comparison.expanded_rate_per_100_observed_km2-comparison.anchor_rate_per_100_observed_km2
comparison.to_csv(OUT/"phase49_36_case_anchor_vs_spatial_expansion.csv",index=False)
paired_cases=comparison.dropna(subset=["expanded_rate_per_100_observed_km2","anchor_rate_per_100_observed_km2"])
aggregate.update({
 "phase49_sampling_design":"same Phase48 county roster, original anchor plus up to two interior maximin 6.5km-separated 4km squares per county",
 "sites_added_outside_phase48_anchor":int((ra.site_index>1).sum()),
 "original_phase48_site_count":63,
 "counties_with_three_sites":int((ra.groupby(["disasterNumber","county_fips"]).site_index.max()==3).sum()),
 "counties_with_fewer_than_three_sites":int((ra.groupby(["disasterNumber","county_fips"]).site_index.max()<3).sum()),
 "paired_sites_quality30pct":int(len(quality)),
 "anchor_sites_quality30pct":int((quality.site_index==1).sum()),
 "additional_sites_quality30pct":int((quality.site_index>1).sum()),
 "paired_cases_both_anchor_and_expanded":len(paired_cases),
 "case_positive_anchor_quality30pct":int((comparison.anchor_new_water_km2.fillna(0)>0).sum()),
 "case_positive_expanded_quality30pct":int((comparison.expanded_new_water_km2.fillna(0)>0).sum()),
 "case_newly_positive_only_after_spatial_expansion":int(((comparison.anchor_new_water_km2.fillna(0)==0)&(comparison.expanded_new_water_km2.fillna(0)>0)).sum()),
 "quality30pct_anchor_new_water_rate_per_100_observed_km2":100*quality.loc[quality.site_index==1,"new_water_sampled_km2_approx"].sum()/quality.loc[quality.site_index==1,"observed_area_km2"].sum() if (quality.site_index==1).any() else None,
 "quality30pct_expanded_new_water_rate_per_100_observed_km2":100*quality.new_water_sampled_km2_approx.sum()/quality.observed_area_km2.sum() if len(quality) else None,
 "geographic_heterogeneity_case_rate_median_absolute_change":float(paired_cases.expanded_minus_anchor_rate.abs().median()) if len(paired_cases) else None,
 "case_rank_spearman_anchor_vs_expanded":float(paired_cases.anchor_rate_per_100_observed_km2.rank().corr(paired_cases.expanded_rate_per_100_observed_km2.rank())) if len(paired_cases)>2 else None,
 "not_a_probabilistic_or_population_weighted_disaster_sampling_design":True,
 "phase49_predeclared_before_funding_test":True
})
# Compare anchor raster values to prior Phase48 where BOTH runs read usable
# scenes. Changes from transient source failures are not silently censored.
prior_pairs=pd.read_csv(IN/"phase48"/"phase48_paired_prior_year_surface_water_transitions.csv")
prior_evt=prior_pairs[(prior_pairs.comparison_phase=="event")&(prior_pairs.status=="paired_read_ok")].copy()
prior_evt["sample_id"]=prior_evt.sample_id+"_1"
now_anchor=main[(main.site_index==1)&(main.status=="paired_read_ok")].copy()
check=prior_evt.merge(now_anchor,on="sample_id",suffixes=("_p48","_p49"),validate="one_to_one")
if len(check):
    assert np.allclose(check.new_water_cells_prior_dry_event_water_p48,
                       check.new_water_cells_prior_dry_event_water_p49,atol=0,rtol=0)
aggregate["phase48_anchor_new_water_reproducibility_cases"]=int(len(check))
# Explicit 0.3, 0.6 and 0.8 observation coverage sensitivity.
sensitivity=[]
for threshold in (0,.3,.6,.8):
    sub=main[main.both_observed_frac_of_site>=threshold]
    for site_group,subsub in (("anchor_only",sub[sub.site_index==1]),("all_spatial_sites",sub)):
        area=float(subsub.observed_area_km2.sum())
        sensitivity.append({"threshold":threshold,"group":site_group,"sites":len(subsub),
          "cases_covered":int(subsub.disasterNumber.nunique()),
          "new_water_area_sample_km2":float(subsub.new_water_sampled_km2_approx.sum()),
          "observed_sample_area_km2":area,
          "water_change_km2_per_100_observed_km2":100*subsub.new_water_sampled_km2_approx.sum()/area if area>0 else None})
pd.DataFrame(sensitivity).to_csv(OUT/"phase49_sampling_coverage_threshold_sensitivity.csv",index=False)

(OUT/"phase49_summary.json").write_text(json.dumps(aggregate,indent=2))
(OUT/"PHASE49_REPORT.md").write_text(
 "# Phase 49 — Outcome-blind, geographically distributed disaster-level flood source screen\n\n"
 "The cohort is 36 FEMA Flood disasters with 2023/2024 incident starts and strict designated counties. Phase49 expands the same Phase48 anchor sites to up to three widely separated fixed windows per selected county. "
 "Up to two separated county interiors per disaster were selected without using NOAA impact reports, radar signatures or dollar targets. "
 "A clipped 4km square was evaluated from 2023/24 monthly Landsat JRC surface-water classifications, "
 "using the same calendar month of the preceding year as a control. "
 "**This is a limited sampled-water source audit, not a disaster-wide flood estimate.**\n\n"
 "## Aggregate measures\n\n"+json.dumps({k:v for k,v in aggregate.items() if k not in ("restrictions","declared_exclusions")},indent=2)+
 "\n\n## Cases (prior-year same-month vs incident month)\n\n"+
 case_agg[case_agg.comparison_phase=="event"].to_markdown(index=False)+
 "\n\n## Phase49 anchor vs expanded samples: normalized new-water rate per 100 observed km²\n\n"+comparison.to_markdown(index=False)+
 "\n\n## Coverage sensitivity\n\n"+pd.DataFrame(sensitivity).to_markdown(index=False)+
 "\n\n## Observability limitations\n\n"+"\n".join("- "+x for x in aggregate["restrictions"]))
print("PHASE49_SUMMARY",json.dumps({k:v for k,v in aggregate.items() if k not in ("restrictions","declared_exclusions")},indent=2),flush=True)
