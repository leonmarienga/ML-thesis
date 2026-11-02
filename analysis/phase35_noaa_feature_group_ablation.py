#!/usr/bin/env python3
"""Phase35 post-selection ablation: NOAA report counts vs reported damage.

Exploratory retrospective FY2021–24 FLOOD test, 59 cases. Same original
FY-earlier-only ExtraTrees architecture; original frozen Track A untouched.
The FEMA/NOAA geographic association does NOT certify event attribution, and
2026 NOAA archive revisions are NOT historical decision-time snapshots.
"""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from sklearn.metrics import r2_score
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"analysis"))
import flood_forward_model_ablation as base
INPUT=ROOT/"audit_inputs"/"phase35"
OUT=ROOT/"audit_outputs"/"phase35"
OUT.mkdir(parents=True,exist_ok=True)
base.IN=INPUT/"phase25"/"model_inputs"
base.OUT=OUT/"underlying_baseline"
base.OUT.mkdir(exist_ok=True)
d,feat,_=base.prepare()
c=pd.read_csv(INPUT/"phase34"/"fema_disaster_candidate_impact_features.csv")
orig=pd.read_csv(INPUT/"phase34"/"ablation"/"exploratory_forward_year_predictions.csv")
assert len(d)==92 and len(c)==92 and orig.disasterNumber.nunique()==59
a=c[["disasterNumber","eligible_FEMA_county_areas","matched_FEMA_counties",
     "unambiguous_noaa_core_count","unambiguous_nonmissing_damage_reports",
     "unambiguous_reported_sum_usd","unambiguous_reported_max_usd"]].copy()
a["noaa_has_coverage"]=(a.unambiguous_noaa_core_count>0).astype(int)
a["noaa_reported_damage_available"]=(a.unambiguous_nonmissing_damage_reports>0).astype(int)
for new,source in [
 ("noaa_log1p_sum_usd","unambiguous_reported_sum_usd"),
 ("noaa_log1p_max_usd","unambiguous_reported_max_usd"),
 ("noaa_log1p_report_count","unambiguous_noaa_core_count"),
 ("noaa_log1p_reported_damage_count","unambiguous_nonmissing_damage_reports"),
 ("noaa_log1p_eligible_counties","eligible_FEMA_county_areas"),
 ("noaa_log1p_counties_matched","matched_FEMA_counties")]:
 a[new]=np.log1p(pd.to_numeric(a[source],errors="coerce").clip(lower=0))
G=["noaa_log1p_eligible_counties","noaa_log1p_counties_matched","noaa_has_coverage"]
E=["noaa_log1p_report_count","noaa_log1p_reported_damage_count","noaa_reported_damage_available"]
L=["noaa_log1p_sum_usd","noaa_log1p_max_usd"]
d=d.merge(a[["disasterNumber"]+G+E+L],on="disasterNumber",how="left",validate="one_to_one")
arms={
 "fema_only":feat["fema_only"],
 "fema_geography_only":feat["fema_only"]+G,
 "fema_geo_event_counts":feat["fema_only"]+G+E,
 "fema_damage_only":feat["fema_only"]+L+["noaa_reported_damage_available"],
 "fema_geo_counts_and_damage":feat["fema_only"]+L+E.slice(0,2)+G.slice(0,2)+G.slice(2)+E.slice(2),
}
rows=[]
for fy in base.YEARS:
 train=d[d.fyDeclared<fy].copy();test=d[d.fyDeclared==fy].copy()
 assert len(train)>0 and len(test)>0 and int(train.fyDeclared.max())<fy
 for name,cols in arms.items():
  amounts,classes=base.fit_fold(train,test,cols,fy)
  for (_,r),amount,cls in zip(test.iterrows(),amounts,classes):
   rows.append({"disasterNumber":int(r.disasterNumber),"fyDeclared":fy,
    "arm":name,"target":float(r.target),"pred_amount":float(amount),
    "true_band":str(r.band),"pred_band":str(cls)})
 print("FY",fy,"train",len(train),"test",len(test),flush=True)
p=pd.DataFrame(rows)
p.to_csv(OUT/"phase35_predictions_59x5.csv",index=False)
# Verify that the two unchanged configurations exactly reproduce Phase34.
for old,new in [("fema_only","fema_only"),("fema_noaa_damage_candidates","fema_geo_counts_and_damage")]:
 o=orig[orig.arm==old].sort_values("disasterNumber")
 n=p[p.arm==new].sort_values("disasterNumber")
 assert (o.disasterNumber.to_numpy()==n.disasterNumber.to_numpy()).all()
 assert np.allclose(o.pred_amount,n.pred_amount,rtol=0,atol=1e-5), (old,"amount mismatch")
 assert (o.pred_band.to_numpy()==n.pred_band.to_numpy()).all(),(old,"band mismatch")
print("REPRODUCED_OLD_ARMS_EXACTLY",flush=True)
summary=[]
for name,g in p.groupby("arm"):
 err=g.pred_amount-g.target
 z={"arm":name,"n":len(g),
    "MAE_USD":float(err.abs().mean()),
    "RMSE_USD":float(np.sqrt((err**2).mean())),
    "R2":float(r2_score(g.target,g.pred_amount)),
    "six_band_accuracy":float((g.pred_band==g.true_band).mean()),
    "positive_target_MAE_USD":float(np.abs(g[g.target>0].pred_amount-g[g.target>0].target).mean()),
    "n_positive_targets":int((g.target>0).sum())}
 summary.append(z)
S=pd.DataFrame(summary).sort_values("MAE_USD")
S.to_csv(OUT/"phase35_factorial_model_metrics.csv",index=False)
base_pred=p[p.arm=="fema_only"].set_index("disasterNumber")
paired=[];peryear=[]
for name,g in p.groupby("arm"):
 if name=="fema_only":continue
 h=g.set_index("disasterNumber").reindex(base_pred.index)
 delta=(h.pred_amount-h.target).abs()-(base_pred.pred_amount-base_pred.target).abs()
 for fy in base.YEARS:
  sub=delta[h.fyDeclared==fy]
  peryear.append({"arm":name,"test_fy":fy,"n":len(sub),"MAE_change_new_minus_baseline_USD":float(sub.mean())})
 year=sorted(set(h.fyDeclared))
 rng=np.random.default_rng(35261102)
 boot=[]
 for j in range(10000):
  ys=rng.choice(year,len(year),replace=True)
  vals=np.concatenate([delta[h.fyDeclared==y].to_numpy() for y in ys])
  boot.append(vals.mean())
 ci=np.quantile(boot,[.025,.975])
 paired.append({"arm":name,"n":len(delta),"MAE_change_new_minus_baseline_USD":float(delta.mean()),
  "improved_cases":int((delta<0).sum()),"worsened_cases":int((delta>0).sum()),"unchanged_cases":int(np.isclose(delta,0,atol=1e-6).sum()),
  "fiscal_year_block_boot_95pct_lower_USD":float(ci[0]),
  "fiscal_year_block_boot_95pct_upper_USD":float(ci[1])})
pd.DataFrame(paired).to_csv(OUT/"phase35_paired_fy_bootstrap.csv",index=False)
pd.DataFrame(peryear).to_csv(OUT/"phase35_by_fiscal_year.csv",index=False)
# Describe discrepancy between original NOAA anchor location and designated FEMA counties.
anchor=pd.read_csv(INPUT/"phase34"/"phase33_anchor_reconciliation.csv")
fema=pd.read_csv(INPUT/"phase34"/"fema_official_designated_areas.csv")
design=set(map(tuple,fema[fema.is_strict_county][["disasterNumber","county_fips"]].drop_duplicates().astype(int).to_numpy()))
anchor["county_is_FEMA_designated"]=[(int(r.disasterNumber),int(r.county_fips)) in design for _,r in anchor.iterrows()]
anchor["audit_reason"]=np.where(anchor.in_expanded_county_date_candidate,"found_in_full_expansion",
 np.where(anchor.strict_match,"outside_FEMA_designated_counties_strict_NOAA_match",
          "outside_FEMA_counties_and_phase33_date_failed"))
# Exceptions: 5 strict_match false but inside expanded list (date near boundary)
assert (anchor.audit_reason=="outside_FEMA_designated_counties_strict_NOAA_match").sum()==123
assert (anchor.audit_reason=="outside_FEMA_counties_and_phase33_date_failed").sum()==7
assert anchor.in_expanded_county_date_candidate.sum()==118
anchor.to_csv(OUT/"phase35_248_anchor_reconciliation_explained.csv",index=False)
reason=anchor.groupby("audit_reason").size().rename("count").reset_index()
reason.to_csv(OUT/"phase35_anchor_reason_counts.csv",index=False)
# Gain breakdown by which heldout cases had data.
wide=p.pivot(index="disasterNumber",columns="arm",values="pred_amount").reset_index()
labels=p[p.arm=="fema_only"][["disasterNumber","target","fyDeclared","true_band"]]
case=wide.merge(labels,on="disasterNumber").merge(c,on="disasterNumber")
for name in arms:
 case["abs_error_"+name]=abs(case[name]-case.target)
case.to_csv(OUT/"phase35_casewise_impact_and_errors.csv",index=False)
ret={
 "n_flood_cases":len(c),"n_eval":59,
 "n_noaa_covered":int(c.unambiguous_noaa_core_count.gt(0).sum()),
 "n_positive_reported_damage":int(c.unambiguous_reported_sum_usd.gt(0).sum()),
 "recomputed_both_old_arms_identical":True,
 "model_metrics":summary,"paired_vs_fema_only":paired,
 "anchor_reasons":reason.to_dict(orient="records"),
 "warnings":[
 "Retrospective NOAA damage values were present in archived records revised after 2024; not T0/T72-valid.",
 "County/date co-occurrence is not causal impact attribution.",
 "FY2021-2024 evaluation repeatedly inspected during feature engineering, so there is no independent test.",
 "Confidence intervals based on four fiscal-year clusters are exploratory only.",
 "NOAA damages are reported estimates, not comprehensive disaster economic losses.",
 "No modifications to 912-case accepted Track A."]}
(OUT/"summary.json").write_text(json.dumps(ret,indent=2))
(OUT/"REPORT.md").write_text(
 "# Phase 35 — Flood NOAA source and feature ablation\n\n"
 "## Main question\nDoes the 59-case retrospective gain come from geography, NOAA event counts, reported damage values, or their combination?\n\n"
 "## Model comparison\n\n"+S.to_markdown(index=False)+"\n\n"
 "## Paired comparisons (new minus FEMA baseline; negative is better)\n\n"
 +pd.DataFrame(paired).to_markdown(index=False)+"\n\n"
 "## Prior NOAA sample anchors\n\n"+reason.to_markdown(index=False)+"\n\n"
 "## Research interpretation\nThis is a retrospective **feature-family diagnostic**, not independent validation. The original Track A remains frozen. Do not treat NOAA reported damages as decision-time available.\n")
print("FINAL_PHASE35",json.dumps({k:v for k,v in ret.items() if k!="warnings"},indent=2),flush=True)
