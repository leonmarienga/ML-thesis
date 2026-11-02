#!/usr/bin/env python3
"""Predeclared paired chronological flood-only funding ablation, FY2021-2024.

Compare on the same 59 FEMA Flood records:
  FEMA_ONLY                 retrospective FEMA operational attributes,
  FEMA_SINGLE_RAIN          same + Daymet 1 NOAA report location,
  FEMA_MULTI_RAIN           same + median/max/range across <=3 NOAA locations,
  FEMA_MULTI_RAIN_USGS      same + normalized selected gauge for 11/92 cases.

Each held-out FY2021-24 model trains only on earlier FY2019-24 floods
(33 -> 43 -> 63 -> 71 prior-year disasters). All location matches and physical
features were defined WITHOUT using funding targets. No train/test leakage:
fitted imputation and state one-hot features derive only from training rows.

Crucial: this is a retrospective, small-sample, fully exploratory study.
Current FEMA operation totals and NOAA/Daymet report records are NOT verified
available at emergency declaration. No direct valid comparison of n=59
retrospective flood-only metrics to accepted 912-row Track A LFYO metrics.
No changes to accepted frozen router, stack, or trained models.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor,ExtraTreesClassifier
from sklearn.metrics import r2_score
from sklearn.base import clone

ROOT=Path(__file__).resolve().parents[1]
IN=ROOT/"audit_inputs"/"flood_ablation"
OUT=ROOT/"audit_outputs"/"flood_forward_ablation"
OUT.mkdir(parents=True,exist_ok=True)
YEARS=[2021,2022,2023,2024]
BINS=np.array([0,1e5,1e6,50e6,200e6,500e6,np.inf],float)
LABELS=["0-100K","100K-1M","1M-50M","50M-200M","200M-500M","500M+"]
NUMERIC_FEMA=["durationDays","declarationDelayDays","missionAssignmentCount",
     "uniqueAgencyCount","uniqueMaTypeCount","uniquePriorityCount",
     "missionDensity","agencyDensity","responseComplexityScore"]
NUMERIC_SINGLE=["event_precip_total_mm","event_max_1day_mm",
    "event_max_3day_mm","event_max_7day_mm",
    "antecedent_7day_total_mm",
    "event_peak_3day_prior_season_percentile"]
NUMERIC_MULTI=["n_usable_locations","n_requested_locations",
    "multi_event_precip_total_mm_median",
    "multi_event_precip_total_mm_max","multi_event_precip_total_mm_range",
    "multi_event_max_1day_mm_median","multi_event_max_3day_mm_max",
    "multi_event_max_3day_mm_range",
    "multi_event_peak_3day_prior_season_percentile_median"]
NUMERIC_FLOW=["event_peak_daily_mean_cfs",
    "peak_to_prior_seasonal_median_ratio",
    "peak_percentile_in_prior_seasonal_daily_means",
    "event_daily_completeness"]
ARMS=["fema_only","fema_single_rain","fema_multi_rain","fema_multi_rain_usgs"]
N_TREES=350
def band(x):
    return LABELS[int(np.searchsorted(BINS,max(0,float(x)),side="right")-1)]

def prepare():
    master=pd.read_excel(ROOT/"master_openfema_40plus.xlsx")
    assert len(master)==971 and master.disasterNumber.nunique()==971
    m=master[(master.incidentType=="Flood") & master.fyDeclared.between(2019,2024)].copy()
    assert len(m)==92 and m.disasterNumber.nunique()==92
    # Prevent accidental use of target as explanatory data.
    m["target"]=pd.to_numeric(m.totalObligatedFunding,errors="raise").clip(lower=0)
    m["band"]=m.target.map(band)
    m["declaration_month"]=pd.to_datetime(m.declarationDate,utc=True).dt.month
    m["declaration_year"]=pd.to_datetime(m.declarationDate,utc=True).dt.year
    m["incident_month"]=pd.to_datetime(m.incidentBeginDate,utc=True).dt.month
    m["incident_duration_days"]=(pd.to_datetime(m.incidentEndDate,utc=True)-
                  pd.to_datetime(m.incidentBeginDate,utc=True)).dt.days
    # Only whitelist original FEMA fields; don't include FEMA funding variables.
    available=[c for c in NUMERIC_FEMA if c in m.columns]
    assert "missionAssignmentCount" in available
    # Omit the highly correlated ratios if missing and always explicitly report.
    single=pd.read_csv(IN/"single"/"all_92_flood_daymet_rainfall.csv")
    multi=pd.read_csv(IN/"multi"/"all_92_multilocation_rainfall_profiles.csv")
    flow=pd.read_csv(IN/"flow"/"flood_stratified_24_normalized_discharge.csv")
    for title,df in [("single",single),("multi",multi),("flow",flow)]:
        assert df.disasterNumber.nunique()==len(df),title
    assert len(single)==92 and len(multi)==92
    assert len(flow)==24
    singles=single[["disasterNumber"]+NUMERIC_SINGLE+["rainfall_available"]].copy()
    singles=singles.rename(columns={"rainfall_available":"single_rain_available",
        **{c:"single_"+c for c in NUMERIC_SINGLE}})
    multis=multi[["disasterNumber"]+NUMERIC_MULTI+["any_rainfall",
         "at_least_two_locations"]].copy()
    multis=multis.rename(columns={"any_rainfall":"multi_rain_available",
          "at_least_two_locations":"multi_two_location_available",
         **{c:"multi_"+c for c in NUMERIC_MULTI}})
    flows=flow[["disasterNumber"]+NUMERIC_FLOW+
         ["normalized_discharge_available"]].copy()
    flows=flows.rename(columns={"normalized_discharge_available":"flow_available",
          **{c:"flow_"+c for c in NUMERIC_FLOW}})
    d=m.merge(singles,on="disasterNumber",how="left",validate="one_to_one")
    d=d.merge(multis,on="disasterNumber",how="left",validate="one_to_one")
    d=d.merge(flows,on="disasterNumber",how="left",validate="one_to_one")
    assert len(d)==92 and d.target.notna().all()
    for c in ["single_rain_available","multi_rain_available",
              "multi_two_location_available","flow_available"]:
        d[c]=d[c].fillna(False).astype(bool)
    assert 69<=int(d.single_rain_available.sum())<=87
    assert 69<=int(d.multi_rain_available.sum())<=87
    assert int(d.flow_available.sum())==11
    assert not d.disasterNumber.duplicated().any()
    # Rainfall derived from NOAAs whose source versions may be finalized
    # *after* disaster and FEMA full-event operations. This is retrospective.
    features={
      "fema_only":available,
      "fema_single_rain":available+[f"single_{x}" for x in NUMERIC_SINGLE]+[
          "single_rain_available"],
      "fema_multi_rain":available+[f"multi_{x}" for x in NUMERIC_MULTI]+[
          "multi_rain_available","multi_two_location_available"],
      "fema_multi_rain_usgs":available+[f"multi_{x}" for x in NUMERIC_MULTI]+[
          "multi_rain_available","multi_two_location_available"]+
          [f"flow_{x}" for x in NUMERIC_FLOW]+["flow_available"]
    }
    return d,features,available

def design(train,test,cols):
    # Restrict all input columns to whitelist and source-independent state/month.
    original_categorical=["state"]
    a=pd.get_dummies(train[original_categorical].astype(str).fillna("UNKNOWN"),dtype=float)
    b=pd.get_dummies(test[original_categorical].astype(str).fillna("UNKNOWN"),dtype=float)
    b=b.reindex(columns=a.columns,fill_value=0.0)
    num=["declaration_month","declaration_year","incident_month",
         "incident_duration_days"]+cols
    xx=train[num].copy()
    yy=test[num].copy()
    # Missingness indicators distinguish unobserved rainfall/flow from zero.
    for c in num:
        xx[c]=pd.to_numeric(xx[c],errors="coerce")
        yy[c]=pd.to_numeric(yy[c],errors="coerce")
        if c.endswith("_available"):
            xx[c]=xx[c].fillna(0).astype(float)
            yy[c]=yy[c].fillna(0).astype(float)
    na_x=xx.isna().astype(float).add_prefix("missing_")
    na_y=yy.isna().astype(float).add_prefix("missing_")
    fill=xx.median().fillna(0)
    for c in num:
        # Transform high-skew inputs with deterministic target-independent map.
        if c in NUMERIC_FEMA or c.startswith(("single_","multi_","flow_")):
            if c.endswith("_available") or c.startswith(("multi_n_",)):
                continue
            if ("percentile" in c or "completeness" in c):
                continue
            # Ratios of observed median flows are nonnegative when defined.
            x=xx[c];y=yy[c]
            if x.dropna().ge(0).all() and y.dropna().ge(0).all():
                xx[c]=np.log1p(x)
                yy[c]=np.log1p(y)
                fill[c]=np.log1p(fill[c]) if fill[c]>=0 else fill[c]
    xx=xx.fillna(fill);yy=yy.fillna(fill)
    for c in num:
        assert xx[c].notna().all() and yy[c].notna().all()
    X=pd.concat([a.reset_index(drop=True),xx.reset_index(drop=True),
                 na_x.reset_index(drop=True)],axis=1)
    Y=pd.concat([b.reset_index(drop=True),yy.reset_index(drop=True),
                 na_y.reset_index(drop=True)],axis=1)
    assert list(X.columns)==list(Y.columns)
    assert np.isfinite(X.to_numpy(float)).all() and np.isfinite(Y.to_numpy(float)).all()
    return X,Y

def fit_fold(train,test,features,year):
    X,Y=design(train,test,features)
    reg=ExtraTreesRegressor(n_estimators=N_TREES,min_samples_leaf=2,
          max_features=.85,random_state=20261102+year,n_jobs=-1)
    reg.fit(X,np.log1p(train.target.to_numpy(float)))
    amount=np.expm1(reg.predict(Y)).clip(0)
    # Independent classifier for six-band recall; same model/seed across arms.
    clf=ExtraTreesClassifier(n_estimators=N_TREES,min_samples_leaf=2,
          max_features=.85,class_weight="balanced_subsample",
          random_state=20261202+year,n_jobs=-1)
    clf.fit(X,train.band.to_numpy())
    labels=clf.predict(Y)
    return amount,labels

def summarize(g):
    truth=g.target.to_numpy(float);p=g.pred_amount.to_numpy(float)
    ae=np.abs(truth-p)
    logae=np.abs(np.log1p(truth)-np.log1p(p))
    out={"n":int(len(g)),"MAE_USD":float(np.mean(ae)),
      "RMSE_USD":float(np.sqrt(np.mean((truth-p)**2))),
      "log1p_MAE":float(np.mean(logae)),
      "MedianAE_USD":float(np.median(ae)),
      "R2":float(r2_score(truth,p)) if len(g)>1 and np.var(truth)>0 else None,
      "funding_band_accuracy":float(np.mean(g.band==g.pred_band)),
      "training_target_not_as_feature":True}
    for name in LABELS:
        sub=g[g.band==name]
        out["recall_"+name]=float(np.mean(sub.pred_band==name)) if len(sub) else None
        out["support_"+name]=int(len(sub))
    return out

def paired(pred,source):
    a=pred[pred.arm=="fema_only"].set_index("disasterNumber").sort_index()
    b=pred[pred.arm==source].set_index("disasterNumber").sort_index()
    assert list(a.index)==list(b.index)
    delta=np.abs(b.target-b.pred_amount).to_numpy(float)-np.abs(
           a.target-a.pred_amount).to_numpy(float)
    # Uncertainty *descriptive only*. Out-of-sample test revisited many times
    # in related work; pairs can be storm-correlated.
    rng=np.random.default_rng(22061102)
    arr=np.array([delta[rng.integers(0,len(delta),len(delta))].mean()
                 for _ in range(2000)])
    return {
       "n":len(delta),
       "MAE_change_new_minus_baseline_USD":float(delta.mean()),
       "paired_bootstrap_95pct_exploratory":[float(v) for v in np.quantile(arr,[.025,.975])],
       "fraction_cases_improved":float(np.mean(delta<0)),
       "fraction_cases_degraded":float(np.mean(delta>0)),
       "baseline_MAE_USD":float(np.mean(np.abs(a.target-a.pred_amount))),
       "challenger_MAE_USD":float(np.mean(np.abs(b.target-b.pred_amount)))}

def main():
    d,features,available=prepare()
    records=[]
    provenance=[]
    for fy in YEARS:
        training=d[d.fyDeclared<fy].copy()
        testing=d[d.fyDeclared==fy].copy()
        assert len(training)>=33 and len(testing)>0
        assert training.fyDeclared.max()<fy
        assert (training.disasterNumber.isin(testing.disasterNumber)).sum()==0
        provenance.append({"test_fiscal_year":fy,"n_train":len(training),
             "n_test":len(testing),
             "n_train_with_rain":int(training.multi_rain_available.sum()),
             "n_test_with_rain":int(testing.multi_rain_available.sum()),
             "n_train_with_flow":int(training.flow_available.sum()),
             "n_test_with_flow":int(testing.flow_available.sum()),
             "training_end_year":int(training.fyDeclared.max())})
        for arm in ARMS:
            p,l=fit_fold(training,testing,features[arm],fy)
            for (_,r),est,cl in zip(testing.iterrows(),p,l):
                records.append({"disasterNumber":int(r.disasterNumber),"fyDeclared":fy,
                   "state":r.state,"arm":arm,"target":float(r.target),
                   "band":r.band,"pred_amount":float(est),"pred_band":str(cl),
                   "rain_available":bool(r.multi_rain_available),
                   "flow_available":bool(r.flow_available)})
        print("YEAR",fy,"train",len(training),"test",len(testing),flush=True)
    pred=pd.DataFrame(records)
    assert pred.disasterNumber.nunique()==59,(pred.disasterNumber.nunique(),len(pred))
    assert (pred.groupby("arm").size()==59).all()
    assert len(pred)==59*len(ARMS)
    overall={arm:summarize(pred[pred.arm==arm]) for arm in ARMS}
    byyear={str(fy):{arm:summarize(pred[(pred.fyDeclared==fy)&
             (pred.arm==arm)]) for arm in ARMS} for fy in YEARS}
    by_rain_avail={
      status:{arm:summarize(pred[(pred.rain_available==status)&
               (pred.arm==arm)]) for arm in ARMS}
      for status in [True,False]}
    comparisons={arm:paired(pred,arm) for arm in ARMS if arm!="fema_only"}
    summary={
       "study":"Experimental flood-only forward fiscal-year out-of-time feature ablation",
       "cohort_fema_flood_fy2019_2024":92,
       "predicted_fy2021_2024_cases":59,
       "folds":provenance,
       "feature_families":features,
       "included_fema_numeric_features":available,
       "models":"Same ExtraTrees 350-tree log1p dollar regressors and independent six-band classifiers in all arms; min leaf 2; year-seeded.",
       "results":overall,"by_year":byyear,"by_rain_observation_availability":by_rain_avail,
       "paired_vs_fema_only":comparisons,
       "accepted_stack_unchanged":True,
       "NOT_directly_comparable_to_accepted_912_rows":True,
       "limitation":[
         "This comparison is ONLY within 2019-24 Flood cases, using 2019-20 for initial training (33 examples), testing 2021-24. Small data and severe band imbalance.",
         "Current FEMA retrospective mission assignments and completed duration fields are finalized later, so this is not a real T0/T+72 decision model.",
         "Daymet and final NOAA report records are retrospectively reconstructed, not publication-time certified.",
         "Rows from a single storm may be dependent. Pairs bootstrap ignores storm clustering.",
         "We repeatedly inspected these recent years in earlier research; these tests are not an untouched independent research set.",
         "No changes to the frozen accepted 830/912 six-band router or $5.015M 912-case model.",
         "We compare same model and test cases, not directly the very different accepted Track A models.",
         "The USGS feature is only available for 11/92, selected through gauge/watershed screens; no claim of disaster hydrological causal validation.",
         "Missing data are imputed from previous-year training examples plus explicit missingness markers, not set to zero.",
         "Physical features from NOAA report point samples cannot establish full FEMA disaster footprint, affected housing, or humanitarian need."
       ]}
    pred.to_csv(OUT/"paired_forward_flood_funding_predictions.csv",index=False)
    pd.DataFrame(provenance).to_csv(OUT/"training_test_provenance.csv",index=False)
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2))
    print("BEGIN_FLOOD_ABLATION")
    print(json.dumps({"cohort":summary["cohort_fema_flood_fy2019_2024"],
       "test":summary["predicted_fy2021_2024_cases"],"folds":provenance,
       "overall":overall,"paired":comparisons},indent=2))
    print("END_FLOOD_ABLATION")

if __name__=="__main__":main()
