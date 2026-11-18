#!/usr/bin/env python3
"""Compare fold-local augmentation while preserving real-only scoring and calibration."""
import argparse
import json
from pathlib import Path

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder

from thesis_augmentation_generators import CATEGORY, CLASS_ONLY, DELAY, METHODS, PREDICTORS, RATIOS, TAIL_BANDS, add_rows, report_support
from thesis_forward_models import LABELS, YEARS, estimators, read_table, score, split
from thesis_hazard_features import BANDS
from thesis_uncertainty import conformal_quantile

SETTINGS = [("real",1), *((method,ratio) for method in METHODS for ratio in RATIOS)]
BASELINE = "real_1x"
LEVELS = (.8,.9)
REVIEW = .10
BOOT_SEED = 246885


def name(config):
    return f"{config[0]}_{config[1]}x"


def tail_mask(frame):
    return frame.actual_band.isin(TAIL_BANDS).to_numpy()


def raw_fit(real, hold, identity, configs=SETTINGS, seed_namespace=None):
    """A fold has its own donor fitting, encoder and holdout prediction ledger."""
    seed_namespace = seed_namespace or identity
    assert real.fyDeclared.max() < hold.fyDeclared.min()
    assert not set(real.event_group)&set(hold.event_group)
    assert not set(real.disasterNumber)&set(hold.disasterNumber)
    assert real[PREDICTORS].notna().all().all() and hold[PREDICTORS].notna().all().all()
    encoder = ColumnTransformer([
        ("categorical",OneHotEncoder(handle_unknown="ignore",sparse_output=False),CATEGORY),
        ("delay","passthrough",[DELAY]),
    ])
    x=encoder.fit_transform(real[PREDICTORS])
    z=encoder.transform(hold[PREDICTORS])
    band=real.actual_band.to_numpy()
    amount=real.target.to_numpy()
    rare=tail_mask(real)
    predictions,synthetic,log=[],[],[]
    base=None
    roles=real[["disasterNumber","event_group","fyDeclared"]].assign(fold=identity, role="train")
    held=hold[["disasterNumber","event_group","fyDeclared"]].assign(fold=identity,role="real_holdout")
    for method,ratio in configs:
        rows, details=add_rows(real,method,ratio,identity,seed_namespace)
        log.extend(details)
        if not rows.empty:
            assert rows.synthetic_id.is_unique
            s=encoder.transform(rows[PREDICTORS])
            assert set(rows.source_a)|set(rows.source_b) <= set(real.disasterNumber)
            synthetic.append(rows)
        train_weight=np.where(rare,ratio,1.).astype(float) if method=="weight" else np.ones(len(real))
        if method!="real" and method!="weight" and rows.empty:
            assert base is not None
            prob,pred=base
        else:
            cls,reg=estimators("extra_trees",6)
            class_x=np.vstack([x,s]) if not rows.empty else x
            class_y=np.r_[band,rows.actual_band.to_numpy()] if not rows.empty else band
            additional_weight=.25 if method=="hybrid" else 1.
            weight=np.r_[train_weight,np.repeat(additional_weight,len(rows))]
            # Override the existing balanced class weighting to isolate reweighting in this study.
            cls.set_params(class_weight=None)
            cls.fit(class_x,class_y,sample_weight=weight)
            prob=np.zeros((len(hold),6))
            prob[:,cls.classes_]=cls.predict_proba(z)
            if method in CLASS_ONLY:
                pred=base[1]
                assert rows.target.isna().all()
            else:
                y=np.r_[amount,rows.target.to_numpy(float)] if not rows.empty else amount
                assert np.isfinite(y).all() and y.max()<=np.nextafter(amount.max(),np.inf)
                reg.fit(class_x,np.log1p(y),sample_weight=weight)
                pred=np.expm1(reg.predict(z)).clip(0,amount.max())
            if method=="real":
                base=(prob,pred)
        records=hold[["disasterNumber","event_group","fyDeclared","incidentType","actual_band","target"]].copy()
        records=records.assign(fold=identity,config=name((method,ratio)),method=method,ratio=ratio,
                  amount=pred,generated_n=len(rows),active=method in ("real","weight") or len(rows)>0)
        for b in LABELS:
            records[f"p{b}"]=prob[:,b]
        predictions.append(records)
    return pd.concat(predictions,ignore_index=True), synthetic,log,pd.concat([roles,held],ignore_index=True)


def calibrate(calibrating, record, reference_train):
    """Estimate band prevalence and prediction errors using a disjoint real block."""
    calibration_y=tail_mask(calibrating)
    pcl=[f"p{x}" for x in LABELS]
    original=calibrating[pcl].to_numpy()
    logits=logit(original[:,3:].sum(axis=1).clip(1e-6,1-1e-6))
    prevalence=(calibration_y.sum()+.5)/(len(calibration_y)+1.)
    offset=brentq(lambda a:expit(logits+a).mean()-prevalence,-40,40)
    p=record[pcl].to_numpy()
    target_prob=expit(logit(p[:,3:].sum(axis=1).clip(1e-6,1-1e-6))+offset)
    prior=reference_train.actual_band.value_counts().reindex(LABELS,fill_value=0).to_numpy(float)+.5
    adjusted=p.copy()
    for block,mass in [(slice(0,3),1-target_prob),(slice(3,6),target_prob)]:
        observed=p[:,block]
        marginal=observed.sum(axis=1,keepdims=True)
        conditional=np.repeat((prior[block]/prior[block].sum())[None,:],len(p),axis=0)
        np.divide(observed,marginal,out=conditional,where=marginal>0)
        adjusted[:,block]=conditional*mass[:,None]
    assert np.allclose(adjusted.sum(axis=1),1.)
    result=record.copy()
    result["calibration_intercept"]=offset
    result["calibration_cases"]=len(calibrating)
    result["calibration_tail_cases"]=int(calibration_y.sum())
    for b in LABELS:
        result[f"cal_p{b}"]=adjusted[:,b]
    errors=abs(calibrating.target-calibrating.amount).to_numpy()
    for level in LEVELS:
        width=conformal_quantile(errors,level)
        assert np.isfinite(width)
        low=np.maximum(record.amount-width,0)
        result[f"lower{int(100*level)}"]=low
        result[f"upper{int(100*level)}"]=record.amount+width
    return result


def choose(tuning):
    """Ratios are ranked using eligible real inner predictions only."""
    rows=[]
    for config,frame in tuning.groupby("config"):
        y=tail_mask(frame)
        p=frame[["p3","p4","p5"]].sum(axis=1)
        loss=abs(frame.amount-frame.target)
        use=frame.active
        rows.append({"config":config,"method":frame.method.iloc[0],"ratio":int(frame.ratio.iloc[0]),
            "active_inner_origins":frame.loc[use,"fold"].nunique(),"tune_n":len(frame),"tune_tail_n":int(y.sum()),
            "average_precision":average_precision_score(y,p) if y.any() else np.nan,
            "tail_mae":loss[y].mean() if y.any() else np.nan,"mae":loss.mean()})
    metrics=pd.DataFrame(rows)
    selections={}
    for method,q in metrics.groupby("method"):
        q=q[q.active_inner_origins.gt(0)]
        if q.empty:
            selections[method]=(BASELINE,BASELINE)
            continue
        c=q.sort_values(["average_precision","ratio"],ascending=[False,True],na_position="last").iloc[0]
        r=q.sort_values(["tail_mae","mae","ratio"],na_position="last").iloc[0]
        selections[method]=(c.config,r.config)
    return metrics,selections


def outer_policy(real, pred, calibrating, tuning, year):
    """Keep ratio-selection episodes out of the calibration score pool."""
    keep = ~calibrating.event_group.isin(tuning.event_group)
    calibration_used = calibrating[keep]
    assert set(calibration_used.event_group).isdisjoint(tuning.event_group)
    assert set(calibration_used.event_group).isdisjoint(pred.event_group)
    grid, selections = choose(tuning)
    grid["outer_year"] = year
    intervals, settings = [], []
    for config, g in pred.groupby("config"):
        cal=calibration_used[calibration_used.config.eq(config)]
        x=calibrate(cal,g,real)
        x["variant"]=config
        x["classification_config"]=config
        x["amount_config"]=config
        x["probability_generation_active"]=x.active
        x["amount_generation_active"]=x.active & ~x.method.isin(CLASS_ONLY)
        intervals.append(x)
    calibrated=pd.concat(intervals,ignore_index=True)
    chosen_frames=[]
    for method,(c,r) in selections.items():
        settings.append({"outer_year":year,"method":method,"classification_config":c,"amount_config":r})
        a=calibrated[calibrated.variant.eq(c)].set_index("disasterNumber").copy()
        b=calibrated[calibrated.variant.eq(r)].set_index("disasterNumber")
        columns=["amount","amount_generation_active","lower80","upper80","lower90","upper90"]
        a[columns]=b[columns]
        a["variant"]=f"nested_{method}"
        a["classification_config"]=c
        a["amount_config"]=r
        chosen_frames.append(a.reset_index())
    inner = pd.concat([calibrating.assign(used_for_selection_or_calibration=keep),
                       tuning.assign(used_for_selection_or_calibration=True)], ignore_index=True)
    return (pd.concat([calibrated, *chosen_frames], ignore_index=True), grid,
            pd.DataFrame(settings), inner)


def year_trials(cohort, year):
    real,held=split(cohort,year)
    ledgers, synth, rosters, support = [], [], [], []
    partitions={}
    for role,years in [("calibration",(year-4,year-3)),("tuning",(year-2,year-1))]:
        rows=[]
        for date in years:
            inner,valid=split(real,date)
            assert set(inner.disasterNumber)|set(valid.disasterNumber) <= set(real.disasterNumber)
            key=f"{year}/{role}/{date}"
            pred,s,registry,roster=raw_fit(inner,valid,key)
            rows.append(pred)
            ledgers+=registry
            synth+=s
            rosters.append(roster)
            support.extend({"fold":key,**line} for line in report_support(inner))
        partitions[role]=pd.concat(rows,ignore_index=True)
    calibrating=partitions["calibration"]
    tuning=partitions["tuning"]
    assert not set(calibrating.disasterNumber)&set(tuning.disasterNumber)
    assert not set(calibrating.disasterNumber)&set(held.disasterNumber)
    key=f"{year}/outer/{year}"
    pred,s,registry,roster=raw_fit(real,held,key)
    ledgers+=registry
    synth+=s
    rosters.append(roster)
    support.extend({"fold":key,**row} for row in report_support(real))
    predictions, grid, selections, inner_predictions = outer_policy(real, pred, calibrating, tuning, year)
    # Omit the largest eligible tail episode, identified without the held-out outcomes.
    extreme=real[real.target.ge(50e6)].sort_values("target",ascending=False).iloc[0]
    omitted=extreme.event_group
    sensitivity_configs=[("real",1),("weight",5),("random",5),("smoter",5),("smogn",5),("hybrid",5)]
    withheld=real[real.event_group.ne(omitted)]
    skey=f"{year}/sensitivity/{year}"
    sen,ss,sl,sr=raw_fit(withheld,held,skey,configs=sensitivity_configs,seed_namespace=key)
    sen["omission_reason"]="largest_training_tail_episode"
    sen["omitted_episode"]=omitted
    sen["omitted_peak_target_usd"]=extreme.target
    rosters.append(sr)
    ledgers+=sl
    synth+=ss
    support.extend({"fold":skey,**line} for line in report_support(withheld))
    donor_source=[d for d in s if d.trial_id.iloc[0]==f"{key}|smoter|5"]
    if donor_source:
        donors=donor_source[0]
        count=pd.concat([donors.episode_a,donors.episode_b]).value_counts()
        used=int(min(count[count.eq(count.max())].index))
        withheld=real[real.event_group.ne(used)]
        dkey=f"{year}/sensitivity_donor/{year}"
        dp,ds,dl,dr=raw_fit(withheld,held,dkey,configs=[("real",1),("smoter",5),("smogn",5),("hybrid",5)],seed_namespace=key)
        dp["omission_reason"]="most_used_smoter_training_donor"
        dp["omitted_episode"]=used
        dp["omitted_peak_target_usd"]=real.loc[real.event_group.eq(used),"target"].max()
        sen=pd.concat([sen,dp],ignore_index=True)
        rosters.append(dr)
        ledgers+=dl
        synth+=ds
        support.extend({"fold":dkey,**line} for line in report_support(withheld))
    synthetic = pd.concat([rows.astype({"target": float}) for rows in synth], ignore_index=True)
    return {"predictions":predictions,"tuning_grid":grid,"selections":selections,"sensitivity":sen,
            "synthetic":synthetic,
            "registry":pd.DataFrame(ledgers),"membership":pd.concat(rosters,ignore_index=True),
            "support":pd.DataFrame(support),"inner_predictions":inner_predictions}


def risk_rows(d,cols):
    p=d[cols].to_numpy()
    y=tail_mask(d)
    risk=p[:,3:].sum(axis=1)
    alert=risk>=REVIEW
    ece=0.
    for left,right in zip([0,.025,.05,.10,.25,.50],[.025,.05,.10,.25,.50,1.01],strict=True):
        ind=(left<=risk)&(risk<right)
        if ind.any():
            ece+=ind.mean()*abs(risk[ind].mean()-y[ind].mean())
    return {"tail_average_precision":average_precision_score(y,risk) if y.any() else np.nan,
            "500m_plus_average_precision":average_precision_score(d.actual_band.eq(5),p[:,5]) if d.actual_band.eq(5).any() else np.nan,
            "500m_plus_brier":float(np.mean((p[:,5]-d.actual_band.eq(5).to_numpy())**2)),
            "tail_brier":float(np.mean((risk-y)**2)),"tail_calibration_error":ece,
            "mean_tail_probability":float(risk.mean()),"alerts":int(alert.sum()),
            "true_tail_alerts":int((alert&y).sum()),"false_tail_alerts":int((alert&~y).sum())}


def metric_tables(pred):
    scores,bands,intervals,reliability=[],[],[],[]
    for variant, g in pred.groupby("variant"):
        y=tail_mask(g)
        for scope, cols in [("raw",[f"p{i}" for i in LABELS]),("calibrated",[f"cal_p{i}" for i in LABELS])]:
            amount=g.amount.to_numpy()
            calc=score(g,amount,g[cols].to_numpy())
            scores.append({"variant":variant,"probability_scope":scope,
                        **calc,**risk_rows(g,cols),"tail_n":int(y.sum()),
                        "tail_mae_usd":abs(g.loc[y,"target"]-g.loc[y,"amount"]).mean(),
                        "historical_tail_episode_groups":g.loc[y,"event_group"].nunique()})
            p=g[cols].to_numpy()
            classes=p.argmax(axis=1)
            for band in LABELS:
                actual=g.actual_band.eq(band)
                bands.append({"variant":variant,"probability_scope":scope,"band":BANDS[band],
                              "real_test_n":int(actual.sum()),"recall":float(np.mean(classes[actual]==band)),
                              "predictions_in_band":int((classes==band).sum())})
            for lo,hi in zip([0,.025,.05,.1,.25,.5],[.025,.05,.10,.25,.5,1.01],strict=True):
                risk=p[:,3:].sum(axis=1)
                index=(lo<=risk)&(risk<hi)
                if index.any():
                    reliability.append({"variant":variant,"probability_scope":scope,"bin_lower":lo,"bin_upper":hi,
                         "n":int(index.sum()),"predicted_risk":risk[index].mean(),"observed_event_rate":y[index].mean()})
        for level in (80,90):
            for subset,frame in [("all",g),("at_least_50m",g[y]),*[(BANDS[band],g[g.actual_band.eq(band)]) for band in TAIL_BANDS]]:
                hits=frame.target.between(frame[f"lower{level}"],frame[f"upper{level}"])
                widths=frame[f"upper{level}"]-frame[f"lower{level}"]
                intervals.append({"variant":variant,"level":level,"subset":subset,"n":len(frame),
                                  "covered":int(hits.sum()),"coverage":hits.mean(),
                                  "median_width_usd":widths.median() if len(widths) else np.nan,
                                  "mean_width_usd":widths.mean()})
    return {"metrics":pd.DataFrame(scores),"per_band":pd.DataFrame(bands),
            "intervals":pd.DataFrame(intervals),"reliability":pd.DataFrame(reliability)}


def paired(pred):
    base=pred[pred.variant.eq(BASELINE)].sort_values("disasterNumber").reset_index(drop=True)
    groups=[frame.index.to_numpy() for _,frame in base.groupby("event_group")]
    rng=np.random.default_rng(BOOT_SEED)
    tail=tail_mask(base)
    draws=[]
    for keys in rng.integers(0,len(groups),size=(800,len(groups))):
        idx=np.concatenate([groups[i] for i in keys])
        if tail[idx].any() and not tail[idx].all():
            draws.append(idx)
    metric_names = (
        "accuracy", "tail_average_precision", "tail_brier", "mae", "tail_mae",
        "raw_tail_average_precision", "raw_tail_brier", "coverage90", "tail_coverage90",
        "mean_width90",
    )

    def summaries(d, indices):
        p = d[[f"cal_p{i}" for i in LABELS]].to_numpy()
        y = tail_mask(d)
        risk = p[:,3:].sum(axis=1)
        raw_risk = d[["p3", "p4", "p5"]].sum(axis=1).to_numpy()
        correct = p.argmax(axis=1) == d.actual_band.to_numpy()
        errors = (d.target-d.amount).abs().to_numpy()
        covered = d.target.between(d.lower90, d.upper90).to_numpy()
        width = (d.upper90-d.lower90).to_numpy()
        def measure(idx):
            use = idx[y[idx]]
            return [correct[idx].mean(), average_precision_score(y[idx], risk[idx]),
                    np.mean((y[idx]-risk[idx])**2), errors[idx].mean(), errors[use].mean(),
                    average_precision_score(y[idx], raw_risk[idx]),
                    np.mean((y[idx]-raw_risk[idx])**2), covered[idx].mean(), covered[use].mean(),
                    width[idx].mean()]
        return np.array([measure(idx) for idx in indices])

    reference = summaries(base, [np.arange(len(base)), *draws])
    rows = []
    for variant, g in pred.groupby("variant"):
        if not variant.startswith("nested_") or variant == "nested_real":
            continue
        g = g.sort_values("disasterNumber").reset_index(drop=True)
        assert g.disasterNumber.equals(base.disasterNumber)
        paired_delta = summaries(g, [np.arange(len(base)), *draws]) - reference
        bounds = np.quantile(paired_delta[1:], [.025, .975], axis=0)
        for i, metric in enumerate(metric_names):
            rows.append({"variant": variant, "metric": metric, "difference": paired_delta[0, i],
                         "p025": bounds[0, i], "p975": bounds[1, i], "episodes": len(groups),
                         "draws": len(draws)})
    return pd.DataFrame(rows)


def sensitivity_table(sensitivity,pred):
    results=[]
    for (config,year,reason),g in sensitivity.groupby(["config","fyDeclared","omission_reason"]):
        a=pred[(pred.variant.eq(config))&pred.fyDeclared.eq(year)].set_index("disasterNumber").loc[g.disasterNumber]
        p=["p3","p4","p5"]
        high=tail_mask(g)
        amount_delta=abs(g.amount.to_numpy()-a.amount.to_numpy())
        prob_delta=abs(g[p].sum(axis=1).to_numpy()-a[p].sum(axis=1).to_numpy())
        results.append({"variant":config,"year":year,"omission_reason":reason,"excluded_episode":g.omitted_episode.iloc[0],
                        "excluded_episode_peak_usd":g.omitted_peak_target_usd.iloc[0],
                        "real_held_n":len(g),"tail_real_held_n":int(high.sum()),
                        "mean_abs_prediction_change_usd":amount_delta.mean(),
                        "tail_mean_abs_prediction_change_usd":amount_delta[high].mean() if high.any() else np.nan,
                        "mean_abs_tail_probability_change":prob_delta.mean()})
    return pd.DataFrame(results)


def generator_distribution(synthetic):
    rows=[]
    if synthetic.empty:
        return pd.DataFrame()
    for (method,fold,band),d in synthetic.groupby(["method","fold","actual_band"]):
        source_counts=pd.concat([d.source_a,d.source_b]).value_counts()
        corr=d[["declaration_delay_days","target"]].corr(method="spearman").iloc[0,1]
        rows.append({"method":method,"fold":fold,"band":band,"generated_rows":len(d),
                     "unique_source_episodes":len(set(d.episode_a)|set(d.episode_b)),
                     "distinct_generated_feature_rows":len(d[PREDICTORS].drop_duplicates()),
                     "reused_real_feature_rows":int(d.feature_duplicate.sum()),
                     "reused_real_row_and_amount":int(d.joint_duplicate.sum()),
                     "largest_source_endpoint_share":source_counts.max()/(2*len(d)),
                     "median_nearest_distance":d.min_distance_to_real_stratum.median(),
                     "synthetic_delay_min":d.declaration_delay_days.min(),
                     "synthetic_delay_median":d.declaration_delay_days.median(),
                     "synthetic_delay_max":d.declaration_delay_days.max(),
                     "synthetic_target_min":d.target.min(),"synthetic_target_median":d.target.dropna().median(),
                     "synthetic_target_max":d.target.max(),"synthetic_spearman_delay_target":corr})
    return pd.DataFrame(rows)


def write_tables(tables, output):
    for table, d in tables.items():
        if table=="synthetic":
            d=d.rename(columns={"target":"synthetic_mission_obligation_proxy_usd","actual_band":"synthetic_funding_band_index"})
        if table in ("predictions","inner_predictions","synthetic","membership","registry","sensitivity"):
            d.to_csv(output/f"{table}.csv.gz",index=False,compression={"method":"gzip","mtime":0})
        else:
            d.to_csv(output/f"{table}.csv",index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--cohort",type=Path,default=Path("research/results/hazards/cohort_features.csv"))
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    cohort=read_table(args.cohort)
    assert cohort.fyDeclared.max()==2024
    jobs=Parallel(n_jobs=args.workers, prefer="threads")(delayed(year_trials)(cohort,year) for year in YEARS)
    tables={key:pd.concat([trial[key] for trial in jobs],ignore_index=True) for key in jobs[0]}
    preds=tables["predictions"]
    assert tables["support"].state_hazard_strata_ctgan_gate.eq(0).all()
    assert preds.groupby("variant").disasterNumber.nunique().eq(376).all()
    assert not preds.duplicated(["variant","disasterNumber"]).any()
    assert set(preds.disasterNumber) == set(pd.read_csv(
        "research/results/forward/forward_predictions.csv", usecols=["disasterNumber"]
    ).disasterNumber)
    assert preds.groupby("variant").actual_band.apply(lambda x:x.isin(TAIL_BANDS).sum()).eq(13).all()
    tables.update(metric_tables(preds))
    print("Fitted all six outer origins; computing paired episode uncertainty",flush=True)
    tables["paired_episode_bootstrap"]=paired(preds)
    tables["sensitivity_summary"]=sensitivity_table(tables["sensitivity"],preds)
    tables["generator_distributions"]=generator_distribution(tables["synthetic"])
    write_tables(tables, args.output)
    (args.output/"study.json").write_text(json.dumps({"n_real_held":376,"test_years":list(YEARS),
         "high_band_real_held":13,"configs":len(SETTINGS),"real_calibration_only":True,
         "ctgan_research_support_floor":100,"ctgan_fit":False,"bootstrap_seed":BOOT_SEED},indent=2)+"\n")


if __name__=="__main__":
    main()
