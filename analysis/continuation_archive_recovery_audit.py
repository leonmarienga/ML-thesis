#!/usr/bin/env python3
"""Verify newly recovered originals and recompute frozen predictions exactly.

No retraining, new parameter selection, or independent holdout claim. Fiscal-year
block bootstrap is descriptive uncertainty for fixed historical predictions.
"""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
import pandas as pd
from shapely.geometry import shape
from sklearn.metrics import classification_report,confusion_matrix,mean_absolute_error,mean_squared_error,median_absolute_error,r2_score

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'research';REC=R/'recovered_artifacts';REC.mkdir(exist_ok=True)
OUT=R/'frozen_prediction_audit';OUT.mkdir(exist_ok=True)
BANDS=['0-100K','100K-1M','1M-50M','50-200M','200-500M','500M+']
EXPECTED={
 'phase49':('phase49-original.zip','aac7444581c593c4dd9970513611b2b7564a19f94ad5ca2d0ce9f3c92dcca19e',41581240598,13205946936),
 'phase49b':('phase49b-original.zip','1293b6e4b0c28634f52eada96d20a03a70df61bcf2fe122db76bedd9c1ef571c',41581789674,13205487776),
 'phase48':('phase48-original.zip','5ab2851283252b81dcd337f73574ca17cca67ee50f9379e0ae090c119254388d',41578898108,13204987935),
 'accepted':('accepted-original.zip','e0a683ae7f3196e560e920c44664672d01e7057f348bb8ef0472ea35fa351f20',41397678781,13118979999)}

def recover():
    rows=[]
    for key,(name,digest,run,artifact) in EXPECTED.items():
        src=ROOT/'work/recovery'/name
        if not src.exists():src=REC/name
        assert hashlib.sha256(src.read_bytes()).hexdigest()==digest,name
        if src!=REC/name:shutil.copy2(src,REC/name)
        if (ROOT/'work/recovery'/key).exists():shutil.copytree(ROOT/'work/recovery'/key,REC/key,dirs_exist_ok=True)
        rows.append(dict(name=name,sha256=digest,run_id=run,artifact_id=artifact,
                         source_url=f'https://github.com/leonmarienga/ML-thesis/actions/runs/{run}/artifacts/{artifact}',
                         recovery='GitHub connector download_workflow_artifact; SHA matches GitHub digest',
                         retrieved_date='2026-11-17'))
    pd.DataFrame(rows).to_csv(REC/'ARCHIVE_MANIFEST.csv',index=False)

def geometry_and_pixels():
    p49=REC/'phase49';p48=REC/'phase48';p49b=REC/'phase49b'
    roster=pd.read_csv(p49/'phase49_predeclared_36_disaster_county_sampling_roster.csv')
    original48=pd.read_csv(p48/'phase48_predeclared_36_disaster_county_sampling_roster.csv')
    assert len(roster)==187 and roster.disasterNumber.nunique()==36
    keys=['disasterNumber','county_fips','site_index']
    assert not roster.duplicated(keys).any()
    anchors=roster[roster.site_index==1].merge(original48,on=['disasterNumber','county_fips'],validate='one_to_one',suffixes=('_49','_48'))
    assert len(anchors)==63
    assert np.allclose(anchors.anchor_lon_49,anchors.anchor_lon_48,rtol=0,atol=1e-12)
    assert np.allclose(anchors.anchor_lat_49,anchors.anchor_lat_48,rtol=0,atol=1e-12)
    monthly=pd.read_csv(p49/'phase49_real_JRC_monthly_2022_24_county_samples.csv')
    correction=pd.read_csv(p49b/'phase49b_Alaska_monthly_source_window_correction.csv')
    monthly=pd.concat([monthly[monthly.disasterNumber!=4730],correction],ignore_index=True)
    pairs=pd.read_csv(p49/'phase49_paired_prior_year_surface_water_transitions.csv')
    fixpairs=pd.read_csv(p49b/'phase49b_Alaska_paired_source_window_correction.csv')
    pairs=pd.concat([pairs[pairs.disasterNumber!=4730],fixpairs],ignore_index=True)
    assert len(monthly)==561 and len(pairs)==374
    assert not monthly.duplicated(['sample_id','phase']).any()
    assert not pairs.duplicated(['sample_id','comparison_phase']).any()
    assert monthly.status.eq('read_ok').all() and pairs.status.eq('paired_read_ok').all()
    monthly.to_csv(REC/'phase49_corrected_561_months.csv',index=False)
    pairs.to_csv(REC/'phase49_corrected_374_pairs.csv',index=False)
    chosen=pd.read_csv(R/'phase50/frozen_67_site_roster.csv')
    merged=chosen.merge(roster,on=keys,validate='one_to_one',suffixes=('_now','_original'))
    assert len(merged)==67
    assert merged.event_month_now.eq(merged.event_month_original).all()
    assert np.allclose(merged.anchor_lon_now,merged.anchor_lon_original,rtol=0,atol=1e-12)
    assert np.allclose(merged.anchor_lat_now,merged.anchor_lat_original,rtol=0,atol=1e-12)
    original_geom=json.loads((p49/'phase49_county_windows_2023_2024.geojson').read_text())
    original_polys={f"{f['properties']['disasterNumber']}_{f['properties']['county_fips']}_{f['properties']['site_index']}":shape(f['geometry']) for f in original_geom['features']}
    reconstructed=json.loads((R/'phase50/reconstructed_all187_windows.geojson').read_text())
    distances=[]
    for f in reconstructed['features']:
        ident=f['properties']['sample_id']
        distance=original_polys[ident].hausdorff_distance(shape(f['geometry']))
        assert distance<1e-10,(ident,distance)
        distances.append(distance)
    gains=pd.read_csv(R/'phase50/historical_log_replication.csv')
    exact=pairs[pairs.comparison_phase=='event'][['sample_id','new_water_cells_prior_dry_event_water','both_observed_n','both_observed_frac_of_site']]
    check=gains.merge(exact,on='sample_id',validate='one_to_one')
    assert len(check)==67
    check['exact_integer_gain_matches']=check.event_gain_cells.eq(check.new_water_cells_prior_dry_event_water)
    assert check.exact_integer_gain_matches.all()
    source=pd.read_csv(R/'phase50/source_reads.csv')
    ev=source.merge(chosen[['sample_id','event_month']],on='sample_id',validate='many_to_one').query('month==event_month')
    old=monthly[monthly.phase=='event'][['sample_id','observation_fraction']]
    obs=ev.merge(old,on='sample_id',validate='one_to_one')
    assert len(obs)==67 and np.allclose(obs.observed_fraction,obs.observation_fraction,rtol=0,atol=1e-12)
    check.to_csv(OUT/'exact_original_pixel_replication_67.csv',index=False)
    return dict(original_sites=187,corrected_monthly_rows=561,corrected_comparison_rows=374,
                original_phase48_anchors_verified=63,phase50_integer_gain_matches=67,
                phase50_event_observability_matches=67,max_geometry_hausdorff_degrees=max(distances))

def monetary(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float)
    return dict(n=len(y),MAE_USD=float(mean_absolute_error(y,p)),RMSE_USD=float(np.sqrt(mean_squared_error(y,p))),
                MedAE_USD=float(median_absolute_error(y,p)),R2=float(r2_score(y,p)) if len(y)>1 and np.var(y)>0 else None,
                log_MAE=float(np.abs(np.log1p(y)-np.log1p(p)).mean()))

def predictions():
    x=pd.read_csv(REC/'accepted/all_912_immutable_overlay.csv')
    assert len(x)==912 and x.disasterNumber.is_unique
    assert (x.router_pred==x.actual_band).sum()==830
    master=pd.read_excel(ROOT/'master_openfema_40plus.xlsx')
    assert set(x.disasterNumber)==set(master.loc[master.incidentType!='Biological','disasterNumber'])
    metadata=master[['disasterNumber','state','fyDeclared','incidentType']].merge(x,on='disasterNumber',validate='one_to_one',suffixes=('_master','_original'))
    for field in ['state','fyDeclared','incidentType']:
        assert metadata[field+'_master'].eq(metadata[field+'_original']).all(),field
    target=master[['disasterNumber','totalObligatedFunding']].merge(x,on='disasterNumber',validate='one_to_one')
    assert len(target)==912 and np.allclose(target.totalObligatedFunding.clip(lower=0),target.target,rtol=0,atol=.01)
    variants=['frozen_stack_prediction','accepted_low_band_prediction','challenger_stack_prediction']
    amount=[]
    for variant in variants:
        amount.append(dict(variant=variant,scope='overall',group='all912',**monetary(x.target,x[variant])))
        for field in ['actual_band','router_pred','incidentType','fyDeclared']:
            for label,g in x.groupby(field):amount.append(dict(variant=variant,scope=field,group=str(label),**monetary(g.target,g[variant])))
        for label,g in [('zero',x[x.target==0]),('positive',x[x.target>0]),('rare_ge50M',x[x.target>=50e6])]:
            amount.append(dict(variant=variant,scope='special',group=label,**monetary(g.target,g[variant])))
    pd.DataFrame(amount).to_csv(OUT/'monetary_metrics.csv',index=False)
    final=monetary(x.target,x.challenger_stack_prediction)
    assert abs(final['MAE_USD']-5014819.011614494)<.01
    assert abs(final['R2']-.8867765710842239)<1e-9
    report=classification_report(x.actual_band,x.router_pred,labels=BANDS,output_dict=True,zero_division=0)
    pd.DataFrame(report).T.to_csv(OUT/'router_classification_report.csv')
    cm=pd.DataFrame(confusion_matrix(x.actual_band,x.router_pred,labels=BANDS),index=BANDS,columns=BANDS)
    cm.index.name='actual_band';cm.to_csv(OUT/'router_confusion_matrix.csv')
    subsets=[]
    for field in ['incidentType','fyDeclared']:
        for label,g in x.groupby(field):
            rpt=classification_report(g.actual_band,g.router_pred,labels=BANDS,output_dict=True,zero_division=0)
            present=g.actual_band.unique()
            subsets.append(dict(scope=field,group=str(label),n=len(g),accuracy=float(g.actual_band.eq(g.router_pred).mean()),
                balanced_accuracy_present_bands=float(np.mean([rpt[b]['recall'] for b in present])),
                macro_precision_six_bands=rpt['macro avg']['precision'],macro_recall_six_bands=rpt['macro avg']['recall'],
                macro_f1_six_bands=rpt['macro avg']['f1-score']))
    pd.DataFrame(subsets).to_csv(OUT/'router_hazard_year_metrics.csv',index=False)
    influence=x.copy();influence['final_absolute_error']=abs(x.target-x.challenger_stack_prediction)
    influence['original_absolute_error']=abs(x.target-x.frozen_stack_prediction)
    influence['accepted_minus_original_absolute_error']=influence.final_absolute_error-influence.original_absolute_error
    influence.sort_values('final_absolute_error',ascending=False).to_csv(OUT/'case_influence.csv',index=False)
    rng=np.random.default_rng(20261117);years=sorted(x.fyDeclared.unique())
    groups={y:influence[influence.fyDeclared==y] for y in years};deltas=[]
    for _ in range(2000):
        draws=rng.choice(years,len(years),replace=True)
        total=sum(len(groups[y]) for y in draws)
        deltas.append(sum(groups[y].accepted_minus_original_absolute_error.sum() for y in draws)/total)
    np.save(OUT/'descriptive_fiscal_year_bootstrap_deltas.npy',np.array(deltas))
    bootstrap=dict(repetitions=2000,seed=20261117,clusters='fiscal years, not independently verified physical events',
        metric='accepted minus original MAE_USD',observed=float(influence.accepted_minus_original_absolute_error.mean()),
        percentile95_lower=float(np.quantile(deltas,.025)),percentile95_upper=float(np.quantile(deltas,.975)),
        inference='descriptive fixed historical predictions; repeated model development; no independent confirmation or prospective validity')
    (OUT/'descriptive_difference_uncertainty.json').write_text(json.dumps(bootstrap,indent=2))
    result=dict(final=final,router_correct=830,router_total=912,router_accuracy=830/912,
        balanced_accuracy=report['macro avg']['recall'],macro_precision=report['macro avg']['precision'],
        macro_recall=report['macro avg']['recall'],macro_f1=report['macro avg']['f1-score'],
        validation='fixed historical LFYO; not independent or prospective',trained_models=0,
        bootstrap=bootstrap)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2));return result

def main():
    recover();geo=geometry_and_pixels();pred=predictions()
    summary=dict(archive_recovery='RESOLVED through GitHub connector; CLI403 remains',geometry=geo,predictions=pred)
    (REC/'RECOVERY_AUDIT.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
