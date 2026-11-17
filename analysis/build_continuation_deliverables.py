#!/usr/bin/env python3
"""Build actual-result registries, figures and an executable research notebook.

Historical report values retain an explicit evidence level. New observational
and simulated outputs remain separate. This does not manufacture missing models.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess
import numpy as np
import pandas as pd
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'research';FIG=R/'figures';FIG.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'work/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def registries():
    log=pd.read_csv(R/'source_audit/source_retrieval_log.csv').fillna('')
    rows=[]
    for x in log.to_dict('records'):
        name=x['name']
        publisher=('NOAA NCEI' if name.startswith('noaa') or name.startswith('StormEvents') else
                   'NOAA NHC' if name.startswith('hurdat') else
                   'USGS' if name.startswith(('usgs','mtbs')) else
                   'NOAA NWS' if name.startswith('nws') else 'European Commission JRC / mirror Epoch')
        observation=('2024' if name.startswith('StormEvents') else '1851-2024, processed subset 2010-2024' if name.startswith('hurdat') else
                     'see source event time fields' if name.startswith('usgs_earthquake') else 'documentation only')
        units=('wind knots; hail inches; tornado path miles/yards per NOAA format' if name.startswith('StormEvents') else
               'wind knots; pressure hPa; lat/lon degrees; -999 missing' if name.startswith('hurdat') else
               'magnitude type specific; depth km; lat/lon degrees; epoch ms' if name.startswith('usgs_earthquake') else 'not applicable')
        rows.append(dict(dataset_id=name,publisher=publisher,dataset_name=name,version=name,
            official_source_url=x['url'],actual_download_url=x.get('actual_download_url',x['url']),
            retrieval_date=x['retrieved_utc'],observation_date=observation,
            publication_or_availability_date='not independently authenticated; revision filename retained where present',
            geographic_coverage='see original source; no automatic FEMA linkage',variables_units=units,
            original_record_identifiers='EVENT_ID/EPISODE_ID, storm_id, or USGS event id where applicable',
            sha256=x.get('sha256',''),license='official publisher reuse terms; exact dataset license not independently audited',
            transformation='see continuation_source_audit.py; untouched response checksummed',
            known_limitations='retrospective current source; candidate matching is not verified attribution',
            status=x['status'],classification='UNVERIFIED CANDIDATE for model inclusion' if x['status']=='retrieved' else 'MISSING'))
        snapshot=R/'source_audit/source_snapshots'/name
        if snapshot.exists():rows[-1]['durable_raw_snapshot']=str(snapshot.relative_to(ROOT))
    rows.append(dict(dataset_id='original_master_971',publisher='Original researcher; FEMA-derived dataset, raw target extraction not independently reconstructed',
        dataset_name='master_openfema_40plus.xlsx',version='immutable research-branch original',
        official_source_url='https://github.com/leonmarienga/ML-thesis',
        actual_download_url='https://github.com/leonmarienga/ML-thesis/blob/041afba8a10da9c2a962222e0a2644284a84fb7b/master_openfema_40plus.xlsx',
        retrieval_date='git checkout 2026-11-17',observation_date='FY2010-2024',
        publication_or_availability_date='original extraction timestamp unknown',geographic_coverage='971 FEMA disaster identifiers',
        variables_units='45 original columns; obligation USD; dates and administrative/derived features',
        original_record_identifiers='disasterNumber',sha256=sha(ROOT/'master_openfema_40plus.xlsx'),
        license='researcher-provided; upstream reuse terms not comprehensively audited',
        transformation='frozen original; one negative target clamped only by original band convention, workbook unchanged',
        known_limitations='raw target and contemporaneous feature provenance incomplete; DR4816 incident-date discrepancy',
        status='preserved',classification='original frozen benchmark; not independently certified prospective data'))
    for x in pd.read_csv(R/'phase50/source_reads.csv').to_dict('records'):
        rows.append(dict(dataset_id=x['sample_id']+'_'+x['month'],publisher='European Commission JRC; Epoch Source Cooperative mirror',
            dataset_name='Global Surface Water Monthly History',version='v1.5 Landsat C2 2022-2024',
            official_source_url='https://global-surface-water.appspot.com/download',actual_download_url=x['source_url'],
            retrieval_date=x['retrieval_date_utc'],observation_date=x['month'],publication_or_availability_date='unknown; not T0 certified',
            geographic_coverage=x['sample_id'],variables_units='class 0 unobserved, 1 dry, 2 water; geodesic km2',
            original_record_identifiers=x['sample_id']+'; monthly tile URL',sha256=x['cached_pixel_sha256'],
            checksum_scope='saved cropped NPZ, not the complete remote TIFF',license='Copernicus full/open data; attribution EC JRC/Google; mirror terms not separately audited',
            transformation='fixed county-clipped window; shared observable pixels; geodesic row areas',
            known_limitations='monthly classifier, not validated event inundation; no inferential probability sample',
            status=x['status'],classification='VERIFIED DERIVED MEASUREMENT of classified surface water; not flood truth'))
    extra=[('Census2023_counties','US Census Bureau','https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip',
            json.loads((R/'phase50/summary.json').read_text())['provenance']['census_sha256'],'generalized county geometry; official source downloaded, reproducible URL'),
           ('FEMA36_current_designations','FEMA','https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries',
            sha(R/'phase50/source_snapshots/fema36_current_designations.json'),'current event metadata; no authenticated historical version; raw IDs saved')]
    for name,publisher,url,digest,lim in extra:
        rows.append(dict(dataset_id=name,publisher=publisher,dataset_name=name,version='snapshot retrieved 2026-11-17',
            official_source_url=url,actual_download_url=url,retrieval_date='2026-11-17',sha256=digest,
            observation_date='county vintage2023 / event dates in raw FEMA snapshot',publication_or_availability_date='unknown',
            geographic_coverage='US counties / 36 FEMA floods',variables_units='geometry degrees / administrative dates and IDs',
            original_record_identifiers='GEOID / FEMA id',license='US government source; publisher reuse conditions apply',
            transformation='committed geographic algorithm and log-key reconstruction',known_limitations=lim,
            status='retrieved',classification='VERIFIED DERIVED MEASUREMENT / administrative metadata'))
    for x in pd.read_csv(R/'flood_validation/nasa_mn_render_source_log.csv').to_dict('records'):
        rows.append(dict(dataset_id='NASA_'+x['sample_id']+'_'+x['observation_date'],publisher='NASA LANCE / GIBS',
            dataset_name=x['layer'],version='current rendered historical layer; generation version unauthenticated',
            official_source_url='https://lance.modaps.eosdis.nasa.gov/flood/',actual_download_url=x.get('url',''),
            retrieval_date=x['retrieval_utc'],observation_date=x['observation_date'],publication_or_availability_date='historical publication time not authenticated',
            geographic_coverage=x['bbox'],variables_units='RGBA visualization, resampled from ~250m source',
            original_record_identifiers=x['sample_id']+'; WMS layer and date',sha256=x.get('sha256',''),
            license='NASA public data; publisher terms and attribution apply',transformation='WMS 1.1.1 bbox for fixed county polygon; visual evidence only',
            known_limitations='not raw QA/class arrays or surveyed flood labels; gray insufficient data; no dry inference from transparency',
            status=x['status'],classification='MODELLED ESTIMATE (satellite flood-product rendered map)'))
    pd.DataFrame(rows).to_csv(ROOT/'DATA_SOURCE_REGISTRY.csv',index=False)
    master=pd.read_excel(ROOT/'master_openfema_40plus.xlsx')
    target_derived={'totalObligatedFunding','avgObligationPerMission','medianObligationPerMission','maxObligationPerMission',
        'fundingPerMission','fundingPerAgency','fundingPerDay','minDateObligated','maxDateObligated',
        'daysToFirstObligationFromDeclaration','obligationSpanDays'}
    future={'incidentEndDate','incidentEndDate_dt','durationDays','durationClass','lastIAFilingDate',
        'missionAssignmentCount','uniqueAgencyCount','uniqueMaTypeCount','uniquePriorityCount',
        'responseComplexityScore','missionDensity','agencyDensity','eventScale'}
    manifest=[]
    for column in master.columns:
        issue=('TARGET / financial leakage: prohibit as predictor' if column in target_derived else
               'post-event/finalized feature: prohibit T0/T72 without authentic cutoff snapshot' if column in future else
               'identifier only; group events, not continuous numeric predictor' if column in {'disasterNumber','declarationRequestNumber','placeCode'} else
               'current metadata or engineered field: T0/T72 availability unverified')
        manifest.append(dict(feature=column,dataset='original master workbook',classification='UNVERIFIED CANDIDATE for prospective use',
            source_version=sha(ROOT/'master_openfema_40plus.xlsx'),units='original column convention; inspect original notebooks',
            event_linkage='FEMA disasterNumber; shared physical event grouping incomplete',T0_status=issue,T72_status=issue,
            retrospective_status='original benchmark frozen',missing_policy='do not fabricate',accepted_new_model=False))
    for field,dataset,kind,units in [('JRC gain/loss/net','phase50','VERIFIED DERIVED MEASUREMENT','km2 per100 common observed km2'),
        ('NOAA wind/hail/tornado','NOAA2024','UNVERIFIED CANDIDATE','knots/inches/EF/path miles/yards; measurements vs estimates distinguished in source'),
        ('NHC best-track wind/pressure','HURDAT2','MODELLED ESTIMATE','knots/hPa; -999 missing sentinel'),
        ('USGS magnitude/depth','ComCat','UNVERIFIED CANDIDATE for FEMA linkage','magnitude type, km'),
        ('MODIS rendered flood map','NASA GIBS','MODELLED ESTIMATE','RGBA rendered classes; no area computed'),
        ('population/buildings in verified footprint','none','MISSING','no exposure estimate accepted')]:
        manifest.append(dict(feature=field,dataset=dataset,classification=kind,units=units,
            event_linkage='not independently certified',T0_status='not certified',T72_status='not certified',
            retrospective_status='exploratory evidence only',missing_policy='preserve missing; never substitute zero',accepted_new_model=False))
    pd.DataFrame(manifest).to_csv(ROOT/'SOURCE_AND_FEATURE_MANIFEST.csv',index=False)
    model_rows=[
        dict(model_id='TrackA_original_amount_stack',status='FROZEN historical version',code='analysis/nonbio_final_830_accepted_amount_stack.py',
             validation='leave-fiscal-year-out; future years can enter training; repeatedly inspected',run_id=34796746603,
             artifact='nonbio-final-830-accepted-amount-stack; ZIP retrieval currently blocked',seed='original implementation; not refit'),
        dict(model_id='TrackA_final_5014819',status='FROZEN user-accepted reconstruction',code='analysis/reconstruct_accepted_912_exact.py; immutable overlay',
             validation='912 LFYO; historical log recovered, predictions not recovered this session',run_id=41397678781,
             artifact='exact-accepted-912-predictions; ZIP retrieval currently blocked',seed='no refit; four original artifacts'),
        dict(model_id='TrackB_Phase36_NOAA_density',status='EXPLORATORY; not replacement',code='analysis/phase36_noaa_report_intensity.py',
             validation='59 FY2021-24 earlier-year tests, extensively examined',run_id=41447664858,artifact='historical Actions artifact',seed='original source'),
        dict(model_id='TrackB_Phase37_physical',status='NEGATIVE exploratory result vs density',code='analysis/phase37_physical_rainflow_vs_reporting.py',
             validation='same 59 floods; retrospective physical features',run_id=41483049801,artifact='historical Actions artifact',seed='original source'),
        dict(model_id='TrackB_T72_NHC',status='EXPLORATORY; rare-tail failure',code='analysis/early_response_912_t0_t72_forward.py',
             validation='376 FY2019-24 earlier-year tests; contemporaneous feature versions unverified',run_id=41393614710,artifact='historical Actions artifact',seed='original source'),
        dict(model_id='TrackB_CQR',status='EXPLORATORY; extreme width, nine rare tests',code='analysis/cqr_track_b_fixed_split.py',
             validation='536 train pre2019,68 calibration2019,308 test2020-24; not Track A wrapper',run_id=41397857911,artifact='historical Actions artifact',seed='original source')]
    for row in model_rows:
        row.update(continuation_code_checkpoint='5a1907e5f46b629b6c30bee148fd3412957bdacf',
                   source_baseline_commit='041afba8a10da9c2a962222e0a2644284a84fb7b',
                   new_training_performed=False,artifact_recovery='source/log/report retained; exact binary recovery incomplete')
    pd.DataFrame(model_rows).to_csv(ROOT/'MODEL_REGISTRY.csv',index=False)
    original=(R/'track_a_recovered_log_evidence.txt').read_text()
    report=json.loads(original.split('BEGIN_IMMUTABLE_OVERLAY_RESULTS\n')[1].split('\nEND_IMMUTABLE_OVERLAY_RESULTS')[0])
    metrics=[]
    for variant,stats in report['overall'].items():
        for metric,value in stats.items():
            metrics.append(dict(study='TrackA',variant=variant,metric=metric,value=value,n=912,
                evidence='RECOVERED HISTORICAL EXECUTION LOG; not per-row recomputed this session',
                validation='LFYO retrospective',source='research/track_a_recovered_log_evidence.txt',run_id=41397678781))
    metrics.append(dict(study='TrackA',variant='frozen_router',metric='accuracy',value=830/912,n=912,
        evidence='RECOVERED HISTORICAL LOG',validation='LFYO retrospective',source='research/track_a_recovered_log_evidence.txt',run_id=41397678781))
    a=pd.read_csv(R/'phase50/aggregate_temporal_comparisons.csv')
    for row in a.to_dict('records'):
        if row['denominator']=='common_pixels' and row['threshold']==.3:
            for m in ('event_gain_per100','placebo_gain_per100','delta_gain_per100'):
                metrics.append(dict(study='Phase50',variant=row['scheme']+'_'+row['group'],metric=m,value=row[m],n=row['cases'],
                    evidence='EXECUTED THIS CONTINUATION; actual JRC pixels',validation='exploratory fixed windows; no flood truth',
                    source='research/phase50/aggregate_temporal_comparisons.csv'))
    for row in pd.read_csv(R/'allocation/simulation_policy_summary.csv').to_dict('records'):
        for metric in ('unmet_need_units','fulfilled_fraction','min_disaster_service','budget_utilization'):
            metrics.append(dict(study='allocation',variant=f"{row['setting']}_{row['budget_ratio']}_{row['policy']}",
                metric=metric,value=row[metric],n=300,evidence='SIMULATED; no historical outcome evidence',
                validation='fixed generator and policies; common random numbers',source='research/allocation/simulation_policy_summary.csv'))
    pd.DataFrame(metrics).to_csv(ROOT/'FINAL_METRICS.csv',index=False)
    refs=[
        dict(id='Pekel2016',title='High-resolution mapping of global surface water and its long-term changes',authors='Pekel; Cottam; Gorelick; Belward',year=2016,
             url='https://www.nature.com/articles/nature20584',doi='10.1038/nature20584',verification='original publisher search metadata and JRC institutional record verified; full text not reviewed',
             claim_scope='original surface-water mapping method; not validation of 2022-24 flood labels'),
        dict(id='Romano2019',title='Conformalized Quantile Regression',authors='Romano; Patterson; Candes',year=2019,
             url='https://proceedings.neurips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html',doi='',
             verification='original NeurIPS proceedings abstract and authors verified',claim_scope='CQR method already exists; not proof of validity under temporal shift'),
        dict(id='NASA_LANCE',title='Flood Location Observation and Overview Dashboard',authors='NASA LANCE',year=2026,
             url='https://lance.modaps.eosdis.nasa.gov/flood/',doi='',verification='official page retrieved and checksummed',
             claim_scope='MODIS/VIIRS flood products and version chronology; current visualizations not ground truth'),
        dict(id='NWS2023',title='Mississippi River Flood of 2023',authors='National Weather Service La Crosse',year=2023,
             url='https://www.weather.gov/arx/flood2023',doi='',verification='official page retrieved; hydrographs and dated regional context read',
             claim_scope='regional April-May2023 flooding and gauge-stage context; not precise fixed-window footprints'),
        dict(id='Slayback2023',title='Update on LANCE Flood Product (MCDWD)',authors='Slayback; Davies; Devadiga; Landis; Shrestha',year=2023,
             url='https://www.ssec.wisc.edu/meetings/wp-content/uploads/sites/33/2023/06/FloodUpdate_Slayback_LANCEUWG_June2023.pdf',doi='',
             verification='original meeting PDF downloaded; title, presenter affiliations, date and text reviewed',
             claim_scope='2023 sensor product and threshold/masking limitations, Mississippi dated maps'),
        dict(id='ReliefDSS_prior_art',title='A decision support tool for allocating temporary-disaster-response facilities',authors='not independently verified',year='',
             url='https://www.sciencedirect.com/science/article/pii/S0167923619301745',doi='',
             verification='publisher search excerpt found; direct full page403; incomplete reference, exclude formal bibliography pending verification',
             claim_scope='novelty search lead only; no detailed performance claims')]
    pd.DataFrame(refs).assign(verification_date='2026-11-17').to_csv(ROOT/'REFERENCE_REGISTRY.csv',index=False)
    for name in ('SOURCE_AND_FEATURE_MANIFEST.csv','FINAL_METRICS.csv'):
        (R/name).write_bytes((ROOT/name).read_bytes())

def figures():
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    h=pd.read_csv(R/'source_audit/hazard_inventory.csv').sort_values('records')
    fig,ax=plt.subplots(figsize=(9,7));ax.barh(h.incidentType,h.records,color='#317e9d')
    ax.set_xlabel('Original disaster records');ax.set_title('Actual hazard support in the 971-row workbook')
    fig.tight_layout();fig.savefig(FIG/'hazard_support.png',dpi=150);plt.close(fig)
    a=pd.read_csv(R/'phase50/aggregate_temporal_comparisons.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,scheme in zip(axes,['seasonal','interannual']):
        for group,label in [('primary_63_anchors','Primary anchors'),('sensitivity_67_with_MN_extras','Add four MN sites')]:
            z=a[(a.scheme==scheme)&(a.denominator=='common_pixels')&(a.group==group)]
            ax.plot(z.threshold,z.delta_gain_per100,marker='o',label=label)
        ax.axhline(0,color='gray',lw=.7);ax.set_title(scheme.capitalize());ax.set_xlabel('Minimum shared observable fraction')
        ax.set_ylabel('Event minus placebo gain\nkm² per100 observed km²');ax.legend(fontsize=8)
    fig.suptitle('Temporal water-change sensitivity; no flood-truth interpretation')
    fig.tight_layout();fig.savefig(FIG/'phase50_placebo_sensitivity.png',dpi=160);plt.close(fig)
    c=pd.read_csv(R/'phase50/case_temporal_comparisons.csv')
    z=c[(c.scheme=='seasonal')&(c.denominator=='common_pixels')&(c.threshold==.3)&(c.group=='primary_63_anchors')].sort_values('delta_gain_per100')
    fig,ax=plt.subplots(figsize=(10,4));ax.bar(z.disasterNumber.astype(str),z.delta_gain_per100,color='#317e9d')
    ax.axhline(0,color='black',lw=.6);ax.tick_params(axis='x',rotation=90);ax.set_ylabel('Paired gain difference per100 observed km²')
    ax.set_title('Primary case-level seasonal differences, same eligible sites in both arms')
    fig.tight_layout();fig.savefig(FIG/'phase50_case_differences.png',dpi=160);plt.close(fig)
    alloc=pd.read_csv(R/'allocation/simulation_policy_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,setting in zip(axes,['simultaneous','sequential']):
        z=alloc[(alloc.setting==setting)&(alloc.budget_ratio==.7)]
        for r in z.itertuples():
            label='upper-quantile weighting' if r.policy=='robust_upper_need' else r.policy.replace('_',' ')
            ax.scatter(r.fulfilled_fraction,r.min_disaster_service,s=45,label=label)
        ax.set_xlabel('Mean fulfilled fraction');ax.set_ylabel('Mean minimum disaster service');ax.set_title(setting.capitalize())
        ax.set_ylim(0,.52);ax.legend(loc='upper left',fontsize=7,framealpha=.9)
        ax.margins(x=.25)
    fig.suptitle('SYNTHETIC scenarios: efficiency and worst-served disaster, planned budget ratio 0.7')
    fig.tight_layout();fig.savefig(FIG/'simulation_efficiency_fairness.png',dpi=160);plt.close(fig)

def notebook():
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    md('# AI-Assisted Disaster Relief Decision Support: Reproducible Research Audit\n\n'
       'This notebook explains and recomputes the completed continuation. It preserves the historical Track A benchmark, '
       'executes the Phase50 observable-pixel analysis from saved real pixels, audits multi-hazard support, and evaluates '
       'explicitly synthetic allocation scenarios. It does **not** establish a fully validated operational multi-hazard system. '
       'Historical reports, newly executed observations, simulation and blocked work have separate evidence labels.')
    code("from pathlib import Path\nimport os, sys, json, hashlib, subprocess\nimport numpy as np, pandas as pd\nROOT = Path.cwd()\nif ROOT.name == 'research': ROOT = ROOT.parent\nassert (ROOT/'master_openfema_40plus.xlsx').exists()\nos.environ['MPLCONFIGDIR'] = str(ROOT/'work/matplotlib')\nimport matplotlib.pyplot as plt\nfrom IPython.display import display, Image, Markdown\nsys.path.insert(0, str(ROOT/'analysis'))\nR = ROOT/'research'\nprint('Python', sys.version.split()[0], 'repository', ROOT)")
    md('## 1. Preserve and inspect the original data\n\nThe workbook has one row per FEMA identifier. '
       'One negative net-obligation target is clamped to zero by the original band function. Bands are lower-inclusive, '
       'upper-exclusive: [0,100K), [100K,1M), [1M,50M), [50M,200M), [200M,500M), [500M,∞). '
       'These are accounting obligations, not independently observed needs. The original workbook is never rewritten.')
    code("master=pd.read_excel(ROOT/'master_openfema_40plus.xlsx')\nassert len(master)==971 and master.disasterNumber.is_unique\nassert (master.incidentType!='Biological').sum()==912\ndisplay(pd.read_csv(R/'source_audit/hazard_inventory.csv'))\nprint('Negative original targets:', (master.totalObligatedFunding<0).sum())\ndisplay(Image(filename=str(R/'figures/hazard_support.png')))")
    md('## 2. Historical benchmark evidence and validation boundary\n\nTrack A reports 830/912 routing, MAE $5,014,819.0116 '
       'and R²0.88677657. Its original execution log was recovered. Per-row artifact ZIP retrieval is blocked; this notebook '
       'does not pretend to recompute these aggregate values from recovered predictions. Leave-fiscal-year-out excludes the '
       'held year but may use later years. It is not strict forward prediction. Existing CQR models are separate Track B models. '
       'The original specialist sources also contradict the handoff’s simplified statement that all three upper bands use medians: '
       'the accepted reconstruction includes a 50–200M hybrid and retained upper-band specialists.')
    code("metrics=pd.read_csv(ROOT/'FINAL_METRICS.csv')\ndisplay(metrics.query('study == \"TrackA\"')[['variant','metric','value','evidence']])\noriginal=json.loads((R/'frozen_source_checksums.json').read_text())\nassert all(hashlib.sha256((ROOT/x['path']).read_bytes()).hexdigest()==x['sha256'] for x in original)\nprint('Original source files unchanged:',len(original))")
    md('## 3. Phase50 question and mathematics\n\nFor fixed window $s$, let $C_s$ be the pixels with water/dry classifications '
       'at every date in both comparison arms. Class0 is unobserved. For pixel geodesic area $a_i$, '
       '$G_e=\\sum_{i\\in C_s}a_i1(x_{e,0}=1,x_{e,1}=2)$, and likewise $G_p$ for the placebo. '
       'The primary rate difference is $100(G_e-G_p)/\\sum_{i\\in C_s}a_i$. Loss and net change are also saved. '
       'Seasonal comparison is previous-month→incident-month in the event year versus the prior year. '
       'For2024, interannual comparison is2023→2024 versus2022→2023 at the same calendar month. '
       'No2021 imagery is mixed with the2022–24 edition. Tests cover the January boundary even though the final cohort '
       'has no comparison requiring that exclusion. Primary anchors and four added Minnesota sites are separate analyses.')
    code("from phase50_temporal_placebo import main as phase50_main\nfrom unittest import TestLoader, TextTestRunner\nimport test_phase50_temporal_placebo\nresult=TextTestRunner(verbosity=1).run(TestLoader().loadTestsFromModule(test_phase50_temporal_placebo))\nassert result.wasSuccessful()\nprint(json.loads((R/'phase50/summary.json').read_text())['provenance']['official_vs_master_date_discrepancies'])")
    md('## 4. Recompute from saved real pixels without network\n\nThe saved cropped arrays include unobserved cells and exact '
       'window grids. Historical log checks are limited to reported observability and four-decimal area values, not original '
       'CSV bytes. The single DR4816 date mismatch was resolved against current official FEMA metadata and historical log '
       'replay. Floating-point geodesic reconstruction tolerates1e-9km² after CSV coordinate round-trips; integer counts '
       'and historical match flags remain identical.')
    code("destination=ROOT/'work/notebook-phase50-reproduction'\nrun=subprocess.run([sys.executable,str(ROOT/'analysis/phase50_temporal_placebo.py'),'--offline-inputs',str(R/'phase50'),'--out',str(destination)],capture_output=True,text=True,timeout=120)\nassert run.returncode==0,run.stderr[-2000:]\nfor name in ['site_temporal_comparisons.csv','aggregate_temporal_comparisons.csv','case_temporal_comparisons.csv','historical_log_replication.csv']:\n    pd.testing.assert_frame_equal(pd.read_csv(R/'phase50'/name),pd.read_csv(destination/name),check_exact=False,rtol=1e-9,atol=1e-9)\nprint('Offline reproduction passed for all tables.')")
    code("a=pd.read_csv(R/'phase50/aggregate_temporal_comparisons.csv')\ndisplay(a.query('denominator == \"common_pixels\" and threshold == .3')[['scheme','group','sites','cases','event_gain_per100','placebo_gain_per100','delta_gain_per100']])\ndisplay(Image(filename=str(R/'figures/phase50_placebo_sensitivity.png')))\ndisplay(Image(filename=str(R/'figures/phase50_case_differences.png')))")
    md('## 5. Geographic and independent-sensor evidence\n\nNASA MODIS renders were obtained for six Minnesota windows '
       'on April12, April26 and May10,2023, chosen from regional NWS flood timing before inspecting the renders. '
       'The official legend marks gray as insufficient data; most inland renders are gray. Two added Houston County '
       'windows show the product’s flood color. These are coarse independently sensed classifier maps, not surveyed labels '
       'or a certified FEMA footprint. No precision, recall, IoU or false-positive rate is computed. Render transparency is '
       'not a dry-land label. Raw product/QA acquisition and contemporaneous version validation remain unresolved.')
    code("display(pd.read_csv(R/'flood_validation/nasa_mn_render_source_log.csv')[['sample_id','observation_date','status','displayed_flood_color_inside_polygon','displayed_insufficient_data_fraction_inside_polygon']])\ndisplay(Image(filename=str(R/'figures/six_sites_dated_maps.png')))\ndisplay(Image(filename=str(R/'flood_validation/MODIS_Flood_H.png')))")
    md('## 6. Other hazards and source limitations\n\nThe actual NOAA2024 archive contains wind and hail magnitudes and '
       'tornado EF/path fields. Winter-storm records lack equivalent numeric snow/ice fields in this archive, so event '
       'counts cannot substitute for intensity. NHC best tracks are retrospective estimates; -999 means missing, not a '
       'physical pressure or wind measurement. Eight earthquake regional queries succeeded, but date/box proximity is '
       'not event attribution. MTBS documentation is accessible; its advertised map service failed. No standalone model '
       'is fit to the eight earthquake records or nonexistent drought population.')
    code("display(pd.read_csv(R/'source_audit/noaa2024_physical_field_availability.csv'))\ndisplay(pd.read_csv(R/'source_audit/earthquake_query_coverage.csv'))\ndisplay(pd.read_csv(ROOT/'SOURCE_AND_FEATURE_MANIFEST.csv')[['feature','classification','T0_status','accepted_new_model']])")
    md('## 7. Prediction and uncertainty synthesis\n\nPreviously reported Phase36 flood NOAA-density MAE was $355,021 '
       'against $537,057 FEMA-only on59 examined tests; Phase37 physical additions did not improve that density baseline. '
       'These are historical report values, not newly reproduced metrics. Existing chronological uncertainty experiments '
       'showed high pooled coverage while failing rare high-value disasters; CQR covered7/9 rare tests at90% nominal '
       'with mean interval width about$140.5M. Those tests are too few for reliable conditional coverage. No newly '
       'certified physical feature set or untouched cohort exists here, so no additional integration model is accepted.')
    code("display(pd.read_csv(ROOT/'MODEL_REGISTRY.csv')[['model_id','status','validation','run_id']])\nprint((R/'PROSPECTIVE_VALIDATION_PROTOCOL.md').read_text()[:2000])")
    md('## 8. Allocation under uncertainty: simulation only\n\nThe original RL notebook rewards historical funding '
       'agreement; that is not a humanitarian utility measure. This separate experiment generates synthetic needs, '
       'vulnerability and budget releases with fixed seed20261117,300 episodes,8 arrivals and3 budget ratios. '
       'It compares five simultaneous and three sequential policies. The LP maximizes weighted delivered units '
       'subject to allocation bounds and cumulative cash constraints. A hindsight need-aware LP supplies a regret '
       'upper bound with the same sequential cash constraints. Fairness floors are an illustrative governance choice. '
       'The synthetic generator and policy definitions are fixed; no tuning against the simulated outcomes is used.')
    code("run=subprocess.run([sys.executable,str(ROOT/'analysis/continuation_allocation_simulation.py')],capture_output=True,text=True,timeout=120)\nassert run.returncode==0,run.stderr\nprint(run.stdout)\nfor name in ['simulation_policy_summary.csv','simulation_episode_metrics.csv']:\n    pd.testing.assert_frame_equal(pd.read_csv(R/'allocation'/name),pd.read_csv(ROOT/'work/allocation'/name),check_exact=False,rtol=1e-9,atol=1e-9)\ndisplay(pd.read_csv(R/'allocation/simulation_policy_summary.csv').query('budget_ratio==.7'))\ndisplay(Image(filename=str(R/'figures/simulation_efficiency_fairness.png')))")
    md('## 9. Human decision-support demonstration and scientific conclusion\n\nThe demonstration identifies actual '
       'source evidence, withheld/unavailable case predictions, insufficient interval evidence and independently synthetic '
       'allocation examples. It abstains instead of fabricating a case funding estimate. Policy choices and final allocation '
       'remain with human decision-makers. Read the final conclusion for completed work, genuine blockers and exact next steps.')
    code("run=subprocess.run([sys.executable,str(ROOT/'analysis/research_decision_support_demo.py')],capture_output=True,text=True,timeout=60)\nassert run.returncode==0,run.stderr\ndemo=json.loads((ROOT/'work/decision_support_demo.json').read_text())\ndisplay(demo['historical_funding']);display(demo['uncertainty'])\ndisplay(Markdown((R/'FINAL_CONCLUSION.md').read_text()))")
    md('## 10. Provenance and references\n\nSource checksums apply to the exact saved bytes identified in each record. '
       'Cropped JRC array checksums do not identify entire remote tiles. Unknown release dates and licensing details '
       'remain explicitly unknown. References verified only at abstract/metadata level are labelled accordingly. '
       'A comprehensive novelty claim is not established.')
    code("display(pd.read_csv(ROOT/'REFERENCE_REGISTRY.csv'))\nprint('Source-registry records:',len(pd.read_csv(ROOT/'DATA_SOURCE_REGISTRY.csv')))\nprint('Current git checkpoint:',subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())")
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3 research','language':'python','name':'python3'},
                                               'language_info':{'name':'python','version':'3.12'}})
    nbf.write(nb,R/'REPRODUCIBLE_RESEARCH.ipynb')

if __name__=='__main__':
    registries();figures();notebook()
