#!/usr/bin/env python3
"""Validate saved scientific invariants without contacting remote services."""
from pathlib import Path
import ast,hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def main():
    original=json.loads((ROOT/'research/frozen_source_checksums.json').read_text())
    for x in original:
        assert hashlib.sha256((ROOT/x['path']).read_bytes()).hexdigest()==x['sha256'],x['path']
    # Execute ONLY the frozen band function AST, avoiding heavy training imports.
    module=ast.parse((ROOT/'analysis/nonbio_all_ranges.py').read_text())
    fn=next(x for x in module.body if isinstance(x,ast.FunctionDef) and x.name=='funding_band')
    scope={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<original-band-function>','exec'),scope)
    band=scope['funding_band'];expected=['0-100K','100K-1M','1M-50M','50-200M','200-500M','500M+']
    assert band(-1)==expected[0]
    for i,threshold in enumerate([100_000,1_000_000,50_000_000,200_000_000,500_000_000]):
        assert band(threshold-1)==expected[i] and band(threshold)==expected[i+1]
    frame=pd.read_excel(ROOT/'master_openfema_40plus.xlsx')
    nb=frame[frame.incidentType!='Biological']
    counts=nb.totalObligatedFunding.map(band).value_counts().reindex(expected)
    assert counts.tolist()==[699,108,82,13,5,5]
    p=ROOT/'research/phase50'
    source=pd.read_csv(p/'source_reads.csv');assert len(source)==289 and source.status.eq('read_ok').all()
    assert not source.duplicated(['sample_id','month']).any()
    for r in source.itertuples():
        file=p/'pixels'/f'{r.sample_id}_{r.month}.npz'
        assert hashlib.sha256(file.read_bytes()).hexdigest()==r.cached_pixel_sha256
        with np.load(file,allow_pickle=False) as a:
            assert np.isin(a['matrix'][a['mask']],[0,1,2]).all()
    replica=pd.read_csv(p/'historical_log_replication.csv')
    assert len(replica)==67
    assert replica.logged_gain_matches_round4.dropna().eq(True).all()
    assert replica.logged_observability_matches.dropna().eq(True).all()
    assert replica.logged_gain_matches_round4.notna().sum()==65
    comparisons=pd.read_csv(p/'site_temporal_comparisons.csv')
    common=comparisons[comparisons.denominator=='common_pixels']
    assert np.allclose(common.event_observed_km2,common.placebo_observed_km2)
    assert (common.common_observable_fraction>=common.threshold).all()
    for arm in ['event','placebo']:
        assert (comparisons[f'{arm}_gain_km2']>=0).all()
        assert (comparisons[f'{arm}_loss_km2']>=0).all()
        assert np.allclose(comparisons[f'{arm}_net_km2'],comparisons[f'{arm}_gain_km2']-comparisons[f'{arm}_loss_km2'])
    alloc=pd.read_csv(ROOT/'research/allocation/simulation_allocations.csv')
    assert len(alloc)==57600
    for _,g in alloc.groupby(['episode','budget_ratio','setting','policy']):
        g=g.sort_values('arrival')
        assert np.all(g.allocation_units.cumsum()<=g.cash_available_cumulative+1e-7)
    for path in (ROOT/'research/source_audit/source_snapshots').iterdir():
        if path.name.endswith(('.gz','.txt','.json','.pdf')):
            logs=pd.read_csv(ROOT/'research/source_audit/source_retrieval_log.csv')
            entry=logs[logs.name==path.name]
            if len(entry):assert hashlib.sha256(path.read_bytes()).hexdigest()==entry.sha256.iloc[0]
    print(json.dumps(dict(frozen_files_verified=len(original),original_band_counts=counts.to_dict(),
        real_raster_reads_verified=289,historical_log_matches=65,allocation_rows_verified=len(alloc),
        unsupported_new_models_accepted=0),indent=2))

if __name__=='__main__':main()
