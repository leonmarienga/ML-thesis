#!/usr/bin/env python3
"""Human-review demonstration. No invented historical predictions or need.

The optional exact accepted artifact is verified before being displayed.
Without it, the prediction layer explicitly abstains. Allocation uses separately
labelled synthetic needs, not the FEMA historical accounting target.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def accepted_predictions(path):
    x=pd.read_csv(path)
    required=['disasterNumber','target','router_pred','actual_band','challenger_stack_prediction']
    if any(c not in x for c in required):raise ValueError('Exact accepted artifact schema required')
    assert len(x)==912 and x.disasterNumber.is_unique
    assert int((x.router_pred==x.actual_band).sum())==830
    assert abs(np.abs(x.target-x.challenger_stack_prediction).mean()-5014819.011614494)<.01
    return x

def main():
    p=argparse.ArgumentParser();p.add_argument('--accepted-predictions',type=Path)
    p.add_argument('--out',type=Path,default=ROOT/'work/decision_support_demo.json');args=p.parse_args()
    source=pd.read_csv(ROOT/'research/phase50/source_reads.csv')
    real=source[source.sample_id=='4722_27055_3'].copy()
    scenario=pd.read_csv(ROOT/'research/allocation/synthetic_scenarios.csv').query('episode==0')
    alloc=pd.read_csv(ROOT/'research/allocation/simulation_allocations.csv').query(
        'episode==0 and budget_ratio==.7 and setting=="sequential" and policy=="forecast_reserve"')
    report=dict(
        evidence=dict(source='actual JRC monthly window records',sample_id='4722_27055_3',
            records=real[['month','observed_fraction','water_pixels','source_url']].to_dict('records'),
            meaning='observed/derived monthly surface-water classification; not observed damage or disaster-wide flood extent'),
        historical_funding=dict(status='ABSTAIN: exact per-row accepted archive unavailable in this checkout',
            aggregate_reported_benchmark={'n':912,'router_correct':830,'MAE_USD':5014819.011614494},
            target='historical obligations, not humanitarian need',validation='LFYO retrospective, future years may enter training'),
        uncertainty=dict(status='human review required',reason='catastrophic calibration support insufficient; no valid event-specific interval recovered'),
        allocation=dict(label='SYNTHETIC DEMONSTRATION ONLY',episode=0,
            policy='illustrative forecast reserve; not a validated operational recommendation',
            needs=scenario.to_dict('records'),allocations=alloc.to_dict('records')),
        human_review=['validate event attribution and map observability',
            'obtain independently justified real need and vulnerability evidence',
            'approve fairness priorities and budget scenario before any real allocation',
            'verify publication timing before claiming declaration-time predictions'],
        action='Research evidence report only; final decisions remain with accountable human officials')
    if args.accepted_predictions:
        x=accepted_predictions(args.accepted_predictions)
        report['historical_funding']['status']='exact frozen artifact verified; historical case outputs available'
        report['historical_funding']['verified_rows']=len(x)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(args.out)

if __name__=='__main__':main()
