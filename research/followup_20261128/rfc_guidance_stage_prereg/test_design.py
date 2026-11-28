"""Synthetic fixtures only. No truth downloaded, loaded from disk or scored by tests."""
import copy,json,shutil,tempfile
from collections import Counter
from fractions import Fraction
from urllib.parse import parse_qs,urlparse
from build_plan import plan,ROOT
from outcome_rules import classify,describe

p,qs=plan()
assert len(p)==11 and qs['truth_requests_executed']==0
assert Counter(x['episode'] for x in p)=={'KY2022':4,'OH2024':2,'HELENE2024':5}
assert qs['guidance_pair_count']==11 and qs['strict_public_eligible']==0
for pair,q in zip(p,qs['future_requests']):
    params=parse_qs(urlparse(q['would_get']).query)
    assert params['datetime']==[pair['target_utc']] and params['monitoring_location_id']==[pair['usgs_id']]
    assert params['parameter_code']==['00065'] and q['expected_time_series_id']==pair['series']
with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
    from pathlib import Path
    b=Path(tmp)
    shutil.copytree(ROOT/'source',b/'source')
    for n in ['SOURCE_SHA256SUMS','design.json']:
        shutil.copy2(ROOT/n,b/n)
    registry=b/'design.json'
    d=json.loads(registry.read_text())
    d['cohort']['KY2022'].pop()
    registry.write_text(json.dumps(d))
    try: plan(b)
    except ValueError as e: assert 'eligible tuple' in str(e)
    else: raise AssertionError('Removed time did not fail')
    shutil.copy2(ROOT/'design.json',registry)
    source=b/'source'/'fixed-lead-dispositions.csv'
    source.write_text(source.read_text().replace('8.6,HGIFF','8.7,HGIFF',1))
    try: plan(b)
    except ValueError as e: assert 'hash mismatch' in str(e)
    else: raise AssertionError('Changed source did not fail')

def fictitious(pair,value='10.0',**changes):
    x=dict(monitoring_location_id=pair['usgs_id'],time_series_id=pair['series'],parameter_code='00065',
        statistic_id='00011',unit_of_measure='ft',time=pair['target_utc'],approval_status='Approved',qualifier=None,
        value=value)
    return dict(type='FeatureCollection',numberReturned=1,features=[dict(properties=dict(x,**changes))],links=[])

v='verified_documented_stage_tie'
sample=p[0]
assert classify(sample,fictitious(sample),v)[0]=='EXACT_APPROVED_PAIR'
assert classify(sample,fictitious(sample),'pending')[0]=='WITHHELD_DATUM_OR_SITE_NOT_VERIFIED'
assert classify(sample,None,v)[0]=='NO_RESPONSE_CAPTURED'
assert classify(sample,fictitious(sample,time="2022-07-28T12:01:00Z"),v)[0]=='NONEXACT_TIME'
assert classify(sample,fictitious(sample,unit_of_measure='m'),v)[0]=='SERIES_UNIT_OR_PARAMETER_MISMATCH'
assert classify(sample,fictitious(sample,time_series_id='mismatched'),v)[0]=='SERIES_UNIT_OR_PARAMETER_MISMATCH'
assert classify(sample,fictitious(sample,qualifier='estimated'),v)[0]=='QUALIFIED_OR_MISSING_QUALIFIER_FIELD'
assert classify(sample,fictitious(sample,approval_status='Provisional'),v)[0]=='NOT_APPROVED_OR_MISSING_STATUS'
assert classify(sample,fictitious(sample,value='NaN'),v)[0]=='NONNUMERIC_OR_NONFINITE'
e=fictitious(sample);e['links']=[dict(rel='next')]
assert classify(sample,e,v)[0]=='PAGINATED_OR_AMBIGUOUS'
e=fictitious(sample);e['numberReturned']=2;e['features']*=2
assert classify(sample,e,v)[0]=='MULTIPLE_FEATURES'

# Synthetic fixture: constant +1, -2, +4 ft prediction-minus-reading error by episode.
# The fictitious values generated below are only in memory and are not historical truth.
offset={'KY2022':1,'OH2024':-2,'HELENE2024':4}
from decimal import Decimal
envelopes={x['pair_id']:fictitious(x,str(Decimal(x['stage_guidance_ft'])-offset[x['episode']])) for x in p}
gates={k:v for k in offset}
rows,s=describe(p,envelopes,gates)
assert len(rows)==11 and s['primary_status']=='complete'
assert s['macro_mae_ft']==Fraction(7,3)
assert s['macro_bias_ft']==1
assert s['pooled_mae_ft']==Fraction(28,11)
del envelopes[sample['pair_id']]
_,s=describe(p,envelopes,gates)
assert s['primary_status']=='not_evaluable_incomplete'
assert s['macro_mae_ft'] is None and s['pooled_mae_ft'] is None
assert s['event_descriptives']['KY2022']['eligible']==3
print('Locked-corpus, no-network prospective plan and synthetic QC checks passed.')
