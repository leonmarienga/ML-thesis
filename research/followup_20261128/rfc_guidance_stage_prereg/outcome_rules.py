"""Future offline scoring rules; intentionally no data acquisition or file entry point."""
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from build_plan import t, check

def classify(pair, envelope, gate):
    if gate != 'verified_documented_stage_tie':
        return 'WITHHELD_DATUM_OR_SITE_NOT_VERIFIED',None
    if envelope is None:
        return 'NO_RESPONSE_CAPTURED',None
    if not isinstance(envelope,dict) or envelope.get('type')!='FeatureCollection':
        return 'INVALID_RESPONSE',None
    fs=envelope.get('features')
    links=envelope.get('links')
    if not isinstance(fs,list) or not isinstance(links,list) or type(envelope.get('numberReturned')) is not int:
        return 'INVALID_RESPONSE',None
    if envelope['numberReturned']!=len(fs) or any(not isinstance(x,dict) for x in links):
        return 'INVALID_RESPONSE',None
    if any(x.get('rel')=='next' for x in links):
        return 'PAGINATED_OR_AMBIGUOUS',None
    if len(fs)==0:
        return 'NO_EXACT_OBSERVATION',None
    if len(fs)!=1:
        return 'MULTIPLE_FEATURES',None
    f=fs[0]
    if not isinstance(f,dict) or not isinstance(p:=f.get('properties'),dict):
        return 'INVALID_RESPONSE',None
    identity={'monitoring_location_id':pair['usgs_id'],'time_series_id':pair['series'],
              'parameter_code':'00065','statistic_id':'00011','unit_of_measure':'ft'}
    if any(p.get(k)!=v for k,v in identity.items()):
        return 'SERIES_UNIT_OR_PARAMETER_MISMATCH',None
    try:
        if t(p['time'])!=t(pair['target_utc']):
            return 'NONEXACT_TIME',None
    except (ValueError,TypeError,AttributeError,KeyError):
        return 'INVALID_TIME',None
    if p.get('approval_status')!='Approved':
        return 'NOT_APPROVED_OR_MISSING_STATUS',None
    if 'qualifier' not in p or p['qualifier'] is not None:
        return 'QUALIFIED_OR_MISSING_QUALIFIER_FIELD',None
    try:
        v=Decimal(str(p['value']))
        if not v.is_finite():
            return 'NONNUMERIC_OR_NONFINITE',None
    except (InvalidOperation,KeyError,TypeError):
        return 'NONNUMERIC_OR_NONFINITE',None
    return 'EXACT_APPROVED_PAIR',v

def describe(pairs, envelopes, event_gates):
    """Requires a reviewer-provided disposition for each event. No network or file IO."""
    pairs=list(pairs)
    check(len(pairs)==11 and len({p['pair_id'] for p in pairs})==11,'Must use all unique locked pairs')
    check(set(event_gates)=={'KY2022','OH2024','HELENE2024'},'Explicit gate disposition for each event required')
    check(set(envelopes)<={p['pair_id'] for p in pairs},'Unregistered observations supplied')
    rows=[]; by_event=defaultdict(list)
    for p in pairs:
        status,value=classify(p,envelopes.get(p['pair_id']),event_gates[p['episode']])
        err=Fraction(Decimal(p['stage_guidance_ft']))-Fraction(value) if value is not None else None
        row={k:p[k] for k in ('pair_id','episode','target_utc','lead_minutes')}
        row.update(status=status,residual_ft=err,absolute_residual_ft=abs(err) if err is not None else None)
        rows.append(row)
        by_event[p['episode']].append(row)
    check({k:len(v) for k,v in by_event.items()}=={'KY2022':4,'OH2024':2,'HELENE2024':5},'Wrong event allocation')
    s={}
    for event,rs in by_event.items():
        v=[x['residual_ft'] for x in rs if x['status']=='EXACT_APPROVED_PAIR']
        s[event]=dict(eligible=len(v),locked=len(rs),mae_ft=sum(map(abs,v))/len(v) if v else None,
            bias_ft=sum(v)/len(v) if v else None)
    full=all(x['eligible']==x['locked'] for x in s.values())
    scores=dict(primary_status='complete' if full else 'not_evaluable_incomplete',
       macro_mae_ft=sum(x['mae_ft'] for x in s.values())/3 if full else None,
       macro_bias_ft=sum(x['bias_ft'] for x in s.values())/3 if full else None,
       pooled_mae_ft=sum(x['absolute_residual_ft'] for x in rows)/11 if full else None,
       event_descriptives=s)
    return rows,scores
