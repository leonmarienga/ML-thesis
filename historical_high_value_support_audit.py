from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DECL_URL = 'https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries'
MA_URL = 'https://www.fema.gov/api/open/v2/MissionAssignments'
MASTER = Path('master_openfema_40plus.xlsx')
OUT = Path('historical_high_value_support_results')
OUT.mkdir(exist_ok=True)
MA_CACHE = OUT / 'MissionAssignments_2000_2009.jsonl'

BANDS6 = ['0-100K','100K-1M','1M-50M','50M-200M','200M-500M','500M+']
BINS6 = [-np.inf,1e5,1e6,50e6,200e6,500e6,np.inf]
BANDS7 = ['0-100K','100K-1M','1M-50M','50M-200M','200M-500M','500M-1B','1B+']
BINS7 = [-np.inf,1e5,1e6,50e6,200e6,500e6,1e9,np.inf]


def extract_list(payload: object, expected: str) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise RuntimeError(f'Unexpected OpenFEMA response: {type(payload)}')
    for key in (expected, expected.rstrip('s'), 'items', 'data'):
        if isinstance(payload.get(key), list):
            return payload[key]
    lists = [v for v in payload.values() if isinstance(v, list)]
    if len(lists) == 1:
        return lists[0]
    raise RuntimeError(f'Could not find list in response keys {list(payload)}')


def fetch_historical_declarations() -> pd.DataFrame:
    session = requests.Session()
    rows = []
    top = 10000
    skip = 0
    params_base = {
        '$filter': "declarationType eq 'DR' and fyDeclared ge 2000 and fyDeclared le 2009",
        '$select': 'disasterNumber,state,declarationType,fyDeclared,declarationDate,incidentType,incidentBeginDate,incidentEndDate,ihProgramDeclared,iaProgramDeclared,paProgramDeclared,hmProgramDeclared,designatedArea,placeCode',
        '$metadata': 'off',
    }
    while True:
        params = dict(params_base, **{'$top': top, '$skip': skip})
        r = session.get(DECL_URL, params=params, timeout=240, headers={'User-Agent':'ML-thesis-research/1.0'})
        r.raise_for_status()
        batch = extract_list(r.json(), 'DisasterDeclarationsSummaries')
        rows.extend(batch)
        if len(batch) < top:
            break
        skip += top
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise RuntimeError('No 2000-2009 DR declaration rows returned')
    raw['disasterNumber'] = pd.to_numeric(raw['disasterNumber'], errors='coerce')
    raw = raw[raw.disasterNumber.notna()].copy()
    raw['disasterNumber'] = raw.disasterNumber.astype(int)
    raw['fyDeclared'] = pd.to_numeric(raw['fyDeclared'], errors='coerce')
    raw.to_csv(OUT/'historical_declaration_rows.csv', index=False)

    # One row per federal disaster number, matching the structure of the 2010-2024 master.
    group_cols = ['disasterNumber']
    first_cols = ['state','declarationType','fyDeclared','declarationDate','incidentType','incidentBeginDate','incidentEndDate']
    agg = {c:'first' for c in first_cols if c in raw.columns}
    for c in ['ihProgramDeclared','iaProgramDeclared','paProgramDeclared','hmProgramDeclared']:
        if c in raw.columns:
            agg[c] = 'max'
    one = raw.groupby(group_cols, as_index=False).agg(agg)
    geography = raw.groupby('disasterNumber').agg(
        declarationGeographyRows=('disasterNumber','size'),
        uniquePlaceCodeCount=('placeCode','nunique'),
    ).reset_index()
    one = one.merge(geography, on='disasterNumber', how='left')
    one.to_csv(OUT/'historical_disasters_2000_2009.csv', index=False)
    return one


def fetch_historical_missions(disaster_numbers: list[int], chunk_size: int = 35) -> pd.DataFrame:
    if MA_CACHE.exists():
        return pd.read_json(MA_CACHE, lines=True)
    session = requests.Session()
    rows: list[dict] = []
    for start in range(0, len(disaster_numbers), chunk_size):
        chunk = disaster_numbers[start:start+chunk_size]
        filt = ' or '.join(f'disasterNumber eq {int(n)}' for n in chunk)
        skip = 0
        while True:
            params = {'$filter':filt, '$top':1000, '$skip':skip, '$metadata':'off'}
            r = session.get(MA_URL, params=params, timeout=180, headers={'User-Agent':'ML-thesis-research/1.0'})
            r.raise_for_status()
            batch = extract_list(r.json(), 'MissionAssignments')
            rows.extend(batch)
            if len(batch) < 1000:
                break
            skip += 1000
        print(f'MissionAssignments: {min(start+chunk_size,len(disaster_numbers))}/{len(disaster_numbers)} disasters, rows={len(rows):,}')
        time.sleep(0.03)
    raw = pd.DataFrame(rows)
    raw.to_json(MA_CACHE, orient='records', lines=True)
    return raw


def aggregate_missions(raw: pd.DataFrame, disasters: list[int]) -> pd.DataFrame:
    if raw.empty:
        raw = pd.DataFrame(columns=['disasterNumber','obligationAmount','agency','maType','priority'])
    m = raw.copy()
    m['disasterNumber'] = pd.to_numeric(m.get('disasterNumber'), errors='coerce')
    m = m[m.disasterNumber.notna()].copy()
    m['disasterNumber'] = m.disasterNumber.astype(int)
    m['obligationAmount'] = pd.to_numeric(m.get('obligationAmount'), errors='coerce').fillna(0.0)
    for c in ['agency','maType','priority']:
        if c not in m.columns:
            m[c] = np.nan
    a = m.groupby('disasterNumber').agg(
        missionAssignmentCount=('disasterNumber','size'),
        uniqueAgencyCount=('agency','nunique'),
        uniqueMaTypeCount=('maType','nunique'),
        uniquePriorityCount=('priority','nunique'),
        totalObligatedFunding=('obligationAmount','sum'),
        positiveObligationRows=('obligationAmount', lambda x:int((x>0).sum())),
        negativeObligationRows=('obligationAmount', lambda x:int((x<0).sum())),
    ).reset_index()
    base = pd.DataFrame({'disasterNumber':disasters})
    out = base.merge(a, on='disasterNumber', how='left')
    for c in ['missionAssignmentCount','uniqueAgencyCount','uniqueMaTypeCount','uniquePriorityCount','totalObligatedFunding','positiveObligationRows','negativeObligationRows']:
        out[c] = pd.to_numeric(out[c], errors='coerce').fillna(0)
    return out


def validate_2010_2011_current_cache() -> dict:
    # The main frozen MissionAssignments cache is restored by the workflow.
    cache = Path('mission_composition_results/MissionAssignments_v2.jsonl')
    if not cache.exists() or not MASTER.exists():
        return {'available':False}
    raw = pd.read_json(cache, lines=True)
    master = pd.read_excel(MASTER)
    master = master[master.fyDeclared.isin([2010,2011])].copy()
    raw['disasterNumber'] = pd.to_numeric(raw.disasterNumber, errors='coerce')
    raw = raw[raw.disasterNumber.isin(master.disasterNumber)].copy()
    raw['obligationAmount'] = pd.to_numeric(raw.obligationAmount, errors='coerce').fillna(0)
    a = raw.groupby('disasterNumber').agg(rawMissionCount=('disasterNumber','size'),rawFunding=('obligationAmount','sum')).reset_index()
    z = master[['disasterNumber','missionAssignmentCount','totalObligatedFunding']].merge(a,on='disasterNumber',how='left').fillna({'rawMissionCount':0,'rawFunding':0})
    count_exact = np.isclose(z.missionAssignmentCount,z.rawMissionCount).all()
    funding_exact = np.isclose(z.totalObligatedFunding,z.rawFunding,rtol=1e-10,atol=.01).all()
    return {
        'available':True,'cases':int(len(z)),'mission_count_exact_all':bool(count_exact),'funding_exact_all':bool(funding_exact),
        'max_abs_funding_difference':float(np.max(np.abs(z.totalObligatedFunding-z.rawFunding))),
    }


def main():
    dec = fetch_historical_declarations()
    nums = sorted(dec.disasterNumber.astype(int).unique().tolist())
    raw = fetch_historical_missions(nums)
    agg = aggregate_missions(raw, nums)
    data = dec.merge(agg, on='disasterNumber', how='left')
    data['targetClipped'] = data.totalObligatedFunding.clip(lower=0)
    data['band6'] = pd.cut(data.targetClipped, bins=BINS6, labels=BANDS6, right=True).astype(str)
    data['band7'] = pd.cut(data.targetClipped, bins=BINS7, labels=BANDS7, right=True).astype(str)
    data.to_csv(OUT/'historical_support_with_funding.csv', index=False)

    counts6 = {b:int((data.band6==b).sum()) for b in BANDS6}
    counts7 = {b:int((data.band7==b).sum()) for b in BANDS7}
    high = data[data.targetClipped>50e6].sort_values('targetClipped', ascending=False)
    high.to_csv(OUT/'historical_cases_over_50M.csv', index=False)

    summary = {
        'period':'2000-2009',
        'selection':"OpenFEMA DisasterDeclarationsSummaries where declarationType='DR'",
        'disaster_count':int(len(data)),
        'mission_assignment_rows':int(len(raw)),
        'band_counts_6':counts6,
        'band_counts_7':counts7,
        'extra_support_50M_plus':int((data.targetClipped>50e6).sum()),
        'extra_support_200M_500M':int(((data.targetClipped>200e6)&(data.targetClipped<=500e6)).sum()),
        'extra_support_500M_plus':int((data.targetClipped>500e6).sum()),
        'extra_support_1B_plus':int((data.targetClipped>1e9).sum()),
        'validation_against_2010_2011_master':validate_2010_2011_current_cache(),
        'notes':[
            'Historical target is the sum of OpenFEMA MissionAssignments obligationAmount by disasterNumber, with negative totals clipped to zero only for routing-band assignment.',
            'No historical case is used as a 2010-2024 test observation; this audit only measures potential extra training support.',
            '2010-2011 are used as a source-stability check because the current frozen MissionAssignments snapshot exactly matches the existing master for those years when validated.',
        ],
    }
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
