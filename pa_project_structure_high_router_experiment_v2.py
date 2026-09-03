from __future__ import annotations

import re
from urllib.parse import urlparse

import pandas as pd
import requests

import pa_project_structure_high_router_experiment as exp

META_URL = 'https://www.fema.gov/api/open/v1/OpenFemaDataSets'


def extract_any_list(payload):
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for v in payload.values():
        if isinstance(v, list):
            return v
    return []


def discover_pa_endpoint(session: requests.Session):
    candidates = []
    try:
        r = session.get(META_URL, params={'$top': 1000, '$metadata': 'off'}, timeout=120,
                        headers={'User-Agent': 'ML-thesis-research/1.0'})
        r.raise_for_status()
        records = extract_any_list(r.json())
        for rec in records:
            blob = ' '.join(str(v) for v in rec.values()).lower()
            if ('public assistance' in blob or 'publicassistance' in blob) and ('funded project' in blob or 'fundedproject' in blob):
                # Prefer an explicit API URL if FEMA publishes one in metadata.
                for v in rec.values():
                    s = str(v)
                    if s.startswith('https://www.fema.gov/api/open/'):
                        candidates.append(s.rstrip('/'))
                names = []
                for key in ['name', 'entityName', 'openFemaDataSet', 'datasetName']:
                    v = rec.get(key)
                    if v:
                        names.append(str(v))
                versions = []
                for key in ['version', 'datasetVersion']:
                    try:
                        versions.append(int(float(rec.get(key))))
                    except Exception:
                        pass
                if not versions:
                    versions = [4, 3, 2, 1]
                for name in names:
                    clean = re.sub(r'[^A-Za-z0-9_]', '', name)
                    if clean:
                        for ver in sorted(set(versions + [4, 3, 2, 1]), reverse=True):
                            candidates.append(f'https://www.fema.gov/api/open/v{ver}/{clean}')
    except Exception as e:
        print(f'Metadata discovery warning: {e}', flush=True)

    # Robust fallbacks for known historical entity naming.
    for ver in [4, 3, 2, 1]:
        for name in [
            'PublicAssistanceFundedProjectsDetails',
            'PublicAssistanceFundedProjectDetails',
            'PublicAssistanceFundedProjects',
            'PublicAssistanceProjects',
        ]:
            candidates.append(f'https://www.fema.gov/api/open/v{ver}/{name}')

    seen = set()
    for url in candidates:
        if url in seen:
            continue
        seen.add(url)
        try:
            r = session.get(url, params={'$top': 1, '$metadata': 'off'}, timeout=60,
                            headers={'User-Agent': 'ML-thesis-research/1.0'})
            if r.status_code != 200:
                continue
            rows = extract_any_list(r.json())
            if rows:
                print(f'Discovered PA endpoint: {url}', flush=True)
                print(f'PA fields: {sorted(rows[0].keys())}', flush=True)
                return url, rows[0]
        except Exception:
            continue
    raise RuntimeError('Could not discover a working OpenFEMA Public Assistance funded-project endpoint')


def first_present(fields, choices):
    lower = {str(f).lower(): f for f in fields}
    for c in choices:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def fetch_pa_current(disaster_numbers, chunk_size=12):
    s = requests.Session()
    endpoint, sample = discover_pa_endpoint(s)
    fields = list(sample.keys())

    mapping = {
        'disasterNumber': first_present(fields, ['disasterNumber', 'disasterNum', 'declarationNumber']),
        'pwNumber': first_present(fields, ['pwNumber', 'projectNumber', 'projectWorksheetNumber', 'projectId', 'pwId']),
        'applicantId': first_present(fields, ['applicantId', 'applicantNumber', 'applicantCode', 'applicantName']),
        'damageCategoryCode': first_present(fields, ['damageCategoryCode', 'categoryCode', 'projectCategoryCode']),
        'damageCategory': first_present(fields, ['damageCategory', 'projectCategory', 'category']),
        'county': first_present(fields, ['county', 'countyName', 'countyParish']),
        'state': first_present(fields, ['state', 'stateAbbreviation', 'stateCode']),
    }
    if mapping['disasterNumber'] is None:
        raise RuntimeError(f'PA dataset has no recognizable disaster-number field: {fields}')

    select_fields = []
    for v in mapping.values():
        if v and v not in select_fields:
            select_fields.append(v)

    rows = []
    for start in range(0, len(disaster_numbers), chunk_size):
        chunk = disaster_numbers[start:start + chunk_size]
        filt = ' or '.join(f"{mapping['disasterNumber']} eq {int(x)}" for x in chunk)
        skip = 0
        while True:
            params = {'$filter': filt, '$top': 1000, '$skip': skip, '$metadata': 'off'}
            if select_fields:
                params['$select'] = ','.join(select_fields)
            r = s.get(endpoint, params=params, timeout=180,
                      headers={'User-Agent': 'ML-thesis-research/1.0'})
            r.raise_for_status()
            batch = extract_any_list(r.json())
            for rec in batch:
                norm = {}
                for canonical, actual in mapping.items():
                    norm[canonical] = rec.get(actual) if actual else None
                rows.append(norm)
            if len(batch) < 1000:
                break
            skip += 1000
        print(f'PA fetch {min(start+chunk_size, len(disaster_numbers))}/{len(disaster_numbers)} rows={len(rows):,}', flush=True)

    raw = pd.DataFrame(rows, columns=['disasterNumber','pwNumber','applicantId','damageCategoryCode','damageCategory','county','state'])
    raw.to_csv(exp.OUT / 'pa_project_rows_nonfinancial.csv', index=False)
    (exp.OUT / 'discovered_endpoint.txt').write_text(endpoint + '\n', encoding='utf-8')
    return raw


if __name__ == '__main__':
    exp.fetch_pa = fetch_pa_current
    exp.main()
