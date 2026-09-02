from __future__ import annotations

import json
import numpy as np
import pandas as pd
import requests

import declared_population_exposure_router_experiment as exp

FEMA_V2_URL = 'https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries'


def fetch_declarations_v2(master: pd.DataFrame) -> pd.DataFrame:
    years = pd.to_numeric(master['fyDeclared'], errors='coerce').dropna().astype(int)
    min_year = int(years.min())
    max_year = int(years.max())
    wanted = set(pd.to_numeric(master['disasterNumber'], errors='coerce').dropna().astype(int))

    rows = []
    skip = 0
    top = 10000
    params_base = {
        '$filter': f'fyDeclared ge {min_year} and fyDeclared le {max_year}',
        '$select': 'disasterNumber,state,placeCode,designatedArea,fipsStateCode,fipsCountyCode',
        '$metadata': 'off',
    }

    while True:
        params = dict(params_base)
        params['$top'] = str(top)
        params['$skip'] = str(skip)
        r = requests.get(FEMA_V2_URL, params=params, timeout=240, headers={'User-Agent': 'ML-thesis-research/1.0'})
        r.raise_for_status()
        payload = r.json()
        if isinstance(payload, list):
            chunk = payload
        elif isinstance(payload, dict):
            chunk = None
            # Current OpenFEMA responses normally use the entity name as the list key.
            for key in ('DisasterDeclarationsSummaries', 'DisasterDeclarationSummaries', 'items', 'data'):
                if isinstance(payload.get(key), list):
                    chunk = payload[key]
                    break
            if chunk is None:
                list_values = [v for v in payload.values() if isinstance(v, list)]
                if len(list_values) == 1:
                    chunk = list_values[0]
                else:
                    raise RuntimeError(f'Could not locate record list in OpenFEMA response keys: {list(payload.keys())}')
        else:
            raise RuntimeError(f'Unexpected OpenFEMA response type: {type(payload)}')

        if not chunk:
            break
        rows.extend(chunk)
        if len(chunk) < top:
            break
        skip += top

    dec = pd.DataFrame(rows)
    if dec.empty:
        raise RuntimeError('OpenFEMA v2 returned no declaration rows')
    dec['disasterNumber'] = pd.to_numeric(dec['disasterNumber'], errors='coerce')
    dec = dec[dec['disasterNumber'].isin(wanted)].copy()
    dec['disasterNumber'] = dec['disasterNumber'].astype(int)
    if 'designatedArea' in dec.columns and 'declaredCountyArea' not in dec.columns:
        dec = dec.rename(columns={'designatedArea': 'declaredCountyArea'})
    return dec


def build_population_exposure_v2(master: pd.DataFrame) -> pd.DataFrame:
    dec = fetch_declarations_v2(master)
    dec['state'] = dec['state'].astype(str).str.strip()

    # Prefer the explicit v2 county FIPS field. Fall back to the legacy 99+county placeCode convention.
    county = dec.get('fipsCountyCode', pd.Series(index=dec.index, dtype=object)).astype(str).str.replace(r'\.0$', '', regex=True).str.extract(r'(\d+)', expand=False)
    county = county.where(county.str.len().between(1, 3)).str.zfill(3)
    fallback = dec.get('placeCode', pd.Series(index=dec.index, dtype=object)).map(exp.normalize_place_code)
    dec['county3'] = county.where(county.notna(), fallback)

    explicit_state_fips = dec.get('fipsStateCode', pd.Series(index=dec.index, dtype=object)).astype(str).str.replace(r'\.0$', '', regex=True).str.extract(r'(\d+)', expand=False)
    explicit_state_fips = explicit_state_fips.where(explicit_state_fips.str.len().between(1, 2)).str.zfill(2)
    dec['stateFips'] = explicit_state_fips.where(explicit_state_fips.notna(), dec['state'].map(exp.STATE_FIPS))
    dec['candidateGEOID'] = np.where(
        dec['stateFips'].notna() & dec['county3'].notna(),
        dec['stateFips'].fillna('') + dec['county3'].fillna(''),
        None,
    )

    area_text = dec.get('declaredCountyArea', pd.Series('', index=dec.index)).fillna('').astype(str)
    dec['statewideFlagRow'] = area_text.str.contains('statewide', case=False, regex=False)

    gaz = exp.load_census_gazetteer()
    valid_geoids = set(gaz['GEOID'])
    dec['matchedGEOID'] = dec['candidateGEOID'].where(dec['candidateGEOID'].isin(valid_geoids))

    state_tot = gaz.groupby('USPS', as_index=False).agg(
        stateGazetteerPopulation2010=('POP10','sum'),
        stateGazetteerLandSqMi=('ALAND_SQMI','sum'),
    )
    lookup = gaz.set_index('GEOID')[['POP10','ALAND_SQMI']]

    out_rows = []
    for disaster, g in dec.groupby('disasterNumber'):
        st = str(g['state'].dropna().iloc[0]) if g['state'].notna().any() else ''
        statewide = bool(g['statewideFlagRow'].any())
        geoids = sorted(set(g['matchedGEOID'].dropna().astype(str)))
        if statewide and st in set(state_tot['USPS']):
            sg = gaz[gaz['USPS'] == st]
            geoids = sorted(set(sg['GEOID'].astype(str)))
        vals = lookup.loc[geoids] if geoids else pd.DataFrame(columns=['POP10','ALAND_SQMI'])
        pops = pd.to_numeric(vals.get('POP10', pd.Series(dtype=float)), errors='coerce').fillna(0.0)
        lands = pd.to_numeric(vals.get('ALAND_SQMI', pd.Series(dtype=float)), errors='coerce').fillna(0.0)
        out_rows.append({
            'disasterNumber': int(disaster),
            'state': st,
            'declaredCountyMatchedCount': int(len(geoids)),
            'declaredUniquePlaceCount': int(g['placeCode'].astype(str).nunique()) if 'placeCode' in g.columns else int(len(g)),
            'declaredGeographyRowCount': int(len(g)),
            'statewideDeclarationFlag': int(statewide),
            'declaredPopulation2010': float(pops.sum()),
            'declaredCountyPopulationMean': float(pops.mean()) if len(pops) else 0.0,
            'declaredCountyPopulationMedian': float(pops.median()) if len(pops) else 0.0,
            'declaredCountyPopulationMax': float(pops.max()) if len(pops) else 0.0,
            'declaredLandSqMi': float(lands.sum()),
        })

    out = pd.DataFrame(out_rows)
    out = out.merge(state_tot, left_on='state', right_on='USPS', how='left').drop(columns=['USPS'], errors='ignore')
    out['stateGazetteerPopulation2010'] = pd.to_numeric(out['stateGazetteerPopulation2010'], errors='coerce').fillna(0.0)
    out['stateGazetteerLandSqMi'] = pd.to_numeric(out['stateGazetteerLandSqMi'], errors='coerce').fillna(0.0)
    out['declaredCountyMatchRatio'] = out['declaredCountyMatchedCount'] / out['declaredUniquePlaceCount'].replace(0, np.nan)
    out['declaredCountyMatchRatio'] = out['declaredCountyMatchRatio'].replace([np.inf,-np.inf], np.nan).fillna(0.0).clip(0,1)
    out['declaredPopulationShareState'] = out['declaredPopulation2010'] / out['stateGazetteerPopulation2010'].replace(0,np.nan)
    out['declaredPopulationShareState'] = out['declaredPopulationShareState'].replace([np.inf,-np.inf],np.nan).fillna(0.0).clip(0,1.5)
    out['declaredLandShareState'] = out['declaredLandSqMi'] / out['stateGazetteerLandSqMi'].replace(0,np.nan)
    out['declaredLandShareState'] = out['declaredLandShareState'].replace([np.inf,-np.inf],np.nan).fillna(0.0).clip(0,1.5)
    out['declaredPopulationDensity'] = out['declaredPopulation2010'] / out['declaredLandSqMi'].replace(0,np.nan)
    out['declaredPopulationDensity'] = out['declaredPopulationDensity'].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    out['logDeclaredPopulation2010'] = np.log1p(out['declaredPopulation2010'].clip(lower=0))
    out['logDeclaredLandSqMi'] = np.log1p(out['declaredLandSqMi'].clip(lower=0))
    out.to_csv(exp.OUT/'declared_population_exposure_features.csv', index=False)
    return out


if __name__ == '__main__':
    exp.FEMA_DECL_URL = FEMA_V2_URL
    exp.build_population_exposure = build_population_exposure_v2
    exp.main()
