from pathlib import Path

src_path = Path('systematic_feature_engineering_high_router.py')
src = src_path.read_text(encoding='utf-8')
old = """    for c in ['declaredAreaCount','uniquePlaceCodeCount','declarationGeographyRowCount']:\n        if c not in d.columns:\n            d[c] = np.nan\n\n    # Numeric coercion only for safe, non-dollar descriptors.\n"""
new = """    for c in ['declaredAreaCount','uniquePlaceCodeCount','declarationGeographyRowCount']:\n        if c not in d.columns:\n            d[c] = np.nan\n\n    # Rebuild the event grouping after the geographic columns have been merged.\n    # The earlier GroupBy object was created before the merge and therefore\n    # could not see declaredAreaCount/other newly merged columns.\n    g = d.groupby('event_key', dropna=False)\n\n    # Numeric coercion only for safe, non-dollar descriptors.\n"""
if old not in src:
    raise RuntimeError('Expected regrouping patch location was not found; refusing to run a modified experiment.')
fixed = src.replace(old, new, 1)
code = compile(fixed, str(src_path), 'exec')
globals_dict = {'__name__': '__main__', '__file__': str(src_path)}
exec(code, globals_dict)
