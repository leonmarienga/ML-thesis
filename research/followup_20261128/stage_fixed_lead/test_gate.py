import json,shutil,tempfile
from pathlib import Path
import audit
with tempfile.TemporaryDirectory(dir=audit.D) as tmp:
    tmp=Path(tmp)
    for name in ('protocol.json',*[f'forecast-{k}.json' for k in audit.E],
                 *[f'metadata-{v[0]}.json' for v in audit.E.values()]):
        shutil.copy2(audit.D/name,tmp/name)
    audit.D=tmp
    p=tmp/'forecast-KY2022.json'
    original=p.read_text()
    for mutator in (
        lambda r:r.update(guidance_disclaimer=False),
        lambda r:r.update(forecast_header=r['forecast_header'].replace('/HGIFF/','/HGIFE/')),
        lambda r:r.update(forecast_continuations=['unreadable']),
    ):
        data=json.loads(original);mutator(data['wmo_forecast_only'][0])
        p.write_text(json.dumps(data))
        try: audit.run()
        except ValueError as e:
            assert 'contrary public-stage path' in str(e), e
        else: raise AssertionError('Fail-closed gate accepted a changed input')
print('3/3 changed-corpus tests failed closed')
