"""Transform public gauge-zero metadata only; never transform stage values."""
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
from hashlib import sha256
import json

D=Path(__file__).resolve().parent
base='https://geodesy.noaa.gov/api/ncat/llh'
# SI conversion uses US survey feet for historical NGVD29; inspect metadata only.
f=Decimal(1200)/Decimal(3937)
def case(job):
    event,ngvd,lat,lon=job
    p=dict(lat=lat,lon=lon,inDatum='NAD83(1986)',outDatum='NAD83(1986)',
        inVertDatum='NGVD29',outVertDatum='NAVD88',orthoHt=str(Decimal(ngvd)*f))
    url=base+'?'+urlencode(p)
    with urlopen(url,timeout=20) as r:
        raw=r.read(100_001); assert len(raw)<=100_000
        js=json.loads(raw)
        return dict(event=event,input_coordinate_frame='NAD83(1986) assumed for orthometric grid evaluation',
                    input_zero_ngvd29_ft=ngvd,status=r.status,sha256=sha256(raw).hexdigest(),url=r.geturl(),result=js)
inputs=[('KY2022','697.674','37.551383','-83.384725'),
        ('OH2024','608.1','40.057222222222','-80.728611111111')]
if __name__=='__main__':
    with ThreadPoolExecutor(2) as p:
        results=list(p.map(case,inputs))
    (D/'ncat-gauge-zero.json').write_text(json.dumps(results,sort_keys=True,indent=2)+'\n')
