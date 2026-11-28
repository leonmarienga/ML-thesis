"""Bounded acquisition of a NOAA archive cycle and an official same-site hydrograph."""
import concurrent.futures,datetime,hashlib,json,pathlib,urllib.error,urllib.parse,urllib.request
ROOT=pathlib.Path(__file__).resolve().parent
LIM=2_000_000
WATER='https://api.waterdata.usgs.gov/ogcapi/v1/collections/continuous/items'
URLS={
 'cnrfc_2022123112.zip':'https://www.cnrfc.noaa.gov/csv/2022123112_N_SanJoaquin_csv_export.zip',
 'cnrfc_2022123118.zip':'https://www.cnrfc.noaa.gov/csv/2022123118_N_SanJoaquin_csv_export.zip',
 'usgs_11335000.geojson':WATER+'?'+urllib.parse.urlencode(dict(f='json',monitoring_location_id='USGS-11335000',parameter_code='00060',time='2022-12-27T00:00:00Z/2023-01-03T00:00:00Z',limit='1600')),
 'cnrfc_group_definitions.pdf':'https://www.cnrfc.noaa.gov/documentation/North_SanJoaquin_definitions.pdf',
}
def fetch(item):
 name,url=item
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 r={'asset':name,'url':url,'attempt_utc':now}
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'ThesisPhysicalEvidenceFeasibility/1.0'})
  with urllib.request.urlopen(req,timeout=25) as f:
   r.update(status=f.status,type=f.headers.get('Content-Type'),final_url=f.url,last_modified=f.headers.get('Last-Modified'),etag=f.headers.get('ETag'))
   data=f.read(LIM+1)
  if len(data)>LIM: raise ValueError('response exceeds byte cap')
  if name.endswith('.zip') and not data.startswith(b'PK'): raise ValueError('not ZIP')
  if name.endswith('.pdf') and not data.startswith(b'%PDF'): raise ValueError('not PDF')
  if name.endswith('.geojson') and json.loads(data).get('type')!='FeatureCollection': raise ValueError('not GeoJSON FeatureCollection')
  (ROOT/'raw'/name).write_bytes(data)
  r.update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
 except urllib.error.HTTPError as e:
  r.update(status=e.code,error=str(e))
 except (OSError,ValueError) as e:
  r['error']=f'{type(e).__name__}: {e}'
 return r
def main():
 (ROOT/'raw').mkdir(exist_ok=False)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  rows=list(pool.map(fetch,URLS.items()))
 (ROOT/'raw'/'MANIFEST.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(json.dumps(rows,indent=2))
if __name__=='__main__': main()
