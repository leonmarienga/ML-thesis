import hashlib,json
from pathlib import Path
from forecast_probe import E,D
package={'protocol_sha256':hashlib.sha256((D/'protocol.json').read_bytes()).hexdigest(),'episodes':{}}
for event,(lid,*_) in E.items():
    n=D/f'metadata-{lid}.json';q=D/f'forecast-{event}.json'
    data=json.loads(n.read_text())
    gs=data['noaa']['metadata']
    result=next(f['properties'] for f in data['usgs']['metadata']['features']
                if f['properties']['statistic_id']=='00011' and f['properties']['parameter_code']=='00065')
    parsed=json.loads(q.read_text())
    package['episodes'][event]={'noaa_site':gs,'usgs_series':{k:result[k] for k in (
      'id','monitoring_location_id','parameter_code','statistic_id','unit_of_measure',
      'begin_utc','end_utc','computation_identifier')},'noaa_metadata_source':data['noaa']['url'],
      'usgs_metadata_source':data['usgs']['url'],'metadata_replay_inputs_sha256':hashlib.sha256(n.read_bytes()).hexdigest(),
      'forecast_replay_inputs_sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'source_requests':parsed['sources']}
print(json.dumps(package,sort_keys=True,indent=2))
