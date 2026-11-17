#!/usr/bin/env python3
"""Independent dated NASA MODIS rendered-map feasibility for six frozen MN sites.

GIBS visualization images are not raw product QA/class arrays or ground truth.
Never infer dry from transparent pixels, calculate flood-label accuracy from
rendered colors, or call a satellite product an independently surveyed footprint.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,io,xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests
from PIL import Image
from shapely.geometry import shape
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/flood_validation';OUT.mkdir(parents=True,exist_ok=True)
ENDPOINT='https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi'

def main():
    r=requests.get(ENDPOINT,params={'SERVICE':'WMS','REQUEST':'GetCapabilities'},timeout=40)
    r.raise_for_status();(OUT/'gibs_capabilities.xml').write_bytes(r.content)
    root=ET.fromstring(r.content);ns={'w':'http://www.opengis.net/wms'}
    layer='MODIS_Combined_Flood_3-Day'
    matches=[e for e in root.findall('.//w:Layer',ns) if e.find('w:Name',ns) is not None and e.find('w:Name',ns).text==layer]
    assert len(matches)==1
    (OUT/'selected_layer_metadata.xml').write_text(ET.tostring(matches[0],encoding='unicode'))
    legend='https://gibs.earthdata.nasa.gov/legends/MODIS_Flood_H.png'
    r=requests.get(legend,timeout=25);r.raise_for_status();(OUT/'MODIS_Flood_H.png').write_bytes(r.content)
    ro=pd.read_csv(ROOT/'research/phase50/frozen_67_site_roster.csv')
    doc=json.loads((ROOT/'research/phase50/reconstructed_all187_windows.geojson').read_text())
    geoms={f['properties']['sample_id']:shape(f['geometry']) for f in doc['features']}
    sources=[]
    # Fixed regional flood timing from NWS report, not selected using new pixels.
    dates=['2023-04-12','2023-04-26','2023-05-10']
    for row in ro[ro.disasterNumber==4722].itertuples():
        bounds=geoms[row.sample_id].bounds
        for date in dates:
            params=dict(SERVICE='WMS',VERSION='1.1.1',REQUEST='GetMap',LAYERS=layer,STYLES='',
                FORMAT='image/png',TRANSPARENT='TRUE',SRS='EPSG:4326',
                BBOX=','.join(map(str,bounds)),WIDTH=128,HEIGHT=128,TIME=date)
            path=OUT/f'{row.sample_id}_{date}_MODIS_render.png'
            audit=dict(sample_id=row.sample_id,observation_date=date,layer=layer,bbox=json.dumps(bounds),
                       retrieval_utc=datetime.now(timezone.utc).isoformat(),
                       classification='MODELLED ESTIMATE: satellite flood classification rendered visualization',
                       limitations='250m native product, resampled PNG, no raw QA; transparency is not dry land; no footprint truth')
            try:
                r=requests.get(ENDPOINT,params=params,timeout=30);r.raise_for_status()
                assert r.content[:8]==b'\x89PNG\r\n\x1a\n'
                im=np.array(Image.open(io.BytesIO(r.content)).convert('RGBA'))
                inside=geometry_mask([geoms[row.sample_id].__geo_interface__],out_shape=(128,128),
                    transform=from_bounds(*bounds,128,128),invert=True)
                # Document presence only, using palette from the returned render
                # plus the official legend. Display-cell counts are NOT areas.
                red=(im[:,:,0]>230)&(im[:,:,1]<60)&(im[:,:,2]<70)&(im[:,:,3]>0)&inside
                gray=(im[:,:,0]==175)&(im[:,:,1]==175)&(im[:,:,2]==175)&(im[:,:,3]>0)&inside
                path.write_bytes(r.content)
                audit.update(status='render_read_ok',url=r.url,sha256=hashlib.sha256(r.content).hexdigest(),
                             rendered_nontransparent_pixels=int((im[:,:,3]>0).sum()),
                             displayed_flood_color_inside_polygon=bool(red.any()),
                             displayed_insufficient_data_fraction_inside_polygon=float(gray.sum()/inside.sum()),
                             saved_path=str(path.relative_to(ROOT)))
            except Exception as e:audit.update(status='failed',error=type(e).__name__)
            sources.append(audit)
            pd.DataFrame(sources).to_csv(OUT/'nasa_mn_render_source_log.csv',index=False)
    (OUT/'summary.json').write_text(json.dumps(dict(
        layer=layer,sites=6,dates=dates,renders_successful=sum(x['status']=='render_read_ok' for x in sources),
        raw_product_arrays_obtained=0,independent_ground_truth_labels=0,
        result='dated independent-sensor rendered maps successfully requested where available; no label validation',
        next_gate='authorized raw MODIS historical-product and QA access, georegistration and explicit independent-label protocol',
        no_FPR_precision_recall_IoU_computed=True),indent=2))
    print('NASA_MAP_AUDIT',len(sources),'renders attempted',flush=True)

if __name__=='__main__':main()
