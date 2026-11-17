#!/usr/bin/env python3
"""Retrieve exact recorded public URLs and require matching original bytes.

For revised dynamic APIs, a mismatch is reported, never silently accepted as
reproduction. Original saved snapshots should be used where available.
"""
from pathlib import Path
import argparse,hashlib,json
import pandas as pd
import requests

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for x in pd.read_csv(a.manifest).fillna('').to_dict('records'):
        if x['status']!='retrieved':continue
        result={'name':x['name']}
        try:
            url=x.get('actual_download_url') or x['url']
            with requests.get(url,timeout=(20,50),stream=True) as r:
                r.raise_for_status();parts=[];size=0
                for chunk in r.iter_content(256*1024):
                    size+=len(chunk)
                    if size>25_000_000:raise ValueError('download budget exceeded')
                    parts.append(chunk)
            content=b''.join(parts);digest=hashlib.sha256(content).hexdigest()
            result.update(sha256=digest,exact_bytes_match=digest==x['sha256'])
            (a.out/x['name']).write_bytes(content)
        except Exception as e:result.update(error=type(e).__name__,exact_bytes_match=False)
        rows.append(result)
        print(x['name'],result.get('exact_bytes_match'),flush=True)
    pd.DataFrame(rows).to_csv(a.out/'reproduction_report.csv',index=False)
    if not all(r['exact_bytes_match'] for r in rows):raise SystemExit('Some dynamic source responses differ or are unavailable; see report')

if __name__=='__main__':main()
