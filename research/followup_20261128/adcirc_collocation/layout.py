"""Bounded byte-range inspection; never download the full model file."""
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen
import h5py

ROOT = Path(__file__).resolve().parent
MODEL = next(x for x in json.loads((ROOT/'zenodo.json').read_text())['files']
             if x['key'] == 'AdcSwan_StormSurge_Data_2011_2021.nc')
URL, SIZE = MODEL['links']['self'], MODEL['size']
BLOCK, CAP = 65536, 12*1024*1024

class Range(io.RawIOBase):
    def __init__(self, fetch=False):
        self.fetch, self.pos, self.blocks, self.new = fetch, 0, {}, []
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset + (0,self.pos,SIZE)[whence]
        if self.pos < 0: raise ValueError('negative seek')
        return self.pos
    def read(self, count):
        if count < 0: raise ValueError('unbounded read')
        end, out = min(SIZE,self.pos+count), bytearray()
        while self.pos < end:
            start = self.pos//BLOCK*BLOCK
            stop = min(start+BLOCK,SIZE)
            if start not in self.blocks:
                if (len(self.blocks)+1)*BLOCK > CAP:
                    raise RuntimeError('range cap exceeded')
                p = ROOT/f'range-{start}.bin'
                if not p.exists():
                    if not self.fetch: raise FileNotFoundError(p)
                    req = Request(f'{URL}?range_probe={start}',headers={
                      'Range':f'bytes={start}-{stop-1}','Accept-Encoding':'identity'})
                    with urlopen(req,timeout=15) as r:
                        if r.status != 206 or r.headers.get('Content-Range') != f'bytes {start}-{stop-1}/{SIZE}':
                            raise ValueError((r.status,r.headers.get('Content-Range')))
                        b = r.read(stop-start+1)
                        result={'start':start,'stop_inclusive':stop-1,'status':r.status,
                                'Content-Range':r.headers.get('Content-Range'),'url':r.geturl()}
                    if len(b)!=stop-start: raise ValueError('incorrect range size')
                    p.write_bytes(b)
                    self.new.append(result)
                    with (ROOT/'new_ranges.jsonl').open('a') as m: m.write(json.dumps(result)+'\n')
                b = p.read_bytes()
                if len(b)!=stop-start: raise ValueError('incorrect range size')
                self.blocks[start] = b
            finish=min(end,stop)
            out.extend(self.blocks[start][self.pos-start:finish-start])
            self.pos=finish
        return bytes(out)
    def readinto(self,b):
        d=self.read(len(b)); b[:len(d)]=d
        return len(d)

if __name__ == '__main__':
    import sys
    source=Range('--fetch' in sys.argv)
    with h5py.File(source,'r',driver='fileobj') as ds:
        for k,v in ds.items():
            print(k, v.shape,v.dtype, v.chunks,
                  {a:str(b) for a,b in v.attrs.items() if not a.startswith(('DIMENSION','REFERENCE','_Netcdf'))})
    print('ranges',sorted(source.blocks))
