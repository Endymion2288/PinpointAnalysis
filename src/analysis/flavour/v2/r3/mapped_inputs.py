"""Allocation-local ragged arrays shared through file-backed, read-only views."""

import pickle
from pathlib import Path

import numpy as np


class MappedItems:
    def __init__(self,directory):
        self.directory=Path(directory)
        with (self.directory/'manifest.pkl').open('rb') as stream:
            self.manifest=pickle.load(stream)
        self.arrays={}
        self._fast_specs=None

    def __len__(self):
        return len(self.manifest['records'])

    def __getstate__(self):
        # Pickle the cache identity and small metadata, never mapped contents.
        return {'directory':self.directory,'manifest':self.manifest,'arrays':{},'_fast_specs':None}

    def _ensure_open(self):
        specs=[]
        for name,spec in self.manifest['arrays'].items():
            if name not in self.arrays:
                path=self.directory/(name+'.bin')
                self.arrays[name]=(np.memmap(path,dtype=spec['dtype'],mode='r')
                    if path.stat().st_size else np.empty(0,dtype=spec['dtype']))
            specs.append((name,self.arrays[name],spec['offsets'],spec['shapes']))
        self._fast_specs=specs

    def __getitem__(self,index):
        if self._fast_specs is None:
            self._ensure_open()
        item=dict(self.manifest['records'][index])
        for name,arr,offsets,shapes in self._fast_specs:
            start=int(offsets[index])
            stop=int(offsets[index+1])
            item[name]=arr[start:stop].reshape(shapes[index])
        return item

    def close(self):
        self.arrays.clear()
        self._fast_specs=None


def write_mapped_items(groups,directory):
    """Stream worker batches to scratch; keep at most one returned batch in RAM."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    records=[];arrays={};streams={}
    try:
        for group in groups:
            for item in group:
                record={}
                for name,value in item.items():
                    if not isinstance(value,np.ndarray):
                        record[name]=value
                        continue
                    if name not in arrays:
                        arrays[name]={'dtype':value.dtype.str,'offsets':[0],'shapes':[]}
                        streams[name]=(directory/(name+'.bin')).open('wb')
                    spec=arrays[name]
                    if value.dtype.str!=spec['dtype']:raise ValueError('cache array dtype changed')
                    streams[name].write(value.tobytes(order='C'))
                    spec['offsets'].append(spec['offsets'][-1]+value.size)
                    spec['shapes'].append(value.shape)
                records.append(record)
    finally:
        for stream in streams.values():stream.close()
    n_rec=len(records)
    for name,spec in arrays.items():
        if len(spec['offsets'])!=n_rec+1:
            raise ValueError(f"mapped_inputs array {name!r} present in {len(spec['offsets'])-1}/{n_rec} records; all records must share identical ndarray keys")
        spec['offsets']=np.asarray(spec['offsets'],dtype='int64')
    with (directory/'manifest.pkl').open('wb') as stream:
        pickle.dump({'records':records,'arrays':arrays},stream,protocol=5)
    return MappedItems(directory)
