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

    def __len__(self):
        return len(self.manifest['records'])

    def __getstate__(self):
        # Pickle the cache identity and small metadata, never mapped contents.
        return {'directory':self.directory,'manifest':self.manifest,'arrays':{}}

    def __getitem__(self,index):
        item=dict(self.manifest['records'][index])
        for name,spec in self.manifest['arrays'].items():
            if name not in self.arrays:
                path=self.directory/(name+'.bin')
                self.arrays[name]=(np.memmap(path,dtype=spec['dtype'],mode='r')
                    if path.stat().st_size else np.empty(0,dtype=spec['dtype']))
            start,stop=spec['offsets'][index:index+2]
            item[name]=self.arrays[name][start:stop].reshape(spec['shapes'][index])
        return item


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
    for spec in arrays.values():spec['offsets']=np.asarray(spec['offsets'],dtype='int64')
    with (directory/'manifest.pkl').open('wb') as stream:
        pickle.dump({'records':records,'arrays':arrays},stream,protocol=5)
    return MappedItems(directory)
