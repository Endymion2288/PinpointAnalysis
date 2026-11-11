"""Verify newly converted S2 catalog coverage and summarize actual token costs."""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.flavour.datasets import NativeShard
from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.s2.representation import token_inputs


def run(args):
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    started=time.perf_counter()
    plan=json.loads((args.root/'plan.json').read_text())
    geometry=json.loads(Path(plan['geometry']).read_text())
    expected=pd.read_parquet(args.root/'manifest.parquet')
    key=['run','chunk','row']
    meta,stats=[],[]
    total_bytes=0
    for path in sorted((args.root/'shards').iterdir()):
        summary=json.loads((path/'summary.json').read_text())
        if summary['PE_input']!=summary['PE_output']:
            raise ValueError('S2 conversion PE conservation mismatch')
        frame=pd.read_parquet(path/'provenance.parquet')
        targets=pd.read_parquet(path/'targets.parquet')
        dataset=NativeShard(path,target_columns=['label','label_status'])
        if not len(dataset)==len(frame)==len(targets) or not np.array_equal(frame.label,targets.label):
            raise ValueError('S2 observation-target-provenance mismatch')
        if not targets.label_status.eq('consistent').all():
            raise ValueError('quarantined labels entered S2')
        total_bytes+=sum(p.stat().st_size for p in path.glob('*.parquet'))
        count=np.stack([np.diff(dataset.offsets[k]) for k in ['pixel','bar','tracker_hypothesis']],axis=1)
        frame[['pixel_channels','bar_channels','tracker_channels']]=count
        frame['all_empty']=(count.sum(1)==0)
        meta.append(frame)
        # Deterministic response-blind row sample for representation overhead;
        # occupancy tails for the entire catalog come from Arrow offsets above.
        for i in np.linspace(0,len(frame)-1,min(32,len(frame)),dtype=int):
            event=dataset[int(i)]['observations']
            for pitch,projection in [(0.,False),(0.,True),(.1,False),(.2,False),(1.,False),(1.,True)]:
                t=time.perf_counter()
                nodes,groups,_=token_inputs(event,geometry,pitch_mm=pitch,projection=projection)
                stats.append({'shard':path.name,'row':int(i),'pitch_mm':pitch,'projection':projection,
                              'nodes':len(nodes),'blocks':int(groups.max())+1,'seconds':time.perf_counter()-t,
                              'label':int(frame.iloc[i].label),'split':frame.iloc[i].split})
        del dataset
        print(f'audited {path.name}',flush=True)
    observed=pd.concat(meta,ignore_index=True)
    if observed.duplicated(key).any():
        raise ValueError('repeated S2 observation')
    matched=expected[key+['split','label','family_hi','family_lo']].merge(observed[key+['split','label','family_hi','family_lo']],
            on=key,how='outer',validate='one_to_one',indicator=True,suffixes=('_expected','_observed'))
    if not matched['_merge'].eq('both').all() or len(matched)!=len(expected):
        raise ValueError('converted subset differs from frozen S2 selection')
    for field in ['split','label','family_hi','family_lo']:
        if not matched[field+'_expected'].eq(matched[field+'_observed']).all():
            raise ValueError(f'frozen manifest field changed: {field}')
    if observed.groupby(['family_hi','family_lo']).split.nunique().max()!=1:
        raise ValueError('accepted family group crosses S2 partitions')
    if not observed.split.isin(['train','validation']).all():
        raise ValueError('forbidden partition in S2 catalog')
    observed.to_parquet(args.root/'catalog.parquet',index=False,compression=COMPRESSION)
    token_frame=pd.DataFrame(stats)
    token_frame.to_parquet(args.root/'token_profile.parquet',index=False,compression=COMPRESSION)
    profile=token_frame.groupby(['pitch_mm','projection'])[['nodes','blocks','seconds']].quantile([0,.5,.9,.99,1]).reset_index().to_dict('records')
    write_json(args.root/'catalog_summary.json',{'events':len(observed),'shards':len(meta),
        'counts':observed.groupby(['split','run']).size().rename('events').reset_index().to_dict('records'),
        'tau_branch':observed[observed.label.eq(2)].groupby(['split','tau_branch']).size().rename('events').reset_index().to_dict('records'),
        'region_counts':{r:observed[observed[r]].groupby(['split','label']).size().rename('events').reset_index().to_dict('records')
                         for r in ['pixel_centered','large_scintillator','all_empty']},
        'all_event_occupancy_quantiles':observed[['pixel_channels','bar_channels','tracker_channels']].quantile([0,.5,.9,.99,.999,1]).to_dict(),
        'representation_profile_sampled_events':len(stats)//6,'representation_profile_quantiles':profile,
        'frozen_subset_exact_join':True,'shared_split_unchanged':True,'signature_group_overlap':0,
        'calibration_test_emitted':False,'total_parquet_bytes':total_bytes,
        'seconds':time.perf_counter()-started,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    run(parser.parse_args())


if __name__=='__main__':
    main()
