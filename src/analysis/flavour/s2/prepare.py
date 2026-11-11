"""Freeze S2 subsets and convert selected approved rows to mixed-class shards.

Uses the accepted S0 manifest directly; no production certification or hardware
calibration is repeated. No calibration/test observations are emitted or used.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import resource
import time

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import uproot

from analysis.flavour.create_data import BRANCHES, ROOT_TREES, lists
from analysis.flavour.features import event_features, FEATURE_NAMES
from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.readout import pixel_channels, bar_channels, tracker_measurements, modality_state


def regions(frame, geometry):
    """S2 coordinate definitions; truth flags used only in selection/diagnostics.

    Pixel-centered = pixel edges inset 20 mm, first 500 mm from target start.
    Large = scintillator edges inset 15 mm, last 200 mm of target excluded.
    Boundaries use stored sensor/bar geometry, not guessed historical coordinates.
    """
    g = geometry
    x, y, z = frame.initX, frame.initY, frame.initZ
    z0 = min(g['tungsten_Zpos']) - g['tungsten_thickness'] / 2
    z1 = max(g['tungsten_Zpos']) + g['tungsten_thickness'] / 2
    px, py = np.asarray(g['pixel_Xpos']), np.asarray(g['pixel_Ypos'])
    dx, dy = np.median(np.diff(px)), np.median(np.diff(py))
    cx, cy = np.mean(g['scint_bar_Xcenter']), np.mean(g['scint_bar_Ycenter'])
    center = (x.between(px.min()-dx/2+20, px.max()+dx/2-20)
              & y.between(py.min()-dy/2+20, py.max()+dy/2-20) & z.between(z0, z0+500))
    large = ((x-cx).abs().le(g['scint_detector_width']/2-15)
             & (y-cy).abs().le(g['scint_detector_height']/2-15) & z.between(z0, z1-200))
    return center.to_numpy(), large.to_numpy(), {'target_z_mm': [z0,z1],
        'pixel_centered_inset_mm':20., 'pixel_centered_first_mm':500.,
        'large_scint_inset_mm':15., 'large_downstream_inset_mm':200.,
        'status':'S2_explicit_geometry_definition_not_claimed_identical_to_historical_study'}


def plan(args):
    started = time.perf_counter()
    cols = ['run','chunk','row','evtID','label','label_status','usable_join','split',
            'geometry_id','family_hi','family_lo','initE','initX','initY','initZ','initPDG',
            'fslPDG','intType','scatteringType','tgtPDG','n_pixel','n_bar','n_tracker']
    frame = pd.read_parquet(args.inventory, columns=cols)
    rng = np.random.default_rng(args.seed)
    chunks, selected = {}, []
    for run in [10081,10082,10083,10084]:
        available = np.sort(frame.loc[frame.run==run,'chunk'].unique())
        chosen = rng.choice(available,size=min(args.chunks,len(available)),replace=False)
        chunks[str(run)] = sorted(map(int,chosen))
        eligible = frame[(frame.run==run) & frame.chunk.isin(chosen)
                         & frame.usable_join & frame.label_status.eq('consistent')]
        for split, limit in [('train',args.train_per_class),('validation',args.val_per_class)]:
            pool = eligible[eligible.split.eq(split)]
            picked = pool.iloc[rng.permutation(len(pool))[:limit]].copy()
            selected.append(picked)
    manifest = pd.concat(selected).sort_values(['run','chunk','row']).reset_index(drop=True)
    geometry = json.loads(args.geometry.read_text())
    center, large, definition = regions(manifest, geometry)
    manifest['pixel_centered'], manifest['large_scintillator'] = center, large
    manifest['shard'] = 0
    for run, chosen in chunks.items():
        mapping = {chunk:i for i,chunk in enumerate(chosen)}
        keep = manifest.run.eq(int(run))
        manifest.loc[keep,'shard'] = manifest.loc[keep,'chunk'].map(mapping)
    args.output.mkdir(parents=True,exist_ok=True)
    manifest.to_parquet(args.output/'manifest.parquet',index=False,compression=COMPRESSION)
    tasks=[list(range(i,min(i+2,args.chunks))) for i in range(0,args.chunks,2)]
    counts=manifest.groupby(['split','run']).size().rename('events').reset_index().to_dict('records')
    write_json(args.output/'plan.json', {'seed':args.seed,'inventory':str(args.inventory),
        'geometry':str(args.geometry),'input_root':str(args.input),'chunks':chunks,'tasks':tasks,
        'selection':'seeded_chunk_sample_then_seeded_events_within_frozen_split',
        'max_train_per_class':args.train_per_class,'max_validation_per_class':args.val_per_class,
        'counts':counts,'regions':definition,'acceptance':'user_attested_existing_detector_proxy',
        'calibration_test_payloads_used':False,'seconds':time.perf_counter()-started})
    print(json.dumps({'counts':counts,'tasks':len(tasks),'output':str(args.output)}))


def tau_branches(root, event_ids):
    """Direct primary tau daughters, grouped by the stored parent track ID.

    Flat tau tree can contain secondary taus. Resolve primary tau track IDs from
    primaries PDG; do not assume trackID==1 or classify from any event lepton.
    Missing/ambiguous direct decays retain an explicit unknown label.
    """
    primary=pd.DataFrame(root['primaries'].arrays(['evtID','PDG','trackID'],library='np'))
    tau=pd.DataFrame(root['tau'].arrays(['evtID','PID','PDG','ParentPDG'],library='np'))
    parents=primary[primary.PDG.abs().eq(15)][['evtID','trackID']].drop_duplicates()
    daughter=tau[tau.ParentPDG.abs().eq(15)].merge(parents,left_on=['evtID','PID'],
                right_on=['evtID','trackID'],how='inner')
    result={int(e):'unknown' for e in event_ids}
    for evt,group in daughter.groupby('evtID'):
        pdgs=set(group.PDG.abs().astype(int))
        if 11 in pdgs and 13 in pdgs:
            branch='ambiguous'
        elif 11 in pdgs:
            branch='electron'
        elif 13 in pdgs:
            branch='muon'
        elif any(p>100 for p in pdgs):
            branch='hadronic'
        else:
            branch='unknown'
        result[int(evt)]=branch
    return result


def convert(args):
    pa.set_cpu_count(args.cpus)
    pa.set_io_thread_count(args.cpus)
    plan_data=json.loads((args.root/'plan.json').read_text())
    frame=pd.read_parquet(args.root/'manifest.parquet',filters=[('shard','in',plan_data['tasks'][args.task])])
    geometry=json.loads(Path(plan_data['geometry']).read_text())
    for shard_id, shard in frame.groupby('shard',sort=True):
        if args.smoke_per_class:
            shard=shard.groupby('run',group_keys=False).head(args.smoke_per_class)
        started=time.perf_counter()
        timings=dict(root_read=0.,aggregation=0.,write=0.)
        observations, targets, conserved, features = [], [], [], []
        ordered=[]
        with ThreadPoolExecutor(args.cpus) as executor:
            for (run,chunk), group in shard.groupby(['run','chunk'],sort=True):
                path=Path(plan_data['input_root'])/str(run)/f'{run}_{chunk:03d}.root'
                with uproot.open(path,decompression_executor=executor,interpretation_executor=executor,
                                 object_cache=None,array_cache=None) as root:
                    branches=tau_branches(root,group.evtID) if run==10083 else {}
                    # Read only windows containing selected train/validation rows.
                    for block, selected in group.groupby(group.row//32,sort=True):
                        t=time.perf_counter()
                        start=int(block)*32
                        stop=min(start+32,root['event'].num_entries)
                        batch={name:root[ROOT_TREES[name]].arrays(keys,entry_start=start,entry_stop=stop,library='ak')
                               for name,keys in BRANCHES.items()}
                        timings['root_read']+=time.perf_counter()-t
                        t=time.perf_counter()
                        for event in selected.itertuples(index=False):
                            i=int(event.row)-start
                            pixel=pixel_channels(batch['pixel'][i],geometry)
                            bars,check=bar_channels(batch['bar'][i],geometry)
                            seed=int(event.family_hi)^int(event.family_lo)^20261111
                            _,tracker=tracker_measurements(batch['tracker'][i],seed)
                            state=modality_state(pixel,bars,tracker)
                            observations.append({'pixel':lists(pixel),'bar':lists(bars),
                                'tracker_hypothesis':lists(tracker),'has_response':state['has_response'].tolist(),
                                'outside_acceptance':state['outside_acceptance'].tolist(),
                                'fit_attempted':False,'momentum_censored':-1})
                            targets.append({'label':int(event.label),'label_status':'consistent'})
                            features.append(event_features(pixel,bars))
                            conserved.append(check)
                            provenance=event._asdict()
                            provenance['tau_branch']=branches.get(int(event.evtID),'not_tau')
                            ordered.append(provenance)
                        timings['aggregation']+=time.perf_counter()-t
        output=(args.output_root or args.root)/'shards'/f'{int(shard_id):03d}'
        output.mkdir(parents=True,exist_ok=True)
        t=time.perf_counter()
        # Seeded mixed-class ordering independent of class/file iteration order.
        order=np.random.default_rng(plan_data['seed']+int(shard_id)).permutation(len(ordered))
        pq.write_table(pa.Table.from_pylist([observations[i] for i in order]),output/'observations.parquet',
                       compression=COMPRESSION,row_group_size=64)
        pq.write_table(pa.Table.from_pylist([targets[i] for i in order]),output/'targets.parquet',compression=COMPRESSION)
        pd.DataFrame([ordered[i] for i in order]).to_parquet(output/'provenance.parquet',index=False,compression=COMPRESSION)
        pd.DataFrame(np.asarray(features)[order],columns=FEATURE_NAMES).to_parquet(output/'b0_features.parquet',
                       index=False,compression=COMPRESSION)
        timings['write']=time.perf_counter()-t
        write_json(output/'summary.json',{'shard':int(shard_id),'events':len(order),
            'seconds':time.perf_counter()-started,'timing_seconds':timings,
            'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'parquet_bytes':{p.name:p.stat().st_size for p in output.glob('*.parquet')},
            'PE_input':sum(c['input_pe'] for c in conserved),'PE_output':sum(c['output_pe'] for c in conserved),
            'response_level':'user_accepted_detector_proxy'})
        print(f'shard {shard_id}: {len(order)} events',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('plan')
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--input',type=Path,default=Path('/eos/project-f/fasersim-bonn/public/pinpoint/data'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--chunks',type=int,default=32)
    p.add_argument('--train-per-class',type=int,default=20000)
    p.add_argument('--val-per-class',type=int,default=2500)
    p.add_argument('--seed',type=int,default=42)
    p=sub.add_parser('convert')
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--task',type=int,required=True)
    p.add_argument('--cpus',type=int,default=2)
    p.add_argument('--output-root',type=Path)
    p.add_argument('--smoke-per-class',type=int,default=0)
    args=parser.parse_args()
    (plan if args.command=='plan' else convert)(args)


if __name__=='__main__':
    main()
