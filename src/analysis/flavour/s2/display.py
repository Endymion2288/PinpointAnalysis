"""Native-response pages for reproducible S2 error groups, without rebinning."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import pyarrow as pa

from analysis.flavour.datasets import CachedNativeDataset
from analysis.flavour.inventory import write_json


CLASSES=['CC nu_e','CC nu_mu','CC nu_tau','NC']


def run(args):
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    groups_path=getattr(args,'groups_file',None) or args.root/args.evaluation_tag/'error_event_groups.parquet'
    groups=pd.read_parquet(groups_path)
    groups=groups[groups.experiment.eq(args.experiment)]
    confusion=groups[groups.example_group.str.contains('high_confidence')].groupby('example_group',sort=True).head(1)
    random=groups[groups.example_group.eq('random_reference')].head(4)
    selected=pd.concat([confusion,random]).drop_duplicates(['run','chunk','row']).head(args.pages)
    if selected.empty:
        raise ValueError('no selected error examples')
    paths=sorted(p for p in (args.root/'shards').iterdir() if (p/'observations.parquet').exists())
    native=CachedNativeDataset(paths,max_shards=2)
    metadata=[]
    for i,path in enumerate(paths):
        frame=pd.read_parquet(path/'provenance.parquet',columns=['run','chunk','row'])
        frame['native_index']=np.arange(len(frame))+native.boundaries[i]
        metadata.append(frame)
    selected=selected.merge(pd.concat(metadata),on=['run','chunk','row'],validate='one_to_one',how='left')
    output=args.root/args.evaluation_tag/'event_displays'/args.experiment
    output.mkdir(parents=True,exist_ok=True)
    with PdfPages(output/'native_error_groups.pdf') as pdf:
        for number,row in enumerate(selected.itertuples(index=False)):
            obs=native[int(row.native_index)]['observations']
            xyz=np.asarray(obs['pixel']['xyz_mm']).reshape(-1,3)
            bounds=np.asarray(obs['bar']['bounds_mm']).reshape(-1,5)
            xy=np.asarray(obs['tracker_hypothesis']['xy_mm']).reshape(-1,2)
            station=np.asarray(obs['tracker_hypothesis']['station'])
            fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
            for axis,coordinate in [(axes[0,0],0),(axes[1,0],1)]:
                if len(xyz):
                    axis.scatter(xyz[:,2],xyz[:,coordinate],s=.8,alpha=.6,rasterized=True)
                axis.set(xlabel='z [mm]',ylabel=f'{"xy"[coordinate]} [mm]',title='Native binary pixels')
            for axis,coordinate in [(axes[0,1],0),(axes[1,1],1)]:
                if len(bounds):
                    segments=np.array([np.column_stack([bounds[:,4],bounds[:,coordinate*2]]),
                                       np.column_stack([bounds[:,4],bounds[:,coordinate*2+1]])]).transpose(1,0,2)
                    collection=LineCollection(segments,array=np.log1p(obs['bar']['pe_proxy']),
                                              cmap='viridis',linewidths=.6,rasterized=True)
                    axis.add_collection(collection)
                    axis.autoscale()
                axis.set(xlabel='z [mm]',ylabel=f'{"xy"[coordinate]} readout interval [mm]',
                         title='Scintillator bar extents (color: log(1+PE))')
            if len(xy):
                for s in np.unique(station):
                    flag=station==s
                    axes[0,2].scatter(xy[flag,0],xy[flag,1],s=8,alpha=.7,rasterized=True,label=f'Station {s}')
                axes[0,2].legend(fontsize=7)
            axes[0,2].set(xlabel='x [mm]',ylabel='y [mm]',title='Accepted tracker proxy (color: station)')
            if len(xyz):
                points=axes[1,2].scatter(xyz[:,0],xyz[:,1],c=xyz[:,2],cmap='plasma',s=.8,alpha=.6,rasterized=True)
                fig.colorbar(points,ax=axes[1,2],label='z [mm]',fraction=.04)
            axes[1,2].set(xlabel='x [mm]',ylabel='y [mm]',title='Native pixel joint positions (color: z)')
            for present,panels,message in [(len(xy),[axes[0,2]],'No accepted tracker hits'),
                    (len(xyz),[axes[0,0],axes[1,0],axes[1,2]],'No active pixel channels'),
                    (len(bounds),[axes[0,1],axes[1,1]],'No active bar channels')]:
                if not present:
                    for axis in panels:
                        axis.text(.5,.5,message,ha='center',va='center',transform=axis.transAxes)
                        axis.set_axis_off()
            scores=(f'p(e, mu, tau, NC)=({row.p0:.3f}, {row.p1:.3f}, {row.p2:.3f}, {row.p3:.3f})'
                    if getattr(args,'all_class_scores',False) else f'p(tau)={row.p2:.3f}')
            fig.suptitle(f'{args.experiment}: true {CLASSES[row.label]}, predicted {CLASSES[row.prediction]}; '
                f'{scores}\nRun {row.run}, chunk {row.chunk}, row {row.row}; {row.example_group}\n'
                f'Truth diagnostics: E={row.initE/1000:.1f} GeV, vertex=({row.initX:.1f}, {row.initY:.1f}, {row.initZ:.1f}) mm',fontsize=10)
            pdf.savefig(fig,dpi=160)
            if number==0:
                fig.savefig(output/'first_example.png',dpi=140)
            plt.close(fig)
    selected.to_csv(output/'examples.csv',index=False)
    write_json(output/'summary.json',{'experiment':args.experiment,'pages':len(selected),
        'selection':'first high-confidence example per confusion pair, then four fixed random references; no manual cherry-picking',
        'observations':'native binary pixel centers, full bar intervals/PE, accepted tracker hypothesis',
        'truth_annotations':'plot-only E and vertex; absent from hardware model inputs'})
    native.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--experiment',default='b0_all_s42')
    parser.add_argument('--pages',type=int,default=20)
    parser.add_argument('--evaluation-tag',default='evaluation')
    run(parser.parse_args())


if __name__=='__main__':
    main()
