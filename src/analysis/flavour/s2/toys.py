"""Controlled lattice toys for representation loss; no neutrino performance claim."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.flavour.inventory import write_json
from analysis.flavour.s2.representation import pixel_view


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    geometry=json.loads(args.geometry.read_text())
    rng=np.random.default_rng(42)
    dx=np.median(np.diff(geometry['pixel_Xpos']))
    dy=np.median(np.diff(geometry['pixel_Ypos']))
    z=np.asarray(geometry['pixel_Zpos'])
    records=[]
    # Uniform unknown sub-cell phase; native floor models finite detector lattice.
    for pitch in [0.,.1,.2,1.]:
        cell=dx if pitch==0 else pitch
        for separation in np.geomspace(.005,2.,40):
            phase=rng.uniform(0,2,size=(4000,2))
            angle=rng.uniform(0,2*np.pi,4000)
            other=phase+separation*np.column_stack([np.cos(angle),np.sin(angle)])
            native_a=np.floor(phase/[dx,dy])*[dx,dy]+np.array([dx,dy])/2
            native_b=np.floor(other/[dx,dy])*[dx,dy]+np.array([dx,dy])/2
            a=native_a if pitch==0 else np.floor(native_a/cell)
            b=native_b if pitch==0 else np.floor(native_b/cell)
            resolved=np.any(a!=b,axis=1)
            records.append({'toy':'two_track','pitch_mm':pitch,'x':separation,
                            'fraction':float(resolved.mean()),'trials':len(resolved)})
        for theta in np.geomspace(1e-5,.02,40):
            base=rng.uniform(0,2,size=(4000,1))
            # Kink just after an early instrumented plane; compare with matched
            # straight track having the same upstream direction/intercept.
            kink_z=z[3]+.5*(z[4]-z[3])
            straight=base+np.zeros((1,len(z)))
            kink=base+np.maximum(0,z-kink_z)[None,:]*theta
            a=np.floor(straight/dx)*dx+dx/2
            b=np.floor(kink/dx)*dx+dx/2
            if pitch:
                a,b=np.floor(a/cell),np.floor(b/cell)
            distinguish=np.any(a!=b,axis=1)
            records.append({'toy':'kink','pitch_mm':pitch,'x':theta*1000,
                            'fraction':float(distinguish.mean()),'trials':len(distinguish)})
        for separation in np.linspace(0,2,25):
            fractions=[]
            for trial in range(100):
                phase=rng.uniform(0,2,size=2)
                first=phase+rng.normal(0,.12,size=(100,2))
                second=phase+[separation,0]+rng.normal(0,.12,size=(100,2))
                a=np.floor(first/[dx,dy]).astype('int64')
                b=np.floor(second/[dx,dy]).astype('int64')
                if pitch:
                    a=np.floor((a+.5)*[dx,dy]/cell).astype('int64')
                    b=np.floor((b+.5)*[dx,dy]/cell).astype('int64')
                aset,bset=set(map(tuple,a)),set(map(tuple,b))
                fractions.append(len(aset&bset)/max(1,len(aset|bset)))
            records.append({'toy':'shower_overlap','pitch_mm':pitch,'x':separation,
                            'fraction':float(np.mean(fractions)),'trials':100})
    frame=pd.DataFrame(records)
    frame.to_csv(args.output/'toy_curves.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    labels={'two_track':('Track separation [mm]','Fraction in distinct occupied cells'),
            'kink':('Kink angle [mrad]','Fraction distinguishable from straight support'),
            'shower_overlap':('Shower separation [mm]','Occupied-cell intersection / union')}
    for ax,toy in zip(axes,labels):
        for pitch,group in frame[frame.toy.eq(toy)].groupby('pitch_mm'):
            ax.plot(group.x,group.fraction,label='Native' if pitch==0 else f'{pitch:g} mm')
        ax.set(xlabel=labels[toy][0],ylabel=labels[toy][1],title=toy.replace('_',' ').title(),ylim=(-.03,1.03))
        if toy!='shower_overlap':
            ax.set_xscale('log')
        ax.legend()
    fig.savefig(args.output/'toy_topology.png',dpi=160)
    fig.savefig(args.output/'toy_topology.pdf')
    plt.close(fig)
    # Exact adversarial x-y association example, including the actual transform.
    nx=len(geometry['pixel_Xpos'])
    def pixel(cells):
        cells=np.asarray(cells)
        ids=cells[:,1]*nx+cells[:,0]
        xyz=np.column_stack([np.asarray(geometry['pixel_Xpos'])[cells[:,0]],
                             np.asarray(geometry['pixel_Ypos'])[cells[:,1]],np.full(len(cells),z[0])])
        return {'channel_id':ids,'xyz_mm':xyz}
    a=pixel([[1000,1000],[1100,1100]])
    b=pixel([[1000,1100],[1100,1000]])
    av,bv=pixel_view(a,geometry,projection=True),pixel_view(b,geometry,projection=True)
    equal=all(np.array_equal(av[k],bv[k]) for k in ['xyz_mm','count','measured','view'])
    if not equal:
        raise ValueError('projection retained forbidden pairing')
    write_json(args.output/'summary.json',{'seed':42,'pixel_pitch_mm':[dx,dy],
        'assumptions':'noise-free binary plane lattice; no material scattering, detector inefficiency or fitted-track claim',
        'two_track':'fraction of distinct occupied cells; count can preserve multiplicity despite unresolved layout',
        'kink':'support difference from fixed matched upstream straight track; not a fitted-kink detection efficiency',
        'shower_overlap':'two independent Gaussian point supports sigma=0.12mm, 100 deposits each; geometry toy only',
        'projection_pairing_collision_verified':equal,'rows':len(frame),
        'scope':'representation_information_only_no_neutrino_classification_inference'})
    print(json.dumps({'rows':len(frame),'projection_collision':equal,'output':str(args.output)}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())


if __name__=='__main__':
    main()
