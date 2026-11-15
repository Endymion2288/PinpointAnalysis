"""Summarize actual optimizer-time branch gradients, updates and activations."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.v2.r3.models import build_model,branch_groups
from analysis.flavour.v2.r1.models import branch_groups as historical_groups


def summarize(experiment):
    manifest=json.loads((experiment/'manifest.json').read_text())
    if manifest['status']!='complete':raise ValueError('final health review needs the completed exposure endpoint')
    health=manifest['health']
    if not health:raise ValueError('missing actual training-time health records')
    model=build_model(manifest['config'])
    groups=branch_groups(model)
    recorded=set(health[0]['gradient_squared_norms'])
    if set(groups)!=recorded:
        # M0 and historical references recorded the aggregate image branch.
        # Use their actual grouping, never invent unmeasured stem/stage data.
        groups=historical_groups(model)
    if set(groups)!=recorded:
        raise ValueError('recorded gradient scope differs from both known grouping definitions')
    rows=[]
    for name,parameters in groups.items():
        n=sum(p.numel() for _,p in parameters)
        g=np.array([h['gradient_squared_norms'][name] for h in health])
        update=np.array([h['parameter_update_RMS'][name] for h in health])
        row={'model':experiment.name,'branch':name,'parameters':n,'health_windows':len(health),
            'positive_gradient_windows':int((g>0).sum()),'positive_update_windows':int((update>0).sum()),
            'gradient_RMS_min':float(np.sqrt(g.min()/n)),
            'gradient_RMS_median':float(np.sqrt(np.median(g)/n)),
            'gradient_RMS_max':float(np.sqrt(g.max()/n)),
            'representative_parameter_update_RMS_min':float(update.min()),
            'representative_parameter_update_RMS_median':float(np.median(update)),
            'representative_parameter_update_RMS_max':float(update.max()),
            'all_finite':bool(np.isfinite(g).all() and np.isfinite(update).all()),
            'actual_gradient_and_update_observed':bool((g>0).any() and (update>0).any())}
        rows.append(row)
    activation_names=sorted({name for h in health for name in h['activations']})
    activations={}
    for name in activation_names:
        values=[h['activations'][name] for h in health if name in h['activations']]
        activations[name]={'observations':len(values),'all_finite':all(v['finite'] for v in values),
            'RMS_min':min(v['RMS'] for v in values),'RMS_max':max(v['RMS'] for v in values),
            'maximum_absolute':max(v['maximum_absolute'] for v in values)}
    std=np.asarray([h['probability_std'] for h in health])
    return {'status':'complete','config':manifest['config'],'GPU':manifest['GPU'],
        'precision':manifest['precision'],'branches':rows,'activation_scopes':activations,
        'probability_std_min_by_class':std.min(0).tolist(),
        'probability_std_max_by_class':std.max(0).tolist(),
        'finite_activations':all(v['all_finite'] for v in activations.values()),
        'all_branches_received_gradient_and_update':all(r['all_finite'] and r['actual_gradient_and_update_observed'] for r in rows),
        'scope':'actual sampled optimizer steps; gradient RMS normalized by all branch parameters; update RMS is the runner representative tensor, not the whole branch; zero-gradient windows can be empty inputs',
        'scientific_interpretation':'gradient/update evidence measures use during fitting; frozen-feature probes measure readability; retrained ablation determines utility'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--experiments',type=Path,nargs='+',required=True)
    parser.add_argument('--attempt',required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    destination=args.root/'health'/args.attempt
    destination.mkdir(parents=True,exist_ok=False)
    models={p.name:summarize(p) for p in args.experiments}
    result={'status':'complete','models':models,'Cal_A_Cut_B_Confirm_payload_read':False}
    if not all(m['finite_activations'] and m['all_branches_received_gradient_and_update'] for m in models.values()):
        result['status']='failed'
    write_json(destination/'results.json',result)
    pd.DataFrame([row for m in models.values() for row in m['branches']]).to_csv(destination/'branches.csv',index=False)
    if result['status']!='complete':raise ValueError('actual branch learning-health evidence failed')
    print(json.dumps({'status':'complete','models':len(models)}),flush=True)


if __name__=='__main__':main()
