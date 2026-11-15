"""Explicit R3 within-factor contrasts using frozen Tune and paired blocks."""

import argparse
import json
from pathlib import Path

from analysis.flavour.v2.r2.evaluate import run


def configuration(root, path):
    contract=json.loads(path.read_text())
    entries=[]
    for item in contract['experiments']:
        experiment=Path(item['path'])
        config=json.loads((experiment/'config.json').read_text())
        for kind,table in [('endpoint','predictions.parquet'),('best_screen','best_predictions.parquet')]:
            entries.append({'name':item['id']+'_'+kind,'experiment':str(experiment),
                'family':config['family'],'task':config['task'],'seed':config['seed'],
                'design':config['design'],'kind':kind,'predictions':str(experiment/table)})
    pairs=[]
    for item in contract['contrasts']:
        for kind in ['endpoint','best_screen']:
            for domain in item['domains']:
                pairs.append({'a':item['a']+'_'+kind,'b':item['b']+'_'+kind,
                    'domain':domain,'reason':item['reason']})
    if len({e['name'] for e in entries})!=len(entries):raise ValueError('duplicate R3 model identity')
    available={e['name'] for e in entries}
    identities={(p['a'],p['b'],p['domain'],p['reason']) for p in pairs}
    if len(identities)!=len(pairs) or any(p[s] not in available for p in pairs for s in ['a','b']):
        raise ValueError('duplicate or unavailable R3 contrast')
    return {'models':entries,'pairs':pairs,'endpoint_primary':True,
        'best_screen_secondary':'minimum own-task fixed4k NLL; no full Tune checkpoint selection',
        'seed_scope':'paired physical-block intervals; realized three-seed mean/SD only for complete matched triplets',
        'contract':str(path)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--replicates',type=int,default=400)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    run(args.root,args.replicates,args.workers,False,
        evaluation_config=configuration(args.root,args.contract),output_name=args.output)


if __name__=='__main__':
    main()
