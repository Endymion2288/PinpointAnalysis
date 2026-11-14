"""Read back internal activation evidence for every frozen inference shard."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from analysis.flavour.inventory import write_json


def summarize(args):
    plan = json.loads((args.root/'plan.json').read_text())
    names = [m['name'] for m in json.loads((args.root/'analysis_freeze.json').read_text())['candidates']['models']]
    output = {}
    for split in ['calibration', 'test']:
        summaries = []
        for task, spec in enumerate(plan['tasks']):
            if spec['split'] != split:
                continue
            summary = json.loads((args.root/'predictions/nominal'/f'{task:04d}/summary.json').read_text())
            if summary['candidates'] != names or summary['events'] != spec['events']:
                raise ValueError('inference health coverage differs')
            if set(summary['health']) != set(names) or any(not h['healthy'] for h in summary['health'].values()):
                raise ValueError('internal inference health requires review')
            summaries.append(summary)
        models = {}
        for name in names:
            layers = defaultdict(list)
            for s in summaries:
                for layer, rows in s['health'][name]['activations'].items():
                    layers[layer].extend(rows)
            models[name] = {'healthy_shards': len(summaries),
                'layers': {layer: {'sampled_forward_calls': len(rows),
                    'RMS_range': [min(v['RMS'] for v in rows), max(v['RMS'] for v in rows)],
                    'maximum_zero_fraction': max(v['zero_fraction'] for v in rows),
                    'maximum_fraction_below_minus5': max(v['fraction_below_minus5'] for v in rows),
                    'all_finite': all(v['all_finite'] for v in rows)} for layer, rows in layers.items()},
                'per_shard_probability_std_minimum_by_class': [min(s['probability_std'][name][c] for s in summaries) for c in range(4)]}
        output[split] = {'events': sum(s['events'] for s in summaries), 'shards': len(summaries), 'models': models}
    write_json(args.root/'audit/inference_health_readback.json', {'status': 'complete', 'splits': output,
        'scope': 'first batch in every shard, all supplied modules and 13 checkpoints; inference has no gradient measurement; training-gradient and fitting evidence retained in S3 repair/S4/S5',
        'all_sampled_finite': all(v['all_finite'] for s in output.values() for m in s['models'].values() for v in m['layers'].values()),
        'test_used_to_retune': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    summarize(parser.parse_args())


if __name__ == '__main__':
    main()
