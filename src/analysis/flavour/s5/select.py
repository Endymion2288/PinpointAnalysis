"""Select complete-screen candidates and freeze a separate equal-pass confirmation."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path


def select(root, campaign, output):
    matrix = json.loads(campaign.read_text())
    rankings = []
    by_variant = {}
    for config in matrix['experiments']:
        directory = root/'experiments'/config['name']
        manifest = json.loads((directory/'manifest.json').read_text())
        replay = json.loads((directory/'readback.json').read_text())
        measured = json.loads((directory/'metrics.json').read_text())
        if manifest['status'] != 'complete' or not manifest['all_internal_health_pass'] or not replay['checkpoint_replay_pass'] or not replay['encoder_update_pass']:
            raise ValueError('screen candidate lacks complete learning/replay evidence: '+config['name'])
        by_variant[config['variant']] = config
        rankings.append({'variant': config['variant'], 'name': config['name'], 'NLL': measured['NLL'],
                         'macro_AP': sum(v['PR_AUC'] for v in measured['classwise'].values())/4,
                         'tau_AP': measured['classwise']['2']['PR_AUC']})
    rankings.sort(key=lambda r: (r['NLL'], r['name']))
    # NLL chooses one candidate; six-way is an independent FASERCal hypothesis.
    chosen = list(dict.fromkeys(['direct4', rankings[0]['variant'], 'six']))
    experiments = []
    for variant in chosen:
        for seed in [17, 42, 2026]:
            config = copy.deepcopy(by_variant[variant])
            checkpoint = Path(config['pretrained_checkpoint']).parent.parent/f's4_mae_m75_e3_s{seed}'/'best.pt'
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            config.update(name=f's5_confirm8_{variant}_n10000_s{seed}', seed=seed, epochs=8,
                          validation_workers=0, analysis_cohort='confirmation8', runtime_revision=2,
                          pretrained_checkpoint=str(checkpoint))
            experiments.append(config)
    result = {'stage': 'S5_confirmation8', 'analysis_cohort': 'confirmation8',
              'selection': 'minimum full-screen validation NLL plus preplanned six-way hypothesis and direct4; all from fresh pretrained initialization, no screen continuation',
              'experiments': experiments}
    output.write_text(json.dumps(result, indent=2)+'\n')
    record = {'utc': datetime.now(timezone.utc).isoformat(), 'screen_matrix': str(campaign), 'ranking': rankings,
              'chosen_variants': chosen, 'criterion': result['selection'], 'confirmation': str(output),
              'seeds': [17, 42, 2026], 'complete_passes': 8, 'calibration_test_used': False,
              'inference_scope': 'reused validation and seed confirmation; independent final test belongs to S6'}
    (root/'selection.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    select(args.root, args.campaign, args.output)


if __name__ == '__main__':
    main()
