"""Fit real nonempty events before accepting the new S5 paths for a campaign."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.s3.health import TrainingHealth
from analysis.flavour.s4.dataset import restrict_dataset
from analysis.flavour.s5.dataset import PhysicsDataset, collate_physics
from analysis.flavour.s5.model import build_model, losses
from analysis.flavour.s5.train import device_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--task', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    config = json.loads(args.campaign.read_text())['experiments'][args.task]
    torch.manual_seed(config['seed'])
    torch.set_num_threads(1)
    device = torch.device(args.device)
    source = PhysicsDataset(args.root, 'train', {**config, 'label_budget': 1000, 'compile_local': False})
    key = ['run', 'chunk', 'row']
    catalog = pd.read_parquet(args.root/'catalog.parquet')
    joined = source.metadata[key+['label']].merge(catalog[key+['pixel_channels', 'bar_channels', 'tracker_channels']], on=key, validate='one_to_one')
    occupancy = joined[['pixel_channels', 'bar_channels', 'tracker_channels']].sum(1)
    selected = []
    balance_six = config.get('pilot_balance_six', False)
    groups = source.targets['six'] if balance_six else joined.label.to_numpy()
    classes = 6 if balance_six else 4
    for label in range(classes):
        eligible = np.flatnonzero((groups == label)&(occupancy > 0)&(occupancy < 2000)&(source.targets['six'] >= 0))
        selected.extend(eligible[:4])
    selected = np.asarray(selected)
    if len(selected) != 4*classes:
        raise ValueError('pilot requires four real nonempty events per requested class')
    source.targets = {k: v[selected] for k, v in source.targets.items()}
    source.objects = source.objects[selected]
    restrict_dataset(source, selected)
    source.object_mean = source.objects.mean(0)
    source.object_scale = np.maximum(source.objects.std(0), .001)
    items = [source[i] for i in range(len(source))]
    batches = [device_batch(collate_physics(items[i:i+4]), device) for i in range(0, len(items), 4)]
    model = build_model({**config, 'compile_local': False}).to(device)
    model.set_kinematics_reference(source.targets['kinematics'], source.targets['kinematics_mask'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.001)
    health = TrainingHealth(model)
    output = args.root/'pilot'
    output.mkdir(exist_ok=True)
    records = []
    def measure():
        model.eval()
        predictions, kinematics, logits = [], [], []
        with torch.no_grad():
            for batch in batches:
                output_values = model(batch['observation'], batch['objects'])
                predictions.append(output_values['probability'])
                kinematics.append(output_values['kinematics'])
                logits.append(output_values['logits'])
        probability = torch.cat(predictions)
        y = torch.tensor(source.metadata.label.to_numpy(), device=device)
        valid = torch.as_tensor(source.targets['kinematics_mask'], device=device)
        target = torch.as_tensor(source.targets['kinematics'], device=device)
        component_MSE = (torch.cat(kinematics)[valid]-target[valid]).square().mean(0).tolist()
        six_accuracy = float((torch.cat(logits).argmax(-1) == torch.as_tensor(source.targets['six'], device=device)).float().mean()) if config.get('six_way') else None
        return float(-probability[torch.arange(len(y), device=device), y].log().mean()), float((probability.argmax(-1) == y).float().mean()), component_MSE, six_accuracy
    initial_loss, initial_accuracy, initial_kinematics, initial_six_accuracy = measure()
    started = time.monotonic()
    rng = np.random.default_rng(config['seed'])
    for step in range(config.get('pilot_steps', 256)):
        batch = batches[int(rng.integers(len(batches)))]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sample = step % 16 == 0
        if sample:
            health.start(batch['observation'])
        prediction = model(batch['observation'], batch['objects'])
        total, _, _ = losses(model, prediction, batch['targets'], batch['observation']['label'])
        total.backward()
        if sample:
            record = health.finish(prediction['logits'], step)
            records.append(record)
            if not record['healthy']:
                write_json(output/'health.json', records)
                raise RuntimeError(str(record['failure_reasons']))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if (step+1) % 32 == 0:
            loss, accuracy, _, six_accuracy = measure()
            write_json(output/'progress.json', {'step': step+1, 'NLL': loss, 'accuracy': accuracy, 'seconds': time.monotonic()-started})
            print(json.dumps({'step': step+1, 'NLL': loss, 'accuracy': accuracy, 'six_accuracy': six_accuracy}), flush=True)
    final_loss, final_accuracy, final_kinematics, final_six_accuracy = measure()
    passed = final_loss < .7*initial_loss and final_accuracy >= .75 and all(r['healthy'] for r in records)
    if config.get('kinematics_revision', 1) >= 2:
        passed &= all(f < .5*i for f, i in zip(final_kinematics, initial_kinematics))
    if balance_six:
        passed &= final_six_accuracy >= .9
    gate = {'gate_passed': passed, 'events': len(source), 'initial_NLL': initial_loss, 'final_NLL': final_loss,
            'initial_accuracy': initial_accuracy, 'final_accuracy': final_accuracy, 'steps': config.get('pilot_steps', 256),
            'seconds': time.monotonic()-started, 'config': config, 'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
            'initial_kinematics_component_MSE': initial_kinematics, 'final_kinematics_component_MSE': final_kinematics,
            'initial_six_accuracy': initial_six_accuracy, 'final_six_accuracy': final_six_accuracy,
            'event_keys': source.metadata[key].to_dict('records'), 'calibration_test_used': False,
            'scope': f'same-event fit gate on {len(source)} real nonempty train events, all S5 paths; generalization unmeasured'}
    write_json(output/'health.json', records)
    write_json(output/'gate.json', gate)
    health.close()
    if not passed:
        raise RuntimeError('real-event fitting gate failed; architecture/training review required')


if __name__ == '__main__':
    main()
