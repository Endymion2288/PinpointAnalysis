"""Replay one fixed final pass to isolate the effect of mixed-shard ordering."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.s2.dataset import SpatialBatchSampler
from analysis.flavour.s3.metrics import bootstrap_weights
from analysis.flavour.s4.stage_inputs import stage_shards
from analysis.flavour.v2.partitions import KEY, equal_identity_rows
from analysis.flavour.v2.profile import device_batch, init_worker, MemoryMonitor, shutdown_loader
from analysis.flavour.v2.r1.data import R1Dataset, collate
from analysis.flavour.v2.r1.evaluate import domain_report, interval, scalar_summary
from analysis.flavour.v2.r1.fit import gradients
from analysis.flavour.v2.r1.health import Activations
from analysis.flavour.v2.r1.interval_review import review_intervals
from analysis.flavour.v2.r1.metrics import quick_report
from analysis.flavour.v2.r1.models import branch_groups, build_model
from analysis.flavour.v2.r1.tasks import task_loss
from analysis.flavour.v2.r1.train import predict, save_predictions
from analysis.flavour.v2.r2.evaluate import domains, independent_points
from analysis.flavour.v2.r2.schedule import learning_rate
from analysis.flavour.v2.r2.train import make_loader


def read(path):
    return json.loads(path.read_text())


def remixed_batches(batches, seed):
    """Permute the same event indices without consulting their labels."""
    sizes = np.asarray([len(batch) for batch in batches], dtype='int64')
    events = np.concatenate(batches)
    order = np.random.default_rng(seed).permutation(events)
    return [part.tolist() for part in np.split(order, np.cumsum(sizes)[:-1])]


def composition(batches, labels):
    counts = np.asarray([np.bincount(labels[batch], minlength=4) for batch in batches])
    proportions = counts/np.asarray([len(batch) for batch in batches])[:, None]
    return {'events': int(counts.sum()), 'updates': len(batches),
        'class_counts': counts.sum(0).tolist(),
        'terminal_100_batches_class_counts': counts[-100:].sum(0).tolist(),
        'lag1_class_proportions': [float(np.corrcoef(proportions[:-1, c], proportions[1:, c])[0, 1])
            if len(proportions)>2 and proportions[:-1, c].std() and proportions[1:, c].std()
            else None for c in range(4)]}


def plan(root):
    folder = root/'sampling'
    folder.mkdir(exist_ok=True)
    if (folder/'contract.json').exists():
        raise FileExistsError('sampling replay contract already frozen')
    name = 'b1_d1wd_t4_s42'
    record = read(root/'experiments'/name/'manifest.json')
    config = record['config']
    checkpoint = root/'experiments'/name/'best.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    data = R1Dataset(Path(config['data_root']), 'train', config)
    sampler = SpatialBatchSampler(data, config['batch_size'], config['seed'], shuffle=True)
    sampler.epoch = 10
    if (tuple(state['position']) != (10, len(sampler))
            or not np.all(state['counts'] == 11)
            or state['record']['counters']['training_exposures'] != 1100000):
        raise ValueError('diagnostic requires the frozen end-of-pass11 prefix')
    sampler.epoch = 11
    original = list(sampler)
    remixed = remixed_batches(original, 20261118)
    original_events = np.concatenate(original)
    if (len(original_events) != len(data) or not np.array_equal(np.sort(original_events), np.arange(len(data)))
            or [len(b) for b in original] != [len(b) for b in remixed]):
        raise ValueError('tail replay must preserve every event and batch-size position')
    np.savez_compressed(folder/'orders.npz', original=original_events,
        global_random=np.concatenate(remixed), sizes=np.asarray([len(b) for b in original], dtype='int16'))
    data.metadata.to_parquet(folder/'train_keys.parquet', index=False, compression=COMPRESSION)
    start = state['record']['counters']['updates']
    total = record['counters']['updates']
    if start+len(original) != total:
        raise ValueError('tail update count differs from the original endpoint')
    lr = np.asarray([learning_rate(i, total, config['learning_rate']) for i in range(start, total)])
    contract = {'status': 'frozen_before_replay_predictions', 'UTC': datetime.now(timezone.utc).isoformat(),
        'parent': name, 'checkpoint': str(checkpoint), 'config': config,
        'tail_epoch': 11, 'parent_passes': 11, 'final_passes': 12,
        'parent_updates': start, 'total_updates': total, 'actual_replayed_updates': len(original),
        'actual_replayed_events': len(data), 'tail_LR_sum': float(lr.sum()),
        'parent_counters': state['record']['counters'], 'variants': ['original', 'global_random'],
        'global_random_seed': 20261118,
        'composition': {key: composition(batches, data.metadata.label.to_numpy())
            for key, batches in [('original', original), ('global_random', remixed)]},
        'same_events_batch_size_sequence_updates_LR_optimizer_prefix_RNG': True,
        'scope': 'conditional final-pass diagnostic from one Tune-selected fixed prefix; not fresh12-pass training or independent confirmation',
        'Cal_A_Cut_B_Confirm_payload_read': False}
    write_json(folder/'contract.json', contract)
    print(json.dumps({k: contract[k] for k in ['parent_updates', 'total_updates', 'actual_replayed_events', 'composition']}))


def keep_events(events):
    return events


class MemoryInputs(Dataset):
    def __init__(self, items, metadata):
        self.items, self.metadata = items, metadata

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def cached_inputs(data, config):
    """Transform in sequential shard order once; random training never reloads shards."""
    generator = torch.Generator().manual_seed(config['seed']+7819)
    loader = DataLoader(data, batch_size=128, shuffle=False, collate_fn=keep_events,
        num_workers=2, worker_init_fn=init_worker, generator=generator,
        multiprocessing_context='spawn', prefetch_factor=2)
    started = time.perf_counter()
    items = [item for group in loader for item in group]
    if [item['index'] for item in items] != list(range(len(data))):
        raise ValueError('transformed cache changed event order or coverage')
    bytes_used = sum(value.nbytes for item in items for value in item.values() if isinstance(value, np.ndarray))
    audit = {'events': len(items), 'array_bytes': bytes_used, 'seconds': time.perf_counter()-started,
        'storage': 'allocation-local transformed arrays; reused by both variants; not persisted on EOS'}
    cached = MemoryInputs(items, data.metadata)
    for i in np.linspace(0, len(data)-1, 32, dtype=int):
        original, reused = collate([data[int(i)]]), collate([cached[int(i)]])
        if any(not torch.equal(original[k], reused[k]) for k in original if torch.is_tensor(original[k])):
            raise ValueError('transformed cache changed actual model inputs')
    audit['scattered_exact_input_checks'] = 32
    data.native.close()
    return cached, audit


def run(root):
    folder = root/'sampling'
    contract = read(folder/'contract.json')
    config = contract['config']
    torch.set_num_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.zeros(1, device='cuda:0').sum().item()
    if torch.cuda.get_device_capability()[0] < 8:
        raise ValueError('frozen BF16 replay requires SM80')
    started = time.perf_counter()
    record = {'status': 'running', 'GPU': torch.cuda.get_device_name(),
        'allocation_id': os.environ.get('PINPOINT_JOB_ID'), 'variants': {},
        'Cal_A_Cut_B_Confirm_payload_read': False}
    diagnostic_path = root/'diagnostics/sampling_replay/manifest.json'
    write_json(diagnostic_path, record)
    parent = torch.load(contract['checkpoint'], map_location='cpu', weights_only=False)
    orders = np.load(folder/'orders.npz')
    sizes = orders['sizes']
    boundaries = np.cumsum(sizes)[:-1]
    scratch_parent = os.environ.get('PINPOINT_V2_SCRATCH', '/tmp')
    with tempfile.TemporaryDirectory(prefix='r2-sampling-', dir=scratch_parent) as scratch, MemoryMonitor() as memory:
        data_root = Path(config['data_root'])
        paths = sorted(p for p in (data_root/'shards').iterdir() if (p/'observations.parquet').exists())
        staging = stage_shards(paths, Path(scratch)/'native', workers=3)
        os.environ['PINPOINT_NATIVE_SHARD_ROOT'] = staging['local_shard_root']
        source = R1Dataset(data_root, 'train', config, cache_dir=scratch)
        if not equal_identity_rows(source.metadata, pd.read_parquet(folder/'train_keys.parquet')):
            raise ValueError('replay cache does not cover the frozen Train identities')
        cache, cache_audit = cached_inputs(source, config)
        write_json(folder/'cache_audit.json', cache_audit)
        tune = R1Dataset(data_root, 'validation', config, cache_dir=scratch)
        tune_loader, _, _ = make_loader(tune, config, False)
        selected = pd.read_parquet(root/'gap/train_screen.parquet')
        positions = pd.MultiIndex.from_frame(cache.metadata[KEY]).get_indexer(pd.MultiIndex.from_frame(selected[KEY]))
        if (positions < 0).any():
            raise ValueError('common Train diagnostic is absent from the prefix corpus')
        for variant in contract['variants']:
            destination = folder/variant
            destination.mkdir(exist_ok=False)
            batches = [part.tolist() for part in np.split(orders[variant], boundaries)]
            generator = torch.Generator().manual_seed(config['seed']+7819)
            loader = DataLoader(cache, batch_sampler=batches, collate_fn=collate,
                num_workers=0, pin_memory=True, generator=generator)
            torch.manual_seed(config['seed'])
            model = build_model(config).cuda()
            model.load_state_dict(parent['model'])
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
            optimizer.load_state_dict(copy.deepcopy(parent['optimizer']))
            initial_delta = max(float((value.cpu()-parent['model'][key]).abs().max())
                for key, value in model.state_dict().items())
            if initial_delta != 0:
                raise ValueError('replay did not start from exactly the fixed prefix weights')
            torch.set_rng_state(parent['torch_rng'])
            torch.cuda.set_rng_state(parent['cuda_rng'])
            np.random.set_state(parent['numpy_rng'])
            groups = branch_groups(model)
            watch = {name: params[0][1] for name, params in groups.items()}
            activations = Activations(model)
            counts = parent['counts'].copy()
            health, lr_sum, exposures = [], 0., 0
            model.train()
            began = time.perf_counter()
            for local_step, batch in enumerate(loader):
                step = contract['parent_updates']+local_step
                lr = learning_rate(step, contract['total_updates'], config['learning_rate'])
                for group in optimizer.param_groups:
                    group['lr'] = lr
                sampled = local_step in [0, 15] or (local_step+1) % 1000 == 0
                before = {name: value.detach().clone() for name, value in watch.items()} if sampled else {}
                if sampled:
                    activations.start()
                gpu = device_batch(batch, 'cuda:0')
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(gpu).float()
                    loss = task_loss(logits, gpu['label'], config['task'])
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite replay loss')
                loss.backward()
                grad = gradients(groups) if sampled else None
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'], error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize()
                np.add.at(counts, batch['index'].numpy(), 1)
                exposures += len(batch['index'])
                lr_sum += lr
                if sampled:
                    health.append({'update': step+1, 'gradient_squared_norms': grad,
                        'preclip_gradient_norm': float(norm), 'activations': activations.finish(),
                        'parameter_update_RMS': {name: float((value-before[name]).detach().float().square().mean().sqrt())
                            for name, value in watch.items()}})
                if (local_step+1) % 1000 == 0:
                    write_json(folder/'progress.json', {'variant': variant, 'tail_updates': local_step+1,
                        'tail_exposures': exposures, 'seconds': time.perf_counter()-began})
            train_seconds = time.perf_counter()-began
            if (exposures != contract['actual_replayed_events'] or not np.all(counts == 12)
                    or abs(lr_sum-contract['tail_LR_sum']) > 1e-10):
                raise ValueError('replay changed the frozen exposure or LR budget')
            if not all(v['finite'] for h in health for v in h['activations'].values()):
                raise ValueError('replay branch activation is nonfinite')
            if not all(any(h['gradient_squared_norms'][key] > 0 and h['parameter_update_RMS'][key] > 0 for h in health)
                    for key in groups):
                raise ValueError('replay branch failed to learn')
            torch.save({'model': model.state_dict(), 'parent': contract['checkpoint'],
                'variant': variant, 'counts': counts}, destination/'endpoint.pt')
            coverage = cache.metadata[KEY+['label', 'family_hi', 'family_lo']].copy()
            coverage['exposures'] = counts
            coverage.to_parquet(destination/'train_exposures.parquet', index=False, compression=COMPRESSION)
            indices, p, logits, seconds = predict(model, tune_loader, config['task'], True)
            save_predictions(destination/'predictions.parquet', tune, indices, p, logits)
            metrics = quick_report(tune.metadata.iloc[indices].label.to_numpy(), p)
            train_loader = DataLoader(Subset(cache, positions.tolist()), batch_size=config['batch_size'],
                collate_fn=collate, num_workers=0, generator=torch.Generator().manual_seed(config['seed']+7819))
            ti, tp, tl, train_eval_seconds = predict(model, train_loader, config['task'], True)
            save_predictions(destination/'train_predictions.parquet', cache, ti, tp, tl)
            gap_metrics = quick_report(cache.metadata.iloc[ti].label.to_numpy(), tp)
            item = {'status': 'complete', 'variant': variant, 'metrics': metrics, 'Train_metrics': gap_metrics,
                'prefix_initial_parameter_max_delta': initial_delta, 'health': health,
                'actual_tail_exposures': exposures, 'actual_tail_updates': local_step+1,
                'actual_tail_LR_sum': lr_sum, 'train_seconds': train_seconds,
                'Tune_inference_seconds': seconds, 'Train_inference_seconds': train_eval_seconds,
                'GPU': record['GPU'], 'allocation_id': record['allocation_id']}
            write_json(destination/'manifest.json', item)
            record['variants'][variant] = {key: value for key, value in item.items() if key != 'health'}
            write_json(diagnostic_path, record)
            del model, optimizer
            torch.cuda.empty_cache()
        shutdown_loader(tune_loader)
        tune.native.close()
        del os.environ['PINPOINT_NATIVE_SHARD_ROOT']
    record.update(status='complete', wall_seconds=time.perf_counter()-started,
        memory=memory.peak, cache=cache_audit, staging=staging)
    write_json(diagnostic_path, record)
    write_json(folder/'training.json', record)
    print(json.dumps({'status': 'complete', 'variants': list(record['variants']), 'wall_seconds': record['wall_seconds']}))


def evaluate(root, workers=4):
    folder = root/'sampling'
    contract = read(folder/'contract.json')
    if read(folder/'training.json')['status'] != 'complete':
        raise ValueError('both replay variants must finish')
    parent_path = f"experiments/{contract['parent']}/predictions.parquet"
    models = [{'name': name, 'predictions': path, 'family': 'B1', 'task': 'T4', 'seed': 42}
        for name, path in [('parent_endpoint', parent_path)]
        + [(name, f'sampling/{name}/predictions.parquet') for name in contract['variants']]]
    selected_domains = ['all', 'pixel_centered', 'large_scintillator', 'occupancy_0_100']
    config = {'models': models, 'pairs': [{'a': a, 'b': b, 'reason': reason, 'domain': domain}
        for a, b, reason in [('global_random', 'original', 'same_prefix_tail_order'),
            ('original', 'parent_endpoint', 'original_tail_reproduction')]
        for domain in selected_domains]}
    catalog = pd.read_parquet(root/'data/catalog.parquet', filters=[('split', '=', 'validation')]).sort_values(KEY).reset_index(drop=True)
    frames, reports = {}, {}
    masks = domains(catalog)
    for entry in models:
        frame = pd.read_parquet(root/entry['predictions']).sort_values(KEY).reset_index(drop=True)
        if not equal_identity_rows(frame, catalog) or not np.array_equal(frame.label, catalog.label):
            raise ValueError('replay predictions changed the frozen Tune identities or labels')
        frames[entry['name']] = frame
        reports[entry['name']] = {domain: domain_report(frame, mask) for domain, mask in masks.items()}
    weights, design = bootstrap_weights(catalog, 400, 20261116)
    needed = sorted({(pair[side], pair['domain']) for pair in config['pairs'] for side in ['a', 'b']})
    def resample(key):
        name, domain = key
        mask = masks[domain]
        compute = scalar_summary(frames[name].loc[mask].reset_index(drop=True))
        return key, (compute(), np.asarray([compute(w[mask]) for w in weights]))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        cache = dict(executor.map(resample, needed))
    pairs = []
    for pair in config['pairs']:
        a, b = cache[(pair['a'], pair['domain'])], cache[(pair['b'], pair['domain'])]
        pairs.append({**pair, 'metrics': interval(a[0]-b[0], a[1]-b[1])})
    result = {'status': 'complete', 'models': reports, 'pairs': pairs, 'mean_seed_pairs': [],
        'replicates': 400, 'bootstrap': design, 'scope': contract['scope'],
        'Cal_A_Cut_B_Confirm_payload_read': False}
    destination = folder/'evaluation'
    write_json(destination/'config.json', config)
    write_json(destination/'results.json', result)
    points = independent_points(frames, reports)
    core = review_intervals(root, config, result)
    write_json(destination/'independent_points.json', points)
    write_json(destination/'independent_core_intervals.json', core)
    if points['failed'] or core['failed']:
        raise ValueError('independent sampling replay review failed')
    check = {'status': 'complete', 'models': 3, 'pairs': len(pairs),
        'independent_points': points['passed'], 'independent_core_intervals': core['passed'],
        'all_frozen_identities': True, 'same_prefix_events_batch_sizes_LR': True,
        'scope': contract['scope'], 'Cal_A_Cut_B_Confirm_payload_read': False}
    write_json(folder/'acceptance.json', check)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['parent_endpoint', 'original', 'global_random']
    labels = ['Saved endpoint', 'Original-order replay', 'Remixed-order replay']
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, domain, metric, title, scale in [
        (axes[0, 0], 'all', 'BA', 'Common three-class BA (%)', 100.),
        (axes[0, 1], 'all', 'NLL', 'Common three-class NLL', 1.),
        (axes[1, 0], 'occupancy_0_100', 'BA', 'Sparse common BA (%)', 100.),
    ]:
        ax.bar(labels, [reports[name][domain]['T3'][metric]*scale for name in names])
        ax.set(title=title)
        ax.tick_params(axis='x', labelrotation=20)
        ax.grid(axis='y', alpha=.2)
    x = np.arange(3)
    for i, name in enumerate(names):
        axes[1, 1].bar(x+(i-1)*.25,
            [100*v['recall'] for v in reports[name]['all']['T3']['per_class']],
            width=.25, label=labels[i])
    axes[1, 1].set(xticks=x, xticklabels=['Electron', 'Muon', 'NC'],
        title='Common per-class recall (%)')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(axis='y', alpha=.2)
    fig.suptitle('Fixed pass-11 prefix; same final-pass events, batch sizes and learning rates')
    fig.tight_layout()
    for suffix in ['pdf', 'png']:
        fig.savefig(root/'figures'/f'r2_sampling_replay.{suffix}', dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(check))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'run', 'evaluate'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.command == 'plan':
        plan(args.root)
    elif args.command == 'run':
        run(args.root)
    else:
        evaluate(args.root, args.workers)


if __name__ == '__main__':
    main()
