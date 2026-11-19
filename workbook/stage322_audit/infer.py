"""Run frozen models on observations read without targets or sidecars."""
import argparse
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from analysis.flavour.v3.common.domains import compute_observable_containment, detector_bounds
from analysis.flavour.v3.v1_b1pp.models import ZAlignedStereoNet
from analysis.flavour.v3.v1_b1pp.representation import collate_b1pp, extract_b1pp_item, static_plane_layout
from analysis.flavour.v3.v2_pinpointformer.pixel_tokenizer import collate_pf, extract_pf_item
from analysis.flavour.v3.v6_clean_sim.run_stage321_clean_v2_campaign import _load_pf_model

KEY = ['run', 'chunk', 'row']
MODALITIES = ['pixel', 'bar', 'tracker_hypothesis']
SEEDS = (17, 42, 2026)


def get_models(prefix, frozen, geometry, bounds, device, seeds):
    models = {}
    for seed in seeds:
        name = f'{prefix}_s{seed}'
        record = frozen[name]
        path = Path(record['path'])
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        assert digest == record['sha256'], name
        config = json.loads(Path(record['config_path']).read_text())
        if prefix == 'v2_b1pp_clean':
            model = ZAlignedStereoNet(static_plane_layout(geometry, bounds), config)
            state = torch.load(path, map_location='cpu', weights_only=False)
            model.load_state_dict(state['model'])
            model = model.to(device).eval()
        else:
            model = _load_pf_model(path, config, geometry, device)
        models[name] = model, config
    return models


def observe_shard(path, limit=None):
    parquet = pq.ParquetFile(path / 'observations.parquet')
    table = (parquet.read(columns=MODALITIES) if limit is None else
             next(parquet.iter_batches(batch_size=limit, columns=MODALITIES)))
    meta = pq.read_table(path / 'provenance.parquet', columns=KEY).slice(0, len(table))
    widths = {'xyz_mm': 3, 'bounds_mm': 5, 'xy_mm': 2, 'sigma_xy_mm': 2}
    events = [{} for _ in range(len(table))]
    for modality in MODALITIES:
        array = table[modality]
        if isinstance(table, pa.Table):
            array = array.combine_chunks()
        fields = {}
        for field in array.type:
            vector = array.field(field.name)
            offsets = vector.offsets.to_numpy()
            values = vector.flatten()
            width = widths.get(field.name)
            if width:
                values = values.flatten().to_numpy().astype(np.float32).reshape(-1, width)
            else:
                values = values.to_numpy().astype(np.int64)
            fields[field.name] = values, offsets
        for i, event in enumerate(events):
            event[modality] = {name: values[offsets[i]:offsets[i + 1]]
                               for name, (values, offsets) in fields.items()}
    return meta.to_pandas(), events


def extract(observations, fn, geometry, bounds, config, labels=None):
    items, domains = [], []
    for index, observation in enumerate(observations):
        # Match the native observation loader's coordinate precision.
        observation['pixel']['xyz_mm'] = np.asarray(observation['pixel']['xyz_mm'], dtype=np.float32).reshape(-1, 3)
        observation['bar']['bounds_mm'] = np.asarray(observation['bar']['bounds_mm'], dtype=np.float32).reshape(-1, 5)
        observation['tracker_hypothesis']['xy_mm'] = np.asarray(observation['tracker_hypothesis']['xy_mm'], dtype=np.float32).reshape(-1, 2)
        dom = compute_observable_containment(observation, geometry, bounds, config=config)
        label = -1 if labels is None else int(labels[index])
        items.append(fn(index, observation, label, dom, geometry, bounds, config))
        domains.append(dom['obs_domain_id'])
    return items, domains


@torch.inference_mode()
def predict(model, collate, items, device, kind, intervention=None):
    batch = {key: value.to(device) for key, value in collate(items).items() if torch.is_tensor(value)}
    if intervention == 'remove':
        batch = {key: value for key, value in batch.items() if key not in ('label', 'index')}
    if intervention == 'replace':
        gen = torch.Generator(device=device).manual_seed(322)
        batch['label'] = torch.randint(0, 4, (len(items),), generator=gen, device=device)
        batch['true_vertex_mm'] = torch.rand((len(items), 3), generator=gen, device=device)
        batch['track_truth_associations'] = torch.randperm(len(items), generator=gen, device=device)
        batch['event_weight'] = torch.rand(len(items), generator=gen, device=device)
    if kind == 'b1':
        return torch.softmax(model(batch).float(), dim=-1).cpu().numpy()
    output = model(batch, grad_blend=1.0, return_aux=False) if kind == 'nomad' else model(batch)
    return torch.cat((output['probs4'], output['charm_prob'].reshape(-1, 1), output['v_hat']), dim=1).cpu().numpy()


def truth_check(root, out, frozen, geometry, bounds, device):
    result = {}
    shard = root / 'data/shards/125'
    keys, observations = observe_shard(shard, 32)
    label = pq.ParquetFile(shard / 'targets.parquet').read_row_group(0, columns=['label']).to_pandas()['label'].to_numpy()[:len(keys)]
    for prefix, kind in [('v2_b1pp_clean', 'b1'), ('v2_pf_pitch_20um', 'pf'),
                         ('v2_pf_hybrid_uvtx', 'pf'), ('v2_pf_hybrid_shuffled', 'pf'),
                         ('v2_pf_hybrid_nomad', 'nomad')]:
        name, (model, config) = next(iter(get_models(prefix, frozen, geometry, bounds, device, [17]).items()))
        fn, collate = (extract_b1pp_item, collate_b1pp) if kind == 'b1' else (extract_pf_item, collate_pf)
        with_truth, domains = extract(observations, fn, geometry, bounds, config, label)
        without_truth, _ = extract(observations, fn, geometry, bounds, config)
        baseline = predict(model, collate, with_truth, device, kind)
        missing_field_error = None
        try:
            predict(model, collate, without_truth, device, kind, 'remove')
        except KeyError as error:
            missing_field_error = str(error)
        removed = predict(model, collate, without_truth, device, kind)
        replaced = predict(model, collate, without_truth, device, kind, 'replace')
        reverse = predict(model, collate, without_truth[::-1], device, kind)[::-1]
        single = predict(model, collate, without_truth[:1], device, kind)
        damage = {}
        for modality in ('pixel', 'bar', 'tracker_hypothesis', 'all'):
            corrupted = deepcopy(observations)
            for observation in corrupted:
                for drop in MODALITIES if modality == 'all' else (modality,):
                    observation[drop] = {key: values[:0] for key, values in observation[drop].items()}
            damaged_items, _ = extract(corrupted, fn, geometry, bounds, config)
            disturbed = predict(model, collate, damaged_items, device, kind)
            damage[modality] = {
                'mean_probability_total_variation': float(np.abs(disturbed[:, :4] - baseline[:, :4]).sum(axis=1).mean() / 2),
                'argmax_flips': int(np.count_nonzero(disturbed[:, :4].argmax(1) != baseline[:, :4].argmax(1))),
                'all_outputs_finite': bool(np.isfinite(disturbed).all()),
            }
        result[name] = {
            'events': len(keys), 'domains': np.bincount(domains, minlength=4).tolist(),
            'missing_field_error': missing_field_error,
            'dummy_label_max_abs': float(np.max(np.abs(removed - baseline))),
            'replace_max_abs': float(np.max(np.abs(replaced - baseline))),
            'reverse_max_abs': float(np.max(np.abs(reverse - baseline))),
            'singleton_max_abs': float(np.max(np.abs(single - baseline[:1]))),
            'probability_sd_by_class': np.std(baseline[:, :4], axis=0).tolist(),
            'drop_modality': damage,
        }
        print(name, result[name], flush=True)
    (out / 'truth_interventions.json').write_text(json.dumps(result, indent=2))


def predict_b1(root, out, frozen, geometry, bounds, device, workers):
    models = get_models('v2_b1pp_clean', frozen, geometry, bounds, device, SEEDS)
    config = next(iter(models.values()))[1]
    plan = pd.read_parquet(root / 'planned_manifest.parquet', columns=['split', 'chunk'])
    roots = {}
    for split in ('cal_a', 'cut_b', 'confirm'):
        roots[split] = [root / 'data/shards' / f'{chunk:03d}'
                        for chunk in sorted(plan.loc[plan.split == split, 'chunk'].unique())]
    for name in ('external_punchthrough', 'nominal_sys_ref', 'tau_helicity_0',
                 'w_density_0p98', 'no_magnet_shells', 'field_0p97', 'pythia6_std_ref', 'pythia6_nux'):
        roots[name] = [root / 'sys_shards' / name / 'shards/000']

    def prepare(path):
        keys, obs = observe_shard(path)
        items, domains = extract(obs, extract_b1pp_item, geometry, bounds, config)
        keys['domain_recomputed'] = domains
        return keys, items

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for split, paths in roots.items():
            target = out / f'b1_{split}.parquet'
            if target.exists():
                continue
            frames = []
            for keys, items in pool.map(prepare, paths):
                for start in range(0, len(items), 128):
                    stop = start + 128
                    take = keys.iloc[start:stop].copy()
                    batch = {k: v.to(device) for k, v in collate_b1pp(items[start:stop]).items()
                             if torch.is_tensor(v) and k != 'index'}
                    with torch.inference_mode():
                        for name, (model, _) in models.items():
                            probability = torch.softmax(model(batch).float(), dim=-1).cpu().numpy()
                            for c in range(4):
                                take[f'{name}_p{c}'] = probability[:, c]
                    frames.append(take)
            pd.concat(frames, ignore_index=True).to_parquet(target, index=False)
            print(split, sum(len(frame) for frame in frames), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('mode', choices=('truth', 'b1'))
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    geometry = json.loads((args.root / 'geometry.json').read_text())
    bounds = detector_bounds(geometry)
    frozen = json.loads((args.root / 'v7_freeze_manifest.json').read_text())['checkpoints']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.mode == 'truth':
        truth_check(args.root, args.output, frozen, geometry, bounds, device)
    else:
        predict_b1(args.root, args.output, frozen, geometry, bounds, device, args.workers)


if __name__ == '__main__':
    main()
