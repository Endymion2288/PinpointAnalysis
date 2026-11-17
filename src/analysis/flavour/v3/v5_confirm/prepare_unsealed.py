"""Post-freeze ROOT-to-Parquet preparation for Cal-A40k, Cut-B160k, and Confirm200k."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import uproot

from analysis.flavour.create_data import BRANCHES, ROOT_TREES, lists
from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.readout import bar_channels, modality_state, pixel_channels, tracker_measurements
from analysis.flavour.s2.prepare import regions, tau_branches
from analysis.flavour.s5.truth import charmed
from analysis.flavour.s6.conditions import select_families
from analysis.flavour.v2.partitions import FAMILY, KEY, equal_identity_rows, row_keys
from analysis.flavour.v2.r1.prepare import TRUTH, assign_shards
from analysis.flavour.v3.common.datasets import EVAL_PROVENANCE_COLUMNS
from analysis.flavour.v3.common.domains import compute_observable_containment, detector_bounds
from analysis.flavour.v3.v5_confirm import DEFAULT_V2_ROOT, DEFAULT_V3_ROOT
from analysis.flavour.v3.v5_confirm.freeze import require_v5_freeze


SPLIT_SEEDS = {'cal_a': 10000, 'cut_b': 20000, 'confirm': 30000}


def _charm_flags_for_chunk(root_file, evt_ids: pd.Series) -> dict[int, bool]:
    """Identify events with any primary charm hadron from the light primaries tree."""
    arr = root_file['primaries'].arrays(['evtID', 'PDG'], library='np')
    ev = arr['evtID']
    pdg = arr['PDG']
    target_set = {int(e) for e in evt_ids}
    out = {e: False for e in target_set}
    charm_mask = np.fromiter((charmed(p) for p in pdg), dtype=bool, count=len(pdg))
    for e in ev[charm_mask]:
        ei = int(e)
        if ei in out:
            out[ei] = True
    return out


def plan_unsealed(
    *,
    v3_root: Path = DEFAULT_V3_ROOT,
    v2_root: Path = DEFAULT_V2_ROOT,
    freeze_path: Path | None = None,
    verify_git_commit: bool = True,
) -> dict:
    """Create conversion manifests/plans for Cal-A40k, Cut-B160k, and Confirm200k after V5 freeze."""
    w3_12 = v3_root / 'W3-12'
    if freeze_path is None:
        freeze_path = w3_12 / 'analysis_freeze_v5.json'
    require_v5_freeze(freeze_path, verify_git_commit=verify_git_commit, verify_checkpoints=True)

    plan_path = w3_12 / 'preparation_plan.json'
    if plan_path.exists():
        return json.loads(plan_path.read_text())

    partitions = json.loads((v2_root / 'partitions.json').read_text())
    cfg = partitions['config']
    geometry_path = Path(cfg['geometry'])
    geometry = json.loads(geometry_path.read_text())
    input_root = Path(cfg['input_root'])

    # Verify strict family disjointness against Train300k & Tune40k BEFORE joining
    used_families: set[tuple[int, int]] = set()
    used_keys: set[tuple[int, int, int]] = set()
    for p_name in ('train_300000', 'tune'):
        df_prev = pd.read_parquet(partitions['manifests'][p_name]['manifest'], columns=KEY + FAMILY)
        used_families.update(zip(df_prev['family_hi'].astype(int), df_prev['family_lo'].astype(int)))
        used_keys.update(zip(df_prev['run'].astype(int), df_prev['chunk'].astype(int), df_prev['row'].astype(int)))

    inventory = pd.read_parquet(cfg['inventory'], columns=KEY + FAMILY + ['evtID', 'label'] + TRUTH)
    inv_lookup = pd.Index(row_keys(inventory))

    split_sources = (
        ('cal_a', Path(partitions['manifests']['cal_a']['manifest']), 40000),
        ('cut_b', Path(partitions['manifests']['cut_b']['manifest']), 160000),
        ('confirm', v2_root / 'sealed' / 'confirm_rows.parquet', 200000),
    )

    all_tasks: list[dict] = []
    manifest_parts: list[pd.DataFrame] = []
    counts_by_split: dict[str, dict] = {}
    regions_def = None

    for split_name, src_path, expected_n in split_sources:
        raw_df = pd.read_parquet(src_path)
        if len(raw_df) != expected_n:
            raise ValueError(f'Split {split_name} expected {expected_n} rows, found {len(raw_df)}')
        if 'split' in raw_df.columns:
            if not (raw_df['split'] == split_name).all():
                raise ValueError(f'Split column mismatch inside {src_path}')
        raw_df['split'] = split_name

        fams = set(zip(raw_df['family_hi'].astype(int), raw_df['family_lo'].astype(int)))
        keys = set(zip(raw_df['run'].astype(int), raw_df['chunk'].astype(int), raw_df['row'].astype(int)))
        if fams & used_families:
            raise RuntimeError(f'Family leakage detected involving split {split_name}')
        if keys & used_keys:
            raise RuntimeError(f'Event key overlap detected involving split {split_name}')
        used_families.update(fams)
        used_keys.update(keys)

        pos = inv_lookup.get_indexer(row_keys(raw_df))
        if (pos < 0).any():
            raise ValueError(f'Split {split_name} rows absent from accepted inventory')
        actual = inventory.iloc[pos]
        if any(not np.array_equal(actual[c].to_numpy(), raw_df[c].to_numpy()) for c in FAMILY + ['label']):
            raise ValueError(f'Inventory identity/label mismatch in split {split_name}')
        for col in ['evtID'] + TRUTH:
            raw_df[col] = actual[col].to_numpy()

        px_cen, l_scint, r_def = regions(raw_df, geometry)
        regions_def = r_def
        raw_df['pixel_centered'] = px_cen
        raw_df['large_scintillator'] = l_scint
        raw_df['energy_nu_GeV'] = (raw_df['initE'].to_numpy(dtype=np.float64) / 1000.0).astype(np.float64)
        raw_df['is_E_le_2p3TeV'] = raw_df['energy_nu_GeV'] <= 2300.0
        raw_df['is_highE_non_tau'] = (raw_df['label'].to_numpy(dtype=np.int64) != 2) & (raw_df['energy_nu_GeV'] > 2300.0)

        sharded, n_shards = assign_shards(raw_df, 0, SPLIT_SEEDS[split_name], target=2500)
        sharded = sharded.sort_values(KEY).reset_index(drop=True)
        split_dir = w3_12 / f'data_{split_name}'
        (split_dir / 'shards').mkdir(parents=True, exist_ok=True)
        sharded.to_parquet(split_dir / 'internal_manifest.parquet', index=False, compression=COMPRESSION)
        split_plan = {
            'stage': 'W3-12',
            'split': split_name,
            'geometry': str(geometry_path),
            'input_root': str(input_root),
            'events': int(len(sharded)),
            'shards': int(n_shards),
            'seed': SPLIT_SEEDS[split_name],
        }
        write_json(split_dir / 'plan.json', split_plan)
        for s_id in range(n_shards):
            sub_cnt = int((sharded['shard'] == s_id).sum())
            all_tasks.append({'split': split_name, 'shard': int(s_id), 'events': sub_cnt})
        counts_by_split[split_name] = {
            'events': int(len(sharded)),
            'shards': int(n_shards),
            'families': int(len(fams)),
            'per_class': {str(int(k)): int(v) for k, v in sharded['label'].value_counts().sort_index().items()},
        }
        manifest_parts.append(sharded)

    (w3_12 / 'offline_truth_meta' / 'parts').mkdir(parents=True, exist_ok=True)
    plan_obj = {
        'status': 'planned',
        'UTC': datetime.now(timezone.utc).isoformat(),
        'geometry': str(geometry_path),
        'input_root': str(input_root),
        'freeze_path': str(freeze_path),
        'regions': regions_def,
        'splits': counts_by_split,
        'tasks': all_tasks,
    }
    write_json(plan_path, plan_obj)
    return plan_obj


def convert_shard_task(
    task: dict,
    *,
    w3_12: Path,
    geometry: dict,
    bounds: dict,
    input_root: Path,
    cpus: int = 4,
) -> dict:
    """Convert one (split, shard) from ROOT into isolated observation shards + offline truth meta."""
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    split = str(task['split'])
    shard = int(task['shard'])
    split_dir = w3_12 / f'data_{split}'
    dest = split_dir / 'shards' / f'{shard:04d}'
    meta_out = w3_12 / 'offline_truth_meta' / 'parts' / f'{split}_{shard:04d}.parquet'
    sum_path = dest / 'summary.json'

    frame = pd.read_parquet(
        split_dir / 'internal_manifest.parquet',
        filters=[('shard', '=', shard)],
    )
    if sum_path.exists() and meta_out.exists():
        prev = json.loads(sum_path.read_text())
        if prev.get('events') == len(frame) and all(
            (dest / fn).is_file() for fn in ('observations.parquet', 'targets.parquet', 'provenance.parquet', 'ideal_tracker.parquet')
        ):
            return prev

    started = time.perf_counter()
    observations, targets, prov_rows, ideal_rows, truth_rows = [], [], [], [], []
    pe_in_sum, pe_out_sum = 0, 0
    edep_max_err = 0.0

    with ThreadPoolExecutor(max_workers=max(1, cpus - 1)) as executor:
        for (run, chunk), group in frame.groupby(['run', 'chunk'], sort=True):
            path = input_root / str(int(run)) / f'{int(run)}_{int(chunk):03d}.root'
            with uproot.open(
                path,
                decompression_executor=executor,
                interpretation_executor=executor,
                object_cache=None,
                array_cache=None,
            ) as root:
                branches = tau_branches(root, group['evtID']) if int(run) == 10083 else {}
                charm_map = _charm_flags_for_chunk(root, group['evtID'])
                for block, selected in group.groupby(group['row'] // 32, sort=True):
                    start = int(block) * 32
                    stop = min(start + 32, root['event'].num_entries)
                    batch = {
                        name: root[ROOT_TREES[name]].arrays(keys, entry_start=start, entry_stop=stop, library='ak')
                        for name, keys in BRANCHES.items()
                    }
                    for rec in selected.to_dict('records'):
                        i = int(rec['row']) - start
                        pixel = pixel_channels(batch['pixel'][i], geometry)
                        bars, check = bar_channels(batch['bar'][i], geometry)
                        seed = int(rec['family_hi']) ^ int(rec['family_lo']) ^ 20261111
                        ideal, tracker = tracker_measurements(batch['tracker'][i], seed)
                        state = modality_state(pixel, bars, tracker)

                        pe_in_sum += int(check['input_pe'])
                        pe_out_sum += int(check['output_pe'])
                        if check['input_pe'] != check['output_pe']:
                            raise ValueError(f'PE non-conservation in {(split, shard, run, chunk, rec["row"])}')
                        edep_err = abs(float(check['input_edep_MeV']) - float(check['output_edep_MeV']))
                        if not np.isclose(check['input_edep_MeV'], check['output_edep_MeV'], rtol=1e-10, atol=1e-8):
                            raise ValueError(f'Edep non-conservation ({edep_err})')
                        edep_max_err = max(edep_max_err, edep_err)

                        obs_entry = {
                            'pixel': lists(pixel),
                            'bar': lists(bars),
                            'tracker_hypothesis': lists(tracker),
                            'has_response': state['has_response'].tolist(),
                            'outside_acceptance': state['outside_acceptance'].tolist(),
                            'fit_attempted': bool(state['fit_attempted']),
                            'momentum_censored': int(state['momentum_censored']),
                        }
                        dom = compute_observable_containment(obs_entry, geometry, bounds)
                        observations.append(obs_entry)
                        targets.append({'label': int(rec['label']), 'label_status': 'consistent'})
                        prov_rows.append({col: rec[col] for col in EVAL_PROVENANCE_COLUMNS})
                        ideal_rows.append(lists(ideal))
                        truth_rows.append({
                            'run': int(rec['run']),
                            'chunk': int(rec['chunk']),
                            'row': int(rec['row']),
                            'family_hi': int(rec['family_hi']),
                            'family_lo': int(rec['family_lo']),
                            'split': split,
                            'label': int(rec['label']),
                            'evtID': int(rec['evtID']),
                            'obs_domain_id': int(dom['obs_domain_id']),
                            'cal_start': int(dom['cal_start']),
                            'pixel_centered': bool(rec['pixel_centered']),
                            'large_scintillator': bool(rec['large_scintillator']),
                            'energy_nu_GeV': float(rec['energy_nu_GeV']),
                            'is_E_le_2p3TeV': bool(rec['is_E_le_2p3TeV']),
                            'is_highE_non_tau': bool(rec['is_highE_non_tau']),
                            'tau_branch': str(branches.get(int(rec['evtID']), 'not_tau')),
                            'has_charm': bool(charm_map.get(int(rec['evtID']), False)),
                            'pixel_hits': int(len(pixel['channel_id'])),
                            'bar_channels': int(len(bars['channel_id'])),
                            'tracker_channels': int(len(tracker['station'])),
                            'shard': shard,
                        })

    order = np.random.default_rng(SPLIT_SEEDS[split] + shard).permutation(len(observations))
    dest.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist([observations[idx] for idx in order]),
        dest / 'observations.parquet',
        compression=COMPRESSION,
        row_group_size=64,
    )
    pq.write_table(
        pa.Table.from_pylist([targets[idx] for idx in order]),
        dest / 'targets.parquet',
        compression=COMPRESSION,
    )
    pd.DataFrame([prov_rows[idx] for idx in order])[list(EVAL_PROVENANCE_COLUMNS)].to_parquet(
        dest / 'provenance.parquet',
        index=False,
        compression=COMPRESSION,
    )
    pq.write_table(
        pa.Table.from_pylist([ideal_rows[idx] for idx in order]),
        dest / 'ideal_tracker.parquet',
        compression=COMPRESSION,
    )
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([truth_rows[idx] for idx in order]).to_parquet(
        meta_out,
        index=False,
        compression=COMPRESSION,
    )
    summary = {
        'split': split,
        'shard': shard,
        'events': int(len(order)),
        'seconds': float(time.perf_counter() - started),
        'PE_input': int(pe_in_sum),
        'PE_output': int(pe_out_sum),
        'edep_max_abs_err_MeV': float(edep_max_err),
        'peak_rss_MiB': float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        'bytes': int(sum(p.stat().st_size for p in dest.glob('*.parquet'))),
    }
    write_json(sum_path, summary)
    return summary


def convert_range(
    *,
    v3_root: Path = DEFAULT_V3_ROOT,
    task_start: int = 0,
    task_count: int | None = None,
    cpus: int = 4,
    freeze_path: Path | None = None,
) -> list[dict]:
    w3_12 = v3_root / 'W3-12'
    if freeze_path is None:
        freeze_path = w3_12 / 'analysis_freeze_v5.json'
    require_v5_freeze(freeze_path, verify_git_commit=True, verify_checkpoints=True)
    plan_obj = json.loads((w3_12 / 'preparation_plan.json').read_text())
    geometry = json.loads(Path(plan_obj['geometry']).read_text())
    bounds = detector_bounds(geometry)
    input_root = Path(plan_obj['input_root'])
    tasks = plan_obj['tasks']
    stop = len(tasks) if task_count is None else min(len(tasks), task_start + task_count)
    summaries = []
    for idx in range(task_start, stop):
        s = convert_shard_task(tasks[idx], w3_12=w3_12, geometry=geometry, bounds=bounds, input_root=input_root, cpus=cpus)
        summaries.append(s)
        print(json.dumps({'task': idx, 'split': s['split'], 'shard': s['shard'], 'events': s['events'], 'seconds': round(s['seconds'], 2)}), flush=True)
    return summaries


def finalize_unsealed(
    *,
    v3_root: Path = DEFAULT_V3_ROOT,
    freeze_path: Path | None = None,
    stress_per_class: int = 2000,
    stress_seed: int = 20261117,
) -> dict:
    """Verify all 400k unsealed events, build offline truth meta, and select 8k paired stress rows."""
    w3_12 = v3_root / 'W3-12'
    if freeze_path is None:
        freeze_path = w3_12 / 'analysis_freeze_v5.json'
    require_v5_freeze(freeze_path, verify_git_commit=True, verify_checkpoints=True)
    plan_obj = json.loads((w3_12 / 'preparation_plan.json').read_text())

    meta_dir = w3_12 / 'offline_truth_meta'
    split_reports = {}
    for split in ('cal_a', 'cut_b', 'confirm'):
        split_dir = w3_12 / f'data_{split}'
        expected = pd.read_parquet(split_dir / 'internal_manifest.parquet', columns=KEY + FAMILY + ['label'])
        tasks = [t for t in plan_obj['tasks'] if t['split'] == split]
        prov_frames, truth_frames, summaries = [], [], []
        for t in tasks:
            sh = int(t['shard'])
            sh_dir = split_dir / 'shards' / f'{sh:04d}'
            s = json.loads((sh_dir / 'summary.json').read_text())
            if s['PE_input'] != s['PE_output']:
                raise RuntimeError(f'PE mismatch in {(split, sh)}')
            prov_frames.append(pd.read_parquet(sh_dir / 'provenance.parquet'))
            truth_frames.append(pd.read_parquet(meta_dir / 'parts' / f'{split}_{sh:04d}.parquet'))
            summaries.append(s)
        prov = pd.concat(prov_frames, ignore_index=True)
        truth = pd.concat(truth_frames, ignore_index=True)
        if not equal_identity_rows(prov, expected) or not equal_identity_rows(truth, expected):
            raise RuntimeError(f'Identity mismatch on finalized split {split}')
        truth = truth.sort_values(KEY).reset_index(drop=True)
        truth.to_parquet(meta_dir / f'{split}_truth_meta.parquet', index=False, compression=COMPRESSION)
        split_reports[split] = {
            'events': int(len(truth)),
            'shards': int(len(tasks)),
            'bytes': int(sum(s['bytes'] for s in summaries)),
            'task_seconds_sum': float(sum(s['seconds'] for s in summaries)),
            'PE_conserved': True,
        }
        if split == 'confirm':
            stress_df = select_families(truth, stress_per_class, stress_seed).sort_values(KEY).reset_index(drop=True)
            stress_df.to_parquet(w3_12 / 'stress_manifest.parquet', index=False, compression=COMPRESSION)

    out = {
        'status': 'complete',
        'UTC': datetime.now(timezone.utc).isoformat(),
        'splits': split_reports,
        'stress_manifest_events': int(4 * stress_per_class),
    }
    write_json(w3_12 / 'preparation_summary.json', out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)
    p_plan = sub.add_parser('plan')
    p_plan.add_argument('--v3-root', type=Path, default=DEFAULT_V3_ROOT)
    p_plan.add_argument('--v2-root', type=Path, default=DEFAULT_V2_ROOT)
    p_plan.add_argument('--freeze-path', type=Path, default=None)

    p_conv = sub.add_parser('convert')
    p_conv.add_argument('--v3-root', type=Path, default=DEFAULT_V3_ROOT)
    p_conv.add_argument('--task-start', type=int, default=0)
    p_conv.add_argument('--task-count', type=int, default=None)
    p_conv.add_argument('--cpus', type=int, default=4)
    p_conv.add_argument('--freeze-path', type=Path, default=None)

    p_fin = sub.add_parser('finalize')
    p_fin.add_argument('--v3-root', type=Path, default=DEFAULT_V3_ROOT)
    p_fin.add_argument('--freeze-path', type=Path, default=None)
    p_fin.add_argument('--stress-per-class', type=int, default=2000)

    args = parser.parse_args()
    if args.cmd == 'plan':
        print(json.dumps(plan_unsealed(v3_root=args.v3_root, v2_root=args.v2_root, freeze_path=args.freeze_path)['splits'], indent=2))
    elif args.cmd == 'convert':
        res = convert_range(v3_root=args.v3_root, task_start=args.task_start, task_count=args.task_count, cpus=args.cpus, freeze_path=args.freeze_path)
        print(json.dumps({'converted_shards': len(res), 'total_events': sum(r['events'] for r in res)}, indent=2))
    else:
        print(json.dumps(finalize_unsealed(v3_root=args.v3_root, freeze_path=args.freeze_path, stress_per_class=args.stress_per_class), indent=2))


if __name__ == '__main__':
    main()
