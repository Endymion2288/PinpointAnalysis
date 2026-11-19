"""Run independent v3 training configs concurrently on one GPU allocation."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import time

import torch
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

from analysis.flavour.inventory import write_json
from analysis.flavour.s4.stage_inputs import stage_shards
from analysis.flavour.v2.partitions import FAMILY, KEY
from analysis.flavour.v2.profile import init_worker, shutdown_loader
from analysis.flavour.v2.r3.mapped_inputs import write_mapped_items
from analysis.flavour.v3.common.datasets import V3EvalDataset
from analysis.flavour.v3.v1_b1pp.representation import extract_b1pp_item
from analysis.flavour.v3.v2_pinpointformer.pixel_tokenizer import extract_pf_item


CLUSTER_SIDECAR_FILES = (
    'pix_offsets.bin',
    'bar_offsets.bin',
    'cluster_first_channel_id.bin',
    'cluster_origin.bin',
    'cluster_ancestry_id.bin',
    'pixel_rel_target.bin',
    'bar_rel_target.bin',
    'manifest.json',
)


def _identity_collate(xs: list[dict]) -> list[dict]:
    return xs


def _build_shared_split(
    data_root: Path,
    split: str,
    out_dir: Path,
    pkl_path: Path,
    scratch: Path,
    workers: int,
    item_transform,
    ds_config: dict | None = None,
) -> dict:
    t0 = time.perf_counter()
    cfg = dict(ds_config or {})
    cfg.setdefault('cache_shards', 2)
    ds = V3EvalDataset(
        data_root,
        split=split,
        config=cfg,
        cache_dir=scratch,
        item_transform=item_transform,
    )
    loader_kw: dict = {
        'batch_sampler': BatchSampler(SequentialSampler(ds), batch_size=64, drop_last=False),
        'num_workers': workers,
        'collate_fn': _identity_collate,
    }
    if workers > 0:
        loader_kw.update(multiprocessing_context='spawn', worker_init_fn=init_worker, prefetch_factor=2)
    loader = DataLoader(ds, **loader_kw)
    try:
        write_mapped_items(loader, out_dir)
    finally:
        shutdown_loader(loader)
    ds.native.close()

    meta = ds.metadata[KEY + FAMILY + ['label']].copy().reset_index(drop=True)
    with pkl_path.open('wb') as stream:
        pickle.dump({'items_dir': str(out_dir), 'metadata': meta}, stream, protocol=5)
    return {
        'split': split,
        'events': int(len(meta)),
        'seconds': float(time.perf_counter() - t0),
    }


def _stage_cluster_sidecar_dir(src_dir: Path, dst_dir: Path, workers: int = 4) -> dict:
    t0 = time.perf_counter()
    if not src_dir.is_dir():
        raise FileNotFoundError(f'cluster sidecar directory not found: {src_dir}')
    files = [name for name in CLUSTER_SIDECAR_FILES if (src_dir / name).exists()]
    if 'pix_offsets.bin' not in files:
        raise FileNotFoundError(f'cluster sidecar missing pix_offsets.bin in {src_dir}')
    dst_dir.mkdir(parents=True, exist_ok=True)

    def _copy_one(name: str) -> int:
        shutil.copyfile(src_dir / name, dst_dir / name)
        return int((dst_dir / name).stat().st_size)

    with ThreadPoolExecutor(max_workers=min(4, max(1, workers))) as pool:
        sizes = list(pool.map(_copy_one, files))
    return {
        'source': str(src_dir),
        'local_dir': str(dst_dir),
        'bytes': int(sum(sizes)),
        'seconds': float(time.perf_counter() - t0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--configs', type=Path, nargs='+', required=True)
    parser.add_argument('--parallel', type=int, default=2)
    parser.add_argument('--share-b1pp-inputs', action='store_true')
    parser.add_argument('--share-pf-inputs', action='store_true')
    parser.add_argument('--stage-cluster-sidecar', type=Path, default=None)
    parser.add_argument('--cache-workers', type=int, default=4)
    parser.add_argument('--train-module', default='analysis.flavour.v3.v1_b1pp.train')
    args = parser.parse_args()

    configs = [json.loads(p.read_text()) for p in args.configs]
    if len({c['data_root'] for c in configs}) != 1:
        raise ValueError('shared group requires all configs to use the same data_root')

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA device not available inside allocation')
    cap = torch.cuda.get_device_capability(0)
    if any(c.get('strict_bf16', False) for c in configs) and cap[0] < 8:
        raise RuntimeError(f'Strict BF16 configs require SM80+, found SM{cap[0]}{cap[1]}')
    if args.stage_cluster_sidecar is not None and not (args.stage_cluster_sidecar / 'pix_offsets.bin').is_file():
        raise FileNotFoundError(f'Invalid --stage-cluster-sidecar directory: {args.stage_cluster_sidecar}')

    (args.root / 'groups').mkdir(parents=True, exist_ok=True)
    (args.root / 'logs').mkdir(parents=True, exist_ok=True)
    record_path = args.root / 'groups' / f'{args.name}.json'
    started = time.perf_counter()
    record = {
        'status': 'running',
        'UTC_started': datetime.now(timezone.utc).isoformat(),
        'GPU': torch.cuda.get_device_name(0),
        'capability': f'SM{cap[0]}{cap[1]}',
        'allocation_id': os.environ.get('PINPOINT_JOB_ID', 'interactive'),
        'concurrent_jobs': int(args.parallel),
        'configs': [str(p) for p in args.configs],
        'completed': [],
        'failed': [],
        'Cal_A_Cut_B_Confirm_payload_read': False,
    }
    write_json(record_path, record)

    parent = os.environ.get('PINPOINT_V3_SCRATCH', '/tmp')
    Path(parent).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'v3-grp-{args.name}-', dir=parent) as scratch:
        scratch_dir = Path(scratch)
        data_root = Path(configs[0]['data_root'])
        shard_paths = sorted(p for p in (data_root / 'shards').iterdir() if (p / 'observations.parquet').exists())
        stage = stage_shards(shard_paths, scratch_dir / 'native', workers=min(4, max(1, args.cache_workers)))
        record['shared_staging'] = stage
        write_json(record_path, record)

        env = dict(os.environ, PINPOINT_V3_SHARED_STAGE=stage['local_shard_root'])
        if args.share_b1pp_inputs and args.share_pf_inputs:
            raise ValueError('Use at most one of --share-b1pp-inputs or --share-pf-inputs')

        if args.share_b1pp_inputs or args.share_pf_inputs:
            if any(c.get('drop_modality') is not None for c in configs):
                raise ValueError('Shared input caching requires drop_modality=None for all configs')
            shared_readout_keys = (
                'quantize_xy_mm',
                'quantize_target_only',
                'pe_only',
                'bar_threshold_pe',
                'sipm_npix',
                'apply_sipm_saturation',
                'invert_sipm_saturation',
                'active_pixel_planes',
                'misalign_sigma_xy_mm',
                'misalign_seed',
                'shuffle_micro_vertex',
            )
            for rk in shared_readout_keys:
                vals = {tuple(v) if isinstance(v, list) else v for v in (c.get(rk) for c in configs)}
                if len(vals) != 1:
                    raise ValueError(f'Shared input caching requires matching {rk} across configs')
            old_stage = os.environ.get('PINPOINT_V3_SHARED_STAGE')
            os.environ['PINPOINT_V3_SHARED_STAGE'] = stage['local_shard_root']
            try:
                prep = {}
                shared_ds_cfg = {rk: configs[0].get(rk) for rk in shared_readout_keys if rk in configs[0]}
                if args.share_b1pp_inputs:
                    spec_vars = (
                        ('train', 'PINPOINT_V3_SHARED_TRAIN_B1PP', 'b1pp', extract_b1pp_item, dict(shared_ds_cfg)),
                        ('validation', 'PINPOINT_V3_SHARED_TUNE_B1PP', 'b1pp', extract_b1pp_item, dict(shared_ds_cfg)),
                    )
                    rec_key = 'shared_b1pp_prep'
                else:
                    spec_vars = (
                        ('train', 'PINPOINT_V3_SHARED_TRAIN_PF', 'pf', extract_pf_item, dict(shared_ds_cfg)),
                        ('validation', 'PINPOINT_V3_SHARED_TUNE_PF', 'pf', extract_pf_item, dict(shared_ds_cfg)),
                    )
                    rec_key = 'shared_pf_prep'
                for split, var, tag, fn, ds_cfg in spec_vars:
                    out_dir = scratch_dir / f'shared_{split}_{tag}'
                    pkl_path = scratch_dir / f'shared_{split}_{tag}.pkl'
                    prep[split] = _build_shared_split(
                        data_root, split, out_dir, pkl_path, scratch_dir, args.cache_workers, fn, ds_cfg
                    )
                    env[var] = str(pkl_path)
                    record[rec_key] = prep
                    write_json(record_path, record)
            finally:
                if old_stage is None:
                    os.environ.pop('PINPOINT_V3_SHARED_STAGE', None)
                else:
                    os.environ['PINPOINT_V3_SHARED_STAGE'] = old_stage
            shutil.rmtree(scratch_dir / 'native', ignore_errors=True)

        if args.stage_cluster_sidecar is not None:
            sc_rep = _stage_cluster_sidecar_dir(
                args.stage_cluster_sidecar,
                scratch_dir / 'cluster_sidecar',
                workers=args.cache_workers,
            )
            env['PINPOINT_V3_LOCAL_CLUSTER_SIDECAR'] = sc_rep['local_dir']
            record['cluster_sidecar_staging'] = sc_rep
            write_json(record_path, record)

        pending = list(zip(args.configs, configs))
        live: list[tuple[subprocess.Popen, dict, object, Path]] = []
        while pending or live:
            failed_names = {entry['name'] for entry in record['failed']}
            while pending and len(live) < args.parallel:
                picked = None
                for idx, (cand_path, cand_cfg) in enumerate(pending):
                    ic_raw = cand_cfg.get('init_checkpoint')
                    if not ic_raw:
                        picked = idx
                        break
                    ic_path = Path(ic_raw)
                    dep_name = ic_path.parent.name
                    if dep_name in failed_names:
                        pending.pop(idx)
                        record['failed'].append({
                            'name': cand_cfg['name'],
                            'returncode': -1,
                            'skipped_failed_dependency': dep_name,
                        })
                        write_json(record_path, record)
                        picked = -1
                        break
                    in_local_exps = (args.root / 'experiments') in ic_path.resolve().parents
                    if in_local_exps:
                        dep_manifest = ic_path.parent / 'manifest.json'
                        if (
                            ic_path.exists()
                            and dep_manifest.exists()
                            and json.loads(dep_manifest.read_text()).get('status') == 'complete'
                        ):
                            picked = idx
                            break
                    elif ic_path.exists():
                        picked = idx
                        break

                if picked == -1:
                    continue
                if picked is None:
                    if not live:
                        for _, blocked_cfg in pending:
                            record['failed'].append({
                                'name': blocked_cfg['name'],
                                'returncode': -1,
                                'unmet_init_checkpoint': blocked_cfg.get('init_checkpoint'),
                            })
                        pending.clear()
                        write_json(record_path, record)
                    break

                cfg_path, cfg = pending.pop(picked)
                manifest = args.root / 'experiments' / cfg['name'] / 'manifest.json'
                if manifest.exists() and json.loads(manifest.read_text()).get('status') == 'complete':
                    record['completed'].append({'name': cfg['name'], 'returncode': 0, 'skipped_already_complete': True})
                    write_json(record_path, record)
                    continue
                log_path = args.root / 'logs' / f"{args.name}_{cfg['name']}.log"
                log_fh = log_path.open('a')
                train_mod = str(cfg.get('train_module', args.train_module))
                cmd = [
                    sys.executable, '-m', train_mod,
                    '--root', str(args.root),
                    '--config', str(cfg_path),
                    '--resume',
                ]
                proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
                live.append((proc, cfg, log_fh, log_path))

            time.sleep(2.0)
            still = []
            for proc, cfg, log_fh, log_path in live:
                rc = proc.poll()
                if rc is None:
                    still.append((proc, cfg, log_fh, log_path))
                else:
                    log_fh.close()
                    entry = {'name': cfg['name'], 'returncode': int(rc), 'log': str(log_path)}
                    if rc == 0:
                        record['completed'].append(entry)
                    else:
                        record['failed'].append(entry)
                    write_json(record_path, record)
            live = still

    record['status'] = 'complete' if not record['failed'] else 'failed'
    record['UTC_finished'] = datetime.now(timezone.utc).isoformat()
    record['wall_seconds'] = float(time.perf_counter() - started)
    write_json(record_path, record)
    print(json.dumps(record, indent=2))
    if record['failed']:
        sys.exit(1)


if __name__ == '__main__':
    main()
