"""Offline-only Tune40k (split == 'validation') truth audit extractor for V2/V3 evaluation.

Strictly rejects sealed splits (Cal-A, Cut-B, Confirm, test) and is never imported or read
by V3EvalDataset or any model inference pass.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.v2.partitions import KEY
from analysis.flavour.v3.common.domains import detector_bounds
from analysis.flavour.v3.common.truth_sidecar import extract_shard_sidecar


DEFAULT_DATA_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R2/data')
DEFAULT_INPUT_ROOT = Path('/eos/project-f/fasersim-bonn/public/pinpoint/data')
DEFAULT_GEOMETRY = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s0_s1_2026-11-11/shards/10083/geometry.json')
DEFAULT_OUT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-08/tune40k_offline_audit')


def _worker_extract(task: tuple[str, str, dict, dict, str]) -> dict:
    shard_str, input_str, geometry, bounds, out_str = task
    return extract_shard_sidecar(
        Path(shard_str),
        Path(input_str),
        geometry,
        bounds,
        Path(out_str),
        target_split='validation',
    )


def build_tune_offline_audit(
    data_root: Path = DEFAULT_DATA_ROOT,
    input_root: Path = DEFAULT_INPUT_ROOT,
    geometry_path: Path = DEFAULT_GEOMETRY,
    output_dir: Path = DEFAULT_OUT,
    workers: int = 8,
    only_shards: list[str] | None = None,
    extract_only: bool = False,
    finalize_only: bool = False,
) -> dict:
    """Extract offline Tune40k event & pixel cluster audit tables from validation shards only."""
    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    cat = pd.read_parquet(data_root / 'catalog.parquet', columns=['split', 'shard'])
    val_shard_ids = sorted(int(s) for s in cat.loc[cat['split'] == 'validation', 'shard'].unique())
    val_shards = [f'{s:03d}' for s in val_shard_ids]

    # Explicitly verify none of the chosen shards belong to Cal-A / Cut-B / Confirm / test
    sealed_splits = set(cat.loc[cat['shard'].astype(int).isin(val_shard_ids), 'split'].unique())
    forbidden = sealed_splits - {'train', 'validation'}
    if forbidden:
        raise RuntimeError(f'Refusing validation shards touching sealed splits: {sorted(forbidden)}')

    run_shards = val_shards
    if only_shards:
        req = {f'{int(s):03d}' for s in only_shards}
        unknown = req - set(val_shards)
        if unknown:
            raise ValueError(f'Requested non-validation shards forbidden: {sorted(unknown)}')
        run_shards = [s for s in val_shards if s in req]

    geometry = json.loads(geometry_path.read_text())
    bounds = detector_bounds(geometry)
    shard_tmp = output_dir / 'shards'
    shard_tmp.mkdir(parents=True, exist_ok=True)

    if not finalize_only:
        tasks = []
        for sh_name in run_shards:
            sh_out = shard_tmp / str(sh_name)
            if (sh_out / 'summary.json').exists() and json.loads((sh_out / 'summary.json').read_text()).get('status') == 'complete':
                continue
            tasks.append((str(data_root / 'shards' / str(sh_name)), str(input_root), geometry, bounds, str(sh_out)))

        if tasks:
            if workers > 1:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    list(pool.map(_worker_extract, tasks))
            else:
                for tk in tasks:
                    _worker_extract(tk)

    if extract_only:
        return {
            'status': 'extracted_subset',
            'shards': run_shards,
            'seconds': float(time.perf_counter() - t0),
        }

    ev_frames = []
    pix_off_parts = [np.zeros(1, dtype=np.int64)]
    pix_cid_parts = []
    pix_or_parts = []
    pix_cursor = 0

    for sh_name in val_shards:
        sh_out = shard_tmp / str(sh_name)
        ev_df = pd.read_parquet(sh_out / 'event_sidecar.parquet')
        ev_frames.append(ev_df)
        npz = np.load(sh_out / 'dense_sidecar.npz')
        p_off = npz['pix_offsets'].astype(np.int64)
        pix_off_parts.append(p_off[1:] + pix_cursor)
        pix_cursor += int(p_off[-1])
        pix_cid_parts.append(npz['cluster_first_channel_id'].astype(np.uint64))
        pix_or_parts.append(npz['cluster_origin'].astype(np.int8))

    full_ev = pd.concat(ev_frames, ignore_index=True)
    full_ev['audit_index'] = np.arange(len(full_ev), dtype=np.int64)
    full_ev.to_parquet(output_dir / 'tune_event_audit.parquet', index=False, compression=COMPRESSION)

    cl_dir = output_dir / 'cluster_audit'
    cl_dir.mkdir(parents=True, exist_ok=True)
    pix_off_all = np.concatenate(pix_off_parts)
    pix_cid_all = np.concatenate(pix_cid_parts)
    pix_or_all = np.concatenate(pix_or_parts)
    pix_off_all.astype('<i8').tofile(cl_dir / 'pix_offsets.bin')
    pix_cid_all.astype('<u8').tofile(cl_dir / 'cluster_first_channel_id.bin')
    pix_or_all.astype('|i1').tofile(cl_dir / 'cluster_origin.bin')

    origin_counts = {str(int(k)): int(v) for k, v in zip(*np.unique(pix_or_all, return_counts=True))}
    summary = {
        'status': 'complete',
        'split_strict': 'validation_only',
        'events': int(len(full_ev)),
        'validation_shards': [str(s) for s in val_shards],
        'total_pixel_clusters': int(len(pix_cid_all)),
        'charm_valid': int((full_ev['charm_valid'] > 0).sum()),
        'charm_positive': int(((full_ev['charm_valid'] > 0) & (full_ev['charm_flag'] == 1)).sum()),
        'vertex_valid': int((full_ev['vertex_valid'] > 0).sum()),
        'cluster_origin_counts': origin_counts,
        'seconds': float(time.perf_counter() - t0),
        'Cal_A_Cut_B_Confirm_payload_read': False,
    }
    write_json(output_dir / 'audit_summary.json', summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--input-root', type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument('--geometry', type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--shards', nargs='*', default=None)
    parser.add_argument('--extract-only', action='store_true')
    parser.add_argument('--finalize-only', action='store_true')
    args = parser.parse_args()
    rep = build_tune_offline_audit(
        data_root=args.data_root,
        input_root=args.input_root,
        geometry_path=args.geometry,
        output_dir=args.output,
        workers=args.workers,
        only_shards=args.shards,
        extract_only=args.extract_only,
        finalize_only=args.finalize_only,
    )
    print(json.dumps(rep, indent=2))


if __name__ == '__main__':
    main()
