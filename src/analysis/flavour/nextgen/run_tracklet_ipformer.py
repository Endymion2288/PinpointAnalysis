"""Stage-I runner for Proposal B (TrackletIPFormer), reference baselines, and causal ablations.

Enforces zero-exception seal + Stage-I isolation (zero access to Tune8k/080..099 or sealed
shards 100..149) across Gate A/B local checks, HTCondor GPU training, IEEE-FP32 Stage-I &
open pure-W OOD evaluation, and paired family-bootstrap aggregation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

from analysis.flavour.datasets import CachedNativeDataset
from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.s4.stage_inputs import stage_shards
from analysis.flavour.v2.partitions import FAMILY, KEY
from analysis.flavour.v2.profile import init_worker, shutdown_loader
from analysis.flavour.v2.r3.mapped_inputs import MappedItems, write_mapped_items
from analysis.flavour.v2.submit import choose_schedd, gpu_requirement
from analysis.flavour.v3.common.calibration import _tau_efficiency_at_fpr
from analysis.flavour.v3.common.datasets import V3ObservationDataset
from analysis.flavour.v3.common.domains import compute_observable_containment, detector_bounds
from analysis.flavour.v3.common.layers import split_adamw_param_groups
from analysis.flavour.v3.common.submit import _package_source
from analysis.flavour.v3.v1_b1pp.models import ZAlignedStereoNet, b1pp_branch_groups
from analysis.flavour.v3.v1_b1pp.representation import collate_b1pp, extract_b1pp_item, static_plane_layout
from analysis.flavour.v3.v2_pinpointformer.pixel_tokenizer import collate_pf, extract_pf_item
from analysis.flavour.v3.v2_pinpointformer.train import move_batch_to_device
from analysis.flavour.v3.v2_pinpointformer.vertex_anchor import VERTEX_AXIS_SCALES_MM
from analysis.flavour.v3.v3_multitask.losses import compute_window_task_denominators
from analysis.flavour.v3.v3_multitask.models import PinpointFormerMultitask, v3_branch_groups
from analysis.flavour.v3.v3_multitask.train import _pf_v3_loss
from analysis.flavour.v3.v4_tau.models import PinpointFormerV4, pf_v4_branch_groups
from analysis.flavour.v3.v6_clean_sim.evaluate_stage317 import paired_family_bootstrap_delta
from analysis.flavour.v3.v6_clean_sim.run_stage325_remediation import _ba4_and_nll, _split_half_tail_eval
from analysis.flavour.nextgen.protocols import compute_student_t_pulls
from analysis.flavour.nextgen.seal_guard import (
    SealAuditGuard,
    SealViolationError,
    assert_unsealed_split,
    load_authorized_split_shards,
    verify_inference_truth_invariance,
)
from analysis.flavour.nextgen.tracklet_ipformer import (
    TrackletIPFormer,
    collate_tracklet_pf,
    extract_tracklet_pf_item,
    tracklet_ipformer_loss,
)


CLEAN_V3_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/clean_sim')
STAGE325_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v7_stage325_2026-11-20')
SPLITS_DIR = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/forensics/splits')
EXP_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_tracklet_ipformer')
CONDOR_DIR = Path.home() / 'condor/Pinpoint/nextgen_tracklet_ipformer'

OOD_W184_NC_CHARM = STAGE325_ROOT / 'ood_shards/gen_ood_w184_epps21_nc_charm'
OOD_RUN10085_10088 = CLEAN_V3_ROOT / 'ood_shards/gen_ood_run10085_10088'

STAGE1_FORBIDDEN_RE = re.compile(
    r'(?:^|/)shards/0(?:8\d|9\d)(?:/|$)|validation_meta|tune_screen\.parquet|tune_truth_meta\.parquet',
    re.IGNORECASE,
)

_STAGE1_HOOK_INSTALLED = False


def install_stage1_seal_guard() -> SealAuditGuard:
    """Install both SealAuditGuard (100..149/sealed) and Stage-I Tune8k/080..099 guard."""
    global _STAGE1_HOOK_INSTALLED
    guard = SealAuditGuard()
    guard.__enter__()
    if not _STAGE1_HOOK_INSTALLED:
        def _stage1_hook(event: str, args: tuple[Any, ...]) -> None:
            if event not in ('open', 'os.stat', 'os.scandir', 'os.listdir') or not args:
                return
            target = args[0]
            if isinstance(target, int):
                return
            try:
                path_str = os.fspath(target) if hasattr(target, '__fspath__') else str(target)
            except Exception:
                return
            if isinstance(path_str, bytes):
                path_str = path_str.decode('utf-8', errors='ignore')
            norm = path_str.replace('\\', '/')
            if STAGE1_FORBIDDEN_RE.search(norm):
                raise SealViolationError(f'Stage-I forbids accessing Tune8k/validation path: {norm}')

        sys.addaudithook(_stage1_hook)
        _STAGE1_HOOK_INSTALLED = True
    return guard


def get_variant_specs() -> dict[str, dict[str, Any]]:
    """Return exact matched-budget configs for all 8 Stage-I runs."""
    base_common = {
        'seed': 42,
        'task': 'T4',
        'passes': 20,
        'batch_size': 128,
        'micro_bs': 64,
        'grad_accum': 2,
        'min_lr': 1.0e-5,
        'weight_decay': 0.03,
        'warmup_fraction': 0.06,
        'ema_decay': 0.999,
        'screen_every_passes': 2,
        'pe_only': True,
        'bar_threshold_pe': 2.5,
        'sipm_npix': 14400.0,
        'apply_sipm_saturation': False,
        'invert_sipm_saturation': True,
        'y_mirror': False,
        'quantize_cluster_coords_only_mm': None,
    }
    pf_common = {
        **base_common,
        'dim': 192,
        'heads': 8,
        'n_local_layers': 4,
        'n_global_layers': 4,
        'drop_path': 0.10,
        'head_dropout': 0.15,
        'learning_rate': 8.0e-4,
        'grad_clip': 2.0,
        'enable_dense_gn2': False,
        'use_fixed_event_loss': True,
        'w_proc': 1.0,
        'w_charm': 0.25,
        'w_vtx': 0.20,
        'w_kin': 0.15,
        'weight_all_tasks': False,
    }
    tip_common = {
        **base_common,
        'dim': 176,
        'heads': 8,
        'n_rel_pre': 2,
        'n_rel_post': 2,
        'n_global_layers': 2,
        'drop_path': 0.10,
        'head_dropout': 0.15,
        'learning_rate': 8.0e-4,
        'grad_clip': 2.0,
    }
    return {
        's1_b1pp': {
            **base_common,
            'family': 'b1pp',
            'dim': 160,
            'n_long_layers': 4,
            'heads': 8,
            'drop_path': 0.10,
            'channel_dropout': 0.03,
            'learning_rate': 9.0e-4,
            'grad_clip': 1.0,
        },
        's1_pf_20um': {
            **pf_common,
            'family': 'pf_v3',
            'enable_micro_vertex': False,
        },
        's1_hybrid_nomad': {
            **pf_common,
            'family': 'pf_v4',
            'enable_micro_vertex': True,
            'model_family': 'pf_v4',
            'enable_nomad': True,
            'legacy_nomad_scale': False,
            'enable_pauc': False,
            'nomad_init_scale': 1.0,
        },
        's1_tracklet_ipformer': {
            **tip_common,
            'family': 'tracklet_ipformer',
        },
        'abl_no_skew_dca': {
            **tip_common,
            'family': 'tracklet_ipformer',
            'disable_skew_dca': True,
        },
        'abl_hard_avr_pv': {
            **tip_common,
            'family': 'tracklet_ipformer',
            'use_hard_avr_pv': True,
        },
        'abl_legacy_16d_only': {
            **tip_common,
            'family': 'tracklet_ipformer',
            'use_legacy_16d_only': True,
        },
        'abl_1mm_quant': {
            **tip_common,
            'family': 'tracklet_ipformer',
            'quantize_cluster_coords_only_mm': 1.00,
        },
    }


def _configure_runtime() -> None:
    torch.set_num_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    torch.multiprocessing.set_sharing_strategy('file_system')


def _build_model_and_ops(variant_cfg: dict[str, Any], geometry: dict) -> tuple[nn.Module, Any, Any, Any]:
    fam = variant_cfg['family']
    bounds = detector_bounds(geometry)
    if fam == 'b1pp':
        layout = static_plane_layout(geometry, bounds)
        model: nn.Module = ZAlignedStereoNet(layout, variant_cfg)
        branch_fn = b1pp_branch_groups
        item_fn = extract_b1pp_item
        collate_fn = collate_b1pp
    elif fam == 'pf_v3':
        model = PinpointFormerMultitask(geometry, variant_cfg)
        branch_fn = v3_branch_groups
        item_fn = extract_pf_item
        collate_fn = lambda evs: collate_pf(evs, mirror_rng=None, y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars'])
    elif fam == 'pf_v4':
        model = PinpointFormerV4(geometry, variant_cfg)
        branch_fn = pf_v4_branch_groups
        item_fn = extract_pf_item
        collate_fn = lambda evs: collate_pf(evs, mirror_rng=None, y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars'])
    elif fam == 'tracklet_ipformer':
        model = TrackletIPFormer(geometry, variant_cfg)
        branch_fn = lambda m: {k: [(k, p) for p in ps] for k, ps in m.branch_parameter_groups().items()}
        item_fn = extract_tracklet_pf_item
        collate_fn = lambda evs: collate_tracklet_pf(evs, mirror_rng=None, y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars'])
    else:
        raise ValueError(f'Unknown variant family: {fam}')
    return model, branch_fn, item_fn, collate_fn


def _compute_variant_loss(
    model: nn.Module,
    family: str,
    batch: dict[str, torch.Tensor],
    grad_blend: float,
    fixed_denoms: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if family == 'b1pp':
        logits = model(batch).float()
        loss = F.cross_entropy(logits, batch['label'].long())
        return loss, torch.softmax(logits, dim=-1)
    if family in ('pf_v3', 'pf_v4'):
        ctx: dict[str, Any] = {'grad_blend': grad_blend, 'task': 'T4'}
        if fixed_denoms is not None:
            ctx['fixed_denoms'] = fixed_denoms
        return _pf_v3_loss(model, batch, ctx)
    out = model(batch, return_dataclass=True)
    return tracklet_ipformer_loss(out, batch, fixed_denoms=fixed_denoms)


@torch.no_grad()
def _predict_variant_batch(model: nn.Module, family: str, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if family == 'b1pp':
        logits = model(batch).float()
        return {'probs': torch.softmax(logits, dim=-1)}
    if family in ('pf_v3', 'pf_v4'):
        out = model(batch, grad_blend=1.0, return_aux=False)
        lv = out['log_var_v'].float()
        ax_sc = out['v_hat'].new_tensor(VERTEX_AXIS_SCALES_MM, dtype=torch.float32)
        pv_scale = (torch.exp(0.5 * lv.clamp(-6.0, 6.0)) * ax_sc[None, :]).clamp(0.02, 150.0)
        return {
            'probs': out['probs4'].float(),
            'pv_pred': out['v_hat'].float(),
            'pv_scale': pv_scale,
        }
    out = model(batch, return_dataclass=True)
    return {
        'probs': out.probs4.float(),
        'pv_pred': out.v_pv_mm.float(),
        'pv_scale': out.scale_pv_mm.float(),
        'sv_disp': out.sv_disp_mm.float() if out.sv_disp_mm is not None else out.v_pv_mm.new_zeros((len(out.probs4), 3)),
    }


def _to_device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _load_stage1_split_keys() -> dict[str, pd.DataFrame]:
    assert_unsealed_split('internal_train', stage=1)
    assert_unsealed_split('internal_screen', stage=1)
    assert_unsealed_split('internal_val', stage=1)
    splits = {
        'train': pd.read_parquet(SPLITS_DIR / 'internal_train_keys.parquet'),
        'dev_screen': pd.read_parquet(SPLITS_DIR / 'internal_dev_screen_keys.parquet'),
        'val_eval': pd.read_parquet(SPLITS_DIR / 'internal_val_eval_keys.parquet'),
    }
    fams = {
        k: set(map(tuple, df[FAMILY].to_numpy().tolist()))
        for k, df in splits.items()
    }
    if fams['train'] & fams['dev_screen'] or fams['train'] & fams['val_eval'] or fams['dev_screen'] & fams['val_eval']:
        raise AssertionError('Family leakage across Stage-I split manifests!')
    return splits


def _extract_shard_worker(task: tuple) -> dict[str, list[tuple[tuple[int, int, int], dict]]]:
    shard_path_str, key_split_map, geom, bounds, cfg, fam = task
    install_stage1_seal_guard()
    shard_path = Path(shard_path_str)
    prov = pd.read_parquet(shard_path / 'provenance.parquet', columns=KEY + ['split', 'label'])
    obs_list = pq.read_table(shard_path / 'observations.parquet').to_pylist()
    tgt_list = pq.read_table(shard_path / 'targets.parquet').to_pylist()
    item_fn = (
        extract_b1pp_item if fam == 'b1pp'
        else (extract_pf_item if fam in ('pf_v3', 'pf_v4') else extract_tracklet_pf_item)
    )
    out: dict[str, list[tuple[tuple[int, int, int], dict]]] = {'train': [], 'dev_screen': [], 'val_eval': []}
    for i, (r, c, rw, sp_name, lbl) in enumerate(zip(prov['run'], prov['chunk'], prov['row'], prov['split'], prov['label'])):
        if sp_name != 'train':
            continue
        k = (int(r), int(c), int(rw))
        sp_dst = key_split_map.get(k)
        if sp_dst is None:
            continue
        ob = obs_list[i]
        t_lbl = int(tgt_list[i]['label'])
        assert t_lbl == int(lbl)
        dom = compute_observable_containment(ob, geom, bounds, config=cfg)
        # Evaluation splits stay strictly label-stripped during inference item extraction; attach label only for train/eval alignment
        it = item_fn(len(out[sp_dst]), ob, t_lbl, dom, geom, bounds, cfg)
        out[sp_dst].append((k, it))
    return out


class _Stage1MappedDataset(Dataset):
    """MappedItems subset with optional Internal-Train-only event_sidecar attachment."""

    def __init__(self, items_dir: Path, sidecar_df: pd.DataFrame | None = None) -> None:
        self.items = MappedItems(items_dir)
        self.sidecar_df = sidecar_df

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        it = self.items[int(idx)]
        if self.sidecar_df is not None:
            row = self.sidecar_df.iloc[int(idx)]
            it['label_7'] = int(row['label_7'])
            it['tau_branch_valid'] = int(row['tau_branch_valid'])
            it['charm_flag'] = int(row['charm_flag'])
            it['charm_valid'] = int(row['charm_valid'])
            it['vertex_xyz_mm'] = np.array(
                [row['vertex_x_mm'], row['vertex_y_mm'], row['vertex_z_mm']],
                dtype=np.float32,
            )
            it['vertex_valid'] = int(row['vertex_valid'])
            it['vis_kinematics'] = np.array(
                [row['vis_pz_log1p'], row['vis_pt_log1p'], row['vis_cos_dphi'], row['vis_sin_dphi']],
                dtype=np.float32,
            )
            it['kin_valid'] = int(row['kin_valid'])
        return it


def _cache_tag_for_variant(cfg: dict[str, Any]) -> str:
    if cfg['family'] == 'b1pp':
        return 'b1pp_20um'
    qc = cfg.get('quantize_cluster_coords_only_mm')
    if qc is not None and float(qc) > 0.0:
        return 'tracklet_pf_1mm'
    return 'tracklet_pf_20um'


def _cache_job_specs() -> dict[str, tuple[dict[str, Any], int]]:
    specs = get_variant_specs()
    return {
        'tracklet_pf_20um': (specs['s1_tracklet_ipformer'], 8),
        'tracklet_pf_1mm': (specs['abl_1mm_quant'], 8),
        'b1pp_20um': (specs['s1_b1pp'], 2),
    }


def prepare_stage1_cache_part(tag: str, part: int, n_parts: int, workers: int = 8) -> dict[str, Any]:
    """Extract one shard-partition of a shared Stage-I MappedItems cache on HTCondor."""
    _configure_runtime()
    job_specs = _cache_job_specs()
    if tag not in job_specs:
        raise ValueError(f'Unknown cache tag {tag!r}')
    cfg, _ = job_specs[tag]
    cache_root = EXP_ROOT / '_tmp_stage1_cache'
    tag_dir = cache_root / tag
    if (tag_dir / 'READY.json').is_file():
        return json.loads((tag_dir / 'READY.json').read_text())

    parts_dir = tag_dir / '_parts'
    parts_dir.mkdir(parents=True, exist_ok=True)
    part_file = parts_dir / f'part_{part:02d}_of_{n_parts:02d}.pkl'
    if part_file.is_file():
        return {'tag': tag, 'part': part, 'n_parts': n_parts, 'status': 'cached'}

    geometry = json.loads((CLEAN_V3_ROOT / 'geometry.json').read_text())
    splits = _load_stage1_split_keys()
    auth_shards = load_authorized_split_shards(CLEAN_V3_ROOT / 'data', 'internal_train', stage=1)
    my_shards = auth_shards[part::n_parts]
    scratch_base = os.environ.get('_CONDOR_SCRATCH_DIR', '/tmp')
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f'ng-part-{tag}-{part}-', dir=scratch_base) as tmp_str:
        local_root = Path(tmp_str) / 'native'
        local_root.mkdir(parents=True, exist_ok=True)

        def _copy_shard(sp: Path) -> Path:
            dst = local_root / sp.name
            dst.mkdir(parents=True, exist_ok=True)
            for fname in ('observations.parquet', 'targets.parquet', 'provenance.parquet'):
                shutil.copyfile(sp / fname, dst / fname)
            return dst

        with ThreadPoolExecutor(max_workers=min(8, max(1, workers))) as pool:
            local_shards = list(pool.map(_copy_shard, my_shards))

        key_split_map: dict[tuple[int, int, int], str] = {}
        for sp_name, df in splits.items():
            for r, c, rw in df[KEY].to_numpy():
                key_split_map[(int(r), int(c), int(rw))] = sp_name
        bounds = detector_bounds(geometry)
        tasks = [(str(sp), key_split_map, geometry, bounds, cfg, cfg['family']) for sp in local_shards]
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
            shard_parts = list(pool.map(_extract_shard_worker, tasks))

    collected: dict[str, list[tuple[tuple[int, int, int], dict]]] = {'train': [], 'dev_screen': [], 'val_eval': []}
    for sp_res in shard_parts:
        for sp_name in ('train', 'dev_screen', 'val_eval'):
            collected[sp_name].extend(sp_res[sp_name])
    tmp_out = parts_dir / f'.tmp_part_{part:02d}_of_{n_parts:02d}.pkl'
    with tmp_out.open('wb') as f:
        pickle.dump(collected, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_out, part_file)
    return {
        'tag': tag,
        'part': part,
        'n_parts': n_parts,
        'shards': len(my_shards),
        'events': sum(len(v) for v in collected.values()),
        'seconds': float(time.perf_counter() - t0),
    }


def merge_stage1_cache_tag(tag: str, n_parts: int) -> dict[str, Any]:
    """Merge sharded part pickles into ordered MappedItems directories and READY.json."""
    _configure_runtime()
    cache_root = EXP_ROOT / '_tmp_stage1_cache'
    tag_dir = cache_root / tag
    if (tag_dir / 'READY.json').is_file():
        return json.loads((tag_dir / 'READY.json').read_text())
    parts_dir = tag_dir / '_parts'
    splits = _load_stage1_split_keys()
    t0 = time.perf_counter()
    collected: dict[str, dict[tuple[int, int, int], dict]] = {'train': {}, 'dev_screen': {}, 'val_eval': {}}
    for part in range(n_parts):
        pfile = parts_dir / f'part_{part:02d}_of_{n_parts:02d}.pkl'
        if not pfile.is_file():
            raise FileNotFoundError(f'Missing cache part {pfile}')
        with pfile.open('rb') as f:
            part_data = pickle.load(f)
        for sp_name in ('train', 'dev_screen', 'val_eval'):
            for k, it in part_data[sp_name]:
                collected[sp_name][k] = it

    for sp_name, df in splits.items():
        ordered = []
        for idx_row, (r, c, rw) in enumerate(df[KEY].to_numpy()):
            it = collected[sp_name][(int(r), int(c), int(rw))]
            it['index'] = int(idx_row)
            ordered.append(it)
        out_dir = tag_dir / f'mapped_{sp_name}'
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        batches = [ordered[i:i + 64] for i in range(0, len(ordered), 64)]
        write_mapped_items(batches, out_dir)

    meta_rep = {'tag': tag, 'n_parts': n_parts, 'merge_seconds': float(time.perf_counter() - t0)}
    write_json(tag_dir / 'READY.json', meta_rep)
    shutil.rmtree(parts_dir, ignore_errors=True)
    return meta_rep


def submit_cache_jobs(phase: str = 'parts') -> dict[str, Any]:
    """Submit sharded CPU cache extraction/merge jobs to HTCondor (bigbird20.cern.ch)."""
    repo = Path(__file__).resolve().parents[4]
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    (EXP_ROOT / 'logs').mkdir(parents=True, exist_ok=True)
    (EXP_ROOT / 'code').mkdir(parents=True, exist_ok=True)
    CONDOR_DIR.mkdir(parents=True, exist_ok=True)

    archive = EXP_ROOT / 'code' / f'nextgen_tracklet_cache_{int(time.time())}.tar.gz'
    _package_source(repo, archive)
    chosen, _, _ = choose_schedd(EXP_ROOT)
    job_specs = _cache_job_specs()
    submissions: dict[str, Any] = {}

    if phase == 'parts':
        for tag, (_, n_parts) in job_specs.items():
            if (EXP_ROOT / '_tmp_stage1_cache' / tag / 'READY.json').is_file():
                continue
            for part in range(n_parts):
                pfile = EXP_ROOT / '_tmp_stage1_cache' / tag / '_parts' / f'part_{part:02d}_of_{n_parts:02d}.pkl'
                if pfile.is_file():
                    continue
                jname = f'cache_{tag}_p{part:02d}'
                job_dir = CONDOR_DIR / jname
                job_dir.mkdir(parents=True, exist_ok=True)
                script = [
                    '#!/bin/bash',
                    'set -o pipefail',
                    'which() { command -v "$@"; }; export -f which',
                    'source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh',
                    'set -e',
                    'ls /eos/home-x/xcheng/FASER/Pinpoint/results >/dev/null',
                    'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1',
                    'export PINPOINT_V3_SCRATCH="${_CONDOR_SCRATCH_DIR:-/tmp}/ng_tip_${PINPOINT_JOB_ID}"',
                    'mkdir -p "$PINPOINT_V3_SCRATCH/code"',
                    f'tar -xzf {shlex.quote(str(archive))} -C "$PINPOINT_V3_SCRATCH/code"',
                    'export PYTHONPATH="$PINPOINT_V3_SCRATCH/code/src:${PYTHONPATH:-}"',
                    'cd "$PINPOINT_V3_SCRATCH"',
                    (
                        f'exec python3 -u -m analysis.flavour.nextgen.run_tracklet_ipformer '
                        f'--prepare-cache-part --cache-tag {shlex.quote(tag)} '
                        f'--part {part} --n-parts {n_parts} --workers 8'
                    ),
                ]
                wrapper = job_dir / 'run.sh'
                wrapper.write_text('\n'.join(script) + '\n')
                wrapper.chmod(0o755)
                sub_lines = [
                    'universe = vanilla',
                    f'executable = {wrapper}',
                    f'initialdir = {job_dir}',
                    f'output = {EXP_ROOT}/logs/{jname}.$(ClusterId).out',
                    f'error = {EXP_ROOT}/logs/{jname}.$(ClusterId).err',
                    f'log = {EXP_ROOT}/logs/{jname}.$(ClusterId).log',
                    'environment = "PINPOINT_JOB_ID=$(ClusterId).$(Process)"',
                    'request_cpus = 8',
                    'request_memory = 16GB',
                    'request_disk = 12GB',
                    'requirements = (OpSysAndVer == "AlmaLinux9")',
                    '+JobFlavour = "microcentury"',
                    'should_transfer_files = YES',
                    'when_to_transfer_output = ON_EXIT',
                    'transfer_output_files = ""',
                    'queue 1',
                ]
                sub_path = job_dir / 'job.sub'
                sub_path.write_text('\n'.join(sub_lines) + '\n')
                resp = subprocess.run(['condor_submit', '-name', chosen, str(sub_path)], capture_output=True, text=True)
                if resp.returncode != 0:
                    raise RuntimeError(f'condor_submit failed for {jname}: {resp.stderr}')
                submissions[jname] = {'schedd': chosen, 'stdout': resp.stdout.strip()}
    elif phase == 'merge':
        for tag, (_, n_parts) in job_specs.items():
            if (EXP_ROOT / '_tmp_stage1_cache' / tag / 'READY.json').is_file():
                continue
            parts_dir = EXP_ROOT / '_tmp_stage1_cache' / tag / '_parts'
            if not all((parts_dir / f'part_{p:02d}_of_{n_parts:02d}.pkl').is_file() for p in range(n_parts)):
                continue
            jname = f'merge_{tag}'
            job_dir = CONDOR_DIR / jname
            job_dir.mkdir(parents=True, exist_ok=True)
            script = [
                '#!/bin/bash',
                'set -o pipefail',
                'which() { command -v "$@"; }; export -f which',
                'source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh',
                'set -e',
                'ls /eos/home-x/xcheng/FASER/Pinpoint/results >/dev/null',
                'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1',
                'export PINPOINT_V3_SCRATCH="${_CONDOR_SCRATCH_DIR:-/tmp}/ng_tip_${PINPOINT_JOB_ID}"',
                'mkdir -p "$PINPOINT_V3_SCRATCH/code"',
                f'tar -xzf {shlex.quote(str(archive))} -C "$PINPOINT_V3_SCRATCH/code"',
                'export PYTHONPATH="$PINPOINT_V3_SCRATCH/code/src:${PYTHONPATH:-}"',
                'cd "$PINPOINT_V3_SCRATCH"',
                (
                    f'exec python3 -u -m analysis.flavour.nextgen.run_tracklet_ipformer '
                    f'--merge-cache --cache-tag {shlex.quote(tag)} --n-parts {n_parts}'
                ),
            ]
            wrapper = job_dir / 'run.sh'
            wrapper.write_text('\n'.join(script) + '\n')
            wrapper.chmod(0o755)
            sub_lines = [
                'universe = vanilla',
                f'executable = {wrapper}',
                f'initialdir = {job_dir}',
                f'output = {EXP_ROOT}/logs/{jname}.$(ClusterId).out',
                f'error = {EXP_ROOT}/logs/{jname}.$(ClusterId).err',
                f'log = {EXP_ROOT}/logs/{jname}.$(ClusterId).log',
                'environment = "PINPOINT_JOB_ID=$(ClusterId).$(Process)"',
                'request_cpus = 2',
                'request_memory = 24GB',
                'request_disk = 8GB',
                'requirements = (OpSysAndVer == "AlmaLinux9")',
                '+JobFlavour = "espresso"',
                'should_transfer_files = YES',
                'when_to_transfer_output = ON_EXIT',
                'transfer_output_files = ""',
                'queue 1',
            ]
            sub_path = job_dir / 'job.sub'
            sub_path.write_text('\n'.join(sub_lines) + '\n')
            resp = subprocess.run(['condor_submit', '-name', chosen, str(sub_path)], capture_output=True, text=True)
            if resp.returncode != 0:
                raise RuntimeError(f'condor_submit failed for {jname}: {resp.stderr}')
            submissions[jname] = {'schedd': chosen, 'stdout': resp.stdout.strip()}
    else:
        raise ValueError(f'Unknown cache phase {phase!r}')
    return submissions


def prepare_stage1_caches(workers: int = 8) -> dict[str, Any]:
    """Pre-extract shared Stage-I MappedItems caches once (eliminating 8x redundant shard extraction)."""
    _configure_runtime()
    geometry = json.loads((CLEAN_V3_ROOT / 'geometry.json').read_text())
    splits = _load_stage1_split_keys()
    cache_root = EXP_ROOT / '_tmp_stage1_cache'
    cache_root.mkdir(parents=True, exist_ok=True)
    specs = get_variant_specs()
    jobs = [
        ('tracklet_pf_20um', specs['s1_tracklet_ipformer']),
        ('tracklet_pf_1mm', specs['abl_1mm_quant']),
        ('b1pp_20um', specs['s1_b1pp']),
    ]
    report: dict[str, Any] = {}
    auth_shards = load_authorized_split_shards(CLEAN_V3_ROOT / 'data', 'internal_train', stage=1)
    with tempfile.TemporaryDirectory(prefix='ng-prep-native-', dir='/tmp') as tmp_str:
        tmp_dir = Path(tmp_str)
        local_root = tmp_dir / 'native'
        local_root.mkdir(parents=True, exist_ok=True)

        def _copy_shard(sp: Path) -> Path:
            dst = local_root / sp.name
            dst.mkdir(parents=True, exist_ok=True)
            for fname in ('observations.parquet', 'targets.parquet', 'provenance.parquet'):
                shutil.copyfile(sp / fname, dst / fname)
            return dst

        with ThreadPoolExecutor(max_workers=min(8, max(1, workers))) as pool:
            local_shards = list(pool.map(_copy_shard, auth_shards))

        key_split_map: dict[tuple[int, int, int], str] = {}
        for sp_name, df in splits.items():
            for r, c, rw in df[KEY].to_numpy():
                key_split_map[(int(r), int(c), int(rw))] = sp_name
        bounds = detector_bounds(geometry)

        for tag, cfg in jobs:
            tag_dir = cache_root / tag
            if (tag_dir / 'READY.json').is_file():
                report[tag] = json.loads((tag_dir / 'READY.json').read_text())
                continue
            if tag_dir.exists():
                shutil.rmtree(tag_dir, ignore_errors=True)
            tag_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            tasks = [(str(sp), key_split_map, geometry, bounds, cfg, cfg['family']) for sp in local_shards]
            collected: dict[str, dict[tuple[int, int, int], dict]] = {'train': {}, 'dev_screen': {}, 'val_eval': {}}
            import multiprocessing as mp
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
                parts = list(pool.map(_extract_shard_worker, tasks))
            for part in parts:
                for sp_name in ('train', 'dev_screen', 'val_eval'):
                    for k, it in part[sp_name]:
                        collected[sp_name][k] = it
            for sp_name, df in splits.items():
                ordered = []
                for idx_row, (r, c, rw) in enumerate(df[KEY].to_numpy()):
                    it = collected[sp_name][(int(r), int(c), int(rw))]
                    it['index'] = int(idx_row)
                    ordered.append(it)
                out_dir = tag_dir / f'mapped_{sp_name}'
                batches = [ordered[i:i + 64] for i in range(0, len(ordered), 64)]
                write_mapped_items(batches, out_dir)
            meta_rep = {'tag': tag, 'seconds': float(time.perf_counter() - t0)}
            write_json(tag_dir / 'READY.json', meta_rep)
            report[tag] = meta_rep
    return report


def _load_or_prepare_stage1(
    clean_root: Path,
    splits: dict[str, pd.DataFrame],
    geometry: dict,
    cfg: dict[str, Any],
    scratch_dir: Path,
    workers: int = 4,
) -> dict[str, dict[str, Any]]:
    cache_tag = _cache_tag_for_variant(cfg)
    shared_dir = EXP_ROOT / '_tmp_stage1_cache' / cache_tag
    if (shared_dir / 'READY.json').is_file():
        t0 = time.perf_counter()
        out_info: dict[str, dict[str, Any]] = {}
        for sp_name, df in splits.items():
            dst = scratch_dir / f'mapped_{sp_name}'
            shutil.copytree(shared_dir / f'mapped_{sp_name}', dst)
            out_info[sp_name] = {'dir': dst, 'count': len(df)}
        sc_all = pd.read_parquet(clean_root / 'truth_sidecar/event_sidecar.parquet')
        tr_df = splits['train']
        pos = pd.MultiIndex.from_frame(sc_all[KEY]).get_indexer(pd.MultiIndex.from_frame(tr_df[KEY]))
        if (pos < 0).any():
            raise ValueError('Internal-Train rows missing from event_sidecar.parquet')
        out_info['train']['sidecar'] = sc_all.iloc[pos].reset_index(drop=True)
        out_info['prep_seconds'] = float(time.perf_counter() - t0)
        return out_info
    return _prepare_stage1_mapped_splits(clean_root, splits, geometry, cfg, scratch_dir, workers=workers)


def _prepare_stage1_mapped_splits(
    clean_root: Path,
    splits: dict[str, pd.DataFrame],
    geometry: dict,
    cfg: dict[str, Any],
    scratch_dir: Path,
    workers: int = 4,
) -> dict[str, dict[str, Any]]:
    """Stage ONLY train shards (000..079) and extract aligned Stage-I MappedItems."""
    t0 = time.perf_counter()
    auth_shards = load_authorized_split_shards(clean_root / 'data', 'internal_train', stage=1)
    local_root = scratch_dir / 'native'
    local_root.mkdir(parents=True, exist_ok=True)

    def _copy_shard(sp: Path) -> Path:
        dst = local_root / sp.name
        dst.mkdir(parents=True, exist_ok=True)
        for fname in ('observations.parquet', 'targets.parquet', 'provenance.parquet'):
            shutil.copyfile(sp / fname, dst / fname)
        return dst

    with ThreadPoolExecutor(max_workers=min(4, max(1, workers))) as pool:
        local_shards = list(pool.map(_copy_shard, auth_shards))

    key_split_map: dict[tuple[int, int, int], str] = {}
    for sp_name, df in splits.items():
        for r, c, rw in df[KEY].to_numpy():
            key_split_map[(int(r), int(c), int(rw))] = sp_name

    bounds = detector_bounds(geometry)
    tasks = [(str(sp), key_split_map, geometry, bounds, cfg, cfg['family']) for sp in local_shards]
    collected: dict[str, dict[tuple[int, int, int], dict]] = {'train': {}, 'dev_screen': {}, 'val_eval': {}}

    if workers <= 1:
        parts = [_extract_shard_worker(t) for t in tasks]
    else:
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
            parts = list(pool.map(_extract_shard_worker, tasks))
    for part in parts:
        for sp_name in ('train', 'dev_screen', 'val_eval'):
            for k, it in part[sp_name]:
                collected[sp_name][k] = it

    shutil.rmtree(scratch_dir / 'native', ignore_errors=True)

    # Write MappedItems strictly in the exact row order of splits[sp_name]!
    out_info: dict[str, dict[str, Any]] = {}
    for sp_name, df in splits.items():
        ordered_items = []
        for idx_row, (r, c, rw) in enumerate(df[KEY].to_numpy()):
            it = collected[sp_name][(int(r), int(c), int(rw))]
            it['index'] = int(idx_row)
            ordered_items.append(it)
        if len(ordered_items) != len(df):
            raise RuntimeError(f'Mismatch for {sp_name}: {len(ordered_items)} vs {len(df)}')
        out_dir = scratch_dir / f'mapped_{sp_name}'
        # Chunk into batches of 64 for write_mapped_items
        batches = [ordered_items[i:i + 64] for i in range(0, len(ordered_items), 64)]
        write_mapped_items(batches, out_dir)
        out_info[sp_name] = {'dir': out_dir, 'count': len(ordered_items)}

    # Attach event_sidecar.parquet filtered strictly to Internal-Train keys ONLY
    sc_all = pd.read_parquet(clean_root / 'truth_sidecar/event_sidecar.parquet')
    tr_df = splits['train']
    pos = pd.MultiIndex.from_frame(sc_all[KEY]).get_indexer(pd.MultiIndex.from_frame(tr_df[KEY]))
    if (pos < 0).any():
        raise ValueError('Internal-Train rows missing from event_sidecar.parquet')
    out_info['train']['sidecar'] = sc_all.iloc[pos].reset_index(drop=True)
    out_info['prep_seconds'] = float(time.perf_counter() - t0)
    return out_info


def _make_loader(ds: Dataset, batch_size: int, shuffle: bool, seed: int, collate_fn: Any, workers: int) -> DataLoader:
    if shuffle:
        gen = torch.Generator()
        gen.manual_seed(seed)
        sampler: Any = RandomSampler(ds, generator=gen)
    else:
        sampler = SequentialSampler(ds)
    bsamp = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
    eff_w = 0 if len(ds) < 512 else max(0, min(workers, 4))
    kw: dict[str, Any] = {'batch_sampler': bsamp, 'num_workers': eff_w, 'collate_fn': collate_fn, 'pin_memory': False}
    if eff_w > 0:
        kw.update(multiprocessing_context='fork', worker_init_fn=init_worker, prefetch_factor=2)
    return DataLoader(ds, **kw)


@torch.no_grad()
def _eval_ieee_fp32(
    model: nn.Module,
    family: str,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Deterministic IEEE-FP32 evaluation (allow_tf32=False, zero autocast)."""
    prev_tf32_mm = torch.backends.cuda.matmul.allow_tf32
    prev_tf32_cd = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model.eval()
    chunks: dict[str, list[np.ndarray]] = {}
    try:
        for batch in loader:
            db = _to_device_batch(batch, device)
            with torch.amp.autocast('cuda', enabled=False):
                out = _predict_variant_batch(model, family, db)
            for k, v in out.items():
                chunks.setdefault(k, []).append(v.detach().float().cpu().numpy())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32_mm
        torch.backends.cudnn.allow_tf32 = prev_tf32_cd
    return {k: np.concatenate(vs, axis=0) for k, vs in chunks.items()}


def _eval_ood_fp32(
    model: nn.Module,
    family: str,
    ood_root: Path,
    geometry: dict,
    bounds: dict,
    cfg: dict[str, Any],
    collate_fn: Any,
    device: torch.device,
) -> np.ndarray:
    """Evaluate open pure-W OOD observations via label-free V3ObservationDataset in IEEE-FP32."""
    obs_p = ood_root / 'shards/000/observations.parquet'
    item_fn = (
        extract_b1pp_item if fam_is_b1(family)
        else (extract_pf_item if family in ('pf_v3', 'pf_v4') else extract_tracklet_pf_item)
    )
    ds = V3ObservationDataset(obs_p, CLEAN_V3_ROOT / 'geometry.json', config=cfg, item_transform=item_fn)
    loader = _make_loader(ds, batch_size=64, shuffle=False, seed=42, collate_fn=collate_fn, workers=4)
    pred = _eval_ieee_fp32(model, family, loader, device)
    return pred['probs'].astype(np.float32)


def fam_is_b1(family: str) -> bool:
    return family == 'b1pp'


def run_gate_ab_checks(output_path: Path | None = None) -> dict[str, Any]:
    """Run Gate A, Gate B (real 32 Internal-Train events), truth-invariance, and grad_accum parity."""
    from tests.test_nextgen_model_health import run_gate_a_health_check, run_gate_b_overfit_32_check

    _configure_runtime()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    geometry = json.loads((CLEAN_V3_ROOT / 'geometry.json').read_text())
    bounds = detector_bounds(geometry)
    auth_shards = load_authorized_split_shards(CLEAN_V3_ROOT / 'data', 'internal_train', stage=1)
    tr_keys = pd.read_parquet(SPLITS_DIR / 'internal_train_keys.parquet')
    sc_df = pd.read_parquet(CLEAN_V3_ROOT / 'truth_sidecar/event_sidecar.parquet')
    tr_set = set(map(tuple, tr_keys[KEY].to_numpy().tolist()))
    sc_idx = pd.MultiIndex.from_frame(sc_df[KEY])

    # Select 32 class-balanced non-empty Internal-Train events (8 per class)
    items_tip: list[dict] = []
    obs_raw_list: list[dict] = []
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for sp in auth_shards[:3]:
        prov = pd.read_parquet(sp / 'provenance.parquet', columns=KEY + ['label'])
        obs_list = pq.read_table(sp / 'observations.parquet').to_pylist()
        for r_i, (r, c, rw, lbl) in enumerate(zip(prov['run'], prov['chunk'], prov['row'], prov['label'])):
            cls = int(lbl)
            if counts[cls] >= 8 or (int(r), int(c), int(rw)) not in tr_set:
                continue
            ob = obs_list[r_i]
            dom = compute_observable_containment(ob, geometry, bounds, {'pe_only': True})
            if dom['obs_domain_id'] == 0:
                continue
            it = extract_tracklet_pf_item(len(items_tip), ob, cls, dom, geometry, bounds, {'pe_only': True})
            srow = sc_df.iloc[int(sc_idx.get_loc((int(r), int(c), int(rw))))]
            it['label_7'] = int(srow['label_7'])
            it['tau_branch_valid'] = int(srow['tau_branch_valid'])
            it['charm_flag'] = int(srow['charm_flag'])
            it['charm_valid'] = int(srow['charm_valid'])
            it['vertex_xyz_mm'] = np.array([srow['vertex_x_mm'], srow['vertex_y_mm'], srow['vertex_z_mm']], dtype=np.float32)
            it['vertex_valid'] = int(srow['vertex_valid'])
            it['vis_kinematics'] = np.array([srow['vis_pz_log1p'], srow['vis_pt_log1p'], srow['vis_cos_dphi'], srow['vis_sin_dphi']], dtype=np.float32)
            it['kin_valid'] = int(srow['kin_valid'])
            items_tip.append(it)
            obs_raw_list.append(ob)
            counts[cls] += 1
        if sum(counts.values()) == 32:
            break
    if len(items_tip) != 32:
        raise RuntimeError(f'Expected 32 balanced Internal-Train events, got {len(items_tip)}')

    b32 = _to_device_batch(
        collate_tracklet_pf(items_tip, y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars']),
        device,
    )

    specs = get_variant_specs()
    results: dict[str, Any] = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
        'variants': {},
    }

    for vname in ('s1_tracklet_ipformer', 'abl_no_skew_dca', 'abl_hard_avr_pv', 'abl_legacy_16d_only'):
        cfg = dict(specs[vname])
        torch.manual_seed(42)
        m = TrackletIPFormer(geometry, cfg).to(device)
        n_params = int(sum(p.numel() for p in m.parameters() if p.requires_grad))

        res_truth = verify_inference_truth_invariance(m, b32, atol=1.0e-7)
        res_ga = run_gate_a_health_check(
            m,
            b32,
            lambda out, b: tracklet_ipformer_loss(out, b)[0],
            expected_logits_shape=(32, 7),
        )

        # bs=1 vs bs=32 consistency
        m.eval()
        b1 = _to_device_batch(
            collate_tracklet_pf([items_tip[0]], y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars']),
            device,
        )
        with torch.inference_mode():
            p32 = m(b32)['probs4'][0]
            p1 = m(b1)['probs4'][0]
            bs1_diff = float((p32 - p1).abs().max().item())
        if bs1_diff > 1.0e-6:
            raise AssertionError(f'{vname} bs=1 vs bs=32 diff {bs1_diff:.3e} > 1e-6')

        # Gate B 32-event overfit check (4-class NLL < 0.05, acc >= 31/32)
        m_gb = TrackletIPFormer(geometry, {**cfg, 'drop_path': 0.0, 'head_dropout': 0.0}).to(device)
        res_gb = run_gate_b_overfit_32_check(
            m_gb,
            b32,
            lambda out, b: F.nll_loss(torch.log(out['probs4'].clamp(min=1e-6)), b['label']),
            steps=100,
            lr=1.5e-3,
            min_acc=31.0 / 32.0,
            max_final_loss=0.05,
            upstream_param_prefixes=('pix_in.', 'stereo_bars.'),
        )
        del m_gb

        vrec: dict[str, Any] = {
            'parameters': n_params,
            'truth_invariance': res_truth,
            'gate_a': {'passed': res_ga['passed'], 'min_grad_l2': min(res_ga['branch_grad_norms'].values())},
            'bs1_vs_bs32_diff': bs1_diff,
            'gate_b_overfit32': res_gb,
        }

        if vname == 's1_tracklet_ipformer' and device.type == 'cuda':
            # Check microbatch grad_accum=2 vs single-batch gradient equivalence in exact FP32 (CPU deterministic + GPU pad-matched)
            m.eval()
            m_cpu = copy.deepcopy(m).cpu().eval()
            b32_c = collate_tracklet_pf(items_tip, y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars'])
            mba_c = collate_tracklet_pf(items_tip[:16], y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars'])
            mbb_c = collate_tracklet_pf(items_tip[16:], y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars'])
            fd_c = compute_window_task_denominators([mba_c, mbb_c], pos_weight=3.0, weight_all_tasks=False)
            m_cpu.zero_grad(set_to_none=True)
            lf_c, _ = tracklet_ipformer_loss(m_cpu(b32_c), b32_c)
            lf_c.backward()
            gf_c = torch.cat([p.grad.detach().float().flatten() for p in m_cpu.parameters() if p.grad is not None])
            m_cpu.zero_grad(set_to_none=True)
            for mbc in (mba_c, mbb_c):
                lm_c, _ = tracklet_ipformer_loss(m_cpu(mbc), mbc, fixed_denoms=fd_c)
                lm_c.backward()
            ga_c = torch.cat([p.grad.detach().float().flatten() for p in m_cpu.parameters() if p.grad is not None])
            rel_err_cpu = float((gf_c - ga_c).norm().item() / max(gf_c.norm().item(), 1e-12))
            del m_cpu
            if rel_err_cpu >= 1.0e-4:
                raise AssertionError(f'CPU microbatch grad_accum rel_err {rel_err_cpu:.3e} >= 1e-4')
            vrec['grad_accum_cpu_rel_err'] = rel_err_cpu

            prev_tf32 = torch.backends.cuda.matmul.allow_tf32
            prev_cd_tf32 = torch.backends.cudnn.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            m.zero_grad(set_to_none=True)
            l_full, _ = tracklet_ipformer_loss(m(b32), b32)
            l_full.backward()
            g_full = torch.cat([p.grad.detach().float().flatten() for p in m.parameters() if p.grad is not None])

            mb_a = _to_device_batch(mba_c, device)
            mb_b = _to_device_batch(mbb_c, device)
            fd = compute_window_task_denominators([mb_a, mb_b], pos_weight=3.0, weight_all_tasks=False)
            m.zero_grad(set_to_none=True)
            for mb in (mb_a, mb_b):
                l_mb, _ = tracklet_ipformer_loss(m(mb), mb, fixed_denoms=fd)
                l_mb.backward()
            g_acc = torch.cat([p.grad.detach().float().flatten() for p in m.parameters() if p.grad is not None])
            rel_err_gpu = float((g_full - g_acc).norm().item() / max(g_full.norm().item(), 1e-12))
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
            torch.backends.cudnn.allow_tf32 = prev_cd_tf32
            if rel_err_gpu >= 5.0e-4:
                raise AssertionError(f'GPU microbatch grad_accum rel_err {rel_err_gpu:.3e} >= 5e-4')
            vrec['grad_accum_gpu_rel_err'] = rel_err_gpu

            # VRAM & throughput microbenchmark (bs=64 FP32 if >= 4 GiB free on shared GPU, else bs=32 FP32)
            del l_full, l_mb, g_full, g_acc, mb_a, mb_b
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            free_bytes, _ = torch.cuda.mem_get_info(device)
            bench_items = (items_tip + items_tip) if free_bytes >= int(4.2 * (1024 ** 3)) else items_tip
            b_bench = _to_device_batch(
                collate_tracklet_pf(bench_items, y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars']),
                device,
            )
            m.train()
            m.zero_grad(set_to_none=True)
            tb0 = time.perf_counter()
            for _ in range(3):
                m.zero_grad(set_to_none=True)
                lb, _ = tracklet_ipformer_loss(m(b_bench), b_bench)
                lb.backward()
            torch.cuda.synchronize()
            dt = time.perf_counter() - tb0
            vrec['bench_bs'] = len(bench_items)
            vrec['bench_fp32_peak_vram_mib'] = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
            vrec['bench_fp32_ev_per_s'] = float((3 * len(bench_items)) / max(dt, 1e-6))
            del b_bench, lb
            torch.cuda.empty_cache()

        results['variants'][vname] = vrec
        del m
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    results['all_passed'] = True
    out_p = output_path or (EXP_ROOT / 'gate_ab_results.json')
    write_json(out_p, results)
    return results


def train_stage1_variant(variant_name: str, resume: bool = False) -> dict[str, Any]:
    """Train one Stage-I variant on Internal-Train (25.6k x 20p) and export IEEE-FP32 predictions."""
    _configure_runtime()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for Stage-I training')
    device = torch.device('cuda:0')
    cap = torch.cuda.get_device_capability(0)
    specs = get_variant_specs()
    if variant_name not in specs:
        raise ValueError(f'Unknown variant {variant_name!r}; expected one of {sorted(specs)}')
    cfg = dict(specs[variant_name])
    family = cfg['family']

    exp_dir = EXP_ROOT / variant_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = exp_dir / 'manifest.json'
    if resume and manifest_path.is_file():
        man = json.loads(manifest_path.read_text())
        if man.get('status') == 'complete':
            return man

    seed = int(cfg['seed'])
    torch.manual_seed(seed)
    np.random.seed(seed)

    geometry = json.loads((CLEAN_V3_ROOT / 'geometry.json').read_text())
    bounds = detector_bounds(geometry)
    splits = _load_stage1_split_keys()

    parent_scratch = os.environ.get('PINPOINT_V3_SCRATCH', '/tmp')
    Path(parent_scratch).mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    utc_started = datetime.now(timezone.utc).isoformat()

    with tempfile.TemporaryDirectory(prefix=f'ng-{variant_name}-', dir=parent_scratch) as tmp_str:
        tmp_dir = Path(tmp_str)
        prep = _load_or_prepare_stage1(CLEAN_V3_ROOT, splits, geometry, cfg, tmp_dir, workers=4)

        train_ds = _Stage1MappedDataset(prep['train']['dir'], sidecar_df=prep['train']['sidecar'])
        dev_ds = _Stage1MappedDataset(prep['dev_screen']['dir'])
        val_ds = _Stage1MappedDataset(prep['val_eval']['dir'])

        model, branch_fn, _, collate_fn = _build_model_and_ops(cfg, geometry)
        model = model.to(device)
        ema_model = copy.deepcopy(model).eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)

        micro_bs = int(cfg['micro_bs'])
        grad_accum = int(cfg['grad_accum'])
        eff_bs = int(cfg['batch_size'])
        passes = int(cfg['passes'])
        base_lr = float(cfg['learning_rate'])
        wd = float(cfg['weight_decay'])
        grad_clip = float(cfg['grad_clip'])
        ema_decay = float(cfg['ema_decay'])

        opt_params = (
            model.build_param_groups(base_lr, wd)
            if hasattr(model, 'build_param_groups')
            else split_adamw_param_groups(model, base_lr, wd)
        )
        opt = torch.optim.AdamW(opt_params, lr=base_lr, weight_decay=wd, betas=(0.9, 0.98))

        steps_per_pass = math.ceil(len(train_ds) / eff_bs)
        total_steps = passes * steps_per_pass
        warmup_steps = max(1, int(float(cfg['warmup_fraction']) * total_steps))
        blend_steps = max(1, int(0.30 * total_steps))
        min_lr_ratio = float(cfg['min_lr']) / base_lr

        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * min(1.0, prog)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

        dev_loader = _make_loader(dev_ds, batch_size=64, shuffle=False, seed=seed, collate_fn=collate_fn, workers=2)
        val_loader = _make_loader(val_ds, batch_size=64, shuffle=False, seed=seed, collate_fn=collate_fn, workers=2)
        y_dev = splits['dev_screen']['label'].to_numpy(dtype=np.int64)

        amp_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
        amp_ok = cap[0] >= 7
        scaler = torch.amp.GradScaler('cuda', enabled=(amp_ok and amp_dtype == torch.float16))

        best_nll = float('inf')
        best_pass = 0
        global_step = 0
        history: list[dict[str, Any]] = []
        best_pt = exp_dir / 'best.pt'
        ema_pt = exp_dir / 'ema.pt'
        resume_pt = exp_dir / 'resume.pt'
        start_pass = 1

        if resume and resume_pt.is_file():
            ck = torch.load(resume_pt, map_location=device, weights_only=False)
            model.load_state_dict(ck['model'])
            ema_model.load_state_dict(ck['ema'])
            opt.load_state_dict(ck['opt'])
            scheduler.load_state_dict(ck['scheduler'])
            start_pass = int(ck['pass']) + 1
            global_step = int(ck['global_step'])
            best_nll = float(ck['best_nll'])
            best_pass = int(ck['best_pass'])
            history = list(ck['history'])

        torch.cuda.reset_peak_memory_stats(device)
        try:
            for p_idx in range(start_pass, passes + 1):
                tp0 = time.perf_counter()
                tr_loader = _make_loader(train_ds, batch_size=micro_bs, shuffle=True, seed=seed + p_idx * 101, collate_fn=collate_fn, workers=2)
                model.train()
                loss_sum = 0.0
                n_seen = 0

                def _step_window(win_buf: list[dict]) -> None:
                    nonlocal global_step, loss_sum, n_seen
                    global_step += 1
                    grad_blend = min(1.0, 0.30 + 0.70 * global_step / float(blend_steps))
                    opt.zero_grad(set_to_none=True)
                    tot_b = sum(len(mb['label']) for mb in win_buf)
                    fd = (
                        compute_window_task_denominators(win_buf, pos_weight=3.0, weight_all_tasks=False)
                        if family != 'b1pp'
                        else None
                    )
                    win_loss = 0.0
                    for mb in win_buf:
                        db = _to_device_batch(mb, device)
                        bs_n = len(db['label'])
                        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_ok):
                            loss, _ = _compute_variant_loss(model, family, db, grad_blend, fixed_denoms=fd)
                            scaled_l = loss if (fd is not None) else (loss * (float(bs_n) / float(max(tot_b, 1))))
                        if scaler.is_enabled():
                            scaler.scale(scaled_l).backward()
                        else:
                            scaled_l.backward()
                        win_loss += float(scaled_l.detach())
                    if scaler.is_enabled():
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        scaler.step(opt)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        opt.step()
                    scheduler.step()
                    with torch.no_grad():
                        for pe, pm in zip(ema_model.parameters(), model.parameters()):
                            pe.data.mul_(ema_decay).add_(pm.data, alpha=1.0 - ema_decay)
                        for be, bm in zip(ema_model.buffers(), model.buffers()):
                            be.copy_(bm)
                    loss_sum += win_loss * tot_b
                    n_seen += tot_b

                try:
                    micro_buf: list[dict] = []
                    for batch in tr_loader:
                        micro_buf.append(batch)
                        if len(micro_buf) >= grad_accum:
                            _step_window(micro_buf)
                            micro_buf.clear()
                    if micro_buf:
                        _step_window(micro_buf)
                        micro_buf.clear()
                finally:
                    shutdown_loader(tr_loader)

                tr_loss = loss_sum / max(n_seen, 1)
                row_log: dict[str, Any] = {
                    'pass': p_idx,
                    'step': global_step,
                    'lr': float(scheduler.get_last_lr()[0]),
                    'train_loss': float(tr_loss),
                    'seconds': float(time.perf_counter() - tp0),
                }
                if p_idx == 1 or (p_idx % 2 == 0) or p_idx == passes:
                    dev_pred = _eval_ieee_fp32(ema_model, family, dev_loader, device)
                    m_dev = _ba4_and_nll(y_dev, dev_pred['probs'])
                    row_log.update({'dev_nll': m_dev['NLL'], 'dev_ba4': m_dev['BA4']})
                    if m_dev['NLL'] < best_nll:
                        best_nll = m_dev['NLL']
                        best_pass = p_idx
                        loc_b = tmp_dir / 'best.local.pt'
                        torch.save({'model': ema_model.state_dict(), 'pass': p_idx, 'config': cfg}, loc_b)
                        shutil.copyfile(loc_b, best_pt)
                        loc_b.unlink(missing_ok=True)
                    print(
                        f"[{variant_name}] pass={p_idx}/{passes} step={global_step} train_loss={tr_loss:.4f} "
                        f"dev_nll={m_dev['NLL']:.4f} dev_ba4={m_dev['BA4']:.4f} ({row_log['seconds']:.1f}s)",
                        flush=True,
                    )
                history.append(row_log)
                if p_idx % 5 == 0 or p_idx == passes:
                    loc_r = tmp_dir / 'resume.local.pt'
                    torch.save({
                        'pass': p_idx, 'global_step': global_step,
                        'model': model.state_dict(), 'ema': ema_model.state_dict(),
                        'opt': opt.state_dict(), 'scheduler': scheduler.state_dict(),
                        'best_nll': best_nll, 'best_pass': best_pass, 'history': history,
                    }, loc_r)
                    shutil.copyfile(loc_r, resume_pt)
                    loc_r.unlink(missing_ok=True)

            # Save final EMA checkpoint & evaluate best.pt in IEEE-FP32
            loc_e = tmp_dir / 'ema.local.pt'
            torch.save({'model': ema_model.state_dict(), 'pass': passes, 'config': cfg}, loc_e)
            shutil.copyfile(loc_e, ema_pt)
            loc_e.unlink(missing_ok=True)

            best_state = torch.load(best_pt, map_location=device, weights_only=False)
            ema_model.load_state_dict(best_state['model'])

            dev_out = _eval_ieee_fp32(ema_model, family, dev_loader, device)
            val_out = _eval_ieee_fp32(ema_model, family, val_loader, device)

            np.save(exp_dir / 'probs_dev_screen.npy', dev_out['probs'].astype(np.float32))
            np.save(exp_dir / 'probs_internal_val_eval.npy', val_out['probs'].astype(np.float32))
            if 'pv_pred' in val_out:
                np.save(exp_dir / 'pv_pred_internal_val_eval.npy', val_out['pv_pred'].astype(np.float32))
                np.save(exp_dir / 'pv_scale_internal_val_eval.npy', val_out['pv_scale'].astype(np.float32))
            if 'sv_disp' in val_out:
                np.save(exp_dir / 'sv_disp_internal_val_eval.npy', val_out['sv_disp'].astype(np.float32))

            # Evaluate open pure-W OOD samples in IEEE-FP32
            ood_w184_probs = _eval_ood_fp32(ema_model, family, OOD_W184_NC_CHARM, geometry, bounds, cfg, collate_fn, device)
            ood_r10085_probs = _eval_ood_fp32(ema_model, family, OOD_RUN10085_10088, geometry, bounds, cfg, collate_fn, device)
            np.save(exp_dir / 'probs_ood_w184_epps21_nc_charm.npy', ood_w184_probs)
            np.save(exp_dir / 'probs_ood_run10085_10088.npy', ood_r10085_probs)

            # Save predictions parquets
            for sp_name, df_meta, out_dict, fname in (
                ('dev_screen', splits['dev_screen'], dev_out, 'dev_screen_predictions.parquet'),
                ('val_eval', splits['val_eval'], val_out, 'val_eval_predictions.parquet'),
            ):
                df_p = df_meta[KEY + FAMILY + ['label', 'obs_domain_id', 'region_id']].copy().reset_index(drop=True)
                pr = out_dict['probs']
                for c in range(4):
                    df_p[f'p_{c}'] = pr[:, c].astype(np.float32)
                if 'pv_pred' in out_dict:
                    df_p['v_hat_x_mm'] = out_dict['pv_pred'][:, 0].astype(np.float32)
                    df_p['v_hat_y_mm'] = out_dict['pv_pred'][:, 1].astype(np.float32)
                    df_p['v_hat_z_mm'] = out_dict['pv_pred'][:, 2].astype(np.float32)
                    df_p['scale_x_mm'] = out_dict['pv_scale'][:, 0].astype(np.float32)
                    df_p['scale_y_mm'] = out_dict['pv_scale'][:, 1].astype(np.float32)
                    df_p['scale_z_mm'] = out_dict['pv_scale'][:, 2].astype(np.float32)
                df_p.to_parquet(exp_dir / fname, index=False, compression=COMPRESSION)

        finally:
            shutdown_loader(dev_loader)
            shutdown_loader(val_loader)

    peak_vram_mib = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    vram_total_mib = float(torch.cuda.get_device_properties(device).total_memory / (1024 ** 2))
    manifest = {
        'status': 'complete',
        'variant': variant_name,
        'family': family,
        'UTC_started': utc_started,
        'UTC_finished': datetime.now(timezone.utc).isoformat(),
        'GPU': torch.cuda.get_device_name(device),
        'GPU_VRAM_MiB': round(vram_total_mib, 1),
        'peak_allocated_VRAM_MiB': round(peak_vram_mib, 1),
        'parameters': int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        'train_events': len(splits['train']),
        'dev_screen_events': len(splits['dev_screen']),
        'val_eval_events': len(splits['val_eval']),
        'effective_batch_size': eff_bs,
        'micro_batch_size': micro_bs,
        'grad_accum': grad_accum,
        'total_optimizer_steps': global_step,
        'best_pass': best_pass,
        'best_dev_screen_nll': best_nll,
        'eval_precision': 'ieee_fp32',
        'prep_seconds': prep['prep_seconds'],
        'wall_seconds': float(time.perf_counter() - t_start),
        'config': cfg,
        'history': history,
        'tune8k_or_sealed_touched': False,
    }
    write_json(manifest_path, manifest)
    if resume_pt.is_file():
        resume_pt.unlink()
    return manifest


def submit_condor_jobs(backend: str = 'condor', variants: list[str] | None = None) -> dict[str, Any]:
    """Submit Stage-I parallel GPU jobs to bigbird20.cern.ch (or Kubeflow backup)."""
    specs = get_variant_specs()
    targets = variants or list(specs.keys())
    repo = Path(__file__).resolve().parents[4]
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    (EXP_ROOT / 'logs').mkdir(parents=True, exist_ok=True)
    (EXP_ROOT / 'code').mkdir(parents=True, exist_ok=True)
    (EXP_ROOT / 'runtime').mkdir(parents=True, exist_ok=True)
    CONDOR_DIR.mkdir(parents=True, exist_ok=True)

    archive = EXP_ROOT / 'code' / f'nextgen_tracklet_{int(time.time())}.tar.gz'
    _package_source(repo, archive)
    h = hashlib.sha256()
    with archive.open('rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''):
            h.update(ch)
    archive_sha = h.hexdigest()

    chosen, snapshot, checks = choose_schedd(EXP_ROOT) if backend == 'condor' else ('', '', [])
    submissions: dict[str, Any] = {}
    for vname in targets:
        job_dir = CONDOR_DIR / vname
        job_dir.mkdir(parents=True, exist_ok=True)
        script = [
            '#!/bin/bash',
            'set -o pipefail',
            'which() { command -v "$@"; }; export -f which',
            'GPU_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d " ")',
            'if [ "$GPU_CAP" = "7.0" ]; then',
            '  source /cvmfs/sft.cern.ch/lcg/views/LCG_107_cuda/x86_64-el9-gcc11-opt/setup.sh',
            'else',
            '  source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh',
            'fi',
            'set -e',
            'ls /eos/home-x/xcheng/FASER/Pinpoint/results >/dev/null',
            'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1',
            f'export PINPOINT_CODE_ARCHIVE_SHA256={shlex.quote(archive_sha)}',
            'export PINPOINT_V3_SCRATCH="${_CONDOR_SCRATCH_DIR:-/tmp}/ng_tip_${PINPOINT_JOB_ID}"',
            'mkdir -p "$PINPOINT_V3_SCRATCH/code"',
            f'tar -xzf {shlex.quote(str(archive))} -C "$PINPOINT_V3_SCRATCH/code"',
            'export PYTHONPATH="$PINPOINT_V3_SCRATCH/code/src:${PYTHONPATH:-}"',
            'cd "$PINPOINT_V3_SCRATCH"',
            f'exec python3 -u -m analysis.flavour.nextgen.run_tracklet_ipformer --train --variant {shlex.quote(vname)}',
        ]
        wrapper = job_dir / 'run.sh'
        wrapper.write_text('\n'.join(script) + '\n')
        wrapper.chmod(0o755)

        if backend == 'condor':
            req = gpu_requirement(8.0, None, 10000)
            sub_lines = [
                'universe = vanilla',
                f'executable = {wrapper}',
                f'initialdir = {job_dir}',
                f'output = {EXP_ROOT}/logs/{vname}.$(ClusterId).out',
                f'error = {EXP_ROOT}/logs/{vname}.$(ClusterId).err',
                f'log = {EXP_ROOT}/logs/{vname}.$(ClusterId).log',
                'environment = "PINPOINT_JOB_ID=$(ClusterId).$(Process)"',
                'request_cpus = 4',
                'request_memory = 15000M',
                'request_disk = 20GB',
                'requirements = (OpSysAndVer == "AlmaLinux9")',
                '+JobFlavour = "workday"',
                'should_transfer_files = YES',
                'when_to_transfer_output = ON_EXIT',
                'transfer_output_files = ""',
                'request_gpus = 1',
                f'require_gpus = {req}',
                'queue 1',
            ]
            sub_path = job_dir / 'job.sub'
            sub_path.write_text('\n'.join(sub_lines) + '\n')
            resp = subprocess.run(['condor_submit', '-name', chosen, str(sub_path)], capture_output=True, text=True)
            rec = {
                'UTC': datetime.now(timezone.utc).isoformat(),
                'variant': vname,
                'schedd': chosen,
                'stdout': resp.stdout,
                'stderr': resp.stderr,
                'returncode': resp.returncode,
                'archive': str(archive),
                'archive_sha256': archive_sha,
            }
            write_json(EXP_ROOT / 'runtime' / f'{vname}.submission.json', rec)
            if resp.returncode != 0:
                raise RuntimeError(f'condor_submit failed for {vname}: {resp.stderr}')
            submissions[vname] = rec
        else:
            raise ValueError(f'Unsupported backend {backend!r} in submit_condor_jobs')
    return submissions


def aggregate_stage1_results() -> dict[str, Any]:
    """Compute Stage-I metrics, OOD metrics, PV pulls, and paired family-bootstrap CIs."""
    splits = _load_stage1_split_keys()
    val_df = splits['val_eval']
    labels = val_df['label'].to_numpy(dtype=np.int64)
    obs_dom = val_df['obs_domain_id'].to_numpy(dtype=np.int64)
    reg_id = val_df['region_id'].to_numpy(dtype=np.int64)
    fam_codes = pd.MultiIndex.from_frame(val_df[FAMILY]).factorize()[0]

    # Attach event_sidecar PV truth strictly for internal_val_eval (000..079 only)
    sc_all = pd.read_parquet(CLEAN_V3_ROOT / 'truth_sidecar/event_sidecar.parquet')
    pos = pd.MultiIndex.from_frame(sc_all[KEY]).get_indexer(pd.MultiIndex.from_frame(val_df[KEY]))
    sc_val = sc_all.iloc[pos].reset_index(drop=True)
    v_true = sc_val[['vertex_x_mm', 'vertex_y_mm', 'vertex_z_mm']].to_numpy(dtype=np.float64)
    v_valid = sc_val['vertex_valid'].to_numpy(dtype=np.int64) > 0

    ood_w184_meta = pd.read_parquet(OOD_W184_NC_CHARM / 'truth_meta.parquet')
    ood_r10085_meta = pd.read_parquet(OOD_RUN10085_10088 / 'truth_meta.parquet')
    ood_r10085_y = ood_r10085_meta['label'].to_numpy(dtype=np.int64)

    slices = {
        'All': np.ones(len(val_df), dtype=bool),
        'D1_pix_contained': obs_dom == 1,
        'D2_scint_contained': obs_dom == 2,
        'D3_partial_or_edge': obs_dom == 3,
        'Pinpoint_PixXY': (reg_id == 0) | (reg_id == 1),
        'Front30': reg_id == 0,
        'Interm150': reg_id == 1,
    }

    specs = get_variant_specs()
    runs: dict[str, Any] = {}
    probs_map: dict[str, np.ndarray] = {}

    for vname in specs:
        vdir = EXP_ROOT / vname
        man_p = vdir / 'manifest.json'
        if not man_p.is_file():
            continue
        man = json.loads(man_p.read_text())
        p_val = np.load(vdir / 'probs_internal_val_eval.npy').astype(np.float64)
        probs_map[vname] = p_val

        slice_metrics = {sname: _ba4_and_nll(labels[mk], p_val[mk]) for sname, mk in slices.items() if mk.any()}

        # Dual 1% FPR nu_tau efficiency metrics
        tau_scores = np.log(np.clip(p_val[:, 2], 1e-8, 1.0)) - np.log(np.clip(1.0 - p_val[:, 2], 1e-8, 1.0))
        within_1pct = {}
        for sname in ('All', 'D1_pix_contained', 'Pinpoint_PixXY', 'Front30'):
            mk = slices[sname]
            wp = _tau_efficiency_at_fpr(tau_scores[mk], labels[mk], 0.01)
            within_1pct[sname] = {
                'tau_eff': float(wp['tau_eff']),
                'actual_fpr': float(wp['actual_fpr']),
                'n_tau': int((labels[mk] == 2).sum()),
                'n_bg': int((labels[mk] != 2).sum()),
            }
        split_half = {
            sname: _split_half_tail_eval(tau_scores, labels, val_df, slices[sname], fpr_targets=(0.01,))
            for sname in ('All', 'D1_pix_contained', 'Pinpoint_PixXY')
        }

        # PV metrics if model outputs PV
        pv_audit: dict[str, Any] | None = None
        pv_path = vdir / 'pv_pred_internal_val_eval.npy'
        sc_path = vdir / 'pv_scale_internal_val_eval.npy'
        if pv_path.is_file():
            v_pred = np.load(pv_path).astype(np.float64)
            pv_audit = {}
            for rtag, rmk in (
                ('All_active', v_valid),
                ('Pinpoint_PixXY', v_valid & slices['Pinpoint_PixXY']),
                ('Front30', v_valid & slices['Front30']),
                ('Interm150', v_valid & slices['Interm150']),
            ):
                if rmk.any():
                    dz = np.abs(v_pred[rmk, 2] - v_true[rmk, 2])
                    d3 = np.linalg.norm(v_pred[rmk] - v_true[rmk], axis=1)
                    pv_audit[rtag] = {
                        'count': int(rmk.sum()),
                        'median_abs_dz_mm': float(np.median(dz)),
                        'median_3d_mm': float(np.median(d3)),
                    }
            if sc_path.is_file():
                v_sc = np.load(sc_path).astype(np.float64)
                pulls = compute_student_t_pulls(
                    torch.from_numpy(v_pred),
                    torch.from_numpy(v_sc),
                    torch.from_numpy(v_true),
                    torch.from_numpy(v_valid.astype(np.int64)),
                    df=3.0,
                )
                pv_audit['student_t_pulls_all'] = pulls
                pulls_pp = compute_student_t_pulls(
                    torch.from_numpy(v_pred[slices['Pinpoint_PixXY']]),
                    torch.from_numpy(v_sc[slices['Pinpoint_PixXY']]),
                    torch.from_numpy(v_true[slices['Pinpoint_PixXY']]),
                    torch.from_numpy(v_valid[slices['Pinpoint_PixXY']].astype(np.int64)),
                    df=3.0,
                )
                pv_audit['student_t_pulls_pinpoint'] = pulls_pp
                for pname, pslice in (
                    ('student_t_pulls_front30', slices['Front30']),
                    ('student_t_pulls_interm150', slices['Interm150']),
                    ('student_t_pulls_d1', slices['D1_pix_contained']),
                    ('student_t_pulls_d2', slices['D2_scint_contained']),
                ):
                    pv_audit[pname] = compute_student_t_pulls(
                        torch.from_numpy(v_pred[pslice]),
                        torch.from_numpy(v_sc[pslice]),
                        torch.from_numpy(v_true[pslice]),
                        torch.from_numpy(v_valid[pslice].astype(np.int64)),
                        df=3.0,
                    )

        # OOD metrics
        ood_metrics: dict[str, Any] = {}
        w184_p = vdir / 'probs_ood_w184_epps21_nc_charm.npy'
        if w184_p.is_file():
            pw = np.load(w184_p).astype(np.float64)
            ood_metrics['gen_ood_w184_epps21_nc_charm'] = {
                'n_events': int(len(pw)),
                'pred_tau_argmax_rate': float((pw.argmax(axis=1) == 2).mean()),
                'p_tau_gt_0p5_rate': float((pw[:, 2] > 0.5).mean()),
                'mean_p_tau': float(pw[:, 2].mean()),
            }
        r85_p = vdir / 'probs_ood_run10085_10088.npy'
        if r85_p.is_file():
            pr85 = np.load(r85_p).astype(np.float64)
            ood_metrics['gen_ood_run10085_10088'] = {
                'n_events': int(len(pr85)),
                **_ba4_and_nll(ood_r10085_y, pr85),
            }

        runs[vname] = {
            'parameters': man['parameters'],
            'best_pass': man['best_pass'],
            'best_dev_screen_nll': man['best_dev_screen_nll'],
            'peak_allocated_VRAM_MiB': man['peak_allocated_VRAM_MiB'],
            'wall_seconds': man['wall_seconds'],
            'GPU': man['GPU'],
            'slices': slice_metrics,
            'within_slice_1pct_fpr': within_1pct,
            'split_half_1pct_fpr': split_half,
            'pv_audit': pv_audit,
            'ood': ood_metrics,
        }

    # Paired family-bootstrap comparisons (seed=20263250, 1000 reps)
    paired_bootstrap: dict[str, Any] = {}
    if 's1_tracklet_ipformer' in probs_map:
        p_full = probs_map['s1_tracklet_ipformer']
        s_full = p_full[:, 2]
        for other in ('s1_b1pp', 's1_pf_20um', 's1_hybrid_nomad', 'abl_no_skew_dca', 'abl_hard_avr_pv', 'abl_legacy_16d_only', 'abl_1mm_quant'):
            if other not in probs_map:
                continue
            p_other = probs_map[other]
            s_other = p_other[:, 2]
            paired_bootstrap[f's1_tracklet_ipformer - {other}'] = {
                sname: paired_family_bootstrap_delta(
                    s_full, s_other, p_full, p_other, labels, fam_codes, slices[sname], n_boot=1000, seed=20263250
                )
                for sname in ('All', 'D1_pix_contained', 'D2_scint_contained', 'Pinpoint_PixXY', 'Front30')
            }

    summary = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'stage': 'Stage-I Internal-Val (000..079 only; Tune8k & sealed 100..149 untouched)',
        'val_eval_events': int(len(val_df)),
        'runs': runs,
        'paired_family_bootstrap_seed20263250_n1000': paired_bootstrap,
    }
    write_json(EXP_ROOT / 'stage1_tracklet_ipformer_results.json', summary)
    return summary


def main() -> None:
    guard = install_stage1_seal_guard()
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--gate-ab', action='store_true', help='Run local Gate A & Gate B checks')
        parser.add_argument('--prepare-cache', action='store_true', help='Pre-extract shared Stage-I item cache')
        parser.add_argument('--prepare-cache-part', action='store_true', help='Extract one partition of Stage-I cache')
        parser.add_argument('--merge-cache', action='store_true', help='Merge sharded cache parts into MappedItems')
        parser.add_argument('--submit-cache', action='store_true', help='Submit HTCondor CPU cache extraction/merge')
        parser.add_argument('--cache-phase', type=str, default='parts', choices=('parts', 'merge'))
        parser.add_argument('--cache-tag', type=str, default='tracklet_pf_20um')
        parser.add_argument('--part', type=int, default=0)
        parser.add_argument('--n-parts', type=int, default=8)
        parser.add_argument('--workers', type=int, default=8)
        parser.add_argument('--train', action='store_true', help='Train one Stage-I variant')
        parser.add_argument('--variant', type=str, help='Variant name for --train')
        parser.add_argument('--resume', action='store_true')
        parser.add_argument('--submit', action='store_true', help='Submit Stage-I jobs to HTCondor')
        parser.add_argument('--variants', nargs='*', default=None, help='Optional subset of variants to submit')
        parser.add_argument('--aggregate', action='store_true', help='Aggregate completed runs & paired bootstrap')
        args = parser.parse_args()

        if args.gate_ab:
            res = run_gate_ab_checks()
            print(json.dumps(res, indent=2))
            return
        if args.prepare_cache:
            res = prepare_stage1_caches(workers=args.workers)
            print(json.dumps(res, indent=2))
            return
        if args.prepare_cache_part:
            res = prepare_stage1_cache_part(args.cache_tag, args.part, args.n_parts, workers=args.workers)
            print(json.dumps(res, indent=2))
            return
        if args.merge_cache:
            res = merge_stage1_cache_tag(args.cache_tag, args.n_parts)
            print(json.dumps(res, indent=2))
            return
        if args.submit_cache:
            res = submit_cache_jobs(phase=args.cache_phase)
            print(json.dumps(res, indent=2))
            return
        if args.train:
            if not args.variant:
                parser.error('--variant is required with --train')
            res = train_stage1_variant(args.variant, resume=args.resume)
            print(json.dumps({
                'status': res['status'],
                'variant': res['variant'],
                'best_pass': res['best_pass'],
                'best_dev_screen_nll': res['best_dev_screen_nll'],
                'peak_allocated_VRAM_MiB': res['peak_allocated_VRAM_MiB'],
                'wall_seconds': res['wall_seconds'],
            }, indent=2))
            return
        if args.submit:
            res = submit_condor_jobs(variants=args.variants)
            print(json.dumps({k: {'schedd': v['schedd'], 'stdout': v['stdout'].strip()} for k, v in res.items()}, indent=2))
            return
        if args.aggregate:
            res = aggregate_stage1_results()
            print(json.dumps({'runs': list(res['runs'].keys()), 'out': str(EXP_ROOT / 'stage1_tracklet_ipformer_results.json')}, indent=2))
            return
        parser.error('Specify one of --gate-ab, --train, --submit, or --aggregate')
    finally:
        guard.__exit__(None, None, None)


if __name__ == '__main__':
    main()
