"""Interactive 32-event GPU health gate and shared production trainer for PinpointFormer."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import tempfile
import time
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.s4.stage_inputs import stage_shards
from analysis.flavour.v2.partitions import FAMILY, KEY
from analysis.flavour.v2.profile import init_worker, shutdown_loader
from analysis.flavour.v2.r1.metrics import Metrics, three_from_four, three_targets
from analysis.flavour.v2.r1.tasks import probabilities, task_loss
from analysis.flavour.v2.r3.mapped_inputs import MappedItems, write_mapped_items
from analysis.flavour.v3.common.datasets import V3EvalDataset, V3TrainDataset
from analysis.flavour.v3.common.domains import detector_bounds
from analysis.flavour.v3.v2_pinpointformer.models import PinpointFormer, pf_branch_groups
from analysis.flavour.v3.v2_pinpointformer.pixel_tokenizer import (
    collate_pf,
    extract_pf_item,
)
from analysis.flavour.v3.v2_pinpointformer.vertex_anchor import heteroscedastic_vertex_loss


DEFAULT_GEOMETRY = Path(
    '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s0_s1_2026-11-11/shards/10083/geometry.json'
)
DEFAULT_DATA_ROOT = Path(
    '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R2/data'
)
DEFAULT_SIDECAR_ROOT = Path(
    '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/truth_sidecar'
)
DEFAULT_HEALTH_OUT = Path(
    '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-06/w3_06_health_report.json'
)


def _identity_collate(xs: list[dict]) -> list[dict]:
    return xs


class CachedPFDataset(Dataset):
    """Memory-mapped PinpointFormer cache with optional training-only sidecar attachment."""

    def __init__(
        self,
        items_dir: Path | str,
        metadata: pd.DataFrame,
        indices: np.ndarray | None = None,
        sample_weights: np.ndarray | None = None,
        event_sidecar: pd.DataFrame | None = None,
        cluster_sidecar_dir: Path | str | None = None,
    ) -> None:
        self.items = MappedItems(items_dir)
        self.indices = None if indices is None else np.asarray(indices, dtype=np.int64)
        if self.indices is None:
            self.metadata = metadata.reset_index(drop=True)
        else:
            self.metadata = metadata.iloc[self.indices].reset_index(drop=True)
        self.sample_weights = sample_weights
        self.event_sidecar = event_sidecar
        self.cluster_dir = Path(cluster_sidecar_dir) if cluster_sidecar_dir is not None else None
        self._memmaps: dict | None = None

    def _get_memmaps(self) -> dict | None:
        if self.cluster_dir is None:
            return None
        if self._memmaps is None:
            pix_off = np.memmap(self.cluster_dir / 'pix_offsets.bin', dtype='<i8', mode='r')
            bar_off = np.memmap(self.cluster_dir / 'bar_offsets.bin', dtype='<i8', mode='r')
            n_pix = int(pix_off[-1])
            n_bar = int(bar_off[-1])
            self._memmaps = {
                'pix_off': pix_off,
                'bar_off': bar_off,
                'first_cid': np.memmap(
                    self.cluster_dir / 'cluster_first_channel_id.bin', dtype='<u8', mode='r', shape=(n_pix,)
                ),
                'origin': np.memmap(
                    self.cluster_dir / 'cluster_origin.bin', dtype='|i1', mode='r', shape=(n_pix,)
                ),
                'ancestry': np.memmap(
                    self.cluster_dir / 'cluster_ancestry_id.bin', dtype='<i4', mode='r', shape=(n_pix, 2)
                ),
                'pix_rel': np.memmap(
                    self.cluster_dir / 'pixel_rel_target.bin', dtype='<f2', mode='r', shape=(n_pix, 3)
                ),
                'bar_rel': np.memmap(
                    self.cluster_dir / 'bar_rel_target.bin', dtype='<f2', mode='r', shape=(n_bar, 3)
                ),
            }
        return self._memmaps

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, pos: int) -> dict:
        raw_idx = pos if self.indices is None else int(self.indices[pos])
        item = dict(self.items[raw_idx])
        item['index'] = int(pos)
        if self.sample_weights is not None:
            item['sample_weight'] = float(self.sample_weights[pos])
        if self.event_sidecar is not None:
            row = self.event_sidecar.iloc[pos]
            item['label_7'] = int(row['label_7'])
            item['tau_branch_valid'] = int(row['tau_branch_valid'])
            item['charm_flag'] = int(row['charm_flag'])
            item['charm_valid'] = int(row['charm_valid'])
            item['vertex_xyz_mm'] = np.array(
                [row['vertex_x_mm'], row['vertex_y_mm'], row['vertex_z_mm']], dtype=np.float32
            )
            item['vertex_valid'] = int(row['vertex_valid'])
            item['vis_kinematics'] = np.array(
                [row['vis_pz_log1p'], row['vis_pt_log1p'], row['vis_cos_dphi'], row['vis_sin_dphi']],
                dtype=np.float32,
            )
            item['kin_valid'] = int(row['kin_valid'])
            mm = self._get_memmaps()
            if mm is not None:
                si = int(row['sidecar_index'])
                ps, pe = int(mm['pix_off'][si]), int(mm['pix_off'][si + 1])
                bs, be = int(mm['bar_off'][si]), int(mm['bar_off'][si + 1])
                item['sc_first_cid'] = np.array(mm['first_cid'][ps:pe], copy=True)
                item['sc_origin'] = np.array(mm['origin'][ps:pe], copy=True)
                item['sc_ancestry'] = np.array(mm['ancestry'][ps:pe], copy=True)
                item['sc_pix_rel'] = np.array(mm['pix_rel'][ps:pe], copy=True)
                item['sc_bar_rel'] = np.array(mm['bar_rel'][bs:be], copy=True)
        return item


class _PFCollator:
    def __init__(self, mirror_seed: int | None, y_scint_center: float, n_bars: int) -> None:
        self.rng = np.random.default_rng(mirror_seed) if mirror_seed is not None else None
        self.y_c = float(y_scint_center)
        self.n_bars = int(n_bars)

    def __call__(self, events: list[dict]) -> dict:
        return collate_pf(events, mirror_rng=self.rng, y_scint_center=self.y_c, n_bars=self.n_bars)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def make_pf_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int = 42,
    y_scint_center: float = -12.0,
    n_bars: int = 40,
    mirror: bool = False,
) -> DataLoader:
    if shuffle:
        gen = torch.Generator()
        gen.manual_seed(seed)
        sampler = RandomSampler(dataset, generator=gen)
    else:
        sampler = SequentialSampler(dataset)
    kw: dict = {
        'batch_sampler': BatchSampler(sampler, batch_size=batch_size, drop_last=False),
        'num_workers': workers,
        'collate_fn': _PFCollator(seed + 999 if mirror else None, y_scint_center, n_bars),
        'pin_memory': torch.cuda.is_available(),
    }
    if workers > 0:
        kw.update(multiprocessing_context='spawn', worker_init_fn=init_worker, prefetch_factor=2)
    return DataLoader(dataset, **kw)


def build_pf_mapped_split(
    data_root: Path,
    split: str,
    out_dir: Path,
    pkl_path: Path,
    scratch: Path,
    workers: int,
    config: dict,
) -> dict:
    """Materialize pure-observation PinpointFormer MappedItems for one split."""
    t0 = time.perf_counter()
    ds_cfg = {
        'cache_shards': 2,
        'quantize_xy_mm': config.get('quantize_xy_mm'),
    }
    ds = V3EvalDataset(
        data_root,
        split=split,
        config=ds_cfg,
        cache_dir=scratch,
        item_transform=extract_pf_item,
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
    return {'split': split, 'events': int(len(meta)), 'seconds': float(time.perf_counter() - t0)}


def _update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for p_e, p_m in zip(ema.parameters(), model.parameters()):
            p_e.data.mul_(decay).add_(p_m.data, alpha=1.0 - decay)
        for b_e, b_m in zip(ema.buffers(), model.buffers()):
            b_e.copy_(b_m)


def _branch_grad_norms(groups: dict[str, list[tuple[str, nn.Parameter]]]) -> dict[str, float]:
    out = {}
    for branch, plist in groups.items():
        sq = sum(float(p.grad.detach().float().pow(2).sum().item()) for _, p in plist if p.grad is not None)
        out[branch] = math.sqrt(sq)
    return out


def _branch_update_rms(
    groups: dict[str, list[tuple[str, nn.Parameter]]],
    initial: dict[str, torch.Tensor],
) -> dict[str, float]:
    out = {}
    for branch, plist in groups.items():
        sq = 0.0
        n = 0
        for name, p in plist:
            diff = p.detach().cpu().float() - initial[name]
            sq += float(diff.pow(2).sum().item())
            n += p.numel()
        out[branch] = math.sqrt(sq / max(n, 1))
    return out


def run_smoke_32_gate(
    model_builder: Callable[[dict, dict], nn.Module],
    branch_groups_fn: Callable[[nn.Module], dict[str, list[tuple[str, nn.Parameter]]]],
    loss_fn: Callable[[nn.Module, dict, dict], tuple[torch.Tensor, torch.Tensor]],
    smoke_configs: list[tuple[str, dict]],
    output_path: Path,
    sidecar_root: Path | None = None,
    load_cluster_sidecar: bool = False,
) -> dict:
    """Shared interactive 32-event GPU overfit, gradient health & bs=64 VRAM gate."""
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA device required for 32-event GPU health gate')
    device = torch.device('cuda:0')
    geometry = json.loads(DEFAULT_GEOMETRY.read_text())

    train_cfg = {
        'cache_shards': 1,
        'load_cluster_sidecar': bool(load_cluster_sidecar),
    }
    ds = V3TrainDataset(
        DEFAULT_DATA_ROOT,
        sidecar_root=sidecar_root,
        split='train',
        config=train_cfg,
        item_transform=extract_pf_item,
    )
    meta = ds.metadata
    head_n = min(3000, len(meta))
    lbl_head = meta['label'].to_numpy()[:head_n]
    items: list[dict] = []
    for c in range(4):
        taken = 0
        for idx in np.flatnonzero(lbl_head == c):
            it = ds[int(idx)]
            # Skip bit-identical D_empty events (all-zero tensors across distinct labels)
            if it['obs_domain_id'] != 0 and (len(it['pix_xyz_mm']) > 0 or len(it['bar_panel']) > 0):
                items.append(it)
                taken += 1
                if taken >= 8:
                    break
        if taken < 8:
            for idx in np.flatnonzero(meta['label'].to_numpy() == c):
                it = ds[int(idx)]
                if it['obs_domain_id'] != 0 and (len(it['pix_xyz_mm']) > 0 or len(it['bar_panel']) > 0):
                    items.append(it)
                    taken += 1
                    if taken >= 8:
                        break
    ds.native.close()

    bounds = detector_bounds(geometry)
    base_batch = collate_pf(items, mirror_rng=None, y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars'])
    bench_items = items + items

    report: dict = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'GPU': torch.cuda.get_device_name(0),
        'capability': f"SM{''.join(map(str, torch.cuda.get_device_capability(0)))}",
        'events': len(items),
        'variants': {},
    }

    for tag, cfg in smoke_configs:
        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats(device)
        model = model_builder(geometry, cfg).to(device)
        groups = branch_groups_fn(model)
        init_snap = {n: p.detach().cpu().float().clone() for n, p in model.named_parameters() if p.requires_grad}
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get('smoke_lr', 1.2e-3)), weight_decay=0.0)
        dev_batch = move_batch_to_device(base_batch, device)

        history = []
        step1_grads = {}
        passed_at = None
        t0 = time.perf_counter()
        for upd in range(1, 101):
            model.train()
            opt.zero_grad(set_to_none=True)
            loss, probs4 = loss_fn(model, dev_batch, {'grad_blend': min(1.0, 0.30 + 0.70 * upd / 25.0)})
            loss.backward()
            if upd == 1:
                step1_grads = _branch_grad_norms(groups)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

            model.eval()
            with torch.no_grad():
                _, eval_p4 = loss_fn(model, dev_batch, {'grad_blend': 1.0})
                y = dev_batch['label']
                nll = float(-torch.log(torch.clamp(eval_p4[torch.arange(len(y), device=device), y], min=1e-7)).mean())
                acc = float((eval_p4.argmax(dim=-1) == y).float().mean())
                distinct = len(torch.unique(eval_p4[:, 2].round(decimals=5)))
            if upd == 1 or upd % 10 == 0 or (nll < 0.05 and acc >= 0.999):
                history.append({'update': upd, 'NLL': nll, 'accuracy': acc, 'distinct_probs': distinct})
            if nll < 0.05 and acc >= 0.999:
                passed_at = upd
                break

        upd_rms = _branch_update_rms(groups, init_snap)
        all_grad_ok = all(math.isfinite(v) and v > 0.0 for v in step1_grads.values())
        all_upd_ok = all(math.isfinite(v) and v > 0.0 for v in upd_rms.values())

        # Free overfit batch before bs=64 VRAM/throughput benchmark (shared GPU safe)
        del dev_batch
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        torch.cuda.reset_peak_memory_stats(device)
        bench_list = bench_items if free_bytes >= 4800 * 1024 * 1024 else items
        b_bench = move_batch_to_device(
            collate_pf(bench_list, mirror_rng=None, y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars']),
            device,
        )
        model.train()
        tb0 = time.perf_counter()
        n_iters = 5
        for _ in range(n_iters):
            opt.zero_grad(set_to_none=True)
            lb, _ = loss_fn(model, b_bench, {'grad_blend': 1.0})
            lb.backward()
            opt.step()
        torch.cuda.synchronize()
        tb_elapsed = time.perf_counter() - tb0
        peak_mib = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        ev_per_s = float((n_iters * len(bench_list)) / max(tb_elapsed, 1e-6))
        del b_bench, opt, model
        torch.cuda.empty_cache()

        variant_ok = (
            passed_at is not None
            and all_grad_ok
            and all_upd_ok
            and peak_mib < 16.0 * 1024.0
        )
        n_params = int(sum(t.numel() for t in init_snap.values()))
        report['variants'][tag] = {
            'parameters': n_params,
            'passed_at_update': passed_at,
            'final_NLL': history[-1]['NLL'],
            'final_accuracy': history[-1]['accuracy'],
            'distinct_probs': history[-1]['distinct_probs'],
            'step1_branch_grad_norms': step1_grads,
            'branch_param_update_rms': upd_rms,
            'all_branches_grad_positive_finite': all_grad_ok,
            'all_branches_updated': all_upd_ok,
            'bench_batch_size': len(bench_list),
            'shared_gpu_free_MiB_before_bench': float(free_bytes / (1024 ** 2)),
            'bs64_peak_GPU_MiB': peak_mib,
            'bs64_throughput_ev_per_s': ev_per_s,
            'seconds': float(time.perf_counter() - t0),
            'passed': variant_ok,
            'history': history,
        }

    report['all_passed'] = all(v['passed'] for v in report['variants'].values())
    write_json(output_path, report)
    if not report['all_passed']:
        raise RuntimeError(f'Smoke-32 gate failed: {json.dumps(report, indent=2)}')
    return report


def _pf_v2_loss(model: nn.Module, batch: dict, ctx: dict) -> tuple[torch.Tensor, torch.Tensor]:
    grad_blend = float(ctx.get('grad_blend', 1.0))
    aux = model(batch, grad_blend=grad_blend, return_aux=True)
    logits = aux['logits']
    task = ctx.get('task', model.config.get('task', 'T4'))
    labels = batch['label']
    if task == 'T3':
        keep = labels != 2
        logits_use = logits[keep]
        labels_use = labels[keep]
    else:
        logits_use = logits
        labels_use = labels
    if 'sample_weight' in batch:
        sw = batch['sample_weight'][keep] if task == 'T3' else batch['sample_weight']
        # Compute weighted cross entropy matching task_loss semantics
        if task == 'T4':
            per = F.cross_entropy(logits_use, labels_use, reduction='none')
            l_cls = (per * sw).sum() / sw.sum().clamp_min(1e-8)
        else:
            l_cls = task_loss(logits_use, labels_use, task)
    else:
        l_cls = task_loss(logits_use, labels_use, task)

    w_vtx = float(model.config.get('vertex_loss_weight', 0.15 if model.use_vertex_anchor else 0.0))
    if model.use_vertex_anchor and w_vtx > 0.0 and 'vertex_xyz_mm' in batch:
        l_vtx = heteroscedastic_vertex_loss(
            aux['v_hat'], aux['log_var_v'], batch['vertex_xyz_mm'], batch['vertex_valid']
        )
        loss = l_cls + w_vtx * l_vtx
    else:
        loss = l_cls

    probs4 = probabilities(logits, task)
    if probs4.shape[1] == 3:
        z = probs4.new_zeros((len(probs4), 1))
        probs4 = torch.cat([probs4[:, :2], z, probs4[:, 2:]], dim=-1)
    return loss, probs4


@torch.no_grad()
def predict_pf_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_bf16: bool,
    pred_fn: Callable[[nn.Module, dict], dict],
) -> dict:
    """Run batched inference over Tune40k returning probs4, logits, v_hat, and optional extra cols."""
    model.eval()
    amp_ok = use_bf16 and device.type == 'cuda'
    chunks: dict[str, list[np.ndarray]] = {}
    for batch in loader:
        db = move_batch_to_device(batch, device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=amp_ok):
            out = pred_fn(model, db)
        for k, v in out.items():
            chunks.setdefault(k, []).append(v.detach().float().cpu().numpy())
    return {k: np.concatenate(vs, axis=0) for k, vs in chunks.items()}


def _pf_v2_predict(model: nn.Module, batch: dict) -> dict[str, torch.Tensor]:
    aux = model(batch, grad_blend=1.0, return_aux=True)
    logits = aux['logits'].float()
    probs4 = probabilities(logits, model.config.get('task', 'T4'))
    if probs4.shape[1] == 3:
        probs4 = torch.cat([probs4[:, :2], probs4.new_zeros((len(probs4), 1)), probs4[:, 2:]], dim=-1)
    return {'logits': logits, 'probs': probs4, 'v_hat': aux['v_hat'].float()}


def _save_pred_parquet(path: Path, meta: pd.DataFrame, pred: dict) -> None:
    df = meta[KEY + FAMILY + ['label']].copy().reset_index(drop=True)
    probs = pred['probs']
    logits = pred['logits']
    v_hat = pred['v_hat']
    for c in range(4):
        df[f'p_{c}'] = probs[:, c].astype(np.float32)
    for j in range(logits.shape[1]):
        df[f'logit_{j}'] = logits[:, j].astype(np.float32)
    df['v_hat_x_mm'] = v_hat[:, 0].astype(np.float32)
    df['v_hat_y_mm'] = v_hat[:, 1].astype(np.float32)
    df['v_hat_z_mm'] = v_hat[:, 2].astype(np.float32)
    if 'charm_prob' in pred:
        df['charm_prob'] = pred['charm_prob'].astype(np.float32)
    df.to_parquet(path, index=False, compression=COMPRESSION)


def _eval_screen(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_bf16: bool,
    pred_fn: Callable[[nn.Module, dict], dict],
) -> dict:
    pred = predict_pf_loader(model, loader, device, use_bf16, pred_fn)
    y = loader.dataset.metadata['label'].to_numpy(dtype=np.int64)
    probs = pred['probs'].astype(np.float64)
    probs = probs / np.maximum(probs.sum(axis=-1, keepdims=True), 1e-12)
    nll = float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-8, 1.0)).mean())
    rep = Metrics(y, probs).report()
    nt = y != 2
    ba3 = (
        float(Metrics(three_targets(y[nt]), three_from_four(probs[nt])).report()['BA'])
        if nt.any() and len(np.unique(y[nt])) == 3
        else float('nan')
    )
    return {'nll': nll, 'ba4': float(rep['BA']), 'ba3': ba3}


def run_pf_training(
    root: Path,
    config: dict,
    model_builder: Callable[[dict, dict], nn.Module],
    branch_groups_fn: Callable[[nn.Module], dict[str, list[tuple[str, nn.Parameter]]]],
    loss_fn: Callable[[nn.Module, dict, dict], tuple[torch.Tensor, torch.Tensor]],
    pred_fn: Callable[[nn.Module, dict], dict],
    resume: bool = True,
    attach_cluster_sidecar: bool = False,
) -> dict:
    """Shared checkpointed, EMA-tracked, mixed-precision PinpointFormer trainer."""
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for PinpointFormer training')
    device = torch.device('cuda:0')
    cap = torch.cuda.get_device_capability(0)
    use_bf16 = bool(config.get('bf16', True)) and cap[0] >= 8
    seed = int(config.get('seed', 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    name = str(config['name'])
    exp_dir = root / 'experiments' / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = exp_dir / 'manifest.json'
    if resume and manifest_path.exists() and json.loads(manifest_path.read_text()).get('status') == 'complete':
        return json.loads(manifest_path.read_text())

    data_root = Path(config['data_root'])
    geometry = json.loads(Path(config.get('geometry_json', DEFAULT_GEOMETRY)).read_text())
    bounds = detector_bounds(geometry)
    sidecar_root_str = config.get('sidecar_root')
    sidecar_root = Path(sidecar_root_str) if sidecar_root_str else None
    local_cluster_sc = os.environ.get('PINPOINT_V3_LOCAL_CLUSTER_SIDECAR')
    cluster_sc_dir: Path | None = None
    if attach_cluster_sidecar:
        if local_cluster_sc and Path(local_cluster_sc).exists():
            cluster_sc_dir = Path(local_cluster_sc)
        elif sidecar_root is not None and (sidecar_root / 'cluster_sidecar').exists():
            cluster_sc_dir = sidecar_root / 'cluster_sidecar'

    parent_scratch = os.environ.get('PINPOINT_V3_SCRATCH', '/tmp')
    Path(parent_scratch).mkdir(parents=True, exist_ok=True)
    cache_workers = int(config.get('cache_workers', 4))
    loader_workers = int(config.get('loader_workers', 3))
    passes = int(config['passes'])
    batch_size = int(config['batch_size'])

    t_start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f'pf-{name}-', dir=parent_scratch) as tmp_dir:
        tmp_path = Path(tmp_dir)
        shared_tr_pkl = os.environ.get('PINPOINT_V3_SHARED_TRAIN_PF')
        shared_tu_pkl = os.environ.get('PINPOINT_V3_SHARED_TUNE_PF')
        if shared_tr_pkl and shared_tu_pkl and Path(shared_tr_pkl).exists() and Path(shared_tu_pkl).exists():
            with open(shared_tr_pkl, 'rb') as f:
                tr_blob = pickle.load(f)
            with open(shared_tu_pkl, 'rb') as f:
                tu_blob = pickle.load(f)
        else:
            stage_root = os.environ.get('PINPOINT_V3_SHARED_STAGE')
            old_stage = os.environ.get('PINPOINT_V3_SHARED_STAGE')
            if not stage_root:
                shard_paths = sorted(p for p in (data_root / 'shards').iterdir() if (p / 'observations.parquet').exists())
                st_rep = stage_shards(shard_paths, tmp_path / 'native', workers=min(4, cache_workers))
                os.environ['PINPOINT_V3_SHARED_STAGE'] = st_rep['local_shard_root']
            try:
                tr_pkl = tmp_path / 'train_pf.pkl'
                tu_pkl = tmp_path / 'tune_pf.pkl'
                build_pf_mapped_split(data_root, 'train', tmp_path / 'train_pf_items', tr_pkl, tmp_path, cache_workers, config)
                build_pf_mapped_split(data_root, 'validation', tmp_path / 'tune_pf_items', tu_pkl, tmp_path, cache_workers, config)
                with open(tr_pkl, 'rb') as f:
                    tr_blob = pickle.load(f)
                with open(tu_pkl, 'rb') as f:
                    tu_blob = pickle.load(f)
            finally:
                shutil.rmtree(tmp_path / 'native', ignore_errors=True)
                if old_stage is None:
                    os.environ.pop('PINPOINT_V3_SHARED_STAGE', None)
                else:
                    os.environ['PINPOINT_V3_SHARED_STAGE'] = old_stage

        tr_meta: pd.DataFrame = tr_blob['metadata']
        tu_meta: pd.DataFrame = tu_blob['metadata']

        # Optional subset (e.g. train_100000) & sample weights
        tr_indices = np.arange(len(tr_meta), dtype=np.int64)
        if config.get('manifest_subset'):
            sub_df = pd.read_parquet(config['manifest_subset'], columns=KEY)
            sub_idx = pd.MultiIndex.from_frame(sub_df[KEY])
            keep_mask = pd.MultiIndex.from_frame(tr_meta[KEY]).isin(sub_idx)
            tr_indices = np.flatnonzero(keep_mask)

        tr_sub_meta = tr_meta.iloc[tr_indices].reset_index(drop=True)
        ev_sidecar = None
        if sidecar_root is not None and (sidecar_root / 'event_sidecar.parquet').exists():
            sc_df = pd.read_parquet(sidecar_root / 'event_sidecar.parquet')
            pos = pd.MultiIndex.from_frame(sc_df[KEY]).get_indexer(pd.MultiIndex.from_frame(tr_sub_meta[KEY]))
            if (pos < 0).any():
                raise ValueError('Train rows missing from event_sidecar.parquet')
            ev_sidecar = sc_df.iloc[pos].reset_index(drop=True)

        tr_weights = None
        if config.get('sample_weight_path'):
            w_df = pd.read_parquet(config['sample_weight_path'], columns=KEY + ['energy_match_weight'])
            w_pos = pd.MultiIndex.from_frame(w_df[KEY]).get_indexer(pd.MultiIndex.from_frame(tr_sub_meta[KEY]))
            if (w_pos < 0).any():
                raise ValueError('Train rows missing from sample_weight_path')
            tr_weights = w_df['energy_match_weight'].to_numpy(dtype=np.float32)[w_pos]

        train_ds = CachedPFDataset(
            tr_blob['items_dir'],
            tr_meta,
            indices=tr_indices,
            sample_weights=tr_weights,
            event_sidecar=ev_sidecar,
            cluster_sidecar_dir=cluster_sc_dir,
        )
        tune_ds = CachedPFDataset(tu_blob['items_dir'], tu_meta)

        scr_path = config.get('tune_screen_path')
        if scr_path and Path(scr_path).exists():
            scr_df = pd.read_parquet(scr_path, columns=KEY)
            scr_pos = pd.MultiIndex.from_frame(tu_meta[KEY]).get_indexer(pd.MultiIndex.from_frame(scr_df[KEY]))
            scr_pos = scr_pos[scr_pos >= 0]
        else:
            scr_pos = np.arange(min(4000, len(tu_meta)), dtype=np.int64)
        screen_ds = CachedPFDataset(tu_blob['items_dir'], tu_meta, indices=scr_pos)

        eval_bs = max(64, batch_size)
        screen_loader = make_pf_loader(
            screen_ds, batch_size=eval_bs, shuffle=False, workers=min(2, loader_workers),
            y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars'], mirror=False,
        )
        tune_loader = make_pf_loader(
            tune_ds, batch_size=eval_bs, shuffle=False, workers=min(2, loader_workers),
            y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars'], mirror=False,
        )

        # Subset eval slice on train for overfitting gap monitoring
        rng_tr_eval = np.random.default_rng(seed)
        tr_eval_idx = np.sort(rng_tr_eval.choice(len(train_ds), size=min(4096, len(train_ds)), replace=False))
        tr_eval_ds = CachedPFDataset(
            tr_blob['items_dir'],
            tr_meta,
            indices=tr_indices[tr_eval_idx],
            sample_weights=tr_weights[tr_eval_idx] if tr_weights is not None else None,
            event_sidecar=ev_sidecar.iloc[tr_eval_idx].reset_index(drop=True) if ev_sidecar is not None else None,
            cluster_sidecar_dir=cluster_sc_dir,
        )
        tr_eval_loader = make_pf_loader(
            tr_eval_ds, batch_size=eval_bs, shuffle=False, workers=min(2, loader_workers),
            y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars'], mirror=False,
        )

        model = model_builder(geometry, config).to(device)
        ema_model = copy.deepcopy(model).eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)

        groups = branch_groups_fn(model)
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(config['learning_rate']),
            weight_decay=float(config.get('weight_decay', 0.03)),
            betas=(0.9, 0.98),
        )
        steps_per_pass = math.ceil(len(train_ds) / batch_size)
        total_steps = passes * steps_per_pass
        warmup_steps = max(1, int(float(config.get('warmup_fraction', 0.05)) * total_steps))
        blend_steps = max(1, int(0.30 * total_steps))
        min_lr_ratio = float(config.get('min_lr', 1e-5)) / float(config['learning_rate'])

        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * min(1.0, prog)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

        start_pass = 1
        global_step = 0
        best_nll = float('inf')
        best_pass = 0
        history: list[dict] = []
        resume_pt = exp_dir / 'resume.pt'
        best_pt = exp_dir / 'best.pt'
        ema_pt = exp_dir / 'ema.pt'
        exp_pt = exp_dir / 'exposure.pt'

        if resume and resume_pt.exists():
            ckpt = torch.load(resume_pt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model'])
            ema_model.load_state_dict(ckpt['ema'])
            opt.load_state_dict(ckpt['opt'])
            scheduler.load_state_dict(ckpt['scheduler'])
            start_pass = int(ckpt['pass']) + 1
            global_step = int(ckpt['global_step'])
            best_nll = float(ckpt['best_nll'])
            best_pass = int(ckpt['best_pass'])
            history = list(ckpt['history'])

        ema_decay = float(config.get('ema_decay', 0.9996))
        screen_every = int(config.get('screen_every_passes', 2))
        y_mirror = bool(config.get('y_mirror', True))
        amp_ok = use_bf16

        try:
            for p_idx in range(start_pass, passes + 1):
                tp0 = time.perf_counter()
                tr_loader = make_pf_loader(
                    train_ds,
                    batch_size=batch_size,
                    shuffle=True,
                    workers=loader_workers,
                    seed=seed + p_idx * 101,
                    y_scint_center=bounds['y_scint_center'],
                    n_bars=bounds['n_bars'],
                    mirror=y_mirror,
                )
                model.train()
                loss_sum = 0.0
                n_seen = 0
                try:
                    for batch in tr_loader:
                        global_step += 1
                        grad_blend = min(1.0, 0.30 + 0.70 * global_step / float(blend_steps))
                        db = move_batch_to_device(batch, device)
                        opt.zero_grad(set_to_none=True)
                        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=amp_ok):
                            loss, _ = loss_fn(model, db, {'grad_blend': grad_blend, 'step': global_step, 'task': config.get('task', 'T4')})
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                        opt.step()
                        scheduler.step()
                        _update_ema(ema_model, model, ema_decay)
                        bs_n = len(db['label'])
                        loss_sum += float(loss.detach()) * bs_n
                        n_seen += bs_n
                finally:
                    shutdown_loader(tr_loader)

                train_loss = loss_sum / max(n_seen, 1)
                do_screen = (p_idx == 1) or (p_idx % screen_every == 0) or (p_idx == passes)
                row_log = {
                    'pass': p_idx,
                    'step': global_step,
                    'lr': float(scheduler.get_last_lr()[0]),
                    'train_loss': float(train_loss),
                    'seconds': float(time.perf_counter() - tp0),
                }
                if do_screen:
                    tr_s = _eval_screen(model, tr_eval_loader, device, use_bf16, pred_fn)
                    on_s = _eval_screen(model, screen_loader, device, use_bf16, pred_fn)
                    em_s = _eval_screen(ema_model, screen_loader, device, use_bf16, pred_fn)
                    row_log.update({
                        'tr_nll': tr_s['nll'],
                        'on_nll': on_s['nll'],
                        'ema_nll': em_s['nll'],
                        'ema_ba4': em_s['ba4'],
                        'ema_ba3': em_s['ba3'],
                    })
                    if em_s['nll'] < best_nll:
                        best_nll = em_s['nll']
                        best_pass = p_idx
                        torch.save({'model': ema_model.state_dict(), 'pass': p_idx, 'config': config}, best_pt)
                    print(
                        f"[{name}] pass={p_idx}/{passes} step={global_step} train_loss={train_loss:.4f} "
                        f"tr_nll={tr_s['nll']:.4f} ema_nll={em_s['nll']:.4f} "
                        f"ema_ba4={em_s['ba4']:.4f} ({row_log['seconds']:.1f}s)",
                        flush=True,
                    )
                history.append(row_log)
                torch.save({
                    'pass': p_idx,
                    'global_step': global_step,
                    'model': model.state_dict(),
                    'ema': ema_model.state_dict(),
                    'opt': opt.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_nll': best_nll,
                    'best_pass': best_pass,
                    'history': history,
                }, resume_pt)

            # Save final exposure and EMA checkpoints and run full Tune40k predictions
            torch.save({'model': ema_model.state_dict(), 'pass': passes, 'config': config}, ema_pt)
            torch.save({'model': model.state_dict(), 'pass': passes, 'config': config}, exp_pt)

            for ckpt_path, pred_name in (
                (best_pt, 'best_predictions.parquet'),
                (ema_pt, 'predictions.parquet'),
                (exp_pt, 'exposure_predictions.parquet'),
            ):
                if ckpt_path.exists():
                    state = torch.load(ckpt_path, map_location=device, weights_only=False)
                    ema_model.load_state_dict(state['model'])
                    pr = predict_pf_loader(ema_model, tune_loader, device, use_bf16, pred_fn)
                    _save_pred_parquet(exp_dir / pred_name, tu_meta, pr)

        finally:
            shutdown_loader(screen_loader)
            shutdown_loader(tune_loader)
            shutdown_loader(tr_eval_loader)

    manifest = {
        'status': 'complete',
        'UTC_started': datetime.now(timezone.utc).isoformat(),
        'UTC_finished': datetime.now(timezone.utc).isoformat(),
        'name': name,
        'config': config,
        'train_events': len(train_ds),
        'tune_events': len(tune_ds),
        'parameters': int(sum(p.numel() for p in model.parameters())),
        'best_pass': best_pass,
        'best_screen_nll': best_nll,
        'learned_scales': model.learned_attention_scales() if hasattr(model, 'learned_attention_scales') else {},
        'history': history,
        'wall_seconds': float(time.perf_counter() - t_start),
        'Cal_A_Cut_B_Confirm_payload_read': False,
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-32', action='store_true')
    parser.add_argument('--output', type=Path, default=DEFAULT_HEALTH_OUT)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    if args.smoke_32:
        cfgs = [
            ('PF_V2_anchored', {'task': 'T4', 'dim': 192, 'heads': 8, 'use_vertex_anchor': True, 'vertex_loss_weight': 0.15}),
            ('PF_V2_no_anchor', {'task': 'T4', 'dim': 192, 'heads': 8, 'use_vertex_anchor': False, 'vertex_loss_weight': 0.0}),
        ]
        rep = run_smoke_32_gate(
            model_builder=PinpointFormer,
            branch_groups_fn=pf_branch_groups,
            loss_fn=_pf_v2_loss,
            smoke_configs=cfgs,
            output_path=args.output,
            sidecar_root=DEFAULT_SIDECAR_ROOT,
            load_cluster_sidecar=False,
        )
        print(json.dumps({
            'all_passed': rep['all_passed'],
            'variants': {k: {m: v[m] for m in ('parameters', 'passed_at_update', 'final_NLL', 'final_accuracy', 'bs64_peak_GPU_MiB', 'bs64_throughput_ev_per_s', 'all_branches_grad_positive_finite')} for k, v in rep['variants'].items()},
        }, indent=2))
        return

    if args.root is None or args.config is None:
        parser.error('--root and --config are required outside --smoke-32')
    cfg = json.loads(args.config.read_text())
    man = run_pf_training(
        root=args.root,
        config=cfg,
        model_builder=PinpointFormer,
        branch_groups_fn=pf_branch_groups,
        loss_fn=_pf_v2_loss,
        pred_fn=_pf_v2_predict,
        resume=args.resume,
        attach_cluster_sidecar=False,
    )
    print(json.dumps({'status': man['status'], 'name': man['name'], 'best_pass': man['best_pass'], 'best_screen_nll': man['best_screen_nll']}, indent=2))


if __name__ == '__main__':
    main()
