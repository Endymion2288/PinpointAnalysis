"""Train, resume, health-gate, and benchmark ZAlignedStereoNet (B1++) for V1."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import pickle
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import BatchSampler, DataLoader, Dataset, SequentialSampler

from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.v2.partitions import FAMILY, KEY
from analysis.flavour.v2.profile import MemoryMonitor, device_batch, init_worker, shutdown_loader
from analysis.flavour.v2.r1.metrics import Metrics, three_from_four, three_targets
from analysis.flavour.v2.r1.tasks import NON_TAU, output_size, probabilities, task_loss, three_labels
from analysis.flavour.v2.r3.mapped_inputs import MappedItems, write_mapped_items
from analysis.flavour.v3.common.datasets import V3EvalDataset, V3TrainDataset
from analysis.flavour.v3.common.layers import split_adamw_param_groups
from analysis.flavour.v3.common.domains import detector_bounds
from analysis.flavour.v3.v1_b1pp.models import ZAlignedStereoNet, b1pp_branch_groups
from analysis.flavour.v3.v1_b1pp.representation import collate_b1pp, extract_b1pp_item, static_plane_layout


class IndexedMappedSubset(Dataset):
    """View a filtered index subset of shared or local MappedItems with optional weights."""

    def __init__(
        self,
        items: Dataset,
        indices: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        self.items = items
        self.indices = np.asarray(indices, dtype=np.int64)
        self.sample_weights = (
            np.asarray(sample_weights, dtype=np.float32) if sample_weights is not None else None
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, pos: int) -> dict:
        item = self.items[int(self.indices[pos])]
        if self.sample_weights is not None:
            item['sample_weight'] = float(self.sample_weights[pos])
        return item


class B1PPCollator:
    """Picklable collator applying independent per-worker exact Y-mirror augmentation."""

    def __init__(self, y_mirror: bool = False, seed: int = 42, n_bars: int = 40) -> None:
        self.y_mirror = bool(y_mirror)
        self.seed = int(seed)
        self.n_bars = int(n_bars)
        self._rng: np.random.Generator | None = None

    def __call__(self, events: list[dict]) -> dict:
        if not self.y_mirror:
            return collate_b1pp(events, mirror_rng=None)
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            wid = int(info.id) if info is not None else 0
            self._rng = np.random.default_rng([self.seed, wid, os.getpid() & 0xFFFF])
        return collate_b1pp(events, mirror_rng=self._rng)


class _ThreadedB1PPLoader:
    """Shared-memory multi-threaded B1++ batch prefetcher (0 extra PyTorch RSS)."""

    def __init__(
        self,
        dataset: Dataset,
        batches: list[list[int]],
        y_mirror: bool,
        seed: int,
        pin_memory: bool = True,
        threads: int = 4,
    ) -> None:
        self.dataset = dataset
        self.batches = batches
        self.y_mirror = bool(y_mirror)
        self.seed = int(seed)
        self.pin_memory = bool(pin_memory)
        self.threads = max(1, int(threads))
        self._stop = threading.Event()

    def __len__(self) -> int:
        return len(self.batches)

    def _build_one(self, b_idx: int, idxs: list[int]) -> dict:
        events = [self.dataset[i] for i in idxs]
        rng = np.random.default_rng(self.seed + 777 + b_idx) if self.y_mirror else None
        batch = collate_b1pp(events, mirror_rng=rng)
        if self.pin_memory:
            batch = {k: v.pin_memory() if torch.is_tensor(v) else v for k, v in batch.items()}
        return batch

    def __iter__(self):
        self._stop.clear()
        if len(self.dataset) > 0:
            _ = self.dataset[0]
        out_q: queue.Queue = queue.Queue(maxsize=self.threads * 3)

        def _producer() -> None:
            try:
                with ThreadPoolExecutor(max_workers=self.threads) as pool:
                    in_flight = []
                    for b_idx, idxs in enumerate(self.batches):
                        if self._stop.is_set():
                            break
                        in_flight.append(pool.submit(self._build_one, b_idx, idxs))
                        if len(in_flight) >= self.threads * 2:
                            out_q.put(('ok', in_flight.pop(0).result()))
                    for fut in in_flight:
                        if self._stop.is_set():
                            break
                        out_q.put(('ok', fut.result()))
            except BaseException as exc:
                out_q.put(('err', exc))
            finally:
                out_q.put(('done', None))

        t = threading.Thread(target=_producer, daemon=True)
        t.start()
        try:
            while True:
                tag, payload = out_q.get()
                if tag == 'done':
                    break
                if tag == 'err':
                    raise payload
                yield payload
        finally:
            self._stop.set()

    def close(self) -> None:
        self._stop.set()


def _configure_runtime() -> None:
    torch.set_num_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    torch.multiprocessing.set_sharing_strategy('file_system')


def _compute_loss(logits: torch.Tensor, batch: dict, task: str) -> torch.Tensor:
    labels = batch['label']
    if 'sample_weight' not in batch:
        return task_loss(logits, labels, task)
    w = batch['sample_weight'].float().clamp_min(0.0)
    target = three_labels(labels) if task == 'T3' else labels
    ce = F.cross_entropy(logits.float(), target, reduction='none')
    return (ce * w).sum() / w.sum().clamp_min(1e-8)


def _branch_grad_norms(groups: dict[str, list[tuple[str, nn.Parameter]]]) -> dict[str, float]:
    out = {}
    for branch, params in groups.items():
        sq = 0.0
        for _, p in params:
            if p.grad is not None:
                sq += float(p.grad.detach().float().square().sum().item())
        out[branch] = float(math.sqrt(max(sq, 0.0)))
    return out


def _branch_param_rms(groups: dict[str, list[tuple[str, nn.Parameter]]], snap: dict[str, torch.Tensor]) -> dict[str, float]:
    out = {}
    for branch, params in groups.items():
        sq = 0.0
        n = 0
        for name, p in params:
            diff = (p.detach().float() - snap[name]).square()
            sq += float(diff.sum().item())
            n += p.numel()
        out[branch] = float(math.sqrt(sq / max(n, 1)))
    return out


def _update_ema(ema_model: nn.Module, model: nn.Module, global_step: int, target_decay: float) -> None:
    decay = min(target_decay, (1.0 + global_step) / (10.0 + global_step))
    one_m = 1.0 - decay
    with torch.no_grad():
        for p_ema, p in zip(ema_model.parameters(), model.parameters()):
            p_ema.data.mul_(decay).add_(p.data.float(), alpha=one_m)
        for b_ema, b in zip(ema_model.buffers(), model.buffers()):
            b_ema.copy_(b)


def _balanced_weights(y: np.ndarray, k: int) -> np.ndarray:
    cnt = np.bincount(y, minlength=k)
    return (len(y) / (float(k) * np.maximum(cnt, 1)))[y]


def _predict_dataset(
    model: nn.Module,
    dataset: Dataset,
    task: str,
    bf16: bool,
    batch_size: int = 128,
    num_workers: int = 2,
    amp_dtype: torch.dtype | None = None,
) -> dict:
    model.eval()
    if amp_dtype is None:
        amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16
    batches_eval = list(BatchSampler(SequentialSampler(dataset), batch_size=batch_size, drop_last=False))
    eff_w = 0 if len(dataset) < 512 else (num_workers if num_workers > 0 else 3)
    loader_kw: dict = {
        'batch_sampler': batches_eval,
        'num_workers': eff_w,
        'collate_fn': B1PPCollator(y_mirror=False),
        'pin_memory': False,
    }
    if eff_w > 0:
        loader_kw.update(multiprocessing_context='fork', worker_init_fn=init_worker, prefetch_factor=2)
    loader = DataLoader(dataset, **loader_kw)
    all_logits = []
    all_labels = []
    all_dom = []
    all_idx = []
    try:
        with torch.inference_mode():
            for batch in loader:
                gpu = device_batch(batch, 'cuda:0')
                with torch.autocast('cuda', dtype=amp_dtype, enabled=bf16):
                    logits = model(gpu).float()
                all_logits.append(logits.cpu().numpy())
                all_labels.append(batch['label'].numpy())
                all_dom.append(batch['obs_domain_id'].numpy())
                all_idx.append(batch['index'].numpy())
    finally:
        shutdown_loader(loader)

    logits_np = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    dom_np = np.concatenate(all_dom, axis=0)
    idx_np = np.concatenate(all_idx, axis=0)
    with torch.inference_mode():
        probs_np = probabilities(torch.from_numpy(logits_np), task).numpy()

    k = probs_np.shape[1]
    if task == 'T3':
        non_tau = labels_np != 2
        y_eval = three_labels(torch.from_numpy(labels_np[non_tau])).numpy()
        p_eval = probs_np[non_tau]
        z_eval = logits_np[non_tau]
        ce = float(F.cross_entropy(torch.from_numpy(z_eval), torch.from_numpy(y_eval)).item())
        ba = float(Metrics(y_eval, p_eval).report(_balanced_weights(y_eval, 3))['BA'])
        metrics = {'NLL3': ce, 'BA3': ba, 'eval_rows': int(non_tau.sum())}
    else:
        ce4 = float(F.cross_entropy(torch.from_numpy(logits_np), torch.from_numpy(labels_np)).item())
        ba4 = float(Metrics(labels_np, probs_np).report(_balanced_weights(labels_np, 4))['BA'])
        non_tau = labels_np != 2
        y3 = three_targets(labels_np[non_tau])
        p3 = three_from_four(probs_np[non_tau])
        m3 = Metrics(y3, p3).report(_balanced_weights(y3, 3))
        metrics = {
            'NLL4': ce4,
            'BA4': ba4,
            'NLL3': float(m3['NLL']),
            'BA3': float(m3['BA']),
            'eval_rows': int(len(labels_np)),
        }
        metrics['NLL'] = ce4
        metrics['BA'] = ba4
    if 'NLL' not in metrics:
        metrics['NLL'] = metrics['NLL3']
        metrics['BA'] = metrics['BA3']
    metrics.update({
        'finite': bool(np.isfinite(logits_np).all()),
        'logits': logits_np,
        'probs': probs_np,
        'labels': labels_np,
        'obs_domain_id': dom_np,
        'indexes': idx_np,
        'distinct_probs': int(len(np.unique(np.round(probs_np, 4), axis=0))),
    })
    return metrics


def _save_predictions(path: Path, metadata: pd.DataFrame, pred: dict) -> None:
    df = metadata[KEY + FAMILY + ['label']].copy().reset_index(drop=True)
    if not np.array_equal(pred['labels'], df['label'].to_numpy(dtype=np.int64)):
        raise ValueError('prediction label order mismatch against metadata')
    df['obs_domain_id'] = pred['obs_domain_id'].astype(np.int8)
    logits = pred['logits']
    probs = pred['probs']
    for c in range(logits.shape[1]):
        df[f'z{c}'] = logits[:, c].astype(np.float32)
    for c in range(probs.shape[1]):
        df[f'p{c}'] = probs[:, c].astype(np.float32)
    df.to_parquet(path, index=False, compression=COMPRESSION)


def _verify_cached_items(raw_ds: Dataset, mapped: MappedItems, count: int = 16) -> dict:
    n = len(raw_ds)
    pos = np.linspace(0, n - 1, min(count, n), dtype=np.int64)
    for idx in pos:
        a = raw_ds[int(idx)]
        b = mapped[int(idx)]
        for k, v in a.items():
            if isinstance(v, np.ndarray):
                if v.dtype.kind in ('i', 'u', 'b'):
                    if not np.array_equal(v, b[k]):
                        raise AssertionError(f'cached array mismatch for key={k} at idx={idx}')
                else:
                    if not np.allclose(v.astype(np.float32), b[k].astype(np.float32), atol=1e-5, rtol=0.0):
                        raise AssertionError(f'cached float mismatch for key={k} at idx={idx}')
            elif int(v) != int(b[k]):
                raise AssertionError(f'cached scalar mismatch for key={k} at idx={idx}: {v} vs {b[k]}')
    return {'checked_events': int(len(pos)), 'bit_exact_ok': True}


def _identity_collate(xs: list[dict]) -> list[dict]:
    return xs


def _build_local_b1pp_cache(ds: Dataset, cache_dir: Path, workers: int = 4) -> tuple[MappedItems, float]:
    t0 = time.perf_counter()
    loader_kw: dict = {
        'batch_sampler': BatchSampler(SequentialSampler(ds), batch_size=64, drop_last=False),
        'num_workers': workers,
        'collate_fn': _identity_collate,
    }
    if workers > 0:
        loader_kw.update(multiprocessing_context='spawn', worker_init_fn=init_worker, prefetch_factor=2)
    loader = DataLoader(ds, **loader_kw)
    try:
        mapped = write_mapped_items(loader, cache_dir)
    finally:
        shutdown_loader(loader)
    return mapped, time.perf_counter() - t0


def _select_active_32_items(ds: V3TrainDataset, task: str) -> list[dict]:
    """Select 32 class-balanced non-empty Train events (obs_domain_id > 0) from shard 000."""
    classes = list(NON_TAU) if task == 'T3' else [0, 1, 2, 3]
    quotas = {c: 8 for c in classes} if task == 'T4' else {0: 11, 1: 11, 3: 10}
    buckets: dict[int, list[dict]] = {c: [] for c in classes}
    labels = ds.metadata['label'].to_numpy(dtype=np.int64)
    for pos in range(min(2500, len(labels))):
        lbl = int(labels[pos])
        if lbl in buckets and len(buckets[lbl]) < quotas[lbl]:
            it = ds[pos]
            if int(it['obs_domain_id']) > 0:
                buckets[lbl].append(it)
            if all(len(buckets[c]) >= quotas[c] for c in classes):
                break
    # Round-robin interleave so every mini-batch covers all classes cleanly
    items: list[dict] = []
    max_q = max(quotas.values())
    for k in range(max_q):
        for c in classes:
            if k < len(buckets[c]):
                items.append(buckets[c][k])
    if len(items) != 32:
        raise RuntimeError(f'expected 32 active smoke events, got {len(items)}')
    return items


def run_smoke_32(
    data_root: Path,
    out_dir: Path,
    batch_benchmark_sec: float = 0.0,
) -> dict:
    """Execute W3-04 32-event interactive GPU overfitting gate and branch gradient check."""
    _configure_runtime()
    out_dir.mkdir(parents=True, exist_ok=True)
    device_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    use_bf16 = cap[0] >= 8
    precision = 'BF16_encoder_FP32_head' if use_bf16 else 'FP32'

    plan = json.loads((data_root / 'plan.json').read_text())
    geometry = json.loads(Path(plan['geometry']).read_text())
    bounds = detector_bounds(geometry)
    layout = static_plane_layout(geometry, bounds)

    results = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'GPU': device_name,
        'capability': f'SM{cap[0]}{cap[1]}',
        'precision': precision,
        'tasks': {},
    }

    with tempfile.TemporaryDirectory(prefix='v3-smoke32-', dir=os.environ.get('PINPOINT_V3_SCRATCH', '/tmp')) as tmp:
        for task in ('T4', 'T3'):
            torch.manual_seed(42)
            np.random.seed(42)
            cfg = {
                'task': task,
                'dim': 160,
                'n_long_layers': 4,
                'heads': 8,
                'channel_dropout': 0.0,
                'drop_path': 0.0,
                'y_mirror': False,
                'cache_shards': 2,
            }
            ds = V3TrainDataset(
                data_root,
                None,
                split='train',
                config=cfg,
                cache_dir=tmp,
                item_transform=extract_b1pp_item,
            )
            items = _select_active_32_items(ds, task)
            ds.native.close()

            model = ZAlignedStereoNet(layout, cfg).cuda()
            n_params = sum(p.numel() for p in model.parameters())
            groups = b1pp_branch_groups(model)
            init_snap = {name: p.detach().float().clone() for ps in groups.values() for name, p in ps}
            opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=0.0)

            batches = [collate_b1pp(items[i:i + 16]) for i in range(0, len(items), 16)]
            full_batch = device_batch(collate_b1pp(items), 'cuda:0')

            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            step1_grads = None
            passed_at = None
            history = []
            for upd in range(1, 101):
                model.train()
                for b in batches:
                    gpu = device_batch(b, 'cuda:0')
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                        logits = model(gpu).float()
                        loss = _compute_loss(logits, gpu, task)
                    loss.backward()
                    if step1_grads is None:
                        step1_grads = _branch_grad_norms(groups)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    opt.step()

                model.eval()
                with torch.inference_mode():
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                        ev_logits = model(full_batch).float()
                    ev_nll = float(_compute_loss(ev_logits, full_batch, task).item())
                    ev_probs = probabilities(ev_logits, task)
                    ev_target = three_labels(full_batch['label']) if task == 'T3' else full_batch['label']
                    ev_acc = float((ev_probs.argmax(-1) == ev_target).float().mean().item())
                    distinct = int(len(torch.unique(torch.round(ev_probs * 1000.0), dim=0)))
                if upd in (1, 10, 25, 50, 75, 100) or (passed_at is None and ev_nll < 0.05 and ev_acc == 1.0):
                    history.append({
                        'update': upd,
                        'NLL': ev_nll,
                        'accuracy': ev_acc,
                        'distinct_probs': distinct,
                    })
                if passed_at is None and ev_nll < 0.05 and ev_acc == 1.0:
                    passed_at = upd
                    break

            upd_rms = _branch_param_rms(groups, init_snap)
            all_grad_ok = all(math.isfinite(v) and v > 0.0 for v in (step1_grads or {}).values())
            all_upd_ok = all(math.isfinite(v) and v > 0.0 for v in upd_rms.values())
            gate_ok = bool(
                passed_at is not None
                and all_grad_ok
                and all_upd_ok
                and math.isfinite(history[-1]['NLL'])
                and history[-1]['NLL'] < 0.05
                and history[-1]['accuracy'] == 1.0
            )
            results['tasks'][task] = {
                'parameters': int(n_params),
                'events': int(len(items)),
                'passed_at_update': passed_at,
                'final_NLL': history[-1]['NLL'],
                'final_accuracy': history[-1]['accuracy'],
                'distinct_probs': history[-1]['distinct_probs'],
                'step1_branch_grad_norms': step1_grads,
                'branch_param_update_rms': upd_rms,
                'all_branches_grad_positive_finite': all_grad_ok,
                'all_branches_updated': all_upd_ok,
                'peak_GPU_MiB': float(torch.cuda.max_memory_allocated() / 2**20),
                'seconds': time.perf_counter() - t0,
                'history': history,
                'gate_passed': gate_ok,
            }
            if not gate_ok:
                raise RuntimeError(f'W3-04 smoke-32 failed for {task}: {json.dumps(results["tasks"][task], indent=2)}')

        # Throughput & VRAM microbenchmark at production batch_size=64 (1 worker vs 2 concurrent procs)
        bench_ds = V3TrainDataset(
            data_root,
            None,
            split='train',
            config={'task': 'T4', 'cache_shards': 2},
            cache_dir=tmp,
            item_transform=extract_b1pp_item,
        )
        sample_items = [bench_ds[i] for i in range(512)]
        bench_ds.native.close()
        bench_batch = device_batch(collate_b1pp(sample_items[:64]), 'cuda:0')
        model_b = ZAlignedStereoNet(layout, {'task': 'T4'}).cuda()
        opt_b = torch.optim.AdamW(model_b.parameters(), lr=6e-4, weight_decay=0.03)
        model_b.train()
        # Warmup
        for _ in range(3):
            opt_b.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                loss = _compute_loss(model_b(bench_batch).float(), bench_batch, 'T4')
            loss.backward()
            opt_b.step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t_start = time.perf_counter()
        n_steps = 40
        for i in range(n_steps):
            b = collate_b1pp(sample_items[(i * 64) % 512 : ((i * 64) % 512) + 64], mirror_rng=np.random.default_rng(i))
            if len(b['label']) < 64:
                b = collate_b1pp(sample_items[:64], mirror_rng=np.random.default_rng(i))
            gpu = device_batch(b, 'cuda:0')
            opt_b.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                loss = _compute_loss(model_b(gpu).float(), gpu, 'T4')
            loss.backward()
            opt_b.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t_start
        results['bs64_single_benchmark'] = {
            'steps': n_steps,
            'batch_size': 64,
            'events_per_sec': float(n_steps * 64 / dt),
            'ms_per_step': float(dt * 1000.0 / n_steps),
            'peak_GPU_MiB': float(torch.cuda.max_memory_allocated() / 2**20),
            'est_parallel_2_VRAM_MiB': float(2.0 * torch.cuda.max_memory_allocated() / 2**20),
        }

    report_path = out_dir / 'w3_04_health_report.json'
    write_json(report_path, results)
    print(json.dumps(results, indent=2))
    return results


def train_one(root: Path, config_path: Path, resume: bool = False) -> dict:
    _configure_runtime()
    config = json.loads(config_path.read_text())
    name = config['name']
    task = config['task']
    seed = int(config.get('seed', 42))
    data_root = Path(config['data_root'])
    passes = int(config['passes'])
    batch_size = int(config.get('batch_size', 64))
    lr = float(config.get('learning_rate', 6e-4))
    min_lr = float(config.get('min_lr', 1e-5))
    weight_decay = float(config.get('weight_decay', 0.03))
    warmup_frac = float(config.get('warmup_fraction', 0.05))
    ema_decay = float(config.get('ema_decay', 0.999))
    default_mirror = not bool(config.get('pe_only', False))
    y_mirror = bool(config.get('y_mirror', default_mirror))
    screen_every = int(config.get('screen_every_passes', 2))
    loader_workers = int(config.get('loader_workers', 2))
    cache_workers = int(config.get('cache_workers', 4))

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA device is required for B1++ training')
    cap = torch.cuda.get_device_capability(0)
    req_bf16 = bool(config.get('bf16', True))
    if req_bf16 and cap[0] < 8 and bool(config.get('strict_bf16', False)):
        raise RuntimeError(f'BF16 training requires SM80+, found SM{cap[0]}{cap[1]}')
    use_bf16 = req_bf16 and cap[0] >= 7
    amp_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
    use_grad_scaler = use_bf16 and (amp_dtype == torch.float16)

    exp_dir = root / 'experiments' / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = exp_dir / 'manifest.json'
    resume_path = exp_dir / 'resume.pt'

    plan = json.loads((data_root / 'plan.json').read_text())
    geometry = json.loads(Path(plan['geometry']).read_text())
    bounds = detector_bounds(geometry)
    layout = static_plane_layout(geometry, bounds)

    manifest = {
        'status': 'running',
        'UTC_started': datetime.now(timezone.utc).isoformat(),
        'name': name,
        'config': config,
        'GPU': torch.cuda.get_device_name(0),
        'capability': f'SM{cap[0]}{cap[1]}',
        'precision': ('BF16_encoder_FP32_head' if amp_dtype == torch.bfloat16 else 'FP16_encoder_FP32_head') if use_bf16 else 'FP32',
        'allocation_id': os.environ.get('PINPOINT_JOB_ID', 'interactive'),
        'PID': os.getpid(),
        'screen_history': [],
        'Cal_A_Cut_B_Confirm_payload_read': False,
    }
    write_json(manifest_path, manifest)

    started_wall = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f'b1pp-{name}-', dir=os.environ.get('PINPOINT_V3_SCRATCH', '/tmp')) as scratch:
        scratch_dir = Path(scratch)
        shared_tr = os.environ.get('PINPOINT_V3_SHARED_TRAIN_B1PP') if config.get('drop_modality') is None else None
        shared_va = os.environ.get('PINPOINT_V3_SHARED_TUNE_B1PP') if config.get('drop_modality') is None else None

        if shared_tr and Path(shared_tr).exists():
            with open(shared_tr, 'rb') as f:
                tr_pkg = pickle.load(f)
            base_tr_items = MappedItems(tr_pkg['items_dir'])
            base_tr_meta = tr_pkg['metadata']
            keep = np.ones(len(base_tr_meta), dtype=bool)
            if config.get('manifest_subset'):
                sub_df = pd.read_parquet(config['manifest_subset'], columns=KEY)
                sub_idx = pd.MultiIndex.from_frame(sub_df[KEY])
                keep &= pd.MultiIndex.from_frame(base_tr_meta[KEY]).isin(sub_idx)
            if task == 'T3':
                keep &= base_tr_meta['label'].isin(NON_TAU).to_numpy()
            pos_idx = np.flatnonzero(keep)
            train_meta = base_tr_meta.iloc[pos_idx].reset_index(drop=True)
            sample_w = None
            if config.get('sample_weight_path'):
                w_df = pd.read_parquet(config['sample_weight_path'], columns=KEY + ['energy_match_weight'])
                w_pos = pd.MultiIndex.from_frame(w_df[KEY]).get_indexer(pd.MultiIndex.from_frame(train_meta[KEY]))
                if (w_pos < 0).any():
                    raise ValueError('missing train rows in sample_weight_path')
                sample_w = w_df['energy_match_weight'].to_numpy(dtype=np.float32)[w_pos]
            train_ds = IndexedMappedSubset(base_tr_items, pos_idx, sample_weights=sample_w)
            cache_audit = {'train_shared': True, 'train_rows': int(len(train_meta))}
        else:
            raw_tr = V3TrainDataset(
                data_root,
                None,
                split='train',
                config=config,
                cache_dir=scratch_dir,
                item_transform=extract_b1pp_item,
            )
            tr_mapped, tr_sec = _build_local_b1pp_cache(raw_tr, scratch_dir / 'train_b1pp', workers=cache_workers)
            chk_tr = _verify_cached_items(raw_tr, tr_mapped, count=16)
            raw_tr.native.close()
            train_meta = raw_tr.metadata[KEY + FAMILY + ['label']].copy().reset_index(drop=True)
            train_ds = IndexedMappedSubset(tr_mapped, np.arange(len(train_meta)), sample_weights=raw_tr.sample_weights)
            cache_audit = {'train_shared': False, 'train_cache_sec': tr_sec, 'train_verify': chk_tr, 'train_rows': int(len(train_meta))}

        if shared_va and Path(shared_va).exists():
            with open(shared_va, 'rb') as f:
                va_pkg = pickle.load(f)
            tune_items = MappedItems(va_pkg['items_dir'])
            tune_meta = va_pkg['metadata']
            cache_audit['tune_shared'] = True
        else:
            val_cfg = {k: v for k, v in config.items() if k not in ('manifest_subset', 'sample_weight_path', 'task')}
            val_cfg['task'] = task
            raw_va = V3EvalDataset(
                data_root,
                split='validation',
                config=val_cfg,
                cache_dir=scratch_dir,
                item_transform=extract_b1pp_item,
            )
            tune_items, va_sec = _build_local_b1pp_cache(raw_va, scratch_dir / 'tune_b1pp', workers=cache_workers)
            chk_va = _verify_cached_items(raw_va, tune_items, count=16)
            raw_va.native.close()
            tune_meta = raw_va.metadata[KEY + FAMILY + ['label']].copy().reset_index(drop=True)
            cache_audit.update({'tune_shared': False, 'tune_cache_sec': va_sec, 'tune_verify': chk_va})

        # Build deterministic 4k Tune screen & 4k Train-gap eval subsets
        if config.get('tune_screen_path') and Path(config['tune_screen_path']).exists():
            scr_df = pd.read_parquet(config['tune_screen_path'], columns=KEY + ['label'])
            if task == 'T3':
                scr_df = scr_df[scr_df['label'].isin(NON_TAU)].reset_index(drop=True)
            scr_pos = pd.MultiIndex.from_frame(tune_meta[KEY]).get_indexer(pd.MultiIndex.from_frame(scr_df[KEY]))
            scr_pos = scr_pos[scr_pos >= 0]
        else:
            rng_s = np.random.default_rng(2026)
            pool = np.flatnonzero(tune_meta['label'].isin(NON_TAU).to_numpy()) if task == 'T3' else np.arange(len(tune_meta))
            scr_pos = np.sort(rng_s.choice(pool, size=min(4000, len(pool)), replace=False))
        tune_screen_ds = IndexedMappedSubset(tune_items, scr_pos)

        rng_g = np.random.default_rng(1000 + seed)
        gap_pos = np.sort(rng_g.choice(np.arange(len(train_meta)), size=min(4000, len(train_meta)), replace=False))
        # Strip sample weights for unweighted eval parity
        train_gap_eval_ds = IndexedMappedSubset(
            train_ds.items, train_ds.indices[gap_pos], sample_weights=None
        )

        torch.manual_seed(seed)
        np.random.seed(seed)
        model = ZAlignedStereoNet(layout, config).cuda()
        ema_model = deepcopy(model).float().cuda()
        for p in ema_model.parameters():
            p.requires_grad_(False)

        groups = b1pp_branch_groups(model)
        n_params = sum(p.numel() for p in model.parameters())
        opt_groups = split_adamw_param_groups(model, lr, weight_decay)
        optimizer = torch.optim.AdamW(opt_groups, lr=lr, weight_decay=weight_decay)

        steps_per_pass = math.ceil(len(train_ds) / batch_size)
        total_steps = max(1, passes * steps_per_pass)
        warmup_steps = max(1, int(round(warmup_frac * total_steps)))

        def _lr_for_step(step: int) -> float:
            if step <= warmup_steps:
                return lr * (0.05 + 0.95 * (step / warmup_steps))
            prog = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
            return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * prog))

        start_pass = 1
        global_step = 0
        best_ema_nll = float('inf')
        best_pass = 0
        screen_history = []
        step1_grads = None

        if resume and resume_path.exists():
            ckpt = torch.load(resume_path, map_location='cuda:0', weights_only=False)
            model.load_state_dict(ckpt['model'])
            ema_model.load_state_dict(ckpt['ema'])
            optimizer.load_state_dict(ckpt['optimizer'])
            start_pass = int(ckpt['completed_pass']) + 1
            global_step = int(ckpt['global_step'])
            best_ema_nll = float(ckpt['best_ema_nll'])
            best_pass = int(ckpt['best_pass'])
            screen_history = list(ckpt['screen_history'])
            step1_grads = ckpt.get('step1_grads')

        train_eval_collator = B1PPCollator(y_mirror=y_mirror, seed=seed, n_bars=layout['n_bars'])
        scaler = torch.amp.GradScaler('cuda', enabled=use_grad_scaler)
        torch.cuda.reset_peak_memory_stats()

        with MemoryMonitor() as mem_mon:
            for p_idx in range(start_pass, passes + 1):
                p_start = time.perf_counter()
                gen = torch.Generator().manual_seed(seed * 1000 + p_idx)
                order = torch.randperm(len(train_ds), generator=gen).tolist()
                sampler = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
                eff_tr_w = 0 if len(train_ds) < 512 else (loader_workers if loader_workers > 0 else 5)
                loader_kw = {
                    'batch_sampler': sampler,
                    'num_workers': eff_tr_w,
                    'collate_fn': train_eval_collator,
                    'pin_memory': False,
                }
                if eff_tr_w > 0:
                    loader_kw.update(multiprocessing_context='fork', worker_init_fn=init_worker, prefetch_factor=2)
                loader = DataLoader(train_ds, **loader_kw)

                model.train()
                loss_sum = 0.0
                ev_sum = 0
                try:
                    for batch in loader:
                        global_step += 1
                        cur_lr = _lr_for_step(global_step)
                        for pg in optimizer.param_groups:
                            pg['lr'] = cur_lr
                        gpu = device_batch(batch, 'cuda:0')
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast('cuda', dtype=amp_dtype, enabled=use_bf16):
                            logits = model(gpu).float()
                            loss = _compute_loss(logits, gpu, task)
                        if use_grad_scaler:
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                        else:
                            loss.backward()
                        if step1_grads is None:
                            step1_grads = _branch_grad_norms(groups)
                            dropped_branch = {'pixel': 'pixel_transverse', 'tracker': 'spectrometer'}.get(
                                config.get('drop_modality')
                            )
                            for b_name, g_val in step1_grads.items():
                                if not math.isfinite(g_val):
                                    raise RuntimeError(f'non-finite grad for {b_name}: {step1_grads}')
                                if b_name == dropped_branch:
                                    if g_val != 0.0:
                                        raise RuntimeError(f'leakage: {b_name} had non-zero grad under drop_modality: {g_val}')
                                elif g_val <= 0.0:
                                    raise RuntimeError(f'zero branch grad at step 1 for {b_name}: {step1_grads}')
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        if use_grad_scaler:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        _update_ema(ema_model, model, global_step, ema_decay)

                        nb = len(batch['label'])
                        loss_sum += float(loss.detach().item()) * nb
                        ev_sum += nb
                finally:
                    shutdown_loader(loader)

                pass_sec = time.perf_counter() - p_start
                do_screen = (p_idx == 1) or (p_idx == passes) or (p_idx % screen_every == 0)
                if do_screen:
                    tr_ev = _predict_dataset(model, train_gap_eval_ds, task, use_bf16, 128, min(2, loader_workers))
                    raw_ev = _predict_dataset(model, tune_screen_ds, task, use_bf16, 128, min(2, loader_workers))
                    ema_ev = _predict_dataset(ema_model, tune_screen_ds, task, use_bf16, 128, min(2, loader_workers))
                    row = {
                        'pass': p_idx,
                        'step': global_step,
                        'lr': float(optimizer.param_groups[0]['lr']),
                        'pass_sec': float(pass_sec),
                        'train_batch_loss': float(loss_sum / max(ev_sum, 1)),
                        'train_gap_NLL': float(tr_ev['NLL']),
                        'train_gap_BA': float(tr_ev['BA']),
                        'tune_screen_raw_NLL': float(raw_ev['NLL']),
                        'tune_screen_raw_BA': float(raw_ev['BA']),
                        'tune_screen_ema_NLL': float(ema_ev['NLL']),
                        'tune_screen_ema_BA': float(ema_ev['BA']),
                        'tune_screen_ema_BA3': float(ema_ev['BA3']),
                        'gap_ema_NLL': float(ema_ev['NLL'] - tr_ev['NLL']),
                    }
                    screen_history.append(row)
                    if ema_ev['NLL'] < best_ema_nll:
                        best_ema_nll = float(ema_ev['NLL'])
                        best_pass = p_idx
                        torch.save({'model': ema_model.state_dict(), 'config': config, 'pass': p_idx}, exp_dir / 'best.pt')

                    tmp_res = exp_dir / 'resume.tmp.pt'
                    torch.save({
                        'model': model.state_dict(),
                        'ema': ema_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'completed_pass': p_idx,
                        'global_step': global_step,
                        'best_ema_nll': best_ema_nll,
                        'best_pass': best_pass,
                        'screen_history': screen_history,
                        'step1_grads': step1_grads,
                    }, tmp_res)
                    tmp_res.replace(resume_path)
                    manifest['screen_history'] = screen_history
                    manifest['last_pass'] = p_idx
                    manifest['best_ema_nll'] = best_ema_nll
                    manifest['best_pass'] = best_pass
                    write_json(manifest_path, manifest)
                    print(
                        f"[{name}] pass={p_idx}/{passes} step={global_step} "
                        f"train_loss={row['train_batch_loss']:.4f} tr_nll={row['train_gap_NLL']:.4f} "
                        f"ema_nll={row['tune_screen_ema_NLL']:.4f} ema_ba={row['tune_screen_ema_BA']:.4f} "
                        f"ema_ba3={row['tune_screen_ema_BA3']:.4f} ({pass_sec:.1f}s)",
                        flush=True,
                    )

        # Save final endpoints & evaluate all 40k Tune events
        torch.save({'model': model.state_dict(), 'config': config, 'pass': passes}, exp_dir / 'exposure.pt')
        torch.save({'model': ema_model.state_dict(), 'config': config, 'pass': passes}, exp_dir / 'ema.pt')

        tune_all_ds = IndexedMappedSubset(tune_items, np.arange(len(tune_meta)))
        pred_ema = _predict_dataset(ema_model, tune_all_ds, task, use_bf16, 128, min(2, loader_workers))
        _save_predictions(exp_dir / 'predictions.parquet', tune_meta, pred_ema)

        pred_exp = _predict_dataset(model, tune_all_ds, task, use_bf16, 128, min(2, loader_workers))
        _save_predictions(exp_dir / 'exposure_predictions.parquet', tune_meta, pred_exp)

        best_model = ZAlignedStereoNet(layout, config).cuda()
        best_ckpt = torch.load(exp_dir / 'best.pt', map_location='cuda:0', weights_only=False)
        best_model.load_state_dict(best_ckpt['model'])
        pred_best = _predict_dataset(best_model, tune_all_ds, task, use_bf16, 128, min(2, loader_workers))
        _save_predictions(exp_dir / 'best_predictions.parquet', tune_meta, pred_best)

        def _strip(d: dict) -> dict:
            return {k: v for k, v in d.items() if k not in ('logits', 'probs', 'labels', 'obs_domain_id', 'indexes')}

        manifest.update({
            'status': 'complete',
            'UTC_finished': datetime.now(timezone.utc).isoformat(),
            'wall_seconds': float(time.perf_counter() - started_wall),
            'parameters': int(n_params),
            'train_rows': int(len(train_ds)),
            'total_steps': int(global_step),
            'cache_audit': cache_audit,
            'step1_branch_grad_norms': step1_grads,
            'peak_GPU_MiB': float(torch.cuda.max_memory_allocated() / 2**20),
            'peak_host_RSS_MiB': mem_mon.peak,
            'best_pass': int(best_pass),
            'best_ema_screen_NLL': float(best_ema_nll),
            'screen_history': screen_history,
            'tune40k_ema': _strip(pred_ema),
            'tune40k_exposure': _strip(pred_exp),
            'tune40k_best_screen': _strip(pred_best),
        })
        write_json(manifest_path, manifest)
        if resume_path.exists():
            resume_path.unlink()

    print(json.dumps({k: v for k, v in manifest.items() if k != 'screen_history'}, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15'))
    parser.add_argument('--data-root', type=Path, default=Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R2/data'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--smoke-32', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    if args.smoke_32:
        run_smoke_32(args.data_root, args.root / 'W3-04')
    else:
        if args.config is None:
            parser.error('--config is required unless --smoke-32 is set')
        train_one(args.root, args.config, resume=args.resume)


if __name__ == '__main__':
    main()
