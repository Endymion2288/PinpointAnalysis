"""Interactive 32-event GPU health gate and unified B1++ V4 / PF V4 trainer."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import pickle
import tempfile
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, Dataset, SequentialSampler

from analysis.flavour.inventory import write_json
from analysis.flavour.v2.partitions import FAMILY, KEY
from analysis.flavour.v2.profile import MemoryMonitor, device_batch, init_worker, shutdown_loader
from analysis.flavour.v2.r1.tasks import NON_TAU, probabilities, three_labels
from analysis.flavour.v2.r3.mapped_inputs import MappedItems
from analysis.flavour.v3.common.datasets import V3EvalDataset, V3TrainDataset
from analysis.flavour.v3.common.domains import detector_bounds
from analysis.flavour.v3.v1_b1pp.representation import collate_b1pp, extract_b1pp_item, static_plane_layout
from analysis.flavour.v3.v1_b1pp.train import (
    B1PPCollator,
    IndexedMappedSubset,
    _build_local_b1pp_cache,
    _compute_loss as _b1pp_ce_loss,
    _configure_runtime,
    _predict_dataset as _b1pp_predict_dataset,
    _save_predictions as _b1pp_save_predictions,
    _select_active_32_items,
    _update_ema as _b1pp_update_ema,
    _verify_cached_items,
)
from analysis.flavour.v3.v2_pinpointformer.train import (
    DEFAULT_DATA_ROOT,
    DEFAULT_GEOMETRY,
    DEFAULT_SIDECAR_ROOT,
    run_pf_training,
    run_smoke_32_gate,
)
from analysis.flavour.v3.v3_multitask.train import _pf_v3_loss, _pf_v3_predict
from analysis.flavour.v3.v4_tau.models import (
    PinpointFormerV4,
    ZAlignedStereoNetV4,
    b1pp_v4_branch_groups,
    pf_v4_branch_groups,
)


DEFAULT_V4_HEALTH_OUT = Path(
    '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-10/w3_10_health_report.json'
)


def _pf_v4_loss(model: nn.Module, batch: dict, ctx: dict) -> tuple[torch.Tensor, torch.Tensor]:
    base_loss, probs4 = _pf_v3_loss(model, batch, ctx)
    if getattr(model, 'enable_pauc', False) and getattr(model, 'pauc_loss', None) is not None:
        w_pauc = float(model.config.get('w_pauc', 0.20))
        l_pauc, _ = model.pauc_loss(probs4, batch['label'], batch['obs_domain_id'], batch.get('sample_weight'))
        return base_loss + w_pauc * l_pauc, probs4
    return base_loss, probs4


def _b1pp_v4_loss(model: ZAlignedStereoNetV4, logits: torch.Tensor, batch: dict, task: str) -> torch.Tensor:
    base_loss = _b1pp_ce_loss(logits, batch, task)
    if model.enable_pauc and model.pauc_loss is not None:
        probs4 = probabilities(logits.float(), task)
        if probs4.shape[1] == 3:
            probs4 = torch.cat([probs4[:, :2], probs4.new_zeros((len(probs4), 1)), probs4[:, 2:]], dim=-1)
        w_pauc = float(model.config.get('w_pauc', 0.20))
        l_pauc, _ = model.pauc_loss(probs4, batch['label'], batch['obs_domain_id'], batch.get('sample_weight'))
        return base_loss + w_pauc * l_pauc
    return base_loss


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


def _run_b1pp_v4_smoke_32(data_root: Path = DEFAULT_DATA_ROOT) -> dict:
    """Interactive 32-event GPU health gate + bs=64 VRAM benchmark for ZAlignedStereoNetV4."""
    _configure_runtime()
    device = torch.device('cuda:0')
    cap = torch.cuda.get_device_capability(0)
    use_bf16 = cap[0] >= 8

    plan = json.loads((data_root / 'plan.json').read_text())
    geometry = json.loads(Path(plan['geometry']).read_text())
    bounds = detector_bounds(geometry)
    layout = static_plane_layout(geometry, bounds)
    layout['geometry'] = geometry

    cfg = {
        'task': 'T4',
        'dim': 160,
        'n_long_layers': 4,
        'heads': 8,
        'channel_dropout': 0.0,
        'drop_path': 0.0,
        'enable_nomad': True,
        'enable_pauc': True,
        'w_pauc': 0.20,
        'pauc_beta': 0.01,
        'pauc_margin': 0.40,
        'cache_shards': 1,
    }
    with tempfile.TemporaryDirectory(prefix='b1pp-v4-smoke32-', dir=os.environ.get('PINPOINT_V3_SCRATCH', '/tmp')) as tmp:
        ds = V3TrainDataset(
            data_root,
            None,
            split='train',
            config=cfg,
            cache_dir=tmp,
            item_transform=extract_b1pp_item,
        )
        items = _select_active_32_items(ds, 'T4')
        ds.native.close()

        torch.manual_seed(42)
        np.random.seed(42)
        torch.cuda.reset_peak_memory_stats(device)
        model = ZAlignedStereoNetV4(layout, cfg, geometry=geometry).to(device)
        groups = b1pp_v4_branch_groups(model)
        init_snap = {name: p.detach().float().clone() for ps in groups.values() for name, p in ps}
        opt = torch.optim.AdamW(model.build_param_groups(1.5e-3, 0.0), lr=1.5e-3, weight_decay=0.0)

        full_batch = device_batch(collate_b1pp(items), 'cuda:0')
        t0 = time.perf_counter()
        step1_grads = None
        passed_at = None
        history = []

        for upd in range(1, 101):
            model.train()
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(full_batch).float()
                loss = _b1pp_v4_loss(model, logits, full_batch, 'T4')
            loss.backward()
            if step1_grads is None:
                step1_grads = _branch_grad_norms(groups)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

            model.eval()
            with torch.inference_mode():
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                    ev_logits = model(full_batch).float()
                ev_nll = float(_b1pp_ce_loss(ev_logits, full_batch, 'T4').item())
                ev_probs = probabilities(ev_logits, 'T4')
                ev_acc = float((ev_probs.argmax(-1) == full_batch['label']).float().mean().item())
                distinct = int(len(torch.unique(torch.round(ev_probs * 1000.0), dim=0)))
            if upd == 1 or upd % 10 == 0 or (ev_nll < 0.05 and ev_acc >= 0.999):
                history.append({'update': upd, 'NLL': ev_nll, 'accuracy': ev_acc, 'distinct_probs': distinct})
            if ev_nll < 0.05 and ev_acc >= 0.999:
                passed_at = upd
                break

        upd_rms = _branch_param_rms(groups, init_snap)
        all_grad_ok = all(math.isfinite(v) and v > 0.0 for v in (step1_grads or {}).values())
        all_upd_ok = all(math.isfinite(v) and v > 0.0 for v in upd_rms.values())

        # bs=64 VRAM & throughput microbenchmark
        del full_batch
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_bytes, _ = torch.cuda.mem_get_info(device)
        torch.cuda.reset_peak_memory_stats(device)
        bench_list = items + items
        b_bench = device_batch(collate_b1pp(bench_list), 'cuda:0')
        model.train()
        tb0 = time.perf_counter()
        n_iters = 5
        for _ in range(n_iters):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                lb = _b1pp_v4_loss(model, model(b_bench).float(), b_bench, 'T4')
            lb.backward()
            opt.step()
        torch.cuda.synchronize()
        tb_elapsed = time.perf_counter() - tb0
        peak_mib = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        ev_per_s = float((n_iters * len(bench_list)) / max(tb_elapsed, 1e-6))
        n_params = int(sum(p.numel() for p in model.parameters()))
        del b_bench, opt, model
        torch.cuda.empty_cache()

        passed = bool(passed_at is not None and all_grad_ok and all_upd_ok and peak_mib < 16.0 * 1024.0)
        return {
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
            'passed': passed,
            'history': history,
        }


def run_v4_smoke_32(output_path: Path = DEFAULT_V4_HEALTH_OUT) -> dict:
    """Run unified 32-event GPU health gate for both B1PP_V4 and PF_V4."""
    pf_tmp = output_path.with_suffix('.pf_tmp.json')
    pf_rep = run_smoke_32_gate(
        model_builder=PinpointFormerV4,
        branch_groups_fn=pf_v4_branch_groups,
        loss_fn=_pf_v4_loss,
        smoke_configs=[
            (
                'PF_V4_gn2_nomad_pauc',
                {
                    'task': 'T4',
                    'dim': 192,
                    'heads': 8,
                    'enable_dense_gn2': True,
                    'enable_nomad': True,
                    'enable_pauc': True,
                    'w_pauc': 0.20,
                    'pauc_beta': 0.01,
                    'pauc_margin': 0.40,
                    'smoke_lr': 1.5e-3,
                },
            ),
        ],
        output_path=pf_tmp,
        sidecar_root=DEFAULT_SIDECAR_ROOT,
        load_cluster_sidecar=True,
    )
    if pf_tmp.exists():
        pf_tmp.unlink()

    b1_rep = _run_b1pp_v4_smoke_32(DEFAULT_DATA_ROOT)
    pf_rep['variants']['B1PP_V4_nomad_pauc'] = b1_rep
    pf_rep['all_passed'] = all(v['passed'] for v in pf_rep['variants'].values())
    write_json(output_path, pf_rep)
    if not pf_rep['all_passed']:
        raise RuntimeError(f'V4 smoke-32 gate failed: {json.dumps(pf_rep, indent=2)}')
    return pf_rep


def train_b1pp_v4(root: Path, config: dict, resume: bool = False) -> dict:
    """Train or tail-finetune ZAlignedStereoNetV4 using shared or local B1++ inputs."""
    _configure_runtime()
    name = str(config['name'])
    task = str(config.get('task', 'T4'))
    seed = int(config.get('seed', 42))
    data_root = Path(config['data_root'])
    passes = int(config['passes'])
    batch_size = int(config.get('batch_size', 128))
    lr = float(config.get('learning_rate', 8.5e-4))
    min_lr = float(config.get('min_lr', 1e-5))
    weight_decay = float(config.get('weight_decay', 0.03))
    warmup_frac = float(config.get('warmup_fraction', 0.05))
    ema_decay = float(config.get('ema_decay', 0.999))
    y_mirror = bool(config.get('y_mirror', True))
    screen_every = int(config.get('screen_every_passes', 2))
    loader_workers = int(config.get('loader_workers', 3))
    cache_workers = int(config.get('cache_workers', 4))

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA device is required for B1++ V4 training')
    cap = torch.cuda.get_device_capability(0)
    req_bf16 = bool(config.get('bf16', True))
    if req_bf16 and cap[0] < 8:
        raise RuntimeError(f'BF16 training requires SM80+, found SM{cap[0]}{cap[1]}')
    use_bf16 = req_bf16 and cap[0] >= 8

    exp_dir = root / 'experiments' / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = exp_dir / 'manifest.json'
    resume_path = exp_dir / 'resume.pt'
    if resume and manifest_path.exists() and json.loads(manifest_path.read_text()).get('status') == 'complete':
        return json.loads(manifest_path.read_text())

    plan = json.loads((data_root / 'plan.json').read_text())
    geometry = json.loads(Path(config.get('geometry_json', plan['geometry'])).read_text())
    bounds = detector_bounds(geometry)
    layout = static_plane_layout(geometry, bounds)
    layout['geometry'] = geometry

    manifest = {
        'status': 'running',
        'UTC_started': datetime.now(timezone.utc).isoformat(),
        'name': name,
        'config': config,
        'GPU': torch.cuda.get_device_name(0),
        'capability': f'SM{cap[0]}{cap[1]}',
        'precision': 'BF16_encoder_FP32_head' if use_bf16 else 'FP32',
        'allocation_id': os.environ.get('PINPOINT_JOB_ID', 'interactive'),
        'PID': os.getpid(),
        'screen_history': [],
        'Cal_A_Cut_B_Confirm_payload_read': False,
    }
    write_json(manifest_path, manifest)

    started_wall = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f'b1ppv4-{name}-', dir=os.environ.get('PINPOINT_V3_SCRATCH', '/tmp')) as scratch:
        scratch_dir = Path(scratch)
        shared_tr = os.environ.get('PINPOINT_V3_SHARED_TRAIN_B1PP')
        shared_va = os.environ.get('PINPOINT_V3_SHARED_TUNE_B1PP')

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
            train_ds = IndexedMappedSubset(base_tr_items, pos_idx, sample_weights=None)
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

        if config.get('tune_screen_path') and Path(config['tune_screen_path']).exists():
            scr_df = pd.read_parquet(config['tune_screen_path'], columns=KEY + ['label'])
            scr_pos = pd.MultiIndex.from_frame(tune_meta[KEY]).get_indexer(pd.MultiIndex.from_frame(scr_df[KEY]))
            scr_pos = scr_pos[scr_pos >= 0]
        else:
            rng_s = np.random.default_rng(2026)
            scr_pos = np.sort(rng_s.choice(np.arange(len(tune_meta)), size=min(4000, len(tune_meta)), replace=False))
        tune_screen_ds = IndexedMappedSubset(tune_items, scr_pos)

        rng_g = np.random.default_rng(1000 + seed)
        gap_pos = np.sort(rng_g.choice(np.arange(len(train_meta)), size=min(4000, len(train_meta)), replace=False))
        train_gap_eval_ds = IndexedMappedSubset(train_ds.items, train_ds.indices[gap_pos], sample_weights=None)

        torch.manual_seed(seed)
        np.random.seed(seed)
        model = ZAlignedStereoNetV4(layout, config, geometry=geometry).cuda()
        init_ckpt = config.get('init_checkpoint')
        if init_ckpt:
            model.load_pretrained_backbone(Path(init_ckpt), 'cuda:0')
        ema_model = deepcopy(model).float().cuda()
        for p in ema_model.parameters():
            p.requires_grad_(False)

        groups = b1pp_v4_branch_groups(model)
        n_params = sum(p.numel() for p in model.parameters())
        param_groups = model.build_param_groups(lr, weight_decay)
        group_base_lrs = [float(pg['lr']) for pg in param_groups]
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

        steps_per_pass = math.ceil(len(train_ds) / batch_size)
        total_steps = max(1, passes * steps_per_pass)
        warmup_steps = max(1, int(round(warmup_frac * total_steps)))

        def _lr_scale(step: int) -> float:
            if step <= warmup_steps:
                return 0.05 + 0.95 * (step / warmup_steps)
            prog = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
            min_ratio = min_lr / lr
            return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * prog))

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

        train_collator = B1PPCollator(y_mirror=y_mirror, seed=seed, n_bars=layout['n_bars'])
        torch.cuda.reset_peak_memory_stats()

        with MemoryMonitor() as mem_mon:
            for p_idx in range(start_pass, passes + 1):
                p_start = time.perf_counter()
                gen = torch.Generator().manual_seed(seed * 1000 + p_idx)
                order = torch.randperm(len(train_ds), generator=gen).tolist()
                sampler = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
                loader_kw: dict = {
                    'batch_sampler': sampler,
                    'num_workers': loader_workers,
                    'collate_fn': train_collator,
                    'pin_memory': True,
                }
                if loader_workers > 0:
                    loader_kw.update(multiprocessing_context='spawn', worker_init_fn=init_worker, prefetch_factor=2)
                loader = DataLoader(train_ds, **loader_kw)

                model.train()
                loss_sum = 0.0
                ev_sum = 0
                try:
                    for batch in loader:
                        global_step += 1
                        scale = _lr_scale(global_step)
                        for pg, g_base in zip(optimizer.param_groups, group_base_lrs):
                            pg['lr'] = g_base * scale
                        gpu = device_batch(batch, 'cuda:0')
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                            logits = model(gpu).float()
                            loss = _b1pp_v4_loss(model, logits, gpu, task)
                        loss.backward()
                        if step1_grads is None:
                            step1_grads = _branch_grad_norms(groups)
                            for b_name, g_val in step1_grads.items():
                                if not math.isfinite(g_val) or g_val <= 0.0:
                                    raise RuntimeError(f'bad step-1 branch grad for {b_name}: {step1_grads}')
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
                        optimizer.step()
                        ema_step = (global_step + 100000) if init_ckpt else global_step
                        _b1pp_update_ema(ema_model, model, ema_step, ema_decay)

                        nb = len(batch['label'])
                        loss_sum += float(loss.detach().item()) * nb
                        ev_sum += nb
                finally:
                    shutdown_loader(loader)

                pass_sec = time.perf_counter() - p_start
                do_screen = (p_idx == 1) or (p_idx == passes) or (p_idx % screen_every == 0)
                if do_screen:
                    tr_ev = _b1pp_predict_dataset(model, train_gap_eval_ds, task, use_bf16, 128, min(2, loader_workers))
                    raw_ev = _b1pp_predict_dataset(model, tune_screen_ds, task, use_bf16, 128, min(2, loader_workers))
                    ema_ev = _b1pp_predict_dataset(ema_model, tune_screen_ds, task, use_bf16, 128, min(2, loader_workers))
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
                    if model.enable_pauc and model.pauc_loss is not None:
                        row['tau_q'] = float(model.pauc_loss.tau_q.detach())
                    if model.enable_nomad and model.nomad_scale is not None:
                        row['nomad_scale'] = float(model.nomad_scale.detach())
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

        torch.save({'model': model.state_dict(), 'config': config, 'pass': passes}, exp_dir / 'exposure.pt')
        torch.save({'model': ema_model.state_dict(), 'config': config, 'pass': passes}, exp_dir / 'ema.pt')

        tune_all_ds = IndexedMappedSubset(tune_items, np.arange(len(tune_meta)))
        pred_ema = _b1pp_predict_dataset(ema_model, tune_all_ds, task, use_bf16, 128, min(2, loader_workers))
        _b1pp_save_predictions(exp_dir / 'predictions.parquet', tune_meta, pred_ema)

        pred_exp = _b1pp_predict_dataset(model, tune_all_ds, task, use_bf16, 128, min(2, loader_workers))
        _b1pp_save_predictions(exp_dir / 'exposure_predictions.parquet', tune_meta, pred_exp)

        best_model = ZAlignedStereoNetV4(layout, config, geometry=geometry).cuda()
        best_ckpt = torch.load(exp_dir / 'best.pt', map_location='cuda:0', weights_only=False)
        best_model.load_state_dict(best_ckpt['model'])
        pred_best = _b1pp_predict_dataset(best_model, tune_all_ds, task, use_bf16, 128, min(2, loader_workers))
        _b1pp_save_predictions(exp_dir / 'best_predictions.parquet', tune_meta, pred_best)

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
    parser.add_argument('--smoke-32', action='store_true')
    parser.add_argument('--output', type=Path, default=DEFAULT_V4_HEALTH_OUT)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    if args.smoke_32:
        rep = run_v4_smoke_32(args.output)
        print(json.dumps({
            'all_passed': rep['all_passed'],
            'variants': {
                k: {
                    m: v[m]
                    for m in (
                        'parameters',
                        'passed_at_update',
                        'final_NLL',
                        'final_accuracy',
                        'bs64_peak_GPU_MiB',
                        'bs64_throughput_ev_per_s',
                        'all_branches_grad_positive_finite',
                    )
                }
                for k, v in rep['variants'].items()
            },
        }, indent=2))
        return

    if args.root is None or args.config is None:
        parser.error('--root and --config are required outside --smoke-32')
    cfg = json.loads(args.config.read_text())
    family = str(cfg.get('model_family', 'pf_v4'))
    if family == 'b1pp_v4':
        man = train_b1pp_v4(args.root, cfg, resume=args.resume)
    elif family == 'pf_v4':
        cfg.setdefault('sidecar_root', str(DEFAULT_SIDECAR_ROOT))
        if cfg.get('enable_dense_gn2', True):
            cfg['load_cluster_sidecar'] = True
        man = run_pf_training(
            root=args.root,
            config=cfg,
            model_builder=PinpointFormerV4,
            branch_groups_fn=pf_v4_branch_groups,
            loss_fn=_pf_v4_loss,
            pred_fn=_pf_v3_predict,
            resume=args.resume,
            attach_cluster_sidecar=bool(cfg.get('enable_dense_gn2', True)),
        )
    else:
        raise ValueError(f'Unknown model_family {family!r}')
    print(json.dumps({'status': man['status'], 'name': man['name'], 'best_pass': man['best_pass']}, indent=2))


if __name__ == '__main__':
    main()
