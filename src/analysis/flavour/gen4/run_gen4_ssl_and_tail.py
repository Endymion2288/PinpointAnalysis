"""Runner for Pinpoint-Gen4 Pillar-III & Full-System Integrated Training:
  - Stage 0: PinpointSSL-v2 pretraining on unlabeled `10085-10088` SSL corpus
  - Stage 1: Label-efficiency (10% / 25% / 100%) & conflict-free optimization ablations
  - Stage 2: 4-Stage curriculum (Conflict-Free Main -> LORT Head Recalibration ->
             Unfrozen Top-65% Live Hard-Negative NSWC-TPAUC + Quantile-Pivot + INFERNO)
             across 3 seeds `{17, 42, 2026}` with Sub-Aperture Orbits & In-Place NC+mu Surgery
  - Stage 2 Aggregate: 3-seed ensemble + orthogonal DiffPhys-SBI fusion + POT-GPD +
             4 Golden Micro-Channels / 3 In-Situ CR NSBI-PL + 1,000x paired family bootstrap
             (`seed=20263250`) + 6 OOD & Shard Stress checks
  - HTCondor GPU job submission (`~/condor/Pinpoint/stage329_gen4_ssl/`), auto-selecting
    the least-loaded compatible schedd via `condor_status -schedd`.

Strictly blocks deprecated `v6_clean_v3` / `10081-10084` / `l2_ssl_300k` / `l1_cluster_sidecar`
and enforces `SealAuditGuard` zero-access on sealed `Cal-A / Cut-B / Confirm`.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import pickle
import re
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from analysis.flavour.gen4.conflict_free_optim import (
    DomainCGGMConfig,
    DomainCGGMProbes,
    DomainCGGMTracker,
    apply_conflict_free_backward,
    compute_mcr_synergy_loss,
    compute_modality_presence_masks,
    compute_probe_loss,
)
from analysis.flavour.gen4.gen4_unified_model import (
    GEN4_TRUTH_BATCH_KEYS,
    PinpointGen4Net,
    compute_gen4_supervised_loss,
    extract_nsbi_observables_from_output,
)
from analysis.flavour.gen4.geometry_contract import (
    Gen4GeometryContract,
    apply_inplace_nc_tau_mu_surgery,
    apply_random_valid_orbit,
    inverse_sat_additive_pe,
    load_geometry_contract,
)
from analysis.flavour.gen4.physics_frontend import (
    N_MAX_CLUSTERS_GEN4,
    N_MAX_PATCHES_GEN4,
    extract_gen4_frontend_event,
)
from analysis.flavour.gen4.run_gen4_supervised import (
    _annotate_eval_meta,
    _evaluate_predictions,
)
from analysis.flavour.gen4.nsbi_discovery import (
    DomainPOTGPDCalibrator,
    compare_pot_gpd_vs_empirical_split_half,
    evaluate_nsbi_vs_single_bin_closure,
    run_nsbi_toy_mc_closure,
)
from analysis.flavour.gen4.ssl_sonata_v2 import (
    PinpointSSLv2Model,
    SonataPrototypeHead,
    SonataSSLConfig,
    compute_sonata_v2_ssl_loss,
)
from analysis.flavour.gen4.upart13_tpauc import (
    FullDatasetScoreTable,
    Stage3TailConfig,
    compute_inferno_fisher_proxy,
    compute_nswc_tpauc_loss,
    compute_quantile_pivot_and_rng_loss,
    compute_upart13_lort_loss,
    configure_unfrozen_top_backbone,
)
from analysis.flavour.inventory import write_json
from analysis.flavour.nextgen.protocols import (
    check_branch_gradient_health,
    compute_student_t_pulls,
)
def _set_ieee_fp32(enable: bool = True) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = not enable
        torch.backends.cudnn.allow_tf32 = not enable
from analysis.flavour.nextgen.seal_guard import (
    SealAuditGuard,
    SealViolationError,
    assert_unsealed_path,
    assert_unsealed_split,
    verify_inference_truth_invariance,
)
from analysis.flavour.v2.partitions import FAMILY, KEY
from analysis.flavour.v2.submit import choose_schedd, gpu_requirement


GEN4_RESULTS_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage329_gen4')
GEN4_MC8588_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage329_gen4_mc8588')
GEN4_CACHE_DIR = GEN4_MC8588_ROOT / 'cache'
EXP_SSL_TAIL_DIR = GEN4_MC8588_ROOT / 'exp_gen4_ssl_tail'
EXP_DIFFPHYS_DIR = GEN4_MC8588_ROOT / 'exp_diffphys_sbi'
EXP_SUP_DIR = GEN4_MC8588_ROOT / 'exp_gen4_supervised'
CONDOR_SSL_DIR = Path.home() / 'condor/Pinpoint/stage329_gen4_ssl'
STAGE2_SEEDS: tuple[int, ...] = (17, 42, 2026)

STAGE1_MATRIX: dict[str, dict[str, Any]] = {
    'ssl_v2_ft_100pct': {'use_ssl': True, 'train_frac': 1.00, 'passes': 18, 'conflict_free': True},
    'scratch_100pct': {'use_ssl': False, 'train_frac': 1.00, 'passes': 18, 'conflict_free': True},
    'ssl_v2_ft_25pct': {'use_ssl': True, 'train_frac': 0.25, 'passes': 28, 'conflict_free': True},
    'scratch_25pct': {'use_ssl': False, 'train_frac': 0.25, 'passes': 28, 'conflict_free': True},
    'ssl_v2_ft_10pct': {'use_ssl': True, 'train_frac': 0.10, 'passes': 36, 'conflict_free': True},
    'scratch_10pct': {'use_ssl': False, 'train_frac': 0.10, 'passes': 36, 'conflict_free': True},
    'abl_no_conflict_free_adamw': {'use_ssl': True, 'train_frac': 1.00, 'passes': 18, 'conflict_free': False},
}

_STAGE1_FORBIDDEN_RE = re.compile(
    r'(?:^|/)shards/0(?:8\d|9\d)(?:/|$)|validation_meta|tune_screen|tune_truth_meta|tune8k|tune6k|tune_eval|stress_diag_pack|diag_truth_meta',
    re.IGNORECASE,
)
_STAGE1_HOOK_INSTALLED = False


def install_stage1_only_hook() -> None:
    """Block Tune/Eval filesystem syscalls strictly during SSL pretraining & Stage-I."""
    global _STAGE1_HOOK_INSTALLED
    if _STAGE1_HOOK_INSTALLED:
        return

    def _hook(event: str, args: tuple[Any, ...]) -> None:
        if event not in ('open', 'os.stat', 'os.scandir', 'os.listdir') or not args:
            return
        target = args[0]
        if isinstance(target, int):
            return
        try:
            p = str(target)
        except Exception:
            return
        if _STAGE1_FORBIDDEN_RE.search(p.replace('\\', '/')):
            raise SealViolationError(f'Stage-I forbids accessing Tune/Eval path: {p}')

    sys.addaudithook(_hook)
    _STAGE1_HOOK_INSTALLED = True


def assert_not_deprecated_mc(path: Path | str) -> None:
    """Enforce directive #44: forbid deprecated pre-10085-10088 MC/caches."""
    low = str(path).lower()
    for bad in ('v6_clean_v3', 'l2_ssl_300k', 'l1_cluster_sidecar'):
        if bad in low:
            raise ValueError(f'Deprecated pre-10085-10088 dataset/cache forbidden by directive #44: {path}')


def _update_ema_module(ema: nn.Module, online: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for pe, po in zip(ema.parameters(), online.parameters()):
            pe.data.mul_(decay).add_(po.data, alpha=1.0 - decay)
        for be, bo in zip(ema.buffers(), online.buffers()):
            be.copy_(bo)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


def _collate_cached(
    items: list[dict[str, Any]],
    geom: Gen4GeometryContract,
    device: torch.device,
    *,
    aug_cfg: dict[str, Any] | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    from analysis.flavour.gen4.data_cache import collate_gen4_batch
    return _move_batch(collate_gen4_batch(items, geom, aug_cfg=aug_cfg, rng=rng), device)


def _strip_truth_keys(batch: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in batch.items() if k not in GEN4_TRUTH_BATCH_KEYS}


# ---------------------------------------------------------------------------
# Gate-A / Gate-B / Integrated SSL + Conflict-Free + Stage-3 Health Check
# ---------------------------------------------------------------------------
def run_gate_ab_health(
    device_str: str = 'cpu',
    batch_size: int = 4,
    steps: int = 20,
) -> dict[str, Any]:
    """Verify Gate-A truth invariance, Gate-B mixed Si+Fortune gradients, SSL & Stage-3 tail."""
    from analysis.flavour.gen4.run_gen4_supervised import make_synthetic_gen4_batch

    dev = torch.device(device_str)
    _set_ieee_fp32(True)
    geom = load_geometry_contract()
    model = PinpointGen4Net(geom).to(dev)
    tea_bb = copy.deepcopy(model).eval()
    for p in tea_bb.parameters():
        p.requires_grad_(False)

    # Mixed batch: half silicon-active, half empty-silicon Fortune_ScintOnly
    b_si = make_synthetic_gen4_batch(batch_size=max(2, batch_size // 2), n_cl=64, has_edep=1.0, fortune_only=False, device=dev, seed=101)
    b_fo = make_synthetic_gen4_batch(batch_size=max(2, batch_size // 2), n_cl=0, has_edep=1.0, fortune_only=True, device=dev, seed=202)
    b_fo['pix_xyz_mm'] = torch.zeros((b_fo['c_obs'].shape[0], 64, 3), device=dev)
    b_fo['pix_plane'] = torch.full((b_fo['c_obs'].shape[0], 64), -1, dtype=torch.long, device=dev)
    b_fo['pix_valid'] = torch.zeros((b_fo['c_obs'].shape[0], 64), dtype=torch.bool, device=dev)
    b_fo['pix_feat'] = torch.zeros((b_fo['c_obs'].shape[0], 64, b_si['pix_feat'].shape[-1]), device=dev)
    b_fo['cl_origin_label'] = torch.full((b_fo['c_obs'].shape[0], 64), -1, dtype=torch.long, device=dev)
    b_fo['cl_secondary_origin_label'] = torch.full((b_fo['c_obs'].shape[0], 64), -1, dtype=torch.long, device=dev)
    b_fo['cl_dominant_edep_frac'] = torch.ones((b_fo['c_obs'].shape[0], 64), device=dev)
    b_fo['cl_dominant_track_id'] = torch.zeros((b_fo['c_obs'].shape[0], 64), dtype=torch.long, device=dev)

    b_mix = {
        k: (torch.cat([b_si[k], b_fo[k]], dim=0) if isinstance(b_si[k], torch.Tensor) else b_si[k])
        for k in b_si
    }

    # 1. Gate-A truth invariance
    gate_a = verify_inference_truth_invariance(model, b_mix, atol=0.0)

    # 2. SSL pretraining step check
    ssl_model = PinpointSSLv2Model(model).to(dev)
    tea_heads = {
        'micro': copy.deepcopy(ssl_model.proto_micro).eval(),
        'meso': copy.deepcopy(ssl_model.proto_meso).eval(),
        'macro': copy.deepcopy(ssl_model.proto_macro).eval(),
    }
    for h in tea_heads.values():
        for p in h.parameters():
            p.requires_grad_(False)

    ssl_opt = torch.optim.AdamW(ssl_model.parameters(), lr=3.0e-4)
    ssl_opt.zero_grad(set_to_none=True)
    l_ssl, ssl_m = compute_sonata_v2_ssl_loss(ssl_model, tea_bb, tea_heads, b_mix)
    l_ssl.backward()
    ssl_opt.step()

    # 3. Conflict-Free Stage-1 step check (Domain-CGGM + MMPareto + MCR)
    probes = DomainCGGMProbes(256, 4).to(dev)
    tracker = DomainCGGMTracker()
    model.probes = probes  # attach training-only probes
    cf_opt = torch.optim.AdamW(model.parameters(), lr=4.0e-4)
    losses_cf: list[float] = []
    for _ in range(steps):
        cf_opt.zero_grad(set_to_none=True)
        out = model.forward_NextGen(b_mix)
        tot, prim, aux_l, _ = compute_gen4_supervised_loss(model, out, b_mix)
        pm = compute_modality_presence_masks(b_mix)
        pl = probes({
            'si_vtx': out.aux['z_si_vtx'],
            'cal_pflow': out.aux['z_cal_pflow'],
            'spec_ipt': out.aux['z_spec_ipt'],
        })
        kap, _ = tracker.update_and_get_kappas(pl, b_mix['label'], pm)
        l_pr, _ = compute_probe_loss(pl, b_mix['label'], pm)
        l_mcr, _ = compute_mcr_synergy_loss(
            model.fusion,
            lambda zf: model.head(
                zf,
                torch.zeros((len(zf), 36), device=dev),
                torch.zeros((len(zf), 16), device=dev),
                torch.ones((len(zf),), dtype=torch.long, device=dev),
                torch.full((len(zf),), 350.0, device=dev),
            ),
            out.aux['z_si_vtx'],
            out.aux['z_cal_pflow'],
            out.aux['z_spec_ipt'],
            b_mix['label'],
            pm,
        )
        diag = apply_conflict_free_backward(
            model, prim, aux_l, presence_masks=pm, kappas=kap, probe_loss=l_pr, mcr_loss=l_mcr
        )
        # Verify zero non-finite grads
        for n, p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise FloatingPointError(f'Non-finite grad in {n} during conflict-free check')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        cf_opt.step()
        losses_cf.append(float(tot.detach().item()))

    # 4. Stage-3 unfrozen top-65% NSWC-TPAUC check
    delattr(model, 'probes')
    pgs, unfreeze_info = configure_unfrozen_top_backbone(model, top_frac=0.65)
    t3_opt = torch.optim.AdamW(pgs)
    t3_opt.zero_grad(set_to_none=True)
    out3 = model.forward_NextGen(b_mix)
    l_tp, _ = compute_nswc_tpauc_loss(
        out3.aux['tau_llr_score'],
        b_mix['label'] == 2,
        b_mix['obs_domain_id'],
        is_charm_bg=(b_mix['label'] != 2) & (b_mix['charm_flag'] > 0),
    )
    l_piv = compute_quantile_pivot_and_rng_loss(
        out3.aux['tau_llr_score'], out3.aux['tau_llr_score'] + 0.01, b_mix['label'] == 2, out3.aux['e_vis_hat_gev']
    )
    l_inf = compute_inferno_fisher_proxy(
        out3.aux['tau_llr_score'], b_mix['label'] == 2, (b_mix['label'] != 2) & (b_mix['charm_flag'] > 0)
    )
    (l_tp + l_piv + l_inf).backward()
    t3_opt.step()

    return {
        'gate_a': gate_a,
        'ssl_metrics': ssl_m,
        'cf_loss_start': losses_cf[0],
        'cf_loss_end': losses_cf[-1],
        'mmpareto_cos': diag['mmpareto']['cos_with_primary'],
        'stage3_unfreeze_info': unfreeze_info,
        'passed': bool(gate_a['passed'] and math.isfinite(losses_cf[-1])),
    }


# ---------------------------------------------------------------------------
# Stage 0: PinpointSSL-v2 Pretraining on `10085-10088` SSL Unlabeled Shards
# ---------------------------------------------------------------------------
def _locate_ssl_shards(cache_dir: Path) -> tuple[list[Path], Path | None]:
    ssl_dir = cache_dir / 'ssl_mc8588'
    ssl_files = sorted(ssl_dir.glob('ssl_train_part_*.pkl')) if ssl_dir.is_dir() else []
    if not ssl_files:
        ssl_files = sorted(cache_dir.glob('ssl*_part_*.pkl'))
    if not ssl_files:
        ssl_files = sorted(cache_dir.glob('ssl_unlabeled/*.pkl'))
    val_p = ssl_dir / 'ssl_val_calib.pkl'
    return ssl_files, (val_p if val_p.is_file() else None)


def run_ssl_pretrain(
    *,
    cache_dir: Path = GEN4_CACHE_DIR,
    out_dir: Path = EXP_SSL_TAIL_DIR / 'ssl_v2',
    epochs: int = 12,
    batch_size: int = 192,
    micro_bs: int = 64,
    lr: float = 4.0e-4,
    max_shards: int = 0,
    seed: int = 42,
    device_str: str = 'cuda:0',
) -> dict[str, Any]:
    """Pretrain >96% of PinpointGen4Net via Sonata-MS + SIGReg/KoLeo + 15 um Ray + HW-Inv."""
    import gc

    assert_not_deprecated_mc(cache_dir)
    assert_unsealed_path(cache_dir)
    install_stage1_only_hook()
    assert_unsealed_split('internal_train', stage=1)

    with SealAuditGuard() as guard:
        t0 = time.perf_counter()
        dev = torch.device(device_str if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(seed)
        np.random.seed(seed)
        geom = load_geometry_contract()

        ssl_files, ssl_val_path = _locate_ssl_shards(cache_dir)
        if not ssl_files:
            raise FileNotFoundError(
                f'No 10085-10088 SSL cache files found under {cache_dir}; '
                'awaiting worker-foundation 10085-10088 cache build.'
            )
        if max_shards > 0:
            ssl_files = ssl_files[:max_shards]
        for sf in ssl_files:
            assert_unsealed_path(sf)
        if ssl_val_path is not None:
            assert_unsealed_path(ssl_val_path)

        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = SonataSSLConfig()
        backbone = PinpointGen4Net(geom).to(dev)
        student = PinpointSSLv2Model(backbone, cfg).to(dev)
        teacher_bb = copy.deepcopy(backbone).eval()
        tea_heads = {
            'micro': copy.deepcopy(student.proto_micro).eval(),
            'meso': copy.deepcopy(student.proto_meso).eval(),
            'macro': copy.deepcopy(student.proto_macro).eval(),
        }
        for m in [teacher_bb, *tea_heads.values()]:
            for p in m.parameters():
                p.requires_grad_(False)

        opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.04, betas=(0.9, 0.95))
        grad_accum = max(1, math.ceil(batch_size / max(1, micro_bs)))
        history: list[dict[str, Any]] = []
        ckpt_path = out_dir / 'pinpoint_ssl_v2_backbone.pt'
        global_step = 0

        # Estimate n_total from the first shard to avoid double-loading 45+ GB before epoch 1
        with ssl_files[0].open('rb') as fh:
            first_items = pickle.load(fh)
        n_est = len(first_items) * len(ssl_files)
        del first_items
        gc.collect()

        total_steps = max(1, epochs * math.ceil(n_est / batch_size))
        warmup_steps = max(1, int(0.05 * total_steps))
        min_lr_ratio = 1.0e-5 / lr

        def _lr_lam(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * min(1.0, prog)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lam)
        gen_t = torch.Generator(device=dev).manual_seed(seed + 13)
        ssl_aug_cfg = {'p_cp_y': 0.5, 'p_px_si': 0.5}
        n_exact_epoch = 0

        val_items_cached: list[dict[str, Any]] | None = None
        if ssl_val_path is not None:
            with ssl_val_path.open('rb') as fh:
                val_items_cached = pickle.load(fh)[:4096]

        for ep in range(1, epochs + 1):
            te0 = time.perf_counter()
            student.train()
            ep_sums: dict[str, float] = {}
            ep_n = 0
            rng = np.random.default_rng(seed + ep)

            for sh_i, sf in enumerate(ssl_files[i] for i in rng.permutation(len(ssl_files))):
                with sf.open('rb') as fh:
                    items = pickle.load(fh)
                order = rng.permutation(len(items))
                opt.zero_grad(set_to_none=True)
                acc = 0
                for start in range(0, len(order), micro_bs):
                    idx = order[start : start + micro_bs]
                    batch = _collate_cached(
                        [items[int(i)] for i in idx], geom, dev, aug_cfg=ssl_aug_cfg, rng=rng
                    )
                    batch = _strip_truth_keys(batch)
                    loss, m_step = compute_sonata_v2_ssl_loss(
                        student, teacher_bb, tea_heads, batch, cfg=cfg, generator=gen_t
                    )
                    (loss / float(grad_accum)).backward()
                    bs_now = int(batch['c_obs'].shape[0])
                    ep_n += bs_now
                    for k, v in m_step.items():
                        ep_sums[k] = ep_sums.get(k, 0.0) + v * bs_now
                    acc += 1
                    if acc >= grad_accum:
                        global_step += 1
                        torch.nn.utils.clip_grad_norm_(student.parameters(), 2.0)
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                        sched.step()
                        prog = min(1.0, global_step / total_steps)
                        ema_m = 0.9995 - 0.5 * (0.9995 - 0.9940) * (1.0 + math.cos(math.pi * prog))
                        _update_ema_module(teacher_bb, backbone, ema_m)
                        _update_ema_module(tea_heads['micro'], student.proto_micro, ema_m)
                        _update_ema_module(tea_heads['meso'], student.proto_meso, ema_m)
                        _update_ema_module(tea_heads['macro'], student.proto_macro, ema_m)
                        acc = 0
                if acc > 0:
                    global_step += 1
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 2.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
                del items
                gc.collect()
                if (sh_i + 1) % 4 == 0 or (sh_i + 1) == len(ssl_files):
                    run_avg = ep_sums.get('ssl_total', 0.0) / max(1, ep_n)
                    print(
                        f"[SSL-v2] ep={ep:02d}/{epochs} shard={sh_i+1:02d}/{len(ssl_files)} "
                        f"n={ep_n} loss={run_avg:.4f} ({time.perf_counter()-te0:.1f}s)",
                        flush=True,
                    )
                if (sh_i + 1) % 8 == 0:
                    torch.save(
                        {
                            'backbone_state_dict': student.export_backbone_checkpoint_dict(),
                            'ssl_config': cfg.__dict__,
                            'epoch': ep,
                            'shard': sh_i + 1,
                            'history': history,
                        },
                        ckpt_path,
                    )

            n_exact_epoch = ep_n
            ep_avg = {k: v / max(1, ep_n) for k, v in ep_sums.items()}

            # Zero-label SSL calibration validation check
            if val_items_cached:
                student.eval()
                val_tot, val_n = 0.0, 0
                with torch.no_grad():
                    for vs in range(0, len(val_items_cached), micro_bs):
                        vb = _strip_truth_keys(_collate_cached(val_items_cached[vs : vs + micro_bs], geom, dev))
                        _, vm = compute_sonata_v2_ssl_loss(
                            student, teacher_bb, tea_heads, vb, cfg=cfg, generator=gen_t
                        )
                        vn = int(vb['c_obs'].shape[0])
                        val_tot += vm['ssl_total'] * vn
                        val_n += vn
                ep_avg['val_ssl_total'] = val_tot / max(1, val_n)

            row = {'epoch': ep, 'step': global_step, 'ep_sec': float(time.perf_counter() - te0), **ep_avg}
            history.append(row)
            torch.save(
                {
                    'backbone_state_dict': student.export_backbone_checkpoint_dict(),
                    'ssl_config': cfg.__dict__,
                    'epoch': ep,
                    'history': history,
                },
                ckpt_path,
            )
            print(
                f"[SSL-v2] ep={ep:02d}/{epochs} n={ep_n} loss={ep_avg.get('ssl_total', 0.0):.4f} "
                f"val={ep_avg.get('val_ssl_total', float('nan')):.4f} ({row['ep_sec']:.1f}s)",
                flush=True,
            )

        summary = {
            'UTC_finished': datetime.now(timezone.utc).isoformat(),
            'n_ssl_events': int(n_exact_epoch),
            'epochs': epochs,
            'wall_seconds': float(time.perf_counter() - t0),
            'ckpt_path': str(ckpt_path),
            'seal_violations': len(guard.violations),
            'history': history,
        }
        write_json(out_dir / 'ssl_pretrain_summary.json', summary)
        return summary


# ---------------------------------------------------------------------------
# 4-Stage Curriculum Fine-Tuning (Conflict-Free -> LORT Recal -> Stage-3 Tail)
# ---------------------------------------------------------------------------
def _empty_obs_for_surgery() -> dict[str, Any]:
    return {
        'pixel': {
            'channel_id': np.zeros((0,), dtype=np.uint64),
            'xyz_mm': np.zeros((0, 3), dtype=np.float32),
            'edep_mev': np.zeros((0,), dtype=np.float32),
        },
        'bar': {
            'layer': np.zeros((0,), dtype=np.int16),
            'panel': np.zeros((0,), dtype=np.int16),
            'horizontal': np.zeros((0,), dtype=np.int8),
            'bar': np.zeros((0,), dtype=np.int16),
            'channel_id': np.zeros((0,), dtype=np.int64),
            'bounds_mm': np.zeros((0, 4), dtype=np.float32),
            'pe_proxy': np.zeros((0,), dtype=np.int64),
            'hit_time_ns': np.zeros((0,), dtype=np.float32),
        },
        'tracker_hypothesis': {
            'station': np.zeros((0,), dtype=np.int8),
            'xy_mm': np.zeros((0, 2), dtype=np.float32),
            'sigma_xy_mm': np.zeros((0, 2), dtype=np.float32),
        },
    }


def _overlay_inplace_nc_surgery_on_cached(
    host: dict[str, Any],
    geom: Gen4GeometryContract,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Overlay polarized tau->mu from `apply_inplace_nc_tau_mu_surgery` onto a cached NC item."""
    pv = np.asarray(host.get('vertex_xyz_mm', [0.0, -12.0, -2950.0]), dtype=np.float64)
    if float(pv[2]) < float(geom.pixel_Zpos[0]) - 50.0 or float(pv[2]) > float(geom.station_z_mm[-1]):
        pv = np.array([0.0, -12.0, float(geom.pixel_Zpos[0]) - 8.5], dtype=np.float64)
    vk = np.asarray(host.get('vis_kinematics', np.zeros(8)), dtype=np.float64)
    p_ht = vk[:2] if len(vk) >= 2 and np.isfinite(vk[:2]).all() else np.array([1.2, -0.7], dtype=np.float64)
    q_mu = -1 if rng.random() < 0.75 else 1

    obs_mu, meta_s = apply_inplace_nc_tau_mu_surgery(_empty_obs_for_surgery(), pv, p_ht, geom, rng, q_mu=q_mu)
    fe_mu = extract_gen4_frontend_event(obs_mu, geom, has_subthr_readout=True, use_edep=True, return_cluster_map=False)

    out = copy.deepcopy(host)

    # 1. Pixel clusters & patches overlay (capped cleanly at N_MAX_CLUSTERS_GEN4 / N_MAX_PATCHES_GEN4)
    n_h = len(out['pix_plane'])
    n_m = len(fe_mu['pix_plane'])
    if n_m > 0:
        n_keep_m = min(n_m, max(0, N_MAX_CLUSTERS_GEN4 - n_h))
        if n_keep_m > 0:
            out['pix_xyz_mm'] = np.concatenate([out['pix_xyz_mm'], fe_mu['pix_xyz_mm'][:n_keep_m].astype(np.float32)], axis=0)
            out['pix_plane'] = np.concatenate([out['pix_plane'], fe_mu['pix_plane'][:n_keep_m].astype(np.int16)], axis=0)
            out['pix_feat'] = np.concatenate([out['pix_feat'], fe_mu['pix_feat'][:n_keep_m].astype(np.float32)], axis=0)
            if 'cl_origin_label' in out:
                out['cl_origin_label'] = np.concatenate([out['cl_origin_label'], np.full(n_keep_m, 1, dtype=np.int64)])
                out['cl_secondary_origin_label'] = np.concatenate([out['cl_secondary_origin_label'], np.full(n_keep_m, 1, dtype=np.int64)])
                out['cl_dominant_edep_frac'] = np.concatenate([out['cl_dominant_edep_frac'], np.ones(n_keep_m, dtype=np.float32)])
                out['cl_sv_group_id'] = np.concatenate([out['cl_sv_group_id'], np.zeros(n_keep_m, dtype=np.int64)])
                out['cl_dominant_track_id'] = np.concatenate([out['cl_dominant_track_id'], np.full(n_keep_m, 999, dtype=np.int64)])
            # Overlay top injected patches if budget permits
            n_hp = len(out['patch_cluster_idx'])
            n_mp = min(len(fe_mu['patch_cluster_idx']), max(0, N_MAX_PATCHES_GEN4 - n_hp))
            if n_mp > 0:
                pidx_m = fe_mu['patch_cluster_idx'][:n_mp].astype(np.int16)
                valid_p = (pidx_m >= 0) & (pidx_m < n_keep_m)
                if valid_p.any():
                    out['pix_patches_u8'] = np.concatenate([out['pix_patches_u8'], fe_mu['pix_patches_u8'][:n_mp][valid_p]], axis=0)
                    out['patch_cluster_idx'] = np.concatenate([out['patch_cluster_idx'], pidx_m[valid_p] + n_h], axis=0)
        out['pix_plane_summary'] = (out['pix_plane_summary'] + fe_mu['pix_plane_summary']).astype(np.float32)

    # 2. Scintillator bars overlay using exact 14,400-cell SiPM inverse saturation
    b_map: dict[tuple[int, int, int], int] = {
        (int(s), int(v), int(b)): idx
        for idx, (s, v, b) in enumerate(zip(out['bar_station'], out['bar_view'], out['bar_idx']))
    }
    bst = list(np.asarray(out['bar_station'], dtype=np.int16).tolist())
    bvw = list(np.asarray(out['bar_view'], dtype=np.int8).tolist())
    bix = list(np.asarray(out['bar_idx'], dtype=np.int16).tolist())
    bf10 = list(np.asarray(out['bar_feat_10'], dtype=np.float32))
    for s_m, v_m, b_m, f_m in zip(fe_mu['bar_station'], fe_mu['bar_view'], fe_mu['bar_idx'], fe_mu['bar_feat_10']):
        key = (int(s_m), int(v_m), int(b_m))
        if key in b_map:
            row = bf10[b_map[key]].copy()
            pe_a = float(max(0.0, row[2]))
            pe_b = float(max(0.0, float(f_m[2])))
            pe_s = float(inverse_sat_additive_pe(pe_a, pe_b))
            row[0] = np.float32(max(0.0, float(row[0])) + max(0.0, float(f_m[0])))
            row[2] = np.float32(pe_s)
            if f_m[5] > 0.5 and (row[5] <= 0.5 or f_m[3] < row[3]):
                row[3] = f_m[3]
                row[4] = f_m[4]
                row[5] = np.float32(1.0)
            row[6] = np.float32(1.0 if pe_s >= 14350.0 else 0.0)
            row[7] = np.float32(1.0 if (0.0 < pe_s < 2.5) else 0.0)
            row[9] = np.float32(max(float(row[9]), float(f_m[9])))
            bf10[b_map[key]] = row
        else:
            b_map[key] = len(bst)
            bst.append(int(s_m))
            bvw.append(int(v_m))
            bix.append(int(b_m))
            bf10.append(f_m.astype(np.float32))
    out['bar_station'] = np.asarray(bst, dtype=np.int16)
    out['bar_view'] = np.asarray(bvw, dtype=np.int8)
    out['bar_idx'] = np.asarray(bix, dtype=np.int16)
    out['bar_feat_10'] = np.stack(bf10, axis=0).astype(np.float32) if bf10 else np.zeros((0, 10), dtype=np.float32)

    # 3. Tracker & Kalman overlay (daughter muon takes top bore candidate when present)
    if bool(fe_mu['trk_cand_mask'].any()):
        host_tc = out['trk_cand']
        host_tm = out['trk_cand_mask']
        inj_tc = fe_mu['trk_cand'][fe_mu['trk_cand_mask']]
        avail = np.flatnonzero(~host_tm)
        for j, slot in enumerate(avail[: len(inj_tc)]):
            host_tc[slot] = inj_tc[j]
            host_tm[slot] = True
        out['trk_station_counts'] = np.maximum(out['trk_station_counts'], fe_mu['trk_station_counts']).astype(np.float32)
    if bool(fe_mu['muon_kalman_mask'].any()):
        host_mk = out['muon_kalman_tracks']
        host_mm = out['muon_kalman_mask']
        inj_mk = fe_mu['muon_kalman_tracks'][fe_mu['muon_kalman_mask']]
        avail_m = np.flatnonzero(~host_mm)
        for j, slot in enumerate(avail_m[: len(inj_mk)]):
            host_mk[slot] = inj_mk[j]
            host_mm[slot] = True

    c_obs = np.asarray(out['c_obs'], dtype=np.float32).copy()
    c_obs[0] = float(len(out['pix_plane']) > 0)
    c_obs[1] = float(len(out['bar_station']) > 0)
    c_obs[2] = float(bool(out['trk_cand_mask'].any()))
    c_obs[4] = float(min(1.0, c_obs[4] + float(fe_mu['c_obs'][4])))
    out['c_obs'] = c_obs

    # 4. Update tau->mu supervised targets & sidecar metadata
    out['label'] = 2
    out['label_7'] = 4  # tau_mu
    out['tau_branch_valid'] = 1
    out['true_tau_submode'] = 1  # tau_mu
    out['charm_flag'] = 0
    out['charm_valid'] = 1
    out['vertex_xyz_mm'] = np.asarray(meta_s['vertex_xyz_mm'], dtype=np.float32)
    out['vertex_valid'] = 1
    out['true_sv_xyz_mm'] = np.asarray(meta_s['true_sv_xyz_mm'], dtype=np.float32)
    out['has_true_sv'] = True
    out['tau_decay_L3d_mm'] = float(meta_s['tau_decay_L3d_mm'])
    out['sample_weight'] = 0.85
    return out


def _generate_inplace_nc_surgery_items(
    train_items: list[dict[str, Any]],
    geom: Gen4GeometryContract,
    rng: np.random.Generator,
    *,
    surgery_frac: float = 0.08,
) -> list[dict[str, Any]]:
    """Create `surgery_frac * N` in-place NC + polarized (tau->mu) short-L_tau training items."""
    if surgery_frac <= 0.0:
        return []
    nc_pool = [it for it in train_items if int(it.get('label', -1)) == 3 and int(it.get('obs_domain_id', 0)) in (1, 2)]
    if not nc_pool:
        return []
    n_surg = max(1, int(round(surgery_frac * len(train_items))))
    surg_items: list[dict[str, Any]] = []
    for _ in range(n_surg):
        host = nc_pool[int(rng.integers(0, len(nc_pool)))]
        surg_items.append(_overlay_inplace_nc_surgery_on_cached(host, geom, rng))
    return surg_items


def _load_ssl_backbone_into_gen4(model: PinpointGen4Net, ssl_ckpt_path: Path, device: torch.device) -> dict[str, Any]:
    assert_unsealed_path(ssl_ckpt_path)
    ckpt = torch.load(ssl_ckpt_path, map_location=device, weights_only=False)
    bb_sd = ckpt['backbone_state_dict'] if 'backbone_state_dict' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(bb_sd, strict=False)
    bad_missing = [k for k in missing if not k.startswith('head.')]
    return {
        'loaded_keys': len(bb_sd),
        'missing_head_keys': len(missing) - len(bad_missing),
        'missing_non_head_keys': bad_missing,
        'unexpected_keys': list(unexpected),
    }


def _run_4stage_curriculum(
    train_items: list[dict[str, Any]],
    eval_items: list[dict[str, Any]],
    *,
    geom: Gen4GeometryContract,
    device: torch.device,
    out_dir: Path,
    seed: int = 42,
    ssl_ckpt_path: Path | None = None,
    passes_stage1: int = 12,
    passes_stage2: int = 2,
    passes_stage3: int = 4,
    batch_size: int = 128,
    micro_bs: int = 64,
    conflict_free: bool = True,
    surgery_frac: float = 0.08,
) -> dict[str, Any]:
    """Execute Stage 1 (Conflict-Free + Orbits + NC+mu Surgery) -> Stage 2 (LORT) -> Stage 3 (Unfrozen NSWC-TPAUC)."""
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    model = PinpointGen4Net(geom).to(device)
    ssl_load_info = None
    if ssl_ckpt_path is not None and ssl_ckpt_path.is_file():
        ssl_load_info = _load_ssl_backbone_into_gen4(model, ssl_ckpt_path, device)

    surg_items = _generate_inplace_nc_surgery_items(train_items, geom, rng, surgery_frac=surgery_frac)
    tr_stage1 = train_items + surg_items
    train_aug_cfg = {'p_cp_y': 0.5, 'p_px_si': 0.5, 'p_subthr_drop': 0.15}

    probes = DomainCGGMProbes(256, 4).to(device)
    tracker = DomainCGGMTracker(DomainCGGMConfig())
    model.probes = probes

    # Stage 1: Conflict-Free Main Training (passes_stage1)
    opt1 = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=0.02, betas=(0.9, 0.98))
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=max(1, passes_stage1))
    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    step1 = 0
    for ep in range(1, passes_stage1 + 1):
        model.train()
        order = rng.permutation(len(tr_stage1))
        ep_loss = 0.0
        n_ep = 0
        for start in range(0, len(order), micro_bs):
            idx = order[start : start + micro_bs]
            batch = _collate_cached([tr_stage1[int(i)] for i in idx], geom, device, aug_cfg=train_aug_cfg, rng=rng)
            opt1.zero_grad(set_to_none=True)
            out = model.forward_NextGen(batch)
            tot, prim, aux_l, _ = compute_gen4_supervised_loss(model, out, batch, delta_lort=0.08)
            if conflict_free:
                pm = compute_modality_presence_masks(batch)
                pl = probes({
                    'si_vtx': out.aux['z_si_vtx'],
                    'cal_pflow': out.aux['z_cal_pflow'],
                    'spec_ipt': out.aux['z_spec_ipt'],
                })
                kap, _ = tracker.update_and_get_kappas(pl, batch['label'], pm)
                l_pr, _ = compute_probe_loss(pl, batch['label'], pm)
                l_mcr, _ = compute_mcr_synergy_loss(
                    model.fusion,
                    lambda zf: model.head(
                        zf,
                        torch.zeros((len(zf), 36), device=device),
                        torch.zeros((len(zf), 16), device=device),
                        torch.ones((len(zf),), dtype=torch.long, device=device),
                        torch.full((len(zf),), 350.0, device=device),
                    ),
                    out.aux['z_si_vtx'],
                    out.aux['z_cal_pflow'],
                    out.aux['z_spec_ipt'],
                    batch['label'],
                    pm,
                )
                apply_conflict_free_backward(
                    model, prim, aux_l, presence_masks=pm, kappas=kap, probe_loss=l_pr, mcr_loss=l_mcr
                )
            else:
                tot.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt1.step()
            step1 += 1
            ep_loss += float(tot.detach().item())
            n_ep += 1
            _update_ema_module(ema_model, model, min(0.9975, (1.0 + step1) / (10.0 + step1)))
        sched1.step()
        print(f'[Gen4-S1 seed={seed} cf={conflict_free}] ep={ep:02d}/{passes_stage1} loss={ep_loss/max(1,n_ep):.4f}', flush=True)

    # Remove training-only probes & free Stage-1 GPU state before Stage 2/3
    if hasattr(model, 'probes'):
        delattr(model, 'probes')
    if hasattr(ema_model, 'probes'):
        delattr(ema_model, 'probes')
    del probes, tracker, opt1, sched1, ema_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Stage 2: Frozen-backbone LORT head recalibration (delta_LORT=0.05) + anchor teacher logits13_ref
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith('head.'))
    opt2 = torch.optim.AdamW(model.head.parameters(), lr=1.0e-4, weight_decay=1.0e-3)
    for ep2 in range(1, passes_stage2 + 1):
        model.train()
        order = rng.permutation(len(train_items))
        for start in range(0, len(order), micro_bs):
            idx = order[start : start + micro_bs]
            batch = _collate_cached([train_items[int(i)] for i in idx], geom, device)
            opt2.zero_grad(set_to_none=True)
            out = model.forward_NextGen(batch)
            l_ce2, _ = compute_upart13_lort_loss(out.aux['logits13'], batch, delta_lort=0.05)
            l_ce2.backward()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), 2.0)
            opt2.step()

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'ssl_load_info': ssl_load_info}, out_dir / 'stage12_ckpt.pt')

    teacher_stage2 = copy.deepcopy(model).eval()
    for p in teacher_stage2.parameters():
        p.requires_grad_(False)

    # Populate 256 KB full-dataset score table on Train items
    n_tr = len(train_items)
    y4_tr = np.array([int(it['label']) for it in train_items], dtype=np.int64)
    dom_tr = np.array([int(it.get('obs_domain_id', 1)) for it in train_items], dtype=np.int64)
    cf_tr = np.array([int(it.get('charm_flag', 0)) for it in train_items], dtype=np.int64)
    l7_tr = np.array([int(it.get('label_7', -1)) for it in train_items], dtype=np.int64)
    ltau_tr = np.array([float(it.get('tau_decay_L3d_mm', -1.0)) for it in train_items], dtype=np.float32)

    score_tbl = FullDatasetScoreTable(
        n_tr, y4_tr, dom_tr, charm_flag=cf_tr, label_7=l7_tr, tau_decay_L3d_mm=ltau_tr
    )
    ref_logits13_all = np.zeros((n_tr, 13), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n_tr, micro_bs):
            end = min(n_tr, start + micro_bs)
            batch = _collate_cached(train_items[start:end], geom, device)
            out_r = teacher_stage2.forward_NextGen(_strip_truth_keys(batch))
            ref_logits13_all[start:end] = out_r.aux['logits13'].cpu().numpy()
            score_tbl.update(np.arange(start, end), out_r.aux['tau_llr_score'])

    del teacher_stage2, opt2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Stage 3: Unfrozen Top-65% Backbone Live Hard-Negative NSWC-TPAUC + Quantile-Pivot + INFERNO
    t3_cfg = Stage3TailConfig(passes=passes_stage3)
    pgs, unfreeze_info = configure_unfrozen_top_backbone(
        model, top_frac=t3_cfg.unfreeze_top_frac, lr_head=t3_cfg.lr_head, weight_decay=t3_cfg.weight_decay
    )
    opt3 = torch.optim.AdamW(pgs)
    t3_full_bs = t3_cfg.n_random + t3_cfg.n_hard_neg + t3_cfg.n_boundary_pos
    t3_bs = min(t3_full_bs, max(16, micro_bs))
    scale3 = float(t3_bs) / float(t3_full_bs)
    n_r3 = max(2, int(round(t3_cfg.n_random * scale3)))
    n_hn3 = max(4, int(round(t3_cfg.n_hard_neg * scale3)))
    n_bp3 = max(2, t3_bs - n_r3 - n_hn3)
    grad_accum3 = max(1, t3_full_bs // max(1, (n_r3 + n_hn3 + n_bp3)))
    steps_per_pass = max(1, math.ceil(n_tr / max(1, n_r3 + n_hn3 + n_bp3)))

    for ep3 in range(1, t3_cfg.passes + 1):
        model.train()
        ep3_loss = 0.0
        opt3.zero_grad(set_to_none=True)
        for st_i in range(steps_per_pass):
            b_idx = score_tbl.sample_hard_batch_indices(
                rng, n_random=n_r3, n_hard_neg=n_hn3, n_boundary_pos=n_bp3,
            )
            batch = _collate_cached([train_items[int(i)] for i in b_idx], geom, device, aug_cfg=train_aug_cfg, rng=rng)
            lg13_ref = torch.from_numpy(ref_logits13_all[b_idx]).to(device=device)

            out3 = model.forward_NextGen(batch)
            tau_llr = out3.aux['tau_llr_score']
            score_tbl.update(b_idx, tau_llr)

            is_tau = batch['label'] == 2
            is_ch_bg = (~is_tau) & (batch.get('charm_flag', torch.zeros_like(batch['label'])) > 0)
            l_tp, _ = compute_nswc_tpauc_loss(
                tau_llr, is_tau, batch['obs_domain_id'], score_tbl,
                fpr_target=t3_cfg.fpr_target, margin=t3_cfg.margin, is_charm_bg=is_ch_bg,
            )
            l_nll, _ = compute_upart13_lort_loss(out3.aux['logits13'], batch, delta_lort=0.05)
            l_kl = F.kl_div(
                F.log_softmax(out3.aux['logits13'].float(), dim=-1),
                F.softmax(lg13_ref.float(), dim=-1),
                reduction='batchmean',
            )
            batch_pert = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            if 'scint_grid' in batch_pert:
                is_del = batch_pert['scint_grid'][..., 9] > 0.5
                jit = 1.0 + 0.15 * (2.0 * torch.rand((batch_pert['scint_grid'].shape[0], 1, 1, 1), device=device) - 1.0)
                batch_pert['scint_grid'][..., 0] = torch.where(is_del, batch_pert['scint_grid'][..., 0] * jit, batch_pert['scint_grid'][..., 0])
            out3_p = model.forward_NextGen(_strip_truth_keys(batch_pert))
            l_piv = compute_quantile_pivot_and_rng_loss(
                tau_llr, out3_p.aux['tau_llr_score'], is_tau, out3.aux['e_vis_hat_gev']
            )
            l_inf = compute_inferno_fisher_proxy(tau_llr, is_tau, is_ch_bg)

            loss3 = (
                t3_cfg.lambda_tpauc * l_tp
                + t3_cfg.lambda_nll * l_nll
                + t3_cfg.lambda_kl * l_kl
                + t3_cfg.lambda_pivot * l_piv
                + t3_cfg.lambda_inferno * l_inf
            )
            (loss3 / float(grad_accum3)).backward()
            ep3_loss += float(loss3.detach().item())
            if (st_i + 1) % grad_accum3 == 0 or (st_i + 1) == steps_per_pass:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt3.step()
                opt3.zero_grad(set_to_none=True)
        print(f'[Gen4-S3 seed={seed}] ep={ep3:02d}/{t3_cfg.passes} tail_loss={ep3_loss/steps_per_pass:.4f}', flush=True)

    # Pure-observable IEEE-FP32 Evaluation
    _set_ieee_fp32(True)
    model.eval()
    probs_l, llr_l, vpv_l, spv_l = [], [], [], []
    nsbi_acc: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for start in range(0, len(eval_items), micro_bs):
            chunk = eval_items[start : start + micro_bs]
            b_obs = _strip_truth_keys(_collate_cached(chunk, geom, device))
            out_e = model.forward_NextGen(b_obs)
            probs_l.append(out_e.probs4.cpu().numpy())
            llr_l.append(out_e.aux['tau_llr_score'].cpu().numpy())
            vpv_l.append(out_e.v_pv_mm.cpu().numpy())
            spv_l.append(out_e.scale_pv_mm.cpu().numpy())
            nb = extract_nsbi_observables_from_output(b_obs, out_e)
            for k, v in nb.items():
                nsbi_acc.setdefault(k, []).append(v)

    probs_ev = np.concatenate(probs_l, axis=0)
    llr_ev = np.concatenate(llr_l, axis=0)
    vpv_ev = np.concatenate(vpv_l, axis=0)
    spv_ev = np.concatenate(spv_l, axis=0)
    nsbi_ev = {k: np.concatenate(v, axis=0) for k, v in nsbi_acc.items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'ssl_load_info': ssl_load_info}, out_dir / 'checkpoint.pt')
    return {
        'model': model,
        'probs4': probs_ev,
        'tau_llr': llr_ev,
        'v_pv_mm': vpv_ev,
        'scale_pv_mm': spv_ev,
        'nsbi_obs': nsbi_ev,
        'ssl_load_info': ssl_load_info,
        'stage3_unfreeze_info': unfreeze_info,
        'score_table_bytes': score_tbl.nbytes,
        'n_surgery_injected': len(surg_items),
        'wall_seconds': float(time.perf_counter() - t0),
    }


def _load_train32k_items(cache_dir: Path) -> list[dict[str, Any]]:
    tr_files = sorted(cache_dir.glob('train*_part_*.pkl'))
    if not tr_files:
        raise FileNotFoundError(f'No train*_part_*.pkl found in {cache_dir}')
    items: list[dict[str, Any]] = []
    for p in tr_files:
        assert_unsealed_path(p)
        with p.open('rb') as fh:
            items.extend(pickle.load(fh))
    return items


def _stratified_subsample(items: list[dict[str, Any]], frac: float, seed: int) -> list[dict[str, Any]]:
    if frac >= 0.9999:
        return list(items)
    rng = np.random.default_rng(seed)
    picked: list[dict[str, Any]] = []
    for c in (0, 1, 2, 3):
        grp = [it for it in items if int(it['label']) == c]
        n_k = max(1, int(round(frac * len(grp))))
        idx = rng.choice(len(grp), size=min(n_k, len(grp)), replace=False)
        picked.extend(grp[int(i)] for i in sorted(idx))
    picked.sort(key=lambda x: (x['run'], x['chunk'], x['evt_id']))
    return picked


def _build_eval_meta_from_items(items: list[dict[str, Any]], geom: Gen4GeometryContract) -> pd.DataFrame:
    raw = pd.DataFrame([
        {
            'run': int(it['run']),
            'chunk': int(it['chunk']),
            'evt_id': int(it['evt_id']),
            'obs_domain_id': int(it.get('obs_domain_id', 0)),
            'label': int(it['label']),
            'label_7': int(it.get('label_7', -1)),
            'charm_flag': int(it.get('charm_flag', 0)),
            'vertex_xyz_mm': it.get('vertex_xyz_mm', [0.0, 0.0, 0.0]),
            'true_tau_submode': int(it.get('true_tau_submode', -1)),
            'tgt_Z': int(it.get('tgt_Z', 74)),
            'tau_decay_L3d_mm': float(it.get('tau_decay_L3d_mm', 0.0)),
        }
        for it in items
    ])
    return _annotate_eval_meta(raw, geom)


def run_stage1_train(
    *,
    cache_dir: Path = GEN4_CACHE_DIR,
    out_dir: Path = EXP_SSL_TAIL_DIR,
    ssl_ckpt_path: Path = EXP_SSL_TAIL_DIR / 'ssl_v2/pinpoint_ssl_v2_backbone.pt',
    variant: str = 'ssl_v2_ft_100pct',
    seed: int = 42,
    micro_bs: int = 64,
    passes_stage1_override: int | None = None,
    device_str: str = 'cuda:0',
) -> dict[str, Any]:
    """Run Stage-I label-efficiency (10% / 25% / 100%) or conflict-free optimization ablation."""
    if variant not in STAGE1_MATRIX:
        raise ValueError(f'Unknown Stage-I variant {variant!r}; expected one of {list(STAGE1_MATRIX)}')
    spec = STAGE1_MATRIX[variant]
    assert_not_deprecated_mc(cache_dir)
    assert_unsealed_path(cache_dir)
    install_stage1_only_hook()
    assert_unsealed_split('internal_train', stage=1)

    with SealAuditGuard() as guard:
        import gc

        dev = torch.device(device_str if torch.cuda.is_available() else 'cpu')
        geom = load_geometry_contract()
        all_tr = _load_train32k_items(cache_dir)
        tr_full = [it for it in all_tr if it.get('internal_split') == 'internal_train']
        ev_sub = [it for it in all_tr if it.get('internal_split') == 'internal_val_eval']
        if not tr_full or not ev_sub:
            n_sp = int(0.8 * len(all_tr))
            tr_full, ev_sub = all_tr[:n_sp], all_tr[n_sp:]
        del all_tr
        tr_sub = _stratified_subsample(tr_full, float(spec['train_frac']), seed + 77)
        del tr_full
        gc.collect()
        ckpt_to_load = ssl_ckpt_path if bool(spec['use_ssl']) else None
        if bool(spec['use_ssl']) and not ssl_ckpt_path.is_file():
            raise FileNotFoundError(f'SSL checkpoint required for {variant}: {ssl_ckpt_path}')

        p1 = passes_stage1_override if passes_stage1_override is not None else min(18, int(spec['passes']))
        ab_dir = out_dir / f'stage1/{variant}'
        res = _run_4stage_curriculum(
            tr_sub, ev_sub,
            geom=geom, device=dev, out_dir=ab_dir, seed=seed,
            ssl_ckpt_path=ckpt_to_load,
            passes_stage1=p1, passes_stage2=2, passes_stage3=3,
            micro_bs=micro_bs, conflict_free=bool(spec['conflict_free']),
        )
        ev_meta = _build_eval_meta_from_items(ev_sub, geom)
        y_ev = ev_meta['label'].to_numpy(dtype=np.int64)
        metrics = _evaluate_predictions(res['probs4'], res['tau_llr'], y_ev, ev_meta, v_pv_mm=res['v_pv_mm'])
        metrics.update({
            'variant': variant,
            'use_ssl': bool(spec['use_ssl']),
            'train_frac': float(spec['train_frac']),
            'n_train_used': len(tr_sub),
            'conflict_free': bool(spec['conflict_free']),
            'ssl_load_info': res['ssl_load_info'],
            'stage3_unfreeze_info': res['stage3_unfreeze_info'],
            'seal_violations': len(guard.violations),
            'wall_seconds': res['wall_seconds'],
        })
        write_json(ab_dir / 'metrics.json', metrics)
        return metrics


def run_stage1_aggregate(*, out_dir: Path = EXP_SSL_TAIL_DIR) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for var in STAGE1_MATRIX:
        mp = out_dir / f'stage1/{var}/metrics.json'
        if mp.is_file():
            rows[var] = json.loads(mp.read_text())
    deltas: dict[str, Any] = {}
    for frac_tag in ('100pct', '25pct', '10pct'):
        k_ft, k_sc = f'ssl_v2_ft_{frac_tag}', f'scratch_{frac_tag}'
        if k_ft in rows and k_sc in rows:
            deltas[f'ssl_gain_{frac_tag}'] = {
                'delta_BA4_pp': float(rows[k_ft]['BA4_pct'] - rows[k_sc]['BA4_pct']),
                'delta_tau_1pct_fpr_pp': float(rows[k_ft]['tau_1pct_fpr_llr_pct'] - rows[k_sc]['tau_1pct_fpr_llr_pct']),
                'delta_NLL': float(rows[k_ft]['NLL'] - rows[k_sc]['NLL']),
            }
    if 'ssl_v2_ft_100pct' in rows and 'abl_no_conflict_free_adamw' in rows:
        a, b = rows['ssl_v2_ft_100pct'], rows['abl_no_conflict_free_adamw']
        deltas['conflict_free_gain'] = {
            'delta_BA4_pp': float(a['BA4_pct'] - b['BA4_pct']),
            'delta_tau_1pct_fpr_pp': float(a['tau_1pct_fpr_llr_pct'] - b['tau_1pct_fpr_llr_pct']),
        }
    out = {'variants': rows, 'deltas': deltas}
    write_json(out_dir / 'stage1/stage1_summary.json', out)
    return out


def _align_tune8k_meta(tune_items: list[dict[str, Any]], meta_pq: Path, geom: Gen4GeometryContract) -> pd.DataFrame:
    meta_full = pd.read_parquet(meta_pq)
    kmap = {
        (int(r), int(c), int(e)): idx
        for idx, (r, c, e) in enumerate(zip(meta_full['run'], meta_full['chunk'], meta_full['evt_id']))
    }
    ev_idx = [kmap[(int(it['run']), int(it['chunk']), int(it['evt_id']))] for it in tune_items]
    return _annotate_eval_meta(meta_full.iloc[ev_idx].reset_index(drop=True), geom)


def run_stage2_train(
    *,
    cache_dir: Path = GEN4_CACHE_DIR,
    out_dir: Path = EXP_SSL_TAIL_DIR,
    ssl_ckpt_path: Path = EXP_SSL_TAIL_DIR / 'ssl_v2/pinpoint_ssl_v2_backbone.pt',
    seed: int = 42,
    micro_bs: int = 64,
    passes_stage1: int = 12,
    device_str: str = 'cuda:0',
) -> dict[str, Any]:
    """Run full 4-Stage curriculum for one seed on Train32k & pure-obs evaluation on Tune8k + Stress400 + Diag_OOD."""
    assert_not_deprecated_mc(cache_dir)
    assert_unsealed_path(cache_dir)
    assert_unsealed_split('train', stage=2)
    assert_unsealed_split('tune6k', stage=2)

    if not ssl_ckpt_path.is_file():
        raise FileNotFoundError(f'Missing SSL checkpoint: {ssl_ckpt_path}')

    with SealAuditGuard() as guard:
        dev = torch.device(device_str if torch.cuda.is_available() else 'cpu')
        geom = load_geometry_contract()
        train_items = _load_train32k_items(cache_dir)

        tune_files = sorted(cache_dir.glob('tune*_part_*.pkl'))
        if not tune_files:
            raise FileNotFoundError(f'No tune*_part_*.pkl found in {cache_dir}')
        tune_all: list[dict[str, Any]] = []
        for tf in tune_files:
            assert_unsealed_path(tf)
            with tf.open('rb') as fh:
                tune_all.extend(pickle.load(fh))

        seed_dir = out_dir / f'stage2/seed_{seed}'
        res = _run_4stage_curriculum(
            train_items, tune_all,
            geom=geom, device=dev, out_dir=seed_dir, seed=seed,
            ssl_ckpt_path=ssl_ckpt_path,
            passes_stage1=passes_stage1, passes_stage2=2, passes_stage3=4,
            micro_bs=micro_bs, conflict_free=True,
        )
        model = res['model']

        # Save pure-observable full 8,000-row Tune8k (Tune_Screen 2k + Tune6k 6k) outputs
        np.save(seed_dir / f'probs4_tune8k_seed{seed}.npy', res['probs4'])
        np.save(seed_dir / f'tau_llr_tune8k_seed{seed}.npy', res['tau_llr'])
        np.save(seed_dir / f'v_pv_tune8k_seed{seed}.npy', res['v_pv_mm'])
        np.save(seed_dir / f'scale_pv_tune8k_seed{seed}.npy', res['scale_pv_mm'])
        np.savez(seed_dir / f'nsbi_obs_tune8k_seed{seed}.npz', **res['nsbi_obs'])

        meta_pq = cache_dir.parent / 'validation_meta/tune_truth_meta.parquet'
        ev_meta = _align_tune8k_meta(tune_all, meta_pq, geom)
        in_scr = np.array([bool(it.get('in_screen', False)) for it in tune_all], dtype=bool)
        t6_m = ~in_scr
        y_all = ev_meta['label'].to_numpy(dtype=np.int64)
        metrics = _evaluate_predictions(
            res['probs4'][t6_m], res['tau_llr'][t6_m], y_all[t6_m],
            ev_meta.loc[t6_m].reset_index(drop=True),
            v_pv_mm=res['v_pv_mm'][t6_m],
        )

        # Evaluate Stress400 & Diag_OOD
        stress_pack_p = cache_dir / 'stress_diag_pack.pkl'
        diag_meta_p = cache_dir.parent / 'validation_meta/diag_truth_meta.pkl'
        if stress_pack_p.is_file() and diag_meta_p.is_file():
            with stress_pack_p.open('rb') as fh:
                spack = pickle.load(fh)
            with diag_meta_p.open('rb') as fh:
                dmeta = pickle.load(fh)
            stress_y = np.asarray(dmeta['stress_y'], dtype=np.int64)
            os_npz: dict[str, np.ndarray] = {}
            stress_res: dict[str, dict[str, float]] = {}
            model.eval()
            for st_name, st_items in spack.get('stress_items', {}).items():
                sp_l, sl_l = [], []
                with torch.no_grad():
                    for si in range(0, len(st_items), micro_bs):
                        bs = _strip_truth_keys(_collate_cached(st_items[si : si + micro_bs], geom, dev))
                        oo = model.forward_NextGen(bs)
                        sp_l.append(oo.probs4.cpu().numpy())
                        sl_l.append(oo.aux['tau_llr_score'].cpu().numpy())
                sp4 = np.concatenate(sp_l, axis=0)
                sllr = np.concatenate(sl_l, axis=0)
                os_npz[f'stress_{st_name}_probs4'] = sp4
                os_npz[f'stress_{st_name}_llr'] = sllr
                sm = _evaluate_predictions(sp4, sllr, stress_y, pd.DataFrame())
                stress_res[st_name] = {
                    'BA4_pct': sm['BA4_pct'],
                    'tau_1pct_fpr_llr_pct': sm['tau_1pct_fpr_llr_pct'],
                }
            metrics['stress400'] = stress_res

            diag_res: dict[str, Any] = {}
            for dk, d_items in spack.get('diag_packs', {}).items():
                if not d_items or dk not in dmeta.get('diag_meta', {}):
                    continue
                dp_l, dl_l, dv_l = [], [], []
                with torch.no_grad():
                    for di in range(0, len(d_items), micro_bs):
                        bd = _strip_truth_keys(_collate_cached(d_items[di : di + micro_bs], geom, dev))
                        od = model.forward_NextGen(bd)
                        dp_l.append(od.probs4.cpu().numpy())
                        dl_l.append(od.aux['tau_llr_score'].cpu().numpy())
                        dv_l.append(od.v_pv_mm.cpu().numpy())
                dp4 = np.concatenate(dp_l, axis=0)
                dllr = np.concatenate(dl_l, axis=0)
                os_npz[f'diag_{dk}_probs4'] = dp4
                os_npz[f'diag_{dk}_llr'] = dllr
                dm_df = _annotate_eval_meta(dmeta['diag_meta'][dk], geom)
                diag_res[dk] = _evaluate_predictions(
                    dp4, dllr, dm_df['label'].to_numpy(dtype=np.int64), dm_df, v_pv_mm=np.concatenate(dv_l, axis=0)
                )
            metrics['diag_ood'] = diag_res
            np.savez(seed_dir / f'ood_and_stress_seed{seed}.npz', **os_npz)

        metrics.update({
            'seed': seed,
            'ssl_load_info': res['ssl_load_info'],
            'stage3_unfreeze_info': res['stage3_unfreeze_info'],
            'score_table_bytes': res['score_table_bytes'],
            'n_surgery_injected': res['n_surgery_injected'],
            'seal_violations': len(guard.violations),
            'wall_seconds': res['wall_seconds'],
        })
        write_json(seed_dir / 'metrics.json', metrics)
        return metrics


def _optimize_fusion_alpha_on_screen(
    p_a: np.ndarray,
    p_b: np.ndarray,
    y_scr: np.ndarray,
) -> float:
    best_a, best_nll = 0.0, 1e9
    la = np.log(np.clip(p_a, 1e-8, 1.0))
    lb = np.log(np.clip(p_b, 1e-8, 1.0))
    for alpha in np.linspace(0.0, 0.5, 51):
        comb = (1.0 - alpha) * la + alpha * lb
        comb -= comb.max(axis=1, keepdims=True)
        pr = np.exp(comb)
        pr /= np.maximum(pr.sum(axis=1, keepdims=True), 1e-12)
        nll = float(-np.log(np.clip(pr[np.arange(len(y_scr)), y_scr], 1e-8, 1.0)).mean())
        if nll < best_nll:
            best_nll = nll
            best_a = float(alpha)
    return best_a


def run_stage2_aggregate(
    *,
    cache_dir: Path = GEN4_CACHE_DIR,
    out_dir: Path = EXP_SSL_TAIL_DIR,
    diffphys_dir: Path = EXP_DIFFPHYS_DIR,
    sup_dir: Path = EXP_SUP_DIR,
    seeds: tuple[int, ...] = STAGE2_SEEDS,
) -> dict[str, Any]:
    """Aggregate 3-seed Stage-II SSL+Tail, orthogonal DiffPhys-SBI fusion, POT-GPD, NSBI-PL, & 1000x Bootstrap."""
    from analysis.flavour.nextgen.run_dual_regime_modalnet import (
        _fit_scalar_temperature_on_screen,
        _paired_family_bootstrap_full,
    )

    assert_not_deprecated_mc(cache_dir)
    assert_unsealed_split('tune6k', stage=2)
    geom = load_geometry_contract()

    tune_files = sorted(cache_dir.glob('tune*_part_*.pkl'))
    tune_all: list[dict[str, Any]] = []
    for tf in tune_files:
        assert_unsealed_path(tf)
        with tf.open('rb') as fh:
            tune_all.extend(pickle.load(fh))
    meta_pq = cache_dir.parent / 'validation_meta/tune_truth_meta.parquet'
    ev_meta = _align_tune8k_meta(tune_all, meta_pq, geom)
    in_scr = np.array([bool(it.get('in_screen', False)) for it in tune_all], dtype=bool)
    t6_m = ~in_scr
    y_all = ev_meta['label'].to_numpy(dtype=np.int64)
    y_t6 = y_all[t6_m]
    meta_t6 = ev_meta.loc[t6_m].reset_index(drop=True)

    # 1. Load 3-seed Gen4 SSL+Tail predictions
    per_seed: dict[str, dict[str, Any]] = {}
    p4_s, llr_s, vpv_s = [], [], []
    nsbi_s: dict[str, list[np.ndarray]] = {}
    os_s: dict[str, list[np.ndarray]] = {}
    for s in seeds:
        sd = out_dir / f'stage2/seed_{s}'
        per_seed[str(s)] = json.loads((sd / 'metrics.json').read_text())
        p4_s.append(np.load(sd / f'probs4_tune8k_seed{s}.npy'))
        llr_s.append(np.load(sd / f'tau_llr_tune8k_seed{s}.npy'))
        vpv_s.append(np.load(sd / f'v_pv_tune8k_seed{s}.npy'))
        with np.load(sd / f'nsbi_obs_tune8k_seed{s}.npz') as nz:
            for k in nz.files:
                nsbi_s.setdefault(k, []).append(nz[k])
        osp = sd / f'ood_and_stress_seed{s}.npz'
        if osp.is_file():
            with np.load(osp) as onz:
                for k in onz.files:
                    os_s.setdefault(k, []).append(onz[k])

    ens_p4 = np.mean(p4_s, axis=0)
    ens_llr = np.mean(llr_s, axis=0)
    ens_vpv = np.mean(vpv_s, axis=0)
    ens_nsbi = {k: np.mean(v, axis=0) for k, v in nsbi_s.items()}

    # Fit scalar temperature strictly on Tune_Screen (2,000 rows)
    t_cal = _fit_scalar_temperature_on_screen(ens_p4[in_scr], y_all[in_scr])
    inv_t = float(t_cal['inv_T'])
    log_p = np.log(np.clip(ens_p4, 1e-8, 1.0)) * inv_t
    log_p -= log_p.max(axis=1, keepdims=True)
    ens_p4_cal = np.exp(log_p)
    ens_p4_cal /= np.maximum(ens_p4_cal.sum(axis=1, keepdims=True), 1e-12)

    np.save(out_dir / 'stage2/probs4_tune8k_ens3.npy', ens_p4_cal.astype(np.float32))
    np.save(out_dir / 'stage2/tau_llr_tune8k_ens3.npy', ens_llr.astype(np.float32))

    gen4_eval = _evaluate_predictions(ens_p4_cal[t6_m], ens_llr[t6_m], y_t6, meta_t6, v_pv_mm=ens_vpv[t6_m])

    # 3-seed summary mean +/- std
    seed_stats: dict[str, dict[str, float]] = {}
    for k in ('BA4_pct', 'NLL', 'tau_1pct_fpr_llr_pct', 'tau_0p1pct_fpr_llr_pct', 'Front30_BA4_pct', 'Interm150_BA4_pct', 'Fortune_ScintOnly_BA4_pct'):
        vals = [float(per_seed[str(s)][k]) for s in seeds if k in per_seed[str(s)]]
        if vals:
            seed_stats[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}

    # Ensemble Stress400 & Diag_OOD
    stress_ens: dict[str, Any] = {}
    diag_ens: dict[str, Any] = {}
    diag_meta_p = cache_dir.parent / 'validation_meta/diag_truth_meta.pkl'
    if os_s and diag_meta_p.is_file():
        with diag_meta_p.open('rb') as fh:
            dmeta = pickle.load(fh)
        stress_y = np.asarray(dmeta['stress_y'], dtype=np.int64)
        nom_ba4 = None
        st_names = sorted({k[7:-7] for k in os_s if k.startswith('stress_') and k.endswith('_probs4')})
        for st_name in st_names:
            sp4 = np.mean(os_s[f'stress_{st_name}_probs4'], axis=0)
            sllr = np.mean(os_s[f'stress_{st_name}_llr'], axis=0)
            sm = _evaluate_predictions(sp4, sllr, stress_y, pd.DataFrame())
            stress_ens[st_name] = {'BA4_pct': sm['BA4_pct'], 'tau_1pct_fpr_llr_pct': sm['tau_1pct_fpr_llr_pct']}
            if st_name == 'nominal':
                nom_ba4 = sm['BA4_pct']
        if nom_ba4 is not None:
            for st_name, d in stress_ens.items():
                d['delta_vs_nominal_pp'] = float(d['BA4_pct'] - nom_ba4)
        for dk, d_df_raw in dmeta.get('diag_meta', {}).items():
            kp = f'diag_{dk}_probs4'
            kl = f'diag_{dk}_llr'
            if kp in os_s and kl in os_s:
                dm_df = _annotate_eval_meta(d_df_raw, geom)
                diag_ens[dk] = _evaluate_predictions(
                    np.mean(os_s[kp], axis=0), np.mean(os_s[kl], axis=0), dm_df['label'].to_numpy(dtype=np.int64), dm_df
                )

    # 2. Orthogonal Fusion with DiffPhys-SBI (if present)
    fusion_res: dict[str, Any] = {}
    dp_p4_path = diffphys_dir / 'stage2/probs4_tune8k_ens3.npy'
    dp_llr_path = diffphys_dir / 'stage2/tau_llr_tune8k_ens3.npy'
    if dp_p4_path.is_file() and dp_llr_path.is_file():
        dp_p4 = np.load(dp_p4_path)
        dp_llr = np.load(dp_llr_path)
        if len(dp_p4) == len(ens_p4_cal):
            alpha = _optimize_fusion_alpha_on_screen(ens_p4_cal[in_scr], dp_p4[in_scr], y_all[in_scr])
            la = np.log(np.clip(ens_p4_cal, 1e-8, 1.0))
            lb = np.log(np.clip(dp_p4, 1e-8, 1.0))
            lc = (1.0 - alpha) * la + alpha * lb
            lc -= lc.max(axis=1, keepdims=True)
            fused_p4 = np.exp(lc)
            fused_p4 /= np.maximum(fused_p4.sum(axis=1, keepdims=True), 1e-12)
            fused_llr = (1.0 - alpha) * ens_llr + alpha * dp_llr
            np.save(out_dir / 'stage2/probs4_tune8k_fused.npy', fused_p4.astype(np.float32))
            np.save(out_dir / 'stage2/tau_llr_tune8k_fused.npy', fused_llr.astype(np.float32))
            fusion_res = {
                'alpha_diffphys': alpha,
                'fused_eval': _evaluate_predictions(fused_p4[t6_m], fused_llr[t6_m], y_t6, meta_t6, v_pv_mm=ens_vpv[t6_m]),
                'diffphys_only_eval': _evaluate_predictions(dp_p4[t6_m], dp_llr[t6_m], y_t6, meta_t6),
            }
        else:
            fused_p4, fused_llr = ens_p4_cal, ens_llr
    else:
        fused_p4, fused_llr = ens_p4_cal, ens_llr

    obs_dom_t6 = meta_t6['obs_dom'].to_numpy(dtype=np.int64)
    pc_t6 = meta_t6['pixel_centered'].to_numpy(dtype=bool)
    fp_t6 = meta_t6['front_pinpoint_pixel_centered'].to_numpy(dtype=bool)

    # 3. POT-GPD + NSBI-PL Statistical Engine
    pot_res = compare_pot_gpd_vs_empirical_split_half(fused_llr[t6_m], y_t6, fpr_targets=(0.01, 0.001), n_splits=150)
    dp_obs_path = diffphys_dir / 'stage2/nsbi_obs_tune8k_ens3.npz'
    if dp_obs_path.is_file():
        with np.load(dp_obs_path) as dnz:
            obs_t6 = {k: dnz[k][t6_m] for k in dnz.files}
    else:
        obs_t6 = {k: v[t6_m] for k, v in ens_nsbi.items()}
    obs_t6.setdefault('obs_domain_id', obs_dom_t6)
    obs_t6.setdefault('has_front_pixel', fp_t6)
    obs_t6.setdefault('has_target_pixel', pc_t6)
    obs_t6['tau_llr'] = ens_llr[t6_m]
    has_ch_t6 = meta_t6['charm_flag'].to_numpy(dtype=np.int64) > 0 if 'charm_flag' in meta_t6.columns else np.zeros(len(y_t6), dtype=bool)
    geom_w_t6 = (
        meta_t6['geom_weight'].to_numpy(dtype=np.float64)
        if 'geom_weight' in meta_t6.columns
        else (meta_t6['sample_weight'].to_numpy(dtype=np.float64) if 'sample_weight' in meta_t6.columns else np.ones(len(y_t6), dtype=np.float64))
    )
    nsbi_gen4_raw = evaluate_nsbi_vs_single_bin_closure(
        obs_t6, y_t6, has_charm=has_ch_t6, mw2t_weights=geom_w_t6,
        total_events=180000.0, tau_fraction=0.003, delta_sys=0.0159,
    )
    obs_t6_fused = dict(obs_t6)
    obs_t6_fused['tau_llr'] = fused_llr[t6_m]
    nsbi_fused_raw = evaluate_nsbi_vs_single_bin_closure(
        obs_t6_fused, y_t6, has_charm=has_ch_t6, mw2t_weights=geom_w_t6,
        total_events=180000.0, tau_fraction=0.003, delta_sys=0.0159,
    )
    nsbi_res = {k: v for k, v in nsbi_fused_raw.items() if k != 'templates'}
    nsbi_res['gen4_only_closure'] = {k: v for k, v in nsbi_gen4_raw.items() if k != 'templates'}
    nsbi_res['toy_mc_closure'] = run_nsbi_toy_mc_closure(
        nsbi_fused_raw['templates'], delta_sys=0.0159, n_toys_h0=200, n_toys_h1=200
    )

    # 4. 1,000x Paired Family Bootstrap (seed=20263250) vs Controlled Baselines
    fam_cols = [c for c in FAMILY if c in meta_t6.columns] or ['run', 'chunk']
    fam_codes = pd.MultiIndex.from_frame(meta_t6[fam_cols]).factorize()[0]
    interm_t6 = pc_t6 & (~fp_t6)
    scint_t6 = (~pc_t6) & (obs_dom_t6 == 2)
    slices_boot = {
        'All_Tune6k': np.ones(len(y_t6), dtype=bool),
        'Front30': fp_t6,
        'Interm150': interm_t6,
        'Fortune_ScintOnly': scint_t6,
    }

    bootstrap_vs: dict[str, Any] = {}
    bootstrap_gen4_vs: dict[str, Any] = {}
    cand_baselines: list[tuple[str, Path, bool]] = [
        ('Ref-BDT', diffphys_dir / 'stage2/ref_bdt_probs4_tune8k.npy', True),
    ]
    ref_gn2_paths = [sup_dir / f'ref_l1_gn2/seed_{s}/probs4_tune8k_seed{s}.npy' for s in seeds if (sup_dir / f'ref_l1_gn2/seed_{s}/probs4_tune8k_seed{s}.npy').is_file()]
    if ref_gn2_paths:
        gn2_ens = np.mean([np.load(p) for p in ref_gn2_paths], axis=0)
        np.save(out_dir / 'stage2/tmp_ref_gn2_tune6k.npy', gn2_ens)
        cand_baselines.append(('Ref-L1-GN2', out_dir / 'stage2/tmp_ref_gn2_tune6k.npy', False))
    sup_seed_paths = [sup_dir / f'stage2/seed_{s}/probs4_tune8k_seed{s}.npy' for s in seeds]
    if all(p.is_file() for p in sup_seed_paths):
        sup_ens = np.mean([np.load(p) for p in sup_seed_paths], axis=0)
        np.save(out_dir / 'stage2/tmp_sup_ens_tune6k.npy', sup_ens)
        cand_baselines.append(('PinpointGen4_SupOnly', out_dir / 'stage2/tmp_sup_ens_tune6k.npy', False))
    if dp_p4_path.is_file():
        cand_baselines.append(('DiffPhys-SBI', dp_p4_path, True))

    for b_name, b_path, is_8k in cand_baselines:
        if not b_path.is_file():
            continue
        bp_arr = np.load(b_path)
        bp_t6 = bp_arr[t6_m] if (is_8k and len(bp_arr) == len(tune_all)) else bp_arr
        if len(bp_t6) == len(y_t6):
            bootstrap_vs[b_name] = {
                sn: _paired_family_bootstrap_full(
                    fused_p4[t6_m], bp_t6, y_t6, fam_codes, sl_mask, n_boot=1000, seed=20263250
                )
                for sn, sl_mask in slices_boot.items()
                if sl_mask.any()
            }
            bootstrap_gen4_vs[b_name] = {
                sn: _paired_family_bootstrap_full(
                    ens_p4_cal[t6_m], bp_t6, y_t6, fam_codes, sl_mask, n_boot=1000, seed=20263250
                )
                for sn, sl_mask in slices_boot.items()
                if sl_mask.any()
            }
    if dp_p4_path.is_file() and len(dp_p4) == len(ens_p4_cal):
        bootstrap_vs['PinpointGen4_SSL_Tail'] = {
            sn: _paired_family_bootstrap_full(
                fused_p4[t6_m], ens_p4_cal[t6_m], y_t6, fam_codes, sl_mask, n_boot=1000, seed=20263250
            )
            for sn, sl_mask in slices_boot.items()
            if sl_mask.any()
        }

    summary = {
        'UTC_aggregated': datetime.now(timezone.utc).isoformat(),
        'seeds': list(seeds),
        'temperature_screen': t_cal,
        'seed_stats': seed_stats,
        'gen4_ssl_tail_ens3': gen4_eval,
        'fusion': fusion_res,
        'stress400_ens3': stress_ens,
        'diag_ood_ens3': diag_ens,
        'pot_gpd': pot_res,
        'nsbi_pl': nsbi_res,
        'paired_bootstrap_1000x': bootstrap_vs,
        'paired_bootstrap_1000x_gen4_only': bootstrap_gen4_vs,
        'per_seed': per_seed,
    }
    write_json(out_dir / 'stage2/stage2_summary.json', summary)
    return summary


def write_condor_submit_files(
    *,
    cache_dir: Path = GEN4_CACHE_DIR,
    schedd: str | None = None,
) -> dict[str, Any]:
    """Select least-loaded compatible schedd and generate HTCondor GPU jobs in ~/condor/Pinpoint/stage329_gen4_ssl/."""
    assert_not_deprecated_mc(cache_dir)
    EXP_SSL_TAIL_DIR.mkdir(parents=True, exist_ok=True)
    CONDOR_SSL_DIR.mkdir(parents=True, exist_ok=True)
    logs_dir = CONDOR_SSL_DIR / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    chosen = schedd if schedd else choose_schedd(EXP_SSL_TAIL_DIR)[0]

    run_sh = CONDOR_SSL_DIR / 'run_ssl_tail.sh'
    run_sh.write_text(
        '#!/bin/bash\n'
        'set -e\n'
        'command -v which >/dev/null 2>&1 || { which() { command -v "$@"; }; export -f which; }\n'
        'SAVED_ARGS=("$@"); set --\n'
        'source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh >/dev/null 2>&1\n'
        'set -- "${SAVED_ARGS[@]}"\n'
        'export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4\n'
        'cd /eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis\n'
        'export PYTHONPATH="src:${PYTHONPATH:-}"\n'
        'exec /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/bin/python -u -m analysis.flavour.gen4.run_gen4_ssl_and_tail "$@"\n'
    )
    run_sh.chmod(0o755)

    req_gpu = (
        '((Capability >= 7.5 && GlobalMemoryMb >= 14000) || '
        '((Capability =?= UNDEFINED) && regexp("^NVIDIA H100.* MIG ", DeviceName)))'
    )
    extra_dir_args = '' if cache_dir == GEN4_CACHE_DIR else f' --cache-dir {cache_dir} --out-dir {EXP_SSL_TAIL_DIR}'
    ssl_out_args = '' if cache_dir == GEN4_CACHE_DIR else f' --cache-dir {cache_dir} --out-dir {EXP_SSL_TAIL_DIR}/ssl_v2'

    ssl_sub = CONDOR_SSL_DIR / 'ssl_v2_pretrain.sub'
    ssl_sub.write_text(
        f'universe = vanilla\n'
        f'executable = {run_sh}\n'
        f'arguments = --mode ssl-pretrain --epochs 10{ssl_out_args}\n'
        f'output = {logs_dir}/ssl_v2.out\n'
        f'error = {logs_dir}/ssl_v2.err\n'
        f'log = {logs_dir}/ssl_v2.log\n'
        f'request_gpus = 1\n'
        f'require_gpus = {req_gpu}\n'
        f'request_cpus = 8\n'
        f'request_memory = 24000\n'
        f'+JobFlavour = "tomorrow"\n'
        f'queue 1\n'
    )

    s1_scratch_sub = CONDOR_SSL_DIR / 'stage1_scratch.sub'
    s1_scratch_sub.write_text(
        f'universe = vanilla\n'
        f'executable = {run_sh}\n'
        f'arguments = --mode stage1-train --variant $(VARIANT) --passes-stage1 12 --micro-bs 16 --seed 42{extra_dir_args}\n'
        f'output = {logs_dir}/stage1_$(VARIANT).out\n'
        f'error = {logs_dir}/stage1_$(VARIANT).err\n'
        f'log = {logs_dir}/stage1_$(VARIANT).log\n'
        f'request_gpus = 1\n'
        f'require_gpus = {req_gpu}\n'
        f'request_cpus = 4\n'
        f'request_memory = 15000\n'
        f'+JobFlavour = "workday"\n'
        f'queue VARIANT in (scratch_25pct scratch_10pct)\n'
    )

    s1_scratch_100_sub = CONDOR_SSL_DIR / 'stage1_scratch_100.sub'
    s1_scratch_100_sub.write_text(
        f'universe = vanilla\n'
        f'executable = {run_sh}\n'
        f'arguments = --mode stage1-train --variant scratch_100pct --passes-stage1 8 --micro-bs 16 --seed 42{extra_dir_args}\n'
        f'output = {logs_dir}/stage1_scratch_100pct.out\n'
        f'error = {logs_dir}/stage1_scratch_100pct.err\n'
        f'log = {logs_dir}/stage1_scratch_100pct.log\n'
        f'request_gpus = 1\n'
        f'require_gpus = {req_gpu}\n'
        f'request_cpus = 4\n'
        f'request_memory = 15000\n'
        f'+JobFlavour = "workday"\n'
        f'queue 1\n'
    )

    s1_ssl_sub = CONDOR_SSL_DIR / 'stage1_ssl_ft.sub'
    s1_ssl_sub.write_text(
        f'universe = vanilla\n'
        f'executable = {run_sh}\n'
        f'arguments = --mode stage1-train --variant $(VARIANT) --passes-stage1 8 --micro-bs 16 --ssl-name pinpoint_ssl_v2_frozen.pt --seed 42{extra_dir_args}\n'
        f'output = {logs_dir}/stage1_$(VARIANT).out\n'
        f'error = {logs_dir}/stage1_$(VARIANT).err\n'
        f'log = {logs_dir}/stage1_$(VARIANT).log\n'
        f'request_gpus = 1\n'
        f'require_gpus = {req_gpu}\n'
        f'request_cpus = 4\n'
        f'request_memory = 15000\n'
        f'+JobFlavour = "workday"\n'
        f'queue VARIANT in (ssl_v2_ft_100pct ssl_v2_ft_25pct ssl_v2_ft_10pct abl_no_conflict_free_adamw)\n'
    )

    s2_sub = CONDOR_SSL_DIR / 'stage2_3seed.sub'
    s2_sub.write_text(
        f'universe = vanilla\n'
        f'executable = {run_sh}\n'
        f'arguments = --mode stage2-train --passes-stage1 7 --micro-bs 16 --ssl-name pinpoint_ssl_v2_frozen.pt --seed $(SEED){extra_dir_args}\n'
        f'output = {logs_dir}/stage2_$(SEED).out\n'
        f'error = {logs_dir}/stage2_$(SEED).err\n'
        f'log = {logs_dir}/stage2_$(SEED).log\n'
        f'request_gpus = 1\n'
        f'require_gpus = {req_gpu}\n'
        f'request_cpus = 4\n'
        f'request_memory = 15000\n'
        f'+JobFlavour = "workday"\n'
        f'queue SEED in (17, 42, 2026)\n'
    )
    return {
        'schedd': chosen,
        'run_sh': str(run_sh),
        'ssl_sub': str(ssl_sub),
        'stage1_scratch_sub': str(s1_scratch_sub),
        'stage1_scratch_100_sub': str(s1_scratch_100_sub),
        'stage1_ssl_ft_sub': str(s1_ssl_sub),
        'stage2_sub': str(s2_sub),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Pinpoint-Gen4 Pillar-III SSL & Tail Runner')
    parser.add_argument(
        '--mode',
        choices=(
            'gate-ab',
            'ssl-pretrain',
            'stage1-train',
            'stage1-aggregate',
            'stage2-train',
            'stage2-aggregate',
            'prepare-condor',
        ),
        default='gate-ab',
    )
    parser.add_argument('--cache-dir', type=Path, default=GEN4_CACHE_DIR)
    parser.add_argument('--out-dir', type=Path, default=EXP_SSL_TAIL_DIR)
    parser.add_argument('--ssl-ckpt', type=Path, default=None)
    parser.add_argument('--ssl-name', type=str, default='pinpoint_ssl_v2_backbone.pt')
    parser.add_argument('--variant', type=str, default='ssl_v2_ft_100pct')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--passes-stage1', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=192)
    parser.add_argument('--micro-bs', type=int, default=64)
    parser.add_argument('--max-shards', type=int, default=0)
    parser.add_argument('--lr', type=float, default=4.0e-4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--schedd', type=str, default=None)
    args = parser.parse_args(argv)
    ssl_ckpt_eff = args.ssl_ckpt if args.ssl_ckpt is not None else (EXP_SSL_TAIL_DIR / 'ssl_v2' / args.ssl_name)

    if args.mode == 'gate-ab':
        dev = 'cpu' if args.device == 'cuda:0' and not torch.cuda.is_available() else args.device
        res = run_gate_ab_health(device_str=dev, steps=args.steps)
        print(json.dumps(res, indent=2))
        return
    if args.mode == 'prepare-condor':
        print(json.dumps(write_condor_submit_files(cache_dir=args.cache_dir, schedd=args.schedd), indent=2))
        return
    if args.mode == 'ssl-pretrain':
        print(json.dumps(run_ssl_pretrain(
            cache_dir=args.cache_dir,
            out_dir=args.out_dir if args.out_dir != EXP_SSL_TAIL_DIR else EXP_SSL_TAIL_DIR / 'ssl_v2',
            epochs=args.epochs, batch_size=args.batch_size, micro_bs=args.micro_bs,
            max_shards=args.max_shards, lr=args.lr, seed=args.seed, device_str=args.device,
        ), indent=2))
        return
    if args.mode == 'stage1-train':
        print(json.dumps(run_stage1_train(
            cache_dir=args.cache_dir, out_dir=args.out_dir, ssl_ckpt_path=ssl_ckpt_eff,
            variant=args.variant, seed=args.seed, micro_bs=args.micro_bs,
            passes_stage1_override=args.passes_stage1 if args.passes_stage1 != 12 else None,
            device_str=args.device,
        ), indent=2))
        return
    if args.mode == 'stage1-aggregate':
        print(json.dumps(run_stage1_aggregate(out_dir=args.out_dir), indent=2))
        return
    if args.mode == 'stage2-train':
        print(json.dumps(run_stage2_train(
            cache_dir=args.cache_dir, out_dir=args.out_dir, ssl_ckpt_path=ssl_ckpt_eff,
            seed=args.seed, micro_bs=args.micro_bs, passes_stage1=args.passes_stage1,
            device_str=args.device,
        ), indent=2))
        return
    if args.mode == 'stage2-aggregate':
        print(json.dumps(run_stage2_aggregate(cache_dir=args.cache_dir, out_dir=args.out_dir), indent=2))
        return


if __name__ == '__main__':
    main()
