"""Gate A & Gate B reusable model-health and 32-event overfitting convergence harness."""

from __future__ import annotations

import math
import os
from typing import Callable
import unittest

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from analysis.flavour.v3.common.domains import compute_observable_containment, detector_bounds
from analysis.flavour.v3.v1_b1pp.models import ZAlignedStereoNet, b1pp_branch_groups
from analysis.flavour.v3.v1_b1pp.representation import (
    collate_b1pp,
    extract_b1pp_item,
    static_plane_layout,
)
from analysis.flavour.v3.v2_pinpointformer.pixel_tokenizer import collate_pf, extract_pf_item
from analysis.flavour.v3.v2_pinpointformer.train import move_batch_to_device
from analysis.flavour.v3.v3_multitask.models import PinpointFormerMultitask, v3_branch_groups
from tests.test_nextgen_representation_probes import SyntheticPinpointPhysicsGenerator
from tests.test_nextgen_seal_and_splits import build_minimal_geometry


def _extract_logits_and_probs4(out: torch.Tensor | dict) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.is_tensor(out):
        logits = out
        probs = torch.softmax(logits.float(), dim=-1)
        return logits, probs
    if isinstance(out, dict):
        logits = None
        for k in ('logits', 'logits_4', 'logits_7', 'cls_logits'):
            if k in out and torch.is_tensor(out[k]):
                logits = out[k]
                break
        if logits is None:
            raise KeyError('Model output dict missing logits tensor')
        if 'probs4' in out and torch.is_tensor(out['probs4']):
            probs = out['probs4'].float()
        else:
            probs = torch.softmax(logits.float(), dim=-1)
        return logits, probs
    raise TypeError(f'Unsupported model output type: {type(out)!r}')


def run_gate_a_health_check(
    model: nn.Module,
    batch: dict,
    loss_fn: Callable[[torch.Tensor | dict, dict], torch.Tensor],
    *,
    expected_logits_shape: tuple[int, ...] | None = None,
    branch_groups_fn: Callable[[nn.Module], dict[str, list[tuple[str, nn.Parameter]]]] | None = None,
    upstream_branches: tuple[str, ...] | None = None,
    min_pred_std: float = 1e-4,
    check_activations: bool = True,
) -> dict:
    """Gate A: verify forward/backward shapes, finite loss, non-constant preds, and non-zero upstream grads."""
    model.train()
    model.zero_grad(set_to_none=True)

    sat_stats: dict[str, dict[str, float]] = {}
    hooks = []
    if check_activations:
        def _make_hook(name: str, is_bounded: bool):
            def _hook(_mod: nn.Module, _inp: tuple, output: torch.Tensor) -> None:
                if not torch.is_tensor(output) or output.numel() == 0:
                    return
                t = output.detach().float()
                if is_bounded:
                    sat_frac = float((t.abs() > 0.999).float().mean().item())
                else:
                    sat_frac = float((t.abs() < 1e-12).float().mean().item())
                sat_stats[name] = {
                    'finite': bool(torch.isfinite(t).all().item()),
                    'std': float(t.std(unbiased=False).item()),
                    'sat_frac': sat_frac,
                }
            return _hook

        for mod_name, mod in model.named_modules():
            if isinstance(mod, (nn.Tanh, nn.Sigmoid)):
                hooks.append(mod.register_forward_hook(_make_hook(mod_name, True)))
            elif isinstance(mod, (nn.ReLU, nn.SiLU, nn.GELU)):
                hooks.append(mod.register_forward_hook(_make_hook(mod_name, False)))

    try:
        out = model(batch)
    finally:
        for h in hooks:
            h.remove()

    logits, probs = _extract_logits_and_probs4(out)
    if expected_logits_shape is not None and tuple(logits.shape) != tuple(expected_logits_shape):
        raise AssertionError(f'Logits shape {tuple(logits.shape)} != expected {expected_logits_shape}')
    if not torch.isfinite(logits).all() or not torch.isfinite(probs).all():
        raise AssertionError('Non-finite logits or probabilities in forward pass')

    bsz = int(logits.shape[0])
    pred_std = float(logits.detach().float().std(dim=0, unbiased=False).mean().item())
    max_pair_diff = float((logits.detach().float() - logits.detach().float()[:1]).abs().max().item()) if bsz >= 2 else 1.0
    if bsz >= 2 and (pred_std <= min_pred_std or max_pair_diff <= min_pred_std):
        raise AssertionError(
            f'Constant or collapsed predictions across batch: pred_std={pred_std:.3e}, max_pair_diff={max_pair_diff:.3e}'
        )

    for act_name, st in sat_stats.items():
        if not st['finite']:
            raise AssertionError(f'Non-finite hidden activation in {act_name}')
        if st['sat_frac'] >= 0.999 and st['std'] < 1e-7:
            raise AssertionError(f'Saturated/dead hidden activation in {act_name}: {st}')

    loss = loss_fn(out, batch)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise AssertionError(f'Loss must be a finite scalar, got shape={tuple(loss.shape)}, val={loss}')

    loss.backward()

    # Check finiteness of all realized gradients
    for pname, param in model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            raise AssertionError(f'Non-finite gradient in parameter {pname}')

    if branch_groups_fn is not None:
        groups = branch_groups_fn(model)
    else:
        groups = {}
        for pname, param in model.named_parameters():
            if not param.requires_grad:
                continue
            top = pname.split('.')[0]
            groups.setdefault(top, []).append((pname, param))

    branch_grad_norms: dict[str, float] = {}
    dead_branches: list[str] = []
    for br, plist in groups.items():
        gnorm = math.sqrt(sum(float(p.grad.detach().float().pow(2).sum().item()) for _, p in plist if p.grad is not None))
        branch_grad_norms[br] = gnorm
        if gnorm <= 0.0:
            dead_branches.append(br)

    check_branches = list(upstream_branches) if upstream_branches is not None else list(groups.keys())
    missing_or_dead = [br for br in check_branches if branch_grad_norms.get(br, 0.0) <= 0.0]
    if missing_or_dead:
        raise AssertionError(
            f'Zero upstream/branch gradient detected in {missing_or_dead}: norms={branch_grad_norms}'
        )

    return {
        'loss': float(loss.detach().item()),
        'logits_shape': tuple(logits.shape),
        'pred_std': pred_std,
        'max_pair_diff': max_pair_diff,
        'branch_grad_norms': branch_grad_norms,
        'dead_branches': dead_branches,
        'passed': True,
    }


def run_gate_b_overfit_32_check(
    model: nn.Module,
    batch: dict,
    loss_fn: Callable[[torch.Tensor | dict, dict], torch.Tensor],
    *,
    steps: int = 75,
    lr: float = 3e-3,
    weight_decay: float = 0.0,
    min_acc: float = 0.90,
    min_loss_drop: float = 0.40,
    upstream_param_prefixes: tuple[str, ...] | None = None,
) -> dict:
    """Gate B: verify 32-event overfitting convergence, non-constant predictions, and encoder weight movement."""
    init_params = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad and (upstream_param_prefixes is None or name.startswith(upstream_param_prefixes))
    }

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    model.eval()
    with torch.inference_mode():
        init_out = model(batch)
        initial_loss = float(loss_fn(init_out, batch).item())

    model.train()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(batch)
        loss = loss_fn(out, batch)
        if not torch.isfinite(loss):
            raise AssertionError(f'Non-finite training loss during Gate B overfit: {loss}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

    model.eval()
    with torch.inference_mode():
        final_out = model(batch)
        final_loss = float(loss_fn(final_out, batch).item())
        logits, probs = _extract_logits_and_probs4(final_out)
        preds = probs.argmax(dim=-1)
        labels = batch['label'].long()
        final_acc = float((preds == labels).float().mean().item())
        unique_preds = int(preds.unique().numel())
        pred_std = float(logits.float().std(dim=0, unbiased=False).mean().item())

    upstream_delta = 0.0
    for name, param in model.named_parameters():
        if name in init_params:
            upstream_delta += float((param.detach() - init_params[name]).pow(2).sum().item())
    upstream_delta = math.sqrt(upstream_delta)

    if final_loss >= initial_loss - min_loss_drop:
        raise AssertionError(
            f'Gate B loss did not converge: initial={initial_loss:.4f}, final={final_loss:.4f}'
        )
    if final_acc < min_acc:
        raise AssertionError(f'Gate B 32-event accuracy {final_acc:.3f} < {min_acc:.3f}')
    if unique_preds < 2 or pred_std <= 1e-3:
        raise AssertionError(f'Gate B collapsed predictions: unique={unique_preds}, std={pred_std:.3e}')
    if upstream_delta <= 0.0:
        raise AssertionError('Gate B upstream encoder parameters did not update')

    return {
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'final_acc': final_acc,
        'unique_pred_classes': unique_preds,
        'pred_std': pred_std,
        'upstream_param_delta': upstream_delta,
        'passed': True,
    }


def _build_32_event_4class_observations(geometry: dict, bounds: dict, seed: int = 325) -> tuple[list[dict], list[int]]:
    """Build 32 deterministic synthetic 4-class events (8 per class) with clear physical signatures."""
    gen = SyntheticPinpointPhysicsGenerator(geometry=geometry, seed=seed)
    nx, ny = int(bounds['nx']), int(bounds['ny'])
    px, py, pz = gen.px, gen.py, gen.pz
    yc = float(bounds['y_scint_center'])

    observations: list[dict] = []
    labels: list[int] = []

    for idx in range(32):
        cls_id = idx % 4
        pv_x = float(gen.rng.uniform(-6.0, 6.0))
        pv_y = float(yc + gen.rng.uniform(-6.0, 6.0))
        pv_z = float(pz[0] + 10.5)
        pv = np.array([pv_x, pv_y, pv_z], dtype=np.float64)

        hits: list[tuple[int, int, int]] = []
        if cls_id == 0:
            # nu_e CC: compact dense EM shower in planes 1..3 + high early bar PE, no 3-station muon
            for k in range(4):
                sl = (float(gen.rng.uniform(-0.02, 0.02)), float(gen.rng.uniform(-0.02, 0.02)))
                hits.extend(gen.propagate_track(pv, sl, planes=(1, 2, 3), momentum_gev=15.0, cluster_2px_prob=0.8))
            # Add extra dense cluster hits on planes 2 & 3
            for dc in range(4):
                col = int(math.floor((pv_x - bounds['px_min']) / bounds['dx_mm'])) + dc
                row = int(math.floor((pv_y - bounds['py_min']) / bounds['dy_mm'])) + (dc % 2)
                if 0 <= col < nx and 0 <= row < ny:
                    hits.extend([(2, row, col), (3, row + 1, col)])
            bar_pe = [2200, 1800, 600]
            bar_panels = [int(bounds['vert_panels'][1]), int(bounds['vert_panels'][2]), int(bounds['horiz_panels'][0])]
            trk_obs = {
                'station': np.empty(0, dtype=np.int8),
                'xy_mm': np.empty((0, 2), dtype=np.float32),
                'sigma_xy_mm': np.empty((0, 2), dtype=np.float32),
            }
        elif cls_id == 1:
            # nu_mu CC: long penetrating MIP across planes 1..5 + 12..14 + clean 3-station spectrometer track
            for sl in ((0.005, -0.004), (-0.04, 0.03)):
                hits.extend(gen.propagate_track(pv, sl, planes=(1, 2, 3, 4, 5, 12, 13, 14), momentum_gev=60.0, cluster_2px_prob=0.1))
            bar_pe = [120, 140, 150]
            bar_panels = [int(bounds['vert_panels'][1]), int(bounds['vert_panels'][10]), int(bounds['horiz_panels'][10])]
            trk_obs = {
                'station': np.array([0, 1, 2], dtype=np.int8),
                'xy_mm': np.array([[0.0, -10.0], [0.2, -11.2], [0.4, -10.0]], dtype=np.float32),
                'sigma_xy_mm': np.full((3, 2), 0.1, dtype=np.float32),
            }
        elif cls_id == 2:
            # nu_tau CC: displaced 3-prong decay (dz ~ 3 mm, d0 ~ 0.18 mm) + moderate bar PE
            obs_tau, _ = gen.generate_prompt_vs_displaced_event(
                displaced=True, n_prongs=3, dz_sv_mm=3.0, d0_target_mm=0.18
            )
            hits = [
                (
                    int(cid // (nx * ny)),
                    int((cid // nx) % ny),
                    int(cid % nx),
                )
                for cid in obs_tau['pixel']['channel_id']
            ]
            bar_pe = [650, 720, 500]
            bar_panels = [int(bounds['vert_panels'][1]), int(bounds['vert_panels'][4]), int(bounds['horiz_panels'][2])]
            trk_obs = {
                'station': np.empty(0, dtype=np.int8),
                'xy_mm': np.empty((0, 2), dtype=np.float32),
                'sigma_xy_mm': np.empty((0, 2), dtype=np.float32),
            }
        else:
            # NC: deeper hadronic activity starting at plane 3..6, wider spread, no spectrometer track
            pv_nc = np.array([pv_x + 15.0, pv_y - 12.0, float(pz[2] + 10.5)], dtype=np.float64)
            for sl in ((-0.09, 0.08), (0.10, -0.07), (0.04, 0.09)):
                hits.extend(gen.propagate_track(pv_nc, sl, planes=(3, 4, 5, 6), momentum_gev=25.0, cluster_2px_prob=0.35))
            bar_pe = [500, 950, 900]
            bar_panels = [int(bounds['vert_panels'][6]), int(bounds['vert_panels'][12]), int(bounds['horiz_panels'][6])]
            trk_obs = {
                'station': np.empty(0, dtype=np.int8),
                'xy_mm': np.empty((0, 2), dtype=np.float32),
                'sigma_xy_mm': np.empty((0, 2), dtype=np.float32),
            }

        obs = gen.hits_to_observation(hits)
        bh = np.array([0, 0, 1], dtype=np.int8)
        bi = np.array([15, 18, 22], dtype=np.int16)
        bb = np.zeros((3, 5), dtype=np.float32)
        for k, (p, h, b) in enumerate(zip(bar_panels, bh, bi)):
            if h == 0:
                xc = geometry['scint_bar_Xcenter'][b]
                bb[k] = [xc - 5.25, xc + 5.25, -222.0, 198.0, geometry['scint_Zpos'][p]]
            else:
                yb = geometry['scint_bar_Ycenter'][b]
                bb[k] = [-133.0, 287.0, yb - 5.25, yb + 5.25, geometry['scint_Zpos'][p]]
        obs['bar'] = {
            'channel_id': np.array([1, 2, 3], dtype=np.int64),
            'layer': np.array(bar_panels, dtype=np.int16),
            'panel': np.array(bar_panels, dtype=np.int16),
            'horizontal': bh,
            'bar': bi,
            'bounds_mm': bb,
            'pe_proxy': np.asarray(bar_pe, dtype=np.int64),
            'hit_time_ns': np.array([3.5, 4.2, 4.3], dtype=np.float32),
        }
        obs['tracker_hypothesis'] = trk_obs
        observations.append(obs)
        labels.append(cls_id)

    return observations, labels


class TestNextgenModelHealth(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(4)
        cls.geometry = build_minimal_geometry()
        cls.bounds = detector_bounds(cls.geometry)
        cls.obs32, cls.lbl32 = _build_32_event_4class_observations(cls.geometry, cls.bounds, seed=325)
        cls.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def _build_pf_batch(self, n_events: int) -> dict:
        cfg = {'pe_only': True}
        items = []
        for i in range(n_events):
            dom = compute_observable_containment(self.obs32[i], self.geometry, self.bounds, cfg)
            items.append(extract_pf_item(i, self.obs32[i], self.lbl32[i], dom, self.geometry, self.bounds, cfg))
        batch = collate_pf(items, y_scint_center=self.bounds['y_scint_center'], n_bars=self.bounds['n_bars'])
        return move_batch_to_device(batch, self.device)

    def _build_b1pp_batch(self, n_events: int) -> dict:
        cfg = {'pe_only': True}
        items = []
        for i in range(n_events):
            dom = compute_observable_containment(self.obs32[i], self.geometry, self.bounds, cfg)
            items.append(extract_b1pp_item(i, self.obs32[i], self.lbl32[i], dom, self.geometry, self.bounds, cfg))
        batch = collate_b1pp(items)
        out = {}
        for k, v in batch.items():
            out[k] = v.to(self.device) if torch.is_tensor(v) else v
        return out

    def test_gate_a_health_and_negative_controls(self) -> None:
        """Gate A passes on PinpointFormerMultitask & B1++ and catches dead encoders / constant preds."""
        torch.manual_seed(42)
        pf_batch = self._build_pf_batch(8)
        pf_cfg = {
            'task': 'T4',
            'dim': 32,
            'heads': 2,
            'n_local_layers': 1,
            'n_global_layers': 1,
            'drop_path': 0.0,
            'enable_dense_gn2': True,
            'enable_micro_vertex': True,
        }
        pf_model = PinpointFormerMultitask(self.geometry, pf_cfg).to(self.device)

        def _pf_loss(out: dict, b: dict) -> torch.Tensor:
            probs4 = out['probs4'].clamp(min=1e-6)
            cls_loss = F.nll_loss(torch.log(probs4), b['label'])
            return cls_loss + 0.01 * out['v_hat'].pow(2).mean() + 0.01 * out['charm_logit'].pow(2).mean() + 0.01 * out['kin_pred'].pow(2).mean()

        res_pf = run_gate_a_health_check(
            pf_model,
            pf_batch,
            _pf_loss,
            expected_logits_shape=(8, 7),
            branch_groups_fn=v3_branch_groups,
            upstream_branches=('pixel_clusters', 'pixel_patch_conv', 'stereo_bars', 'spectrometer', 'vertex_anchor'),
        )
        self.assertTrue(res_pf['passed'])

        b1_batch = self._build_b1pp_batch(8)
        b1_cfg = {'task': 'T4', 'dim': 32, 'heads': 2, 'n_long_layers': 2, 'drop_path': 0.0, 'channel_dropout': 0.0}
        b1_model = ZAlignedStereoNet(static_plane_layout(self.geometry), b1_cfg).to(self.device)
        res_b1 = run_gate_a_health_check(
            b1_model,
            b1_batch,
            lambda out, b: F.cross_entropy(out, b['label']),
            expected_logits_shape=(8, 4),
            branch_groups_fn=b1pp_branch_groups,
        )
        self.assertTrue(res_b1['passed'])

        # Negative control 1: dead upstream encoder
        class _DeadEncoderModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dead_enc = nn.Linear(16, 16)
                self.head = nn.Linear(16, 4)

            def forward(self, b: dict) -> torch.Tensor:
                with torch.no_grad():
                    _ = self.dead_enc(b['c_obs'])
                return self.head(b['c_obs'])

        with self.assertRaises(AssertionError):
            run_gate_a_health_check(
                _DeadEncoderModel().to(self.device),
                pf_batch,
                lambda out, b: F.cross_entropy(out, b['label']),
                upstream_branches=('dead_enc', 'head'),
            )

        # Negative control 2: constant prediction across batch
        class _ConstantModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bias = nn.Parameter(torch.zeros(4))

            def forward(self, b: dict) -> torch.Tensor:
                return self.bias.unsqueeze(0).expand(len(b['c_obs']), -1)

        with self.assertRaises(AssertionError):
            run_gate_a_health_check(
                _ConstantModel().to(self.device),
                pf_batch,
                lambda out, b: F.cross_entropy(out, b['label']),
            )

    def test_gate_b_32_event_overfitting_convergence(self) -> None:
        """Gate B verifies 32-event 4-class memorization convergence and upstream weight updates."""
        torch.manual_seed(17)
        pf_batch = self._build_pf_batch(32)
        pf_cfg = {
            'task': 'T4',
            'dim': 32,
            'heads': 2,
            'n_local_layers': 1,
            'n_global_layers': 1,
            'k_vtx': 16,
            'n_queries': 4,
            'drop_path': 0.0,
            'head_dropout': 0.0,
            'enable_dense_gn2': False,
            'enable_micro_vertex': True,
        }
        pf_model = PinpointFormerMultitask(self.geometry, pf_cfg).to(self.device)

        def _loss_fn(out: dict, b: dict) -> torch.Tensor:
            probs4 = out['probs4'].clamp(min=1e-6)
            return F.nll_loss(torch.log(probs4), b['label'])

        n_steps = 40 if self.device.type == 'cuda' else 18
        lr_val = 6e-3 if self.device.type == 'cuda' else 1e-2
        res = run_gate_b_overfit_32_check(
            pf_model,
            pf_batch,
            _loss_fn,
            steps=n_steps,
            lr=lr_val,
            min_acc=0.90,
            min_loss_drop=0.45,
            upstream_param_prefixes=('backbone.pix_in.', 'backbone.stereo_bars.'),
        )
        self.assertTrue(res['passed'])
        self.assertGreaterEqual(res['final_acc'], 0.90)
        self.assertEqual(res['unique_pred_classes'], 4)


if __name__ == '__main__':
    unittest.main()
