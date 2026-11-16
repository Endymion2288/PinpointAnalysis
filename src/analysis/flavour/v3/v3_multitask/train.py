"""Interactive 32-event GPU health gate and multitask trainer for PinpointFormer-GN2 (V3)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from analysis.flavour.v3.v2_pinpointformer.train import (
    DEFAULT_SIDECAR_ROOT,
    run_pf_training,
    run_smoke_32_gate,
)
from analysis.flavour.v3.v3_multitask.models import PinpointFormerMultitask, v3_branch_groups


DEFAULT_V3_HEALTH_OUT = Path(
    '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-08/w3_08_health_report.json'
)


def _pf_v3_loss(model: nn.Module, batch: dict, ctx: dict) -> tuple[torch.Tensor, torch.Tensor]:
    grad_blend = float(ctx.get('grad_blend', 1.0))
    out = model(batch, grad_blend=grad_blend, return_aux=True)
    l_event, _ = model.event_loss(
        out['logits'],
        out['charm_logit'],
        out['v_hat'],
        out['log_var_v'],
        out['kin_pred'],
        batch,
    )
    loss = l_event
    if model.enable_dense_gn2 and model.dense_heads is not None and 'pix_origin_target' in batch:
        dense_l = model.dense_heads.compute_losses(out['dense'], out['aux'], batch)
        loss = l_event + 0.35 * dense_l['l_rel'] + 0.50 * dense_l['l_origin'] + 0.30 * dense_l['l_compat']
    return loss, out['probs4']


def _pf_v3_predict(model: nn.Module, batch: dict) -> dict[str, torch.Tensor]:
    out = model(batch, grad_blend=1.0, return_aux=False)
    return {
        'logits': out['logits'].float(),
        'probs': out['probs4'].float(),
        'v_hat': out['v_hat'].float(),
        'charm_prob': out['charm_prob'].float(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-32', action='store_true')
    parser.add_argument('--output', type=Path, default=DEFAULT_V3_HEALTH_OUT)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    if args.smoke_32:
        cfgs = [
            ('PF_V3_event4head', {'task': 'T4', 'dim': 192, 'heads': 8, 'enable_dense_gn2': False, 'smoke_lr': 1.5e-3}),
            ('PF_V3_gn2_dense', {'task': 'T4', 'dim': 192, 'heads': 8, 'enable_dense_gn2': True, 'smoke_lr': 1.5e-3}),
        ]
        rep = run_smoke_32_gate(
            model_builder=PinpointFormerMultitask,
            branch_groups_fn=v3_branch_groups,
            loss_fn=_pf_v3_loss,
            smoke_configs=cfgs,
            output_path=args.output,
            sidecar_root=DEFAULT_SIDECAR_ROOT,
            load_cluster_sidecar=True,
        )
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
    cfg.setdefault('sidecar_root', str(DEFAULT_SIDECAR_ROOT))
    if cfg.get('enable_dense_gn2', True):
        cfg['load_cluster_sidecar'] = True
    man = run_pf_training(
        root=args.root,
        config=cfg,
        model_builder=PinpointFormerMultitask,
        branch_groups_fn=v3_branch_groups,
        loss_fn=_pf_v3_loss,
        pred_fn=_pf_v3_predict,
        resume=args.resume,
        attach_cluster_sidecar=bool(cfg.get('enable_dense_gn2', True)),
    )
    print(json.dumps({'status': man['status'], 'name': man['name'], 'best_pass': man['best_pass'], 'best_screen_nll': man['best_screen_nll']}, indent=2))


if __name__ == '__main__':
    main()
