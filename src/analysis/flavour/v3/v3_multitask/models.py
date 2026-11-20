"""PinpointFormerMultitask (V3): 4-head homoscedastic + full GN2 dense supervision."""

from __future__ import annotations

import torch
from torch import nn

from analysis.flavour.v3.common.layers import split_adamw_param_groups
from analysis.flavour.v3.v2_pinpointformer.attention import (
    collect_attention_scales,
    collect_effective_attention_sharpness,
)
from analysis.flavour.v3.v2_pinpointformer.models import PinpointFormer, pf_branch_groups
from analysis.flavour.v3.v3_multitask.dense_heads import DenseGN2Heads, TOPO_SUMMARY_DIM
from analysis.flavour.v3.v3_multitask.losses import (
    FixedWeightEventLoss,
    HomoscedasticEventLoss,
    marginal_four_from_seven,
)


class PinpointFormerMultitask(nn.Module):
    """PinpointFormer backbone augmented with 7-way, Charm, Vertex, Kinematics & GN2 dense heads."""

    def __init__(self, geometry: dict, config: dict) -> None:
        super().__init__()
        self.config = config
        self.enable_dense_gn2 = bool(config.get('enable_dense_gn2', True))
        cfg_bb = dict(config)
        cfg_bb['use_vertex_anchor'] = True
        cfg_bb['num_classes'] = 7
        self.backbone = PinpointFormer(geometry, cfg_bb)
        # Replace unused V2 classification head so zero dead parameters exist
        self.backbone.head = nn.Identity()
        dim = self.backbone.dim

        # Auxiliary event-level Charm & cylindrical visible kinematics heads
        in_global = 3 * dim
        head_drop = float(config.get('head_dropout', 0.15))
        self.charm_head = nn.Sequential(
            nn.LayerNorm(in_global),
            nn.Linear(in_global, 192),
            nn.SiLU(),
            nn.Dropout(head_drop),
            nn.Linear(192, 1),
        )
        self.kin_head = nn.Sequential(
            nn.LayerNorm(in_global),
            nn.Linear(in_global, 192),
            nn.SiLU(),
            nn.Linear(192, 4),
        )
        # GN2 dense heads + final 7-way classifier ingesting zero-truth topological summary
        if self.enable_dense_gn2:
            self.dense_heads: DenseGN2Heads | None = DenseGN2Heads(dim=dim, bar_dim=128)
            self.topo_proj: nn.Module | None = nn.Sequential(
                nn.Linear(TOPO_SUMMARY_DIM, 64),
                nn.LayerNorm(64),
                nn.SiLU(),
            )
            head_in_dim = in_global + 64
        else:
            self.dense_heads = None
            self.topo_proj = None
            head_in_dim = in_global
        self.event_head_7 = nn.Sequential(
            nn.LayerNorm(head_in_dim),
            nn.Linear(head_in_dim, 256),
            nn.SiLU(),
            nn.Dropout(head_drop),
            nn.Linear(256, 7),
        )
        if bool(config.get('use_fixed_event_loss', False)):
            self.event_loss: nn.Module = FixedWeightEventLoss(
                w_proc=float(config.get('w_proc', 1.0)),
                w_charm=float(config.get('w_charm', 0.25)),
                w_vtx=float(config.get('w_vtx', 0.20)),
                w_kin=float(config.get('w_kin', 0.15)),
                weight_all_tasks=bool(config.get('weight_all_tasks', False)),
            )
        else:
            self.event_loss = HomoscedasticEventLoss()

    def build_param_groups(self, base_lr: float, weight_decay: float) -> list[dict]:
        return split_adamw_param_groups(self, base_lr, weight_decay)

    def learned_attention_scales(self) -> dict[str, list[float]]:
        return collect_attention_scales(self.backbone)

    def effective_attention_sharpness(self) -> dict[str, list[dict[str, float]]]:
        return collect_effective_attention_sharpness(self.backbone)

    def forward(
        self,
        batch: dict,
        grad_blend: float = 1.0,
        return_aux: bool = False,
        return_entropy: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        aux = self.backbone.forward_features(batch, grad_blend=grad_blend, return_entropy=return_entropy)
        fused = aux['fused_global']
        if self.enable_dense_gn2 and self.dense_heads is not None:
            dense_out = self.dense_heads(aux, enable_dense=True)
        else:
            dense_out = {'topo_summary': fused.new_zeros((len(fused), TOPO_SUMMARY_DIM), dtype=torch.float32)}

        with torch.autocast(device_type=fused.device.type, enabled=False):
            ff = fused.float()
            charm_logit = self.charm_head(ff).squeeze(-1)
            kin_pred = self.kin_head(ff)
            if self.enable_dense_gn2 and self.topo_proj is not None:
                topo_emb = self.topo_proj(dense_out['topo_summary'].float())
                head_in = torch.cat([ff, topo_emb], dim=-1)
            else:
                head_in = ff
            logits7 = self.event_head_7(head_in)
            probs4 = marginal_four_from_seven(logits7, is_probs=False)

        out = {
            'logits': logits7,
            'probs4': probs4,
            'charm_logit': charm_logit,
            'charm_prob': torch.sigmoid(charm_logit),
            'kin_pred': kin_pred,
            'v_hat': aux['v_hat'],
            'log_var_v': aux['log_var_v'],
            'entropies': aux['entropies'],
        }
        if return_aux:
            out['aux'] = aux
            out['dense'] = dense_out
        return out


def v3_branch_groups(model: PinpointFormerMultitask) -> dict[str, list[tuple[str, nn.Parameter]]]:
    """Group PinpointFormerMultitask parameters into clean physical branches for health audit."""
    groups = {
        br: [(f'backbone.{n}', p) for n, p in plist]
        for br, plist in pf_branch_groups(model.backbone).items()
    }
    # Remove unused base backbone.head since event_head_7 replaces it cleanly
    groups.pop('head', None)
    for name, param in model.named_parameters():
        if not param.requires_grad or name.startswith('backbone.'):
            continue
        if name.startswith('charm_head.'):
            br = 'charm_head'
        elif name.startswith('kin_head.'):
            br = 'kin_head'
        elif name.startswith('dense_heads.'):
            br = 'dense_gn2_heads'
        elif name.startswith(('topo_proj.', 'event_head_7.')):
            br = 'event_head_7'
        elif name.startswith('event_loss.'):
            br = 'homoscedastic_weights'
        else:
            br = 'other'
        groups.setdefault(br, []).append((name, param))
    return groups
