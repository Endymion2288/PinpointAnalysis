"""V4 models adding observable NOMAD transverse kinematics & DRO pAUC loss to V1 B1++ and V3 PF-GN2."""

from __future__ import annotations

import json
from pathlib import Path
import torch
from torch import nn

from analysis.flavour.v2.r1.tasks import output_size
from analysis.flavour.v3.common.layers import is_weight_decay_exempt
from analysis.flavour.v3.v1_b1pp.models import ZAlignedStereoNet, b1pp_branch_groups
from analysis.flavour.v3.v2_pinpointformer.train import DEFAULT_GEOMETRY
from analysis.flavour.v3.v3_multitask.losses import marginal_four_from_seven
from analysis.flavour.v3.v3_multitask.models import PinpointFormerMultitask, v3_branch_groups
from analysis.flavour.v3.v4_tau.nomad_kinematics import NOMAD_FEAT_DIM, NomadTransverseExtractor
from analysis.flavour.v3.v4_tau.pauc_finetune import DROPartialAUCLoss


class NomadFusionBlock(nn.Module):
    """Domain-gated MLP encoding 10-dim NOMAD kinematics + 16-dim c_obs + scint flags."""

    def __init__(self, out_dim: int = 64) -> None:
        super().__init__()
        in_dim = NOMAD_FEAT_DIM + 16 + 2
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 96),
            nn.SiLU(),
            nn.Linear(96, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        nomad_feat: torch.Tensor,
        c_obs: torch.Tensor,
        obs_domain_id: torch.Tensor,
    ) -> torch.Tensor:
        with torch.amp.autocast(device_type=c_obs.device.type, enabled=False):
            is_scint = (obs_domain_id == 2).unsqueeze(-1).to(torch.float32)
            has_bar = (c_obs[:, 1] > 0.5).unsqueeze(-1).to(torch.float32)
            active = (obs_domain_id != 0).unsqueeze(-1).to(torch.float32)
            x = torch.cat([nomad_feat.float(), c_obs.float(), is_scint, has_bar], dim=-1)
            return self.net(x) * active


def _append_split_group(
    groups: list[dict],
    named_params: list[tuple[str, nn.Parameter]],
    lr: float,
    weight_decay: float,
) -> None:
    decay, no_decay = [], []
    for name, param in named_params:
        if is_weight_decay_exempt(name, param):
            no_decay.append(param)
        else:
            decay.append(param)
    if decay:
        groups.append({'params': decay, 'lr': lr, 'weight_decay': weight_decay})
    if no_decay:
        groups.append({'params': no_decay, 'lr': lr, 'weight_decay': 0.0})


def _init_residual_head(seq: nn.Sequential) -> None:
    last = seq[-1]
    if isinstance(last, nn.Linear):
        nn.init.normal_(last.weight, mean=0.0, std=0.02)
        if last.bias is not None:
            nn.init.zeros_(last.bias)


class ZAlignedStereoNetV4(ZAlignedStereoNet):
    """B1++ upgraded with observable NOMAD transverse head and optional DRO pAUC criterion."""

    def __init__(self, layout: dict, config: dict, geometry: dict | None = None) -> None:
        super().__init__(layout, config)
        self.enable_nomad = bool(config.get('enable_nomad', True))
        self.enable_pauc = bool(config.get('enable_pauc', False))

        geom = geometry if geometry is not None else layout.get('geometry')
        if geom is None:
            geom = json.loads(Path(config.get('geometry_json', DEFAULT_GEOMETRY)).read_text())

        if self.enable_nomad:
            init_scale = float(config.get('nomad_init_scale', 0.25 if config.get('init_checkpoint') else 1.0))
            phys_cal = not bool(config.get('legacy_nomad_scale', False))
            pe_scale = float(config.get('nomad_pe_per_mip_mev', 40.0 if config.get('pe_only', False) else 1.0))
            self.nomad_extractor: NomadTransverseExtractor | None = NomadTransverseExtractor(
                geom,
                phys_calibrated=phys_cal,
                pe_per_mip_mev=pe_scale,
            )
            self.nomad_block: NomadFusionBlock | None = NomadFusionBlock(out_dim=64)
            self.nomad_head: nn.Sequential | None = nn.Sequential(
                nn.LayerNorm(self.fused_dim + 64),
                nn.Linear(self.fused_dim + 64, 96),
                nn.SiLU(),
                nn.Linear(96, output_size(config['task'])),
            )
            _init_residual_head(self.nomad_head)
            self.nomad_scale: nn.Parameter | None = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        else:
            self.nomad_extractor = None
            self.nomad_block = None
            self.nomad_head = None
            self.nomad_scale = None

        if self.enable_pauc:
            self.pauc_loss: DROPartialAUCLoss | None = DROPartialAUCLoss(
                beta=float(config.get('pauc_beta', 0.01)),
                margin=float(config.get('pauc_margin', 0.40)),
                margin_p=float(config.get('pauc_margin_p', 0.15)),
                kl_temp=float(config.get('pauc_kl_temp', 0.04)),
            )
        else:
            self.pauc_loss = None

    def load_pretrained_backbone(self, ckpt_path: Path, device: torch.device | str = 'cpu') -> None:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = state.get('model', state)
        inc = self.load_state_dict(sd, strict=False)
        if inc.unexpected_keys:
            raise RuntimeError(f'unexpected keys loading ZAlignedStereoNet checkpoint: {inc.unexpected_keys}')
        bad_missing = [k for k in inc.missing_keys if not k.startswith(('nomad_', 'pauc_loss.'))]
        if bad_missing:
            raise RuntimeError(f'missing backbone keys in ZAlignedStereoNetV4: {bad_missing}')

    def build_param_groups(self, base_lr: float, weight_decay: float) -> list[dict]:
        warm = bool(self.config.get('init_checkpoint'))
        backbone_params, top_head_params, nomad_params, pauc_params = [], [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('pauc_loss.'):
                pauc_params.append(param)
            elif name.startswith(('nomad_block.', 'nomad_head.', 'nomad_scale')):
                nomad_params.append((name, param))
            elif warm and name.startswith('head.'):
                top_head_params.append((name, param))
            else:
                backbone_params.append((name, param))

        groups: list[dict] = []
        _append_split_group(groups, backbone_params, (0.15 * base_lr) if warm else base_lr, weight_decay)
        _append_split_group(groups, top_head_params, 0.80 * base_lr, weight_decay)
        _append_split_group(groups, nomad_params, (3.0 * base_lr) if warm else base_lr, weight_decay)
        if pauc_params:
            groups.append({
                'params': pauc_params,
                'lr': max(5e-3, 30.0 * base_lr),
                'weight_decay': 0.0,
            })
        return groups

    def forward(self, batch: dict) -> torch.Tensor:
        fused = self.forward_fused(batch)
        with torch.autocast(device_type=fused.device.type, enabled=False):
            ff = fused.float()
            logits = self.head(ff)
            if self.enable_nomad and self.nomad_extractor is not None and self.nomad_block is not None and self.nomad_head is not None and self.nomad_scale is not None:
                nomad_raw = self.nomad_extractor.from_b1pp(batch)
                nomad_emb = self.nomad_block(nomad_raw, batch['c_obs'], batch['obs_domain_id'])
                delta = self.nomad_head(torch.cat([ff, nomad_emb], dim=-1))
                logits = logits + self.nomad_scale * delta
            return logits


class PinpointFormerV4(PinpointFormerMultitask):
    """PinpointFormer-GN2 upgraded with NOMAD transverse residual heads and DRO pAUC loss."""

    def __init__(self, geometry: dict, config: dict) -> None:
        super().__init__(geometry, config)
        self.enable_nomad = bool(config.get('enable_nomad', True))
        self.enable_pauc = bool(config.get('enable_pauc', False))

        in_global = 3 * self.backbone.dim
        head_in_dim = in_global + (64 if self.enable_dense_gn2 else 0)

        if self.enable_nomad:
            init_scale = float(config.get('nomad_init_scale', 0.25 if config.get('init_checkpoint') else 1.0))
            phys_cal = not bool(config.get('legacy_nomad_scale', False))
            pe_scale = float(config.get('nomad_pe_per_mip_mev', 40.0 if config.get('pe_only', False) else 1.0))
            self.nomad_extractor: NomadTransverseExtractor | None = NomadTransverseExtractor(
                geometry,
                phys_calibrated=phys_cal,
                pe_per_mip_mev=pe_scale,
            )
            self.nomad_block: NomadFusionBlock | None = NomadFusionBlock(out_dim=64)
            self.nomad_head_7: nn.Sequential | None = nn.Sequential(
                nn.LayerNorm(head_in_dim + 64),
                nn.Linear(head_in_dim + 64, 96),
                nn.SiLU(),
                nn.Linear(96, 7),
            )
            self.nomad_charm_head: nn.Sequential | None = nn.Sequential(
                nn.LayerNorm(in_global + 64),
                nn.Linear(in_global + 64, 64),
                nn.SiLU(),
                nn.Linear(64, 1),
            )
            _init_residual_head(self.nomad_head_7)
            _init_residual_head(self.nomad_charm_head)
            self.nomad_scale: nn.Parameter | None = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        else:
            self.nomad_extractor = None
            self.nomad_block = None
            self.nomad_head_7 = None
            self.nomad_charm_head = None
            self.nomad_scale = None

        if self.enable_pauc:
            self.pauc_loss: DROPartialAUCLoss | None = DROPartialAUCLoss(
                beta=float(config.get('pauc_beta', 0.01)),
                margin=float(config.get('pauc_margin', 0.40)),
                margin_p=float(config.get('pauc_margin_p', 0.15)),
                kl_temp=float(config.get('pauc_kl_temp', 0.04)),
            )
        else:
            self.pauc_loss = None

    def load_pretrained_backbone(self, ckpt_path: Path, device: torch.device | str = 'cpu') -> None:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = state.get('model', state)
        inc = self.load_state_dict(sd, strict=False)
        if inc.unexpected_keys:
            raise RuntimeError(f'unexpected keys loading PinpointFormerMultitask checkpoint: {inc.unexpected_keys}')
        bad_missing = [k for k in inc.missing_keys if not k.startswith(('nomad_', 'pauc_loss.'))]
        if bad_missing:
            raise RuntimeError(f'missing backbone keys in PinpointFormerV4: {bad_missing}')

    def build_param_groups(self, base_lr: float, weight_decay: float) -> list[dict]:
        warm = bool(self.config.get('init_checkpoint'))
        backbone_params, top_head_params, nomad_params, pauc_params = [], [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('pauc_loss.'):
                pauc_params.append(param)
            elif name.startswith(('nomad_block.', 'nomad_head_7.', 'nomad_charm_head.', 'nomad_scale')):
                nomad_params.append((name, param))
            elif warm and name.startswith(('event_head_7.', 'charm_head.', 'topo_proj.')):
                top_head_params.append((name, param))
            else:
                backbone_params.append((name, param))

        groups: list[dict] = []
        _append_split_group(groups, backbone_params, (0.15 * base_lr) if warm else base_lr, weight_decay)
        _append_split_group(groups, top_head_params, 0.80 * base_lr, weight_decay)
        _append_split_group(groups, nomad_params, (3.0 * base_lr) if warm else base_lr, weight_decay)
        if pauc_params:
            groups.append({
                'params': pauc_params,
                'lr': max(5e-3, 30.0 * base_lr),
                'weight_decay': 0.0,
            })
        return groups

    def forward(
        self,
        batch: dict,
        grad_blend: float = 1.0,
        return_aux: bool = False,
        return_entropy: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        out = super().forward(batch, grad_blend=grad_blend, return_aux=True, return_entropy=return_entropy)
        if self.enable_nomad and self.nomad_extractor is not None and self.nomad_block is not None and self.nomad_head_7 is not None and self.nomad_charm_head is not None and self.nomad_scale is not None:
            aux = out['aux']
            dense_out = out['dense']
            nomad_raw = self.nomad_extractor.from_pf(batch)
            with torch.amp.autocast(device_type=aux['fused_global'].device.type, enabled=False):
                ff = aux['fused_global'].float()
                nomad_emb = self.nomad_block(nomad_raw, batch['c_obs'], batch['obs_domain_id'])
                if self.enable_dense_gn2 and self.topo_proj is not None:
                    topo_emb = self.topo_proj(dense_out['topo_summary'].float())
                    head_in = torch.cat([ff, topo_emb, nomad_emb], dim=-1)
                else:
                    head_in = torch.cat([ff, nomad_emb], dim=-1)
                logits7 = out['logits'].float() + self.nomad_scale * self.nomad_head_7(head_in)
                charm_logit = out['charm_logit'].float() + self.nomad_scale * self.nomad_charm_head(
                    torch.cat([ff, nomad_emb], dim=-1)
                ).squeeze(-1)
                out['logits'] = logits7
                out['probs4'] = marginal_four_from_seven(logits7, is_probs=False)
                out['charm_logit'] = charm_logit
                out['charm_prob'] = torch.sigmoid(charm_logit)
        if not return_aux:
            out.pop('aux', None)
            out.pop('dense', None)
        return out


def b1pp_v4_branch_groups(model: ZAlignedStereoNetV4) -> dict[str, list[tuple[str, nn.Parameter]]]:
    """Group ZAlignedStereoNetV4 parameters into physical branches for gradient auditing."""
    groups = b1pp_branch_groups(model)
    head_keep = []
    nomad_list = []
    pauc_list = []
    for name, param in groups.pop('head', []):
        if name.startswith(('nomad_block.', 'nomad_head.', 'nomad_scale')):
            nomad_list.append((name, param))
        elif name.startswith('pauc_loss.'):
            pauc_list.append((name, param))
        else:
            head_keep.append((name, param))
    if head_keep:
        groups['head'] = head_keep
    if nomad_list:
        groups['nomad_kinematics'] = nomad_list
    if pauc_list:
        groups['pauc_dro'] = pauc_list
    return groups


def pf_v4_branch_groups(model: PinpointFormerV4) -> dict[str, list[tuple[str, nn.Parameter]]]:
    """Group PinpointFormerV4 parameters into physical branches for gradient auditing."""
    groups = v3_branch_groups(model)
    other_keep = []
    nomad_list = []
    pauc_list = []
    for name, param in groups.pop('other', []):
        if name.startswith(('nomad_block.', 'nomad_head_7.', 'nomad_charm_head.', 'nomad_scale')):
            nomad_list.append((name, param))
        elif name.startswith('pauc_loss.'):
            pauc_list.append((name, param))
        else:
            other_keep.append((name, param))
    if other_keep:
        groups['other'] = other_keep
    if nomad_list:
        groups['nomad_kinematics'] = nomad_list
    if pauc_list:
        groups['pauc_dro'] = pauc_list
    return groups
