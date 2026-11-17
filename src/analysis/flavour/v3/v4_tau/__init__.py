"""Phase V4 modules: observable NOMAD transverse kinematics, DRO partial-AUC finetuning, and ensembling."""

from analysis.flavour.v3.v4_tau.ensemble import (
    calibrate_single_member_oof,
    extract_member_arrays,
    fit_oof_domain_ensemble,
)
from analysis.flavour.v3.v4_tau.models import (
    NomadFusionBlock,
    PinpointFormerV4,
    ZAlignedStereoNetV4,
    b1pp_v4_branch_groups,
    pf_v4_branch_groups,
)
from analysis.flavour.v3.v4_tau.nomad_kinematics import (
    NOMAD_FEAT_DIM,
    NOMAD_FEATURE_NAMES,
    NomadTransverseExtractor,
)
from analysis.flavour.v3.v4_tau.pauc_finetune import DROPartialAUCLoss

__all__ = [
    'DROPartialAUCLoss',
    'NOMAD_FEAT_DIM',
    'NOMAD_FEATURE_NAMES',
    'NomadFusionBlock',
    'NomadTransverseExtractor',
    'PinpointFormerV4',
    'ZAlignedStereoNetV4',
    'b1pp_v4_branch_groups',
    'calibrate_single_member_oof',
    'extract_member_arrays',
    'fit_oof_domain_ensemble',
    'pf_v4_branch_groups',
]
