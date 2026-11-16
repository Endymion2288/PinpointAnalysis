"""PinpointFormer-GN2 V3 multitask dense vertex/origin/compatibility supervision."""

from analysis.flavour.v3.v3_multitask.dense_heads import (
    CLUSTER_ORIGIN_NAMES,
    TOPO_SUMMARY_DIM,
    DenseGN2Heads,
)
from analysis.flavour.v3.v3_multitask.losses import (
    LABEL_7_NAMES,
    HomoscedasticEventLoss,
    focal_charm_bce,
    marginal_four_from_seven,
    marginal_tau_branch_ce,
)
from analysis.flavour.v3.v3_multitask.models import PinpointFormerMultitask, v3_branch_groups

__all__ = [
    'CLUSTER_ORIGIN_NAMES',
    'DenseGN2Heads',
    'HomoscedasticEventLoss',
    'LABEL_7_NAMES',
    'PinpointFormerMultitask',
    'TOPO_SUMMARY_DIM',
    'focal_charm_bce',
    'marginal_four_from_seven',
    'marginal_tau_branch_ce',
    'v3_branch_groups',
]
