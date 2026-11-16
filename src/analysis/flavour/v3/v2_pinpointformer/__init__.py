"""PinpointFormer V2: vertex-anchored fine-grained micron + stereo calorimeter architecture."""

from analysis.flavour.v3.v2_pinpointformer.attention import (
    AttentionStack,
    PerceiverCrossBlock,
    collect_attention_scales,
)
from analysis.flavour.v3.v2_pinpointformer.models import PinpointFormer, pf_branch_groups
from analysis.flavour.v3.v2_pinpointformer.pixel_tokenizer import (
    N_MAX_CLUSTERS,
    N_MAX_PATCHES,
    PATCH_SIZE,
    PIX_FEAT_DIM,
    PIX_PLANE_SUMMARY_DIM,
    apply_y_mirror_pf_item,
    collate_pf,
    extract_pf_item,
)
from analysis.flavour.v3.v2_pinpointformer.stereo_bars import (
    N_MAX_NEXUS,
    StereoBarNexus,
    static_scint_stations,
)
from analysis.flavour.v3.v2_pinpointformer.vertex_anchor import (
    K_VTX_MAX,
    PAIR_FEAT_DIM,
    VertexAnchorPredictor,
    compute_near_vertex_pair_bias,
    compute_stage2_polar_features,
)

__all__ = [
    'AttentionStack',
    'K_VTX_MAX',
    'N_MAX_CLUSTERS',
    'N_MAX_NEXUS',
    'N_MAX_PATCHES',
    'PAIR_FEAT_DIM',
    'PATCH_SIZE',
    'PIX_FEAT_DIM',
    'PIX_PLANE_SUMMARY_DIM',
    'PerceiverCrossBlock',
    'PinpointFormer',
    'StereoBarNexus',
    'VertexAnchorPredictor',
    'apply_y_mirror_pf_item',
    'collate_pf',
    'collect_attention_scales',
    'compute_near_vertex_pair_bias',
    'compute_stage2_polar_features',
    'extract_pf_item',
    'pf_branch_groups',
    'static_scint_stations',
]
