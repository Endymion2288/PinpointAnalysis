"""Next-generation PINPOINT neutrino flavour & topology reconstruction package.

Strictly isolated under `src/analysis/flavour/nextgen/` with zero modifications to
frozen `v2/` or `v3/` modules and zero access to sealed splits (`Cal-A`, `Cut-B`, `Confirm`).
"""

from analysis.flavour.nextgen.dual_regime_modalnet import DualRegimeModalNet
from analysis.flavour.nextgen.protocols import (
    NextGenFlavourModel,
    NextGenModelOutput,
    check_branch_gradient_health,
    conditional_three_from_four,
    marginal_four_from_seven,
    student_t_vertex_loss,
)
from analysis.flavour.nextgen.seal_guard import (
    FORBIDDEN_SPLIT_TOKENS,
    SealAuditGuard,
    SealViolationError,
    TRUTH_BATCH_KEYS,
    assert_unsealed_path,
    assert_unsealed_split,
    load_authorized_split_shards,
    verify_inference_truth_invariance,
)
from analysis.flavour.nextgen.tail_dro_pauc import (
    ScoreBank,
    TailDROConfig,
    bounded_tau_score,
    compute_anchored_tail_loss,
    compute_multiclass_ece,
    compute_normalized_pauc,
    compute_tau_binary_ece,
    dro_pauc_surrogate_loss,
    extract_penultimate_cache,
    finetune_head_on_cached_latents,
    freeze_backbone_for_tail_finetuning,
    predict_from_cached_latents,
    restore_tf32_mode,
    set_ieee_fp32_mode,
    snapshot_backbone_weights,
    tau_log_odds_from_logits7,
    verify_backbone_bit_identity,
)
from analysis.flavour.nextgen.tau_kinema_diffnet import (
    TauKinemaDiffNet,
    compute_tau_kinema_loss,
    compute_tau_kinema_window_denoms,
)
from analysis.flavour.nextgen.tracklet_ipformer import (
    TrackletIPFormer,
    tracklet_ipformer_loss,
)
from analysis.flavour.nextgen.tracklet_ops import (
    DifferentiableStudentTVertexLayer,
    compute_ipt_spectrometer_interlock,
    compute_scint_tof_residual_ns,
    extract_soft_tracklets_batch,
    highland_scattering_rad,
    pairwise_skew_dca_features,
)
from analysis.flavour.nextgen.unified_pinpoint_net import (
    PinpointUnifiedNet,
    compute_unified_loss,
    compute_unified_window_denominators,
    extract_unified_item,
)

__all__ = [
    'DifferentiableStudentTVertexLayer',
    'DualRegimeModalNet',
    'FORBIDDEN_SPLIT_TOKENS',
    'NextGenFlavourModel',
    'NextGenModelOutput',
    'PinpointUnifiedNet',
    'ScoreBank',
    'SealAuditGuard',
    'SealViolationError',
    'TRUTH_BATCH_KEYS',
    'TailDROConfig',
    'TauKinemaDiffNet',
    'TrackletIPFormer',
    'assert_unsealed_path',
    'assert_unsealed_split',
    'bounded_tau_score',
    'check_branch_gradient_health',
    'compute_anchored_tail_loss',
    'compute_ipt_spectrometer_interlock',
    'compute_multiclass_ece',
    'compute_normalized_pauc',
    'compute_scint_tof_residual_ns',
    'compute_tau_binary_ece',
    'compute_tau_kinema_loss',
    'compute_tau_kinema_window_denoms',
    'compute_unified_loss',
    'compute_unified_window_denominators',
    'conditional_three_from_four',
    'dro_pauc_surrogate_loss',
    'extract_penultimate_cache',
    'extract_soft_tracklets_batch',
    'extract_unified_item',
    'finetune_head_on_cached_latents',
    'freeze_backbone_for_tail_finetuning',
    'highland_scattering_rad',
    'load_authorized_split_shards',
    'marginal_four_from_seven',
    'pairwise_skew_dca_features',
    'predict_from_cached_latents',
    'restore_tf32_mode',
    'set_ieee_fp32_mode',
    'snapshot_backbone_weights',
    'student_t_vertex_loss',
    'tau_log_odds_from_logits7',
    'tracklet_ipformer_loss',
    'verify_backbone_bit_identity',
    'verify_inference_truth_invariance',
]
