"""Next-generation PINPOINT neutrino flavour & topology reconstruction package.

Strictly isolated under `src/analysis/flavour/nextgen/` with zero modifications to
frozen `v2/` or `v3/` modules and zero access to sealed splits (`Cal-A`, `Cut-B`, `Confirm`).
"""

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
    assert_unsealed_path,
    assert_unsealed_split,
    load_authorized_split_shards,
    verify_inference_truth_invariance,
)
from analysis.flavour.nextgen.tracklet_ops import (
    DifferentiableStudentTVertexLayer,
    compute_ipt_spectrometer_interlock,
    compute_scint_tof_residual_ns,
    extract_soft_tracklets_batch,
    highland_scattering_rad,
    pairwise_skew_dca_features,
)

__all__ = [
    'DifferentiableStudentTVertexLayer',
    'FORBIDDEN_SPLIT_TOKENS',
    'NextGenFlavourModel',
    'NextGenModelOutput',
    'SealAuditGuard',
    'SealViolationError',
    'assert_unsealed_path',
    'assert_unsealed_split',
    'check_branch_gradient_health',
    'compute_ipt_spectrometer_interlock',
    'compute_scint_tof_residual_ns',
    'conditional_three_from_four',
    'extract_soft_tracklets_batch',
    'highland_scattering_rad',
    'load_authorized_split_shards',
    'marginal_four_from_seven',
    'pairwise_skew_dca_features',
    'student_t_vertex_loss',
    'verify_inference_truth_invariance',
]
