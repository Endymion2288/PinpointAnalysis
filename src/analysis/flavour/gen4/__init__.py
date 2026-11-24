"""Pinpoint-Gen4 next-generation neutrino flavour & topology reconstruction package.

Strictly isolated under ``src/analysis/flavour/gen4/`` with zero modifications to
frozen ``v2/``, ``v3/``, or ``nextgen/`` modules and zero access to sealed splits
(``Cal-A``, ``Cut-B``, ``Confirm``).
"""

from analysis.flavour.gen4.conflict_free_optim import (
    DomainCGGMConfig,
    DomainCGGMProbes,
    DomainCGGMTracker,
    apply_conflict_free_backward,
    collect_gen4_param_groups,
    mmpareto_combine_gradients,
)
from analysis.flavour.gen4.data_cache import (
    assert_unsealed_mc8588_path,
    collate_gen4_batch,
    parse_mc8588_chunk_id,
    partition_mc8588_chunks,
)
from analysis.flavour.gen4.diffphys_sbi import (
    BrokenLineKalmanKinkDetector,
    DiffPhysSBIModel,
    RossiGreisenFISTAShowerStripper,
    ScintStationDesaturator,
    diffphys_sbi_loss,
    extract_diffphys_sufficient_stats,
)
from analysis.flavour.gen4.gen4_unified_model import (
    CrossModalSynergyFusion,
    PinpointGen4Net,
    compute_gen4_supervised_loss,
    extract_nsbi_observables_from_output,
    multivariate_student_t_vertex_loss,
)
from analysis.flavour.gen4.geometry_contract import (
    Gen4GeometryContract,
    load_geometry_contract,
)
from analysis.flavour.gen4.micro_ndive import (
    AnisoPTv3,
    BackConeSlabPrior,
    CompetitiveSlotNDIVE,
    DualTrackClusterEncoder,
    MicroConePairmixerNDIVE,
    MicroVtxOutput,
)
from analysis.flavour.gen4.nsbi_discovery import (
    DiscoveryRouterConfig,
    DomainPOTGPDCalibrator,
    POTGPDFitResult,
    compare_pot_gpd_vs_empirical_split_half,
    evaluate_nsbi_vs_single_bin_closure,
    fit_pot_gpd,
    route_golden_micro_channels,
)
from analysis.flavour.gen4.pflow_fmpe import (
    DeSatSinkhornTomography,
    IntrinsicFMPE,
    MacroHGPflow,
    PFlowFMPEOutput,
    PinpointPFlowFMPE,
    SpecIPTOutput,
    UnifiedMuonSpecEncoder,
)
from analysis.flavour.gen4.physics_frontend import (
    UnifiedMuonKalman,
    build_observation_from_raw_root_event,
    dark_shield_filter_and_features,
    extract_gen4_frontend_event,
    reconstruct_calorimeter_54_stations,
)
from analysis.flavour.gen4.ssl_sonata_v2 import (
    LeJEPASIGReg,
    PinpointSSLv2Model,
    SonataPrototypeHead,
    SonataSSLConfig,
    compute_sonata_v2_ssl_loss,
)
from analysis.flavour.gen4.upart13_tpauc import (
    FullDatasetScoreTable,
    PinpointUParT13Head,
    Stage3TailConfig,
    bayesian_tau_llr_from_logits13,
    compute_nswc_tpauc_loss,
    compute_upart13_lort_loss,
)

__all__ = [
    "AnisoPTv3",
    "BackConeSlabPrior",
    "BrokenLineKalmanKinkDetector",
    "CompetitiveSlotNDIVE",
    "CrossModalSynergyFusion",
    "DeSatSinkhornTomography",
    "DiffPhysSBIModel",
    "DiscoveryRouterConfig",
    "DomainCGGMConfig",
    "DomainCGGMProbes",
    "DomainCGGMTracker",
    "DomainPOTGPDCalibrator",
    "DualTrackClusterEncoder",
    "FullDatasetScoreTable",
    "Gen4GeometryContract",
    "IntrinsicFMPE",
    "LeJEPASIGReg",
    "MacroHGPflow",
    "MicroConePairmixerNDIVE",
    "MicroVtxOutput",
    "PFlowFMPEOutput",
    "POTGPDFitResult",
    "PinpointGen4Net",
    "PinpointPFlowFMPE",
    "PinpointSSLv2Model",
    "PinpointUParT13Head",
    "RossiGreisenFISTAShowerStripper",
    "ScintStationDesaturator",
    "SonataPrototypeHead",
    "SonataSSLConfig",
    "SpecIPTOutput",
    "Stage3TailConfig",
    "UnifiedMuonKalman",
    "UnifiedMuonSpecEncoder",
    "apply_conflict_free_backward",
    "assert_unsealed_mc8588_path",
    "bayesian_tau_llr_from_logits13",
    "build_observation_from_raw_root_event",
    "collate_gen4_batch",
    "collect_gen4_param_groups",
    "compare_pot_gpd_vs_empirical_split_half",
    "compute_gen4_supervised_loss",
    "compute_nswc_tpauc_loss",
    "compute_sonata_v2_ssl_loss",
    "compute_upart13_lort_loss",
    "dark_shield_filter_and_features",
    "diffphys_sbi_loss",
    "evaluate_nsbi_vs_single_bin_closure",
    "extract_diffphys_sufficient_stats",
    "extract_gen4_frontend_event",
    "extract_nsbi_observables_from_output",
    "fit_pot_gpd",
    "load_geometry_contract",
    "mmpareto_combine_gradients",
    "multivariate_student_t_vertex_loss",
    "parse_mc8588_chunk_id",
    "partition_mc8588_chunks",
    "reconstruct_calorimeter_54_stations",
    "route_golden_micro_channels",
]
