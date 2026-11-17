"""Phase V5 (W3-11 multi-seed stability & W3-12 one-shot Confirm200k unsealing) package."""

from __future__ import annotations

from pathlib import Path

DEFAULT_V3_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15')
DEFAULT_V2_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14')
DEFAULT_S6_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s6_2026-11-14')
DEFAULT_INVENTORY = Path(
    '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s0_s1_2026-11-11/inventory/events.parquet'
)
DEFAULT_GEOMETRY = Path(
    '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s0_s1_2026-11-11/shards/10083/geometry.json'
)
DEFAULT_INPUT_ROOT = Path('/eos/project-f/fasersim-bonn/public/pinpoint/data')

SEEDS: tuple[int, ...] = (17, 42, 2026)

# Canonical 5-role dual-regime ensemble ordering per seed
ROLE_ORDER: tuple[str, ...] = (
    'pf_v3_gn2',
    'pf_v4_nomad_only',
    'pf_v4_nomad_pauc',
    'b1pp_v3',
    'b1pp_v4_nomad_pauc',
)


def experiment_dir_for(v3_root: Path, role: str, seed: int) -> Path:
    """Resolve the experiment directory for a canonical (role, seed) checkpoint."""
    seed = int(seed)
    if seed == 42:
        mapping = {
            'b1pp_v3': v3_root / 'W3-05/experiments/b1pp_300k_24p_t4',
            'b1pp_v4_nomad_pauc': v3_root / 'W3-10/experiments/b1pp_v4_300k_3p_nomad_pauc',
            'pf_v3_gn2': v3_root / 'W3-09/experiments/pf_v3_300k_24p_gn2_flagship',
            'pf_v4_nomad_only': v3_root / 'W3-10/experiments/pf_v4_300k_3p_nomad_only',
            'pf_v4_nomad_pauc': v3_root / 'W3-10/experiments/pf_v4_300k_3p_nomad_pauc',
        }
    elif seed in (17, 2026):
        mapping = {
            'b1pp_v3': v3_root / f'W3-11/experiments/b1pp_300k_24p_t4_s{seed}',
            'b1pp_v4_nomad_pauc': v3_root / f'W3-11/experiments/b1pp_v4_300k_3p_nomad_pauc_s{seed}',
            'pf_v3_gn2': v3_root / f'W3-11/experiments/pf_v3_300k_24p_gn2_s{seed}',
            'pf_v4_nomad_only': v3_root / f'W3-11/experiments/pf_v4_300k_3p_nomad_only_s{seed}',
            'pf_v4_nomad_pauc': v3_root / f'W3-11/experiments/pf_v4_300k_3p_nomad_pauc_s{seed}',
        }
    else:
        raise ValueError(f'Unsupported canonical seed: {seed}')
    if role not in mapping:
        raise ValueError(f'Unknown model role: {role}')
    return mapping[role]


def model_family_kind(role: str) -> str:
    """Return 'pf' or 'b1pp' for a canonical role."""
    if role.startswith('pf_'):
        return 'pf'
    if role.startswith('b1pp_'):
        return 'b1pp'
    raise ValueError(f'Cannot infer family kind for role={role}')
