"""Mandatory pre-unsealing freeze construction and verification for W3-12."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from analysis.flavour.inventory import write_json
from analysis.flavour.v3.v4_tau.evaluate import _primary_tag_from_manifest
from analysis.flavour.v3.v5_confirm import (
    DEFAULT_V3_ROOT,
    ROLE_ORDER,
    SEEDS,
    experiment_dir_for,
    model_family_kind,
)


def resolve_primary_checkpoint(exp_dir: Path) -> tuple[str, Path]:
    """Select 'best.pt' vs 'ema.pt' solely from Tune-screen NLL in manifest.json."""
    tag = _primary_tag_from_manifest(exp_dir)
    if tag == 'best' and (exp_dir / 'best.pt').exists():
        return 'best', exp_dir / 'best.pt'
    if (exp_dir / 'ema.pt').exists():
        return 'ema', exp_dir / 'ema.pt'
    if (exp_dir / 'best.pt').exists():
        return 'best', exp_dir / 'best.pt'
    raise FileNotFoundError(f'Neither ema.pt nor best.pt exists in {exp_dir}')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def build_v5_freeze_manifest(
    v3_root: Path = DEFAULT_V3_ROOT,
    *,
    repo_root: Path | None = None,
    w3_11_summary_path: Path | None = None,
    pre_unsealing_git_commit: str | None = None,
) -> dict:
    """Assemble the complete immutable V5 analysis freeze specification."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[5]
    members = []
    for seed in SEEDS:
        for role in ROLE_ORDER:
            exp_dir = experiment_dir_for(v3_root, role, seed)
            manifest_path = exp_dir / 'manifest.json'
            if not manifest_path.exists():
                raise FileNotFoundError(f'Missing manifest for {(role, seed)}: {manifest_path}')
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('status') != 'complete':
                raise RuntimeError(f'Experiment {(role, seed)} is not complete: {manifest.get("status")}')
            if manifest.get('Cal_A_Cut_B_Confirm_payload_read') is not False:
                raise RuntimeError(f'Experiment {(role, seed)} violated sealed payload flag')
            tag, ckpt_path = resolve_primary_checkpoint(exp_dir)
            members.append({
                'member_id': f'{role}_s{seed}',
                'role': role,
                'seed': int(seed),
                'family': model_family_kind(role),
                'experiment_dir': str(exp_dir),
                'checkpoint_tag': tag,
                'checkpoint_path': str(ckpt_path),
                'checkpoint_sha256': sha256_file(ckpt_path),
                'config': manifest.get('config', {}),
                'tune_screen_selection_rule': 'lowest tune_screen_ema_NLL in manifest.json',
            })

    freeze = {
        'status': 'frozen',
        'version': 'v5_confirm_v1',
        'created_UTC': datetime.now(timezone.utc).isoformat(),
        'sealed_payloads_touched_at_freeze': False,
        'calibration_and_threshold_rules_frozen': True,
        'pre_unsealing_git_commit': pre_unsealing_git_commit,
        'seeds': list(SEEDS),
        'roles': list(ROLE_ORDER),
        'members': members,
        'ensemble_protocol': {
            'primary_deliverable': 'ens_v5_3seed_dual_regime',
            'seed_aggregation': 'equal-weight arithmetic mean of calibrated member probabilities in probability/log-probability space per role across seeds {17, 42, 2026}',
            'cross_family_calibration_split': 'Cal-A40k',
            'cross_family_blend_rule': 'per-domain L-BFGS-B convex blend weights w_d, inverse temperature inv_T_d, and zero-sum class bias b_d for active domains d in {1, 2, 3}; D_empty (d=0) fixed to uniform [0.25, 0.25, 0.25, 0.25]',
            'charm_suppression_split': 'Cut-B160k',
            'charm_suppression_grid': [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0],
            'threshold_locking_split': 'Cut-B160k',
            'strategy_i_rule': 'global balanced-non-tau quantile cut on S_tau at FPR in {0.01, 0.001}',
            'strategy_ii_rule': 'Neyman-Pearson per-domain FPR budget search over grid [(1,1,1),(1.5,1.0,0.6),(2.0,1.0,0.5),(3.0,1.0,0.3),(4.0,0.8,0.2)] at total non-tau FPR in {0.01, 0.001}',
            'template_bins_per_active_domain': 6,
            'template_quantiles': [0.0, 0.50, 0.80, 0.92, 0.97, 0.99, 1.0],
            'confirm_retuning_allowed': False,
        },
        'stress_protocol': {
            'conditions_source': '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s6_2026-11-14/conditions.json',
            'bar_hardware_map': '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s6_2026-11-14/audit/bar_hardware_map.json',
            'confirm_paired_per_class': 2000,
            'confirm_paired_total': 8000,
            'confirm_paired_seed': 20261117,
            'nominal_parity_max_prob_diff': 3e-3,
        },
        'w3_11_summary_path': str(w3_11_summary_path) if w3_11_summary_path is not None else None,
        'git_repository': str(repo_root),
    }
    return freeze


def save_v5_freeze_manifests(
    freeze: dict,
    *,
    eos_freeze_path: Path,
    repo_freeze_path: Path,
) -> None:
    write_json(eos_freeze_path, freeze)
    write_json(repo_freeze_path, freeze)


def require_v5_freeze(
    freeze_path: Path,
    *,
    repo_root: Path | None = None,
    verify_git_commit: bool = True,
    verify_checkpoints: bool = True,
) -> dict:
    """Block any sealed-split conversion or inference unless the V5 freeze is valid."""
    if not freeze_path.exists():
        raise FileNotFoundError(
            f'Pre-unsealing freeze manifest missing ({freeze_path}); sealed splits remain locked.'
        )
    data = json.loads(freeze_path.read_text())
    if data.get('status') != 'frozen':
        raise RuntimeError(f'V5 freeze status must be "frozen", got {data.get("status")!r}')
    if not data.get('calibration_and_threshold_rules_frozen'):
        raise RuntimeError('calibration_and_threshold_rules_frozen must be true before unsealing')
    commit = data.get('pre_unsealing_git_commit')
    if not commit or not isinstance(commit, str) or len(commit) < 7:
        raise RuntimeError('pre_unsealing_git_commit must be populated with a valid commit SHA')
    if len(data.get('members', [])) != len(SEEDS) * len(ROLE_ORDER):
        raise RuntimeError(
            f'Expected {len(SEEDS) * len(ROLE_ORDER)} frozen members, found {len(data.get("members", []))}'
        )
    if verify_checkpoints:
        for m in data['members']:
            cp = Path(m['checkpoint_path'])
            if not cp.is_file():
                raise FileNotFoundError(f'Frozen checkpoint missing: {cp}')
    if verify_git_commit:
        root = repo_root or Path(__file__).resolve().parents[5]
        if not (root / '.git').exists():
            eos_repo = Path('/eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis')
            if (eos_repo / '.git').exists():
                root = eos_repo
        res = subprocess.run(
            ['git', 'merge-base', '--is-ancestor', commit, 'HEAD'],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            raise RuntimeError(
                f'pre_unsealing_git_commit {commit} is not an ancestor of current HEAD in {root}'
            )
    return data
