"""Zero-exception seal guard & Gate-A anti-leakage verification utilities for nextgen."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import sys
import threading
from typing import Any, Callable

import torch


class SealViolationError(PermissionError):
    """Raised immediately upon any attempted access to sealed splits, shards, or metadata."""


FORBIDDEN_SPLIT_TOKENS: frozenset[str] = frozenset({
    'cal_a',
    'cala',
    'cal-a',
    'cut_b',
    'cutb',
    'cut-b',
    'confirm',
    'confirm200k',
    'sealed',
    'sealed_truth_meta',
    'test',
    'calibration',
})

# Clean v3 sealed shards 100..149 (or any >=100) and sealed GENIE source chunks 016..028 (runs 10081..10084)
_SEALED_SHARD_RE = re.compile(r'(?:^|/)shards/(\d+)(?:/|$)')
_SEALED_GENIE_RE = re.compile(r'(?:^|[^0-9])1008[1-4][_/\\-]+0*(?:1[6-9]|2[0-8])(?:$|[^0-9])')
_SEALED_PATH_SEGMENTS = (
    '/sealed_truth_meta',
    '/input_sys_confirm',
    '/w3-12',
    '/unsealed',
    '/sealed/',
    'cal_a',
    'cal-a',
    'cut_b',
    'cut-b',
    'confirm200k',
    'calibration_catalog.parquet',
    'test_catalog.parquet',
)

ALLOWED_NEXTGEN_SPLITS: frozenset[str] = frozenset({
    'train',
    'validation',
    'internal_train',
    'internal_val',
    'internal_screen',
    'tune_screen',
    'tune6k',
    'ood',
})

TRUTH_BATCH_KEYS: tuple[str, ...] = (
    'label',
    'label_4',
    'label_7',
    'tau_branch',
    'tau_branch_valid',
    'charm_flag',
    'has_charm',
    'charm_valid',
    'vertex_xyz_mm',
    'vertex_x_mm',
    'vertex_y_mm',
    'vertex_z_mm',
    'vertex_valid',
    'vis_kinematics',
    'vis_pz_log1p',
    'vis_pt_log1p',
    'vis_cos_dphi',
    'vis_sin_dphi',
    'kin_valid',
    'tau_decay_L3d_mm',
    'tau_decay_dz_mm',
    'charm_decay_L3d_mm',
    'sample_weight',
    'geom_weight',
    'pix_origin_target',
    'pix_ancestry_target',
    'pix_rel_target',
    'pix_rel_valid',
    'bar_rel_target',
    'trackID',
    'Ev_GeV',
    'energy_nu_GeV',
    'initX',
    'initY',
    'initZ',
    'initE',
    'family_hi',
    'family_lo',
)


def assert_unsealed_split(split_name: str, *, stage: int = 1) -> str:
    """Validate requested split name; raise SealViolationError on any sealed identifier.

    In Stage 1 (`stage == 1`), `validation` / `tune6k` are also blocked to prevent
    adaptive selection bias on Tune6k prior to Stage II freeze.
    """
    if not isinstance(split_name, str) or not split_name.strip():
        raise ValueError(f'Invalid empty split_name: {split_name!r}')
    norm = split_name.strip().lower()
    if norm in FORBIDDEN_SPLIT_TOKENS:
        raise SealViolationError(f'Forbidden sealed split requested: {split_name!r}')
    for tok in ('cal_a', 'cal-a', 'cala', 'cut_b', 'cut-b', 'cutb', 'confirm', 'sealed'):
        if tok in norm:
            raise SealViolationError(f'Forbidden sealed token {tok!r} in split {split_name!r}')
    if norm not in ALLOWED_NEXTGEN_SPLITS:
        raise ValueError(f'Unrecognized split_name {split_name!r}; allowed={sorted(ALLOWED_NEXTGEN_SPLITS)}')
    if stage == 1 and norm in ('validation', 'tune6k', 'tune_screen'):
        raise SealViolationError(
            f'Stage-1 exploration forbids accessing {split_name!r}; use internal_train / internal_val only.'
        )
    return norm


def assert_unsealed_path(path: str | Path) -> Path:
    """Check path string against Zero-Exception Seal Registry without calling os.stat()."""
    raw = str(path).replace('\\', '/')
    low = raw.lower()
    for m_shard in _SEALED_SHARD_RE.finditer(low):
        if int(m_shard.group(1)) >= 100:
            raise SealViolationError(f'Access to sealed shard {m_shard.group(1)} is strictly forbidden: {raw}')
    if _SEALED_GENIE_RE.search(low) is not None:
        raise SealViolationError(f'Access to sealed GENIE source chunk is strictly forbidden: {raw}')
    for seg in _SEALED_PATH_SEGMENTS:
        if seg in low:
            raise SealViolationError(f'Access to sealed path pattern {seg!r} is strictly forbidden: {raw}')
    # Also catch `/confirm` shard/prediction directories while allowing benign words like `confirmation` code only if not in results
    if re.search(r'(?:^|/)(?:confirm|test_split)(?:[/_.]|$)', low):
        raise SealViolationError(f'Access to sealed confirm/test path is strictly forbidden: {raw}')
    return Path(path)


def load_authorized_split_shards(data_root: str | Path, split: str = 'train', *, stage: int = 1) -> list[Path]:
    """Resolve shard paths strictly through `split_shards.json` allowlist (never `iterdir` over 100..149)."""
    norm_split = assert_unsealed_split(split, stage=stage)
    root = assert_unsealed_path(data_root)
    allow_file = root / 'split_shards.json'
    if not allow_file.is_file():
        raise FileNotFoundError(f'Mandatory split_shards.json allowlist missing at {allow_file}')
    mapping = json.loads(allow_file.read_text())
    key = 'train' if norm_split.startswith('internal_') else norm_split
    if key not in mapping:
        raise KeyError(f'Split {key!r} absent from split_shards.json keys {sorted(mapping.keys())}')
    out: list[Path] = []
    for name in mapping[key]:
        shard_id = str(name)
        if shard_id.isdigit() and int(shard_id) >= 100:
            raise SealViolationError(f'Sealed shard index {shard_id} encountered in allowlist for {key!r}')
        sp = assert_unsealed_path(root / 'shards' / shard_id)
        out.append(sp)
    return out


_AUDIT_LOCK = threading.Lock()
_ACTIVE_GUARDS: list['SealAuditGuard'] = []
_HOOK_INSTALLED = False


def _global_seal_audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if event not in ('open', 'os.stat', 'os.scandir', 'os.listdir'):
        return
    if not _ACTIVE_GUARDS or not args:
        return
    target = args[0]
    if isinstance(target, int):
        return
    try:
        path_str = os.fspath(target) if hasattr(target, '__fspath__') else str(target)
    except Exception:
        return
    if isinstance(path_str, bytes):
        path_str = path_str.decode('utf-8', errors='ignore')
    try:
        assert_unsealed_path(path_str)
    except SealViolationError as exc:
        for g in list(_ACTIVE_GUARDS):
            g.violations.append(f'{event}:{path_str}')
        raise exc


import os  # noqa: E402


class SealAuditGuard:
    """Runtime `sys.addaudithook` context manager recording & blocking any sealed filesystem syscall."""

    def __init__(self) -> None:
        self.violations: list[str] = []

    def __enter__(self) -> 'SealAuditGuard':
        global _HOOK_INSTALLED
        with _AUDIT_LOCK:
            if not _HOOK_INSTALLED:
                sys.addaudithook(_global_seal_audit_hook)
                _HOOK_INSTALLED = True
            _ACTIVE_GUARDS.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        with _AUDIT_LOCK:
            if self in _ACTIVE_GUARDS:
                _ACTIVE_GUARDS.remove(self)
        return False


@torch.no_grad()
def verify_inference_truth_invariance(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    forward_fn: Callable[[torch.nn.Module, dict[str, Any]], torch.Tensor] | None = None,
    atol: float = 1.0e-7,
) -> dict[str, float | bool]:
    """Gate-A check: mutating or deleting all truth/metadata keys must leave inference logits unchanged."""
    was_training = model.training
    model.eval()
    try:
        run_fwd = forward_fn or (lambda m, b: m.predict_logits4(b) if hasattr(m, 'predict_logits4') else m(b))
        clean_batch = {k: v for k, v in batch.items() if k not in TRUTH_BATCH_KEYS}
        ref_out = run_fwd(model, clean_batch)
        if isinstance(ref_out, dict):
            ref_logits = ref_out['probs4'] if 'probs4' in ref_out else ref_out['logits']
        else:
            ref_logits = ref_out
        ref_f = ref_logits.detach().float()

        mutated = copy.copy(batch)
        bsz = int(ref_f.shape[0])
        device = ref_f.device
        mutated['label'] = torch.full((bsz,), 3, dtype=torch.long, device=device)
        mutated['label_4'] = torch.full((bsz,), 3, dtype=torch.long, device=device)
        mutated['label_7'] = torch.full((bsz,), 6, dtype=torch.long, device=device)
        mutated['tau_branch'] = ['unknown'] * bsz
        mutated['tau_branch_valid'] = torch.zeros((bsz,), dtype=torch.long, device=device)
        mutated['charm_flag'] = torch.ones((bsz,), dtype=torch.long, device=device)
        mutated['has_charm'] = torch.ones((bsz,), dtype=torch.bool, device=device)
        mutated['charm_valid'] = torch.ones((bsz,), dtype=torch.long, device=device)
        mutated['vertex_xyz_mm'] = torch.full((bsz, 3), 999.0, dtype=torch.float32, device=device)
        mutated['vertex_x_mm'] = torch.full((bsz,), 999.0, dtype=torch.float32, device=device)
        mutated['vertex_y_mm'] = torch.full((bsz,), -999.0, dtype=torch.float32, device=device)
        mutated['vertex_z_mm'] = torch.full((bsz,), -1234.0, dtype=torch.float32, device=device)
        mutated['vertex_valid'] = torch.ones((bsz,), dtype=torch.long, device=device)
        mutated['vis_kinematics'] = torch.full((bsz, 4), -7.5, dtype=torch.float32, device=device)
        mutated['vis_pz_log1p'] = torch.full((bsz,), 9.9, dtype=torch.float32, device=device)
        mutated['vis_pt_log1p'] = torch.full((bsz,), 4.2, dtype=torch.float32, device=device)
        mutated['vis_cos_dphi'] = torch.full((bsz,), -1.0, dtype=torch.float32, device=device)
        mutated['vis_sin_dphi'] = torch.zeros((bsz,), dtype=torch.float32, device=device)
        mutated['kin_valid'] = torch.ones((bsz,), dtype=torch.long, device=device)
        mutated['tau_decay_L3d_mm'] = torch.full((bsz,), 55.5, dtype=torch.float32, device=device)
        mutated['tau_decay_dz_mm'] = torch.full((bsz,), 44.4, dtype=torch.float32, device=device)
        mutated['charm_decay_L3d_mm'] = torch.full((bsz,), 33.3, dtype=torch.float32, device=device)
        mutated['energy_nu_GeV'] = torch.full((bsz,), 4321.0, dtype=torch.float32, device=device)
        mutated['Ev_GeV'] = torch.full((bsz,), 4321.0, dtype=torch.float32, device=device)
        mutated['geom_weight'] = torch.full((bsz,), 7.7, dtype=torch.float32, device=device)
        mutated['trackID'] = torch.arange(bsz, dtype=torch.long, device=device) + 12345

        mut_out = run_fwd(model, mutated)
        mut_logits = (mut_out['probs4'] if 'probs4' in mut_out else mut_out['logits']) if isinstance(mut_out, dict) else mut_out
        max_diff = float((ref_f - mut_logits.detach().float()).abs().max().item())
        passed = max_diff <= atol
        if not passed:
            raise AssertionError(f'Gate-A truth leakage detected: max_abs_diff={max_diff:.3e} > {atol:.3e}')
        return {'passed': True, 'max_abs_diff': max_diff}
    finally:
        model.train(was_training)
