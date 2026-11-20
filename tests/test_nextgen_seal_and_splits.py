"""Next-gen anti-seal-leakage, family-disjoint split, and inference truth-isolation tests."""

from __future__ import annotations

from functools import lru_cache
import importlib
import json
import os
from pathlib import Path
import pkgutil
import re
import sys
import tempfile
import unittest

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn

from analysis.flavour.v3.common.datasets import V3EvalDataset, V3ObservationDataset, V3TrainDataset
from analysis.flavour.v3.common.domains import compute_observable_containment, detector_bounds
from analysis.flavour.v3.v1_b1pp.models import ZAlignedStereoNet
from analysis.flavour.v3.v1_b1pp.representation import (
    collate_b1pp,
    extract_b1pp_item,
    static_plane_layout,
)
from analysis.flavour.v3.v2_pinpointformer.pixel_tokenizer import collate_pf, extract_pf_item
from analysis.flavour.v3.v2_pinpointformer.train import move_batch_to_device
from analysis.flavour.v3.v3_multitask.models import PinpointFormerMultitask
from analysis.flavour.v3.v4_tau.models import PinpointFormerV4


SEALED_SPLIT_NAMES = (
    'Cal-A',
    'Cut-B',
    'Confirm',
    'confirm200k',
    'cal_a',
    'cut_b',
    'sealed',
    'test',
    'calibration',
)

_EXACT_SEALED_SPLITS = {
    'cal_a',
    'cal_a40k',
    'cut_b',
    'cut_b160k',
    'confirm',
    'confirm200k',
    'sealed',
    'test',
    'calibration',
    'sealed_truth_meta',
}

_SEALED_PATTERN = re.compile(
    r'(?:^|[/\\_\-\s])(cal[-_]?a(?:40k)?|cut[-_]?b(?:160k)?|confirm(?:\d+k?)?|sealed_truth_meta)(?:$|[/\\_\-\s\.])'
    r'|(?:^|[/\\])sealed(?:[/\\]|$)'
    r'|(?:^|[/\\_\-])(?:input_sys_confirm|(?:calibration|test)_catalog)',
    re.IGNORECASE,
)

_SEALED_SHARD_RE = re.compile(r'(?:^|[/\\])shards[/\\](\d+)(?:[/\\]|$)', re.IGNORECASE)
_SEALED_GENIE_CHUNK_RE = re.compile(r'(?:^|[^0-9])1008[1-4][_/\\-]+0*(1[6-9]|2[0-8])(?:$|[^0-9])')

INTERNAL_SPLIT_SPEC_PATH = Path(
    '/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/forensics/internal_split_spec.json'
)


def is_sealed_split_or_path(value: str | Path) -> bool:
    """Return True if a split name or filesystem path references any sealed partition."""
    text = os.fspath(value).strip()
    low = text.lower().replace('-', '_')
    if low in _EXACT_SEALED_SPLITS:
        return True
    if _SEALED_PATTERN.search(text):
        return True
    if any(tok in low for tok in ('cal_a', 'cut_b', 'confirm200k', 'sealed_truth_meta', 'input_sys_confirm')):
        return True
    for m in _SEALED_SHARD_RE.finditer(text):
        if int(m.group(1)) >= 100:
            return True
    if _SEALED_GENIE_CHUNK_RE.search(text):
        return True
    return False


def assert_unsealed_split_or_path(value: str | Path, *, context: str = 'nextgen') -> None:
    """Raise ValueError immediately if a sealed split name or path is passed."""
    if is_sealed_split_or_path(value):
        raise ValueError(f'Sealed split or path is forbidden in {context}: {value!r}')


def assert_family_disjoint_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    family_cols: tuple[str, ...] = ('family_hi', 'family_lo'),
    extra_group_cols: tuple[str, ...] = ('source_tag',),
    key_cols: tuple[str, ...] = ('run', 'chunk', 'row'),
) -> None:
    """Assert zero overlap of event families, source groups, or row keys between train and val."""
    if not set(family_cols).issubset(train_df.columns) or not set(family_cols).issubset(val_df.columns):
        raise KeyError(f'Missing family columns {family_cols} in train/val DataFrames')

    tr_fam = set(map(tuple, train_df[list(family_cols)].to_numpy().tolist()))
    va_fam = set(map(tuple, val_df[list(family_cols)].to_numpy().tolist()))
    shared_fam = tr_fam & va_fam
    if shared_fam:
        raise AssertionError(
            f'Event-family leakage across internal train/val: {len(shared_fam)} shared families'
        )

    for col in extra_group_cols:
        if col in train_df.columns and col in val_df.columns:
            shared_grp = set(train_df[col].tolist()) & set(val_df[col].tolist())
            if shared_grp:
                raise AssertionError(
                    f'Group column {col!r} leaks across internal train/val: {len(shared_grp)} shared'
                )

    if set(key_cols).issubset(train_df.columns) and set(key_cols).issubset(val_df.columns):
        tr_keys = set(map(tuple, train_df[list(key_cols)].to_numpy().tolist()))
        va_keys = set(map(tuple, val_df[list(key_cols)].to_numpy().tolist()))
        shared_keys = tr_keys & va_keys
        if shared_keys:
            raise AssertionError(f'Duplicate event keys across train/val: {len(shared_keys)} shared')


def _clone_and_mutate_truth_fields(batch: dict, dev: torch.device | None = None) -> dict:
    """Deep-clone tensor/array batch and corrupt all truth/identity fields."""
    mutated: dict = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            mutated[k] = v.clone()
            if dev is None:
                dev = v.device
        elif isinstance(v, np.ndarray):
            mutated[k] = v.copy()
        else:
            mutated[k] = v

    if dev is None:
        dev = torch.device('cpu')

    bsz = 1
    for v in mutated.values():
        if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] > 0:
            bsz = int(v.shape[0])
            break

    # Mutate any existing truth fields AND inject canonical truth keys if absent.
    mutated['trackID'] = torch.full((bsz, 16), 999999, dtype=torch.long, device=dev)
    mutated['vertex_xyz_mm'] = torch.full((bsz, 3), -1234.5, dtype=torch.float32, device=dev)
    mutated['vertex_valid'] = torch.ones(bsz, dtype=torch.long, device=dev)
    mutated['vis_kinematics'] = torch.full((bsz, 4), 42.0, dtype=torch.float32, device=dev)
    mutated['kin_valid'] = torch.ones(bsz, dtype=torch.long, device=dev)
    mutated['initX'] = torch.full((bsz,), 77.7, dtype=torch.float32, device=dev)
    mutated['initY'] = torch.full((bsz,), -88.8, dtype=torch.float32, device=dev)
    mutated['initZ'] = torch.full((bsz,), -2500.0, dtype=torch.float32, device=dev)
    mutated['initE'] = torch.full((bsz,), 999.0, dtype=torch.float32, device=dev)
    mutated['Ev_GeV'] = torch.full((bsz,), 999.0, dtype=torch.float32, device=dev)
    mutated['label'] = torch.full((bsz,), 3, dtype=torch.long, device=dev)
    mutated['label_7'] = torch.full((bsz,), 6, dtype=torch.long, device=dev)
    mutated['charm_flag'] = torch.ones(bsz, dtype=torch.long, device=dev)
    mutated['charm_valid'] = torch.ones(bsz, dtype=torch.long, device=dev)
    mutated['tau_branch_valid'] = torch.ones(bsz, dtype=torch.long, device=dev)

    if 'pix_mask' in mutated and torch.is_tensor(mutated['pix_mask']):
        n_cl = int(mutated['pix_mask'].shape[1])
        mutated['pix_origin_target'] = torch.full((bsz, n_cl), 4, dtype=torch.long, device=dev)
        mutated['pix_ancestry_target'] = torch.full((bsz, n_cl, 2), 99, dtype=torch.long, device=dev)
        mutated['pix_rel_target'] = torch.full((bsz, n_cl, 3), 0.333, dtype=torch.float32, device=dev)
        mutated['pix_rel_valid'] = torch.ones((bsz, n_cl), dtype=torch.bool, device=dev)
    return mutated


def _extract_logits(out: torch.Tensor | dict) -> torch.Tensor:
    if torch.is_tensor(out):
        return out
    if isinstance(out, dict):
        for key in ('logits', 'logits_4', 'logits_7', 'cls_logits'):
            if key in out and torch.is_tensor(out[key]):
                return out[key]
    raise KeyError('Model output must be a Tensor or a dict containing logits')


def assert_zero_truth_leakage_at_inference(
    model: nn.Module,
    batch: dict,
    *,
    atol: float = 1e-7,
) -> float:
    """Assert model inference logits are bit-identical (within atol) when truth fields are mutated."""
    model.eval()
    clean_batch: dict = {}
    for k, v in batch.items():
        clean_batch[k] = v.clone() if torch.is_tensor(v) else v

    with torch.inference_mode():
        raw_logits = _extract_logits(model(clean_batch))
        dev = raw_logits.device
        ref_logits = raw_logits.detach().float()
    bsz = int(ref_logits.shape[0])

    # Compare clean_batch vs zeroed truth fields vs heavily mutated truth fields.
    base_truth = dict(clean_batch)
    base_truth.setdefault('trackID', torch.zeros((bsz, 16), dtype=torch.long, device=dev))
    base_truth.setdefault('vertex_xyz_mm', torch.zeros((bsz, 3), dtype=torch.float32, device=dev))
    base_truth.setdefault('vertex_valid', torch.zeros(bsz, dtype=torch.long, device=dev))
    base_truth.setdefault('vis_kinematics', torch.zeros((bsz, 4), dtype=torch.float32, device=dev))
    base_truth.setdefault('kin_valid', torch.zeros(bsz, dtype=torch.long, device=dev))
    base_truth.setdefault('label', torch.zeros(bsz, dtype=torch.long, device=dev))
    base_truth.setdefault('label_7', torch.zeros(bsz, dtype=torch.long, device=dev))
    mutated_batch = _clone_and_mutate_truth_fields(clean_batch, dev=dev)

    with torch.inference_mode():
        logits_base = _extract_logits(model(base_truth)).detach().float()
        logits_mut = _extract_logits(model(mutated_batch)).detach().float()

    max_diff_clean = float((ref_logits - logits_base).abs().max().item())
    max_diff_mut = float((ref_logits - logits_mut).abs().max().item())
    max_diff_pair = float((logits_base - logits_mut).abs().max().item())
    max_diff = max(max_diff_clean, max_diff_mut, max_diff_pair)
    if not np.isfinite(max_diff) or max_diff > atol:
        raise AssertionError(
            f'Inference logits changed when mutating truth fields: max_abs_diff={max_diff:.3e} > {atol:.1e}'
        )
    return max_diff


@lru_cache(maxsize=1)
def build_minimal_geometry() -> dict:
    """Construct compact Pinpoint_G4-consistent detector geometry dictionary (no EOS required)."""
    w_z: list[float] = []
    sc_z: list[float] = []
    p_z: list[float] = []
    z = -3000.0
    for _ in range(6):
        p_z.append(z + 8.005)
        w_z.extend([z + 15.51, z + 25.51])
        sc_z.append(z + 20.51)
        z += 30.01
    for f in range(7):
        for j in range(6):
            w_z.extend([z + 2.5 + j * 20.0, z + 12.5 + j * 20.0])
            sc_z.extend([z + 7.5 + j * 20.0, z + 17.5 + j * 20.0])
        z += 120.0
        if f < 6:
            p_z.append(z + 8.005)
            w_z.extend([z + 15.51, z + 25.51])
            sc_z.append(z + 20.51)
            z += 30.01
    p_z.extend([z + 65.005, z + 95.015, z + 125.025])
    return {
        'detector_width': 266.0,
        'detector_height': 196.0,
        'silicon_thickness': 10.0,
        'tungsten_thickness': 5.0,
        'nFortuneBlocks': 7,
        'numScintLayers': 6,
        'num_scint_panels_layer': 2,
        'nPinpointBlocks': 6,
        'scint_detector_width': 420.0,
        'scint_detector_height': 420.0,
        'scint_thickness': 5.0,
        'scint_bar_width': 10.0,
        'scint_bar_height': 10.0,
        'nLayers': 15,
        'pixel_Xpos': (-133.0 + (np.arange(12788) + 0.5) * 0.0208).tolist(),
        'pixel_Ypos': (-110.0 + (np.arange(8596) + 0.5) * 0.0228).tolist(),
        'pixel_Zpos': p_z,
        'tungsten_Zpos': w_z,
        'scint_Zpos': sc_z,
        'scint_bar_Xcenter': (77.0 - 210.0 + (np.arange(40) + 0.5) * 10.5).tolist(),
        'scint_bar_Ycenter': (-12.0 - 210.0 + (np.arange(40) + 0.5) * 10.5).tolist(),
    }


def _make_synthetic_observation(geometry: dict, bounds: dict, shift_col: int = 0) -> dict:
    """Create one realistic synthetic Pinpoint observation matching observations.parquet schema."""
    nx, ny = int(bounds['nx']), int(bounds['ny'])
    px = geometry['pixel_Xpos']
    py = geometry['pixel_Ypos']
    pz = geometry['pixel_Zpos']

    cids, xyzs = [], []
    # Start in W0 -> pixel hits on planes 1, 2, 3, 4 (plane 0 quiet so upstream_veto=False)
    for pl in (1, 2, 3, 4):
        for prong, (dc_base, dr_base) in enumerate(((0, 0), (12, -8), (-10, 14))):
            col = 6200 + shift_col + pl * (2 + prong) + dc_base
            row = 4300 + pl * (1 - prong) + dr_base
            for dc in (0, 1):
                c = col + dc
                cids.append((pl * ny + row) * nx + c)
                xyzs.append([px[c], py[row], pz[pl]])

    vp0 = int(bounds['vert_panels'][1])
    vp1 = int(bounds['vert_panels'][6])
    hp0 = int(bounds['horiz_panels'][0])
    bp = np.array([vp0, vp1, hp0], dtype=np.int16)
    bh = np.array([0, 0, 1], dtype=np.int8)
    bi = np.array([15, 16, 20], dtype=np.int16)
    bb = np.zeros((3, 5), dtype=np.float32)
    for k, (p, h, b) in enumerate(zip(bp, bh, bi)):
        if h == 0:
            xc = geometry['scint_bar_Xcenter'][b]
            bb[k] = [xc - 5.25, xc + 5.25, -222.0, 198.0, geometry['scint_Zpos'][p]]
        else:
            yc = geometry['scint_bar_Ycenter'][b]
            bb[k] = [-133.0, 287.0, yc - 5.25, yc + 5.25, geometry['scint_Zpos'][p]]

    return {
        'pixel': {
            'channel_id': np.asarray(cids, dtype=np.int64),
            'xyz_mm': np.asarray(xyzs, dtype=np.float32),
        },
        'bar': {
            'channel_id': np.array([1, 2, 3], dtype=np.int64),
            'layer': bp.copy(),
            'panel': bp,
            'horizontal': bh,
            'bar': bi,
            'bounds_mm': bb,
            'pe_proxy': np.array([320, 640, 480], dtype=np.int64),
            'hit_time_ns': np.array([3.2, 4.1, 4.2], dtype=np.float32),
        },
        'tracker_hypothesis': {
            'station': np.array([0, 1, 2], dtype=np.int8),
            'xy_mm': np.array([[0.0, -10.0], [0.2, -11.008], [0.4, -10.0]], dtype=np.float32),
            'sigma_xy_mm': np.full((3, 2), 0.1, dtype=np.float32),
        },
    }


class TestNextgenSealAndSplits(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        cls.geometry = build_minimal_geometry()
        cls.bounds = detector_bounds(cls.geometry)

    def test_sealed_split_names_and_paths_rejected(self) -> None:
        """Sealed split names/paths must raise immediate errors in guard and training helpers."""
        # Verify IPT z-positions match Pinpoint_G4 / clean_sim/geometry.json to < 1e-9 mm
        np.testing.assert_allclose(
            self.geometry['pixel_Zpos'][-3:],
            [-1734.875, -1704.865, -1674.855],
            atol=1e-9,
        )

        for tok in SEALED_SPLIT_NAMES:
            with self.assertRaises(ValueError, msg=f'Guard failed to reject {tok!r}'):
                assert_unsealed_split_or_path(tok)

        sealed_paths = (
            '/eos/some/dir/Cal-A/observations.parquet',
            '/eos/some/dir/Cut-B/observations.parquet',
            '/eos/some/dir/Confirm/observations.parquet',
            '/eos/some/dir/confirm200k/observations.parquet',
            '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/clean_sim/data/shards/100/observations.parquet',
            '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/clean_sim/data/shards/149/observations.parquet',
            '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/sealed/rows.parquet',
            'calibration_catalog.parquet',
            'test_catalog.parquet',
            'input_sys_confirm1200.parquet',
            '/eos/project-f/fasersim-bonn/public/pinpoint/data/FaserMC-MC24_Genie_all_10081_00016.root',
            '/eos/project-f/fasersim-bonn/public/pinpoint/data/FaserMC-MC24_Genie_all_10084_00028.root',
        )
        for sp in sealed_paths:
            with self.assertRaises(ValueError, msg=f'Guard failed to reject sealed path {sp!r}'):
                assert_unsealed_split_or_path(sp)

        # V3TrainDataset strictly rejects every sealed split name before touching disk.
        for tok in SEALED_SPLIT_NAMES:
            with self.assertRaises(ValueError):
                V3TrainDataset(obs_root='/nonexistent/obs', sidecar_root=None, split=tok)

        # Positive controls: allowed internal/dev splits, /tests/ paths, shards 000..099, and chunks 000..015
        allowed_inputs = (
            'train',
            'validation',
            'internal_train',
            'internal_val',
            '/tmp/dev_split',
            '/tmp/test_nextgen_scratch/observations.parquet',
            '/repo/tests/test_nextgen_seal_and_splits.py',
            '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/clean_sim/data/shards/000/observations.parquet',
            '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/clean_sim/data/shards/099/observations.parquet',
            '/eos/project-f/fasersim-bonn/public/pinpoint/data/FaserMC-MC24_Genie_all_10081_00000.root',
            '/eos/project-f/fasersim-bonn/public/pinpoint/data/FaserMC-MC24_Genie_all_10084_00015.root',
        )
        for allowed in allowed_inputs:
            assert_unsealed_split_or_path(allowed)

        # If analysis.flavour.nextgen exists, probe any exported split/guard/loader helpers
        try:
            nextgen_pkg = importlib.import_module('analysis.flavour.nextgen')
        except ImportError:
            nextgen_pkg = None

        if nextgen_pkg is not None and hasattr(nextgen_pkg, '__path__'):
            for mod_info in pkgutil.walk_packages(nextgen_pkg.__path__, prefix='analysis.flavour.nextgen.'):
                mod = importlib.import_module(mod_info.name)
                for attr_name in ('assert_unsealed_split', 'validate_split_name', 'require_unsealed_split'):
                    fn = getattr(mod, attr_name, None)
                    if callable(fn):
                        for tok in SEALED_SPLIT_NAMES:
                            with self.assertRaises((ValueError, PermissionError, RuntimeError)):
                                fn(tok)

    def test_split_shards_allowlist_blocks_sealed_io(self) -> None:
        """Training, validation, and observation loaders must never open or stat sealed shards."""
        with tempfile.TemporaryDirectory(prefix='nextgen-seal-') as td:
            root = Path(td)
            (root / 'geometry.json').write_text(json.dumps(self.geometry))
            (root / 'plan.json').write_text(json.dumps({'geometry': str(root / 'geometry.json')}))

            shard_specs = (
                ('000', 'train'),
                ('001', 'validation'),
                ('100', 'cal_a'),
                ('101', 'cut_b'),
                ('102', 'confirm'),
            )
            for sname, sp_val in shard_specs:
                sdir = root / 'shards' / sname
                sdir.mkdir(parents=True)
                obs_tbl = pa.Table.from_pylist([{
                    'pixel': {'channel_id': [], 'xyz_mm': []},
                    'bar': {
                        'panel': [], 'bar': [], 'horizontal': [], 'channel_id': [],
                        'layer': [], 'bounds_mm': [], 'pe_proxy': [], 'hit_time_ns': [],
                    },
                    'tracker_hypothesis': {'station': [], 'xy_mm': [], 'sigma_xy_mm': []},
                }])
                pq.write_table(obs_tbl, sdir / 'observations.parquet')
                pq.write_table(pa.Table.from_pylist([{'label': 0}]), sdir / 'targets.parquet')
                pd.DataFrame([{
                    'run': 32500, 'chunk': int(sname), 'row': 0,
                    'family_hi': np.uint64(int(sname) + 1), 'family_lo': np.uint64(1),
                    'split': sp_val, 'label': 0,
                }]).to_parquet(sdir / 'provenance.parquet', index=False)

            (root / 'split_shards.json').write_text(json.dumps({
                'train': ['000'],
                'validation': ['001'],
                'cal_a': ['100'],
                'cut_b': ['101'],
                'confirm': ['102'],
            }))

            sealed_dirs = [str(root / 'shards' / s) for s in ('100', '101', '102')]
            sealed_hits: list[tuple[str, str]] = []
            active = [True]

            def _audit(event: str, args: tuple) -> None:
                if not active[0]:
                    return
                if event in ('open', 'os.scandir', 'os.listdir', 'os.stat'):
                    for arg in args:
                        if isinstance(arg, (str, bytes, os.PathLike)):
                            s = os.fsdecode(arg)
                            if any(sd in s for sd in sealed_dirs):
                                sealed_hits.append((event, s))

            sys.addaudithook(_audit)
            for sd in sealed_dirs:
                os.chmod(sd, 0o000)
            try:
                ds_tr = V3EvalDataset(root, split='train')
                ds_va = V3EvalDataset(root, split='validation')
                ds_obs = V3ObservationDataset(root / 'shards' / '001' / 'observations.parquet', root / 'geometry.json')
                self.assertEqual(len(ds_tr), 1)
                self.assertEqual(len(ds_va), 1)
                self.assertEqual(len(ds_obs), 1)
                ds_tr.native.close()
                ds_va.native.close()
            finally:
                active[0] = False
                for sd in sealed_dirs:
                    os.chmod(sd, 0o755)

            self.assertEqual(sealed_hits, [], f'Sealed shard accessed: {sealed_hits}')

    def test_event_family_group_disjointness(self) -> None:
        """Internal train/val splits must have zero shared event families or source groups."""
        train_df = pd.DataFrame({
            'run': [10081, 10081, 10081, 10082],
            'chunk': [0, 0, 1, 2],
            'row': [0, 1, 0, 0],
            'family_hi': np.array([10, 10, 11, 12], dtype=np.uint64),
            'family_lo': np.array([1, 1, 2, 3], dtype=np.uint64),
            'source_tag': ['src_A', 'src_A', 'src_B', 'src_C'],
        })
        val_df = pd.DataFrame({
            'run': [10083, 10083],
            'chunk': [12, 13],
            'row': [0, 5],
            'family_hi': np.array([20, 21], dtype=np.uint64),
            'family_lo': np.array([1, 2], dtype=np.uint64),
            'source_tag': ['src_D', 'src_E'],
        })
        assert_family_disjoint_splits(train_df, val_df)

        # Negative test: leaking a single event family across train/val must raise
        leaky_val = val_df.copy()
        leaky_val.loc[0, 'family_hi'] = np.uint64(10)
        leaky_val.loc[0, 'family_lo'] = np.uint64(1)
        with self.assertRaises(AssertionError):
            assert_family_disjoint_splits(train_df, leaky_val)

        # Negative test: leaking source_tag must raise
        leaky_src = val_df.copy()
        leaky_src.loc[1, 'source_tag'] = 'src_B'
        with self.assertRaises(AssertionError):
            assert_family_disjoint_splits(train_df, leaky_src)

        # Optional check on forensics internal_split_spec.json if generated on disk
        if INTERNAL_SPLIT_SPEC_PATH.is_file():
            spec = json.loads(INTERNAL_SPLIT_SPEC_PATH.read_text())
            auth_shards = spec.get('source_dataset', {}).get('authorized_train_shards', [])
            self.assertTrue(all(int(s) < 80 for s in auth_shards), f'Unauthorized shard in internal_split_spec: {auth_shards}')
            anti = spec.get('anti_leakage_verification', {})
            if 'sealed_shards_touched' in anti:
                self.assertFalse(anti['sealed_shards_touched'])
            for k, v in anti.items():
                if k.startswith('family_overlap_'):
                    self.assertEqual(v, 0, f'Non-zero family overlap in {k}: {v}')
            for split_name, split_meta in spec.get('splits', {}).items():
                assert_unsealed_split_or_path(split_name, context='internal_split_spec')
                if isinstance(split_meta, dict) and 'path' in split_meta:
                    assert_unsealed_split_or_path(split_meta['path'], context='internal_split_spec')

    def test_inference_zero_truth_leakage(self) -> None:
        """Mutating trackID, truth vertex, or truth kinematics leaves logits identical within 1e-7."""
        torch.manual_seed(42)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        obs0 = _make_synthetic_observation(self.geometry, self.bounds, shift_col=0)
        obs1 = _make_synthetic_observation(self.geometry, self.bounds, shift_col=35)
        cfg = {'pe_only': True, 'task': 'T4', 'dim': 32, 'heads': 2, 'n_local_layers': 1, 'n_global_layers': 1, 'drop_path': 0.0}

        # 1. B1++ (ZAlignedStereoNet)
        b1_items = [
            extract_b1pp_item(i, ob, i, compute_observable_containment(ob, self.geometry, self.bounds, cfg), self.geometry, self.bounds, cfg)
            for i, ob in enumerate((obs0, obs1))
        ]
        b1_batch = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in collate_b1pp(b1_items).items()
        }
        b1_model = ZAlignedStereoNet(static_plane_layout(self.geometry), {**cfg, 'n_long_layers': 2, 'channel_dropout': 0.0}).to(device)
        diff_b1 = assert_zero_truth_leakage_at_inference(b1_model, b1_batch, atol=1e-7)
        self.assertLessEqual(diff_b1, 1e-7)

        # 2. PinpointFormerMultitask & PinpointFormerV4
        pf_items = [
            extract_pf_item(i, ob, i, compute_observable_containment(ob, self.geometry, self.bounds, cfg), self.geometry, self.bounds, cfg)
            for i, ob in enumerate((obs0, obs1))
        ]
        pf_batch = move_batch_to_device(collate_pf(pf_items), device)
        pf_cfg = {**cfg, 'enable_dense_gn2': True, 'enable_micro_vertex': True}
        pf_model = PinpointFormerMultitask(self.geometry, pf_cfg).to(device)
        diff_pf = assert_zero_truth_leakage_at_inference(pf_model, pf_batch, atol=1e-7)
        self.assertLessEqual(diff_pf, 1e-7)

        v4_cfg = {**pf_cfg, 'enable_nomad': True, 'enable_pauc': False}
        v4_model = PinpointFormerV4(self.geometry, v4_cfg).to(device)
        diff_v4 = assert_zero_truth_leakage_at_inference(v4_model, pf_batch, atol=1e-7)
        self.assertLessEqual(diff_v4, 1e-7)

        # 3. Negative control: deliberately leaky model must be caught immediately
        class _LeakyModel(nn.Module):
            def forward(self, batch: dict) -> torch.Tensor:
                vtx = batch.get('vertex_xyz_mm')
                if vtx is None:
                    return batch['c_obs'][:, :4]
                return batch['c_obs'][:, :4] + 0.01 * vtx.sum(dim=-1, keepdim=True)

        with self.assertRaises(AssertionError):
            assert_zero_truth_leakage_at_inference(_LeakyModel().to(device), pf_batch, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
