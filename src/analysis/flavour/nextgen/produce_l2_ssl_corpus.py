"""Experiment L2: 300k+ Pure-Tungsten Contained Unlabeled SSL Corpus Production.

Strictly unsealed & chunk-disjoint pure-W (Z == 74) production from GENIE chunks >= 029
in {10081, 10083, 10084, 10086}, using pinpoint_v3_1 in worker scratch with zero .root
files retained on EOS. Filters strictly to observable-contained (D1_pix_contained U
D2_scint_contained) events and writes compact Zstd-compressed CSR `.npz.zst` shards
containing ONLY pure observables + (family_hi, family_lo) + obs_domain_id (zero truth
flavour/charm/vertex columns).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import uproot
import zstandard as zstd

from analysis.flavour.create_data import BRANCHES, ROOT_TREES
from analysis.flavour.inventory import write_json
from analysis.flavour.nextgen.dual_regime_modalnet import extract_nonexclusive_spec_candidates
from analysis.flavour.nextgen import seal_guard
from analysis.flavour.nextgen.seal_guard import (
    SealAuditGuard,
    SealViolationError,
    TRUTH_BATCH_KEYS,
)
from analysis.flavour.nextgen.tracklet_ipformer import extract_tracklet_pf_item
from analysis.flavour.readout import bar_channels, modality_state, pixel_channels, tracker_measurements
from analysis.flavour.s5.truth import group_flat
from analysis.flavour.s6.gfaser import nominal_record, write_gfaser
from analysis.flavour.v3.common.domains import compute_observable_containment, detector_bounds
from analysis.flavour.v3.v6_clean_sim.produce_clean_sim import _family_pair, _sample_tungsten_vertex_v2_mm
from analysis.flavour.v3.v6_clean_sim.produce_clean_v3 import build_full_volume_tungsten_sampler_spec


PINPOINT_DATA_DIR = Path('/eos/project-f/fasersim-bonn/public/pinpoint/data')
CLEAN_V3_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/clean_sim')
STAGE325_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v7_stage325_2026-11-20')
PINPOINT_V3_1_BIN = STAGE325_ROOT / 'g4/pinpoint_v3_1'
GEOM_V3_1_MACRO = STAGE325_ROOT / 'g4/macros/geom.mac'
GEOM_JSON_PATH = CLEAN_V3_ROOT / 'geometry.json'
DEFAULT_L2_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/l2_ssl_300k')

L2_RUN_IDS: tuple[int, ...] = (10081, 10086, 10083, 10084)
MIN_CHUNK_INDEX = 29
TRAIN_CHUNK_START = 29
TRAIN_CHUNK_END = 143      # [29, 143) = 114 chunks/run * 4 runs = 456 chunks (~422k raw -> ~320k+ D1+D2)
VAL_CHUNK_START = 200
VAL_CHUNK_END = 204        # [200, 204) = 4 chunks/run * 4 runs = 16 chunks (~14.8k raw -> ~11.8k D1+D2)
TRAIN_CHUNKS_PER_WORKER = 3
VAL_CHUNKS_PER_WORKER = 2

OBS_CFG: dict[str, Any] = {
    'pe_only': True,
    'apply_sipm_saturation': False,
    'invert_sipm_saturation': True,
}

# Pure observable SSL schema keys (strictly 0 truth flavour/charm/vertex columns)
SSL_SHARD_KEYS: tuple[str, ...] = (
    'family_hi',
    'family_lo',
    'obs_domain_id',
    'c_obs',
    'cal_start',
    'avr_pv_mm',
    'pix_ptr',
    'pix_xyz_mm',
    'pix_feat_16',
    'pix_plane',
    'pix_plane_summary',
    'pat_ptr',
    'pix_patches',
    'patch_cluster_idx',
    'bar_ptr',
    'bar_panel',
    'bar_horiz',
    'bar_idx',
    'bar_pe',
    'bar_hit_time_ns',
    'trk_cand',
    'trk_cand_mask',
    'trk_station_counts',
    'hit_ptr',
    'trk_hits',
    'trk_kin',
    'trk_morph',
    'trk_mask',
)

FORBIDDEN_TRUTH_KEYS: frozenset[str] = frozenset(TRUTH_BATCH_KEYS) - {'family_hi', 'family_lo'}

# Blocks Stage-I validation/tune paths, sealed shards 080..149, planned_manifest, and GENIE chunks 000..028
_L2_FORBIDDEN_PATH_RE = re.compile(
    r'(?:^|/)shards/(?:0[89]\d|1\d\d)(?:/|$)|'
    r'planned_manifest\.parquet|sealed_truth_meta|validation_meta|tune_screen\.parquet|tune_truth_meta\.parquet|'
    r'input_sys_confirm|'
    r'(?:^|[^0-9])1008[1-8][_/\\-]+0*(?:[01]?\d|2[0-8])(?:$|[^0-9])',
    re.IGNORECASE,
)

_L2_HOOK_INSTALLED = False


class _L2CorpusSealGuard(SealAuditGuard):
    """Scoped SealAuditGuard for L2 corpus provenance and GENIE chunk allowlisting."""


def install_l2_seal_guard() -> SealAuditGuard:
    """Install SealAuditGuard + L2 hard block on shards/080..149, planned_manifest, and GENIE 000..028."""
    global _L2_HOOK_INSTALLED
    guard = _L2CorpusSealGuard()
    guard.__enter__()
    if not _L2_HOOK_INSTALLED:
        def _l2_hook(event: str, args: tuple[Any, ...]) -> None:
            if not any(isinstance(g, _L2CorpusSealGuard) for g in seal_guard._ACTIVE_GUARDS):
                return
            if event not in ('open', 'os.stat', 'os.scandir', 'os.listdir') or not args:
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
            norm = path_str.replace('\\', '/')
            if _L2_FORBIDDEN_PATH_RE.search(norm):
                for g in list(seal_guard._ACTIVE_GUARDS):
                    if isinstance(g, _L2CorpusSealGuard):
                        g.violations.append(f'{event}:{norm}')
                raise SealViolationError(f'L2 provenance guard forbids accessing {norm}')

        sys.addaudithook(_l2_hook)
        _L2_HOOK_INSTALLED = True
    return guard


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _load_excluded_families() -> tuple[np.ndarray, dict[str, int]]:
    """Load all (family_hi, family_lo) pairs from Train32k (000..079) and all 6 OOD shards."""
    pairs: list[tuple[int, int]] = []
    source_counts: dict[str, int] = {}

    ood_dirs = (
        sorted((CLEAN_V3_ROOT / 'ood_shards').glob('*/truth_meta.parquet'))
        + sorted((STAGE325_ROOT / 'ood_shards').glob('*/truth_meta.parquet'))
    )
    for p in ood_dirs:
        df = pq.read_table(p, columns=['family_hi', 'family_lo']).to_pandas()
        hi = df['family_hi'].to_numpy(dtype=np.uint64)
        lo = df['family_lo'].to_numpy(dtype=np.uint64)
        pairs.extend(zip(hi.tolist(), lo.tolist()))
        source_counts[f'ood:{p.parent.name}'] = int(len(df))

    n_train = 0
    for sid in range(80):
        pp = CLEAN_V3_ROOT / 'data/shards' / f'{sid:03d}/provenance.parquet'
        if pp.is_file():
            df = pq.read_table(pp, columns=['family_hi', 'family_lo']).to_pandas()
            hi = df['family_hi'].to_numpy(dtype=np.uint64)
            lo = df['family_lo'].to_numpy(dtype=np.uint64)
            pairs.extend(zip(hi.tolist(), lo.tolist()))
            n_train += int(len(df))
    source_counts['train_000_079'] = n_train

    if not pairs:
        return np.empty((0, 2), dtype=np.uint64), source_counts
    arr = np.unique(np.asarray(pairs, dtype=np.uint64), axis=0)
    return arr, source_counts


def _family_pair_set(arr: np.ndarray) -> set[tuple[int, int]]:
    if len(arr) == 0:
        return set()
    return set(zip(arr[:, 0].tolist(), arr[:, 1].tolist()))


def reconstruct_pix_feat22(
    pix_xyz_mm: np.ndarray,
    pix_plane: np.ndarray,
    pix_feat_16: np.ndarray,
    pix_plane_summary: np.ndarray,
    bounds: dict,
) -> np.ndarray:
    """Reconstruct exact 22-dim `extract_pf_item` `pix_feat` from compact 16-dim `pix_feat_16`."""
    n_cl = len(pix_xyz_mm)
    if n_cl == 0:
        return np.empty((0, 22), dtype=np.float32)
    x_c = float(bounds['x_pix_center'])
    y_c = float(bounds['y_scint_center'])
    z_mid = float(bounds['z_mid'])
    pl = pix_plane.astype(np.int64)
    xyz = pix_xyz_mm.astype(np.float32)
    pf16 = pix_feat_16.astype(np.float32)
    ps = pix_plane_summary.astype(np.float32)

    plane_xc = ps[pl, 7] * 135.0 + x_c
    plane_yc = ps[pl, 6] * 100.0 + y_c
    dx_core = xyz[:, 0] - plane_xc
    dy_core = xyz[:, 1] - plane_yc
    r_core = np.hypot(dx_core, dy_core)

    out = np.empty((n_cl, 22), dtype=np.float32)
    out[:, 0] = (xyz[:, 0] - x_c) / 135.0
    out[:, 1] = (xyz[:, 1] - y_c) / 100.0
    out[:, 2] = (xyz[:, 2] - z_mid) / 1000.0
    out[:, 3:16] = pf16[:, :13]
    out[:, 16] = np.arcsinh(dx_core / 0.10) / 4.0
    out[:, 17] = np.arcsinh(dy_core / 0.10) / 4.0
    out[:, 18] = np.arcsinh(r_core / 0.08) / 4.0
    out[:, 19:22] = pf16[:, 13:16]
    return out


def pack_ssl_csr_zst(
    items: list[dict[str, Any]],
    out_path: Path,
    *,
    zstd_level: int = 12,
) -> dict[str, Any]:
    """Pack pure-observable SSL event dicts into a Zstd-compressed CSR `.npz.zst` shard."""
    n = len(items)
    if n == 0:
        raise ValueError('Cannot write empty SSL shard')
    for it in items:
        bad = set(it.keys()) & FORBIDDEN_TRUTH_KEYS
        if bad:
            raise AssertionError(f'Truth columns detected in SSL item: {sorted(bad)}')

    pix_ptr = np.zeros(n + 1, dtype=np.int32)
    bar_ptr = np.zeros(n + 1, dtype=np.int32)
    hit_ptr = np.zeros(n + 1, dtype=np.int32)
    pat_ptr = np.zeros(n + 1, dtype=np.int32)
    for k, it in enumerate(items):
        pix_ptr[k + 1] = pix_ptr[k] + len(it['pix_plane'])
        bar_ptr[k + 1] = bar_ptr[k] + len(it['bar_panel'])
        hit_ptr[k + 1] = hit_ptr[k] + len(it['trk_hits'])
        pat_ptr[k + 1] = pat_ptr[k] + len(it['pix_patches'])

    def _cat_or_empty(key: str, shape_tail: tuple[int, ...], dtype: Any) -> np.ndarray:
        parts = [np.asarray(it[key], dtype=dtype) for it in items if len(it[key]) > 0]
        if not parts:
            return np.empty((0, *shape_tail), dtype=dtype)
        return np.concatenate(parts, axis=0)

    ht_raw = _cat_or_empty('bar_hit_time_ns', (), np.float32)
    ht_q05 = np.where(
        ht_raw < 0.0,
        -1.0,
        np.minimum(np.round(ht_raw * 20.0) / 20.0, 65000.0),
    ).astype(np.float16)

    payload = {
        'family_hi': np.array([int(it['family_hi']) for it in items], dtype=np.uint64),
        'family_lo': np.array([int(it['family_lo']) for it in items], dtype=np.uint64),
        'obs_domain_id': np.array([int(it['obs_domain_id']) for it in items], dtype=np.uint8),
        'c_obs': np.stack([it['c_obs'] for it in items]).astype(np.float32),
        'cal_start': np.stack([it['cal_start'] for it in items]).astype(np.float32),
        'avr_pv_mm': np.stack([it['avr_pv_mm'] for it in items]).astype(np.float32),
        'pix_ptr': pix_ptr,
        'pix_xyz_mm': _cat_or_empty('pix_xyz_mm', (3,), np.float32),
        'pix_feat_16': _cat_or_empty('pix_feat_16', (16,), np.float16),
        'pix_plane': _cat_or_empty('pix_plane', (), np.int8),
        'pix_plane_summary': np.stack([it['pix_plane_summary'] for it in items]).astype(np.float32),
        'pat_ptr': pat_ptr,
        'pix_patches': _cat_or_empty('pix_patches', (2, 32, 32), np.float16),
        'patch_cluster_idx': _cat_or_empty('patch_cluster_idx', (), np.int16),
        'bar_ptr': bar_ptr,
        'bar_panel': _cat_or_empty('bar_panel', (), np.int16),
        'bar_horiz': _cat_or_empty('bar_horiz', (), np.int8),
        'bar_idx': _cat_or_empty('bar_idx', (), np.int16),
        'bar_pe': np.clip(_cat_or_empty('bar_pe', (), np.int32), 0, 65535).astype(np.uint16),
        'bar_hit_time_ns': ht_q05,
        'trk_cand': np.stack([it['trk_cand'] for it in items]).astype(np.float32),
        'trk_cand_mask': np.stack([it['trk_cand_mask'] for it in items]).astype(bool),
        'trk_station_counts': np.stack([it['trk_station_counts'] for it in items]).astype(np.float32),
        'hit_ptr': hit_ptr,
        'trk_hits': _cat_or_empty('trk_hits', (6,), np.float32),
        'trk_kin': np.stack([it['trk_kin'] for it in items]).astype(np.float16),
        'trk_morph': np.stack([it['trk_morph'] for it in items]).astype(np.float16),
        'trk_mask': np.stack([it['trk_mask'] for it in items]).astype(bool),
    }

    assert tuple(payload.keys()) == SSL_SHARD_KEYS
    buf = io.BytesIO()
    np.savez(buf, **payload)
    raw_bytes = buf.getvalue()
    comp_bytes = zstd.ZstdCompressor(level=int(zstd_level)).compress(raw_bytes)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f'.{out_path.name}.{os.getpid()}.tmp')
    tmp_path.write_bytes(comp_bytes)
    tmp_path.replace(out_path)
    return {
        'events': n,
        'raw_bytes': len(raw_bytes),
        'compressed_bytes': len(comp_bytes),
        'bytes_per_event': float(len(comp_bytes) / n),
        'mb_per_2000_events': float((len(comp_bytes) / n) * 2000.0 / 1e6),
    }


def unpack_ssl_csr_zst(path: Path) -> dict[str, np.ndarray]:
    """Decompress and load a `.npz.zst` CSR shard into memory."""
    dctx = zstd.ZstdDecompressor()
    raw = dctx.decompress(path.read_bytes())
    with np.load(io.BytesIO(raw), allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files}


def _read_pure_w_unlabeled_chunk(run: int, chunk: int, excl_set: set[tuple[int, int]]) -> dict[str, Any]:
    """Read pure-W (Z == 74) GENIE records from an unsealed chunk >= 029 without keeping truth labels."""
    if chunk < MIN_CHUNK_INDEX:
        raise SealViolationError(f'Refusing GENIE chunk {chunk} < {MIN_CHUNK_INDEX}')
    path = PINPOINT_DATA_DIR / str(run) / f'{run}_{chunk:03d}.root'
    if not path.is_file():
        return {'records': [], 'rejected_non_w': 0, 'excluded_overlap': 0}

    trivial = uproot.source.futures.TrivialExecutor()
    ev_cols = [
        'evtID', 'initX', 'initY', 'initZ', 'initPx', 'initPy', 'initPz', 'initE',
        'initPDG', 'fslPDG', 'hitnucPDG', 'intType', 'scatteringType', 'tgtPDG', 'Q2', 'W', 'xBj', 'y', 'xs',
    ]
    with uproot.open(
        path,
        decompression_executor=trivial,
        interpretation_executor=trivial,
        object_cache=None,
        array_cache=None,
    ) as src:
        ev = src['event'].arrays(ev_cols, library='np')
        prim_map = group_flat(src['primaries'], ['evtID', 'PDG', 'Px', 'Py', 'Pz', 'E'])

    records: list[dict[str, Any]] = []
    rejected_non_w = 0
    excluded_overlap = 0
    tag = f'pinpoint:{run}:{chunk:03d}'
    n_rows = len(ev['evtID'])
    for r in range(n_rows):
        evt_id = int(ev['evtID'][r])
        pev = prim_map.get(evt_id)
        if pev is None or len(pev['PDG']) == 0:
            continue
        meta = {k: ev[k][r] for k in ev_cols}
        it = int(meta['intType'])
        if it not in (2, 3):
            continue
        tgt_pdg = int(meta['tgtPDG'])
        tgt_z = (tgt_pdg // 10000) % 1000 if tgt_pdg >= 1000000000 else -1
        if tgt_z != 74:
            rejected_non_w += 1
            continue
        rec = nominal_record(meta, pev)
        if int(rec['Z']) != 74:
            rejected_non_w += 1
            continue
        for k_nan in ('Q2', 'W', 'x', 'y', 'XSec'):
            if not np.isfinite(rec[k_nan]):
                rec[k_nan] = 0.0
        hi, lo = _family_pair(tag, r)
        if (hi, lo) in excl_set:
            excluded_overlap += 1
            continue
        rec['family_hi'] = int(hi)
        rec['family_lo'] = int(lo)
        rec['source_run'] = int(run)
        rec['source_chunk'] = int(chunk)
        records.append(rec)
    return {
        'records': records,
        'rejected_non_w': rejected_non_w,
        'excluded_overlap': excluded_overlap,
    }


def _simulate_and_extract_subchunk(task: tuple) -> dict[str, Any]:
    """Run pinpoint_v3_1 in worker scratch, keep D1 U D2 events, extract pure observables, delete .root."""
    records, scratch_str, sub_idx, seed_offset, geometry, bounds = task
    sub_dir = Path(scratch_str) / f'sub_{sub_idx:02d}'
    sub_dir.mkdir(parents=True, exist_ok=True)
    in_root = sub_dir / 'in.gfaser.root'
    out_root = sub_dir / 'out_g4.root'
    mac = sub_dir / 'run.mac'

    try:
        write_gfaser(records, in_root)
        seed_base = 20263280 + int(seed_offset) * 53 + sub_idx * 17
        mac.write_text('\n'.join([
            '/control/verbose 0',
            '/run/verbose 0',
            f'/control/execute {GEOM_V3_1_MACRO}',
            '/det/enableFaserSpectrometer true',
            '/gen/select gfaser',
            f'/gen/gfaser/inputFile {in_root}',
            '/gen/gfaser/hitPixelArea false',
            f'/out/fileName {out_root}',
            '/out/saveTrack false',
            '/out/saveTruthHits false',
            f'/random/setSeeds {seed_base} {seed_base + 23}',
            '/run/initialize',
            '/Pinpoint/field/value 0.57 tesla',
            f'/run/beamOn {len(records)}',
            '',
        ]))
        cmd = (
            'which() { command -v "$@"; }; export -f which && '
            'source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc11-opt/setup.sh >/dev/null 2>&1 && '
            f'exec {shlex.quote(str(PINPOINT_V3_1_BIN))} {shlex.quote(str(mac))}'
        )
        env = dict(os.environ)
        env.update({
            'PINPOINT_USE_INPUT_Z': '1',
            'PINPOINT_SKIP_GDML': '1',
            'PINPOINT_COMPACT_TRUTH': '1',
            'PINPOINT_REQUIRE_W_MATERIAL': '1',
            'PINPOINT_REQUIRE_W_TARGET': '1',
            'PINPOINT_PIXEL_DIFFUSION_UM': '3.0',
            'PINPOINT_PIXEL_NOISE_E': '10.0',
            'PINPOINT_PIXEL_THR_E': '100.0',
            'PINPOINT_SIPM_TIME_THR_PE': '3',
            'PINPOINT_SIPM_TTS_NS': '0.15',
        })
        cp = subprocess.run(
            ['/bin/bash', '-c', cmd],
            cwd=sub_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if cp.returncode != 0 or not out_root.is_file():
            raise RuntimeError(
                f'pinpoint_v3_1 subchunk {sub_idx} failed rc={cp.returncode}: '
                f'{cp.stderr[-400:]}\n{cp.stdout[-400:]}'
            )

        in_root.unlink(missing_ok=True)

        items: list[dict[str, Any]] = []
        n_d1 = 0
        n_d2 = 0
        n_dropped = 0
        n_w_verified = 0
        zero_pe_viol = 0

        with uproot.open(out_root, object_cache=None, array_cache=None) as src:
            ev_arr = src['event'].arrays(['tgtZ', 'vtxMatZ', 'vtxMatName'], library='np')
            data = {}
            for n, k in BRANCHES.items():
                tree = src[ROOT_TREES[n]]
                cols = list(k)
                if n == 'bar':
                    cols = [c for c in cols if c != 'edep']
                    if 'hitTime' in tree.keys():
                        cols.append('hitTime')
                data[n] = tree.arrays(cols, library='ak')
            n_sub = len(ev_arr['tgtZ'])
            if n_sub != len(records):
                raise RuntimeError(f'G4 event count {n_sub} != input {len(records)}')

            for i in range(n_sub):
                vtx_mat_z = int(ev_arr['vtxMatZ'][i])
                vtx_mat_name = str(ev_arr['vtxMatName'][i])
                tgt_z = int(ev_arr['tgtZ'][i])
                if vtx_mat_name != 'G4_W' or vtx_mat_z != 74 or tgt_z != 74:
                    raise AssertionError(f'Pure-W check failed: mat={vtx_mat_name}, matZ={vtx_mat_z}, tgtZ={tgt_z}')
                n_w_verified += 1

                rec = records[i]
                hi = int(rec['family_hi'])
                lo = int(rec['family_lo'])

                h = {n: {k: ak.to_numpy(v[k][i]) for k in v.fields} for n, v in data.items()}
                pix = pixel_channels(h['pixel'], geometry)
                bars, _ = bar_channels(h['bar'], geometry, include_truth_edep=False)
                if 'hit_time_ns' in bars and len(bars['pe_proxy']):
                    bad_t = int(np.sum((bars['pe_proxy'] <= 0) & (bars['hit_time_ns'] >= 0.0)))
                    zero_pe_viol += bad_t
                    if bad_t > 0:
                        raise AssertionError(f'0-PE bar carried non-negative hit_time_ns ({bad_t})')

                seed = hi ^ lo ^ 20263280
                _, trk = tracker_measurements(h['tracker'], seed)
                state = modality_state(pix, bars, trk)
                obs_dict = {
                    'pixel': pix,
                    'bar': bars,
                    'tracker_hypothesis': trk,
                    'has_response': state['has_response'],
                    'outside_acceptance': state['outside_acceptance'],
                    'fit_attempted': False,
                    'momentum_censored': -1,
                }
                dom = compute_observable_containment(obs_dict, geometry, bounds, config=OBS_CFG)
                dom_id = int(dom['obs_domain_id'])
                if dom_id == 1:
                    n_d1 += 1
                elif dom_id == 2:
                    n_d2 += 1
                else:
                    n_dropped += 1
                    continue

                it = extract_tracklet_pf_item(i, obs_dict, None, dom, geometry, bounds, OBS_CFG)
                c8, c8m, st_cnt = extract_nonexclusive_spec_candidates(trk, dom['cal_start'], bounds)
                pf = it['pix_feat']
                pf_16 = (
                    np.concatenate([pf[:, 3:16], pf[:, 19:22]], axis=1).astype(np.float16)
                    if len(pf) > 0
                    else np.empty((0, 16), dtype=np.float16)
                )
                keep_bar = np.asarray(it['bar_keep_mask'], dtype=bool)
                bar_pe = np.asarray(bars['pe_proxy'], dtype=np.int32)[keep_bar] if len(keep_bar) else np.empty(0, dtype=np.int32)

                ssl_item = {
                    'family_hi': np.uint64(hi),
                    'family_lo': np.uint64(lo),
                    'obs_domain_id': np.int8(dom_id),
                    'c_obs': it['c_obs'].astype(np.float16),
                    'cal_start': it['cal_start'].astype(np.float32),
                    'avr_pv_mm': it['avr_pv_mm'].astype(np.float32),
                    'pix_xyz_mm': it['pix_xyz_mm'].astype(np.float32),
                    'pix_feat_16': pf_16,
                    'pix_plane': it['pix_plane'].astype(np.int8),
                    'pix_plane_summary': it['pix_plane_summary'].astype(np.float32),
                    'pix_patches': it['pix_patches'].astype(np.float16),
                    'patch_cluster_idx': it['patch_cluster_idx'].astype(np.int16),
                    'bar_panel': it['bar_panel'].astype(np.int16),
                    'bar_horiz': it['bar_horiz'].astype(np.int8),
                    'bar_idx': it['bar_idx'].astype(np.int16),
                    'bar_pe': bar_pe,
                    'bar_hit_time_ns': it['bar_hit_time_ns'].astype(np.float32),
                    'trk_cand': c8.astype(np.float32),
                    'trk_cand_mask': c8m.astype(bool),
                    'trk_station_counts': st_cnt.astype(np.float32),
                    'trk_hits': it['trk_hits'].astype(np.float32),
                    'trk_kin': it['trk_kin'].astype(np.float32),
                    'trk_morph': it['trk_morph'].astype(np.float16),
                    'trk_mask': it['trk_mask'].astype(bool),
                }
                items.append(ssl_item)

        return {
            'items': items,
            'n_simulated': n_sub,
            'n_d1': n_d1,
            'n_d2': n_d2,
            'n_dropped': n_dropped,
            'n_w_verified': n_w_verified,
            'zero_pe_violations': zero_pe_viol,
        }
    finally:
        shutil.rmtree(sub_dir, ignore_errors=True)


def survey_and_prepare_l2(root: Path, workers: int = 8) -> dict[str, Any]:
    """Survey unsealed chunks >= 029, build OOD/Train32k exclusion set, and plan 152 8-core tasks."""
    t0 = time.perf_counter()
    guard = install_l2_seal_guard()
    root.mkdir(parents=True, exist_ok=True)
    for d in ('train_shards', 'ssl_val_calib', 'runtime', 'logs'):
        (root / d).mkdir(parents=True, exist_ok=True)

    geom = json.loads(GEOM_JSON_PATH.read_text())
    bounds = detector_bounds(geom)
    sampler_spec = build_full_volume_tungsten_sampler_spec(geom, bounds, margin_xy_mm=0.001, margin_z_mm=0.001)

    excl_arr, excl_counts = _load_excluded_families()
    np.savez(root / 'runtime/excluded_families.npz', pairs=excl_arr)

    # Verify all planned chunks >= 029 exist and check sample pure-W counts
    train_chunks_by_run: dict[int, list[int]] = {}
    val_chunks_by_run: dict[int, list[int]] = {}
    for run in L2_RUN_IDS:
        rdir = PINPOINT_DATA_DIR / str(run)
        avail = sorted(
            int(p.stem.split('_')[1])
            for p in rdir.glob(f'{run}_*.root')
            if int(p.stem.split('_')[1]) >= MIN_CHUNK_INDEX
        )
        avail_set = set(avail)
        tr_ch = [c for c in range(TRAIN_CHUNK_START, TRAIN_CHUNK_END) if c in avail_set]
        va_ch = [c for c in range(VAL_CHUNK_START, VAL_CHUNK_END) if c in avail_set]
        if len(tr_ch) != (TRAIN_CHUNK_END - TRAIN_CHUNK_START):
            raise RuntimeError(f'Run {run}: found {len(tr_ch)} train chunks in [{TRAIN_CHUNK_START}, {TRAIN_CHUNK_END})')
        if len(va_ch) != (VAL_CHUNK_END - VAL_CHUNK_START):
            raise RuntimeError(f'Run {run}: found {len(va_ch)} val chunks in [{VAL_CHUNK_START}, {VAL_CHUNK_END})')
        train_chunks_by_run[run] = tr_ch
        val_chunks_by_run[run] = va_ch

    # Interleave runs round-robin so every worker gets a balanced flavour mix & uniform runtime
    train_pool: list[list[int]] = []
    for idx in range(TRAIN_CHUNK_END - TRAIN_CHUNK_START):
        for run in L2_RUN_IDS:
            train_pool.append([int(run), int(train_chunks_by_run[run][idx])])

    val_pool: list[list[int]] = []
    for idx in range(VAL_CHUNK_END - VAL_CHUNK_START):
        for run in L2_RUN_IDS:
            val_pool.append([int(run), int(val_chunks_by_run[run][idx])])

    # Verify chunk disjointness across splits and >= 029
    tr_set = {(r, c) for r, c in train_pool}
    va_set = {(r, c) for r, c in val_pool}
    if tr_set & va_set:
        raise AssertionError('Train and ssl_val_calib chunk sets overlap!')
    if any(c < MIN_CHUNK_INDEX for _, c in (tr_set | va_set)):
        raise AssertionError('Chunk index < 029 detected in task plan!')

    tasks: list[dict[str, Any]] = []
    shard_idx = 0
    for s in range(0, len(train_pool), TRAIN_CHUNKS_PER_WORKER):
        tasks.append({
            'task_id': len(tasks),
            'split': 'train_shards',
            'shard_name': f'shard_{shard_idx:04d}',
            'chunks': train_pool[s : s + TRAIN_CHUNKS_PER_WORKER],
        })
        shard_idx += 1

    val_idx = 0
    for s in range(0, len(val_pool), VAL_CHUNKS_PER_WORKER):
        tasks.append({
            'task_id': len(tasks),
            'split': 'ssl_val_calib',
            'shard_name': f'val_{val_idx:04d}',
            'chunks': val_pool[s : s + VAL_CHUNKS_PER_WORKER],
        })
        val_idx += 1

    # Quick survey of first chunk per run to record empirical pure-W density
    survey_stats: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=min(4, workers)) as tpool:
        def _probe(run_id: int) -> tuple[int, int, int]:
            fp = PINPOINT_DATA_DIR / str(run_id) / f'{run_id}_{TRAIN_CHUNK_START:03d}.root'
            with uproot.open(fp) as f:
                tgts = f['event'].arrays(['tgtPDG'], library='np')['tgtPDG']
            w_cnt = int(np.sum(((tgts // 10000) % 1000) == 74))
            return run_id, len(tgts), w_cnt
        for rid, n_tot, n_w in tpool.map(_probe, L2_RUN_IDS):
            survey_stats[str(rid)] = {
                'train_chunks': [TRAIN_CHUNK_START, TRAIN_CHUNK_END - 1],
                'val_chunks': [VAL_CHUNK_START, VAL_CHUNK_END - 1],
                'sample_chunk_total': n_tot,
                'sample_chunk_pure_W': n_w,
                'pure_W_frac': float(n_w / max(1, n_tot)),
            }

    plan = {
        'stage': 'Stage328_L2_SSL_300k',
        'UTC': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': float(time.perf_counter() - t0),
        'pinpoint_v3_1_bin': str(PINPOINT_V3_1_BIN),
        'pinpoint_v3_1_sha256': _sha256_file(PINPOINT_V3_1_BIN),
        'geom_macro_sha256': _sha256_file(GEOM_V3_1_MACRO),
        'geometry_json': str(GEOM_JSON_PATH),
        'n_tasks': len(tasks),
        'n_train_tasks': shard_idx,
        'n_val_tasks': val_idx,
        'min_chunk_index': MIN_CHUNK_INDEX,
        'excluded_family_counts': excl_counts,
        'total_excluded_unique_families': int(len(excl_arr)),
        'stratum_volume_fractions': list(sampler_spec['volume_fractions']),
        'stratum_sampling_probs': list(sampler_spec['sampling_probs']),
        'survey_stats': survey_stats,
        'seal_violations': list(guard.violations),
        'tasks': tasks,
    }
    write_json(root / 'runtime/task_plan.json', plan)
    return {k: v for k, v in plan.items() if k != 'tasks'}


def run_l2_worker(
    root: Path,
    task_id: int,
    *,
    cpus: int = 8,
    max_events: int | None = None,
    smoke_out_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute one 8-core L2 worker: read pure-W chunks, run pinpoint_v3_1 in scratch, save .npz.zst."""
    t0 = time.perf_counter()
    guard = install_l2_seal_guard()
    plan = json.loads((root / 'runtime/task_plan.json').read_text())
    task = plan['tasks'][int(task_id)]
    split_name = str(task['split'])
    shard_name = str(task['shard_name'])

    if smoke_out_dir is not None:
        out_dir = smoke_out_dir
    else:
        out_dir = root / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / f'{shard_name}.npz.zst'
    meta_path = out_dir / f'{shard_name}.meta.json'
    if shard_path.is_file() and meta_path.is_file() and smoke_out_dir is None:
        return json.loads(meta_path.read_text())

    geom = json.loads(Path(plan['geometry_json']).read_text())
    bounds = detector_bounds(geom)
    sampler_spec = build_full_volume_tungsten_sampler_spec(geom, bounds, margin_xy_mm=0.001, margin_z_mm=0.001)

    excl_npz = np.load(root / 'runtime/excluded_families.npz', allow_pickle=False)
    excl_set = _family_pair_set(excl_npz['pairs'])

    records: list[dict[str, Any]] = []
    rejected_non_w = 0
    excluded_overlap = 0
    for run, chunk in task['chunks']:
        res = _read_pure_w_unlabeled_chunk(int(run), int(chunk), excl_set)
        records.extend(res['records'])
        rejected_non_w += int(res['rejected_non_w'])
        excluded_overlap += int(res['excluded_overlap'])
        if max_events is not None and max_events > 0 and len(records) >= int(max_events):
            break

    if max_events is not None and max_events > 0:
        records = records[: int(max_events)]

    # Deterministic shuffling within worker so each core gets balanced flavour/stratum load
    rng = np.random.default_rng(20263280 + int(task_id))
    perm = rng.permutation(len(records))
    records = [records[int(i)] for i in perm]

    for r in records:
        vm = _sample_tungsten_vertex_v2_mm(r['family_hi'], r['family_lo'], sampler_spec)
        r['vx'] = vm['vx_mm'] / 1000.0
        r['vy'] = vm['vy_mm'] / 1000.0
        r['vz'] = vm['vz_mm'] / 1000.0

    scratch_base = Path(os.environ.get('_CONDOR_SCRATCH_DIR', os.environ.get('PINPOINT_L2_SCRATCH', '/tmp')))
    scratch_base.mkdir(parents=True, exist_ok=True)

    n_workers = max(1, min(int(cpus), len(records)))
    step = max(1, (len(records) + n_workers - 1) // n_workers)
    with tempfile.TemporaryDirectory(prefix=f'l2-w{int(task_id):03d}-', dir=scratch_base) as tmp_str:
        subtasks = [
            (records[s : s + step], tmp_str, k, int(task_id), geom, bounds)
            for k, s in enumerate(range(0, len(records), step))
        ]
        if n_workers == 1:
            sub_results = [_simulate_and_extract_subchunk(subtasks[0])]
        else:
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context('spawn')) as pool:
                sub_results = list(pool.map(_simulate_and_extract_subchunk, subtasks))

    all_items: list[dict[str, Any]] = []
    n_sim = 0
    n_d1 = 0
    n_d2 = 0
    n_drop = 0
    n_w_ver = 0
    zero_pe_viol = 0
    for sr in sub_results:
        all_items.extend(sr['items'])
        n_sim += int(sr['n_simulated'])
        n_d1 += int(sr['n_d1'])
        n_d2 += int(sr['n_d2'])
        n_drop += int(sr['n_dropped'])
        n_w_ver += int(sr['n_w_verified'])
        zero_pe_viol += int(sr['zero_pe_violations'])

    pack_stats = pack_ssl_csr_zst(all_items, shard_path, zstd_level=14)
    meta = {
        'task_id': int(task_id),
        'split': split_name,
        'shard_name': shard_name,
        'shard_file': shard_path.name,
        'chunks': task['chunks'],
        'rejected_non_w_genie': int(rejected_non_w),
        'excluded_ood_train_overlap': int(excluded_overlap),
        'n_simulated': int(n_sim),
        'n_w_verified': int(n_w_ver),
        'n_contained': int(n_d1 + n_d2),
        'n_d1': int(n_d1),
        'n_d2': int(n_d2),
        'n_dropped_d0_d3': int(n_drop),
        'containment_rate': float((n_d1 + n_d2) / max(1, n_sim)),
        'zero_pe_time_violations': int(zero_pe_viol),
        'seal_violations': list(guard.violations),
        'wall_seconds': float(time.perf_counter() - t0),
        **pack_stats,
    }
    write_json(meta_path, meta)
    return meta


def choose_l2_schedds() -> list[tuple[str, int, int]]:
    """Select compatible least-loaded CERN HTCondor schedds (AFS log layout)."""
    snap = subprocess.check_output(
        ['condor_status', '-schedd', '-af', 'Name', 'TotalRunningJobs', 'TotalIdleJobs', 'TotalHeldJobs'],
        text=True,
    )
    candidates: list[tuple[int, int, int, str]] = []
    for line in snap.splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        name, run_s, idle_s, held_s = parts
        if not name.startswith('bigbird'):
            continue
        candidates.append((int(idle_s), int(held_s), int(run_s), name))

    usable: list[tuple[str, int, int]] = []
    for idle_i, held_i, run_i, name in sorted(candidates):
        reqs = subprocess.run(
            ['condor_config_val', '-name', name, '-schedd', 'SUBMIT_REQUIREMENT_NAMES'],
            capture_output=True, text=True,
        )
        req_names = {p.strip().lower() for p in reqs.stdout.split(',')}
        if reqs.returncode or (req_names & {'alleos', 'abspathineos', 'onlyspool'}):
            continue
        ping = subprocess.run(
            ['condor_ping', '-type', 'SCHEDD', '-name', name, '-table', 'WRITE'],
            capture_output=True, text=True,
        )
        if ping.returncode == 0 and 'ALLOW' in ping.stdout:
            usable.append((name, idle_i, run_i))
        if len(usable) >= 3:
            break
    if not usable:
        raise RuntimeError('No WRITE-authorized bigbird schedd found')
    return usable


def submit_l2_workers(root: Path, cpus: int = 8, job_flavour: str = 'longlunch', max_schedds: int = 2) -> dict[str, Any]:
    """Submit all L2 8-core workers across least-loaded compatible schedds."""
    from analysis.flavour.v3.common.submit import _package_source

    root = root.resolve()
    plan = json.loads((root / 'runtime/task_plan.json').read_text())
    n_tasks = int(plan['n_tasks'])

    repo = Path(__file__).resolve().parents[4]
    archive = root / 'runtime/stage328_l2_code.tar.gz'
    _package_source(repo, archive)

    condor_dir = Path.home() / 'condor/Pinpoint/stage328_l2_sim'
    (condor_dir / 'logs').mkdir(parents=True, exist_ok=True)

    wrapper = condor_dir / 'run_l2_worker.sh'
    wrapper.write_text('\n'.join([
        '#!/bin/bash',
        'set -o pipefail',
        'which() { command -v "$@"; }; export -f which',
        'source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh >/dev/null 2>&1',
        'set -e',
        'ls /eos/home-x/xcheng/FASER/Pinpoint/results >/dev/null',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1',
        'export PINPOINT_L2_SCRATCH="${_CONDOR_SCRATCH_DIR:-/tmp}/l2_ssl_${PINPOINT_JOB_ID}"',
        'mkdir -p "$PINPOINT_L2_SCRATCH/code"',
        'cleanup() { rm -rf "$PINPOINT_L2_SCRATCH"; }',
        'trap cleanup EXIT',
        f'tar -xzf {shlex.quote(str(archive))} -C "$PINPOINT_L2_SCRATCH/code"',
        'export PYTHONPATH="$PINPOINT_L2_SCRATCH/code/src:${PYTHONPATH:-}"',
        'cd "$PINPOINT_L2_SCRATCH"',
        f'exec python3 -m analysis.flavour.nextgen.produce_l2_ssl_corpus worker '
        f'--root {shlex.quote(str(root))} --task-id "$1" --task-offset "$2" --cpus {int(cpus)}',
        '',
    ]))
    wrapper.chmod(0o755)

    schedds = choose_l2_schedds()[:max(1, int(max_schedds))]
    # Partition task IDs evenly across selected low-idle schedds
    per_s = (n_tasks + len(schedds) - 1) // len(schedds)
    submissions: list[dict[str, Any]] = []
    for s_idx, (schedd_name, idle_i, run_i) in enumerate(schedds):
        t_start = s_idx * per_s
        t_end = min(n_tasks, t_start + per_s)
        if t_start >= t_end:
            continue
        sub_path = condor_dir / f'array_{s_idx}.sub'
        sub_path.write_text('\n'.join([
            'universe = vanilla',
            f'executable = {wrapper}',
            f'arguments = $(Process) {t_start}',
            f'initialdir = {condor_dir}',
            f'output = {condor_dir}/logs/worker.$(ClusterId).$(Process).out',
            f'error = {condor_dir}/logs/worker.$(ClusterId).$(Process).err',
            f'log = {condor_dir}/logs/worker.$(ClusterId).log',
            'environment = "PINPOINT_JOB_ID=$(ClusterId).$(Process)"',
            f'request_cpus = {int(cpus)}',
            'request_memory = 16GB',
            'request_disk = 16GB',
            'requirements = (OpSysAndVer == "AlmaLinux9")',
            f'+JobFlavour = "{job_flavour}"',
            'should_transfer_files = YES',
            'when_to_transfer_output = ON_EXIT',
            'transfer_output_files = ""',
            f'queue {t_end - t_start}',
            '',
        ]))
        resp = subprocess.run(['condor_submit', '-name', schedd_name, str(sub_path)], capture_output=True, text=True, check=True)
        m = re.search(r'submitted to cluster (\d+)', resp.stdout)
        cluster_id = int(m.group(1)) if m else -1
        submissions.append({
            'schedd': schedd_name,
            'idle_at_submit': idle_i,
            'running_at_submit': run_i,
            'cluster_id': cluster_id,
            'task_start': t_start,
            'task_count': t_end - t_start,
            'submit_file': str(sub_path),
            'stdout': resp.stdout.strip(),
        })

    sub_summary = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'n_tasks': n_tasks,
        'cpus_per_worker': int(cpus),
        'job_flavour': job_flavour,
        'submissions': submissions,
    }
    write_json(root / 'runtime/condor_submissions.json', sub_summary)
    return sub_summary


def finalize_l2_corpus(root: Path) -> dict[str, Any]:
    """Aggregate shard metadata, verify 0 family overlap, 0 truth columns, 0 .root files, write manifest."""
    t0 = time.perf_counter()
    guard = install_l2_seal_guard()
    plan = json.loads((root / 'runtime/task_plan.json').read_text())

    excl_npz = np.load(root / 'runtime/excluded_families.npz', allow_pickle=False)
    excl_set = _family_pair_set(excl_npz['pairs'])

    split_totals: dict[str, dict[str, Any]] = {}
    seen_families: set[tuple[int, int]] = set()
    train_families: set[tuple[int, int]] = set()
    val_families: set[tuple[int, int]] = set()
    family_overlap_excl = 0
    internal_duplicate_families = 0
    zero_pe_violations = 0
    truth_col_violations = 0
    shard_entries: list[dict[str, Any]] = []

    for split_name in ('train_shards', 'ssl_val_calib'):
        sdir = root / split_name
        metas = sorted(sdir.glob('*.meta.json'))
        n_sim = 0
        n_w = 0
        n_cont = 0
        n_d1 = 0
        n_d2 = 0
        n_drop = 0
        tot_bytes = 0
        for mp in metas:
            m = json.loads(mp.read_text())
            sp_file = sdir / str(m['shard_file'])
            if not sp_file.is_file():
                raise FileNotFoundError(f'Missing shard file {sp_file}')
            tot_bytes += int(sp_file.stat().st_size)
            n_sim += int(m['n_simulated'])
            n_w += int(m['n_w_verified'])
            n_cont += int(m['n_contained'])
            n_d1 += int(m['n_d1'])
            n_d2 += int(m['n_d2'])
            n_drop += int(m['n_dropped_d0_d3'])
            zero_pe_violations += int(m['zero_pe_time_violations'])

            # Verify schema, zero truth keys, and family disjointness
            arrs = unpack_ssl_csr_zst(sp_file)
            bad_keys = set(arrs.keys()) & FORBIDDEN_TRUTH_KEYS
            if bad_keys:
                truth_col_violations += len(bad_keys)
                raise AssertionError(f'Truth columns {bad_keys} found in {sp_file}')
            fams = list(zip(arrs['family_hi'].tolist(), arrs['family_lo'].tolist()))
            fams_set = set(fams)
            if len(fams_set) != len(fams):
                internal_duplicate_families += (len(fams) - len(fams_set))
            if fams_set & excl_set:
                family_overlap_excl += len(fams_set & excl_set)
            if fams_set & seen_families:
                internal_duplicate_families += len(fams_set & seen_families)
            seen_families |= fams_set
            if split_name == 'train_shards':
                train_families |= fams_set
            else:
                val_families |= fams_set

            shard_entries.append({
                'split': split_name,
                'shard_file': f'{split_name}/{sp_file.name}',
                'events': int(m['n_contained']),
                'd1': int(m['n_d1']),
                'd2': int(m['n_d2']),
                'bytes': int(sp_file.stat().st_size),
                'mb_per_2000_events': float(m['mb_per_2000_events']),
            })

        split_totals[split_name] = {
            'n_shards': len(metas),
            'n_simulated': n_sim,
            'n_w_verified': n_w,
            'n_contained': n_cont,
            'n_d1': n_d1,
            'n_d2': n_d2,
            'n_dropped_d0_d3': n_drop,
            'containment_rate': float(n_cont / max(1, n_sim)),
            'total_mb': float(tot_bytes / 1e6),
            'mean_mb_per_2000_events': float((tot_bytes / max(1, n_cont)) * 2000.0 / 1e6),
        }

    train_val_overlap = len(train_families & val_families)
    root_files_on_eos = list(root.rglob('*.root'))
    sub_file = root / 'runtime/condor_submissions.json'
    condor_info = json.loads(sub_file.read_text()) if sub_file.is_file() else {}

    total_bytes = sum(e['bytes'] for e in shard_entries)
    manifest = {
        'stage': 'Stage328_L2_SSL_300k',
        'UTC': datetime.now(timezone.utc).isoformat(),
        'finalize_wall_seconds': float(time.perf_counter() - t0),
        'root': str(root),
        'geometry_json': plan['geometry_json'],
        'pinpoint_v3_1_sha256': plan['pinpoint_v3_1_sha256'],
        'schema_keys': list(SSL_SHARD_KEYS),
        'truth_columns_present': int(truth_col_violations),
        'root_files_on_eos': len(root_files_on_eos),
        'min_genie_chunk_index': int(plan['min_chunk_index']),
        'train_shards_summary': split_totals['train_shards'],
        'ssl_val_calib_summary': split_totals['ssl_val_calib'],
        'total_eos_mb': float(total_bytes / 1e6),
        'total_eos_gb': float(total_bytes / 1e9),
        'family_overlap_with_train32k_and_6ood': int(family_overlap_excl),
        'train_vs_ssl_val_family_overlap': int(train_val_overlap),
        'internal_duplicate_families': int(internal_duplicate_families),
        'zero_pe_time_violations': int(zero_pe_violations),
        'seal_violations': len(guard.violations),
        'condor_submissions': condor_info,
        'shards': shard_entries,
    }
    if family_overlap_excl != 0 or train_val_overlap != 0 or internal_duplicate_families != 0:
        raise AssertionError(
            f'Family overlap violation: excl={family_overlap_excl}, '
            f'train_val={train_val_overlap}, dup={internal_duplicate_families}'
        )
    if len(root_files_on_eos) != 0:
        raise AssertionError(f'Found {len(root_files_on_eos)} .root files on EOS!')
    if guard.violations:
        raise AssertionError(f'Seal violations recorded: {guard.violations}')

    write_json(root / 'l2_corpus_manifest.json', manifest)
    return {k: v for k, v in manifest.items() if k != 'shards'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'worker', 'submit-workers', 'finalize'))
    parser.add_argument('--root', type=Path, default=DEFAULT_L2_ROOT)
    parser.add_argument('--task-id', type=int, default=0)
    parser.add_argument('--task-offset', type=int, default=0)
    parser.add_argument('--cpus', type=int, default=8)
    parser.add_argument('--max-events', type=int, default=None)
    parser.add_argument('--smoke-out-dir', type=Path, default=None)
    parser.add_argument('--job-flavour', type=str, default='longlunch')
    parser.add_argument('--max-schedds', type=int, default=2)
    args = parser.parse_args()

    if args.command == 'prepare':
        out = survey_and_prepare_l2(args.root, workers=args.cpus)
    elif args.command == 'worker':
        out = run_l2_worker(
            args.root,
            args.task_id + args.task_offset,
            cpus=args.cpus,
            max_events=args.max_events,
            smoke_out_dir=args.smoke_out_dir,
        )
    elif args.command == 'submit-workers':
        out = submit_l2_workers(
            args.root,
            cpus=args.cpus,
            job_flavour=args.job_flavour,
            max_schedds=args.max_schedds,
        )
    else:
        out = finalize_l2_corpus(args.root)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
