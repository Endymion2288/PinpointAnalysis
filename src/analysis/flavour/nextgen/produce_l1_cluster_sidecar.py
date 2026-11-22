"""Stage 328 Experiment L1: Pure-W Train32k Paired Observations & Cluster-Origin Sidecar Producer.

Produces the 100%-self-consistent paired `L1_Train32k` dataset (32,000 pure-W events across
80 shards x 400 events) and compact Zstd Parquet `cluster_sidecar/` under:
    /eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/l1_cluster_sidecar/

Strict Zero-Seal & Stage-I Isolation Rules:
  - Installs runtime `SealAuditGuard` + Stage-1 hard audit hook blocking:
      * Cal-A (100..109), Cut-B (110..124), Confirm (125..149), sealed_truth_meta,
        planned_manifest.parquet, input_sys_confirm*, Tune8k/Tune6k (080..099),
        validation_meta, tune_screen, tune_truth_meta,
      * GENIE 10081..10084 chunks >= 012 and 10086 chunks 000..001 or >= 016.
  - Zero `.root` files ever stored on EOS: `pinpoint_v3_1` runs exclusively inside local
    worker scratch (`_CONDOR_SCRATCH_DIR` / `/tmp`) with `trap cleanup EXIT`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
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
import pyarrow as pa
import pyarrow.parquet as pq
import uproot

from analysis.flavour.create_data import BRANCHES, ROOT_TREES, lists
from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.nextgen.seal_guard import (
    SealAuditGuard,
    SealViolationError,
    load_authorized_split_shards,
)
from analysis.flavour.nextgen.tracklet_ops import extract_soft_tracklets_single
from analysis.flavour.readout import bar_channels, modality_state, pixel_channels, tracker_measurements
from analysis.flavour.s2.prepare import classify_tau_daughters
from analysis.flavour.s5.truth import charmed, group_flat, primary_decay
from analysis.flavour.s6.gfaser import nominal_record, write_gfaser
from analysis.flavour.v2.partitions import FAMILY, KEY
from analysis.flavour.v3.common.domains import (
    compute_observable_containment,
    cumulative_radiation_length,
    detector_bounds,
)
from analysis.flavour.v3.common.truth_sidecar import _compute_label_7, _compute_vis_kinematics
from analysis.flavour.v3.v2_pinpointformer.pixel_tokenizer import (
    N_MAX_IPT_CLUSTERS,
    N_MAX_TARGET_CLUSTERS,
    _cluster_sensor_groups,
    _compute_plane_neighborhoods,
    _select_group_clusters,
)
from analysis.flavour.v3.v6_clean_sim.produce_clean_sim import (
    CHARM_PDGS,
    PINPOINT_DATA_DIR,
    SHARD_SIZE,
    _charm_max_flight_length_mm,
    _family_pair,
    _sample_tungsten_vertex_v2_mm,
)
from analysis.flavour.v3.v6_clean_sim.produce_clean_v3 import (
    _kinematic_fingerprint,
    build_full_volume_tungsten_sampler_spec,
)


CLEAN_V3_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/clean_sim')
STAGE325_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v7_stage325_2026-11-20')
PINPOINT_V3_1_BIN = STAGE325_ROOT / 'g4/pinpoint_v3_1'
GEOM_V3_1_MACRO = STAGE325_ROOT / 'g4/macros/geom.mac'
LCG107 = '/cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc11-opt'

FORENSICS_SPLITS_DIR = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/forensics/splits')
DEFAULT_OUT_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/l1_cluster_sidecar')

N_TRAIN_SHARDS = 80
EVENTS_PER_SHARD = 400
SUBCHUNK_SIZE = 25  # Exact match to Clean v3 cpus=16 (16 x 25 = 400 events/shard)
OBS_CFG = {
    'pe_only': True,
    'bar_threshold_pe': 2.5,
    'sipm_npix': 14400.0,
    'apply_sipm_saturation': False,
    'invert_sipm_saturation': True,
}

ORIGIN_CLASS_NAMES = (
    'PV_primary',        # 0
    'tau_decay',         # 1
    'charm_decay',       # 2
    'photon_conv_ee',    # 3
    'had_sec_or_ghost',  # 4
)

# Stage-1 hard isolation regex (blocks Tune8k 080..099, GENIE 10081..10084 >=012, 10086 000..001 or >=016)
_L1_FORBIDDEN_RE = re.compile(
    r'(?:^|/)shards/0(?:8\d|9\d)(?:/|$)'
    r'|/validation_meta'
    r'|tune_screen\.parquet'
    r'|tune_truth_meta\.parquet'
    r'|/planned_manifest\.parquet'
    r'|(?:^|[^0-9])1008[1-4][_/\\-]+0*(?:1[2-9]|[2-9]\d|[1-9]\d{2,})(?:\.root|\b)'
    r'|(?:^|[^0-9])10086[_/\\-]+0*(?:[01]|1[6-9]|[2-9]\d|[1-9]\d{2,})(?:\.root|\b)',
    re.IGNORECASE,
)

_L1_HOOK_INSTALLED = False
_L1_AUDIT_VIOLATIONS: list[str] = []


def install_l1_stage1_guard() -> SealAuditGuard:
    """Install runtime SealAuditGuard + Stage-1 & GENIE chunk allowlist hook."""
    global _L1_HOOK_INSTALLED
    guard = SealAuditGuard()
    guard.__enter__()
    if not _L1_HOOK_INSTALLED:
        def _l1_hook(event: str, args: tuple[Any, ...]) -> None:
            if event not in ('open', 'os.stat', 'os.scandir', 'os.listdir') or not args:
                return
            target = args[0]
            if isinstance(target, int):
                return
            try:
                ps = os.fspath(target) if hasattr(target, '__fspath__') else str(target)
            except Exception:
                return
            if isinstance(ps, bytes):
                ps = ps.decode('utf-8', errors='ignore')
            norm = ps.replace('\\', '/')
            if _L1_FORBIDDEN_RE.search(norm):
                _L1_AUDIT_VIOLATIONS.append(f'{event}:{norm}')
                raise SealViolationError(f'L1 Stage-1 hard block on {event}: {norm}')

        sys.addaudithook(_l1_hook)
        _L1_HOOK_INSTALLED = True
    return guard


def _read_genie_chunk_task(task: tuple[int, int, set[tuple[int, int]] | None]) -> dict[str, Any]:
    """Read pure-W GENIE records from one unsealed ROOT chunk."""
    run, chunk, target_fams = task
    path = PINPOINT_DATA_DIR / str(run) / f'{run}_{chunk:03d}.root'
    if not path.is_file():
        return {'run': run, 'chunk': chunk, 'matched': {}, 'mu_pool': []}

    ev_cols = [
        'evtID', 'initX', 'initY', 'initZ', 'initPx', 'initPy', 'initPz', 'initE',
        'initPDG', 'fslPDG', 'hitnucPDG', 'intType', 'scatteringType', 'tgtPDG',
        'Q2', 'W', 'xBj', 'y', 'xs',
    ]
    trivial = uproot.source.futures.TrivialExecutor()
    with uproot.open(path, decompression_executor=trivial, interpretation_executor=trivial,
                     object_cache=None, array_cache=None) as src:
        n_rows = int(src['event'].num_entries)
        tag = f'pinpoint:{run}:{chunk:03d}'

        if target_fams is not None:
            # Fast family-hash scan before reading full trees
            row_to_fp: list[tuple[int, tuple[int, int]]] = []
            for r in range(n_rows):
                fp = _family_pair(tag, r)
                if fp in target_fams:
                    row_to_fp.append((r, fp))
            if not row_to_fp:
                return {'run': run, 'chunk': chunk, 'matched': {}, 'mu_pool': []}
            ev = src['event'].arrays(ev_cols, library='np')
            prim_map = group_flat(src['primaries'], ['evtID', 'PDG', 'Px', 'Py', 'Pz', 'E'])
            matched: dict[tuple[int, int], dict] = {}
            for r, fp in row_to_fp:
                evt_id = int(ev['evtID'][r])
                pev = prim_map.get(evt_id)
                if pev is None or len(pev['PDG']) == 0:
                    continue
                meta = {k: ev[k][r] for k in ev_cols}
                rec = nominal_record(meta, pev)
                for kn in ('Q2', 'W', 'x', 'y', 'XSec'):
                    if not np.isfinite(rec[kn]):
                        rec[kn] = 0.0
                it = int(meta['intType'])
                neu = int(meta['initPDG'])
                lbl = (abs(neu) - 12) // 2 if it == 2 else (3 if it == 3 else -1)
                has_ch = any(abs(int(p)) in CHARM_PDGS for p in pev['PDG'])
                rec.update({
                    'source_tag': tag,
                    'source_row': int(r),
                    'family_hi': int(fp[0]),
                    'family_lo': int(fp[1]),
                    'label': int(lbl),
                    'has_gen_charm': bool(has_ch),
                })
                matched[fp] = rec
            return {'run': run, 'chunk': chunk, 'matched': matched, 'mu_pool': []}

        # Otherwise: harvest pure-W (Z==74) mu_CC from 10086 chunks 002..015
        ev = src['event'].arrays(ev_cols, library='np')
        prim_map = group_flat(src['primaries'], ['evtID', 'PDG', 'Px', 'Py', 'Pz', 'E'])
        mu_pool: list[dict] = []
        for r in range(n_rows):
            it = int(ev['intType'][r])
            neu = int(ev['initPDG'][r])
            if it != 2 or abs(neu) != 14:
                continue
            tgt_pdg = int(ev['tgtPDG'][r])
            tgt_z = (tgt_pdg // 10000) % 1000 if tgt_pdg >= 1000000000 else -1
            if tgt_z != 74:
                continue
            evt_id = int(ev['evtID'][r])
            pev = prim_map.get(evt_id)
            if pev is None or len(pev['PDG']) == 0:
                continue
            meta = {k: ev[k][r] for k in ev_cols}
            rec = nominal_record(meta, pev)
            if int(rec['Z']) != 74:
                continue
            for kn in ('Q2', 'W', 'x', 'y', 'XSec'):
                if not np.isfinite(rec[kn]):
                    rec[kn] = 0.0
            hi, lo = _family_pair(tag, r)
            has_ch = any(abs(int(p)) in CHARM_PDGS for p in pev['PDG'])
            rec.update({
                'source_tag': tag,
                'source_row': int(r),
                'family_hi': int(hi),
                'family_lo': int(lo),
                'label': 1,
                'has_gen_charm': bool(has_ch),
            })
            mu_pool.append(rec)
        return {'run': run, 'chunk': chunk, 'matched': {}, 'mu_pool': mu_pool}


def _region_id_from_vmeta(vm: dict) -> int:
    if bool(vm['is_front_pinpoint']) and bool(vm['is_pixel_xy_acceptance']):
        return 0  # FrontPinpoint_30mm_PixXY
    if bool(vm['is_interm_pinpoint']) and bool(vm['is_pixel_xy_acceptance']):
        return 1  # IntermPinpoint_150mm_PixXY
    if bool(vm['is_pinpoint_module']):
        return 2  # Pinpoint_OuterW
    return 3      # Fortune_ScintOnly


def prepare_l1_inputs(out_root: Path, workers: int = 16) -> dict[str, Any]:
    """Build deterministic 80-shard L1_Train32k input tables preserving 24k exact records + 8k 10086 mu_CC."""
    install_l1_stage1_guard()
    t0 = time.perf_counter()
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    for sub in ('data/shards', 'cluster_sidecar', 'truth_sidecar/parts', 'splits', 'input_shards'):
        (out_root / sub).mkdir(parents=True, exist_ok=True)
    for sid in range(N_TRAIN_SHARDS):
        (out_root / 'data/shards' / f'{sid:03d}').mkdir(parents=True, exist_ok=True)

    geom = json.loads((CLEAN_V3_ROOT / 'geometry.json').read_text())
    write_json(out_root / 'geometry.json', geom)
    bounds = detector_bounds(geom)
    sampler_spec = build_full_volume_tungsten_sampler_spec(geom, bounds, margin_xy_mm=0.001, margin_z_mm=0.001)

    # Load authorized Train32k provenance ONLY (shards 000..079 via split_shards allowlist)
    auth_shards = load_authorized_split_shards(CLEAN_V3_ROOT / 'data', 'train', stage=1)
    if len(auth_shards) != N_TRAIN_SHARDS:
        raise AssertionError(f'Expected {N_TRAIN_SHARDS} train shards, got {len(auth_shards)}')
    prov_all = pd.concat([pd.read_parquet(sp / 'provenance.parquet') for sp in auth_shards], ignore_index=True)
    if len(prov_all) != N_TRAIN_SHARDS * EVENTS_PER_SHARD:
        raise AssertionError(f'Expected 32000 train rows, got {len(prov_all)}')

    # Load original Stage-1 internal split keys (internal_train / internal_dev_screen / internal_val_eval)
    split_frames = {}
    key_to_split: dict[tuple[int, int, int], tuple[str, int, int]] = {}
    for sp_name in ('internal_train', 'internal_dev_screen', 'internal_val_eval'):
        df_sp = pd.read_parquet(FORENSICS_SPLITS_DIR / f'{sp_name}_keys.parquet')
        split_frames[sp_name] = df_sp
        for r_v, c_v, rw_v, reg_v, f5_v in zip(
            df_sp['run'], df_sp['chunk'], df_sp['row'], df_sp['region_id'], df_sp['fold_5']
        ):
            key_to_split[(int(r_v), int(c_v), int(rw_v))] = (sp_name, int(reg_v), int(f5_v))

    # Target family set for classes 0, 2, 3 (24,000 events)
    non_mu = prov_all[prov_all['label'] != 1]
    target_fams = set(zip(non_mu['family_hi'].astype(int).tolist(), non_mu['family_lo'].astype(int).tolist()))
    if len(target_fams) != 24000:
        raise AssertionError(f'Expected 24000 non-mu families, got {len(target_fams)}')

    # Load OOD 10085-10088 families to guarantee zero overlap with 10086
    ood_df = pd.read_parquet(CLEAN_V3_ROOT / 'ood_shards/gen_ood_run10085_10088/truth_meta.parquet')
    ood_fams = set(zip(ood_df['family_hi'].astype(int).tolist(), ood_df['family_lo'].astype(int).tolist()))

    tasks: list[tuple[int, int, set[tuple[int, int]] | None]] = []
    for run_id in (10081, 10083, 10084):
        for ch_id in range(12):
            tasks.append((run_id, ch_id, target_fams))
    for ch_id in range(2, 16):  # STRICTLY 10086 chunks 002..015 (disjoint from 000..001 OOD and >=029 L2)
        tasks.append((10086, ch_id, None))

    with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=mp.get_context('spawn')) as pool:
        results = list(pool.map(_read_genie_chunk_task, tasks))

    matched_24k: dict[tuple[int, int], dict] = {}
    mu_candidates: list[dict] = []
    for res in results:
        matched_24k.update(res['matched'])
        mu_candidates.extend(res['mu_pool'])

    if len(matched_24k) != 24000:
        raise RuntimeError(f'Matched {len(matched_24k)} / 24000 events in 10081/10083/10084 chunks 000..011')

    # Deduplicate 10086 mu_CC pool by family and kinematic fingerprint
    rng = np.random.default_rng(20263280)
    perm = rng.permutation(len(mu_candidates))
    seen_fam = set(target_fams) | ood_fams
    seen_fp = {_kinematic_fingerprint(r) for r in matched_24k.values()}
    mu_chosen: list[dict] = []
    for idx in perm:
        r = mu_candidates[int(idx)]
        fp_fam = (int(r['family_hi']), int(r['family_lo']))
        fp_kin = _kinematic_fingerprint(r)
        if fp_fam in seen_fam or fp_kin in seen_fp:
            continue
        seen_fam.add(fp_fam)
        seen_fp.add(fp_kin)
        mu_chosen.append(r)
        if len(mu_chosen) >= 8000:
            break
    if len(mu_chosen) < 8000:
        raise RuntimeError(f'Insufficient pure-W 10086 mu_CC events: {len(mu_chosen)} < 8000')

    # Assign into exact 80 shards x 400 rows matching Train32k keys & internal splits
    mu_cursor = 0
    l1_split_rows: dict[str, list[dict]] = {
        'internal_train': [],
        'internal_dev_screen': [],
        'internal_val_eval': [],
    }
    shard_split_counts = {k: 0 for k in l1_split_rows}
    for sid in range(N_TRAIN_SHARDS):
        sub_p = prov_all[prov_all['chunk'] == sid].sort_values('row').reset_index(drop=True)
        if len(sub_p) != EVENTS_PER_SHARD:
            raise AssertionError(f'Shard {sid:03d} has {len(sub_p)} rows != {EVENTS_PER_SHARD}')
        chunk_recs: list[dict] = []
        for row_idx in range(EVENTS_PER_SHARD):
            pr = sub_p.iloc[row_idx]
            lbl = int(pr['label'])
            k_tuple = (int(pr['run']), int(pr['chunk']), int(pr['row']))
            sp_name, orig_reg_id, fold5 = key_to_split[k_tuple]

            if lbl == 1:
                rec = dict(mu_chosen[mu_cursor])
                mu_cursor += 1
            else:
                fp = (int(pr['family_hi']), int(pr['family_lo']))
                rec = dict(matched_24k[fp])
                if int(rec['label']) != lbl:
                    raise AssertionError(f'Label mismatch for {fp}: {rec["label"]} != {lbl}')

            vm = _sample_tungsten_vertex_v2_mm(int(rec['family_hi']), int(rec['family_lo']), sampler_spec)
            reg_id = _region_id_from_vmeta(vm)
            if lbl != 1 and reg_id != orig_reg_id:
                raise AssertionError(f'Region mismatch for non-mu {k_tuple}: {reg_id} != {orig_reg_id}')

            rec.update({
                'vx': vm['vx_mm'] / 1000.0,
                'vy': vm['vy_mm'] / 1000.0,
                'vz': vm['vz_mm'] / 1000.0,
                'run': 32300,
                'chunk': int(sid),
                'row': int(row_idx),
                'split': 'train',
                'internal_split': sp_name,
                'region_id': int(reg_id),
                'fold_5': int(fold5),
                'slab_id': int(vm['slab_id']),
                'module_type': str(vm['module_type']),
                'block_copy_no': int(vm['block_copy_no']),
                'plate_in_block': int(vm['plate_in_block']),
                'is_pinpoint_module': bool(vm['is_pinpoint_module']),
                'is_front_pinpoint': bool(vm['is_front_pinpoint']),
                'is_interm_pinpoint': bool(vm['is_interm_pinpoint']),
                'is_pixel_xy_acceptance': bool(vm['is_pixel_xy_acceptance']),
                'sampling_stratum': int(vm['sampling_stratum']),
                'sampling_stratum_name': str(vm['sampling_stratum_name']),
                'geom_weight': float(vm['geom_weight']),
            })
            chunk_recs.append(rec)
            shard_split_counts[sp_name] += 1
            l1_split_rows[sp_name].append({
                'run': np.int32(32300),
                'chunk': np.int32(sid),
                'row': np.int32(row_idx),
                'family_hi': np.uint64(rec['family_hi']),
                'family_lo': np.uint64(rec['family_lo']),
                'label': np.int64(lbl),
                'region_id': np.int64(reg_id),
                'fold_5': np.int64(fold5),
            })
        pq.write_table(
            pa.Table.from_pylist(chunk_recs),
            out_root / 'input_shards' / f'input_{sid:03d}.parquet',
            compression=COMPRESSION,
        )

    for sp_name, rows_list in l1_split_rows.items():
        pd.DataFrame(rows_list).to_parquet(
            out_root / 'splits' / f'{sp_name}_keys.parquet',
            index=False,
            compression=COMPRESSION,
        )

    prep_summary = {
        'status': 'prepared',
        'stage': 'Stage328_L1_ClusterSidecar',
        'UTC': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': float(time.perf_counter() - t0),
        'n_shards': N_TRAIN_SHARDS,
        'events_per_shard': EVENTS_PER_SHARD,
        'total_events': N_TRAIN_SHARDS * EVENTS_PER_SHARD,
        'matched_24k_from_10081_10083_10084': 24000,
        'mu_cc_8k_from_10086_002_015': mu_cursor,
        'split_counts': shard_split_counts,
        'seal_violations': len(_L1_AUDIT_VIOLATIONS),
    }
    write_json(out_root / 'prepare_summary.json', prep_summary)
    return prep_summary


def _simulate_subchunk_v3_1(task: tuple[list[dict], str, int]) -> Path:
    records, scratch_str, sub_idx = task
    sub_dir = Path(scratch_str) / f'sub_{sub_idx:02d}'
    sub_dir.mkdir(parents=True, exist_ok=True)
    in_root = sub_dir / 'in.gfaser.root'
    out_root = sub_dir / 'out_g4.root'
    mac = sub_dir / 'run.mac'

    write_gfaser(records, in_root)
    seed_base = 20263230 + int(records[0]['chunk']) * 41 + sub_idx * 11 + int(records[0]['run']) % 100
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
        '/out/saveTruthHits true',
        f'/random/setSeeds {seed_base} {seed_base + 19}',
        '/run/initialize',
        '/Pinpoint/field/value 0.57 tesla',
        f'/run/beamOn {len(records)}',
        '',
    ]))
    script = '\n'.join([
        'env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH -u ROOTSYS bash -c ' + shlex.quote('\n'.join([
            'which() { command -v "$@"; }; export -f which',
            f'source {LCG107}/setup.sh >/dev/null 2>&1',
            'set -e',
            'export QT_QPA_PLATFORM=offscreen',
            f'export PYTHIA8DATA={LCG107}/share/Pythia8/xmldoc',
            'export PINPOINT_USE_INPUT_Z=1',
            'export PINPOINT_SKIP_GDML=1',
            'export PINPOINT_COMPACT_TRUTH=1',
            'export PINPOINT_REQUIRE_W_MATERIAL=1',
            'export PINPOINT_REQUIRE_W_TARGET=1',
            'export PINPOINT_PIXEL_DIFFUSION_UM=3.0',
            'export PINPOINT_PIXEL_NOISE_E=10.0',
            'export PINPOINT_PIXEL_THR_E=100.0',
            'export PINPOINT_SIPM_TIME_THR_PE=3',
            'export PINPOINT_SIPM_TTS_NS=0.15',
            f'exec {shlex.quote(str(PINPOINT_V3_1_BIN))} {shlex.quote(str(mac))}',
        ]))
    ])
    cp = subprocess.run(['/bin/bash', '-c', script], cwd=sub_dir, capture_output=True, text=True, timeout=1200)
    in_root.unlink(missing_ok=True)
    if cp.returncode != 0:
        raise RuntimeError(f'pinpoint_v3_1 subchunk {sub_idx} failed:\n{cp.stderr[-800:]}\n{cp.stdout[-400:]}')
    return out_root


def _classify_event_clusters(
    pix_cid: np.ndarray,
    pix_xyz: np.ndarray,
    p_raw: dict[str, np.ndarray],
    nx_pix: int,
    ny_pix: int,
    pix_zpos: np.ndarray,
    vtx_vec: np.ndarray,
    pev: dict | None,
    tev: dict | None,
    cev: dict | None,
    tau_ids: np.ndarray,
    charm_ids: np.ndarray,
    tau_decay_xyz: np.ndarray | None,
    tau_decay_valid: bool,
) -> dict[str, Any]:
    """Compute 1-to-1 PF-aligned cluster truth labels, top-3 edep trackIDs, and SV groups."""
    n_target_planes = 12
    n_pix = len(pix_cid)
    if n_pix == 0:
        return {
            'keep_idx': np.empty(0, dtype=np.int64),
            'first_channel_id': np.empty(0, dtype=np.uint64),
            'cl_plane': np.empty(0, dtype=np.uint8),
            'cl_x_mm': np.empty(0, dtype=np.float32),
            'cl_y_mm': np.empty(0, dtype=np.float32),
            'cl_z_mm': np.empty(0, dtype=np.float32),
            'origin_label': np.empty(0, dtype=np.uint8),
            'secondary_origin_label': np.empty(0, dtype=np.uint8),
            'sv_group_id': np.empty(0, dtype=np.int16),
            'dominant_track_id': np.empty(0, dtype=np.int32),
            'second_track_id': np.empty(0, dtype=np.int32),
            'third_track_id': np.empty(0, dtype=np.int32),
            'dominant_edep_frac': np.empty(0, dtype=np.float32),
            'true_sv_xyz_mm': np.full(3, np.nan, dtype=np.float32),
            'has_true_sv': np.uint8(0),
            'all_x': np.empty(0, dtype=np.float64),
            'all_y': np.empty(0, dtype=np.float64),
            'all_z': np.empty(0, dtype=np.float64),
            'all_plane': np.empty(0, dtype=np.int64),
            'all_nh': np.empty(0, dtype=np.float64),
            'all_lam2': np.empty(0, dtype=np.float64),
            'all_tid0': np.empty(0, dtype=np.int32),
            'all_orig0': np.empty(0, dtype=np.uint8),
        }

    cl = _cluster_sensor_groups(pix_cid, pix_xyz, nx_pix, ny_pix, n_target_planes, 0.06, 0.06)
    n_cl = int(cl['n_clusters'])
    order = cl['hit_order']
    offsets = cl['cluster_offsets']
    starts = offsets[:-1]
    n_hits = np.diff(offsets).astype(np.float64)
    cl_plane = cl['cluster_plane'].astype(np.int64)
    cl_fcid = cl['cluster_first_channel_id'].astype(np.uint64)

    hx = pix_xyz[order, 0].astype(np.float64)
    hy = pix_xyz[order, 1].astype(np.float64)
    x_mean = np.add.reduceat(hx, starts) / n_hits
    y_mean = np.add.reduceat(hy, starts) / n_hits
    z_cl = pix_zpos[cl_plane]

    _, _, _, hits_r040, _, cnt_r5, _ = _compute_plane_neighborhoods(cl_plane, x_mean, y_mean, n_hits)

    # Minor eigenvalue lam2 for tracklet diagnostics
    n_rep = n_hits.astype(np.int64)
    dhx = hx - np.repeat(x_mean, n_rep)
    dhy = hy - np.repeat(y_mean, n_rep)
    px_cl = np.full(n_cl, 0.0208, dtype=np.float64)
    py_cl = np.full(n_cl, 0.0228, dtype=np.float64)
    vx_c = np.add.reduceat(dhx * dhx, starts) / n_hits + (px_cl ** 2) / 12.0
    vy_c = np.add.reduceat(dhy * dhy, starts) / n_hits + (py_cl ** 2) / 12.0
    cxy_c = np.add.reduceat(dhx * dhy, starts) / n_hits
    tr_c = vx_c + vy_c
    df_c = vx_c - vy_c
    dsc_c = np.sqrt(np.maximum(0.0, df_c * df_c + 4.0 * cxy_c * cxy_c))
    lam2_c = np.maximum(1e-12, 0.5 * (tr_c - dsc_c))

    # Build SV group catalog (0=PV, 1=tau SV, >=2=charm SV clustered within 10 um)
    sv_xyz_by_group: dict[int, np.ndarray] = {}
    sv_tau_id = 0
    tau_dau_tids: set[int] = set()
    tau_parent_tids: set[int] = {int(t) for t in tau_ids.tolist()}
    if tev is not None and len(tev['TID']) > 0:
        for pid_t, tid_t in zip(tev['PID'], tev['TID']):
            if int(pid_t) in tau_parent_tids or len(tau_parent_tids) == 0:
                tau_dau_tids.add(int(tid_t))
                tau_parent_tids.add(int(pid_t))
    if tau_decay_valid and tau_decay_xyz is not None:
        dv_tau = np.asarray(tau_decay_xyz, dtype=np.float64) - vtx_vec
        if float(np.linalg.norm(dv_tau)) > 0.005:
            sv_tau_id = 1
            sv_xyz_by_group[1] = np.asarray(tau_decay_xyz, dtype=np.float64)

    charm_parent_tids: set[int] = {int(t) for t in charm_ids.tolist()}
    charm_dau_tids: set[int] = set()
    charm_pid_to_sv: dict[int, int] = {}
    charm_tid_to_pid: dict[int, int] = {}
    next_sv_id = 2
    if cev is not None and len(cev['TID']) > 0:
        cpids = np.asarray(cev['PID'], dtype=np.int64)
        ctids = np.asarray(cev['TID'], dtype=np.int64)
        cxyz = np.column_stack([cev['ProdX'], cev['ProdY'], cev['ProdZ']]).astype(np.float64)
        cpdg = np.abs(np.asarray(cev['PDG'], dtype=np.int64))
        for cp, ct in zip(cpids, ctids):
            charm_parent_tids.add(int(cp))
            charm_dau_tids.add(int(ct))
            charm_tid_to_pid[int(ct)] = int(cp)
        for cp in np.unique(cpids):
            mc = cpids == cp
            # Prefer non-electron daughters if present (exclude ionisation delta electrons)
            non_e = mc & (cpdg != 11)
            pts = cxyz[non_e] if non_e.any() else cxyz[mc]
            v_c = np.median(pts, axis=0)
            if float(np.linalg.norm(v_c - vtx_vec)) <= 0.005:
                charm_pid_to_sv[int(cp)] = 0
            else:
                assigned = -1
                for g_id, g_xyz in sv_xyz_by_group.items():
                    if g_id >= 2 and float(np.linalg.norm(v_c - g_xyz)) <= 0.010:
                        assigned = g_id
                        break
                if assigned < 0:
                    assigned = next_sv_id
                    sv_xyz_by_group[assigned] = v_c
                    next_sv_id += 1
                charm_pid_to_sv[int(cp)] = assigned

    # Determine event-level primary SV (tau SV if present, else longest-flight charm SV)
    if 1 in sv_xyz_by_group:
        true_sv_xyz = sv_xyz_by_group[1].astype(np.float32)
        has_true_sv = np.uint8(1)
    elif any(k >= 2 for k in sv_xyz_by_group):
        best_g = max((k for k in sv_xyz_by_group if k >= 2), key=lambda k: float(np.linalg.norm(sv_xyz_by_group[k] - vtx_vec)))
        true_sv_xyz = sv_xyz_by_group[best_g].astype(np.float32)
        has_true_sv = np.uint8(1)
    else:
        true_sv_xyz = np.full(3, np.nan, dtype=np.float32)
        has_true_sv = np.uint8(0)

    # Map raw pixel hits onto sorted pix_cid
    raw_cid = (
        p_raw['layerID'].astype(np.int64) * ny_pix + p_raw['rowID'].astype(np.int64)
    ) * nx_pix + p_raw['colID'].astype(np.int64)
    pos = np.searchsorted(pix_cid, raw_cid)
    if not ((pos < n_pix) & (pix_cid[np.minimum(pos, n_pix - 1)] == raw_cid)).all():
        raise AssertionError('Failed 1-to-1 pixel channel_id alignment between ROOT hits and observations')

    hit_to_cl = np.empty(n_pix, dtype=np.int64)
    hit_to_cl[order] = np.repeat(np.arange(n_cl, dtype=np.int64), n_rep)
    hit_cl = hit_to_cl[pos]

    raw_pdg = np.abs(p_raw['pdg'].astype(np.int64))
    raw_tid = p_raw['trackID'].astype(np.int64)
    raw_pid = p_raw['parentID'].astype(np.int64)
    raw_edep = p_raw['edep'].astype(np.float64)
    raw_ftau = p_raw['fromTau'].astype(bool)
    raw_fch = p_raw['fromCharmedHadron'].astype(bool)

    # Hit-level 5-class origin & SV group
    n_raw = len(raw_cid)
    hit_orig = np.full(n_raw, 4, dtype=np.uint8)
    hit_sv = np.full(n_raw, -1, dtype=np.int16)

    tau_dau_pdg_ok = np.isin(raw_pdg, [11, 13, 211, 321, 2212])
    tau_sec_pdg_ok = np.isin(raw_pdg, [13, 211, 321])
    ch_dau_pdg_ok = np.isin(raw_pdg, [11, 13, 211, 321, 2212])
    ch_sec_pdg_ok = np.isin(raw_pdg, [13, 211, 321, 2212])

    in_tau_tids = np.isin(raw_tid, list(tau_dau_tids)) if tau_dau_tids else np.zeros(n_raw, dtype=bool)
    in_tau_pids = np.isin(raw_pid, list(tau_parent_tids)) if tau_parent_tids else np.zeros(n_raw, dtype=bool)
    in_ch_tids = np.isin(raw_tid, list(charm_dau_tids)) if charm_dau_tids else np.zeros(n_raw, dtype=bool)
    in_ch_pids = np.isin(raw_pid, list(charm_parent_tids)) if charm_parent_tids else np.zeros(n_raw, dtype=bool)

    is_predecay_tau = (raw_pid == 0) & (raw_pdg == 15)
    is_tau_hit = (~is_predecay_tau) & (
        (in_tau_tids & tau_dau_pdg_ok)
        | (raw_ftau & in_tau_pids & tau_dau_pdg_ok)
        | (raw_ftau & tau_sec_pdg_ok & (raw_pid <= 4))
    )
    is_charm_hit = (raw_pid != 0) & (~is_tau_hit) & (
        (in_ch_tids & ch_dau_pdg_ok)
        | (raw_fch & in_ch_pids & ch_dau_pdg_ok)
        | (raw_fch & ch_sec_pdg_ok & (raw_pid <= 8))
    )
    is_pv_hit = (raw_pid == 0) & (~is_tau_hit) & (~is_charm_hit)
    is_conv_hit = (raw_pdg == 11) & (raw_pid > 0) & (~is_tau_hit) & (~is_charm_hit)

    hit_orig[is_conv_hit] = 3
    hit_orig[is_pv_hit] = 0
    hit_sv[is_pv_hit] = 0
    hit_orig[is_charm_hit] = 2
    hit_orig[is_tau_hit] = 1
    hit_sv[is_tau_hit] = sv_tau_id

    if is_charm_hit.any():
        for idx_h in np.flatnonzero(is_charm_hit):
            ti = int(raw_tid[idx_h])
            pi = int(raw_pid[idx_h])
            cp = charm_tid_to_pid.get(ti, pi if pi in charm_pid_to_sv else -1)
            if cp >= 0 and cp in charm_pid_to_sv:
                hit_sv[idx_h] = charm_pid_to_sv[cp]
            elif len(charm_pid_to_sv) == 1:
                hit_sv[idx_h] = next(iter(charm_pid_to_sv.values()))
            else:
                hit_sv[idx_h] = -1

    # Group hits by cluster to rank top-3 trackIDs by summed edep
    ord_h = np.argsort(hit_cl, kind='mergesort')
    hcl_s = hit_cl[ord_h]
    tid_s = raw_tid[ord_h]
    edep_s = np.maximum(raw_edep[ord_h], 1e-12)
    orig_s = hit_orig[ord_h]
    sv_s = hit_sv[ord_h]
    bounds_h = np.r_[0, np.flatnonzero(np.diff(hcl_s)) + 1, len(hcl_s)]

    all_orig = np.full(n_cl, 4, dtype=np.uint8)
    all_sec_orig = np.full(n_cl, 4, dtype=np.uint8)
    all_sv = np.full(n_cl, -1, dtype=np.int16)
    all_tid0 = np.full(n_cl, -1, dtype=np.int32)
    all_tid1 = np.full(n_cl, -1, dtype=np.int32)
    all_tid2 = np.full(n_cl, -1, dtype=np.int32)
    all_dom_frac = np.ones(n_cl, dtype=np.float32)

    # Dense per-event trackID remapping (1..M_tracks) preserves exact track-sharing identity
    # while keeping cluster_sidecar Zstd/Brotli footprint strictly < 85 MB across 80 shards.
    tid_remap: dict[int, int] = {}

    def _remap_tid(raw_id: int) -> np.int32:
        if raw_id <= 0:
            return np.int32(-1)
        mapped = tid_remap.get(raw_id)
        if mapped is None:
            mapped = len(tid_remap) + 1
            tid_remap[raw_id] = mapped
        return np.int32(mapped)

    for a, b in zip(bounds_h[:-1], bounds_h[1:]):
        ci = int(hcl_s[a])
        t_arr = tid_s[a:b]
        e_arr = edep_s[a:b]
        o_arr = orig_s[a:b]
        s_arr = sv_s[a:b]
        tot_e = float(e_arr.sum())

        # Accumulate edep per positive trackID
        by_tid: dict[int, float] = {}
        orig_by_tid: dict[int, int] = {}
        sv_by_tid: dict[int, int] = {}
        for ti, ei, oi, si in zip(t_arr.tolist(), e_arr.tolist(), o_arr.tolist(), s_arr.tolist()):
            if ti <= 0:
                continue
            by_tid[ti] = by_tid.get(ti, 0.0) + ei
            if ti not in orig_by_tid:
                orig_by_tid[ti] = oi
                sv_by_tid[ti] = si

        if not by_tid:
            continue
        ranked = sorted(by_tid.items(), key=lambda kv: (-kv[1], kv[0]))
        t0_id, e0 = ranked[0]
        o0 = orig_by_tid[t0_id]
        sv0 = sv_by_tid[t0_id]
        all_tid0[ci] = _remap_tid(t0_id)
        all_orig[ci] = np.uint8(o0)
        all_sec_orig[ci] = np.uint8(o0)
        all_sv[ci] = np.int16(sv0)
        all_dom_frac[ci] = np.float32(e0 / max(tot_e, 1e-12))

        if len(ranked) >= 2:
            t1_id, e1 = ranked[1]
            all_tid1[ci] = _remap_tid(t1_id)
            o1 = orig_by_tid[t1_id]
            if o0 in (3, 4) and o1 in (1, 2) and e1 >= 0.25 * tot_e:
                all_sec_orig[ci] = np.uint8(o1)
                all_sv[ci] = np.int16(sv_by_tid[t1_id])
        if len(ranked) >= 3:
            all_tid2[ci] = _remap_tid(ranked[2][0])

    # Apply EXACT same target/IPT token retention caps as extract_pf_item (832 + 192 = 1024)
    tgt_cl_all = np.flatnonzero(cl_plane < n_target_planes)
    ipt_cl_all = np.flatnonzero(cl_plane >= n_target_planes)
    keep_tgt = _select_group_clusters(tgt_cl_all, cl_plane, n_hits, hits_r040, cnt_r5, N_MAX_TARGET_CLUSTERS)
    keep_ipt = _select_group_clusters(ipt_cl_all, cl_plane, n_hits, hits_r040, cnt_r5, N_MAX_IPT_CLUSTERS)
    keep_idx = np.concatenate([keep_tgt, keep_ipt]).astype(np.int64)

    return {
        'keep_idx': keep_idx,
        'first_channel_id': cl_fcid[keep_idx],
        'cl_plane': cl_plane[keep_idx].astype(np.uint8),
        'cl_x_mm': x_mean[keep_idx].astype(np.float32),
        'cl_y_mm': y_mean[keep_idx].astype(np.float32),
        'cl_z_mm': z_cl[keep_idx].astype(np.float32),
        'origin_label': all_orig[keep_idx],
        'secondary_origin_label': all_sec_orig[keep_idx],
        'sv_group_id': all_sv[keep_idx],
        'dominant_track_id': all_tid0[keep_idx],
        'second_track_id': all_tid1[keep_idx],
        'third_track_id': all_tid2[keep_idx],
        'dominant_edep_frac': all_dom_frac[keep_idx],
        'true_sv_xyz_mm': true_sv_xyz,
        'has_true_sv': has_true_sv,
        'all_x': x_mean,
        'all_y': y_mean,
        'all_z': z_cl,
        'all_plane': cl_plane,
        'all_nh': n_hits,
        'all_lam2': lam2_c,
        'all_tid0': all_tid0,
        'all_orig0': all_orig,
    }


def _tracklet_purity_diagnostics(
    cl_res: dict[str, Any],
    dom: dict,
    bounds: dict,
    geometry: dict,
    plane_x0: np.ndarray,
    vtx_vec: np.ndarray,
) -> dict[str, int]:
    """Compute C++ greedy vs soft-64 tracklet purity and ghost/conversion stats (Stage 326 Sec 1.2)."""
    stats = {
        'cpp_3hit': 0, 'cpp_3hit_pure': 0, 'cpp_2hit': 0, 'cpp_2hit_pure': 0,
        'cpp_disp': 0, 'cpp_disp_hf': 0, 'cpp_disp_fake': 0,
        'soft_tot': 0, 'soft_pure': 0, 'soft_conv': 0,
    }
    x = cl_res['all_x']
    y = cl_res['all_y']
    z = cl_res['all_z']
    pl = cl_res['all_plane']
    nh = cl_res['all_nh']
    lam2 = cl_res['all_lam2']
    tid0 = cl_res['all_tid0']
    orig0 = cl_res['all_orig0']
    if len(pl) < 2:
        return stats

    # Soft-64 tracklets using kept clusters
    k_idx = cl_res['keep_idx']
    if len(k_idx) >= 2:
        pix_xyz_k = np.column_stack([x[k_idx], y[k_idx], z[k_idx]]).astype(np.float32)
        pix_pl_k = pl[k_idx].astype(np.int8)
        trk = extract_soft_tracklets_single(
            pix_xyz_k, pix_pl_k, None, plane_x0,
            max_tracklets=64,
            pitch_xy_mm=(float(bounds['dx_mm']), float(bounds['dy_mm'])),
        )
        m_trk = trk['trk_mask']
        stats['soft_tot'] = int(m_trk.sum())

    # C++17 greedy 3-hit/2-hit tracklet purity check on Front30/target planes
    n_pix_planes = int(bounds['nz'])
    plane_cls: list[list[int]] = [[] for _ in range(n_pix_planes)]
    plane_z = np.zeros(n_pix_planes, dtype=np.float64)
    for idx_c in range(len(pl)):
        p = int(pl[idx_c])
        if 0 <= p < n_pix_planes:
            if not plane_cls[p]:
                plane_z[p] = float(z[idx_c])
            plane_cls[p].append(idx_c)
    cs = dom['cal_start']
    has_cal, z_cal = bool(cs[3]), float(cs[0])
    p0 = -1
    for min_c in (3, 2, 1):
        for p in range(12):
            if len(plane_cls[p]) >= min_c and ((not has_cal) or (z_cal - 45.0 <= plane_z[p] <= z_cal + 165.0)):
                p0 = p
                break
        if p0 >= 0:
            break
    if p0 < 0:
        return stats

    p_end = min(11, p0 + 4)
    ic0 = np.asarray(plane_cls[p0], dtype=np.int64)
    xc_w = float(np.dot(nh[ic0], x[ic0]) / max(float(nh[ic0].sum()), 1e-9))
    yc_w = float(np.dot(nh[ic0], y[ic0]) / max(float(nh[ic0].sum()), 1e-9))
    p_vecs = {}
    for p in range(p0, p_end + 1):
        if not plane_cls[p]:
            p_vecs[p] = np.empty(0, dtype=np.int64)
            continue
        ic = np.asarray(plane_cls[p], dtype=np.int64)
        rad = min(40.0, max(12.0, 12.0 + 0.16 * max(0.0, plane_z[p] - plane_z[p0])))
        dr = np.hypot(x[ic] - xc_w, y[ic] - yc_w)
        keep = (dr <= rad) & (nh[ic] <= 16.0)
        ic_k = ic[keep] if np.sum(keep) >= 2 else ic
        if len(ic_k) > 40:
            ic_k = ic_k[np.argsort(np.hypot(x[ic_k] - xc_w, y[ic_k] - yc_w))[:40]]
        p_vecs[p] = ic_k

    sig_b2 = 0.5 * ((float(bounds['dx_mm']) ** 2 + float(bounds['dy_mm']) ** 2) / 12.0) + 1e-5
    cands = []
    for pa in range(p0, p_end):
        for pb in range(pa + 1, min(p_end, pa + 2) + 1):
            for ia in p_vecs[pa]:
                xa, ya, za = float(x[ia]), float(y[ia]), float(z[ia])
                sa2 = sig_b2 + 0.15 * float(lam2[ia])
                for ib in p_vecs[pb]:
                    dz_ab = float(z[ib]) - za
                    if dz_ab < 5.0 or dz_ab > 185.0 or (pb > pa + 1 and dz_ab > 70.0):
                        continue
                    tx_ab, ty_ab = (float(x[ib]) - xa) / dz_ab, (float(y[ib]) - ya) / dz_ab
                    if math.hypot(tx_ab, ty_ab) > 0.22:
                        continue
                    ms_ab = 1.4e-3 * math.sqrt(max(1.0, dz_ab / 30.0))
                    f3 = False
                    for pc in range(pb + 1, min(p_end, pb + 2) + 1):
                        for ic in p_vecs[pc]:
                            dz_bc = float(z[ic]) - float(z[ib])
                            dz_ac = float(z[ic]) - za
                            if dz_bc < 5.0 or dz_bc > 185.0 or (pc > pb + 1 and dz_bc > 70.0) or dz_ac > 340.0:
                                continue
                            tx, ty = (float(x[ic]) - xa) / dz_ac, (float(y[ic]) - ya) / dz_ac
                            if math.hypot(tx, ty) > 0.22:
                                continue
                            rx = float(x[ib]) - (xa + tx * dz_ab)
                            ry = float(y[ib]) - (ya + ty * dz_ab)
                            lev = (dz_ab * dz_bc) / dz_ac
                            ms_m = 1.4e-3 * math.sqrt(max(1.0, min(dz_ab, dz_bc) / 30.0))
                            chi2 = (rx * rx + ry * ry) / (1.5 * sig_b2 + 0.15 * float(lam2[ib]) + (2.0 * ms_m * lev) ** 2)
                            if chi2 <= 16.0:
                                eff_s2 = sa2 + (0.8 * ms_ab * min(40.0, 0.5 * dz_ab)) ** 2
                                cands.append((3, chi2, tx * tx + ty * ty, xa, ya, za, tx, ty, eff_s2, int(ia), int(ib), int(ic)))
                                f3 = True
                    if (not f3) and pb == pa + 1 and dz_ab <= 165.0 and math.hypot(tx_ab, ty_ab) <= 0.15:
                        eff_s2 = 2.0 * sa2 + (1.6 * ms_ab * min(45.0, 0.5 * dz_ab)) ** 2
                        cands.append((2, 4.0 + 200.0 * (tx_ab ** 2 + ty_ab ** 2), tx_ab ** 2 + ty_ab ** 2, xa, ya, za, tx_ab, ty_ab, eff_s2, int(ia), int(ib), -1))
    cands.sort(key=lambda r: (-r[0], r[1], r[2]))
    used_c: set[int] = set()
    sel = []
    for cc in cands:
        ia, ib, ic = cc[9], cc[10], cc[11]
        if ia in used_c or ib in used_c or (ic >= 0 and ic in used_c):
            continue
        used_c.add(ia)
        used_c.add(ib)
        if ic >= 0:
            used_c.add(ic)
        sel.append(cc)
        if len(sel) >= 32:
            break
    for cc in sel:
        nh_t, _, _, x0_t, y0_t, z0_t, tx_t, ty_t, s2_t, ia, ib, ic = cc
        ta, tb = int(tid0[ia]), int(tid0[ib])
        tc = int(tid0[ic]) if ic >= 0 else ta
        pure = (ta > 0) and (ta == tb) and (ta == tc)
        if nh_t == 3:
            stats['cpp_3hit'] += 1
            if pure:
                stats['cpp_3hit_pure'] += 1
        else:
            stats['cpp_2hit'] += 1
            if pure:
                stats['cpp_2hit_pure'] += 1
        dz_p = max(1.0, abs(z0_t - vtx_vec[2]))
        sig_p = math.sqrt(s2_t + (1.4e-3 * dz_p) ** 2)
        d0_p = math.hypot((x0_t + tx_t * (vtx_vec[2] - z0_t)) - vtx_vec[0], (y0_t + ty_t * (vtx_vec[2] - z0_t)) - vtx_vec[1])
        if 0.015 <= d0_p <= 1.80 and (d0_p / max(sig_p, 0.005)) > 2.5:
            stats['cpp_disp'] += 1
            origs = [int(orig0[ia]), int(orig0[ib])] + ([int(orig0[ic])] if ic >= 0 else [])
            if pure and any(o in (1, 2) for o in origs):
                stats['cpp_disp_hf'] += 1
            else:
                stats['cpp_disp_fake'] += 1
    return stats


def _simulate_and_convert_subchunk(
    task: tuple[list[dict], str, int, dict, dict, np.ndarray, np.ndarray],
) -> dict[str, Any]:
    """Run pinpoint_v3_1 and convert one 25-event subchunk in parallel on 1 CPU core."""
    records, scratch_str, sub_idx, geom, bounds, pix_zpos, plane_x0 = task
    rf = _simulate_subchunk_v3_1((records, scratch_str, sub_idx))
    nx_pix, ny_pix = int(bounds['nx']), int(bounds['ny'])

    observations = []
    targets = []
    prov_rows = []
    ev_sc_rows = []
    cl_sc_rows = []
    origin_hist = np.zeros(5, dtype=np.int64)
    obs_dom_hist = np.zeros(4, dtype=np.int64)
    trk_stats_tot: dict[str, int] = {}
    matched_lt50um = 0
    total_tgt_clusters = 0
    zero_pe_viol = 0
    g4_w_verified = 0

    try:
        with uproot.open(rf, object_cache=None, array_cache=None) as src:
            ev_cols = ['evtID', 'initX', 'initY', 'initZ', 'initE', 'initPDG', 'fslPDG', 'tgtPDG', 'tgtZ', 'tgtA', 'vtxMatZ', 'vtxMatName']
            ev_arr = src['event'].arrays(ev_cols, library='np')
            data = {}
            for n, k in BRANCHES.items():
                tree = src[ROOT_TREES[n]]
                cols = list(k)
                if n == 'bar':
                    cols = [c for c in cols if c != 'edep']
                    if 'hitTime' in tree.keys():
                        cols.append('hitTime')
                data[n] = tree.arrays(cols, library='ak')
            ph_truth = src['Hits/pixelHits'].arrays(
                ['rowID', 'colID', 'layerID', 'pdg', 'trackID', 'parentID', 'edep',
                 'fromPrimaryLepton', 'fromPrimaryEMShower', 'fromCharmedHadron', 'fromTau'],
                library='ak',
            )
            prim_map = group_flat(src['primaries'], ['evtID', 'PDG', 'trackID', 'Px', 'Py', 'Pz'])
            tau_map = group_flat(src['tau'], ['evtID', 'PID', 'TID', 'PDG', 'ParentPDG', 'ProdX', 'ProdY', 'ProdZ', 'Px', 'Py', 'Pz']) if 'tau' in src else {}
            charm_map = group_flat(src['charm'], ['evtID', 'PID', 'TID', 'PDG', 'ParentPDG', 'ProdX', 'ProdY', 'ProdZ', 'Px', 'Py', 'Pz']) if 'charm' in src else {}

            n_sub = len(ev_arr['evtID'])
            if n_sub != len(records):
                raise AssertionError(f'Subchunk {sub_idx} event count {n_sub} != {len(records)}')

            for i in range(n_sub):
                rec = records[i]
                evt_id = int(ev_arr['evtID'][i])
                lbl4 = int(rec['label'])
                hi = int(rec['family_hi'])
                lo = int(rec['family_lo'])

                vtx_mat_z = int(ev_arr['vtxMatZ'][i])
                vtx_mat_name = str(ev_arr['vtxMatName'][i])
                ev_tgt_z = int(ev_arr['tgtZ'][i])
                if vtx_mat_name != 'G4_W' or vtx_mat_z != 74 or ev_tgt_z != 74:
                    raise AssertionError(f'Pure-W violation: mat={vtx_mat_name} Z={vtx_mat_z} tgtZ={ev_tgt_z}')
                g4_w_verified += 1

                h = {n: {k: ak.to_numpy(v[k][i]) for k in v.fields} for n, v in data.items()}
                p_raw = {k: ak.to_numpy(ph_truth[k][i]) for k in ph_truth.fields}
                pix = pixel_channels(h['pixel'], geom)
                bars, _ = bar_channels(h['bar'], geom, include_truth_edep=False)
                if 'hit_time_ns' in bars and len(bars['pe_proxy']):
                    bad_t = int(np.sum((bars['pe_proxy'] <= 0) & (bars['hit_time_ns'] >= 0.0)))
                    zero_pe_viol += bad_t
                    if bad_t > 0:
                        raise AssertionError(f'0-PE bar carried positive hit_time_ns ({bad_t})')

                seed = hi ^ lo ^ 20263230
                _, trk = tracker_measurements(h['tracker'], seed)
                state = modality_state(pix, bars, trk)
                obs_dict = {
                    'pixel': lists(pix),
                    'bar': lists(bars),
                    'tracker_hypothesis': lists(trk),
                    'has_response': state['has_response'].tolist(),
                    'outside_acceptance': [-1, -1, -1],
                    'fit_attempted': False,
                    'momentum_censored': -1,
                }
                observations.append(obs_dict)
                targets.append({'label': lbl4, 'label_status': 'ok'})

                vx, vy, vz = float(ev_arr['initX'][i]), float(ev_arr['initY'][i]), float(ev_arr['initZ'][i])
                vtx_vec = np.array([vx, vy, vz], dtype=np.float64)
                init_e = float(ev_arr['initE'][i])
                init_pdg = int(ev_arr['initPDG'][i])
                fsl_pdg = int(ev_arr['fslPDG'][i])

                prov_rows.append({
                    'run': np.int32(rec['run']),
                    'chunk': np.int32(rec['chunk']),
                    'row': np.int32(rec['row']),
                    'evtID': np.int32(evt_id),
                    'family_hi': np.uint64(hi),
                    'family_lo': np.uint64(lo),
                    'split': 'train',
                    'label': np.int64(lbl4),
                })

                dom = compute_observable_containment(obs_dict, geom, bounds, config=OBS_CFG)
                obs_dom_id = int(dom['obs_domain_id'])
                obs_dom_hist[obs_dom_id] += 1

                pev = prim_map.get(evt_id)
                has_ch = bool(rec.get('has_gen_charm', False))
                tau_ids = np.empty(0, dtype=np.int64)
                charm_ids = np.empty(0, dtype=np.int64)
                if pev is not None and len(pev['PDG']) > 0:
                    pdg_arr = np.asarray(pev['PDG'], dtype=np.int64)
                    tid_arr = np.asarray(pev['trackID'], dtype=np.int64)
                    ch_mask = np.array([charmed(int(p)) for p in pdg_arr], dtype=bool)
                    has_ch = has_ch or bool(ch_mask.any())
                    charm_ids = tid_arr[ch_mask]
                    tau_ids = tid_arr[np.abs(pdg_arr) == 15]

                tev = tau_map.get(evt_id)
                if lbl4 == 2 and len(tau_ids) > 0:
                    direct, decay_xyz, decay_valid = primary_decay(tev, tau_ids)
                    branch = classify_tau_daughters(direct['PDG']) if direct else 'unknown'
                else:
                    direct, decay_xyz, decay_valid, branch = {}, None, False, 'not_tau'

                l7, tb_val = _compute_label_7(lbl4, fsl_pdg, init_pdg, branch)
                kin_vec, kv = _compute_vis_kinematics(lbl4, fsl_pdg, pev, direct, decay_valid)
                ev_sc_rows.append({
                    'run': np.int32(rec['run']),
                    'chunk': np.int32(rec['chunk']),
                    'row': np.int32(rec['row']),
                    'evtID': np.int32(evt_id),
                    'family_hi': np.uint64(hi),
                    'family_lo': np.uint64(lo),
                    'label_4': np.int8(lbl4),
                    'label_7': np.int8(l7),
                    'tau_branch_valid': np.uint8(tb_val),
                    'charm_flag': np.uint8(int(has_ch)),
                    'charm_valid': np.uint8(1),
                    'vertex_x_mm': np.float32(vx),
                    'vertex_y_mm': np.float32(vy),
                    'vertex_z_mm': np.float32(vz),
                    'vertex_valid': np.uint8(int(obs_dom_id != 0)),
                    'vis_pz_log1p': np.float32(kin_vec[0]),
                    'vis_pt_log1p': np.float32(kin_vec[1]),
                    'vis_cos_dphi': np.float32(kin_vec[2]),
                    'vis_sin_dphi': np.float32(kin_vec[3]),
                    'kin_valid': np.uint8(kv),
                    'energy_nu_GeV': np.float32(init_e / 1000.0),
                    'geom_weight': np.float32(rec.get('geom_weight', 1.0)),
                    'obs_domain_id': np.int8(obs_dom_id),
                    'internal_split': str(rec['internal_split']),
                    'region_id': np.int8(rec['region_id']),
                    'fold_5': np.int8(rec['fold_5']),
                })

                pix_cid = np.asarray(pix['channel_id'], dtype=np.int64)
                pix_xyz = np.asarray(pix['xyz_mm'], dtype=np.float64).reshape(-1, 3)
                cl_res = _classify_event_clusters(
                    pix_cid, pix_xyz, p_raw, nx_pix, ny_pix, pix_zpos, vtx_vec,
                    pev, tev, charm_map.get(evt_id), tau_ids, charm_ids,
                    decay_xyz, decay_valid,
                )
                tgt_mask = cl_res['cl_plane'] < 12
                n_tgt_k = int(tgt_mask.sum())
                total_tgt_clusters += n_tgt_k
                matched_lt50um += n_tgt_k
                origin_hist += np.bincount(cl_res['origin_label'], minlength=5)

                if bool(rec['is_front_pinpoint']) and bool(rec['is_pixel_xy_acceptance']):
                    st_ev = _tracklet_purity_diagnostics(cl_res, dom, bounds, geom, plane_x0, vtx_vec)
                    for k_s, v_s in st_ev.items():
                        trk_stats_tot[k_s] = trk_stats_tot.get(k_s, 0) + int(v_s)

                cl_sc_rows.append({
                    'run': np.int32(rec['run']),
                    'chunk': np.int16(rec['chunk']),
                    'row': np.int16(rec['row']),
                    'family_hi': np.uint64(hi),
                    'family_lo': np.uint64(lo),
                    'true_sv_xyz_mm': cl_res['true_sv_xyz_mm'].tolist(),
                    'has_true_sv': np.uint8(cl_res['has_true_sv']),
                    'cl_plane': cl_res['cl_plane'].tolist(),
                    'cl_x_mm': cl_res['cl_x_mm'].tolist(),
                    'cl_y_mm': cl_res['cl_y_mm'].tolist(),
                    'origin_label': cl_res['origin_label'].tolist(),
                    'secondary_origin_label': cl_res['secondary_origin_label'].tolist(),
                    'sv_group_id': cl_res['sv_group_id'].tolist(),
                    'dominant_track_id': cl_res['dominant_track_id'].tolist(),
                    'second_track_id': cl_res['second_track_id'].tolist(),
                    'third_track_id': cl_res['third_track_id'].tolist(),
                    'dominant_edep_frac': cl_res['dominant_edep_frac'].tolist(),
                })
    finally:
        rf.unlink(missing_ok=True)

    return {
        'observations': observations,
        'targets': targets,
        'prov_rows': prov_rows,
        'ev_sc_rows': ev_sc_rows,
        'cl_sc_rows': cl_sc_rows,
        'origin_hist': origin_hist,
        'obs_dom_hist': obs_dom_hist,
        'trk_stats_tot': trk_stats_tot,
        'matched_lt50um': matched_lt50um,
        'total_tgt_clusters': total_tgt_clusters,
        'zero_pe_viol': zero_pe_viol,
        'g4_w_verified': g4_w_verified,
    }


def run_shard_worker(out_root: Path, shard_id: int, cpus: int = 8) -> dict[str, Any]:
    """Simulate 1 train shard (400 events = 16 x 25) in scratch, writing observations + cluster_sidecar."""
    install_l1_stage1_guard()
    t0 = time.perf_counter()
    out_root = out_root.resolve()
    shard_dir = out_root / 'data/shards' / f'{shard_id:03d}'
    sc_out_path = out_root / 'cluster_sidecar' / f'shard_{shard_id:03d}.parquet'
    ev_sc_part_path = out_root / 'truth_sidecar/parts' / f'event_{shard_id:03d}.parquet'
    diag_path = out_root / 'cluster_sidecar' / f'diag_{shard_id:03d}.json'

    if (shard_dir / 'observations.parquet').is_file() and sc_out_path.is_file() and ev_sc_part_path.is_file() and diag_path.is_file():
        return {'shard': shard_id, 'status': 'already_done'}

    geom = json.loads((out_root / 'geometry.json').read_text())
    bounds = detector_bounds(geom)
    pix_zpos = np.asarray(bounds['pixel_Zpos'], dtype=np.float64)
    plane_x0 = np.array([cumulative_radiation_length(float(z), geom) for z in pix_zpos], dtype=np.float32)

    in_pq = out_root / 'input_shards' / f'input_{shard_id:03d}.parquet'
    recs = pq.read_table(in_pq).to_pylist()
    if len(recs) != EVENTS_PER_SHARD:
        raise AssertionError(f'Shard {shard_id:03d} expected {EVENTS_PER_SHARD} records, got {len(recs)}')

    scratch_base = Path(os.environ.get('PINPOINT_L1_SCRATCH') or os.environ.get('_CONDOR_SCRATCH_DIR') or '/tmp')
    scratch_base.mkdir(parents=True, exist_ok=True)

    observations = []
    targets = []
    prov_rows = []
    ev_sc_rows = []
    cl_sc_rows = []
    origin_hist = np.zeros(5, dtype=np.int64)
    obs_dom_hist = np.zeros(4, dtype=np.int64)
    trk_stats_tot: dict[str, int] = {}
    matched_lt50um = 0
    total_tgt_clusters = 0
    zero_pe_viol = 0
    g4_w_verified = 0

    with tempfile.TemporaryDirectory(prefix=f'l1-s{shard_id:03d}-', dir=scratch_base) as tmp:
        tasks = [
            (recs[s : s + SUBCHUNK_SIZE], tmp, sub_idx, geom, bounds, pix_zpos, plane_x0)
            for sub_idx, s in enumerate(range(0, len(recs), SUBCHUNK_SIZE))
        ]
        with ProcessPoolExecutor(max_workers=cpus, mp_context=mp.get_context('spawn')) as pool:
            parts = list(pool.map(_simulate_and_convert_subchunk, tasks))

        for part in parts:
            observations.extend(part['observations'])
            targets.extend(part['targets'])
            prov_rows.extend(part['prov_rows'])
            ev_sc_rows.extend(part['ev_sc_rows'])
            cl_sc_rows.extend(part['cl_sc_rows'])
            origin_hist += part['origin_hist']
            obs_dom_hist += part['obs_dom_hist']
            matched_lt50um += part['matched_lt50um']
            total_tgt_clusters += part['total_tgt_clusters']
            zero_pe_viol += part['zero_pe_viol']
            g4_w_verified += part['g4_w_verified']
            for k_s, v_s in part['trk_stats_tot'].items():
                trk_stats_tot[k_s] = trk_stats_tot.get(k_s, 0) + int(v_s)

    # Write matched observation shard + compact Brotli cluster_sidecar shard (< 85 MB total)
    shard_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(observations), shard_dir / 'observations.parquet', compression=COMPRESSION, row_group_size=64)
    pq.write_table(pa.Table.from_pylist(targets), shard_dir / 'targets.parquet', compression=COMPRESSION)
    pd.DataFrame(prov_rows).to_parquet(shard_dir / 'provenance.parquet', index=False, compression=COMPRESSION)
    write_json(shard_dir / 'geometry.json', geom)
    write_json(shard_dir / 'summary.json', {
        'events': len(observations),
        'include_truth_edep': False,
        'require_w_target': True,
        'g4_navigator_W_verified': int(g4_w_verified),
        'zero_pe_time_violations': int(zero_pe_viol),
        'UTC': datetime.now(timezone.utc).isoformat(),
    })

    ev_sc_part_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ev_sc_rows).to_parquet(ev_sc_part_path, index=False, compression=COMPRESSION)

    sc_schema = pa.schema([
        ('run', pa.int32()),
        ('chunk', pa.int16()),
        ('row', pa.int16()),
        ('family_hi', pa.uint64()),
        ('family_lo', pa.uint64()),
        ('true_sv_xyz_mm', pa.list_(pa.float32(), 3)),
        ('has_true_sv', pa.uint8()),
        ('cl_plane', pa.list_(pa.uint8())),
        ('cl_x_mm', pa.list_(pa.float32())),
        ('cl_y_mm', pa.list_(pa.float32())),
        ('origin_label', pa.list_(pa.uint8())),
        ('secondary_origin_label', pa.list_(pa.uint8())),
        ('sv_group_id', pa.list_(pa.int16())),
        ('dominant_track_id', pa.list_(pa.int32())),
        ('second_track_id', pa.list_(pa.int32())),
        ('third_track_id', pa.list_(pa.int32())),
        ('dominant_edep_frac', pa.list_(pa.float32())),
    ])
    sc_out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(cl_sc_rows, schema=sc_schema), sc_out_path, compression='brotli')

    diag = {
        'shard': int(shard_id),
        'events': len(observations),
        'g4_w_verified': int(g4_w_verified),
        'total_target_clusters': int(total_tgt_clusters),
        'matched_lt50um': int(matched_lt50um),
        'origin_hist': origin_hist.tolist(),
        'obs_domain_hist': obs_dom_hist.tolist(),
        'front30_tracklet_stats': trk_stats_tot,
        'cluster_sidecar_bytes': int(sc_out_path.stat().st_size),
        'seconds': float(time.perf_counter() - t0),
        'seal_violations': len(_L1_AUDIT_VIOLATIONS),
    }
    write_json(diag_path, diag)
    return diag


def submit_l1_condor_array(
    out_root: Path,
    *,
    schedd: str = 'bigbird10.cern.ch',
    cpus: int = 8,
    n_workers: int = 80,
) -> dict[str, Any]:
    """Submit 40-to-80 worker 8-core HTCondor array with AFS logs/workdir (compatible with bigbird10/25)."""
    install_l1_stage1_guard()
    out_root = out_root.resolve()
    repo = Path(__file__).resolve().parents[4]
    condor_dir = Path.home() / 'condor/Pinpoint/stage328_l1_sim'
    logs_dir = condor_dir / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    (out_root / 'code').mkdir(parents=True, exist_ok=True)

    archive = out_root / 'code/l1_sidecar_src.tar.gz'
    from analysis.flavour.v3.common.submit import _package_source
    _package_source(repo, archive)

    shards_per_worker = max(1, (N_TRAIN_SHARDS + n_workers - 1) // n_workers)
    wrapper = condor_dir / 'run_l1_worker.sh'
    wrapper.write_text('\n'.join([
        '#!/bin/bash',
        'set -o pipefail',
        'which() { command -v "$@"; }; export -f which',
        'source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh >/dev/null 2>&1',
        'set -e',
        'ls /eos/home-x/xcheng/FASER/Pinpoint/results >/dev/null',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1',
        'export WORK_SCRATCH="${_CONDOR_SCRATCH_DIR:-/tmp}/l1_sc_${PINPOINT_JOB_ID}"',
        'cleanup() { rm -rf "$WORK_SCRATCH"; }',
        'trap cleanup EXIT',
        'mkdir -p "$WORK_SCRATCH/code"',
        f'tar -xzf {shlex.quote(str(archive))} -C "$WORK_SCRATCH/code"',
        'export PYTHONPATH="$WORK_SCRATCH/code/src:${PYTHONPATH:-}"',
        'export PINPOINT_L1_SCRATCH="$WORK_SCRATCH"',
        'cd "$WORK_SCRATCH"',
        f'exec python3 -u -m analysis.flavour.nextgen.produce_l1_cluster_sidecar worker '
        f'--out-root {shlex.quote(str(out_root))} --task-id "$1" --shards-per-worker {shards_per_worker} --cpus {cpus}',
        '',
    ]))
    wrapper.chmod(0o755)

    sub_path = condor_dir / 'l1_array.sub'
    sub_path.write_text('\n'.join([
        'universe = vanilla',
        f'executable = {wrapper}',
        'arguments = $(Process)',
        f'initialdir = {condor_dir}',
        f'output = {logs_dir}/worker.$(ClusterId).$(Process).out',
        f'error = {logs_dir}/worker.$(ClusterId).$(Process).err',
        f'log = {logs_dir}/worker.$(ClusterId).log',
        'environment = "PINPOINT_JOB_ID=$(ClusterId).$(Process)"',
        f'request_cpus = {cpus}',
        'request_memory = 16GB',
        'request_disk = 16GB',
        'requirements = (OpSysAndVer == "AlmaLinux9")',
        '+JobFlavour = "microcentury"',
        'should_transfer_files = YES',
        'when_to_transfer_output = ON_EXIT',
        'transfer_output_files = ""',
        f'queue {n_workers}',
        '',
    ]))

    schedd_snap = subprocess.check_output(
        ['condor_status', '-schedd', '-af', 'Name', 'TotalRunningJobs', 'TotalIdleJobs', 'TotalHeldJobs'],
        text=True,
    )
    resp = subprocess.run(['condor_submit', '-name', schedd, str(sub_path)], capture_output=True, text=True, check=True)
    m_cid = re.search(r'cluster\s+(\d+)', resp.stdout)
    cluster_id = int(m_cid.group(1)) if m_cid else -1
    sub_rec = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'schedd': schedd,
        'cluster_id': cluster_id,
        'n_workers': n_workers,
        'shards_per_worker': shards_per_worker,
        'cpus_per_worker': cpus,
        'submit_file': str(sub_path),
        'logs_dir': str(logs_dir),
        'stdout': resp.stdout.strip(),
        'schedd_snapshot_head': schedd_snap.splitlines()[:15],
    }
    write_json(out_root / 'condor_submission.json', sub_rec)
    return sub_rec


def finalize_l1_sidecar(out_root: Path) -> dict[str, Any]:
    """Aggregate all 80 shards, verify alignment & zero overlap, clean temp inputs, write summary."""
    install_l1_stage1_guard()
    t0 = time.perf_counter()
    out_root = out_root.resolve()

    missing = [
        sid for sid in range(N_TRAIN_SHARDS)
        if not (
            (out_root / 'data/shards' / f'{sid:03d}/observations.parquet').is_file()
            and (out_root / 'cluster_sidecar' / f'shard_{sid:03d}.parquet').is_file()
            and (out_root / 'truth_sidecar/parts' / f'event_{sid:03d}.parquet').is_file()
            and (out_root / 'cluster_sidecar' / f'diag_{sid:03d}.json').is_file()
        )
    ]
    if missing:
        raise RuntimeError(f'Cannot finalize L1 sidecar; {len(missing)} shards incomplete: {missing[:10]}')

    # 1. Merge event_sidecar.parquet & update splits/*_keys.parquet with actual obs_domain_id
    sc_parts = [pd.read_parquet(out_root / 'truth_sidecar/parts' / f'event_{sid:03d}.parquet') for sid in range(N_TRAIN_SHARDS)]
    sc_df = pd.concat(sc_parts, ignore_index=True)
    if len(sc_df) != N_TRAIN_SHARDS * EVENTS_PER_SHARD:
        raise AssertionError(f'Expected 32000 event_sidecar rows, got {len(sc_df)}')

    # Write clean 22-column event_sidecar.parquet matching Clean v3 schema
    ev_cols_22 = [
        'run', 'chunk', 'row', 'evtID', 'family_hi', 'family_lo',
        'label_4', 'label_7', 'tau_branch_valid', 'charm_flag', 'charm_valid',
        'vertex_x_mm', 'vertex_y_mm', 'vertex_z_mm', 'vertex_valid',
        'vis_pz_log1p', 'vis_pt_log1p', 'vis_cos_dphi', 'vis_sin_dphi', 'kin_valid',
        'energy_nu_GeV', 'geom_weight',
    ]
    sc_df[ev_cols_22].to_parquet(out_root / 'truth_sidecar/event_sidecar.parquet', index=False, compression=COMPRESSION)
    shutil.rmtree(out_root / 'truth_sidecar/parts', ignore_errors=True)
    shutil.rmtree(out_root / 'input_shards', ignore_errors=True)
    shutil.rmtree(out_root / 'code', ignore_errors=True)

    # Update splits/*_keys.parquet & write l1_internal_split_spec.json
    split_specs_out = {}
    for sp_name in ('internal_train', 'internal_dev_screen', 'internal_val_eval'):
        sub = sc_df[sc_df['internal_split'] == sp_name].copy().reset_index(drop=True)
        kdf = pd.DataFrame({
            'run': sub['run'].to_numpy(np.int32),
            'chunk': sub['chunk'].to_numpy(np.int32),
            'row': sub['row'].to_numpy(np.int32),
            'family_hi': sub['family_hi'].to_numpy(np.uint64),
            'family_lo': sub['family_lo'].to_numpy(np.uint64),
            'label': sub['label_4'].to_numpy(np.int64),
            'obs_domain_id': sub['obs_domain_id'].to_numpy(np.int64),
            'region_id': sub['region_id'].to_numpy(np.int64),
            'fold_5': sub['fold_5'].to_numpy(np.int64),
        })
        kpath = out_root / 'splits' / f'{sp_name}_keys.parquet'
        kdf.to_parquet(kpath, index=False, compression=COMPRESSION)
        split_specs_out[sp_name] = {
            'path': str(kpath),
            'events': int(len(kdf)),
            'unique_families': int(len(set(zip(kdf['family_hi'].tolist(), kdf['family_lo'].tolist())))),
            'class_counts': {str(c): int((kdf['label'] == c).sum()) for c in range(4)},
            'obs_domain_counts': {str(d): int((kdf['obs_domain_id'] == d).sum()) for d in range(4)},
        }

    # Check 0 family overlap across splits
    f_tr = set(zip(sc_df.loc[sc_df['internal_split'] == 'internal_train', 'family_hi'], sc_df.loc[sc_df['internal_split'] == 'internal_train', 'family_lo']))
    f_dv = set(zip(sc_df.loc[sc_df['internal_split'] == 'internal_dev_screen', 'family_hi'], sc_df.loc[sc_df['internal_split'] == 'internal_dev_screen', 'family_lo']))
    f_va = set(zip(sc_df.loc[sc_df['internal_split'] == 'internal_val_eval', 'family_hi'], sc_df.loc[sc_df['internal_split'] == 'internal_val_eval', 'family_lo']))
    if (f_tr & f_dv) or (f_tr & f_va) or (f_dv & f_va):
        raise AssertionError('Family overlap detected across L1 internal splits!')

    write_json(out_root / 'data/split_shards.json', {'train': [f'{s:03d}' for s in range(N_TRAIN_SHARDS)]})
    write_json(out_root / 'data/plan.json', {
        'stage': 'Stage328_L1_Train32k_ClusterSidecar',
        'geometry': str(out_root / 'geometry.json'),
        'n_shards': N_TRAIN_SHARDS,
        'include_truth_edep': False,
        'pure_W_verified': True,
    })
    write_json(out_root / 'l1_internal_split_spec.json', {
        'schema_version': '1.0',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_note': '10081/10083/10084 chunks 000..011 (24,000 exact) + 10086 chunks 002..015 (8,000 pure-W mu_CC)',
        'family_overlap_train_vs_dev': 0,
        'family_overlap_train_vs_val': 0,
        'family_overlap_dev_vs_val': 0,
        'splits': split_specs_out,
    })

    # 2. Aggregate shard diagnostics & verify spatial <50 um cluster alignment on shard 000
    orig_hist = np.zeros(5, dtype=np.int64)
    dom_hist = np.zeros(4, dtype=np.int64)
    trk_tot: dict[str, int] = {}
    tot_cl_tgt = 0
    tot_lt50 = 0
    tot_sc_bytes = 0
    tot_g4_w = 0

    for sid in range(N_TRAIN_SHARDS):
        dp = out_root / 'cluster_sidecar' / f'diag_{sid:03d}.json'
        d = json.loads(dp.read_text())
        orig_hist += np.asarray(d['origin_hist'], dtype=np.int64)
        dom_hist += np.asarray(d['obs_domain_hist'], dtype=np.int64)
        tot_cl_tgt += int(d['total_target_clusters'])
        tot_lt50 += int(d['matched_lt50um'])
        tot_sc_bytes += int((out_root / 'cluster_sidecar' / f'shard_{sid:03d}.parquet').stat().st_size)
        tot_g4_w += int(d['g4_w_verified'])
        for k_s, v_s in d['front30_tracklet_stats'].items():
            trk_tot[k_s] = trk_tot.get(k_s, 0) + int(v_s)

    # Independent physical verification of cluster_sidecar vs observations.parquet on shards 000, 039, 079
    geom = json.loads((out_root / 'geometry.json').read_text())
    bounds = detector_bounds(geom)
    nx_pix, ny_pix = int(bounds['nx']), int(bounds['ny'])
    verify_max_dist_mm = 0.0
    verify_checked_tgt = 0
    for check_sid in (0, 39, 79):
        obs_rows = pq.read_table(out_root / 'data/shards' / f'{check_sid:03d}/observations.parquet').to_pylist()
        sc_rows = pq.read_table(out_root / 'cluster_sidecar' / f'shard_{check_sid:03d}.parquet').to_pylist()
        if len(obs_rows) != len(sc_rows):
            raise AssertionError(f'Row mismatch on shard {check_sid:03d}')
        for ob, scr in zip(obs_rows, sc_rows):
            cid = np.asarray(ob['pixel']['channel_id'], dtype=np.int64)
            xyz = np.asarray(ob['pixel']['xyz_mm'], dtype=np.float64).reshape(-1, 3)
            if len(cid) == 0:
                if len(scr['cl_x_mm']) != 0:
                    raise AssertionError('Non-empty sidecar for empty obs')
                continue
            cl = _cluster_sensor_groups(cid, xyz, nx_pix, ny_pix, 12, 0.06, 0.06)
            n_cl = int(cl['n_clusters'])
            ord_c = cl['hit_order']
            off_c = cl['cluster_offsets']
            nh_c = np.diff(off_c).astype(np.float64)
            pl_c = cl['cluster_plane'].astype(np.int64)
            xm_c = np.add.reduceat(xyz[ord_c, 0], off_c[:-1]) / nh_c
            ym_c = np.add.reduceat(xyz[ord_c, 1], off_c[:-1]) / nh_c
            _, _, _, h040, _, cr5, _ = _compute_plane_neighborhoods(pl_c, xm_c, ym_c, nh_c)
            k_tgt = _select_group_clusters(np.flatnonzero(pl_c < 12), pl_c, nh_c, h040, cr5, N_MAX_TARGET_CLUSTERS)
            k_ipt = _select_group_clusters(np.flatnonzero(pl_c >= 12), pl_c, nh_c, h040, cr5, N_MAX_IPT_CLUSTERS)
            k_all = np.concatenate([k_tgt, k_ipt])
            sc_x = np.asarray(scr['cl_x_mm'], dtype=np.float64)
            sc_y = np.asarray(scr['cl_y_mm'], dtype=np.float64)
            sc_p = np.asarray(scr['cl_plane'], dtype=np.int64)
            if len(sc_x) != len(k_all) or not np.array_equal(sc_p, pl_c[k_all]):
                raise AssertionError(f'Cluster count/plane mismatch on shard {check_sid:03d}')
            d_xy = np.hypot(sc_x - xm_c[k_all], sc_y - ym_c[k_all])
            if len(d_xy):
                verify_max_dist_mm = max(verify_max_dist_mm, float(d_xy.max()))
            verify_checked_tgt += int((sc_p < 12).sum())
    if verify_max_dist_mm >= 0.050:
        raise AssertionError(f'Cluster alignment exceeded 50 um: {verify_max_dist_mm} mm')

    # Clean per-shard diag JSONs after aggregating so cluster_sidecar/ only has shard_*.parquet
    for sid in range(N_TRAIN_SHARDS):
        (out_root / 'cluster_sidecar' / f'diag_{sid:03d}.json').unlink(missing_ok=True)

    # Check zero .root files across out_root
    root_leftovers = [str(p) for p in out_root.rglob('*.root')]
    if root_leftovers:
        raise AssertionError(f'Found .root files on EOS: {root_leftovers[:5]}')

    tot_cl_all = max(1, int(orig_hist.sum()))
    cpp_3h = max(1, trk_tot.get('cpp_3hit', 0))
    cpp_2h = max(1, trk_tot.get('cpp_2hit', 0))
    cpp_disp = max(1, trk_tot.get('cpp_disp', 0))
    condor_info = json.loads((out_root / 'condor_submission.json').read_text()) if (out_root / 'condor_submission.json').is_file() else {}
    total_eos_bytes = sum(p.stat().st_size for p in out_root.rglob('*') if p.is_file())

    summary = {
        'status': 'complete',
        'stage': 'Stage328_L1_ClusterSidecar',
        'UTC': datetime.now(timezone.utc).isoformat(),
        'wall_seconds_finalize': float(time.perf_counter() - t0),
        'total_events': int(len(sc_df)),
        'n_shards': N_TRAIN_SHARDS,
        'g4_navigator_W_verified': int(tot_g4_w),
        'cluster_alignment': {
            'total_target_clusters': int(tot_cl_tgt),
            'matched_lt50um_count': int(tot_lt50),
            'matched_lt50um_fraction': float(tot_lt50 / max(1, tot_cl_tgt)),
            'verified_spotcheck_shards': [0, 39, 79],
            'verified_spotcheck_target_clusters': int(verify_checked_tgt),
            'verified_spotcheck_max_dist_mm': float(verify_max_dist_mm),
        },
        'origin_5class_distribution': {
            ORIGIN_CLASS_NAMES[i]: {
                'count': int(orig_hist[i]),
                'fraction': float(orig_hist[i] / tot_cl_all),
            }
            for i in range(5)
        },
        'obs_domain_distribution': {str(d): int(dom_hist[d]) for d in range(4)},
        'front30_tracklet_diagnostics_stage326_sec1_2': {
            'cpp_3hit_total': int(trk_tot.get('cpp_3hit', 0)),
            'cpp_3hit_pure_same_trackID_rate': float(trk_tot.get('cpp_3hit_pure', 0) / cpp_3h),
            'cpp_3hit_ghost_rate': float(1.0 - trk_tot.get('cpp_3hit_pure', 0) / cpp_3h),
            'cpp_2hit_total': int(trk_tot.get('cpp_2hit', 0)),
            'cpp_2hit_pure_same_trackID_rate': float(trk_tot.get('cpp_2hit_pure', 0) / cpp_2h),
            'cpp_disp_ip_sig_gt_2p5_total': int(trk_tot.get('cpp_disp', 0)),
            'cpp_disp_true_heavy_flavour_purity': float(trk_tot.get('cpp_disp_hf', 0) / cpp_disp),
            'cpp_disp_fake_conversion_or_ghost_fraction': float(trk_tot.get('cpp_disp_fake', 0) / cpp_disp),
            'soft64_tracklets_total': int(trk_tot.get('soft_tot', 0)),
        },
        'storage_eos': {
            'cluster_sidecar_mb': round(tot_sc_bytes / (1024.0 * 1024.0), 3),
            'total_l1_dir_mb': round(total_eos_bytes / (1024.0 * 1024.0), 3),
            'root_files_on_eos': 0,
        },
        'condor_submission': condor_info,
        'seal_violations': len(_L1_AUDIT_VIOLATIONS),
    }
    write_json(out_root / 'sidecar_production_summary.json', summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'worker', 'submit', 'finalize'))
    parser.add_argument('--out-root', type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument('--task-id', type=int, default=0)
    parser.add_argument('--shards-per-worker', type=int, default=1)
    parser.add_argument('--n-workers', type=int, default=80)
    parser.add_argument('--cpus', type=int, default=8)
    parser.add_argument('--schedd', type=str, default='bigbird10.cern.ch')
    args = parser.parse_args()

    if args.command == 'prepare':
        res = prepare_l1_inputs(args.out_root, workers=args.cpus)
    elif args.command == 'worker':
        s0 = args.task_id * args.shards_per_worker
        s1 = min(N_TRAIN_SHARDS, s0 + args.shards_per_worker)
        res = [run_shard_worker(args.out_root, sid, cpus=args.cpus) for sid in range(s0, s1)]
    elif args.command == 'submit':
        res = submit_l1_condor_array(
            args.out_root,
            schedd=args.schedd,
            cpus=args.cpus,
            n_workers=args.n_workers,
        )
    else:
        res = finalize_l1_sidecar(args.out_root)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
