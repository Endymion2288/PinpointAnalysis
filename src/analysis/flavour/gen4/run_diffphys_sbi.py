"""Stage-I & Stage-II Runner for Pillar IV (`DiffPhys-SBI` + `Ref-BDT` + `POT-GPD` + `NSBI-PL`) on `10085-10088`.

Contract v2.0 Compliance:
  - 100% isolated on official MC `10085-10088` (`stage329_gen4_mc8588/cache/`); ZERO access to
    legacy `10081-10084` / `v6_clean_v3` / `l1_cluster_sidecar` / `l2_ssl_300k` or sealed chunks
    `r = chunk_id % 20 in {16, 17, 18, 19}` (`Cal-A / Cut-B / Confirm`).
  - Stage-I physically blocks `Tune_Screen` (`r=12`) and `Tune6k / Stress400 / Diag_OOD` (`r=13`).
  - Extracts 36D inverse-physics sufficient statistics (with Mode-B single-pixel `edep` & `<2.5 PE`
    halo de-saturation) and runs Rossi-Greisen FISTA segment purity audit against `cl_origin_label`.
  - Runs controlled `Ref-BDT` (historical binary-pixel + `>=2.5 PE` + 3-station baseline) on the
    exact same `Train32k -> Tune_Screen -> Tune6k` split for apples-to-apples paired bootstrap.
  - Evaluates `DiffPhys-SBI` (3 seeds `{17, 42, 2026}`) on `Tune6k`, `tau_e/mu/1h/3h @1% FPR`,
    `L_tau < 30 mm`, `Diag_OOD` (`InDet_NC_Charm`, `NonW_Al_Scint`, `HighE_gt1p5TeV`), `Stress400`,
    `POT-GPD`, and `NSBI-PL` Asimov & Toy-MC closure (`f_tau=0.3%`, `MW2t`, `delta_sys`).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from datetime import datetime, timezone
import gc
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import re
import shlex
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
import torch

from analysis.flavour.gen4.diffphys_sbi import (
    DIFFPHYS_STAT_DIM,
    DiffPhysSBIModel,
    RossiGreisenFISTAShowerStripper,
    diffphys_sbi_loss,
    extract_diffphys_from_gen4_item,
    strip_shower_and_score_segments,
)
from analysis.flavour.gen4.geometry_contract import load_geometry_contract
from analysis.flavour.gen4.nsbi_discovery import (
    compare_pot_gpd_vs_empirical_split_half,
    evaluate_nsbi_vs_single_bin_closure,
    run_nsbi_toy_mc_closure,
)
from analysis.flavour.inventory import write_json
from analysis.flavour.nextgen.protocols import check_branch_gradient_health
from analysis.flavour.nextgen.run_dual_regime_modalnet import (
    STRESS_SPECS,
    _ba4_and_nll,
    _collaboration_section12_slice,
    _fit_scalar_temperature_on_screen,
    _paired_family_bootstrap_full,
    _set_ieee_fp32,
    _split_half_tail_eval,
    _weighted_ba4_pct,
    _within_slice_tail_eval,
)
from analysis.flavour.nextgen.seal_guard import (
    SealAuditGuard,
    SealViolationError,
    assert_unsealed_path,
    assert_unsealed_split,
    verify_inference_truth_invariance,
)
from analysis.flavour.v2.partitions import FAMILY
from analysis.flavour.v2.submit import choose_schedd
from analysis.flavour.v3.common.domains import OBS_DOMAIN_NAMES
from analysis.flavour.v3.common.submit import _package_source


NEXTGEN_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research')
CACHE_DIR = NEXTGEN_ROOT / 'stage329_gen4_mc8588/cache'
VAL_META_DIR = NEXTGEN_ROOT / 'stage329_gen4_mc8588/validation_meta'
EXP_ROOT = NEXTGEN_ROOT / 'stage329_gen4_mc8588/exp_diffphys_sbi'
DIFFPHYS_CACHE_DIR = EXP_ROOT / '_diffphys_cache'
STAGE1_DIR = EXP_ROOT / 'stage1'
STAGE2_DIR = EXP_ROOT / 'stage2'
CONDOR_DIR = Path.home() / 'condor/Pinpoint/stage329_diffphys'
GEOM_MC8588_JSON = CACHE_DIR / 'geometry_mc8588.json'

STAGE2_SEEDS: tuple[int, ...] = (17, 42, 2026)

# Runtime seal guard blocking legacy 10081-10084 caches and MC 10085-10088 sealed chunks (r in 16..19)
_LEGACY_OR_SEALED_RE = re.compile(
    r'v6_clean_v3|l1_cluster_sidecar|l2_ssl_300k|1008[1-4][_/\\-]|cal_a|cut_b|confirm',
    re.IGNORECASE,
)
_ROOT_CHUNK_RE = re.compile(r'1008[5-8]_(\d+)\.root$', re.IGNORECASE)
_STAGE1_TUNE_RE = re.compile(r'tune8k|tune_screen|tune6k|ood_stress|stress400|diag_ood', re.IGNORECASE)


def install_mc8588_seal_hook(*, stage: int) -> None:
    """Install runtime sys.addaudithook enforcing zero legacy access and zero MC8588 sealed-chunk access."""

    def _hook(event: str, args: tuple[Any, ...]) -> None:
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
        if _LEGACY_OR_SEALED_RE.search(norm):
            raise SealViolationError(f'Forbidden legacy/sealed path under Contract v2.0: {norm}')
        m_rc = _ROOT_CHUNK_RE.search(norm)
        if m_rc is not None:
            cid = int(m_rc.group(1))
            r = cid % 20
            if r >= 16:
                raise SealViolationError(f'Sealed MC8588 chunk r={r} (chunk {cid}) is strictly forbidden: {norm}')
            if stage == 1 and r in (12, 13):
                raise SealViolationError(f'Stage-I forbids accessing Tune chunk r={r} (chunk {cid}): {norm}')
        if stage == 1 and _STAGE1_TUNE_RE.search(norm):
            raise SealViolationError(f'Stage-I forbids accessing Tune/OOD cache: {norm}')

    sys.addaudithook(_hook)


def _resolve_geom_path() -> Path:
    if GEOM_MC8588_JSON.is_file():
        return assert_unsealed_path(GEOM_MC8588_JSON)
    # Fallback search for exported geometry json inside CACHE_DIR
    cands = sorted(CACHE_DIR.glob('*geometry*.json'))
    if cands:
        return assert_unsealed_path(cands[0])
    raise FileNotFoundError(f'Missing geometry_mc8588.json in {CACHE_DIR}; wait for [foundation-ready]')


def _load_pkl_obj(path: Path) -> Any:
    with assert_unsealed_path(path).open('rb') as fh:
        return pickle.load(fh)


def _extract_worker(task: tuple[list[dict[str, Any]], str]) -> list[dict[str, Any]]:
    items_slice, geom_path_str = task
    geom_c = load_geometry_contract(geom_path_str)
    strip_op = RossiGreisenFISTAShowerStripper(geom_c.raw_geometry, pixel_x0=getattr(geom_c, 'pixel_X0', None))
    out = []
    for it in items_slice:
        dp = extract_diffphys_from_gen4_item(it, geom_c, stripper=strip_op)
        # Preserve historical-parity BDT features for Ref-BDT (binary pixels, >=2.5 PE, 3-station M1)
        dp['ref_bdt_feat'] = _extract_historical_bdt_features(it)
        for k in (
            'run', 'chunk', 'evt_id', 'row', 'family_hi', 'family_lo', 'internal_split',
            'label', 'label_7', 'charm_flag', 'charm_valid', 'vertex_xyz_mm', 'vertex_valid',
            'sample_weight', 'true_tau_submode', 'tau_decay_L3d_mm', 'tgt_Z',
            'true_nu_energy_gev', 'energy_nu_GeV',
        ):
            if k in it:
                dp[k] = it[k]
        out.append(dp)
    return out


def _extract_historical_bdt_features(it: Mapping[str, Any]) -> np.ndarray:
    """Extract historical-parity 24D hand-crafted features (no Mode-B edep, no <2.5 PE, no M2)."""
    c_obs = np.asarray(it.get('c_obs', np.zeros(16)), dtype=np.float32).reshape(-1)
    c_pad = np.zeros(16, dtype=np.float32)
    c_pad[:min(16, len(c_obs))] = c_obs[:16]

    pl = np.asarray(it.get('pix_plane', ()), dtype=np.int64)
    n_cl_tgt = float(np.sum(pl < 12))
    n_cl_front = float(np.sum(pl < 6))
    n_cl_ipt = float(np.sum(pl >= 12))

    # Scintillator >= 2.5 PE summary only
    if 'scint_grid' in it:
        sg = np.asarray(it['scint_grid'], dtype=np.float32)
        pe = sg[:, :, :, 2]
        pe_thr = np.where(pe >= 2.5, pe, 0.0)
        tot_pe = float(np.sum(pe_thr))
        n_bars_thr = float(np.sum(pe >= 2.5))
        early_pe = float(np.sum(pe_thr[:12]))
    elif 'bar_feat_10' in it and 'bar_station' in it:
        bf = np.asarray(it['bar_feat_10'], dtype=np.float32)
        bs = np.asarray(it['bar_station'], dtype=np.int64)
        if len(bf) > 0:
            pe = bf[:, 2]
            thr_m = pe >= 2.5
            tot_pe = float(np.sum(pe[thr_m]))
            n_bars_thr = float(np.sum(thr_m))
            early_pe = float(np.sum(pe[thr_m & (bs < 12)]))
        else:
            tot_pe, n_bars_thr, early_pe = 0.0, 0.0, 0.0
    else:
        tot_pe, n_bars_thr, early_pe = 0.0, 0.0, 0.0

    # Only 3-station M1 spectrometer track presence & p
    m1_p, m1_cnt = 0.0, 0.0
    if 'muon_kalman_tracks' in it and 'muon_kalman_mask' in it:
        mk = np.asarray(it['muon_kalman_tracks'], dtype=np.float32)
        mm = np.asarray(it['muon_kalman_mask'], dtype=bool)
        if mm.any():
            m1_m = mm & (mk[:, 6] > 0.5)
            m1_cnt = float(m1_m.sum())
            if m1_m.any():
                m1_p = float(mk[m1_m][0, 2])

    extra = np.array([
        math.log1p(n_cl_tgt),
        math.log1p(n_cl_front),
        math.log1p(n_cl_ipt),
        math.log1p(tot_pe),
        math.log1p(n_bars_thr),
        early_pe / max(tot_pe, 1.0),
        m1_cnt,
        math.log1p(m1_p),
    ], dtype=np.float32)
    return np.concatenate([c_pad, extra], axis=0)


def _parallel_extract_dp(items: list[dict[str, Any]], geom_path: Path, n_workers: int = 8) -> list[dict[str, Any]]:
    if not items:
        return []
    chunk_sz = max(64, math.ceil(len(items) / max(1, n_workers * 2)))
    tasks = [(items[i : i + chunk_sz], str(geom_path)) for i in range(0, len(items), chunk_sz)]
    if n_workers <= 1 or len(tasks) <= 1:
        return [x for t in tasks for x in _extract_worker(t)]
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context('spawn')) as pool:
        parts = list(pool.map(_extract_worker, tasks))
    return [x for p in parts for x in p]


def _pack_dp_arrays(dp_list: list[dict[str, Any]], *, include_truth: bool) -> dict[str, Any]:
    n = len(dp_list)
    z = np.stack([d['diffphys_z'] for d in dp_list], axis=0).astype(np.float32)
    zs = np.stack([d['diffphys_z_sigma'] for d in dp_list], axis=0).astype(np.float32)
    bdt_x = np.stack([d['ref_bdt_feat'] for d in dp_list], axis=0).astype(np.float32)
    dom = np.array([int(d['obs_domain_id']) for d in dp_list], dtype=np.int64)
    ev = np.array([float(d['e_vis_gev']) for d in dp_list], dtype=np.float32)
    pv = np.stack([d['pv_xyz_mm'] for d in dp_list], axis=0).astype(np.float32)
    pvs = np.stack([d['pv_sigma_mm'] for d in dp_list], axis=0).astype(np.float32)

    obs_keys = list(dp_list[0]['obs_summary'].keys())
    obs_summary = {k: np.array([d['obs_summary'][k] for d in dp_list]) for k in obs_keys}
    pack: dict[str, Any] = {
        'diffphys_z': z,
        'diffphys_z_sigma': zs,
        'ref_bdt_feat': bdt_x,
        'obs_domain_id': dom,
        'e_vis_gev': ev,
        'pv_xyz_mm': pv,
        'pv_sigma_mm': pvs,
        'obs_summary': obs_summary,
    }
    for k in ('run', 'chunk', 'evt_id', 'row', 'family_hi', 'family_lo', 'internal_split'):
        if k in dp_list[0]:
            pack[k] = np.array([d[k] for d in dp_list])

    if include_truth:
        for k, dt in (
            ('label', np.int64),
            ('label_7', np.int64),
            ('charm_flag', np.int64),
            ('charm_valid', np.int64),
            ('vertex_valid', np.int64),
            ('sample_weight', np.float32),
            ('true_tau_submode', np.int64),
            ('tau_decay_L3d_mm', np.float32),
            ('tgt_Z', np.int64),
            ('true_nu_energy_gev', np.float32),
            ('energy_nu_GeV', np.float32),
        ):
            if k in dp_list[0]:
                pack[k] = np.array([d[k] for d in dp_list], dtype=dt)
        if 'vertex_xyz_mm' in dp_list[0]:
            pack['vertex_xyz_mm'] = np.stack([d['vertex_xyz_mm'] for d in dp_list], axis=0).astype(np.float32)
        if 'sample_weight' not in pack:
            pack['sample_weight'] = np.ones(n, dtype=np.float32)
    return pack


def _load_cache_parts(pattern: str) -> list[dict[str, Any]]:
    files = sorted(CACHE_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f'No cache files matching {pattern!r} in {CACHE_DIR}; wait for [foundation-ready]')
    items: list[dict[str, Any]] = []
    for p_f in files:
        obj = _load_pkl_obj(p_f)
        if isinstance(obj, list):
            items.extend(obj)
        elif isinstance(obj, dict) and 'items' in obj:
            items.extend(obj['items'])
        else:
            raise TypeError(f'Unsupported cache payload type in {p_f}: {type(obj)!r}')
    return items


def _extract_cache_parts_streaming(
    pattern: str,
    geom_path: Path,
    n_workers: int,
    *,
    include_truth: bool,
) -> dict[str, Any]:
    files = sorted(CACHE_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f'No cache files matching {pattern!r} in {CACHE_DIR}; wait for [foundation-ready]')
    dp_all: list[dict[str, Any]] = []
    for p_f in files:
        obj = _load_pkl_obj(p_f)
        if isinstance(obj, list):
            part_items = obj
        elif isinstance(obj, dict) and 'items' in obj:
            part_items = obj['items']
        else:
            raise TypeError(f'Unsupported cache payload type in {p_f}: {type(obj)!r}')
        dp_all.extend(_parallel_extract_dp(part_items, geom_path, n_workers=n_workers))
        del obj, part_items
        gc.collect()
    return _pack_dp_arrays(dp_all, include_truth=include_truth)


def run_preextract_diffphys(stage: int = 1, workers: int = 8, include_stress_diag: bool = True) -> dict[str, Any]:
    """Pre-extract 36D DiffPhys-SBI & Ref-BDT caches from MC 10085-10088 foundation cache."""
    assert_unsealed_split('train', stage=stage)
    if stage >= 2:
        for sp in ('validation', 'tune_screen', 'tune6k', 'ood'):
            assert_unsealed_split(sp, stage=2)

    DIFFPHYS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    geom_path = _resolve_geom_path()
    t0 = time.perf_counter()
    report: dict[str, Any] = {'stage': stage, 'geom': str(geom_path)}

    tr_out = DIFFPHYS_CACHE_DIR / 'train32k_diffphys.pkl'
    if not tr_out.is_file():
        pack_tr = _extract_cache_parts_streaming('train32k_part_*.pkl', geom_path, workers, include_truth=True)
        tmp = tr_out.with_suffix('.pkl.tmp')
        with tmp.open('wb') as fh:
            pickle.dump(pack_tr, fh, protocol=5)
        tmp.replace(tr_out)
        report['train32k_events'] = int(len(pack_tr['diffphys_z']))
        del pack_tr
        gc.collect()
    else:
        report['train32k_cached'] = True

    if stage >= 2:
        t8_out = DIFFPHYS_CACHE_DIR / 'tune8k_diffphys.pkl'
        if not t8_out.is_file():
            pack_t8 = _extract_cache_parts_streaming('tune8k_part_*.pkl', geom_path, workers, include_truth=False)
            tmp = t8_out.with_suffix('.pkl.tmp')
            with tmp.open('wb') as fh:
                pickle.dump(pack_t8, fh, protocol=5)
            tmp.replace(t8_out)
            report['tune8k_events'] = int(len(pack_t8['diffphys_z']))
            del pack_t8
            gc.collect()
        else:
            report['tune8k_cached'] = True

        os_out = DIFFPHYS_CACHE_DIR / 'ood_stress_diffphys.pkl'
        if include_stress_diag and not os_out.is_file():
            sd_f = CACHE_DIR / 'stress_diag_pack.pkl'
            dm_f = VAL_META_DIR / 'diag_truth_meta.pkl'
            os_f = CACHE_DIR / 'ood_stress_pack.pkl'
            ood_dp: dict[str, tuple[dict[str, Any], np.ndarray]] = {}
            stress_dp: dict[str, dict[str, Any]] = {}
            if sd_f.is_file():
                sd_pack = _load_pkl_obj(sd_f)
                dm_pack = _load_pkl_obj(dm_f) if dm_f.is_file() else {}
                diag_meta = dm_pack.get('diag_meta', {})
                for ov, items_o in sd_pack.get('diag_packs', {}).items():
                    if not items_o:
                        continue
                    dp_o = _parallel_extract_dp(items_o, geom_path, n_workers=workers)
                    if ov in diag_meta and 'label' in diag_meta[ov].columns:
                        y_o = diag_meta[ov]['label'].to_numpy(dtype=np.int64)
                    else:
                        y_o = np.full(len(items_o), 3 if 'NC' in ov else -1, dtype=np.int64)
                    ood_dp[ov] = (_pack_dp_arrays(dp_o, include_truth=False), y_o)
                for sn, items_s in sd_pack.get('stress_items', {}).items():
                    dp_s = _parallel_extract_dp(items_s, geom_path, n_workers=workers)
                    stress_dp[sn] = _pack_dp_arrays(dp_s, include_truth=False)
                y_stress = dm_pack.get('stress_y')
                prov_stress = None
            elif os_f.is_file():
                os_pack = _load_pkl_obj(os_f)
                ood_src = os_pack.get('ood_pre', os_pack.get('diag_ood', {}))
                for ov, (items_o, y_o) in ood_src.items():
                    dp_o = _parallel_extract_dp(items_o, geom_path, n_workers=workers)
                    ood_dp[ov] = (_pack_dp_arrays(dp_o, include_truth=False), np.asarray(y_o, dtype=np.int64))
                for sn, items_s in os_pack['stress_items'].items():
                    dp_s = _parallel_extract_dp(items_s, geom_path, n_workers=workers)
                    stress_dp[sn] = _pack_dp_arrays(dp_s, include_truth=False)
                y_stress = os_pack.get('y_stress', os_pack.get('y_080'))
                prov_stress = os_pack.get('prov_stress', os_pack.get('prov_080'))
            else:
                raise FileNotFoundError(f'Missing {sd_f} and {os_f}')
            pack_os = {
                'ood_dp': ood_dp,
                'stress_dp': stress_dp,
                'y_stress': y_stress,
                'prov_stress': prov_stress,
            }
            tmp = os_out.with_suffix('.pkl.tmp')
            with tmp.open('wb') as fh:
                pickle.dump(pack_os, fh, protocol=5)
            tmp.replace(os_out)
            report['ood_stress_extracted'] = True
        else:
            report['ood_stress_cached'] = True

    report['wall_seconds'] = float(time.perf_counter() - t0)
    return report


def run_fista_purity_audit(max_events: int = 2000) -> dict[str, Any]:
    """Quantify candidate 3-hit segment truth purity & tau cluster recall using native MC8588 sidecars."""
    assert_unsealed_split('train', stage=1)
    geom_path = _resolve_geom_path()
    geom_c = load_geometry_contract(geom_path)
    strip_op = RossiGreisenFISTAShowerStripper(geom_c.raw_geometry, pixel_x0=getattr(geom_c, 'pixel_X0', None))

    p_files = sorted(CACHE_DIR.glob('train32k_part_*.pkl'))
    if not p_files:
        raise FileNotFoundError(f'Missing train32k_part_*.pkl in {CACHE_DIR}; wait for [foundation-ready]')
    per_part = max(1, max_events // len(p_files))

    n_raw_seg, n_raw_pure = 0, 0
    n_str_seg, n_str_pure = 0, 0
    tau_rec_sum, n_tau_cl_ev, n_audited = 0.0, 0, 0
    tau_mip_rec_sum, n_tau_mip_ev = 0.0, 0
    em_strip_sum, n_em_ev = 0.0, 0
    by_run: dict[str, dict[str, float | int]] = {}

    for p_f in p_files:
        part_items = _load_pkl_obj(p_f)[:per_part]
        for it in part_items:
            n_audited += 1
            orig = it.get('cl_origin_label')
            tid = it.get('cl_dominant_track_id')
            xyz = np.asarray(it['pix_xyz_mm'], dtype=np.float64)
            pl = np.asarray(it['pix_plane'], dtype=np.int64)
            if orig is None or tid is None or len(xyz) < 3:
                continue
            feat = np.asarray(it.get('pix_feat', np.empty((0, 36))), dtype=np.float64)
            nh = np.expm1(np.clip(feat[:, 3] * 6.0, 0.0, 6.0)) if (len(feat) == len(xyz) and feat.shape[1] >= 4) else None
            bip = np.clip(feat[:, 24], 0.0, 1.5) if (len(feat) == len(xyz) and feat.shape[1] >= 25) else None
            if bip is not None and nh is not None:
                bip = np.where(nh >= 3.5, bip, 0.0)
            if len(feat) == len(xyz) and feat.shape[1] >= 32 and float(it.get('has_edep', 0.0)) > 0.5:
                b_2mip = np.clip(feat[:, 31], 0.0, 1.0)
                bip = b_2mip if bip is None else np.maximum(bip, b_2mip)
            ed_mip = np.expm1(np.clip(feat[:, 30], 0.0, 6.0)) if (len(feat) == len(xyz) and feat.shape[1] >= 31 and float(it.get('has_edep', 0.0)) > 0.5) else None
            r = strip_shower_and_score_segments(
                xyz, pl, strip_op, bipole_scores=bip, cluster_n_hits=nh, cluster_edep_mip=ed_mip,
                w_mip_thr=0.50, cl_origin_label=orig, cl_dominant_track_id=tid,
            )
            rp_cnt = int(round(r['raw_purity'] * r['n_raw_segments']))
            sp_cnt = int(round(r['stripped_purity'] * r['n_stripped_segments']))
            n_raw_seg += r['n_raw_segments']
            n_raw_pure += rp_cnt
            n_str_seg += r['n_stripped_segments']
            n_str_pure += sp_cnt
            rk = str(it.get('run', 'unk'))
            ent = by_run.setdefault(rk, {'raw_seg': 0, 'raw_pure': 0, 'str_seg': 0, 'str_pure': 0, 'n_ev': 0})
            ent['raw_seg'] += r['n_raw_segments']
            ent['raw_pure'] += rp_cnt
            ent['str_seg'] += r['n_stripped_segments']
            ent['str_pure'] += sp_cnt
            ent['n_ev'] += 1
            orig_arr = np.asarray(orig, dtype=np.int64)
            if np.any(orig_arr == 3):
                em_strip_sum += r.get('em_stripped_frac', 0.0)
                n_em_ev += 1
            if np.any(orig_arr == 1):
                tau_rec_sum += r['tau_cluster_recall']
                n_tau_cl_ev += 1
                if int(it.get('true_tau_submode', -1)) in (1, 2, 3):
                    tau_mip_rec_sum += r['tau_cluster_recall']
                    n_tau_mip_ev += 1
        del part_items
        gc.collect()

    raw_pur = float(n_raw_pure / max(n_raw_seg, 1))
    str_pur = float(n_str_pure / max(n_str_seg, 1))
    tau_rec = float(tau_rec_sum / max(n_tau_cl_ev, 1))
    tau_mip_rec = float(tau_mip_rec_sum / max(n_tau_mip_ev, 1))
    em_str = float(em_strip_sum / max(n_em_ev, 1))
    res = {
        'n_events_audited': int(n_audited),
        'n_tau_events_with_clusters': int(n_tau_cl_ev),
        'n_tau_mip_events_with_clusters': int(n_tau_mip_ev),
        'raw_3hit_segments': int(n_raw_seg),
        'raw_segment_truth_purity_pct': raw_pur * 100.0,
        'stripped_3hit_segments': int(n_str_seg),
        'stripped_segment_truth_purity_pct': str_pur * 100.0,
        'purity_improvement_factor': float(str_pur / max(raw_pur, 1e-6)),
        'tau_daughter_cluster_recall_pct': tau_rec * 100.0,
        'tau_non_e_mip_recall_pct': tau_mip_rec * 100.0,
        'em_cluster_stripped_pct': em_str * 100.0,
        'by_run': {
            rk: {
                'n_events': int(v['n_ev']),
                'raw_purity_pct': 100.0 * float(v['raw_pure'] / max(v['raw_seg'], 1)),
                'stripped_purity_pct': 100.0 * float(v['str_pure'] / max(v['str_seg'], 1)),
            }
            for rk, v in sorted(by_run.items())
        },
    }
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(EXP_ROOT / 'fista_purity_audit.json', res)
    return res


def _build_tensor_batch(pack: dict[str, Any], idx: np.ndarray, device: torch.device, *, include_truth: bool) -> dict[str, torch.Tensor]:
    b: dict[str, torch.Tensor] = {
        'diffphys_z': torch.from_numpy(pack['diffphys_z'][idx]).to(device),
        'diffphys_z_sigma': torch.from_numpy(pack['diffphys_z_sigma'][idx]).to(device),
        'obs_domain_id': torch.from_numpy(pack['obs_domain_id'][idx]).to(device),
        'e_vis_gev': torch.from_numpy(pack['e_vis_gev'][idx]).to(device),
        'pv_xyz_mm': torch.from_numpy(pack['pv_xyz_mm'][idx]).to(device),
    }
    if include_truth:
        for k in ('label', 'label_7', 'charm_flag', 'charm_valid', 'vertex_xyz_mm', 'vertex_valid'):
            if k in pack:
                b[k] = torch.from_numpy(pack[k][idx]).to(device)
    return b


def run_health_and_small_sample_overfit(seed: int = 42) -> dict[str, Any]:
    """Run Gate-A branch gradient health, Gate-A truth invariance (atol=0.0), and N=64 overfit check."""
    _set_ieee_fp32(True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    tr_f = DIFFPHYS_CACHE_DIR / 'train32k_diffphys.pkl'
    if tr_f.is_file():
        pack = _load_pkl_obj(tr_f)
        idx64 = np.concatenate([np.flatnonzero(pack['label'] == c)[:16] for c in range(4)])
        b64 = _build_tensor_batch(pack, idx64, device, include_truth=True)
    else:
        n = 64
        y4 = np.tile(np.arange(4, dtype=np.int64), n // 4)
        z = rng.normal(0.0, 0.7, size=(n, DIFFPHYS_STAT_DIM)).astype(np.float32)
        for c in range(4):
            z[y4 == c, c * 4 : c * 4 + 4] += 1.8
        b64 = {
            'diffphys_z': torch.from_numpy(z).to(device),
            'diffphys_z_sigma': torch.full((n, DIFFPHYS_STAT_DIM), 0.12, dtype=torch.float32, device=device),
            'obs_domain_id': torch.ones(n, dtype=torch.long, device=device),
            'e_vis_gev': torch.full((n,), 300.0, dtype=torch.float32, device=device),
            'pv_xyz_mm': torch.zeros((n, 3), dtype=torch.float32, device=device),
            'label': torch.from_numpy(y4).to(device),
            'label_7': torch.from_numpy(np.where(y4 == 0, 0, np.where(y4 == 1, 1, np.where(y4 == 2, 5, 6)))).to(device),
            'charm_flag': torch.zeros(n, dtype=torch.long, device=device),
            'charm_valid': torch.ones(n, dtype=torch.long, device=device),
            'vertex_xyz_mm': torch.zeros((n, 3), dtype=torch.float32, device=device),
            'vertex_valid': torch.ones(n, dtype=torch.long, device=device),
        }

    model = DiffPhysSBIModel().to(device)
    model.set_normalization(b64['diffphys_z'].mean(dim=0), b64['diffphys_z'].std(dim=0).clamp_min(0.1))
    n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    inv_res = verify_inference_truth_invariance(model, b64, atol=0.0)
    model.train()
    loss0, _ = diffphys_sbi_loss(model, b64)
    grad_res = check_branch_gradient_health(model, loss0, min_grad_l2=1e-6)
    if not grad_res['passed']:
        raise RuntimeError(f'Gate-A gradient check failed: {grad_res}')

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    losses = []
    for _ in range(150):
        opt.zero_grad(set_to_none=True)
        loss, _ = diffphys_sbi_loss(model, b64)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step()
        losses.append(float(loss.detach().item()))

    model.eval()
    with torch.no_grad():
        out = model.forward_NextGen(b64)
        acc = float((out.probs4.argmax(dim=-1) == b64['label']).float().mean().item())
        pred_std = float(out.logits7.std(dim=0).mean().item())

    report = {
        'n_params': n_params,
        'in_param_budget_85k_150k': bool(85000 <= n_params <= 150000),
        'truth_invariance_atol0': inv_res,
        'branch_grad_health': grad_res,
        'overfit_n64_initial_loss': losses[0],
        'overfit_n64_final_loss': losses[-1],
        'overfit_n64_acc': acc,
        'overfit_n64_pred_std': pred_std,
        'passed': bool(inv_res['passed'] and grad_res['passed'] and losses[-1] < losses[0] - 0.5 and acc >= 0.90),
    }
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(EXP_ROOT / 'health_check.json', report)
    return report


def _train_diffphys_single(
    tr_pack: dict[str, Any],
    tr_idx: np.ndarray,
    val_pack: dict[str, Any],
    val_idx: np.ndarray,
    *,
    seed: int,
    epochs: int = 24,
    batch_size: int = 512,
    lr: float = 3.0e-3,
    device: torch.device,
    tag: str = 's1',
) -> tuple[DiffPhysSBIModel, list[dict[str, float]]]:
    _set_ieee_fp32(True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = DiffPhysSBIModel().to(device)
    z_tr = tr_pack['diffphys_z'][tr_idx]
    z_mean = torch.from_numpy(z_tr.mean(axis=0)).float()
    z_std = torch.from_numpy(z_tr.std(axis=0)).float().clamp_min(1e-2)
    model.set_normalization(z_mean, z_std)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2.0e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=5.0e-5)

    best_nll = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, float]] = []
    val_y = val_pack['label'][val_idx]

    for ep in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(tr_idx)
        ep_loss, n_steps = 0.0, 0
        for st in range(0, len(perm), batch_size):
            sub = perm[st : st + batch_size]
            batch = _build_tensor_batch(tr_pack, sub, device, include_truth=True)
            opt.zero_grad(set_to_none=True)
            loss, _ = diffphys_sbi_loss(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            ep_loss += float(loss.detach().item())
            n_steps += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            v_probs_l = []
            for st in range(0, len(val_idx), 512):
                vb = _build_tensor_batch(val_pack, val_idx[st : st + 512], device, include_truth=False)
                v_probs_l.append(model.forward_NextGen(vb).probs4.cpu().numpy())
            v_probs = np.concatenate(v_probs_l, axis=0)
            vm = _ba4_and_nll(val_y, v_probs)

        history.append({
            'epoch': ep,
            'train_loss': ep_loss / max(n_steps, 1),
            'screen_BA4_pct': vm['BA4_pct'],
            'screen_NLL': vm['NLL'],
        })
        if ep == 1 or ep % 4 == 0 or ep == epochs:
            print(
                f"[{tag} seed={seed}] ep={ep:02d}/{epochs} loss={history[-1]['train_loss']:.4f} "
                f"screen_BA4={vm['BA4_pct']:.2f}% screen_NLL={vm['NLL']:.4f}",
                flush=True,
            )
        if vm['NLL'] < best_nll:
            best_nll = vm['NLL']
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    return model, history


@torch.no_grad()
def _predict_diffphys_pack(
    model: DiffPhysSBIModel,
    pack: dict[str, Any],
    device: torch.device,
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    model.eval()
    n = len(pack['diffphys_z'])
    p4_l, llr_l, ch_l, vp_l, sp_l = [], [], [], [], []
    for st in range(0, n, batch_size):
        idx = np.arange(st, min(n, st + batch_size), dtype=np.int64)
        b = _build_tensor_batch(pack, idx, device, include_truth=False)
        out = model.forward_NextGen(b)
        p4_l.append(out.probs4.float().cpu().numpy())
        llr_l.append(out.aux['tau_llr'].float().cpu().numpy())
        ch_l.append(torch.sigmoid(out.charm_logit.float()).cpu().numpy())
        vp_l.append(out.v_pv_mm.float().cpu().numpy())
        sp_l.append(out.scale_pv_mm.float().cpu().numpy())
    return {
        'probs4': np.concatenate(p4_l, axis=0).astype(np.float32),
        'tau_llr': np.concatenate(llr_l, axis=0).astype(np.float32),
        'charm_prob': np.concatenate(ch_l, axis=0).astype(np.float32),
        'v_pv_mm': np.concatenate(vp_l, axis=0).astype(np.float32),
        'scale_pv_mm': np.concatenate(sp_l, axis=0).astype(np.float32),
    }


def _build_nsbi_obs_dict(pack: dict[str, Any], pred: dict[str, Any]) -> dict[str, np.ndarray]:
    """Assemble [gen4-contract] Section 5.2 pure-observable NSBI routing dictionary."""
    os_raw = pack['obs_summary']
    n = len(pred['tau_llr'])
    z = pack['diffphys_z']
    return {
        'obs_domain_id': np.asarray(os_raw['obs_domain_id'], dtype=np.int64),
        'tau_llr': np.asarray(pred['tau_llr'], dtype=np.float32),
        'charm_prob': np.asarray(pred['charm_prob'], dtype=np.float32),
        'n_sv_prongs': np.asarray(os_raw['n_sv_prongs'], dtype=np.int64),
        'sl_xy': np.asarray(os_raw['sv_lxy_sig'], dtype=np.float32),
        'm_sv_corr_gev': np.asarray(os_raw['sv_mass_gev'], dtype=np.float32),
        'n_mu_veto': np.asarray(os_raw['n_pen_muons'], dtype=np.int64),
        'q_mu1': np.sign(z[:, 27]).astype(np.float32),
        'q_mu2': np.zeros(n, dtype=np.float32),
        'lambda_kink': np.asarray(os_raw['kink_llr'], dtype=np.float32),
        'dphi_mu_ht': np.asarray(os_raw['dphi_mu_had'], dtype=np.float32),
        'ip_sig_max': np.asarray(os_raw['ip_sig_max'], dtype=np.float32),
        'm_vis_rho_gev': np.asarray(os_raw['rho_vis_mass_gev'], dtype=np.float32),
        'e_vis_hat_gev': np.asarray(os_raw['e_vis_gev'], dtype=np.float32),
        'pt_miss_hat_gev': np.asarray(os_raw['pt_miss_gev'], dtype=np.float32),
        'm_gg_pi0_gev': np.where(np.asarray(os_raw['pi0_em_frac'], dtype=np.float32) > 0.2, 0.135, 0.0).astype(np.float32),
        'has_prompt_e': np.asarray(os_raw['has_prompt_e'], dtype=bool),
        'has_front_pixel': np.asarray(os_raw['has_front_pixel'], dtype=bool),
        'has_target_pixel': np.asarray(os_raw['has_target_pixel'], dtype=bool),
    }


def _train_ref_bdt(tr_pack: dict[str, Any], tr_idx: np.ndarray, eval_pack: dict[str, Any], seed: int = 42) -> np.ndarray:
    """Controlled Ref-BDT baseline (historical binary pixels + >=2.5 PE + 3-station M1)."""
    clf = HistGradientBoostingClassifier(
        max_iter=350,
        learning_rate=0.05,
        max_leaf_nodes=63,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=seed,
    )
    clf.fit(tr_pack['ref_bdt_feat'][tr_idx], tr_pack['label'][tr_idx])
    probs = clf.predict_proba(eval_pack['ref_bdt_feat'])
    return probs.astype(np.float32)


def run_stage1(epochs: int = 28, seed: int = 42) -> dict[str, Any]:
    """Train & evaluate DiffPhys-SBI + Ref-BDT on Stage-I (`internal_train 25.6k / dev_screen 1.6k / val_eval 4.8k`)."""
    assert_unsealed_split('internal_train', stage=1)
    assert_unsealed_split('internal_val', stage=1)
    pack = _load_pkl_obj(DIFFPHYS_CACHE_DIR / 'train32k_diffphys.pkl')

    spl = np.asarray(pack['internal_split'])
    idx_tr = np.flatnonzero(spl == 'internal_train')
    idx_dv = np.flatnonzero(spl == 'internal_dev_screen')
    idx_va = np.flatnonzero(spl == 'internal_val_eval')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model, hist = _train_diffphys_single(pack, idx_tr, pack, idx_dv, seed=seed, epochs=epochs, device=device)

    STAGE1_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'history': hist}, STAGE1_DIR / 'diffphys_s1_best.pt')

    pred_all = _predict_diffphys_pack(model, pack, device)
    p4_va = pred_all['probs4'][idx_va]
    llr_va = pred_all['tau_llr'][idx_va]
    y_va = pack['label'][idx_va]
    y7_va = pack['label_7'][idx_va]
    hc_va = pack['charm_flag'][idx_va] > 0
    dom_va = pack['obs_domain_id'][idx_va]

    # Ref-BDT on exact same internal_train -> internal_val_eval split
    bdt_p_all = _train_ref_bdt(pack, idx_tr, pack, seed=seed)
    bdt_va = bdt_p_all[idx_va]

    t1_thr = float(np.quantile(llr_va[y_va != 2], 0.99))
    sub_eff = {
        tag: float(np.mean(llr_va[y7_va == c7] >= t1_thr) * 100.0) if (y7_va == c7).any() else 0.0
        for tag, c7 in (('tau_e', 3), ('tau_mu', 4), ('tau_had', 5))
    }
    if 'true_tau_submode' in pack:
        tsm_va = pack['true_tau_submode'][idx_va]
        for tag, sm_id in (('tau_1h', 2), ('tau_3h', 3)):
            m_sm = tsm_va == sm_id
            sub_eff[tag] = float(np.mean(llr_va[m_sm] >= t1_thr) * 100.0) if m_sm.any() else 0.0

    pot_cmp = compare_pot_gpd_vs_empirical_split_half(llr_va, y_va, fpr_targets=(0.01, 0.001), n_splits=120)
    obs_va = {k: v[idx_va] for k, v in _build_nsbi_obs_dict(pack, pred_all).items()}
    nsbi_cmp = evaluate_nsbi_vs_single_bin_closure(
        obs_va, y_va, has_charm=hc_va, total_events=180000.0, tau_fraction=0.003, delta_sys=0.0159
    )
    toy_cls = run_nsbi_toy_mc_closure(nsbi_cmp['templates'], delta_sys=0.0159, n_toys_h0=150, n_toys_h1=150)

    out_doc = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'n_params': int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        'val_eval_events': int(len(idx_va)),
        'best_screen_epoch': int(min(hist, key=lambda r: r['screen_NLL'])['epoch']),
        'history': hist,
        'internal_val_eval_all': _ba4_and_nll(y_va, p4_va),
        'ref_bdt_val_eval_all': _ba4_and_nll(y_va, bdt_va),
        'by_obs_domain': {
            OBS_DOMAIN_NAMES[d]: _ba4_and_nll(y_va[dom_va == d], p4_va[dom_va == d])
            for d in range(4) if (dom_va == d).any()
        },
        'within_slice_1pct_fpr_llr': _within_slice_tail_eval(llr_va, y_va, np.ones(len(y_va), dtype=bool), 0.01),
        'within_slice_0p1pct_fpr_llr': _within_slice_tail_eval(llr_va, y_va, np.ones(len(y_va), dtype=bool), 0.001),
        'tau_submode_at_1pct_fpr_pct': sub_eff,
        'pot_gpd_split_half': pot_cmp,
        'nsbi_pl_closure': {k: v for k, v in nsbi_cmp.items() if k != 'templates'},
        'nsbi_toy_mc_closure': toy_cls,
    }
    write_json(STAGE1_DIR / 'stage1_diffphys_results.json', out_doc)
    return out_doc


def _load_mc8588_tune_meta() -> pd.DataFrame:
    """Locate tune_truth_meta parquet in MC8588 validation_meta or cache."""
    for p in (
        VAL_META_DIR / 'tune_truth_meta.parquet',
        CACHE_DIR / 'tune_truth_meta.parquet',
        CACHE_DIR / 'tune8k_truth_meta.parquet',
    ):
        if p.is_file():
            return pd.read_parquet(assert_unsealed_path(p))
    for d in (VAL_META_DIR, CACHE_DIR):
        if d.is_dir():
            cands = sorted(d.glob('*tune*meta*.parquet'))
            if cands:
                return pd.read_parquet(assert_unsealed_path(cands[0]))
    raise FileNotFoundError(f'Missing tune_truth_meta.parquet in {VAL_META_DIR} or {CACHE_DIR}')


def _eval_saved_ood_and_stress(os_pack: dict[str, Any], seeds: list[int], device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    for seed in seeds:
        s_dir = STAGE2_DIR / f'seed_{seed}'
        npz_path = s_dir / f'ood_and_stress_seed{seed}.npz'
        if not npz_path.is_file():
            ckpt = torch.load(s_dir / f'diffphys_seed{seed}.pt', map_location=device, weights_only=True)
            model = DiffPhysSBIModel().to(device)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()
            os_npz: dict[str, np.ndarray] = {}
            for ov, (o_pack, _) in os_pack['ood_dp'].items():
                os_npz[f'ood_{ov}'] = _predict_diffphys_pack(model, o_pack, device)['probs4']
            for sn, s_pack in os_pack['stress_dp'].items():
                os_npz[f'stress_{sn}'] = _predict_diffphys_pack(model, s_pack, device)['probs4']
            np.savez(npz_path, **os_npz)

    ood_res: dict[str, Any] = {}
    for ov, (_, y_o) in os_pack['ood_dp'].items():
        po = np.mean([np.load(STAGE2_DIR / f'seed_{s}/ood_and_stress_seed{s}.npz')[f'ood_{ov}'].astype(np.float64) for s in seeds], axis=0)
        if len(np.unique(y_o)) >= 4:
            ood_res[ov] = _ba4_and_nll(y_o, po)
        else:
            ood_res[ov] = {
                'n_events': int(len(y_o)),
                'pred_tau_argmax_pct': float((po.argmax(1) == 2).mean() * 100.0),
                'p_tau_gt_0p5_pct': float((po[:, 2] > 0.5).mean() * 100.0),
                'mean_p_tau': float(po[:, 2].mean()),
            }

    y_st = np.asarray(os_pack['y_stress'], dtype=np.int64)
    npz0 = np.load(STAGE2_DIR / f'seed_{seeds[0]}/ood_and_stress_seed{seeds[0]}.npz')
    p_nom = np.mean([np.load(STAGE2_DIR / f'seed_{s}/ood_and_stress_seed{s}.npz')['stress_nominal'].astype(np.float64) for s in seeds], axis=0)
    nom_ba4 = _ba4_and_nll(y_st, p_nom)['BA4_pct']
    stress_names = [k.removeprefix('stress_') for k in npz0.files if k.startswith('stress_')]
    stress_res: dict[str, Any] = {}
    for sn in stress_names:
        ps = np.mean([np.load(STAGE2_DIR / f'seed_{s}/ood_and_stress_seed{s}.npz')[f'stress_{sn}'].astype(np.float64) for s in seeds], axis=0)
        ms = _ba4_and_nll(y_st, ps)
        ms['delta_BA4_vs_nominal_pp'] = float(ms['BA4_pct'] - nom_ba4)
        stress_res[sn] = ms
    return ood_res, stress_res


def run_eval_ood_stress(seeds: list[int]) -> dict[str, Any]:
    """Evaluate already-trained Stage-II DiffPhys-SBI checkpoints on fresh Diag_OOD + Stress400."""
    assert_unsealed_split('ood', stage=2)
    os_pack = _load_pkl_obj(DIFFPHYS_CACHE_DIR / 'ood_stress_diffphys.pkl')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ood_res, stress_res = _eval_saved_ood_and_stress(os_pack, seeds, device)
    res_path = STAGE2_DIR / 'stage2_diffphys_results.json'
    if res_path.is_file():
        doc = json.loads(res_path.read_text())
        doc['ood_evaluation'] = ood_res
        doc['stress400_tests'] = stress_res
        write_json(res_path, doc)
    return {'ood_evaluation': ood_res, 'stress400_tests': stress_res}


def run_stage2(seeds: list[int], epochs: int = 28, *, eval_stress_diag: bool = True) -> dict[str, Any]:
    """Train Ref-BDT & DiffPhys-SBI (3 seeds) on MC8588 Train32k; evaluate Tune6k, Diag_OOD, Stress400, NSBI-PL."""
    for sp in ('train', 'validation', 'tune_screen', 'tune6k', 'ood'):
        assert_unsealed_split(sp, stage=2)

    tr_pack = _load_pkl_obj(DIFFPHYS_CACHE_DIR / 'train32k_diffphys.pkl')
    t8_pack = _load_pkl_obj(DIFFPHYS_CACHE_DIR / 'tune8k_diffphys.pkl')
    os_path = DIFFPHYS_CACHE_DIR / 'ood_stress_diffphys.pkl'
    os_pack = _load_pkl_obj(os_path) if (eval_stress_diag and os_path.is_file()) else None
    tune_meta = _load_mc8588_tune_meta()

    in_screen = tune_meta['in_screen'].to_numpy(dtype=bool) if 'in_screen' in tune_meta.columns else (
        tune_meta['split'].astype(str).to_numpy() == 'tune_screen'
    )
    scr_idx = np.flatnonzero(in_screen)
    tune6k_m = ~in_screen

    y_t8 = tune_meta['label'].to_numpy(dtype=np.int64)
    t8_screen_pack = dict(t8_pack)
    t8_screen_pack['label'] = y_t8

    idx_all_tr = np.arange(len(tr_pack['diffphys_z']), dtype=np.int64)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    STAGE2_DIR.mkdir(parents=True, exist_ok=True)

    # Controlled Ref-BDT baseline on exact same split
    bdt_p4 = _train_ref_bdt(tr_pack, idx_all_tr, t8_pack, seed=42)
    np.save(STAGE2_DIR / 'ref_bdt_probs4_tune8k.npy', bdt_p4)

    seed_p4, seed_llr, seed_ch, seed_vp, seed_sp = [], [], [], [], []
    seed_ba4s, seed_nlls = [], []
    for seed in seeds:
        s_dir = STAGE2_DIR / f'seed_{seed}'
        s_dir.mkdir(parents=True, exist_ok=True)
        model, hist = _train_diffphys_single(tr_pack, idx_all_tr, t8_screen_pack, scr_idx, seed=seed, epochs=epochs, device=device, tag='s2')
        torch.save({'model_state_dict': model.state_dict(), 'history': hist}, s_dir / f'diffphys_seed{seed}.pt')

        pred_t8 = _predict_diffphys_pack(model, t8_pack, device)
        np.save(s_dir / f'probs4_tune8k_seed{seed}.npy', pred_t8['probs4'])
        np.save(s_dir / f'tau_llr_tune8k_seed{seed}.npy', pred_t8['tau_llr'])
        np.save(s_dir / f'v_pv_tune8k_seed{seed}.npy', pred_t8['v_pv_mm'])
        np.save(s_dir / f'scale_pv_tune8k_seed{seed}.npy', pred_t8['scale_pv_mm'])
        nsbi_obs_s = _build_nsbi_obs_dict(t8_pack, pred_t8)
        np.savez(s_dir / f'nsbi_obs_tune8k_seed{seed}.npz', **nsbi_obs_s)

        if os_pack is not None:
            os_npz: dict[str, np.ndarray] = {}
            for ov, (o_pack, _) in os_pack['ood_dp'].items():
                os_npz[f'ood_{ov}'] = _predict_diffphys_pack(model, o_pack, device)['probs4']
            for sn, s_pack in os_pack['stress_dp'].items():
                os_npz[f'stress_{sn}'] = _predict_diffphys_pack(model, s_pack, device)['probs4']
            np.savez(s_dir / f'ood_and_stress_seed{seed}.npz', **os_npz)

        m_s = _ba4_and_nll(y_t8[tune6k_m], pred_t8['probs4'][tune6k_m])
        seed_ba4s.append(m_s['BA4_pct'])
        seed_nlls.append(m_s['NLL'])
        seed_p4.append(pred_t8['probs4'].astype(np.float64))
        seed_llr.append(pred_t8['tau_llr'].astype(np.float64))
        seed_ch.append(pred_t8['charm_prob'].astype(np.float64))
        seed_vp.append(pred_t8['v_pv_mm'].astype(np.float64))
        seed_sp.append(pred_t8['scale_pv_mm'].astype(np.float64))

    ens_p4 = np.mean(seed_p4, axis=0)
    ens_llr = np.mean(seed_llr, axis=0)
    ens_ch = np.mean(seed_ch, axis=0)

    np.save(STAGE2_DIR / 'probs4_tune8k_ens3.npy', ens_p4.astype(np.float32))
    np.save(STAGE2_DIR / 'tau_llr_tune8k_ens3.npy', ens_llr.astype(np.float32))
    ens_nsbi = _build_nsbi_obs_dict(t8_pack, {'tau_llr': ens_llr, 'charm_prob': ens_ch})
    np.savez(STAGE2_DIR / 'nsbi_obs_tune8k_ens3.npz', **ens_nsbi)

    t_cal = _fit_scalar_temperature_on_screen(ens_p4[scr_idx], y_t8[scr_idx])
    inv_t = float(t_cal['inv_T'])
    cal_log = np.log(np.clip(ens_p4, 1e-8, 1.0)) * inv_t
    cal_log -= cal_log.max(axis=1, keepdims=True)
    ens_p4_cal = np.exp(cal_log)
    ens_p4_cal /= np.maximum(ens_p4_cal.sum(axis=1, keepdims=True), 1e-12)

    if 'row' not in tune_meta.columns and 'evt_id' in tune_meta.columns:
        tune_meta = tune_meta.assign(row=tune_meta['evt_id'])
    obs_dom = tune_meta['obs_domain_id'].to_numpy(dtype=np.int64)
    pc_m = tune_meta['pixel_centered'].to_numpy(dtype=bool) if 'pixel_centered' in tune_meta.columns else (obs_dom == 1)
    fp_m = tune_meta['front_pinpoint_pixel_centered'].to_numpy(dtype=bool) if 'front_pinpoint_pixel_centered' in tune_meta.columns else pc_m
    interm_m = pc_m & (~fp_m)
    scint_only_m = (~pc_m) & (obs_dom == 2)
    geom_w = (
        tune_meta['geom_weight'].to_numpy(dtype=np.float64)
        if 'geom_weight' in tune_meta.columns
        else (tune_meta['sample_weight'].to_numpy(dtype=np.float64) if 'sample_weight' in tune_meta.columns else np.ones(len(y_t8), dtype=np.float64))
    )
    has_ch = (
        tune_meta['has_charm'].to_numpy(dtype=bool)
        if 'has_charm' in tune_meta.columns
        else (tune_meta['charm_flag'].to_numpy(dtype=bool) if 'charm_flag' in tune_meta.columns else np.zeros(len(y_t8), dtype=bool))
    )
    tau_l3d = tune_meta['tau_decay_L3d_mm'].to_numpy(dtype=np.float64) if 'tau_decay_L3d_mm' in tune_meta.columns else np.full(len(y_t8), -1.0)
    fam_cols = [c for c in FAMILY if c in tune_meta.columns] or ['run', 'chunk']
    fam_codes = pd.MultiIndex.from_frame(tune_meta[fam_cols]).factorize()[0]

    t6k_slices = {
        'All_Tune6k': tune6k_m,
        'D1_obs': tune6k_m & (obs_dom == 1),
        'D2_obs': tune6k_m & (obs_dom == 2),
        'D3_obs': tune6k_m & (obs_dom == 3),
        'Pinpoint_PixXY': tune6k_m & pc_m,
        'Front30': tune6k_m & fp_m,
        'Interm150': tune6k_m & interm_m,
        'Fortune_ScintOnly': tune6k_m & scint_only_m,
    }

    thr_1pct_llr = float(np.quantile(ens_llr[tune6k_m & (y_t8 != 2)], 0.99))
    submode_eff: dict[str, Any] = {}
    if 'true_tau_submode' in tune_meta.columns:
        tsm = tune_meta['true_tau_submode'].to_numpy(dtype=np.int64)
        for sb, sid in (('tau_e', 0), ('tau_mu', 1), ('tau_1h', 2), ('tau_3h', 3)):
            sm = tune6k_m & (y_t8 == 2) & (tsm == sid)
            submode_eff[sb] = {
                'n_events': int(sm.sum()),
                'eff_1pct_fpr_pct': float(np.mean(ens_llr[sm] >= thr_1pct_llr) * 100.0) if sm.any() else 0.0,
            }
        sm_had = tune6k_m & (y_t8 == 2) & np.isin(tsm, [2, 3])
        submode_eff['tau_had'] = {
            'n_events': int(sm_had.sum()),
            'eff_1pct_fpr_pct': float(np.mean(ens_llr[sm_had] >= thr_1pct_llr) * 100.0) if sm_had.any() else 0.0,
        }

    ltau_eff: dict[str, Any] = {}
    for lo, hi in ((0.0, 1.0), (1.0, 3.0), (3.0, 10.0), (10.0, 30.0), (30.0, 1e9)):
        lm = tune6k_m & (y_t8 == 2) & (tau_l3d >= lo) & (tau_l3d < hi)
        ltau_eff[f'[{lo:g},{hi:g})_mm'] = {
            'n_events': int(lm.sum()),
            'eff_1pct_fpr_pct': float(np.mean(ens_llr[lm] >= thr_1pct_llr) * 100.0) if lm.any() else 0.0,
        }
    lm_lt30 = tune6k_m & (y_t8 == 2) & (tau_l3d >= 0.0) & (tau_l3d < 30.0)
    ltau_eff['L_lt_30mm'] = {
        'n_events': int(lm_lt30.sum()),
        'eff_1pct_fpr_pct': float(np.mean(ens_llr[lm_lt30] >= thr_1pct_llr) * 100.0) if lm_lt30.any() else 0.0,
    }

    pot_t6k = compare_pot_gpd_vs_empirical_split_half(ens_llr[tune6k_m], y_t8[tune6k_m], fpr_targets=(0.01, 0.001), n_splits=150)
    obs_t6k = {k: v[tune6k_m] for k, v in ens_nsbi.items()}
    nsbi_t6k = evaluate_nsbi_vs_single_bin_closure(
        obs_t6k, y_t8[tune6k_m], has_charm=has_ch[tune6k_m], mw2t_weights=geom_w[tune6k_m],
        total_events=180000.0, tau_fraction=0.003, delta_sys=0.0159,
    )
    toy_t6k = run_nsbi_toy_mc_closure(nsbi_t6k['templates'], delta_sys=0.0159, n_toys_h0=200, n_toys_h1=200)

    if os_pack is not None:
        ood_res, stress_res = _eval_saved_ood_and_stress(os_pack, seeds, device)
    else:
        ood_res, stress_res = {}, {}

    boot_vs_bdt = {
        sn: _paired_family_bootstrap_full(ens_p4, bdt_p4.astype(np.float64), y_t8, fam_codes, mk, n_boot=1000, seed=20263250)
        for sn, mk in (('All_Tune6k', t6k_slices['All_Tune6k']), ('D1_obs', t6k_slices['D1_obs']), ('Fortune_ScintOnly', t6k_slices['Fortune_ScintOnly']))
        if mk.any()
    }

    bdt_s_tau = np.log(np.clip(bdt_p4[:, 2], 1e-8, 1.0)) - np.log(np.clip(1.0 - bdt_p4[:, 2], 1e-8, 1.0))
    ref_bdt_doc = {
        'slices': {sn: _ba4_and_nll(y_t8[mk], bdt_p4[mk]) for sn, mk in t6k_slices.items() if mk.any()},
        'mass_weighted_2t_BA4_pct': _weighted_ba4_pct(y_t8[tune6k_m], bdt_p4[tune6k_m], geom_w[tune6k_m]),
        'within_slice_1pct_fpr': {
            sn: _within_slice_tail_eval(bdt_s_tau, y_t8, t6k_slices[sn], 0.01)
            for sn in ('All_Tune6k', 'D1_obs', 'Fortune_ScintOnly') if t6k_slices[sn].any()
        },
    }
    write_json(STAGE2_DIR / 'ref_bdt_results.json', ref_bdt_doc)

    out_doc = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'cross_seed_summary': {
            'tune6k_BA4_mean_pct': float(np.mean(seed_ba4s)),
            'tune6k_BA4_std_pp': float(np.std(seed_ba4s, ddof=0)),
            'tune6k_NLL_mean': float(np.mean(seed_nlls)),
            'tune6k_NLL_std': float(np.std(seed_nlls, ddof=0)),
        },
        'temperature_tune_screen': t_cal,
        'slices_raw': {sn: _ba4_and_nll(y_t8[mk], ens_p4[mk]) for sn, mk in t6k_slices.items() if mk.any()},
        'slices_cal': {sn: _ba4_and_nll(y_t8[mk], ens_p4_cal[mk]) for sn, mk in t6k_slices.items() if mk.any()},
        'mass_weighted_2t_BA4_pct': _weighted_ba4_pct(y_t8[tune6k_m], ens_p4[tune6k_m], geom_w[tune6k_m]),
        'ref_bdt_tune6k': ref_bdt_doc,
        'within_slice_1pct_fpr_llr': {
            sn: _within_slice_tail_eval(ens_llr, y_t8, t6k_slices[sn], 0.01)
            for sn in ('All_Tune6k', 'D1_obs', 'Pinpoint_PixXY', 'Front30', 'Fortune_ScintOnly') if t6k_slices[sn].any()
        },
        'within_slice_0p1pct_fpr_llr': {
            sn: _within_slice_tail_eval(ens_llr, y_t8, t6k_slices[sn], 0.001)
            for sn in ('All_Tune6k', 'D1_obs') if t6k_slices[sn].any()
        },
        'split_half_1pct_fpr_llr': _split_half_tail_eval(ens_llr, y_t8, tune_meta, tune6k_m, 0.01),
        'tau_submode_at_1pct_fpr': submode_eff,
        'ltau_at_1pct_fpr': ltau_eff,
        'pot_gpd_split_half': pot_t6k,
        'nsbi_pl_closure': {k: v for k, v in nsbi_t6k.items() if k != 'templates'},
        'nsbi_toy_mc_closure': toy_t6k,
        'ood_evaluation': ood_res,
        'stress400_tests': stress_res,
        'paired_family_bootstrap_vs_ref_bdt': boot_vs_bdt,
    }
    write_json(STAGE2_DIR / 'stage2_diffphys_results.json', out_doc)
    return out_doc


def _submit_condor_job(
    jname: str,
    cmd_args: str,
    *,
    cpus: int = 8,
    mem: str = '32GB',
    disk: str = '16GB',
    flavour: str = 'workday',
    use_gpu: bool = True,
) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[4]
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    (EXP_ROOT / 'logs').mkdir(parents=True, exist_ok=True)
    (EXP_ROOT / 'code').mkdir(parents=True, exist_ok=True)
    CONDOR_DIR.mkdir(parents=True, exist_ok=True)

    archive = EXP_ROOT / 'code' / f'gen4_diffphys_{jname}_{int(time.time())}.tar.gz'
    _package_source(repo, archive)
    chosen, _, _ = choose_schedd(EXP_ROOT)

    job_dir = CONDOR_DIR / jname
    job_dir.mkdir(parents=True, exist_ok=True)
    n_omp = 1 if '--preextract-diffphys' in cmd_args else min(2, max(1, int(cpus)))
    script = [
        '#!/bin/bash',
        'set -e',
        '{ which() { command -v "$@"; }; export -f which; }',
        'SAVED_ARGS=("$@"); set --',
        'source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh >/dev/null 2>&1',
        'set -- "${SAVED_ARGS[@]}"',
        f'export OMP_NUM_THREADS={n_omp} OPENBLAS_NUM_THREADS={n_omp} MKL_NUM_THREADS={n_omp} NUMEXPR_NUM_THREADS={n_omp}',
        'export PINPOINT_V3_SCRATCH="${_CONDOR_SCRATCH_DIR:-/tmp}/gen4_dp_${PINPOINT_JOB_ID}"',
        'mkdir -p "$PINPOINT_V3_SCRATCH/code"',
        f'tar -xzf {shlex.quote(str(archive))} -C "$PINPOINT_V3_SCRATCH/code"',
        'export PYTHONPATH="$PINPOINT_V3_SCRATCH/code/src:${PYTHONPATH:-}"',
        'cd "$PINPOINT_V3_SCRATCH"',
        f'exec python3 -u -m analysis.flavour.gen4.run_diffphys_sbi {cmd_args}',
    ]
    wrapper = job_dir / 'run.sh'
    wrapper.write_text('\n'.join(script) + '\n')
    wrapper.chmod(0o755)

    sub_lines = [
        'universe = vanilla',
        f'executable = {wrapper}',
        f'initialdir = {job_dir}',
        f'output = {EXP_ROOT}/logs/{jname}.$(ClusterId).out',
        f'error = {EXP_ROOT}/logs/{jname}.$(ClusterId).err',
        f'log = {EXP_ROOT}/logs/{jname}.$(ClusterId).log',
        'environment = "PINPOINT_JOB_ID=$(ClusterId).$(Process)"',
        f'request_cpus = {cpus}',
        f'request_memory = {mem}',
        f'request_disk = {disk}',
        'requirements = (OpSysAndVer == "AlmaLinux9")',
        f'+JobFlavour = "{flavour}"',
        'should_transfer_files = YES',
        'when_to_transfer_output = ON_EXIT',
        'transfer_output_files = ""',
    ]
    if use_gpu:
        sub_lines.extend([
            'request_gpus = 1',
            'require_gpus = ((Capability >= 7.5) || ((Capability =?= UNDEFINED) && regexp("^NVIDIA H100.* MIG ", DeviceName)))',
        ])
    sub_lines.append('queue 1')
    sub_path = job_dir / 'job.sub'
    sub_path.write_text('\n'.join(sub_lines) + '\n')
    resp = subprocess.run(['condor_submit', '-name', chosen, str(sub_path)], capture_output=True, text=True)
    rec = {'job': jname, 'schedd': chosen, 'stdout': resp.stdout.strip(), 'stderr': resp.stderr.strip(), 'returncode': resp.returncode}
    write_json(EXP_ROOT / f'{jname}.submission.json', rec)
    if resp.returncode != 0:
        raise RuntimeError(f'condor_submit failed: {resp.stderr}')
    return rec


def run_reeval_nsbi() -> dict[str, Any]:
    """Re-evaluate Stage-I & Stage-II NSBI-PL + Toy-MC closure using saved models/predictions."""
    out: dict[str, Any] = {}
    s1_ckpt = STAGE1_DIR / 'diffphys_s1_best.pt'
    s1_json = STAGE1_DIR / 'stage1_diffphys_results.json'
    if s1_ckpt.is_file() and s1_json.is_file():
        assert_unsealed_split('internal_val', stage=1)
        pack = _load_pkl_obj(DIFFPHYS_CACHE_DIR / 'train32k_diffphys.pkl')
        idx_va = np.flatnonzero(np.asarray(pack['internal_split']) == 'internal_val_eval')
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = DiffPhysSBIModel().to(device)
        model.load_state_dict(torch.load(s1_ckpt, map_location=device, weights_only=True)['model_state_dict'])
        pred_all = _predict_diffphys_pack(model, pack, device)
        obs_va = {k: v[idx_va] for k, v in _build_nsbi_obs_dict(pack, pred_all).items()}
        y_va = pack['label'][idx_va]
        hc_va = pack['charm_flag'][idx_va] > 0
        nsbi_cmp = evaluate_nsbi_vs_single_bin_closure(
            obs_va, y_va, has_charm=hc_va, total_events=180000.0, tau_fraction=0.003, delta_sys=0.0159
        )
        toy_cls = run_nsbi_toy_mc_closure(nsbi_cmp['templates'], delta_sys=0.0159, n_toys_h0=150, n_toys_h1=150)
        doc1 = json.loads(s1_json.read_text())
        doc1['nsbi_pl_closure'] = {k: v for k, v in nsbi_cmp.items() if k != 'templates'}
        doc1['nsbi_toy_mc_closure'] = toy_cls
        write_json(s1_json, doc1)
        out['stage1'] = doc1['nsbi_pl_closure']

    s2_obs_p = STAGE2_DIR / 'nsbi_obs_tune8k_ens3.npz'
    s2_json = STAGE2_DIR / 'stage2_diffphys_results.json'
    if s2_obs_p.is_file() and s2_json.is_file():
        assert_unsealed_split('tune6k', stage=2)
        tune_meta = _load_mc8588_tune_meta()
        in_screen = tune_meta['in_screen'].to_numpy(dtype=bool) if 'in_screen' in tune_meta.columns else (
            tune_meta['split'].astype(str).to_numpy() == 'tune_screen'
        )
        tune6k_m = ~in_screen
        y_t8 = tune_meta['label'].to_numpy(dtype=np.int64)
        has_ch = (
            tune_meta['has_charm'].to_numpy(dtype=bool)
            if 'has_charm' in tune_meta.columns
            else (tune_meta['charm_flag'].to_numpy(dtype=bool) if 'charm_flag' in tune_meta.columns else np.zeros(len(y_t8), dtype=bool))
        )
        geom_w = (
            tune_meta['geom_weight'].to_numpy(dtype=np.float64)
            if 'geom_weight' in tune_meta.columns
            else (tune_meta['sample_weight'].to_numpy(dtype=np.float64) if 'sample_weight' in tune_meta.columns else np.ones(len(y_t8), dtype=np.float64))
        )
        d_obs = dict(np.load(s2_obs_p))
        obs_t6k = {k: v[tune6k_m] for k, v in d_obs.items()}
        nsbi_t6k = evaluate_nsbi_vs_single_bin_closure(
            obs_t6k, y_t8[tune6k_m], has_charm=has_ch[tune6k_m], mw2t_weights=geom_w[tune6k_m],
            total_events=180000.0, tau_fraction=0.003, delta_sys=0.0159,
        )
        toy_t6k = run_nsbi_toy_mc_closure(nsbi_t6k['templates'], delta_sys=0.0159, n_toys_h0=200, n_toys_h1=200)
        doc2 = json.loads(s2_json.read_text())
        doc2['nsbi_pl_closure'] = {k: v for k, v in nsbi_t6k.items() if k != 'templates'}
        doc2['nsbi_toy_mc_closure'] = toy_t6k
        write_json(s2_json, doc2)
        out['stage2'] = doc2['nsbi_pl_closure']
    return out


def main() -> None:
    guard = SealAuditGuard()
    guard.__enter__()
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--health-check', action='store_true')
        parser.add_argument('--audit-fista-purity', action='store_true')
        parser.add_argument('--preextract-diffphys', action='store_true')
        parser.add_argument('--skip-stress-diag', action='store_true')
        parser.add_argument('--stage', type=int, default=2)
        parser.add_argument('--workers', type=int, default=8)
        parser.add_argument('--s1-train', action='store_true')
        parser.add_argument('--s2-train', action='store_true')
        parser.add_argument('--eval-ood-stress', action='store_true')
        parser.add_argument('--reeval-nsbi', action='store_true')
        parser.add_argument('--seeds', nargs='+', type=int, default=[17, 42, 2026])
        parser.add_argument('--epochs', type=int, default=28)
        parser.add_argument('--submit', action='store_true')
        parser.add_argument('--no-gpu', action='store_true')
        parser.add_argument('--cpus', type=int, default=8)
        parser.add_argument('--mem', type=str, default='24GB')
        parser.add_argument('--job-name', type=str, default='diffphys_full')
        parser.add_argument('--cmd-args', type=str, default='--preextract-diffphys --stage 2 --s1-train --s2-train')
        args = parser.parse_args()

        is_stage1_only = (
            (not args.s2_train)
            and (not args.eval_ood_stress)
            and (not args.reeval_nsbi)
            and (not (args.preextract_diffphys and args.stage >= 2))
            and (not args.submit)
        )
        if not args.submit:
            install_mc8588_seal_hook(stage=1 if is_stage1_only else 2)

        if args.health_check:
            print(json.dumps(run_health_and_small_sample_overfit(), indent=2))
        if args.audit_fista_purity:
            print(json.dumps(run_fista_purity_audit(), indent=2))
        if args.preextract_diffphys:
            print(json.dumps(run_preextract_diffphys(stage=args.stage, workers=args.workers, include_stress_diag=not args.skip_stress_diag), indent=2))
        if args.s1_train:
            res = run_stage1(epochs=args.epochs)
            print(json.dumps({'val_eval_all': res['internal_val_eval_all'], 'ref_bdt': res['ref_bdt_val_eval_all'], 'nsbi_pl': res['nsbi_pl_closure']}, indent=2))
        if args.s2_train:
            res = run_stage2(seeds=args.seeds, epochs=args.epochs, eval_stress_diag=not args.skip_stress_diag)
            print(json.dumps({'cross_seed_summary': res['cross_seed_summary'], 'ref_bdt': res['ref_bdt_tune6k'], 'nsbi_pl': res['nsbi_pl_closure']}, indent=2))
        if args.eval_ood_stress:
            print(json.dumps(run_eval_ood_stress(seeds=args.seeds), indent=2))
        if args.reeval_nsbi:
            print(json.dumps(run_reeval_nsbi(), indent=2))
        if args.submit:
            print(json.dumps(_submit_condor_job(args.job_name, args.cmd_args, cpus=args.cpus, mem=args.mem, use_gpu=not args.no_gpu), indent=2))
    finally:
        guard.__exit__(None, None, None)


if __name__ == '__main__':
    main()
