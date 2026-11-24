"""High-speed 10085-10088 ROOT -> Pinpoint-Gen4 feature cache builder & collate_gen4_batch.

Implements `[gen4-contract-v2]` (`chunk_id % 20` interleaved stratified splits):
  - `Train32k`    : r in {0..11}  (8,000 / flavour = 32,000; full truth & cluster sidecars)
  - `Tune_Screen` : r == 12       (500 / flavour = 2,000; 0 truth in cache pkl)
  - `Tune6k`      : r == 13       (1,500 / flavour = 6,000; 0 truth in cache pkl)
  - `Stress400`   : r == 13       (100 / flavour = 400 x 7 STAGE2_STRESS_SPECS; 0 truth)
  - `Diag_OOD`    : r == 13 rem   (`InDet_NC_Charm`, `NonW_Al_Scint`, `HighE_gt1p5TeV`; 0 truth in obs)
  - `SSL_320k`    : r in {14, 15} (D1 U D2 natural flux; 0 truth anywhere)
  - `SEALED`      : r in {16..19} (`Cal-A / Cut-B / Confirm`) -> PHYSICALLY BLOCKED!

Zero deprecated `10081-10084` / `v6_clean_v3` paths are ever referenced or read.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import re
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import uproot

from analysis.flavour.gen4.geometry_contract import (
    COMMON_Y_AXIS_MM,
    Gen4GeometryContract,
    SI_SUB_X_AXIS_MM,
    load_geometry_contract,
)
from analysis.flavour.gen4.physics_frontend import (
    N_MAX_CLUSTERS_GEN4,
    N_MAX_PATCHES_GEN4,
    PATCH_SIZE,
    PIX_FEAT_GEN4_DIM,
    PIX_PLANE_SUMMARY_DIM,
    PIXEL_ENC_MEV,
    PIXEL_THR_MEV,
    SCINT_GRID_CHANNELS,
    build_observation_from_raw_root_event,
    digitize_raw_faser_hits,
    digitize_raw_pixel_hits,
    digitize_raw_scint_hits,
    extract_gen4_frontend_event,
)
from analysis.flavour.nextgen.seal_guard import (
    SealViolationError,
    assert_unsealed_path,
    assert_unsealed_split,
)


RAW_MC_ROOT = Path('/eos/project-f/fasersim-bonn/public/pinpoint/data')
GEN4_MC8588_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage329_gen4_mc8588')
GEN4_CACHE_DIR = GEN4_MC8588_ROOT / 'cache'
GEN4_VAL_META_DIR = GEN4_MC8588_ROOT / 'validation_meta'

# Dataset ID -> (name, label_4, base_label_7)
MC8588_DATASETS: dict[str, tuple[str, int, int]] = {
    '10085': ('nue_CC', 0, 0),
    '10086': ('numu_CC', 1, 1),
    '10087': ('nutau_CC', 2, 3),
    '10088': ('NC', 3, 6),
}

# Exact 7 stress perturbation specs matching run_tracklet_ipformer.py
STAGE2_STRESS_SPECS: tuple[tuple[str, dict[str, Any]], ...] = (
    ('nominal', {}),
    ('pixel_dead_2pct', {'pixel_dead': 0.02}),
    ('pixel_dead_5pct', {'pixel_dead': 0.05}),
    ('pixel_dark_1e-5', {'pixel_dark_rate': 1e-5}),
    ('tracker_dead_2pct_noise1', {'tracker_dead': 0.02, 'tracker_noise': 1}),
    ('tracker_dead_5pct_noise2', {'tracker_dead': 0.05, 'tracker_noise': 2}),
    ('sipm_sat_cap_0p95', {'sipm_sat_cap': 0.95}),
)

GEN4_PURE_OBS_KEYS: tuple[str, ...] = (
    'c_obs',
    'obs_domain_id',
    'cal_start_mm',
    'has_subthr_readout',
    'has_edep',
    's_mir',
    'pix_xyz_mm',
    'pix_plane',
    'pix_valid',
    'pix_feat',
    'pix_plane_summary',
    'pix_patches',
    'patch_cluster_idx',
    'scint_grid',
    'muon_kalman_tracks',
    'muon_kalman_mask',
    'trk_cand',
    'trk_cand_mask',
    'trk_station_counts',
)

GEN4_TRUTH_BATCH_KEYS: tuple[str, ...] = (
    'label',
    'label_7',
    'tau_branch_valid',
    'charm_flag',
    'charm_valid',
    'vertex_xyz_mm',
    'vertex_valid',
    'vis_kinematics',
    'kin_valid',
    'sample_weight',
    'cl_sidecar_valid',
    'cl_origin_label',
    'cl_secondary_origin_label',
    'cl_dominant_edep_frac',
    'cl_sv_group_id',
    'cl_dominant_track_id',
    'true_sv_xyz_mm',
    'has_true_sv',
    'tau_n_charged_prongs_si',
    'tier1b_genie_valid',
    'true_tau_submode',
    'true_n_pi0',
    'true_nu_energy_gev',
    'true_branch_p4_gev',
    'tgt_Z',
    'tau_decay_L3d_mm',
)

_FORBIDDEN_PATH_TOKENS = (
    '10081',
    '10082',
    '10083',
    '10084',
    'v6_clean_v3',
    'l1_cluster_sidecar',
    'l2_ssl_300k',
    'sealed_truth_meta',
    'planned_manifest',
)

_CHUNK_RE = re.compile(r'1008[5-8]_(\d+)\.root$')


def parse_mc8588_chunk_id(path: Path | str) -> int:
    m = _CHUNK_RE.search(str(path))
    if not m:
        raise SealViolationError(f'Not a recognized 10085-10088 ROOT file path: {path}')
    return int(m.group(1))


def assert_unsealed_mc8588_path(path: Path | str) -> Path:
    """Enforce zero access to deprecated MCs and sealed chunks (r = chunk_id % 20 >= 16)."""
    s = str(path)
    for tok in _FORBIDDEN_PATH_TOKENS:
        if tok in s:
            raise SealViolationError(f'Forbidden deprecated/sealed path token "{tok}" in {s}')
    p = assert_unsealed_path(path)
    if p.suffix == '.root':
        cid = parse_mc8588_chunk_id(p)
        if (cid % 20) >= 16:
            raise SealViolationError(
                f'SEALED 10085-10088 chunk blocked (chunk_id={cid}, mod20={cid % 20} in {{16..19}}): {p}'
            )
    return p


def partition_mc8588_chunks() -> dict[str, dict[str, list[tuple[int, Path]]]]:
    """Return unsealed 10085-10088 ROOT files strictly partitioned by `r = chunk_id % 20`."""
    splits: dict[str, dict[str, list[tuple[int, Path]]]] = {
        'train': {ds: [] for ds in MC8588_DATASETS},
        'tune_screen': {ds: [] for ds in MC8588_DATASETS},
        'tune6k': {ds: [] for ds in MC8588_DATASETS},
        'ssl_train': {ds: [] for ds in MC8588_DATASETS},
        'ssl_val': {ds: [] for ds in MC8588_DATASETS},
    }
    for ds in MC8588_DATASETS:
        ds_dir = RAW_MC_ROOT / ds
        pairs: list[tuple[int, Path]] = []
        for f in ds_dir.glob(f'{ds}_*.root'):
            cid = parse_mc8588_chunk_id(f)
            r = cid % 20
            if r >= 16:
                continue  # Sealed Cal-A / Cut-B / Confirm: never touched
            pairs.append((cid, f))
        pairs.sort(key=lambda x: x[0])

        r15: list[tuple[int, Path]] = []
        for cid, f in pairs:
            r = cid % 20
            if r <= 11:
                splits['train'][ds].append((cid, f))
            elif r == 12:
                splits['tune_screen'][ds].append((cid, f))
            elif r == 13:
                splits['tune6k'][ds].append((cid, f))
            elif r == 14:
                splits['ssl_train'][ds].append((cid, f))
            elif r == 15:
                r15.append((cid, f))
        n_val = max(1, int(round(0.15 * len(r15))))
        splits['ssl_train'][ds].extend(r15[:-n_val])
        splits['ssl_val'][ds].extend(r15[-n_val:])
    return splits


def _build_evt_group_map(tree_evt_ids: np.ndarray, event_evt_ids: np.ndarray) -> list[np.ndarray]:
    """O(N log N) group index builder mapping `event.evtID` -> row indices for primaries/tau/charm trees."""
    n_ev = len(event_evt_ids)
    groups: list[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(n_ev)]
    if n_ev == 0 or len(tree_evt_ids) == 0:
        return groups
    ev_to_local = {int(ev_id): idx for idx, ev_id in enumerate(np.asarray(event_evt_ids, dtype=np.int64).tolist())}
    order = np.argsort(tree_evt_ids, kind='mergesort')
    e_sorted = tree_evt_ids[order]
    uniq_e, starts, counts = np.unique(e_sorted, return_index=True, return_counts=True)
    for ue, s, c in zip(uniq_e.tolist(), starts.tolist(), counts.tolist()):
        loc = ev_to_local.get(int(ue))
        if loc is not None:
            groups[loc] = order[s : s + c]
    return groups


def _extract_event_truth_record(
    ds: str,
    eid: int,
    ev_tree: dict[str, np.ndarray],
    prim_rows: np.ndarray,
    pr_tree: dict[str, np.ndarray],
    tau_rows: np.ndarray,
    tau_tree: dict[str, np.ndarray],
    charm_rows: np.ndarray,
    charm_tree: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Extract Gen4 truth fields from native 10085-10088 trees (converted to GeV, mm)."""
    _, label_4, base_l7 = MC8588_DATASETS[ds]
    init_e_gev = float(ev_tree['initE'][eid]) * 1e-3
    pv = np.array([ev_tree['initX'][eid], ev_tree['initY'][eid], ev_tree['initZ'][eid]], dtype=np.float32)
    fsl_pdg = int(ev_tree['fslPDG'][eid])
    tgt_z = int(ev_tree['tgtZ'][eid])

    has_charm = len(charm_rows) > 0
    charm_flag = 1 if has_charm else 0

    # Label 7 & tau decay submode
    true_tau_submode = -1
    tau_branch_valid = 0
    true_n_pi0 = 0
    has_true_sv = False
    true_sv = np.zeros(3, dtype=np.float32)
    tau_l3d = 0.0
    lep_p4 = np.zeros(4, dtype=np.float64)
    sec_p4 = np.zeros(4, dtype=np.float64)

    if label_4 == 0:
        label_7 = 0
    elif label_4 == 1:
        label_7 = 1 if fsl_pdg == 13 else 2
    elif label_4 == 3:
        label_7 = 6
    else:
        # nutau_CC: classify direct tau decay daughters (sharing initial decay ProdZ)
        if len(tau_rows) > 0:
            tz = tau_tree['ProdZ'][tau_rows]
            direct = np.abs(tz - tz[0]) < 1e-3
            d_rows = tau_rows[direct]
            d_pdg = tau_tree['PDG'][d_rows]
            abs_pdg = np.abs(d_pdg)
            true_n_pi0 = int(np.count_nonzero(abs_pdg == 111))
            true_sv = np.array(
                [tau_tree['ProdX'][d_rows[0]], tau_tree['ProdY'][d_rows[0]], tau_tree['ProdZ'][d_rows[0]]],
                dtype=np.float32,
            )
            has_true_sv = True
            tau_l3d = float(np.linalg.norm(true_sv - pv))
            tau_branch_valid = 1

            # Visible tau daughters (excluding neutrinos 12/14/16)
            vis_tau = ~np.isin(abs_pdg, [12, 14, 16])
            if vis_tau.any():
                masses = np.where(abs_pdg[vis_tau] == 11, 0.000511, np.where(abs_pdg[vis_tau] == 13, 0.10566, np.where(abs_pdg[vis_tau] == 111, 0.13498, 0.13957)))
                px_v = tau_tree['Px'][d_rows[vis_tau]] * 1e-3
                py_v = tau_tree['Py'][d_rows[vis_tau]] * 1e-3
                pz_v = tau_tree['Pz'][d_rows[vis_tau]] * 1e-3
                e_v = np.sqrt(px_v**2 + py_v**2 + pz_v**2 + masses**2)
                lep_p4 = np.array([px_v.sum(), py_v.sum(), pz_v.sum(), e_v.sum()], dtype=np.float64)

            if np.any(abs_pdg == 13):
                label_7 = 4
                true_tau_submode = 1
            elif np.any(abs_pdg == 11):
                label_7 = 3
                true_tau_submode = 0
            else:
                label_7 = 5
                n_ch_h = int(np.count_nonzero(np.isin(abs_pdg, [211, 321, 2212])))
                true_tau_submode = 3 if n_ch_h >= 3 else 2
        else:
            label_7 = 5

    if not has_true_sv and has_charm:
        cz = charm_tree['DecayZ'][charm_rows]
        # Finite physical decay inside detector (Sentinel 10000.0 means undecayed)
        valid_c = np.abs(cz) < 9000.0
        if valid_c.any():
            r_c = charm_rows[valid_c][0]
            true_sv = np.array([charm_tree['DecayX'][r_c], charm_tree['DecayY'][r_c], charm_tree['DecayZ'][r_c]], dtype=np.float32)
            has_true_sv = True

    # Visible kinematics & 4-branch p4 from primaries (excluding neutrinos 12/14/16)
    had_p4 = np.zeros(4, dtype=np.float64)
    if len(prim_rows) > 0:
        ppdg = np.abs(pr_tree['PDG'][prim_rows])
        is_nu = np.isin(ppdg, [12, 14, 16])
        is_fsl = ppdg == abs(fsl_pdg)
        px_p = pr_tree['Px'][prim_rows] * 1e-3
        py_p = pr_tree['Py'][prim_rows] * 1e-3
        pz_p = pr_tree['Pz'][prim_rows] * 1e-3
        e_p = pr_tree['E'][prim_rows] * 1e-3

        if label_4 in (0, 1) and np.any(is_fsl):
            m_f = is_fsl & (~is_nu)
            lep_p4 = np.array([px_p[m_f].sum(), py_p[m_f].sum(), pz_p[m_f].sum(), e_p[m_f].sum()], dtype=np.float64)
            m_h = (~is_fsl) & (~is_nu)
        else:
            m_h = (~is_fsl) & (~is_nu)
        if m_h.any():
            had_p4 = np.array([px_p[m_h].sum(), py_p[m_h].sum(), pz_p[m_h].sum(), e_p[m_h].sum()], dtype=np.float64)
        if label_4 != 2:
            true_n_pi0 += int(np.count_nonzero(ppdg[m_h] == 111))

    vis_p4 = lep_p4 + had_p4
    vis_pz = max(0.0, float(vis_p4[2]))
    vis_pt = float(math.hypot(vis_p4[0], vis_p4[1]))
    dphi = math.atan2(float(lep_p4[1]), float(lep_p4[0])) - math.atan2(float(had_p4[1]), float(had_p4[0]))
    vis_kin = np.array([
        math.log1p(vis_pz),
        math.log1p(vis_pt),
        math.cos(dphi),
        math.sin(dphi),
    ], dtype=np.float32)

    init_p4 = np.array([
        float(ev_tree['initPx'][eid]) * 1e-3,
        float(ev_tree['initPy'][eid]) * 1e-3,
        float(ev_tree['initPz'][eid]) * 1e-3,
        init_e_gev,
    ], dtype=np.float64)
    miss_p4 = init_p4 - vis_p4
    branch_p4 = np.vstack([lep_p4, had_p4, miss_p4, sec_p4]).astype(np.float32)

    return {
        'label': int(label_4),
        'label_7': int(label_7),
        'tau_branch_valid': int(tau_branch_valid),
        'charm_flag': int(charm_flag),
        'charm_valid': 1,
        'vertex_xyz_mm': pv,
        'vertex_valid': 1,
        'vis_kinematics': vis_kin,
        'kin_valid': 1,
        'sample_weight': 1.0,
        'true_sv_xyz_mm': true_sv,
        'has_true_sv': bool(has_true_sv),
        'tier1b_genie_valid': True,
        'true_tau_submode': int(true_tau_submode),
        'true_n_pi0': int(true_n_pi0),
        'true_nu_energy_gev': float(init_e_gev),
        'true_branch_p4_gev': branch_p4,
        'tgt_Z': int(tgt_z),
        'tau_decay_L3d_mm': float(tau_l3d),
    }


def _extract_cluster_sidecar_for_event(
    surv_mask: np.ndarray,
    ph_from_tau: np.ndarray,
    ph_from_charm: np.ndarray,
    ph_from_em: np.ndarray,
    ph_tid: np.ndarray,
    ph_edep: np.ndarray,
    cluster_map: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, Any]:
    """Map native 10085-10088 pixelHits truth tags 1-to-1 to truncated Gen4 clusters."""
    hit_order, offsets, sel_raw = cluster_map
    n_sel = len(sel_raw)
    if n_sel == 0 or len(hit_order) == 0:
        return {
            'cl_sidecar_valid': True,
            'cl_origin_label': np.zeros(0, dtype=np.int64),
            'cl_secondary_origin_label': np.zeros(0, dtype=np.int64),
            'cl_dominant_edep_frac': np.zeros(0, dtype=np.float32),
            'cl_sv_group_id': np.zeros(0, dtype=np.int64),
            'cl_dominant_track_id': np.zeros(0, dtype=np.int64),
            'tau_n_charged_prongs_si': -1,
        }

    ft = np.asarray(ph_from_tau, dtype=np.int8)[surv_mask] != 0
    fc = np.asarray(ph_from_charm, dtype=np.int8)[surv_mask] != 0
    fe = np.asarray(ph_from_em, dtype=np.int8)[surv_mask] != 0
    tid = np.asarray(ph_tid, dtype=np.int64)[surv_mask]
    ed = np.maximum(1e-9, np.asarray(ph_edep, dtype=np.float64)[surv_mask])

    orig = np.full(n_sel, 4, dtype=np.int64)
    sec_orig = np.full(n_sel, 4, dtype=np.int64)
    dom_frac = np.ones(n_sel, dtype=np.float32)
    sv_gid = np.zeros(n_sel, dtype=np.int64)
    dom_tid = np.zeros(n_sel, dtype=np.int64)
    tau_tids: set[int] = set()

    # Label priority: 1=tau, 2=charm, 3=conv_ee/EM, 0=PV/primary, 4=had_or_ghost
    hit_orig = np.where(ft, 1, np.where(fc, 2, np.where(fe, 3, np.where(tid <= 40, 0, 4))))
    for out_i, rk in enumerate(sel_raw):
        s, e = int(offsets[rk]), int(offsets[rk + 1])
        idx_h = hit_order[s:e]
        w_h = ed[idx_h]
        tot_w = float(w_h.sum())
        best_pos = int(np.argmax(w_h))
        o_best = int(hit_orig[idx_h[best_pos]])
        orig[out_i] = o_best
        sec_orig[out_i] = o_best
        best_tid = int(tid[idx_h[best_pos]])
        dom_tid[out_i] = best_tid
        dom_frac[out_i] = float(w_h[tid[idx_h] == best_tid].sum() / max(tot_w, 1e-12))
        if o_best == 1:
            sv_gid[out_i] = 1
            tau_tids.add(best_tid)
        elif o_best == 2:
            sv_gid[out_i] = 2

    return {
        'cl_sidecar_valid': True,
        'cl_origin_label': orig,
        'cl_secondary_origin_label': sec_orig,
        'cl_dominant_edep_frac': dom_frac,
        'cl_sv_group_id': sv_gid,
        'cl_dominant_track_id': dom_tid,
        'tau_n_charged_prongs_si': len(tau_tids) if tau_tids else -1,
    }


def _digitize_pixel_with_surv_mask(
    row_id: np.ndarray,
    col_id: np.ndarray,
    layer_id: np.ndarray,
    edep_mev: np.ndarray,
    contract: Gen4GeometryContract,
    event_seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Digitize raw pixelHits returning both canonical pixel dict and boolean survival mask."""
    r = np.asarray(row_id, dtype=np.int64)
    c = np.asarray(col_id, dtype=np.int64)
    l = np.asarray(layer_id, dtype=np.int64)
    e = np.asarray(edep_mev, dtype=np.float64)
    rng = np.random.default_rng(int(event_seed))
    if len(r) == 0:
        return digitize_raw_pixel_hits(r, c, l, e, contract, event_seed=event_seed), np.zeros(0, dtype=bool)
    valid = (l >= 0) & (l < contract.nz) & (r >= 0) & (r < contract.ny) & (c >= 0) & (c < contract.nx)
    e_noisy = e + rng.normal(0.0, PIXEL_ENC_MEV, size=len(e))
    surv = valid & (e_noisy >= PIXEL_THR_MEV)
    if not surv.any():
        return digitize_raw_pixel_hits(np.empty(0), np.empty(0), np.empty(0), np.empty(0), contract, event_seed=event_seed), surv
    rs, cs, ls, es = r[surv], c[surv], l[surv], e_noisy[surv]
    cid = ((ls * contract.ny + rs) * contract.nx + cs).astype(np.uint64)
    px = np.asarray(contract.raw_geometry['pixel_Xpos'], dtype=np.float64)
    py = np.asarray(contract.raw_geometry['pixel_Ypos'], dtype=np.float64)
    xyz = np.column_stack([px[cs], py[rs], contract.pixel_Zpos[ls]]).astype(np.float32)
    return {'channel_id': cid, 'xyz_mm': xyz, 'edep_mev': es.astype(np.float32)}, surv


def _process_chunk_worker(task: tuple[Any, ...]) -> list[dict[str, Any]]:
    """Multiprocess worker extracting pure-obs (and optional truth) items from one ROOT chunk."""
    ds, chunk_id, root_path_str, selected_indices, mode, internal_split_mode, seed_base = task
    root_path = assert_unsealed_mc8588_path(root_path_str)
    geom = load_geometry_contract()

    entry_stop: int | None = None
    if isinstance(selected_indices, int):
        entry_stop = int(selected_indices)
    elif selected_indices is not None and len(selected_indices) > 0:
        entry_stop = int(max(selected_indices)) + 1

    with uproot.open(str(root_path)) as f:
        ph = f['Hits/pixelHits'].arrays(
            ['rowID', 'colID', 'layerID', 'edep', 'fromTau', 'fromCharmedHadron', 'fromPrimaryEMShower', 'trackID']
            if mode == 'train'
            else ['rowID', 'colID', 'layerID', 'edep'],
            entry_stop=entry_stop,
            library='np',
        )
        sh = f['Hits/scintHits'].arrays(
            ['panelID', 'isHorizontal', 'rowID', 'colID', 'nPE', 'hitTime'],
            entry_stop=entry_stop,
            library='np',
        )
        fh = f['Hits/faserHits'].arrays(['trackerID', 'x', 'y', 'z'], entry_stop=entry_stop, library='np')

        ev_tree = pr_tree = tau_tree = charm_tree = None
        prim_groups = tau_groups = charm_groups = None
        n_ev = len(ph['edep'])
        if mode in ('train', 'tune_with_meta', 'diag_scan'):
            ev_tree = f['event'].arrays(
                ['evtID', 'initE', 'initX', 'initY', 'initZ', 'initPx', 'initPy', 'initPz', 'fslPDG', 'tgtZ'],
                entry_stop=entry_stop,
                library='np',
            )
            ev_ids = np.asarray(ev_tree['evtID'], dtype=np.int64)
            if entry_stop is not None:
                cut_expr = (
                    f'(evtID >= {int(ev_ids.min())}) & (evtID <= {int(ev_ids.max())})'
                    if len(ev_ids) > 0
                    else 'evtID < -1'
                )
            else:
                cut_expr = None
            pr_tree = f['primaries'].arrays(['evtID', 'PDG', 'Px', 'Py', 'Pz', 'E'], cut=cut_expr, library='np')
            tau_tree = f['tau'].arrays(['evtID', 'PDG', 'ProdX', 'ProdY', 'ProdZ', 'Px', 'Py', 'Pz'], cut=cut_expr, library='np')
            charm_tree = f['charm'].arrays(['evtID', 'DecayX', 'DecayY', 'DecayZ'], cut=cut_expr, library='np')
            prim_groups = _build_evt_group_map(pr_tree['evtID'], ev_ids)
            tau_groups = _build_evt_group_map(tau_tree['evtID'], ev_ids)
            charm_groups = _build_evt_group_map(charm_tree['evtID'], ev_ids)

    if isinstance(selected_indices, int):
        idx_iter: Any = range(min(n_ev, int(selected_indices)))
    elif selected_indices is not None:
        idx_iter = [int(i) for i in selected_indices if int(i) < n_ev]
    else:
        idx_iter = range(n_ev)
    n_cc_kept = 0
    n_nw_kept = 0
    n_he_kept = 0
    results: list[dict[str, Any]] = []
    for pos_i, eid in enumerate(idx_iter):
        eid_i = int(eid)
        ev_seed = int(seed_base) + chunk_id * 2000 + eid_i

        # For diag_scan, prefilter OOD condition cleanly before heavy frontend work
        if mode == 'diag_scan':
            assert ev_tree is not None and charm_groups is not None
            is_nc_charm = (ds == '10088') and (len(charm_groups[eid_i]) > 0)
            is_cc_charm = (ds in ('10085', '10086', '10087')) and (len(charm_groups[eid_i]) > 0) and (n_cc_kept < 15)
            is_nonw = (int(ev_tree['tgtZ'][eid_i]) != 74) and (n_nw_kept < 15)
            is_highe = (float(ev_tree['initE'][eid_i]) * 1e-3 > 1500.0) and (n_he_kept < 15)
            if not (is_nc_charm or is_cc_charm or is_nonw or is_highe):
                continue
            if is_cc_charm:
                n_cc_kept += 1
            if is_nonw:
                n_nw_kept += 1
            if is_highe:
                n_he_kept += 1

        if mode == 'train':
            pix_dict, surv_mask = _digitize_pixel_with_surv_mask(
                ph['rowID'][eid_i], ph['colID'][eid_i], ph['layerID'][eid_i], ph['edep'][eid_i],
                geom, ev_seed,
            )
            obs = {
                'pixel': pix_dict,
                'bar': digitize_raw_scint_hits(
                    sh['panelID'][eid_i], sh['isHorizontal'][eid_i], sh['rowID'][eid_i],
                    sh['colID'][eid_i], sh['nPE'][eid_i], sh['hitTime'][eid_i], geom,
                ),
                'tracker_hypothesis': digitize_raw_faser_hits(
                    fh['trackerID'][eid_i], fh['x'][eid_i], fh['y'][eid_i], fh['z'][eid_i],
                    event_seed=ev_seed,
                ),
            }
        else:
            obs = build_observation_from_raw_root_event(
                ph['rowID'][eid_i], ph['colID'][eid_i], ph['layerID'][eid_i], ph['edep'][eid_i],
                sh['panelID'][eid_i], sh['isHorizontal'][eid_i], sh['rowID'][eid_i],
                sh['colID'][eid_i], sh['nPE'][eid_i], sh['hitTime'][eid_i],
                fh['trackerID'][eid_i], fh['x'][eid_i], fh['y'][eid_i], fh['z'][eid_i],
                geom, event_seed=ev_seed,
            )

        # Skip full-detector zero-response empty events (~0.5%)
        if len(obs['pixel']['channel_id']) == 0 and len(obs['bar']['panel']) == 0 and len(obs['tracker_hypothesis']['station']) == 0:
            continue

        fe = extract_gen4_frontend_event(
            obs, geom, has_subthr_readout=True, use_edep=True, return_cluster_map=(mode == 'train')
        )
        if mode == 'ssl' and int(fe['obs_domain_id']) not in (1, 2):
            continue

        item: dict[str, Any] = {
            'run': int(ds),
            'chunk': int(chunk_id),
            'evt_id': int(eid_i),
            'c_obs': fe['c_obs'],
            'obs_domain_id': int(fe['obs_domain_id']),
            'cal_start_mm': fe['cal_start_mm'],
            'has_subthr_readout': float(fe['has_subthr_readout']),
            'has_edep': float(fe['has_edep']),
            's_mir': float(fe['s_mir']),
            'pix_xyz_mm': fe['pix_xyz_mm'].astype(np.float32),
            'pix_plane': fe['pix_plane'].astype(np.int16),
            'pix_feat': fe['pix_feat'].astype(np.float32),
            'pix_plane_summary': fe['pix_plane_summary'].astype(np.float32),
            'pix_patches_u8': fe['pix_patches_u8'],
            'patch_cluster_idx': fe['patch_cluster_idx'].astype(np.int16),
            'bar_station': fe['bar_station'],
            'bar_view': fe['bar_view'],
            'bar_idx': fe['bar_idx'],
            'bar_feat_10': fe['bar_feat_10'].astype(np.float32),
            'muon_kalman_tracks': fe['muon_kalman_tracks'],
            'muon_kalman_mask': fe['muon_kalman_mask'],
            'trk_cand': fe['trk_cand'],
            'trk_cand_mask': fe['trk_cand_mask'],
            'trk_station_counts': fe['trk_station_counts'],
        }

        if mode == 'train':
            assert ev_tree is not None and prim_groups is not None and tau_groups is not None and charm_groups is not None
            cl_map = fe.pop('_cluster_map')
            tr = _extract_event_truth_record(
                ds, eid_i, ev_tree,
                prim_groups[eid_i], pr_tree,
                tau_groups[eid_i], tau_tree,
                charm_groups[eid_i], charm_tree,
            )
            cl_tr = _extract_cluster_sidecar_for_event(
                surv_mask,
                ph['fromTau'][eid_i], ph['fromCharmedHadron'][eid_i],
                ph['fromPrimaryEMShower'][eid_i], ph['trackID'][eid_i], ph['edep'][eid_i],
                cl_map,
            )
            item.update(tr)
            item.update(cl_tr)
            r = chunk_id % 20
            if internal_split_mode == 'r9_split':
                item['internal_split'] = 'internal_train' if (pos_i % 2 == 0) else 'internal_dev_screen'
            elif r in (10, 11):
                item['internal_split'] = 'internal_val_eval'
            else:
                item['internal_split'] = 'internal_train'
        elif mode in ('tune_with_meta', 'diag_scan'):
            assert ev_tree is not None and prim_groups is not None and tau_groups is not None and charm_groups is not None
            tr = _extract_event_truth_record(
                ds, eid_i, ev_tree,
                prim_groups[eid_i], pr_tree,
                tau_groups[eid_i], tau_tree,
                charm_groups[eid_i], charm_tree,
            )
            # Attach temporary `_meta` removed BEFORE writing pure-obs cache!
            tr['run'] = int(ds)
            tr['chunk'] = int(chunk_id)
            tr['evt_id'] = int(eid_i)
            tr['obs_domain_id'] = int(fe['obs_domain_id'])
            item['_meta'] = tr
            if mode == 'diag_scan':
                item['_diag_flags'] = (
                    bool((ds == '10088') and (len(charm_groups[eid_i]) > 0)),
                    bool((ds in ('10085', '10086', '10087')) and (len(charm_groups[eid_i]) > 0)),
                    bool(int(ev_tree['tgtZ'][eid_i]) != 74),
                    bool(float(ev_tree['initE'][eid_i]) * 1e-3 > 1500.0),
                )
        results.append(item)
    return results


def _process_stress_chunk_worker(task: tuple[Any, ...]) -> dict[str, list[dict[str, Any]]]:
    """Apply all 7 STAGE2_STRESS_SPECS on selected events from one r==13 ROOT chunk (0 truth)."""
    ds, chunk_id, root_path_str, selected_indices, seed_base = task
    root_path = assert_unsealed_mc8588_path(root_path_str)
    geom = load_geometry_contract()
    entry_stop = int(max(selected_indices)) + 1 if len(selected_indices) > 0 else 0
    with uproot.open(str(root_path)) as f:
        ph = f['Hits/pixelHits'].arrays(['rowID', 'colID', 'layerID', 'edep'], entry_stop=entry_stop, library='np')
        sh = f['Hits/scintHits'].arrays(['panelID', 'isHorizontal', 'rowID', 'colID', 'nPE', 'hitTime'], entry_stop=entry_stop, library='np')
        fh = f['Hits/faserHits'].arrays(['trackerID', 'x', 'y', 'z'], entry_stop=entry_stop, library='np')

    out: dict[str, list[dict[str, Any]]] = {st_name: [] for st_name, _ in STAGE2_STRESS_SPECS}
    for eid in selected_indices:
        eid_i = int(eid)
        ev_seed = int(seed_base) + chunk_id * 2000 + eid_i
        for st_name, kw in STAGE2_STRESS_SPECS:
            rng_st = np.random.default_rng(ev_seed + 41)
            r_p, c_p, l_p, e_p = ph['rowID'][eid_i], ph['colID'][eid_i], ph['layerID'][eid_i], ph['edep'][eid_i]
            if 'pixel_dead' in kw and len(r_p) > 0:
                keep_m = rng_st.random(len(r_p)) >= float(kw['pixel_dead'])
                r_p, c_p, l_p, e_p = r_p[keep_m], c_p[keep_m], l_p[keep_m], e_p[keep_m]
            npe_arr = sh['nPE'][eid_i].astype(np.float64)
            if 'sipm_sat_cap' in kw and len(npe_arr) > 0:
                npe_arr = npe_arr * float(kw['sipm_sat_cap'])
            obs = build_observation_from_raw_root_event(
                r_p, c_p, l_p, e_p,
                sh['panelID'][eid_i], sh['isHorizontal'][eid_i], sh['rowID'][eid_i],
                sh['colID'][eid_i], npe_arr, sh['hitTime'][eid_i],
                fh['trackerID'][eid_i], fh['x'][eid_i], fh['y'][eid_i], fh['z'][eid_i],
                geom, event_seed=ev_seed,
                dark_rate=float(kw.get('pixel_dark_rate', 0.0)),
                tracker_dead_prob=float(kw.get('tracker_dead', 0.0)),
                tracker_noise=int(kw.get('tracker_noise', 0)),
            )
            fe = extract_gen4_frontend_event(obs, geom, has_subthr_readout=True, use_edep=True)
            out[st_name].append({
                'run': int(ds),
                'chunk': int(chunk_id),
                'evt_id': int(eid_i),
                'c_obs': fe['c_obs'],
                'obs_domain_id': int(fe['obs_domain_id']),
                'cal_start_mm': fe['cal_start_mm'],
                'has_subthr_readout': float(fe['has_subthr_readout']),
                'has_edep': float(fe['has_edep']),
                's_mir': float(fe['s_mir']),
                'pix_xyz_mm': fe['pix_xyz_mm'].astype(np.float32),
                'pix_plane': fe['pix_plane'].astype(np.int16),
                'pix_feat': fe['pix_feat'].astype(np.float32),
                'pix_plane_summary': fe['pix_plane_summary'].astype(np.float32),
                'pix_patches_u8': fe['pix_patches_u8'],
                'patch_cluster_idx': fe['patch_cluster_idx'].astype(np.int16),
                'bar_station': fe['bar_station'],
                'bar_view': fe['bar_view'],
                'bar_idx': fe['bar_idx'],
                'bar_feat_10': fe['bar_feat_10'].astype(np.float32),
                'muon_kalman_tracks': fe['muon_kalman_tracks'],
                'muon_kalman_mask': fe['muon_kalman_mask'],
                'trk_cand': fe['trk_cand'],
                'trk_cand_mask': fe['trk_cand_mask'],
                'trk_station_counts': fe['trk_station_counts'],
            })
    return out


def _stratified_chunk_quotas(chunks: list[tuple[int, Path]], total_target: int, seed: int) -> list[tuple[int, Path, int]]:
    """Deterministically allocate stratified per-chunk quotas with evenly strided chunk sampling."""
    if not chunks or total_target <= 0:
        return []
    # Cap opened chunks per bucket to ~240 evenly across sorted chunk IDs (~20-40 events/chunk)
    # so large datasets like 10086 don't open 1,761 files to take 4 events each.
    max_chunks = max(48, min(len(chunks), max(total_target // 16, 120)))
    if len(chunks) > max_chunks:
        pick_idx = np.linspace(0, len(chunks) - 1, max_chunks, dtype=int)
        chunks_use = [chunks[int(i)] for i in pick_idx]
    else:
        chunks_use = chunks
    n_ch = len(chunks_use)
    base_q = total_target // n_ch
    rem = total_target % n_ch
    alloc: list[tuple[int, Path, int]] = []
    for idx, (cid, p) in enumerate(chunks_use):
        want = base_q + (1 if idx < rem else 0) + 3  # +3 margin for ~0.5% empty event skip
        alloc.append((cid, p, want))
    return alloc


def build_train32k_cache(
    cache_dir: Path = GEN4_CACHE_DIR,
    workers: int = 24,
    n_per_flavour: int = 8000,
    seed: int = 20260001,
) -> dict[str, Any]:
    """Build `train32k_part_{0..7}.pkl` (8,000/flavour = 32,000 from r in {0..11})."""
    assert_unsealed_split('internal_train', stage=1)
    cache_dir.mkdir(parents=True, exist_ok=True)
    splits = partition_mc8588_chunks()['train']

    # Split per-flavour quotas into: 6000 (r in 0..8 -> train), 800 (r==9 -> 400 train + 400 dev), 1200 (r in 10..11 -> val)
    q_08 = int(round(n_per_flavour * 0.75))
    q_r9 = int(round(n_per_flavour * 0.10))
    q_1011 = n_per_flavour - q_08 - q_r9

    tasks: list[tuple[Any, ...]] = []
    for ds_idx, ds in enumerate(MC8588_DATASETS):
        ch_08 = [c for c in splits[ds] if (c[0] % 20) <= 8]
        ch_r9 = [c for c in splits[ds] if (c[0] % 20) == 9]
        ch_1011 = [c for c in splits[ds] if (c[0] % 20) in (10, 11)]
        for tag, ch_sub, q_sub, sp_mode in (
            ('08', ch_08, q_08, 'train'),
            ('r9', ch_r9, q_r9, 'r9_split'),
            ('1011', ch_1011, q_1011, 'val'),
        ):
            for cid, p, ev_idx in _stratified_chunk_quotas(ch_sub, q_sub, seed + ds_idx * 100 + len(tag)):
                tasks.append((ds, cid, str(p), ev_idx, 'train', sp_mode, 10_000_000 * (ds_idx + 1)))

    all_items_by_ds: dict[int, list[dict[str, Any]]] = {0: [], 1: [], 2: [], 3: []}
    with ProcessPoolExecutor(max_workers=min(workers, max(1, len(tasks))), mp_context=mp.get_context('spawn')) as pool:
        for res in pool.map(_process_chunk_worker, tasks):
            for it in res:
                all_items_by_ds[int(it['label'])].append(it)

    final_items: list[dict[str, Any]] = []
    counts_by_split: dict[str, int] = {'internal_train': 0, 'internal_dev_screen': 0, 'internal_val_eval': 0}
    for lbl in (0, 1, 2, 3):
        ds_items = sorted(all_items_by_ds[lbl], key=lambda x: (x['run'], x['chunk'], x['evt_id']))
        # Trim strictly to exact target per internal_split
        tr_sub = [x for x in ds_items if x['internal_split'] == 'internal_train'][: int(round(n_per_flavour * 0.80))]
        dv_sub = [x for x in ds_items if x['internal_split'] == 'internal_dev_screen'][: int(round(n_per_flavour * 0.05))]
        va_sub = [x for x in ds_items if x['internal_split'] == 'internal_val_eval'][: int(round(n_per_flavour * 0.15))]
        picked = tr_sub + dv_sub + va_sub
        for x in picked:
            counts_by_split[x['internal_split']] += 1
        final_items.extend(picked)

    final_items.sort(key=lambda x: (x['run'], x['chunk'], x['evt_id']))
    n_parts = 8
    part_size = (len(final_items) + n_parts - 1) // n_parts
    paths: list[str] = []
    for p_i in range(n_parts):
        sub = final_items[p_i * part_size : (p_i + 1) * part_size]
        out_p = cache_dir / f'train32k_part_{p_i}.pkl'
        tmp_p = cache_dir / f'train32k_part_{p_i}.pkl.tmp'
        with tmp_p.open('wb') as fh:
            pickle.dump(sub, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_p.replace(out_p)
        paths.append(str(out_p))

    return {
        'total': len(final_items),
        'splits': counts_by_split,
        'paths': paths,
    }


def build_tune8k_and_stress_diag_cache(
    cache_dir: Path = GEN4_CACHE_DIR,
    val_meta_dir: Path = GEN4_VAL_META_DIR,
    workers: int = 24,
    n_screen_per_flavour: int = 500,
    n_eval_per_flavour: int = 1500,
    n_stress_per_flavour: int = 100,
    seed: int = 20260002,
) -> dict[str, Any]:
    """Build `tune8k_part_{0..1}.pkl` (0 truth), `tune_truth_meta.parquet`, `stress_diag_pack.pkl`."""
    assert_unsealed_split('tune_screen', stage=2)
    assert_unsealed_split('tune6k', stage=2)
    cache_dir.mkdir(parents=True, exist_ok=True)
    val_meta_dir.mkdir(parents=True, exist_ok=True)
    all_splits = partition_mc8588_chunks()

    # 1. Tune_Screen (r == 12, 500/flavour = 2000) & Tune6k (r == 13, 1500/flavour = 6000)
    tune_tasks: list[tuple[Any, ...]] = []
    used_tune6k_keys: set[tuple[int, int, int]] = set()
    for ds_idx, ds in enumerate(MC8588_DATASETS):
        for cid, p, ev_idx in _stratified_chunk_quotas(all_splits['tune_screen'][ds], n_screen_per_flavour, seed + ds_idx):
            tune_tasks.append((ds, cid, str(p), ev_idx, 'tune_with_meta', 'screen', 20_000_000 * (ds_idx + 1)))
        for cid, p, ev_idx in _stratified_chunk_quotas(all_splits['tune6k'][ds], n_eval_per_flavour, seed + 50 + ds_idx):
            tune_tasks.append((ds, cid, str(p), ev_idx, 'tune_with_meta', 'eval', 30_000_000 * (ds_idx + 1)))

    raw_tune: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(workers, max(1, len(tune_tasks))), mp_context=mp.get_context('spawn')) as pool:
        for res in pool.map(_process_chunk_worker, tune_tasks):
            raw_tune.extend(res)

    screen_items: list[dict[str, Any]] = []
    eval_items: list[dict[str, Any]] = []
    for ds_int in (10085, 10086, 10087, 10088):
        scr = sorted([x for x in raw_tune if x['run'] == ds_int and (x['chunk'] % 20) == 12], key=lambda x: (x['chunk'], x['evt_id']))[:n_screen_per_flavour]
        evl = sorted([x for x in raw_tune if x['run'] == ds_int and (x['chunk'] % 20) == 13], key=lambda x: (x['chunk'], x['evt_id']))[:n_eval_per_flavour]
        for x in scr:
            x['_meta']['in_screen'] = True
            x['in_screen'] = True
        for x in evl:
            x['_meta']['in_screen'] = False
            x['in_screen'] = False
            used_tune6k_keys.add((x['run'], x['chunk'], x['evt_id']))
        screen_items.extend(scr)
        eval_items.extend(evl)

    tune8k_all = screen_items + eval_items
    meta_rows = [x.pop('_meta') for x in tune8k_all]
    # Enforce ZERO truth fields in tune8k_part_{0..1}.pkl!
    for x in tune8k_all:
        for k in list(x.keys()):
            if k in GEN4_TRUTH_BATCH_KEYS:
                raise SealViolationError(f'Truth key {k} leaked into pure-obs Tune8k item!')

    for p_i, sl in enumerate((slice(0, len(tune8k_all) // 2), slice(len(tune8k_all) // 2, len(tune8k_all)))):
        out_p = cache_dir / f'tune8k_part_{p_i}.pkl'
        tmp_p = cache_dir / f'tune8k_part_{p_i}.pkl.tmp'
        with tmp_p.open('wb') as fh:
            pickle.dump(tune8k_all[sl], fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_p.replace(out_p)

    meta_df = pd.DataFrame([{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in m.items()} for m in meta_rows])
    pq.write_table(pa.Table.from_pandas(meta_df), val_meta_dir / 'tune_truth_meta.parquet')

    # 2. Stress400 (100/flavour from r==13, 7 STAGE2_STRESS_SPECS)
    stress_tasks: list[tuple[Any, ...]] = []
    for ds_idx, ds in enumerate(MC8588_DATASETS):
        ds_eval = [x for x in eval_items if x['run'] == int(ds)][:n_stress_per_flavour]
        by_chunk: dict[int, list[int]] = {}
        for x in ds_eval:
            by_chunk.setdefault(x['chunk'], []).append(x['evt_id'])
        ch_map = {cid: p for cid, p in all_splits['tune6k'][ds]}
        for cid, evs in sorted(by_chunk.items()):
            stress_tasks.append((ds, cid, str(ch_map[cid]), evs, 40_000_000 * (ds_idx + 1)))

    stress_items: dict[str, list[dict[str, Any]]] = {st_name: [] for st_name, _ in STAGE2_STRESS_SPECS}
    with ProcessPoolExecutor(max_workers=min(workers, max(1, len(stress_tasks))), mp_context=mp.get_context('spawn')) as pool:
        for res_d in pool.map(_process_stress_chunk_worker, stress_tasks):
            for st_name, lst in res_d.items():
                stress_items[st_name].extend(lst)
    for st_name in stress_items:
        stress_items[st_name].sort(key=lambda x: (x['run'], x['chunk'], x['evt_id']))

    stress_y = np.array([MC8588_DATASETS[str(x['run'])][1] for x in stress_items['nominal']], dtype=np.int64)

    # 3. Diag_OOD from remaining r==13 events disjoint from Tune6k
    diag_tasks: list[tuple[Any, ...]] = []
    for ds_idx, ds in enumerate(MC8588_DATASETS):
        for cid, p in all_splits['tune6k'][ds]:
            diag_tasks.append((ds, cid, str(p), None, 'diag_scan', 'diag', 50_000_000 * (ds_idx + 1)))

    diag_packs: dict[str, list[dict[str, Any]]] = {
        'InDet_NC_Charm': [],
        'InDet_CC_Charm': [],
        'NonW_Al_Scint': [],
        'HighE_gt1p5TeV': [],
    }
    with ProcessPoolExecutor(max_workers=min(workers, max(1, len(diag_tasks))), mp_context=mp.get_context('spawn')) as pool:
        for res in pool.map(_process_chunk_worker, diag_tasks):
            for it in res:
                if (it['run'], it['chunk'], it['evt_id']) in used_tune6k_keys:
                    continue
                is_nc_c, is_cc_c, is_nw, is_he = it.pop('_diag_flags')
                if is_nc_c:
                    diag_packs['InDet_NC_Charm'].append(dict(it))
                if is_cc_c and len(diag_packs['InDet_CC_Charm']) < 3000:
                    diag_packs['InDet_CC_Charm'].append(dict(it))
                if is_nw and len(diag_packs['NonW_Al_Scint']) < 3000:
                    diag_packs['NonW_Al_Scint'].append(dict(it))
                if is_he and len(diag_packs['HighE_gt1p5TeV']) < 3000:
                    diag_packs['HighE_gt1p5TeV'].append(dict(it))

    diag_meta: dict[str, pd.DataFrame] = {}
    for k_d, lst in diag_packs.items():
        lst.sort(key=lambda x: (x['run'], x['chunk'], x['evt_id']))
        m_list = [x.pop('_meta') for x in lst]
        diag_meta[k_d] = pd.DataFrame([{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in m.items()} for m in m_list])

    pack_path = cache_dir / 'stress_diag_pack.pkl'
    tmp_pack = cache_dir / 'stress_diag_pack.pkl.tmp'
    with tmp_pack.open('wb') as fh:
        pickle.dump({'stress_items': stress_items, 'diag_packs': diag_packs}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_pack.replace(pack_path)

    diag_meta_out = val_meta_dir / 'diag_truth_meta.pkl'
    with diag_meta_out.open('wb') as fh:
        pickle.dump({'stress_y': stress_y, 'diag_meta': diag_meta}, fh, protocol=pickle.HIGHEST_PROTOCOL)

    return {
        'tune_screen': len(screen_items),
        'tune6k': len(eval_items),
        'stress_per_spec': {k: len(v) for k, v in stress_items.items()},
        'diag_counts': {k: len(v) for k, v in diag_packs.items()},
        'pack_path': str(pack_path),
    }


def build_ssl_mc8588_part(
    part_idx: int,
    num_parts: int = 32,
    cache_dir: Path = GEN4_CACHE_DIR,
    workers: int = 8,
    seed: int = 20260003,
) -> dict[str, Any]:
    """HTCondor-parallel shard builder for `SSL_320k` (part 0..num_parts-1) & `SSL_Val12k` (part == -2)."""
    assert_unsealed_split('train', stage=1)
    ssl_dir = cache_dir / 'ssl_mc8588'
    ssl_dir.mkdir(parents=True, exist_ok=True)
    splits = partition_mc8588_chunks()

    if part_idx == -2:
        tasks: list[tuple[Any, ...]] = []
        for ds_idx, ds in enumerate(MC8588_DATASETS):
            for cid, p in splits['ssl_val'][ds]:
                tasks.append((ds, cid, str(p), None, 'ssl', 'ssl', 60_000_000 * (ds_idx + 1) + seed))
        out_name = 'ssl_val_calib.pkl'
    else:
        if not (0 <= part_idx < num_parts):
            raise ValueError(f'part_idx must be in 0..{num_parts - 1} or -2 for ssl_val, got {part_idx}')
        # Interleave datasets so every shard has identical natural flux ratio & energy distribution
        per_ds = [
            [(ds, cid, str(p), None, 'ssl', 'ssl', 60_000_000 * (ds_idx + 1) + seed) for cid, p in splits['ssl_train'][ds]]
            for ds_idx, ds in enumerate(MC8588_DATASETS)
        ]
        all_tasks: list[tuple[Any, ...]] = []
        max_len = max(len(x) for x in per_ds)
        for k in range(max_len):
            for lst in per_ds:
                if k < len(lst):
                    all_tasks.append(lst[k])
        tasks = all_tasks[part_idx::num_parts]
        out_name = f'ssl_train_part_{part_idx:02d}.pkl'

    out_items: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(workers, max(1, len(tasks))), mp_context=mp.get_context('spawn')) as pool:
        for res in pool.map(_process_chunk_worker, tasks):
            out_items.extend(res)

    out_p = ssl_dir / out_name
    tmp_p = ssl_dir / f'{out_name}.tmp'
    with tmp_p.open('wb') as fh:
        pickle.dump(out_items, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_p.replace(out_p)
    return {'part': int(part_idx), 'count': len(out_items), 'path': str(out_p)}


def build_ssl_mc8588_cache(
    cache_dir: Path = GEN4_CACHE_DIR,
    workers: int = 24,
    max_ssl_train: int = 320_000,
    max_ssl_val: int = 16_000,
    seed: int = 20260003,
) -> dict[str, Any]:
    """Build strictly truth-free `SSL_320k` + `SSL_Val12k` from r in {14, 15} (D1 U D2)."""
    assert_unsealed_split('train', stage=1)
    ssl_dir = cache_dir / 'ssl_mc8588'
    ssl_dir.mkdir(parents=True, exist_ok=True)
    splits = partition_mc8588_chunks()

    def _run_split(split_key: str, cap: int) -> list[dict[str, Any]]:
        tasks: list[tuple[Any, ...]] = []
        for ds_idx, ds in enumerate(MC8588_DATASETS):
            for cid, p in splits[split_key][ds]:
                tasks.append((ds, cid, str(p), None, 'ssl', 'ssl', 60_000_000 * (ds_idx + 1) + seed))
        out_items: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=min(workers, max(1, len(tasks))), mp_context=mp.get_context('spawn')) as pool:
            for res in pool.map(_process_chunk_worker, tasks):
                out_items.extend(res)
                if len(out_items) >= cap:
                    break
        return out_items[:cap]

    tr_items = _run_split('ssl_train', max_ssl_train)
    va_items = _run_split('ssl_val', max_ssl_val)

    n_shards = max(1, min(16, (len(tr_items) + 19999) // 20000))
    sh_sz = (len(tr_items) + n_shards - 1) // n_shards
    tr_paths: list[str] = []
    for s_i in range(n_shards):
        p_out = ssl_dir / f'ssl_train_part_{s_i:02d}.pkl'
        tmp_p = ssl_dir / f'ssl_train_part_{s_i:02d}.pkl.tmp'
        with tmp_p.open('wb') as fh:
            pickle.dump(tr_items[s_i * sh_sz : (s_i + 1) * sh_sz], fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_p.replace(p_out)
        tr_paths.append(str(p_out))

    val_p = ssl_dir / 'ssl_val_calib.pkl'
    with val_p.open('wb') as fh:
        pickle.dump(va_items, fh, protocol=pickle.HIGHEST_PROTOCOL)

    return {'ssl_train': len(tr_items), 'ssl_val': len(va_items), 'ssl_dir': str(ssl_dir)}


# =============================================================================
# High-Speed Batch Collation & Symmetry / Ablation Augmentation
# =============================================================================

_CPY_FLIP_PIX_FEAT = (1, 11, 17, 29, 34)
_CPY_FLIP_PS_FEAT = (4, 6)
_CPY_FLIP_MUON_FEAT = (0, 4, 13)


def _apply_item_symmetry_in_batch(
    item: dict[str, Any],
    geom: Gen4GeometryContract,
    do_cp_y: bool,
    do_px_si: bool,
) -> dict[str, Any]:
    """Clone item and apply exact Gen4 feature/truth-consistent P_x^{Si} and/or CP_y flips."""
    it = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in item.items()}
    pxyz = it['pix_xyz_mm']
    pfeat = it['pix_feat']
    ps = it['pix_plane_summary']
    patches = it['pix_patches_u8']
    bst = it['bar_station']
    bvw = it['bar_view']
    bix = it['bar_idx']
    bf10 = it['bar_feat_10']
    mtrk = it['muon_kalman_tracks']
    mmask = it['muon_kalman_mask']

    if do_px_si and len(pxyz) > 0:
        pxyz[:, 0] = 2.0 * SI_SUB_X_AXIS_MM - pxyz[:, 0]
        pfeat[:, 0] = (pxyz[:, 0] - geom.x_pix_center) / 135.0
        pfeat[:, (11, 29, 33)] *= -1.0
        ps[:, 3] = ((2.0 * SI_SUB_X_AXIS_MM - (ps[:, 3] * 135.0 + geom.x_pix_center)) - geom.x_pix_center) / 135.0
        ps[:, 7] = ((2.0 * SI_SUB_X_AXIS_MM - (ps[:, 7] * 135.0 + geom.x_pix_center)) - geom.x_pix_center) / 135.0
        if len(patches) > 0:
            patches = patches[..., ::-1].copy()
        if len(bix) > 0:
            xm = bvw == 0
            bix[xm] = 24 - bix[xm]
            bf10[bvw == 1, 8] *= -1.0
        if mmask.any():
            mtrk[mmask, 10] = 2.0 * SI_SUB_X_AXIS_MM - mtrk[mmask, 10]
            mtrk[mmask, 12] *= -1.0
        if 'vertex_xyz_mm' in it:
            it['vertex_xyz_mm'][0] = 2.0 * SI_SUB_X_AXIS_MM - it['vertex_xyz_mm'][0]
        if 'true_sv_xyz_mm' in it and it.get('has_true_sv', False):
            it['true_sv_xyz_mm'][0] = 2.0 * SI_SUB_X_AXIS_MM - it['true_sv_xyz_mm'][0]

    if do_cp_y:
        it['s_mir'] = -float(it.get('s_mir', 1.0))
        if len(pxyz) > 0:
            pxyz[:, 1] = 2.0 * COMMON_Y_AXIS_MM - pxyz[:, 1]
            pfeat[:, _CPY_FLIP_PIX_FEAT] *= -1.0
            ps[:, _CPY_FLIP_PS_FEAT] *= -1.0
            if len(patches) > 0:
                patches = patches[..., ::-1, :].copy()
        if len(bix) > 0:
            ym = bvw == 1
            bix[ym] = (geom.n_bars - 1) - bix[ym]
            bf10[ym, 8] *= -1.0
        if mmask.any():
            mtrk[mmask, 11] = 2.0 * COMMON_Y_AXIS_MM - mtrk[mmask, 11]
            for j_m in _CPY_FLIP_MUON_FEAT:
                mtrk[mmask, j_m] *= -1.0
        if 'label_7' in it:
            l7 = int(it['label_7'])
            if l7 == 1:
                it['label_7'] = 2
            elif l7 == 2:
                it['label_7'] = 1
        if 'vertex_xyz_mm' in it:
            it['vertex_xyz_mm'][1] = 2.0 * COMMON_Y_AXIS_MM - it['vertex_xyz_mm'][1]
        if 'true_sv_xyz_mm' in it and it.get('has_true_sv', False):
            it['true_sv_xyz_mm'][1] = 2.0 * COMMON_Y_AXIS_MM - it['true_sv_xyz_mm'][1]

    it['pix_xyz_mm'] = pxyz
    it['pix_feat'] = pfeat
    it['pix_plane_summary'] = ps
    it['pix_patches_u8'] = patches
    it['bar_idx'] = bix
    it['bar_feat_10'] = bf10
    it['muon_kalman_tracks'] = mtrk
    return it


def collate_gen4_batch(
    items: list[dict[str, Any]],
    geom: Gen4GeometryContract,
    aug_cfg: dict[str, Any] | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Collate cached Gen4 event items into dynamic-padded `Gen4EventBatch` tensors."""
    bsz = len(items)
    if bsz == 0:
        raise ValueError('collate_gen4_batch received empty items list')
    if rng is None:
        rng = np.random.default_rng()
    cfg = aug_cfg or {}
    p_cp_y = float(cfg.get('p_cp_y', 0.0))
    p_px_si = float(cfg.get('p_px_si', 0.0))
    p_subthr_drop = float(cfg.get('p_subthr_drop', 0.0))
    drop_edep = bool(cfg.get('drop_edep', False))

    proc: list[dict[str, Any]] = []
    for it in items:
        do_cpy = (p_cp_y > 0.0) and (rng.random() < p_cp_y)
        do_px = False
        if p_px_si > 0.0 and rng.random() < p_px_si:
            px_x = it['pix_xyz_mm'][:, 0] if len(it['pix_xyz_mm']) else np.empty(0)
            bx_x = it['bar_idx'][it['bar_view'] == 0] if len(it['bar_idx']) else np.empty(0)
            if (len(px_x) == 0 or np.all((px_x >= -126.0) & (px_x <= 122.5))) and (
                len(bx_x) == 0 or np.all((bx_x >= 2) & (bx_x <= 22))
            ):
                do_px = True
        proc.append(_apply_item_symmetry_in_batch(it, geom, do_cpy, do_px) if (do_cpy or do_px) else it)

    max_cl_raw = max((len(it['pix_plane']) for it in proc), default=0)
    n_cl_pad = max(64, min(N_MAX_CLUSTERS_GEN4, ((max_cl_raw + 63) // 64) * 64))

    c_obs = np.zeros((bsz, 16), dtype=np.float32)
    obs_dom = np.zeros(bsz, dtype=np.int64)
    cal_start = np.zeros((bsz, 3), dtype=np.float32)
    has_subthr = np.zeros(bsz, dtype=np.float32)
    has_edep = np.zeros(bsz, dtype=np.float32)
    s_mir = np.ones(bsz, dtype=np.float32)

    pix_xyz = np.zeros((bsz, n_cl_pad, 3), dtype=np.float32)
    pix_plane = np.full((bsz, n_cl_pad), -1, dtype=np.int64)
    pix_valid = np.zeros((bsz, n_cl_pad), dtype=bool)
    pix_feat = np.zeros((bsz, n_cl_pad, PIX_FEAT_GEN4_DIM), dtype=np.float32)
    pix_ps = np.zeros((bsz, geom.nz, PIX_PLANE_SUMMARY_DIM), dtype=np.float32)
    pix_patches = np.zeros((bsz, N_MAX_PATCHES_GEN4, 2, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    patch_cl_idx = np.full((bsz, N_MAX_PATCHES_GEN4), -1, dtype=np.int64)

    scint_grid = np.zeros((bsz, geom.n_stations, 2, geom.n_bars, SCINT_GRID_CHANNELS), dtype=np.float32)
    mu_tracks = np.zeros((bsz, 4, 16), dtype=np.float32)
    mu_mask = np.zeros((bsz, 4), dtype=bool)
    trk_cand = np.zeros((bsz, 8, 16), dtype=np.float32)
    trk_mask = np.zeros((bsz, 8), dtype=bool)
    trk_counts = np.zeros((bsz, 3), dtype=np.float32)

    for b, it in enumerate(proc):
        c_obs[b] = it['c_obs']
        obs_dom[b] = int(it['obs_domain_id'])
        cal_start[b] = it['cal_start_mm']
        s_mir[b] = float(it.get('s_mir', 1.0))

        n_c = min(n_cl_pad, len(it['pix_plane']))
        if n_c > 0:
            pix_xyz[b, :n_c] = it['pix_xyz_mm'][:n_c]
            pix_plane[b, :n_c] = it['pix_plane'][:n_c]
            pix_valid[b, :n_c] = True
            pf = it['pix_feat'][:n_c]
            if pf.shape[1] < PIX_FEAT_GEN4_DIM:
                pix_feat[b, :n_c, : pf.shape[1]] = pf
            else:
                pix_feat[b, :n_c] = pf[:, :PIX_FEAT_GEN4_DIM]
        pix_ps[b] = it['pix_plane_summary']

        n_p = min(N_MAX_PATCHES_GEN4, len(it['patch_cluster_idx']))
        if n_p > 0:
            pix_patches[b, :n_p] = it['pix_patches_u8'][:n_p].astype(np.float32) / 255.0
            pidx = it['patch_cluster_idx'][:n_p].astype(np.int64)
            patch_cl_idx[b, :n_p] = np.where(pidx < n_c, pidx, -1)

        if drop_edep:
            has_edep[b] = 0.0
            pix_feat[b, :n_c, 30:36] = 0.0
            pix_patches[b, :, 1] = 0.0
        else:
            has_edep[b] = float(it.get('has_edep', 1.0))

        # Scatter sparse bars -> dense (54, 2, 40, 10) scint_grid
        bst = it['bar_station']
        bvw = it['bar_view']
        bix = it['bar_idx']
        bf10 = it['bar_feat_10']
        mask_subthr = (p_subthr_drop > 0.0) and (rng.random() < p_subthr_drop)
        if mask_subthr:
            has_subthr[b] = 0.0
            if len(bf10) > 0:
                keep_b = bf10[:, 2] >= 2.5
                bst, bvw, bix, bf10 = bst[keep_b], bvw[keep_b], bix[keep_b], bf10[keep_b]
        else:
            has_subthr[b] = float(it.get('has_subthr_readout', 1.0))

        if len(bst) > 0:
            scint_grid[b, bst, bvw, bix] = bf10

        mu_tracks[b] = it['muon_kalman_tracks']
        mu_mask[b] = it['muon_kalman_mask']
        trk_cand[b] = it['trk_cand']
        trk_mask[b] = it['trk_cand_mask']
        trk_counts[b] = it['trk_station_counts']

    batch: dict[str, torch.Tensor] = {
        'c_obs': torch.from_numpy(c_obs),
        'obs_domain_id': torch.from_numpy(obs_dom),
        'cal_start_mm': torch.from_numpy(cal_start),
        'has_subthr_readout': torch.from_numpy(has_subthr),
        'has_edep': torch.from_numpy(has_edep),
        's_mir': torch.from_numpy(s_mir),
        'pix_xyz_mm': torch.from_numpy(pix_xyz),
        'pix_plane': torch.from_numpy(pix_plane),
        'pix_valid': torch.from_numpy(pix_valid),
        'pix_feat': torch.from_numpy(pix_feat),
        'pix_plane_summary': torch.from_numpy(pix_ps),
        'pix_patches': torch.from_numpy(pix_patches),
        'patch_cluster_idx': torch.from_numpy(patch_cl_idx),
        'scint_grid': torch.from_numpy(scint_grid),
        'muon_kalman_tracks': torch.from_numpy(mu_tracks),
        'muon_kalman_mask': torch.from_numpy(mu_mask),
        'trk_cand': torch.from_numpy(trk_cand),
        'trk_cand_mask': torch.from_numpy(trk_mask),
        'trk_station_counts': torch.from_numpy(trk_counts),
    }

    # Attach truth keys ONLY when present in training items (never present in Tune8k/SSL/inference!)
    if 'label' in proc[0]:
        batch['label'] = torch.tensor([int(it['label']) for it in proc], dtype=torch.long)
        batch['label_7'] = torch.tensor([int(it['label_7']) for it in proc], dtype=torch.long)
        batch['tau_branch_valid'] = torch.tensor([int(it.get('tau_branch_valid', 0)) for it in proc], dtype=torch.long)
        batch['charm_flag'] = torch.tensor([int(it.get('charm_flag', 0)) for it in proc], dtype=torch.long)
        batch['charm_valid'] = torch.tensor([int(it.get('charm_valid', 0)) for it in proc], dtype=torch.long)
        batch['vertex_xyz_mm'] = torch.from_numpy(np.stack([it['vertex_xyz_mm'] for it in proc]).astype(np.float32))
        batch['vertex_valid'] = torch.tensor([int(it.get('vertex_valid', 0)) for it in proc], dtype=torch.long)
        batch['vis_kinematics'] = torch.from_numpy(np.stack([it['vis_kinematics'] for it in proc]).astype(np.float32))
        batch['kin_valid'] = torch.tensor([int(it.get('kin_valid', 0)) for it in proc], dtype=torch.long)
        batch['sample_weight'] = torch.tensor([float(it.get('sample_weight', 1.0)) for it in proc], dtype=torch.float32)

        cl_orig = np.full((bsz, n_cl_pad), -1, dtype=np.int64)
        cl_sec = np.full((bsz, n_cl_pad), -1, dtype=np.int64)
        cl_frac = np.zeros((bsz, n_cl_pad), dtype=np.float32)
        cl_svg = np.full((bsz, n_cl_pad), -1, dtype=np.int64)
        cl_tid = np.zeros((bsz, n_cl_pad), dtype=np.int64)
        for b, it in enumerate(proc):
            n_c = min(n_cl_pad, len(it.get('cl_origin_label', ())))
            if n_c > 0:
                cl_orig[b, :n_c] = it['cl_origin_label'][:n_c]
                cl_sec[b, :n_c] = it['cl_secondary_origin_label'][:n_c]
                cl_frac[b, :n_c] = it['cl_dominant_edep_frac'][:n_c]
                cl_svg[b, :n_c] = it['cl_sv_group_id'][:n_c]
                cl_tid[b, :n_c] = it['cl_dominant_track_id'][:n_c]

        batch['cl_sidecar_valid'] = torch.tensor([bool(it.get('cl_sidecar_valid', False)) for it in proc], dtype=torch.bool)
        batch['cl_origin_label'] = torch.from_numpy(cl_orig)
        batch['cl_secondary_origin_label'] = torch.from_numpy(cl_sec)
        batch['cl_dominant_edep_frac'] = torch.from_numpy(cl_frac)
        batch['cl_sv_group_id'] = torch.from_numpy(cl_svg)
        batch['cl_dominant_track_id'] = torch.from_numpy(cl_tid)
        batch['true_sv_xyz_mm'] = torch.from_numpy(np.stack([it['true_sv_xyz_mm'] for it in proc]).astype(np.float32))
        batch['has_true_sv'] = torch.tensor([bool(it.get('has_true_sv', False)) for it in proc], dtype=torch.bool)
        batch['tau_n_charged_prongs_si'] = torch.tensor([int(it.get('tau_n_charged_prongs_si', -1)) for it in proc], dtype=torch.long)
        batch['tier1b_genie_valid'] = torch.tensor([bool(it.get('tier1b_genie_valid', False)) for it in proc], dtype=torch.bool)
        batch['true_tau_submode'] = torch.tensor([int(it.get('true_tau_submode', -1)) for it in proc], dtype=torch.long)
        batch['true_n_pi0'] = torch.tensor([int(it.get('true_n_pi0', 0)) for it in proc], dtype=torch.long)
        batch['true_nu_energy_gev'] = torch.tensor([float(it.get('true_nu_energy_gev', 0.0)) for it in proc], dtype=torch.float32)
        batch['true_branch_p4_gev'] = torch.from_numpy(np.stack([it['true_branch_p4_gev'] for it in proc]).astype(np.float32))
        batch['tgt_Z'] = torch.tensor([int(it.get('tgt_Z', 74)) for it in proc], dtype=torch.long)
        batch['tau_decay_L3d_mm'] = torch.tensor([float(it.get('tau_decay_L3d_mm', 0.0)) for it in proc], dtype=torch.float32)

    return batch


def adapt_ssl_shard_to_gen4_batch(
    ssl_items: list[dict[str, Any]],
    indices: np.ndarray,
    geom: Gen4GeometryContract,
    deg_cfg: dict[str, Any] | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Slice pure-observation SSL shard items (`has_subthr_readout=1.0`, `has_edep=1.0`) into Gen4EventBatch."""
    picked = [ssl_items[int(i)] for i in np.asarray(indices, dtype=np.int64).tolist()]
    for x in picked:
        for k in GEN4_TRUTH_BATCH_KEYS:
            if k in x:
                raise SealViolationError(f'Truth key {k} found in SSL shard item!')
    return collate_gen4_batch(picked, geom, aug_cfg=deg_cfg, rng=rng)


def main() -> None:
    import argparse
    import time

    parser = argparse.ArgumentParser(description='Build 10085-10088 Pinpoint-Gen4 feature caches.')
    parser.add_argument('--stage', choices=('train', 'tune', 'supervised', 'ssl', 'ssl-part', 'all'), default='supervised')
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--ssl-part', type=int, default=0, help='0..num_parts-1 for ssl_train, -2 for ssl_val')
    parser.add_argument('--ssl-num-parts', type=int, default=32)
    args = parser.parse_args()

    if args.stage in ('train', 'supervised', 'all'):
        t0 = time.time()
        r_tr = build_train32k_cache(workers=args.workers)
        print(f'[Train32k] done in {time.time() - t0:.1f}s: total={r_tr["total"]} splits={r_tr["splits"]}', flush=True)
    if args.stage in ('tune', 'supervised', 'all'):
        t1 = time.time()
        r_tu = build_tune8k_and_stress_diag_cache(workers=args.workers)
        print(f'[Tune8k+Stress+Diag] done in {time.time() - t1:.1f}s: {r_tu}', flush=True)
    if args.stage == 'ssl-part':
        t2 = time.time()
        r_p = build_ssl_mc8588_part(args.ssl_part, num_parts=args.ssl_num_parts, workers=args.workers)
        print(f'[SSL part {args.ssl_part}] done in {time.time() - t2:.1f}s: {r_p}', flush=True)
    if args.stage in ('ssl', 'all'):
        t2 = time.time()
        r_ssl = build_ssl_mc8588_cache(workers=args.workers)
        print(f'[SSL_320k] done in {time.time() - t2:.1f}s: {r_ssl}', flush=True)


if __name__ == '__main__':
    main()
