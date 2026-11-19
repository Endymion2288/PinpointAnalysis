"""Event-level and pixel-cluster / bar-level truth sidecar extraction (W3-02).

Strictly used ONLY during training via V3TrainDataset. Evaluation and inference
use V3EvalDataset which cannot import or accept truth sidecars.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import uproot

from analysis.flavour.datasets import NativeShard
from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.s5.truth import CHARM_PARENTS, charmed, group_flat, primary_decay
from analysis.flavour.s2.prepare import classify_tau_daughters
from analysis.flavour.v2.partitions import FAMILY, KEY, equal_identity_rows
from analysis.flavour.v3.common.domains import compute_observable_containment, detector_bounds
from analysis.flavour.v3.common.pixel_clusters import cluster_pixel_hits


CLUSTER_ORIGIN_NAMES = (
    'Primary_lepton',
    'Primary_hadron',
    'Tau_daughter',
    'Charm_daughter',
    'Secondary_shower_or_other',
)

LABEL_7_NAMES = (
    'e_CC',
    'mu_minus_CC',
    'mu_plus_CC',
    'tau_e',
    'tau_mu',
    'tau_had',
    'NC',
)

NEUTRINOS = {12, 14, 16, 18}


def _pdg_to_rel_class(pdg_arr: np.ndarray) -> np.ndarray:
    ap = np.abs(pdg_arr.astype(np.int64))
    out = np.full(len(ap), 2, dtype=np.int64)  # default: had
    out[(ap == 11) | (ap == 22)] = 0  # EM
    out[ap == 13] = 1  # mu
    return out


def _compute_label_7(label_4: int, fsl_pdg: int, init_pdg: int, tau_branch: str) -> tuple[int, int]:
    if label_4 == 0:
        return 0, 1
    if label_4 == 1:
        if fsl_pdg == 13 or (fsl_pdg not in (13, -13) and init_pdg == 14):
            return 1, 1
        if fsl_pdg == -13 or (fsl_pdg not in (13, -13) and init_pdg == -14):
            return 2, 1
        return 1, 0
    if label_4 == 2:
        mapping = {'electron': 3, 'muon': 4, 'hadronic': 5}
        if tau_branch in mapping:
            return mapping[tau_branch], 1
        return -1, 0
    return 6, 1


def _compute_vis_kinematics(
    label_4: int,
    fsl_pdg: int,
    primary_ev: dict | None,
    direct_tau: dict,
    tau_decay_valid: bool,
) -> tuple[np.ndarray, int]:
    """Compute [log1p(pz_vis_GeV), log1p(pt_vis_GeV), cos(dphi_lh), sin(dphi_lh)]."""
    if not primary_ev or len(primary_ev['PDG']) == 0:
        return np.zeros(4, dtype=np.float32), 0

    pdgs = np.asarray(primary_ev['PDG'], dtype=np.int64)
    tids = np.asarray(primary_ev['trackID'], dtype=np.int64)
    px = np.asarray(primary_ev['Px'], dtype=np.float64) / 1000.0
    py = np.asarray(primary_ev['Py'], dtype=np.float64) / 1000.0
    pz = np.asarray(primary_ev['Pz'], dtype=np.float64) / 1000.0

    p_lep = np.zeros(3, dtype=np.float64)
    p_had = np.zeros(3, dtype=np.float64)
    has_lep = False
    has_had = False

    if label_4 == 2:
        if not tau_decay_valid or not direct_tau:
            return np.zeros(4, dtype=np.float32), 0
        # Tau visible daughters
        t_pdg = np.asarray(direct_tau['PDG'], dtype=np.int64)
        t_px = np.asarray(direct_tau['Px'], dtype=np.float64) / 1000.0
        t_py = np.asarray(direct_tau['Py'], dtype=np.float64) / 1000.0
        t_pz = np.asarray(direct_tau['Pz'], dtype=np.float64) / 1000.0
        vis_tau = ~np.isin(np.abs(t_pdg), list(NEUTRINOS))
        if vis_tau.any():
            p_lep = np.array([t_px[vis_tau].sum(), t_py[vis_tau].sum(), t_pz[vis_tau].sum()], dtype=np.float64)
            has_lep = True
        for i in range(len(pdgs)):
            ap = abs(int(pdgs[i]))
            if ap in NEUTRINOS or ap >= 10**9 or ap == 15:
                continue
            p_had += np.array([px[i], py[i], pz[i]], dtype=np.float64)
            has_had = True
    else:
        lep_idx = -1
        if label_4 in (0, 1):
            matches = np.flatnonzero(np.abs(pdgs) == abs(int(fsl_pdg)))
            if len(matches):
                lep_idx = int(matches[np.argmin(tids[matches])])
        for i in range(len(pdgs)):
            ap = abs(int(pdgs[i]))
            if ap in NEUTRINOS or ap >= 10**9:
                continue
            vec = np.array([px[i], py[i], pz[i]], dtype=np.float64)
            if i == lep_idx:
                p_lep += vec
                has_lep = True
            else:
                p_had += vec
                has_had = True

    p_tot = p_lep + p_had
    pz_vis = max(0.0, float(p_tot[2]))
    pt_vis = float(np.hypot(p_tot[0], p_tot[1]))
    pt_lep = float(np.hypot(p_lep[0], p_lep[1]))
    pt_had = float(np.hypot(p_had[0], p_had[1]))

    if has_lep and has_had and pt_lep > 1e-4 and pt_had > 1e-4:
        phi_l = float(np.arctan2(p_lep[1], p_lep[0]))
        phi_h = float(np.arctan2(p_had[1], p_had[0]))
        dphi = phi_l - phi_h
        cos_dphi = float(np.cos(dphi))
        sin_dphi = float(np.sin(dphi))
        kin_valid = 1
    else:
        cos_dphi = 0.0
        sin_dphi = 0.0
        kin_valid = 0

    vec = np.array([np.log1p(pz_vis), np.log1p(pt_vis), cos_dphi, sin_dphi], dtype=np.float32)
    return vec, kin_valid


def extract_shard_sidecar(
    shard_path: Path,
    input_root: Path,
    geometry: dict,
    bounds: dict,
    out_dir: Path,
    target_split: str = 'train',
) -> dict:
    """Extract event & cluster/bar truth sidecars for allowed splits (train or validation audit)."""
    if target_split not in ('train', 'validation'):
        raise ValueError(f'truth sidecar extraction refuses split={target_split!r}; sealed splits forbidden')
    started = time.perf_counter()
    prov = pd.read_parquet(shard_path / 'provenance.parquet')
    target_mask = (prov['split'] == target_split).to_numpy()
    target_indices = np.flatnonzero(target_mask)
    if len(target_indices) == 0:
        return {
            'shard': shard_path.name,
            'events': 0,
            'skipped_empty_split': True,
            'seconds': time.perf_counter() - started,
        }

    ns = NativeShard(shard_path, target_columns=('label',))
    nx, ny = bounds['nx'], bounds['ny']
    n_scint = bounds['n_scint_panels']
    n_bars = bounds['n_bars']

    sub_prov = prov.iloc[target_indices].copy().reset_index(drop=True)
    n_ev = len(sub_prov)

    # Preallocate output buffers
    ev_cols = {
        'run': sub_prov['run'].to_numpy(dtype=np.int32),
        'chunk': sub_prov['chunk'].to_numpy(dtype=np.int32),
        'row': sub_prov['row'].to_numpy(dtype=np.int32),
        'evtID': sub_prov['evtID'].to_numpy(dtype=np.int32),
        'family_hi': sub_prov['family_hi'].to_numpy(dtype=np.uint64),
        'family_lo': sub_prov['family_lo'].to_numpy(dtype=np.uint64),
        'label_4': sub_prov['label'].to_numpy(dtype=np.int8),
        'label_7': np.full(n_ev, -1, dtype=np.int8),
        'tau_branch_valid': np.zeros(n_ev, dtype=np.uint8),
        'charm_flag': np.zeros(n_ev, dtype=np.uint8),
        'charm_valid': np.zeros(n_ev, dtype=np.uint8),
        'vertex_x_mm': sub_prov['initX'].to_numpy(dtype=np.float32),
        'vertex_y_mm': sub_prov['initY'].to_numpy(dtype=np.float32),
        'vertex_z_mm': sub_prov['initZ'].to_numpy(dtype=np.float32),
        'vertex_valid': np.zeros(n_ev, dtype=np.uint8),
        'vis_pz_log1p': np.zeros(n_ev, dtype=np.float32),
        'vis_pt_log1p': np.zeros(n_ev, dtype=np.float32),
        'vis_cos_dphi': np.zeros(n_ev, dtype=np.float32),
        'vis_sin_dphi': np.zeros(n_ev, dtype=np.float32),
        'kin_valid': np.zeros(n_ev, dtype=np.uint8),
        'energy_nu_GeV': (sub_prov['initE'].to_numpy(dtype=np.float64) / 1000.0).astype(np.float32),
    }

    pix_offsets = np.zeros(n_ev + 1, dtype=np.int64)
    bar_offsets = np.zeros(n_ev + 1, dtype=np.int64)
    pix_first_cids: list[np.ndarray] = [None] * n_ev  # type: ignore[list-item]
    pix_origins: list[np.ndarray] = [None] * n_ev  # type: ignore[list-item]
    pix_ancestry: list[np.ndarray] = [None] * n_ev  # type: ignore[list-item]
    pix_rels: list[np.ndarray] = [None] * n_ev  # type: ignore[list-item]
    bar_rels: list[np.ndarray] = [None] * n_ev  # type: ignore[list-item]

    sub_prov['local_idx'] = np.arange(n_ev)
    sub_prov['shard_idx'] = target_indices

    for (run, chunk), group in sub_prov.groupby(['run', 'chunk'], sort=True):
        root_path = input_root / str(int(run)) / f'{int(run)}_{int(chunk):03d}.root'
        with uproot.open(root_path, object_cache=None, array_cache=None) as root:
            prim_map = group_flat(root['primaries'], ['evtID', 'PDG', 'trackID', 'Px', 'Py', 'Pz'])
            tau_map = group_flat(root['tau'], ['evtID', 'PID', 'TID', 'PDG', 'ParentPDG', 'ProdX', 'ProdY', 'ProdZ', 'Px', 'Py', 'Pz']) if 'tau' in root else {}
            charm_map = group_flat(root['charm'], ['evtID', 'PID', 'TID', 'PDG', 'ParentPDG', 'ProdX', 'ProdY', 'ProdZ']) if 'charm' in root else {}

            row_arr = group['row'].to_numpy(dtype=np.int64)
            start = int(row_arr.min())
            stop = int(row_arr.max()) + 1
            ph = root['Hits/pixelHits'].arrays(
                ['rowID', 'colID', 'layerID', 'pdg', 'trackID', 'parentID',
                 'fromPrimaryLepton', 'fromPrimaryEMShower', 'fromCharmedHadron', 'fromTau'],
                entry_start=start, entry_stop=stop, library='ak',
            )
            sh = root['Hits/scintHits'].arrays(
                ['layerID', 'panelID', 'colID', 'rowID', 'isHorizontal', 'pdg', 'edep'],
                entry_start=start, entry_stop=stop, library='ak',
            )

            for rec in group.itertuples(index=False):
                li = int(rec.local_idx)
                si = int(rec.shard_idx)
                r_off = int(rec.row) - start
                evt_id = int(rec.evtID)
                lbl4 = int(rec.label)

                obs = ns[si]['observations']
                dom = compute_observable_containment(obs, geometry, bounds)
                vtx = np.array([rec.initX, rec.initY, rec.initZ], dtype=np.float64)
                ev_cols['vertex_valid'][li] = int(np.isfinite(vtx).all() and (dom['obs_domain_id'] != 0))

                # Primaries & tau decay truth
                pev = prim_map.get(evt_id)
                if pev is not None and len(pev['PDG']) > 0:
                    ev_cols['charm_valid'][li] = 1
                    ev_cols['charm_flag'][li] = int(any(charmed(int(p)) for p in pev['PDG']))
                    tau_ids = np.asarray(pev['trackID'], dtype=np.int64)[np.abs(pev['PDG']) == 15]
                else:
                    tau_ids = np.empty(0, dtype=np.int64)

                tev = tau_map.get(evt_id)
                if lbl4 == 2 and len(tau_ids) > 0:
                    direct, _, decay_valid = primary_decay(tev, tau_ids)
                    branch = classify_tau_daughters(direct['PDG']) if direct else 'unknown'
                else:
                    direct, decay_valid = {}, False
                    branch = 'not_tau'

                l7, tb_val = _compute_label_7(lbl4, int(rec.fslPDG), int(rec.initPDG), branch)
                ev_cols['label_7'][li] = l7
                ev_cols['tau_branch_valid'][li] = tb_val

                kin_vec, kv = _compute_vis_kinematics(lbl4, int(rec.fslPDG), pev, direct, decay_valid)
                ev_cols['vis_pz_log1p'][li] = kin_vec[0]
                ev_cols['vis_pt_log1p'][li] = kin_vec[1]
                ev_cols['vis_cos_dphi'][li] = kin_vec[2]
                ev_cols['vis_sin_dphi'][li] = kin_vec[3]
                ev_cols['kin_valid'][li] = kv

                # Pixel clusters alignment & truth
                obs_cid = np.asarray(obs['pixel']['channel_id'], dtype=np.uint64)
                obs_xyz = np.asarray(obs['pixel']['xyz_mm'], dtype=np.float32).reshape(-1, 3)
                cinfo = cluster_pixel_hits(obs_cid, obs_xyz, nx, ny)
                n_cl = cinfo['n_clusters']

                if n_cl == 0:
                    pix_first_cids[li] = np.empty(0, dtype=np.uint64)
                    pix_origins[li] = np.empty(0, dtype=np.int8)
                    pix_ancestry[li] = np.empty((0, 2), dtype=np.int32)
                    pix_rels[li] = np.empty((0, 3), dtype=np.float16)
                else:
                    p_row = np.asarray(ph['rowID'][r_off], dtype=np.int64)
                    p_col = np.asarray(ph['colID'][r_off], dtype=np.int64)
                    p_lay = np.asarray(ph['layerID'][r_off], dtype=np.int64)
                    raw_cid = ((p_lay * ny + p_row) * nx + p_col).astype(np.uint64)

                    # Map raw ROOT hits to obs_cid (both unique channel_ids)
                    pos = np.searchsorted(obs_cid, raw_cid)
                    valid_pos = (pos < len(obs_cid)) & (obs_cid[np.minimum(pos, len(obs_cid) - 1)] == raw_cid)
                    if not valid_pos.all():
                        raise ValueError(f'pixel channel alignment failed at shard={shard_path.name} row={rec.row}')

                    raw_pdg = np.asarray(ph['pdg'][r_off], dtype=np.int64)
                    raw_tid = np.asarray(ph['trackID'][r_off], dtype=np.int32)
                    raw_pid = np.asarray(ph['parentID'][r_off], dtype=np.int32)
                    raw_fl = np.asarray(ph['fromPrimaryLepton'][r_off], dtype=bool)
                    raw_fe = np.asarray(ph['fromPrimaryEMShower'][r_off], dtype=bool)
                    raw_fc = np.asarray(ph['fromCharmedHadron'][r_off], dtype=bool)
                    raw_ft = np.asarray(ph['fromTau'][r_off], dtype=bool)

                    # Hit-level origin in {0..4}
                    hit_orig = np.full(len(raw_cid), 4, dtype=np.int8)
                    is_prim_lep = raw_fl | ((lbl4 == 0) & raw_fe)
                    is_prim_had = (raw_pid == 0) & (~is_prim_lep)
                    hit_orig[is_prim_had] = 1
                    hit_orig[is_prim_lep] = 0
                    hit_orig[raw_fc] = 3
                    hit_orig[raw_ft] = 2

                    hit_rel = _pdg_to_rel_class(raw_pdg)
                    hit_cl = cinfo['hit_to_cluster'][pos]

                    # Vectorized cluster aggregation over hits sorted by hit_cl
                    cl_rel_counts = np.zeros((n_cl, 3), dtype=np.float32)
                    np.add.at(cl_rel_counts, (hit_cl, hit_rel), 1.0)
                    cl_rel = cl_rel_counts / np.maximum(cl_rel_counts.sum(axis=1, keepdims=True), 1.0)

                    cl_orig_counts = np.zeros((n_cl, 5), dtype=np.int32)
                    np.add.at(cl_orig_counts, (hit_cl, hit_orig), 1)
                    non_sec = cl_orig_counts[:, :4].sum(axis=1)
                    cl_orig = np.full(n_cl, 4, dtype=np.int8)
                    has_signal = (non_sec > 0) & ((non_sec * 2 >= cl_orig_counts[:, 4]) | (cl_orig_counts[:, 4] == 0))
                    sig_mat = cl_orig_counts[has_signal][:, [2, 3, 0, 1]]
                    best_sig = np.array([2, 3, 0, 1], dtype=np.int8)[sig_mat.argmax(axis=1)]
                    cl_orig[has_signal] = best_sig

                    # Dominant trackID and true decaying-parent vertex ID per cluster.
                    # Resolves Stage 320 P1: distinct charmed hadrons in c-cbar events get distinct
                    # vertex IDs (encoded as 10000 + parent_track_ID) instead of collapsing all charm into vg=2.
                    cev = charm_map.get(evt_id)
                    charm_parent_ids: set[int] = set()
                    charm_tid_to_pid: dict[int, int] = {}
                    if cev is not None and len(cev['PID']) > 0:
                        for c_pid, c_tid in zip(cev['PID'], cev['TID']):
                            cp, ct = int(c_pid), int(c_tid)
                            charm_parent_ids.add(cp)
                            charm_tid_to_pid[ct] = cp
                    if pev is not None and len(pev['PDG']) > 0:
                        for p_pdg, p_tid in zip(pev['PDG'], pev['trackID']):
                            if charmed(int(p_pdg)):
                                charm_parent_ids.add(int(p_tid))
                    tau_parent_ids: set[int] = {int(t) for t in tau_ids.tolist()}
                    tau_tid_to_pid: dict[int, int] = {}
                    if tev is not None and len(tev['PID']) > 0:
                        for t_pid, t_tid in zip(tev['PID'], tev['TID']):
                            tp, tt = int(t_pid), int(t_tid)
                            tau_parent_ids.add(tp)
                            tau_tid_to_pid[tt] = tp

                    order_h = np.argsort(hit_cl, kind='mergesort')
                    hcl_s = hit_cl[order_h]
                    htid_s = raw_tid[order_h]
                    hpid_s = raw_pid[order_h]
                    horig_s = hit_orig[order_h]
                    cl_b = np.r_[0, np.flatnonzero(np.diff(hcl_s)) + 1, len(hcl_s)]
                    dom_tids = np.zeros(n_cl, dtype=np.int32)
                    vg = np.full(n_cl, -1, dtype=np.int32)
                    for a, b in zip(cl_b[:-1], cl_b[1:]):
                        cid_k = int(hcl_s[a])
                        tids_k = htid_s[a:b]
                        pids_k = hpid_s[a:b]
                        orig_k = horig_s[a:b]
                        sig_k = orig_k != 4
                        pool_tids = tids_k[sig_k] if sig_k.any() else tids_k
                        uniq_t, cnt_t = np.unique(pool_tids, return_counts=True)
                        dom_t = int(uniq_t[np.argmax(cnt_t)])
                        dom_tids[cid_k] = dom_t
                        c_o = int(cl_orig[cid_k])
                        if c_o in (0, 1):
                            vg[cid_k] = 0
                        elif c_o == 2:
                            # Tau decay vertex ID: unique per parent tau trackID
                            matched_p: list[int] = []
                            for t_i, p_i in zip(tids_k, pids_k):
                                ti, pi = int(t_i), int(p_i)
                                if ti in tau_tid_to_pid:
                                    matched_p.append(tau_tid_to_pid[ti])
                                elif pi in tau_parent_ids:
                                    matched_p.append(pi)
                            if matched_p:
                                up, cp = np.unique(matched_p, return_counts=True)
                                vg[cid_k] = 10000 + int(up[np.argmax(cp)])
                            elif len(tau_parent_ids) == 1:
                                vg[cid_k] = 10000 + next(iter(tau_parent_ids))
                            else:
                                vg[cid_k] = -1
                        elif c_o == 3:
                            # Charm decay vertex ID: distinct per charmed hadron parent trackID!
                            # If >=2 charmed hadrons exist and cluster is an unresolved deep-shower
                            # descendant, mark vg = -1 so two distinct charm vertices are NEVER merged.
                            matched_c: list[int] = []
                            for t_i, p_i in zip(tids_k, pids_k):
                                ti, pi = int(t_i), int(p_i)
                                if ti in charm_tid_to_pid:
                                    matched_c.append(charm_tid_to_pid[ti])
                                elif pi in charm_parent_ids:
                                    matched_c.append(pi)
                            if matched_c:
                                uc, cc = np.unique(matched_c, return_counts=True)
                                vg[cid_k] = 20000 + int(uc[np.argmax(cc)])
                            elif len(charm_parent_ids) == 1:
                                vg[cid_k] = 20000 + next(iter(charm_parent_ids))
                            else:
                                vg[cid_k] = -1

                    pix_first_cids[li] = cinfo['cluster_first_channel_id']
                    pix_origins[li] = cl_orig
                    pix_ancestry[li] = np.column_stack([dom_tids, vg]).astype(np.int32)
                    pix_rels[li] = cl_rel.astype(np.float16)

                # Scintillator bar Rel truth aligned to obs['bar']['channel_id']
                obs_bcid = np.asarray(obs['bar']['channel_id'], dtype=np.uint64)
                n_b = len(obs_bcid)
                if n_b == 0:
                    bar_rels[li] = np.empty((0, 3), dtype=np.float16)
                else:
                    b_lay = np.asarray(sh['layerID'][r_off], dtype=np.int64)
                    b_pan = np.asarray(sh['panelID'][r_off], dtype=np.int64)
                    b_h = np.asarray(sh['isHorizontal'][r_off], dtype=np.int64)
                    b_row = np.asarray(sh['rowID'][r_off], dtype=np.int64)
                    b_col = np.asarray(sh['colID'][r_off], dtype=np.int64)
                    b_bar = np.where(b_h == 1, b_row, b_col)
                    b_key = ((b_lay * n_scint + b_pan) * (2 * n_bars) + b_h * n_bars + b_bar).astype(np.uint64)
                    b_pos = np.searchsorted(obs_bcid, b_key)
                    if (b_pos >= n_b).any() or not np.array_equal(obs_bcid[b_pos], b_key):
                        raise ValueError(f'bar channel alignment failed at shard={shard_path.name} row={rec.row}')
                    b_edep = np.asarray(sh['edep'][r_off], dtype=np.float64)
                    b_pdg = np.asarray(sh['pdg'][r_off], dtype=np.int64)
                    b_rel_cls = _pdg_to_rel_class(b_pdg)
                    b_sums = np.zeros((n_b, 3), dtype=np.float64)
                    np.add.at(b_sums, (b_pos, b_rel_cls), b_edep)
                    tot_b = np.maximum(b_sums.sum(axis=1, keepdims=True), 1e-30)
                    bar_rels[li] = (b_sums / tot_b).astype(np.float16)

    for li in range(n_ev):
        pix_offsets[li + 1] = pix_offsets[li] + len(pix_origins[li])
        bar_offsets[li + 1] = bar_offsets[li] + len(bar_rels[li])

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ev_cols).to_parquet(out_dir / 'event_sidecar.parquet', index=False, compression=COMPRESSION)
    np.savez(
        out_dir / 'dense_sidecar.npz',
        pix_offsets=pix_offsets,
        bar_offsets=bar_offsets,
        cluster_first_channel_id=np.concatenate(pix_first_cids),
        cluster_origin=np.concatenate(pix_origins),
        cluster_ancestry_id=np.concatenate(pix_ancestry, axis=0),
        pixel_rel_target=np.concatenate(pix_rels, axis=0),
        bar_rel_target=np.concatenate(bar_rels, axis=0),
    )
    summary = {
        'status': 'complete',
        'shard': shard_path.name,
        'events': n_ev,
        'total_pixel_clusters': int(pix_offsets[-1]),
        'total_active_bars': int(bar_offsets[-1]),
        'seconds': time.perf_counter() - started,
    }
    write_json(out_dir / 'summary.json', summary)
    return summary


def _shard_task(args: tuple[str, str, dict, dict, str]) -> dict:
    s_dir, i_root, geom, bounds, o_dir = args
    return extract_shard_sidecar(Path(s_dir), Path(i_root), geom, bounds, Path(o_dir))


def run_worker(
    data_root: Path,
    input_root: Path,
    output_root: Path,
    shard_start: int,
    shard_end: int,
    workers: int = 4,
) -> None:
    plan = json.loads((data_root / 'plan.json').read_text())
    geometry = json.loads(Path(plan['geometry']).read_text())
    bounds = detector_bounds(geometry)
    shards = sorted(p for p in (data_root / 'shards').iterdir() if (p / 'observations.parquet').exists())
    selected = shards[shard_start:shard_end]
    tasks = []
    for sp in selected:
        if not pd.read_parquet(sp / 'provenance.parquet', columns=['split'])['split'].eq('train').any():
            continue
        od = output_root / 'shards' / sp.name
        if (od / 'summary.json').exists() and (od / 'event_sidecar.parquet').exists() and (od / 'dense_sidecar.npz').exists():
            continue
        tasks.append((str(sp), str(input_root), geometry, bounds, str(od)))
    if not tasks:
        print(json.dumps({'status': 'already_complete', 'shard_start': shard_start, 'shard_end': shard_end}))
        return
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_shard_task, tasks))
    else:
        results = [_shard_task(t) for t in tasks]
    print(json.dumps({'status': 'complete', 'processed_shards': len(results), 'events': sum(r['events'] for r in results)}))


def finalize_sidecar(data_root: Path, manifests_root: Path, output_root: Path) -> dict:
    started = time.perf_counter()
    shards = sorted(p for p in (data_root / 'shards').iterdir() if (p / 'observations.parquet').exists())
    ev_frames = []
    pix_cid, pix_orig, pix_anc, pix_rel, bar_rel = [], [], [], [], []
    pix_off = [np.zeros(1, dtype=np.int64)]
    bar_off = [np.zeros(1, dtype=np.int64)]
    p_cur = 0
    b_cur = 0
    shard_event_ranges = {}

    ev_cursor = 0
    for sp in shards:
        if not pd.read_parquet(sp / 'provenance.parquet', columns=['split'])['split'].eq('train').any():
            continue
        sd = output_root / 'shards' / sp.name
        if not (sd / 'summary.json').exists():
            raise FileNotFoundError(f'missing truth_sidecar shard {sp.name}')
        df = pd.read_parquet(sd / 'event_sidecar.parquet')
        df['shard'] = sp.name
        df['sidecar_index'] = np.arange(ev_cursor, ev_cursor + len(df), dtype=np.int64)
        ev_frames.append(df)
        shard_event_ranges[sp.name] = [ev_cursor, ev_cursor + len(df)]
        ev_cursor += len(df)

        z = np.load(sd / 'dense_sidecar.npz')
        po = z['pix_offsets']
        bo = z['bar_offsets']
        pix_off.append(po[1:] + p_cur)
        bar_off.append(bo[1:] + b_cur)
        p_cur += int(po[-1])
        b_cur += int(bo[-1])
        pix_cid.append(z['cluster_first_channel_id'])
        pix_orig.append(z['cluster_origin'])
        pix_anc.append(z['cluster_ancestry_id'])
        pix_rel.append(z['pixel_rel_target'])
        bar_rel.append(z['bar_rel_target'])

    events = pd.concat(ev_frames, ignore_index=True)
    m300 = pd.read_parquet(manifests_root / 'train_300000.parquet')
    if not equal_identity_rows(events, m300):
        raise ValueError('merged truth_sidecar events differ from frozen Train300k identities')

    events.to_parquet(output_root / 'event_sidecar.parquet', index=False, compression=COMPRESSION)

    cdir = output_root / 'cluster_sidecar'
    cdir.mkdir(parents=True, exist_ok=True)
    arrays_to_save = {
        'pix_offsets': np.concatenate(pix_off),
        'bar_offsets': np.concatenate(bar_off),
        'cluster_first_channel_id': np.concatenate(pix_cid),
        'cluster_origin': np.concatenate(pix_orig),
        'cluster_ancestry_id': np.concatenate(pix_anc, axis=0),
        'pixel_rel_target': np.concatenate(pix_rel, axis=0),
        'bar_rel_target': np.concatenate(bar_rel, axis=0),
    }
    meta = {'events': int(len(events)), 'shard_event_ranges': shard_event_ranges, 'arrays': {}}
    for name, arr in arrays_to_save.items():
        arr = np.ascontiguousarray(arr)
        bin_path = cdir / f'{name}.bin'
        bin_path.write_bytes(arr.tobytes(order='C'))
        meta['arrays'][name] = {'dtype': arr.dtype.str, 'shape': list(arr.shape)}
    write_json(cdir / 'manifest.json', meta)

    orig_all = arrays_to_save['cluster_origin']
    orig_counts = np.bincount(orig_all, minlength=5)
    l7 = events['label_7'].to_numpy()
    tau_mask = events['label_4'].to_numpy() == 2
    coverage = {
        'status': 'complete',
        'seconds': time.perf_counter() - started,
        'events': int(len(events)),
        'total_pixel_clusters': int(p_cur),
        'total_active_bars': int(b_cur),
        'label_7_counts': {
            name: int((l7 == i).sum()) for i, name in enumerate(LABEL_7_NAMES)
        },
        'unknown_tau_count': int((tau_mask & (l7 == -1)).sum()),
        'tau_branch_valid_rate_within_tau': float(events.loc[tau_mask, 'tau_branch_valid'].mean()),
        'charm_valid_rate': float(events['charm_valid'].mean()),
        'charm_positive_rate': float(events['charm_flag'].mean()),
        'vertex_valid_rate': float(events['vertex_valid'].mean()),
        'kin_valid_rate': float(events['kin_valid'].mean()),
        'cluster_origin_counts': {
            name: int(orig_counts[i]) for i, name in enumerate(CLUSTER_ORIGIN_NAMES)
        },
        'cluster_origin_fractions': {
            name: float(orig_counts[i] / max(1, len(orig_all))) for i, name in enumerate(CLUSTER_ORIGIN_NAMES)
        },
        'Cal_A_Cut_B_Confirm_payload_read': False,
    }
    write_json(output_root / 'coverage_summary.json', coverage)
    return coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('worker')
    p.add_argument('--data-root', type=Path, default=Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R2/data'))
    p.add_argument('--input-root', type=Path, default=Path('/eos/project-f/fasersim-bonn/public/pinpoint/data'))
    p.add_argument('--output', type=Path, default=Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/truth_sidecar'))
    p.add_argument('--shard-start', type=int, default=0)
    p.add_argument('--shard-end', type=int, default=136)
    p.add_argument('--workers', type=int, default=4)

    p = sub.add_parser('finalize')
    p.add_argument('--data-root', type=Path, default=Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R2/data'))
    p.add_argument('--manifests-root', type=Path, default=Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/manifests'))
    p.add_argument('--output', type=Path, default=Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/truth_sidecar'))

    args = parser.parse_args()
    if args.command == 'worker':
        run_worker(args.data_root, args.input_root, args.output, args.shard_start, args.shard_end, args.workers)
    else:
        cov = finalize_sidecar(args.data_root, args.manifests_root, args.output)
        print(json.dumps(cov, indent=2))


if __name__ == '__main__':
    main()
