"""Stage 318 (Exp-4): Clean Pinpoint_G4 dataset production & HTCondor multi-core worker array.

Resolves Audit P0-2, P1-1, and P2-1 end-to-end:
  1. Uses Stage 315 `pinpoint_fixed` (G4PVPlacement-derived geometry, 100% G4Navigator verified,
     Birks' law kB=0.126 mm/MeV + 14,400-cell SiPM saturation, Sm2Co17 magnet shells, V-A tau
     helicity polarization, monotonic truth ancestry).
  2. Eliminates the 2.31 TeV nu_tau energy ceiling by harvesting tungsten nu_tau CC events up to
     > 4 TeV from Pythia6_Standard forward GENIE files alongside Trench GENIE files.
  3. Supplements GENIE NC events with physical lab-frame Pythia8 fixed-target tungsten NC DIS
     charm events (nu + p/n -> nu + c + cbar + X, charmed hadrons decayed in tungsten by Geant4).
  4. Samples 100% of interaction vertices inside physical G4_W tungsten slabs [z_W - 2.45, z_W + 2.45] mm
     and partitions strictly unique generator events (zero family overlap) into 5 splits:
       train (32,000), validation (8,000), cal_a (4,000), cut_b (6,000), confirm (10,000)
     plus 3 paired Geant4 physics-systematic shards (tau_helicity_0, w_density_0p98, no_magnet_shells).
  5. Converts Geant4 output directly in worker scratch to Zstd Parquet shards + train/tune sidecars,
     leaving zero intermediate .root files on EOS.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import time

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import uproot

from analysis.flavour.create_data import BRANCHES, ROOT_TREES, lists
from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.readout import bar_channels, modality_state, pixel_channels, tracker_measurements
from analysis.flavour.s2.prepare import classify_tau_daughters
from analysis.flavour.s5.truth import charmed, group_flat, primary_decay
from analysis.flavour.s6.gfaser import legacy_record, nominal_record, write_gfaser
from analysis.flavour.v2.partitions import FAMILY, KEY
from analysis.flavour.v3.common.domains import (
    compute_observable_containment,
    detector_bounds,
    pinpoint_slab_catalog,
)
from analysis.flavour.v3.common.submit import submit_job
from analysis.flavour.v3.common.truth_sidecar import _compute_label_7, _compute_vis_kinematics


LCG107 = '/cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc11-opt'
PINPOINT_FIXED_BIN = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v4_audit_2026-11-18/g4/pinpoint_fixed')
GEOM_MACRO = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v4_audit_2026-11-18/g4/macros/geom.mac')
PINPOINT_DATA_DIR = Path('/eos/project-f/fasersim-bonn/public/pinpoint/data')
GENIE_TUNES_ROOT = Path('/eos/experiment/fasernu-data0/MC/Genie_files/genie_output_Pythia6tunes')
DEFAULT_CLEAN_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v4_audit_2026-11-18/clean_sim')
DEFAULT_CLEAN_V2_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v5_clean_v2_2026-11-19/clean_sim')

CHARM_PDGS = {411, 421, 431, 4122, 4132, 4232, 4332}

SPLIT_QUOTAS_PER_CLASS = (
    ('train', 8000),
    ('validation', 2000),
    ('cal_a', 1000),
    ('cut_b', 1500),
    ('confirm', 2500),
)
SHARD_SIZE = 400
SYS_CONFIRM_PER_CLASS = 300  # 1,200 paired Confirm events for Geant4 physics systematics

# Exact 3-stratum physical-volume tungsten sampling probabilities (summing to 1.0):
#   0: front PinpointBlock slabs (0..11) inside silicon pixel XY acceptance
#   1: intermediate PinpointBlock slabs (24..25, 38..39, 52..53, 66..67, 80..81, 94..95) inside pixel XY
#   2: exact area-weighted uniform over ALL remaining G4_W volume (24 Pinpoint outer XY + 84 Fortune full XY)
STRATUM_PROBS = (0.40, 0.15, 0.45)
STRATUM_NAMES = ('front_pinpoint_pix_xy', 'interm_pinpoint_pix_xy', 'remaining_tungsten_uniform')


def _run_lcg107(cmd: list[str], *, env_extra: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    export_lines = []
    for k, v in (env_extra or {}).items():
        if k == 'PYTHONPATH':
            export_lines.append(f'export PYTHONPATH={shlex.quote(str(v))}:"${{PYTHONPATH:-}}"')
        else:
            export_lines.append(f'export {k}={shlex.quote(str(v))}')
    script = '\n'.join([
        'env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH -u ROOTSYS bash -c ' + shlex.quote('\n'.join([
            'which() { command -v "$@"; }; export -f which',
            f'source {LCG107}/setup.sh >/dev/null 2>&1',
            'set -e',
            'export QT_QPA_PLATFORM=offscreen',
            f'export PYTHIA8DATA={LCG107}/share/Pythia8/xmldoc',
            *export_lines,
            'exec ' + shlex.join([str(x) for x in cmd]),
        ]))
    ])
    subprocess.run(['/bin/bash', '-c', script], cwd=cwd, check=True)


def _export_clean_geometry(out_json: Path) -> dict:
    if out_json.exists():
        return json.loads(out_json.read_text())
    with tempfile.TemporaryDirectory(prefix='clean-geom-') as tmp:
        tmp_p = Path(tmp)
        in_root = tmp_p / 'probe_in.gfaser.root'
        root_out = tmp_p / 'geom_probe.root'
        mac = tmp_p / 'geom_probe.mac'
        probe_rec = [{
            'n': 3,
            'pdgc': [14, 1000741840, 13],
            'status': [0, 0, 1],
            'name': ['', '', ''],
            'px': [0.0, 0.0, 0.1],
            'py': [0.0, 0.0, 0.0],
            'pz': [200.0, 0.0, 199.9],
            'E': [200.0, 171.4, 199.9001],
            'vx': 0.0,
            'vy': 0.0,
            'vz': -2.98449,
            'pxv': 0.0,
            'pyv': 0.0,
            'pzv': 200.0,
            'Ev': 200.0,
            'Q2': 1.0,
            'W': 2.0,
            'x': 0.1,
            'y': 0.01,
            'XSec': 1.0,
            'intType': 2,
            'scatteringType': 1,
            'Z': 74,
            'A': 184,
            'tgt': 1000741840,
            'neu': 14,
            'fspl': 13,
            'hitnuc': 2112,
            'cc': True,
            'nc': False,
        }]
        write_gfaser(probe_rec, in_root)
        mac.write_text('\n'.join([
            '/control/verbose 0',
            '/run/verbose 0',
            f'/control/execute {GEOM_MACRO}',
            '/det/enableFaserSpectrometer true',
            '/gen/select gfaser',
            f'/gen/gfaser/inputFile {in_root}',
            '/gen/gfaser/hitPixelArea false',
            f'/out/fileName {root_out}',
            '/out/saveTrack false',
            '/out/saveTruthHits false',
            '/random/setSeeds 20261118 20261135',
            '/run/initialize',
            '/Pinpoint/field/value 0.57 tesla',
            '/run/beamOn 1',
            '',
        ]))
        _run_lcg107(
            [str(PINPOINT_FIXED_BIN), str(mac)],
            env_extra={'PINPOINT_USE_INPUT_Z': '1', 'PINPOINT_SKIP_GDML': '1', 'PINPOINT_COMPACT_TRUTH': '1'},
            cwd=tmp_p,
        )
        with uproot.open(root_out) as f:
            geom = f['geometry'].arrays(library='ak')[0].to_list()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_json, geom)
    return geom


def _family_pair(source_tag: str, source_row: int) -> tuple[int, int]:
    dig = hashlib.blake2b(f'{source_tag}:{source_row}'.encode('utf-8'), digest_size=16).digest()
    lo = int.from_bytes(dig[:8], 'little') & 0x7FFFFFFFFFFFFFFF
    hi = int.from_bytes(dig[8:], 'little') & 0x7FFFFFFFFFFFFFFF
    return hi, lo


def _build_tungsten_sampler_spec(geometry: dict, bounds: dict | None = None) -> dict:
    """Precompute exact 3D physical-volume tungsten strata matching DetectorConstruction.cc."""
    if bounds is None:
        bounds = detector_bounds(geometry)
    cat = pinpoint_slab_catalog(geometry)
    margin_xy = 0.5  # mm inset inside 430x430 mm G4_W plate and 266x196 mm PixelSensor
    w_x0 = float(cat['w_x_min_mm']) + margin_xy
    w_x1 = float(cat['w_x_max_mm']) - margin_xy
    w_y0 = float(cat['w_y_min_mm']) + margin_xy
    w_y1 = float(cat['w_y_max_mm']) - margin_xy

    p_x0 = max(float(bounds['px_min']), float(cat['w_x_min_mm'])) + margin_xy
    p_x1 = min(float(bounds['px_max']), float(cat['w_x_max_mm'])) - margin_xy
    p_y0 = max(float(bounds['py_min']), float(cat['w_y_min_mm'])) + margin_xy
    p_y1 = min(float(bounds['py_max']), float(cat['w_y_max_mm'])) - margin_xy

    a_w = (w_x1 - w_x0) * (w_y1 - w_y0)
    a_pix = max(0.0, p_x1 - p_x0) * max(0.0, p_y1 - p_y0)
    # Exact disjoint rectangle decomposition of (W_box \ Pix_box)
    outer_rects: list[tuple[float, float, float, float, float]] = []
    for rx0, rx1, ry0, ry1 in (
        (w_x0, p_x0, w_y0, w_y1),  # left strip (0 if flush)
        (p_x1, w_x1, w_y0, w_y1),  # right strip
        (p_x0, p_x1, w_y0, p_y0),  # bottom-middle strip
        (p_x0, p_x1, p_y1, w_y1),  # top-middle strip
    ):
        area = max(0.0, rx1 - rx0) * max(0.0, ry1 - ry0)
        if area > 1e-6:
            outer_rects.append((rx0, rx1, ry0, ry1, area))
    a_outer = sum(r[4] for r in outer_rects)
    if abs((a_pix + a_outer) - a_w) > 1e-4:
        raise AssertionError(f'Rectangle decomposition area mismatch: {a_pix + a_outer} != {a_w}')

    n_front = len(cat['front_pinpoint_indices'])
    n_interm = len(cat['intermediate_pinpoint_indices'])
    n_pin = len(cat['pinpoint_indices'])
    n_for = len(cat['fortune_indices'])
    v_tot = float(cat['n_slabs']) * a_w

    f0 = (n_front * a_pix) / v_tot
    f1 = (n_interm * a_pix) / v_tot
    v_rem_pin_outer = n_pin * a_outer
    v_rem_fortune = n_for * a_w
    f2 = (v_rem_pin_outer + v_rem_fortune) / v_tot
    if abs((f0 + f1 + f2) - 1.0) > 1e-9:
        raise AssertionError(f'Stratum volume fractions sum to {f0 + f1 + f2} != 1.0')

    p0, p1, p2 = STRATUM_PROBS
    weights = (f0 / p0, f1 / p1, f2 / p2)
    outer_probs = np.asarray([r[4] / max(a_outer, 1e-12) for r in outer_rects], dtype=np.float64)
    outer_cdf = np.cumsum(outer_probs)
    outer_cdf[-1] = 1.0

    return {
        'catalog': cat,
        'w_box': (w_x0, w_x1, w_y0, w_y1),
        'pix_box': (p_x0, p_x1, p_y0, p_y1),
        'outer_rects': outer_rects,
        'outer_cdf': outer_cdf,
        'p_rem_pin_outer': float(v_rem_pin_outer / max(v_rem_pin_outer + v_rem_fortune, 1e-12)),
        'volume_fractions': (float(f0), float(f1), float(f2)),
        'sampling_probs': (float(p0), float(p1), float(p2)),
        'geom_weights': tuple(float(w) for w in weights),
        'half_w_mm': 0.5 * float(cat['w_thickness_mm']) - 0.05,  # 50 um safety margin inside G4_W face
    }


def _sample_tungsten_vertex_v2_mm(fam_hi: int, fam_lo: int, spec: dict) -> dict:
    """Deterministically sample a physical G4_W tungsten vertex with exact module & mass weight metadata."""
    rng = np.random.default_rng([fam_hi & 0xFFFFFFFF, (fam_hi >> 32) & 0xFFFFFFFF, fam_lo & 0xFFFFFFFF])
    cat = spec['catalog']
    p0, p1, _ = spec['sampling_probs']
    u = float(rng.random())

    if u < p0:
        stratum = 0
        slab_id = int(rng.choice(cat['front_pinpoint_indices']))
        x0, x1, y0, y1 = spec['pix_box']
        vx = float(rng.uniform(x0, x1))
        vy = float(rng.uniform(y0, y1))
        in_pix_xy = True
    elif u < p0 + p1:
        stratum = 1
        slab_id = int(rng.choice(cat['intermediate_pinpoint_indices']))
        x0, x1, y0, y1 = spec['pix_box']
        vx = float(rng.uniform(x0, x1))
        vy = float(rng.uniform(y0, y1))
        in_pix_xy = True
    else:
        stratum = 2
        if float(rng.random()) < spec['p_rem_pin_outer']:
            slab_id = int(rng.choice(cat['pinpoint_indices']))
            r_idx = int(np.searchsorted(spec['outer_cdf'], float(rng.random()), side='right'))
            r_idx = min(r_idx, len(spec['outer_rects']) - 1)
            rx0, rx1, ry0, ry1, _ = spec['outer_rects'][r_idx]
            vx = float(rng.uniform(rx0, rx1))
            vy = float(rng.uniform(ry0, ry1))
            in_pix_xy = False
        else:
            slab_id = int(rng.choice(cat['fortune_indices']))
            x0, x1, y0, y1 = spec['w_box']
            vx = float(rng.uniform(x0, x1))
            vy = float(rng.uniform(y0, y1))
            px0, px1, py0, py1 = spec['pix_box']
            in_pix_xy = (px0 <= vx <= px1) and (py0 <= vy <= py1)

    vz = float(cat['z_center_mm'][slab_id] + rng.uniform(-spec['half_w_mm'], spec['half_w_mm']))
    return {
        'vx_mm': vx,
        'vy_mm': vy,
        'vz_mm': vz,
        'slab_id': int(slab_id),
        'module_type': str(cat['module_type'][slab_id]),
        'block_copy_no': int(cat['block_copy_no'][slab_id]),
        'plate_in_block': int(cat['plate_in_block'][slab_id]),
        'is_pinpoint_module': bool(cat['is_pinpoint_slab'][slab_id]),
        'is_front_pinpoint': bool(cat['is_front_pinpoint'][slab_id]),
        'is_interm_pinpoint': bool(cat['is_intermediate_pinpoint'][slab_id]),
        'is_pixel_xy_acceptance': bool(in_pix_xy),
        'sampling_stratum': int(stratum),
        'sampling_stratum_name': STRATUM_NAMES[stratum],
        'geom_weight': float(spec['geom_weights'][stratum]),
    }


def _sample_tungsten_vertex_mm(fam_hi: int, fam_lo: int, w_z: np.ndarray, w_t: float, *, spec: dict | None = None) -> tuple[float, float, float]:
    """Backwards-compatible tuple wrapper delegating to exact physical-volume sampler when spec is provided."""
    if spec is not None:
        v = _sample_tungsten_vertex_v2_mm(fam_hi, fam_lo, spec)
        return v['vx_mm'], v['vy_mm'], v['vz_mm']
    raise ValueError('_sample_tungsten_vertex_mm requires physical sampler spec from _build_tungsten_sampler_spec(geometry)')


def _read_pinpoint_root_chunk(task: tuple[int, int, int]) -> list[dict]:
    run, chunk, target_label = task
    path = PINPOINT_DATA_DIR / str(run) / f'{run}_{chunk:03d}.root'
    if not path.exists():
        return []
    trivial = uproot.source.futures.TrivialExecutor()
    ev_cols = [
        'evtID', 'initX', 'initY', 'initZ', 'initPx', 'initPy', 'initPz', 'initE',
        'initPDG', 'fslPDG', 'hitnucPDG', 'intType', 'scatteringType', 'tgtPDG', 'Q2', 'W', 'xBj', 'y', 'xs',
    ]
    with uproot.open(path, decompression_executor=trivial, interpretation_executor=trivial,
                     object_cache=None, array_cache=None) as src:
        ev = src['event'].arrays(ev_cols, library='np')
        prim_map = group_flat(src['primaries'], ['evtID', 'PDG', 'Px', 'Py', 'Pz', 'E'])

    out = []
    tag = f'pinpoint:{run}:{chunk:03d}'
    n_rows = len(ev['evtID'])
    for r in range(n_rows):
        evt_id = int(ev['evtID'][r])
        pev = prim_map.get(evt_id)
        if pev is None or len(pev['PDG']) == 0:
            continue
        meta = {k: ev[k][r] for k in ev_cols}
        it = int(meta['intType'])
        neu = int(meta['initPDG'])
        lbl = (abs(neu) - 12) // 2 if it == 2 else (3 if it == 3 else -1)
        if lbl != target_label:
            continue
        rec = nominal_record(meta, pev)
        for k_nan in ('Q2', 'W', 'x', 'y', 'XSec'):
            if not np.isfinite(rec[k_nan]):
                rec[k_nan] = 0.0
        has_ch = any(abs(int(p)) in CHARM_PDGS for p in pev['PDG'])
        hi, lo = _family_pair(tag, r)
        rec.update({
            'source_tag': tag,
            'source_row': int(r),
            'family_hi': int(hi),
            'family_lo': int(lo),
            'label': int(lbl),
            'has_gen_charm': bool(has_ch),
        })
        out.append(rec)
    return out


def _read_legacy_gfaser_file(task: tuple[str, set[int], float]) -> list[dict]:
    path_str, wanted_labels, min_e_gev = task
    path = Path(path_str)
    if not path.exists():
        return []
    with uproot.open(path) as f:
        t = f['gFaser']
        arr = t.arrays(['pdgc', 'status', 'firstMother', 'px', 'py', 'pz', 'E'], library='ak')
    p = arr['pdgc']
    s = arr['status']
    m = arr['firstMother']
    en = arr['E']
    nu_mask = (s == 0) & ((abs(p) == 12) | (abs(p) == 14) | (abs(p) == 16))
    tgt_mask = (s == 0) & (p >= 1000000000) & (p < 1100000000)
    nu = ak.to_numpy(ak.fill_none(ak.firsts(p[nu_mask]), 0))
    ev = ak.to_numpy(ak.fill_none(ak.firsts(en[nu_mask]), 0.0))
    tgt = ak.to_numpy(ak.fill_none(ak.firsts(p[tgt_mask]), 0))
    lep_mask = (s == 1) & (m == 0) & ((abs(p) >= 11) & (abs(p) <= 16))
    lep = ak.to_numpy(ak.fill_none(ak.firsts(p[lep_mask]), 0))
    cc = lep == np.sign(nu) * (np.abs(nu) - 1)
    nc = lep == nu
    lbl = np.where(cc, (np.abs(nu) - 12) // 2, np.where(nc, 3, -1))
    # Tungsten Z == 74; pre-filter high-E (>2 TeV) for e/mu so we don't parse 80k low-E legacy records
    is_w = ((tgt // 10000) % 1000) == 74
    keep = is_w & np.isin(lbl, list(wanted_labels)) & ((lbl == 2) | (ev > max(min_e_gev, 2000.0)))
    rows = np.flatnonzero(keep)

    out = []
    tag = f'legacy:{path.parent.parent.parent.name}/{path.parent.parent.name}/{path.parent.name}/{path.name}'
    for r in rows.tolist():
        try:
            rec = legacy_record(ak.to_list(arr[r]), [0.0, 0.0, 0.0])
        except ValueError:
            continue
        has_ch = any(abs(pdg_i) in CHARM_PDGS for pdg_i, st_i in zip(rec['pdgc'], rec['status']) if st_i == 1)
        hi, lo = _family_pair(tag, r)
        for k_nan in ('W', 'x', 'XSec'):
            if not np.isfinite(rec[k_nan]):
                rec[k_nan] = 0.0
        rec.update({
            'source_tag': tag,
            'source_row': int(r),
            'family_hi': int(hi),
            'family_lo': int(lo),
            'label': int(lbl[r]),
            'has_gen_charm': bool(has_ch),
        })
        out.append(rec)
    return out


def _pythia8_nc_charm_worker(args_tuple: tuple[int, int, list[float]]) -> list[dict]:
    worker_id, quota, energy_pool = args_tuple
    import pythia8

    rng = np.random.default_rng(20261119 + 997 * worker_id)
    out: list[dict] = []

    # Continuous energy spectrum matched to empirical tungsten NC energies above charm threshold (>= 60 GeV)
    ep = np.asarray(energy_pool, dtype=np.float64) if energy_pool else np.empty(0, dtype=np.float64)
    ep = ep[np.isfinite(ep) & (ep >= 60.0) & (ep <= 4500.0)]
    if len(ep) < 100:
        # Smooth fallback log-normal / forward-LHC spectrum
        ep = np.exp(rng.normal(np.log(550.0), 0.75, size=2000)).clip(60.0, 4000.0)

    # All 8 physical (nu_flavour, W184 nucleon) combinations weighted by forward LHC NC flux * (74p / 110n)
    p_proton = 74.0 / 184.0
    p_neutron = 110.0 / 184.0
    flav_weights = ((14, 0.60), (-14, 0.24), (12, 0.11), (-12, 0.05))
    channels: list[tuple[int, int, float]] = []
    for nu_pdg, fw in flav_weights:
        channels.append((nu_pdg, 2212, fw * p_proton))
        channels.append((nu_pdg, 2112, fw * p_neutron))
    ch_probs = np.asarray([c[2] for c in channels], dtype=np.float64)
    ch_probs /= ch_probs.sum()

    # Use 24 continuous energy-channel batches per worker (24 * 12 workers = 288 continuous beam energies!)
    n_batches = min(24, max(8, quota // 8))
    base_per_batch = max(1, (quota + n_batches - 1) // n_batches)

    # Draw 24 continuous beam energies directly from empirical NC energy distribution (with small +-3% continuous jitter)
    e_draws = rng.choice(ep, size=n_batches, replace=True) * np.exp(rng.uniform(-0.03, 0.03, size=n_batches))
    e_draws = np.clip(e_draws, 60.0, 4200.0)
    # Ensure all 8 (nu_pdg, nuc_pdg) channels are represented proportionally
    ch_draws = rng.choice(len(channels), size=n_batches, replace=True, p=ch_probs)
    for k_ensure in range(min(len(channels), n_batches)):
        ch_draws[(worker_id + k_ensure) % n_batches] = k_ensure

    seq = 0
    for b_idx in range(n_batches):
        if len(out) >= quota:
            break
        e_beam = float(e_draws[b_idx])
        nu_pdg, nuc_pdg, _ = channels[int(ch_draws[b_idx])]
        m_nuc = 0.938272 if nuc_pdg == 2212 else 0.939565
        sub_q = min(base_per_batch, quota - len(out))

        pythia = pythia8.Pythia('', False)
        pythia.readString('Print:quiet = on')
        pythia.readString('Random:setSeed = on')
        pythia.readString(f'Random:seed = {20000 + worker_id * 1000 + seq}')
        seq += 1
        pythia.readString('Beams:frameType = 2')
        pythia.readString(f'Beams:idA = {nu_pdg}')
        pythia.readString(f'Beams:idB = {nuc_pdg}')
        pythia.readString(f'Beams:eA = {e_beam:.4f}')
        pythia.readString(f'Beams:eB = {m_nuc:.6f}')
        pythia.readString('WeakBosonExchange:ff2ff(t:gmZ) = on')
        pythia.readString('PhaseSpace:Q2Min = 4.0')
        for c_pdg in CHARM_PDGS:
            pythia.readString(f'{c_pdg}:mayDecay = off')
        if not pythia.init():
            continue

        got = 0
        tries = 0
        while got < sub_q and len(out) < quota and tries < sub_q * 220:
            tries += 1
            if not pythia.next():
                continue
            ev = pythia.event
            finals = []
            out_nu = None
            has_c = False
            for i in range(1, ev.size()):
                if not ev[i].isFinal():
                    continue
                pid = int(ev[i].id())
                p4 = (pid, float(ev[i].px()), float(ev[i].py()), float(ev[i].pz()), float(ev[i].e()))
                finals.append(p4)
                if pid == nu_pdg and (out_nu is None or p4[4] > out_nu[4]):
                    out_nu = p4
                if abs(pid) in CHARM_PDGS:
                    has_c = True
            if not has_c or out_nu is None or out_nu[4] <= 0.0:
                continue

            # Exact event-by-event DIS kinematics from q = p_nu_in - p_nu_out (fixed-target frame)
            q_e = e_beam - out_nu[4]
            q_x = -out_nu[1]
            q_y = -out_nu[2]
            q_z = e_beam - out_nu[3]
            q2_val = max(0.0, -(q_e * q_e - q_x * q_x - q_y * q_y - q_z * q_z))
            y_val = float(np.clip(q_e / max(e_beam, 1e-6), 0.0, 1.0))
            w2_val = m_nuc * m_nuc + 2.0 * m_nuc * q_e - q2_val
            w_val = math.sqrt(max(w2_val, 0.0))
            x_val = float(np.clip(q2_val / max(2.0 * m_nuc * q_e, 1e-6), 0.0, 1.0))

            row_id = worker_id * 100000 + len(out)
            tag = 'pythia8_nc_charm_W184_v2'
            hi, lo = _family_pair(tag, row_id)
            pdgc = [nu_pdg, 1000741840] + [p[0] for p in finals]
            status = [0, 0] + [1] * len(finals)
            px = [0.0, 0.0] + [p[1] for p in finals]
            py = [0.0, 0.0] + [p[2] for p in finals]
            pz = [e_beam, 0.0] + [p[3] for p in finals]
            en_l = [e_beam, 171.4] + [p[4] for p in finals]
            out.append({
                'source_tag': tag,
                'source_row': int(row_id),
                'family_hi': int(hi),
                'family_lo': int(lo),
                'label': 3,
                'has_gen_charm': True,
                'n': len(pdgc),
                'pdgc': pdgc,
                'status': status,
                'name': [''] * len(pdgc),
                'px': px,
                'py': py,
                'pz': pz,
                'E': en_l,
                'pxv': 0.0,
                'pyv': 0.0,
                'pzv': float(e_beam),
                'Ev': float(e_beam),
                'Q2': float(q2_val),
                'W': float(w_val),
                'x': float(x_val),
                'y': float(y_val),
                'XSec': 0.0,
                'intType': 3,
                'scatteringType': 1,
                'Z': 74,
                'A': 184,
                'tgt': 1000741840,
                'neu': int(nu_pdg),
                'fspl': int(nu_pdg),
                'hitnuc': int(nuc_pdg),
                'cc': False,
                'nc': True,
            })
            got += 1
    return out


def prepare_clean_sim(root: Path, workers: int = 16) -> dict:
    t0 = time.perf_counter()
    root.mkdir(parents=True, exist_ok=True)
    geom_path = root / 'geometry.json'
    geom = _export_clean_geometry(geom_path)
    bounds = detector_bounds(geom)
    sampler_spec = _build_tungsten_sampler_spec(geom, bounds)

    total_per_class = sum(q for _, q in SPLIT_QUOTAS_PER_CLASS)  # 15,000 per class -> 60,000 total

    # 1. Harvest primary generator records in parallel from runs 10081..10084 (20 chunks per run = ~20k/class)
    chunk_tasks: list[tuple[int, int, int]] = []
    for run_id, lbl_id, n_ch in ((10081, 0, 20), (10082, 1, 20), (10083, 2, 20), (10084, 3, 20)):
        for ch_id in range(n_ch):
            chunk_tasks.append((run_id, ch_id, lbl_id))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        chunk_lists = list(pool.map(_read_pinpoint_root_chunk, chunk_tasks))
    by_cls_base: dict[int, list[dict]] = {0: [], 1: [], 2: [], 3: []}
    for sub in chunk_lists:
        for r in sub:
            by_cls_base[r['label']].append(r)

    # 2. Harvest high-energy tungsten events (>2.31 TeV nu_tau CC included!) from Pythia6_Standard & Pythia6_NUX
    legacy_tasks: list[tuple[str, set[int], float]] = []
    for tune_name in ('Pythia6_Standard', 'Pythia6_NUX'):
        tdir = GENIE_TUNES_ROOT / tune_name
        if not tdir.exists():
            continue
        for flux_sub, n_files in (('Powheg_pythia_charm_central', 100), ('EPOSLHC_k', 40), ('EPOSLHC_pi', 20)):
            fdir = tdir / flux_sub
            if not fdir.exists():
                continue
            for fid in range(n_files):
                fp = fdir / str(fid) / f'faser.{fid}.gfaser.root'
                if fp.exists():
                    wanted = {0, 1, 2} if tune_name == 'Pythia6_Standard' else {2}
                    legacy_tasks.append((str(fp), wanted, 0.0))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        leg_lists = list(pool.map(_read_legacy_gfaser_file, legacy_tasks))
    leg_all = [r for sub in leg_lists for r in sub]

    leg_tau_hi = [r for r in leg_all if r['label'] == 2 and r['Ev'] > 2312.0]
    leg_tau_lo = [r for r in leg_all if r['label'] == 2 and r['Ev'] <= 2312.0]
    leg_nue_hi = [r for r in leg_all if r['label'] == 0 and r['Ev'] > 2000.0]
    leg_num_hi = [r for r in leg_all if r['label'] == 1 and r['Ev'] > 2000.0]

    # 3. Generate 2,400 genuine continuous-energy multi-flavour Pythia8 tungsten NC_charm events in parallel
    nc_energy_pool = [float(r['Ev']) for r in by_cls_base[3] if float(r['Ev']) >= 60.0]
    n_nc_charm = 2400
    n_pw = min(12, workers)
    per_w = (n_nc_charm + n_pw - 1) // n_pw
    with ProcessPoolExecutor(max_workers=n_pw) as ppool:
        nc_c_lists = list(ppool.map(_pythia8_nc_charm_worker, [(w, per_w, nc_energy_pool) for w in range(n_pw)]))
    nc_c_recs = [r for sub in nc_c_lists for r in sub][:n_nc_charm]

    rng = np.random.default_rng(20261119)
    # Build priority-first pools so 100% of >2.31 TeV nu_tau CC and 100% of NC_charm enter the 15,000/class selection
    def _shuffle_list(lst: list[dict]) -> list[dict]:
        if not lst:
            return []
        idx_p = rng.permutation(len(lst))
        return [lst[int(i)] for i in idx_p]

    pools = {
        0: _shuffle_list(leg_nue_hi)[:1500] + _shuffle_list(by_cls_base[0]),
        1: _shuffle_list(leg_num_hi)[:1500] + _shuffle_list([r for r in by_cls_base[1] if r['has_gen_charm']])[:2200] + _shuffle_list(by_cls_base[1]),
        2: _shuffle_list(leg_tau_hi) + _shuffle_list(leg_tau_lo)[:3000] + _shuffle_list(by_cls_base[2]),
        3: _shuffle_list(nc_c_recs) + _shuffle_list(by_cls_base[3]),
    }

    # Deduplicate globally by (family_hi, family_lo) taking priority-first up to 15,000, then shuffle across splits
    selected_by_split: dict[str, list[dict]] = {sp: [] for sp, _ in SPLIT_QUOTAS_PER_CLASS}
    class_stats = {}
    global_seen_fam: set[tuple[int, int]] = set()
    for cls_id in range(4):
        uniq = []
        for r in pools[cls_id]:
            fk = (r['family_hi'], r['family_lo'])
            if fk not in global_seen_fam:
                global_seen_fam.add(fk)
                uniq.append(r)
        if len(uniq) < total_per_class:
            raise RuntimeError(f'Class {cls_id} has {len(uniq)} unique events < required {total_per_class}')
        top_chosen = uniq[:total_per_class]
        perm = rng.permutation(len(top_chosen))
        chosen = [top_chosen[int(i)] for i in perm]
        cursor = 0
        for sp_name, q_cls in SPLIT_QUOTAS_PER_CLASS:
            selected_by_split[sp_name].extend(chosen[cursor : cursor + q_cls])
            cursor += q_cls
        evs = np.array([r['Ev'] for r in chosen], dtype=np.float64)
        class_stats[str(cls_id)] = {
            'available_unique': len(uniq),
            'selected': len(chosen),
            'E_max_GeV': float(evs.max()),
            'count_E_gt_2312_GeV': int((evs > 2312.0).sum()),
            'count_gen_charm': int(sum(1 for r in chosen if r['has_gen_charm'])),
        }

    # Assign exact physical-volume tungsten vertices and pack into single-split 400-event shards (run=32100)
    shards_dir = root / 'input_shards'
    shards_dir.mkdir(parents=True, exist_ok=True)
    shard_id = 0
    manifest_rows = []
    for sp_name, _ in SPLIT_QUOTAS_PER_CLASS:
        sp_recs = selected_by_split[sp_name]
        sp_perm = rng.permutation(len(sp_recs))
        sp_recs = [sp_recs[int(i)] for i in sp_perm]
        for start in range(0, len(sp_recs), SHARD_SIZE):
            chunk_recs = sp_recs[start : start + SHARD_SIZE]
            for local_row, rec in enumerate(chunk_recs):
                vmeta = _sample_tungsten_vertex_v2_mm(rec['family_hi'], rec['family_lo'], sampler_spec)
                vx, vy, vz = vmeta['vx_mm'], vmeta['vy_mm'], vmeta['vz_mm']
                rec['vx'] = vx / 1000.0
                rec['vy'] = vy / 1000.0
                rec['vz'] = vz / 1000.0
                rec['run'] = 32100
                rec['chunk'] = int(shard_id)
                rec['row'] = int(local_row)
                rec['split'] = sp_name
                rec['slab_id'] = int(vmeta['slab_id'])
                rec['module_type'] = str(vmeta['module_type'])
                rec['block_copy_no'] = int(vmeta['block_copy_no'])
                rec['plate_in_block'] = int(vmeta['plate_in_block'])
                rec['is_pinpoint_module'] = bool(vmeta['is_pinpoint_module'])
                rec['is_front_pinpoint'] = bool(vmeta['is_front_pinpoint'])
                rec['is_interm_pinpoint'] = bool(vmeta['is_interm_pinpoint'])
                rec['is_pixel_xy_acceptance'] = bool(vmeta['is_pixel_xy_acceptance'])
                rec['sampling_stratum'] = int(vmeta['sampling_stratum'])
                rec['sampling_stratum_name'] = str(vmeta['sampling_stratum_name'])
                rec['geom_weight'] = float(vmeta['geom_weight'])
                manifest_rows.append({
                    'run': 32100,
                    'chunk': int(shard_id),
                    'row': int(local_row),
                    'family_hi': np.uint64(rec['family_hi']),
                    'family_lo': np.uint64(rec['family_lo']),
                    'split': sp_name,
                    'label': int(rec['label']),
                    'Ev_GeV': float(rec['Ev']),
                    'has_gen_charm': bool(rec['has_gen_charm']),
                    'initX': float(vx),
                    'initY': float(vy),
                    'initZ': float(vz),
                    'slab_id': int(vmeta['slab_id']),
                    'module_type': str(vmeta['module_type']),
                    'block_copy_no': int(vmeta['block_copy_no']),
                    'plate_in_block': int(vmeta['plate_in_block']),
                    'is_pinpoint_module': bool(vmeta['is_pinpoint_module']),
                    'is_front_pinpoint': bool(vmeta['is_front_pinpoint']),
                    'is_interm_pinpoint': bool(vmeta['is_interm_pinpoint']),
                    'is_pixel_xy_acceptance': bool(vmeta['is_pixel_xy_acceptance']),
                    'sampling_stratum': int(vmeta['sampling_stratum']),
                    'sampling_stratum_name': str(vmeta['sampling_stratum_name']),
                    'geom_weight': float(vmeta['geom_weight']),
                    'source_tag': rec['source_tag'],
                })
            pq.write_table(
                pa.Table.from_pylist(chunk_recs),
                shards_dir / f'input_{shard_id:03d}.parquet',
                compression=COMPRESSION,
            )
            shard_id += 1

    # Save paired 1,200-event Confirm subset input shard (300/class) for Geant4 physics systematics
    conf_recs = selected_by_split['confirm']
    sys_sub = []
    for cls_id in range(4):
        c_sub = [dict(r) for r in conf_recs if r['label'] == cls_id][:SYS_CONFIRM_PER_CLASS]
        sys_sub.extend(c_sub)
    pq.write_table(pa.Table.from_pylist(sys_sub), shards_dir / 'input_sys_confirm1200.parquet', compression=COMPRESSION)

    man_df = pd.DataFrame(manifest_rows)
    # Verify 100% family disjointness across all 5 splits!
    fam_sets = {
        sp: set(zip(sub['family_hi'].tolist(), sub['family_lo'].tolist()))
        for sp, sub in man_df.groupby('split')
    }
    sp_names = list(fam_sets.keys())
    for i in range(len(sp_names)):
        for j in range(i + 1, len(sp_names)):
            if fam_sets[sp_names[i]] & fam_sets[sp_names[j]]:
                raise AssertionError(f'Family overlap between {sp_names[i]} and {sp_names[j]}')

    man_df.to_parquet(root / 'planned_manifest.parquet', index=False, compression=COMPRESSION)
    # Pre-create all shard output directories up-front to prevent EOS FUSE concurrent mkdir cache races
    for sub_d in ('data/shards', 'truth_sidecar/parts', 'validation_meta/parts', 'sealed_truth_meta/parts'):
        (root / sub_d).mkdir(parents=True, exist_ok=True)
    for sid_pre in range(shard_id):
        (root / 'data/shards' / f'{sid_pre:03d}').mkdir(parents=True, exist_ok=True)
    prep_summary = {
        'status': 'prepared',
        'UTC': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': float(time.perf_counter() - t0),
        'total_events': len(man_df),
        'n_shards': int(shard_id),
        'shard_size': SHARD_SIZE,
        'sys_confirm_events': int(len(sys_sub)),
        'family_disjoint_verified': True,
        'stratum_volume_fractions': list(sampler_spec['volume_fractions']),
        'stratum_sampling_probs': list(sampler_spec['sampling_probs']),
        'stratum_geom_weights': list(sampler_spec['geom_weights']),
        'pinpoint_module_count': int(man_df['is_pinpoint_module'].sum()),
        'pinpoint_pix_xy_count': int((man_df['is_pinpoint_module'] & man_df['is_pixel_xy_acceptance']).sum()),
        'fortune_module_count': int((~man_df['is_pinpoint_module']).sum()),
        'mean_geom_weight': float(man_df['geom_weight'].mean()),
        'class_stats': class_stats,
        'geometry_json': str(geom_path),
    }
    write_json(root / 'prepare_summary.json', prep_summary)
    return prep_summary


def _simulate_subchunk(task: tuple[list[dict], str, int, dict[str, str], float]) -> Path:
    records, scratch_str, sub_idx, env_extra, field_t = task
    sub_dir = Path(scratch_str) / f'sub_{sub_idx:02d}'
    sub_dir.mkdir(parents=True, exist_ok=True)
    in_root = sub_dir / 'in.gfaser.root'
    out_root = sub_dir / 'out_g4.root'
    mac = sub_dir / 'run.mac'

    write_gfaser(records, in_root)

    seed_base = 20263180 + int(records[0].get('chunk', 0)) * 37 + sub_idx * 7
    mac.write_text('\n'.join([
        '/control/verbose 0',
        '/run/verbose 0',
        f'/control/execute {GEOM_MACRO}',
        '/det/enableFaserSpectrometer true',
        '/gen/select gfaser',
        f'/gen/gfaser/inputFile {in_root}',
        '/gen/gfaser/hitPixelArea false',
        f'/out/fileName {out_root}',
        '/out/saveTrack false',
        '/out/saveTruthHits false',
        f'/random/setSeeds {seed_base} {seed_base + 19}',
        '/run/initialize',
        f'/Pinpoint/field/value {field_t} tesla',
        f'/run/beamOn {len(records)}',
        '',
    ]))
    g4_env = {
        'PINPOINT_USE_INPUT_Z': '1',
        'PINPOINT_SKIP_GDML': '1',
        'PINPOINT_COMPACT_TRUTH': '1',
        **env_extra,
    }
    _run_lcg107([str(PINPOINT_FIXED_BIN), str(mac)], env_extra=g4_env, cwd=sub_dir)
    in_root.unlink(missing_ok=True)
    return out_root


def _charm_max_flight_length_mm(cev: dict | None, charm_ids: np.ndarray, vtx: np.ndarray) -> float:
    """Compute max 3D flight length (mm) across primary charmed hadrons from G4 charm decay tree."""
    if not cev or len(charm_ids) == 0:
        return -1.0
    pids = np.asarray(cev['PID'], dtype=np.int64)
    prod = np.column_stack([cev['ProdX'], cev['ProdY'], cev['ProdZ']]).astype(np.float64)
    best = -1.0
    for cid in charm_ids.tolist():
        m = pids == int(cid)
        if not np.any(m):
            continue
        pts = prod[m]
        # Decay daughters share the terminal (max-z) production point of the charmed hadron;
        # exclude delta-rays produced earlier along the trajectory.
        z_max = float(pts[:, 2].max())
        term = pts[np.abs(pts[:, 2] - z_max) < 0.05]
        dv = np.median(term, axis=0) if len(term) else pts[int(np.argmax(pts[:, 2]))]
        d3 = float(np.linalg.norm(dv - vtx))
        if d3 > best:
            best = d3
    return best


def _convert_subchunks_to_shard(
    root_files: list[Path],
    records: list[dict],
    geometry: dict,
    bounds: dict,
    shard_out_dir: Path,
    sidecar_part_path: Path | None,
    sealed_meta_part_path: Path | None,
    *,
    include_truth_edep: bool = False,
) -> dict:
    observations = []
    targets = []
    prov_rows = []
    sc_rows = []
    sealed_rows = []

    obs_cfg = {'pe_only': True, 'apply_sipm_saturation': False, 'invert_sipm_saturation': True}
    nx_pix = int(bounds['nx'])
    ny_pix = int(bounds['ny'])

    rec_cursor = 0
    for rf in root_files:
        with uproot.open(rf, object_cache=None, array_cache=None) as src:
            ev_arr = src['event'].arrays(['evtID', 'initX', 'initY', 'initZ', 'initE', 'initPDG', 'fslPDG'], library='np')
            data = {}
            for n, k in BRANCHES.items():
                tree = src[ROOT_TREES[n]]
                cols = list(k)
                if n == 'bar':
                    if not include_truth_edep:
                        cols = [c for c in cols if c != 'edep']
                    if 'hitTime' in tree.keys():
                        cols.append('hitTime')
                data[n] = tree.arrays(cols, library='ak')
            prim_map = group_flat(src['primaries'], ['evtID', 'PDG', 'trackID', 'Px', 'Py', 'Pz'])
            tau_map = group_flat(src['tau'], ['evtID', 'PID', 'TID', 'PDG', 'ParentPDG', 'ProdX', 'ProdY', 'ProdZ', 'Px', 'Py', 'Pz']) if 'tau' in src else {}
            charm_map = group_flat(src['charm'], ['evtID', 'PID', 'TID', 'PDG', 'ParentPDG', 'ProdX', 'ProdY', 'ProdZ', 'Px', 'Py', 'Pz']) if 'charm' in src else {}
            n_sub = len(ev_arr['evtID'])

            for i in range(n_sub):
                rec = records[rec_cursor]
                rec_cursor += 1
                evt_id = int(ev_arr['evtID'][i])
                lbl4 = int(rec['label'])
                hi = int(rec['family_hi'])
                lo = int(rec['family_lo'])

                h = {n: {k: ak.to_numpy(v[k][i]) for k in v.fields} for n, v in data.items()}
                pix = pixel_channels(h['pixel'], geometry)
                bars, _ = bar_channels(h['bar'], geometry, include_truth_edep=include_truth_edep)
                seed = hi ^ lo ^ 20261119
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

                vx = float(ev_arr['initX'][i])
                vy = float(ev_arr['initY'][i])
                vz = float(ev_arr['initZ'][i])
                vtx_vec = np.array([vx, vy, vz], dtype=np.float64)
                init_e = float(ev_arr['initE'][i])
                init_pdg = int(ev_arr['initPDG'][i])
                fsl_pdg = int(ev_arr['fslPDG'][i])

                prov_rows.append({
                    'run': int(rec['run']),
                    'chunk': int(rec['chunk']),
                    'row': int(rec['row']),
                    'evtID': evt_id,
                    'family_hi': np.uint64(hi),
                    'family_lo': np.uint64(lo),
                    'split': str(rec['split']),
                    'label': lbl4,
                })

                dom = compute_observable_containment(obs_dict, geometry, bounds, config=obs_cfg)
                pcid_arr = np.asarray(pix['channel_id'], dtype=np.int64)
                p0_pix_hits = int(np.count_nonzero((pcid_arr // (nx_pix * ny_pix)) == 0)) if len(pcid_arr) else 0

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
                tau_l3d = -1.0
                tau_dz = -1.0
                if lbl4 == 2 and len(tau_ids) > 0:
                    direct, decay_xyz, decay_valid = primary_decay(tev, tau_ids)
                    branch = classify_tau_daughters(direct['PDG']) if direct else 'unknown'
                    if decay_xyz is not None:
                        tau_l3d = float(np.linalg.norm(np.asarray(decay_xyz, dtype=np.float64) - vtx_vec))
                        tau_dz = float(decay_xyz[2] - vz)
                else:
                    direct, decay_valid = {}, False
                    branch = 'not_tau'

                charm_l3d = _charm_max_flight_length_mm(charm_map.get(evt_id), charm_ids, vtx_vec)

                if sidecar_part_path is not None:
                    l7, tb_val = _compute_label_7(lbl4, fsl_pdg, init_pdg, branch)
                    kin_vec, kv = _compute_vis_kinematics(lbl4, fsl_pdg, pev, direct, decay_valid)
                    sc_rows.append({
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
                        'vertex_valid': np.uint8(int(dom['obs_domain_id'] != 0)),
                        'vis_pz_log1p': np.float32(kin_vec[0]),
                        'vis_pt_log1p': np.float32(kin_vec[1]),
                        'vis_cos_dphi': np.float32(kin_vec[2]),
                        'vis_sin_dphi': np.float32(kin_vec[3]),
                        'kin_valid': np.uint8(kv),
                        'energy_nu_GeV': np.float32(init_e / 1000.0),
                        'geom_weight': np.float32(rec.get('geom_weight', 1.0)),
                    })
                if sealed_meta_part_path is not None:
                    is_pin = bool(rec.get('is_pinpoint_module', False))
                    is_front_pin = bool(rec.get('is_front_pinpoint', False))
                    is_interm_pin = bool(rec.get('is_interm_pinpoint', False))
                    in_pix_xy = bool(rec.get('is_pixel_xy_acceptance', (-132.5 <= vx <= 132.5) and (-109.5 <= vy <= 85.5)))
                    # Exact physical definition: true PinpointBlock slabs inside silicon pixel XY acceptance
                    pix_cent = bool(is_pin and in_pix_xy)
                    front_pix_cent = bool(is_front_pin and in_pix_xy)
                    interm_pix_cent = bool(is_interm_pin and in_pix_xy)
                    lg_scint = (-133.0 <= vx <= 297.0) and (-222.0 <= vy <= 208.0) and (vz <= bounds['z_85'])
                    sealed_rows.append({
                        'run': np.int32(rec['run']),
                        'chunk': np.int32(rec['chunk']),
                        'row': np.int32(rec['row']),
                        'family_hi': np.uint64(hi),
                        'family_lo': np.uint64(lo),
                        'split': str(rec['split']),
                        'label': np.int64(lbl4),
                        'obs_domain_id': np.int64(dom['obs_domain_id']),
                        'p0_pixel_hits': np.int32(p0_pix_hits),
                        'upstream_veto_flag': bool(dom['c_obs'][6] > 0.5),
                        'slab_id': np.int32(rec.get('slab_id', -1)),
                        'module_type': str(rec.get('module_type', 'external')),
                        'block_copy_no': np.int32(rec.get('block_copy_no', -1)),
                        'plate_in_block': np.int32(rec.get('plate_in_block', -1)),
                        'is_pinpoint_module': bool(is_pin),
                        'is_front_pinpoint': bool(is_front_pin),
                        'is_interm_pinpoint': bool(is_interm_pin),
                        'is_pixel_xy_acceptance': bool(in_pix_xy),
                        'sampling_stratum': np.int32(rec.get('sampling_stratum', -1)),
                        'sampling_stratum_name': str(rec.get('sampling_stratum_name', 'external')),
                        'geom_weight': np.float64(rec.get('geom_weight', 1.0)),
                        'pixel_centered': bool(pix_cent),
                        'front_pinpoint_pixel_centered': bool(front_pix_cent),
                        'interm_pinpoint_pixel_centered': bool(interm_pix_cent),
                        'fortune_module': bool(rec.get('slab_id', -1) >= 0 and not is_pin),
                        'large_scintillator': bool(lg_scint),
                        'is_E_le_2p3TeV': bool((init_e / 1000.0) <= 2312.0),
                        'is_highE_non_tau': bool(((init_e / 1000.0) > 2312.0) and (lbl4 != 2)),
                        'tau_branch': branch,
                        'has_charm': bool(has_ch),
                        'n_charm_hadrons': np.int32(len(charm_ids)),
                        'tau_decay_L3d_mm': np.float32(tau_l3d),
                        'tau_decay_dz_mm': np.float32(tau_dz),
                        'charm_decay_L3d_mm': np.float32(charm_l3d),
                        'vertex_x_mm': np.float32(vx),
                        'vertex_y_mm': np.float32(vy),
                        'vertex_z_mm': np.float32(vz),
                        'energy_nu_GeV': np.float32(init_e / 1000.0),
                    })
        rf.unlink(missing_ok=True)

    shard_out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(observations), shard_out_dir / 'observations.parquet', compression=COMPRESSION, row_group_size=64)
    pq.write_table(pa.Table.from_pylist(targets), shard_out_dir / 'targets.parquet', compression=COMPRESSION)
    pd.DataFrame(prov_rows).to_parquet(shard_out_dir / 'provenance.parquet', index=False, compression=COMPRESSION)
    write_json(shard_out_dir / 'geometry.json', geometry)
    write_json(shard_out_dir / 'summary.json', {'events': len(observations), 'include_truth_edep': bool(include_truth_edep), 'UTC': datetime.now(timezone.utc).isoformat()})

    if sidecar_part_path is not None and sc_rows:
        sidecar_part_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(sc_rows).to_parquet(sidecar_part_path, index=False, compression=COMPRESSION)
    if sealed_meta_part_path is not None and sealed_rows:
        sealed_meta_part_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(sealed_rows).to_parquet(sealed_meta_part_path, index=False, compression=COMPRESSION)
    return {'events': len(observations)}


def run_worker(root: Path, shard_start: int, shard_count: int, cpus: int = 8, sys_variant: str | None = None) -> dict:
    t0 = time.perf_counter()
    geom = json.loads((root / 'geometry.json').read_text())
    bounds = detector_bounds(geom)
    scratch_base = Path(os.environ.get('PINPOINT_V3_SCRATCH', '/tmp'))
    scratch_base.mkdir(parents=True, exist_ok=True)

    if sys_variant:
        sys_specs = {
            'tau_helicity_0': ({'PINPOINT_TAU_HELICITY': '0'}, 0.57, 32101),
            'w_density_0p98': ({'PINPOINT_W_DENSITY_SCALE': '0.98'}, 0.57, 32102),
            'no_magnet_shells': ({'PINPOINT_DISABLE_MAGNET_SHELLS': '1'}, 0.57, 32103),
            'field_0p97': ({}, 0.57 * 0.97, 32104),
            'nominal_sys_ref': ({}, 0.57, 32100),
        }
        env_ex, field_t, sys_run = sys_specs[sys_variant]
        out_dir = root / 'sys_shards' / sys_variant / 'shards' / '000'
        if (out_dir / 'observations.parquet').exists():
            return {'status': 'already_done', 'sys_variant': sys_variant}
        sys_in = root / 'input_shards' / 'input_sys_confirm1200.parquet'
        if not sys_in.exists():
            sys_in = root / 'input_shards' / 'input_sys_confirm600.parquet'
        recs = pq.read_table(sys_in).to_pylist()
        for idx_r, r in enumerate(recs):
            r['run'] = sys_run
            r['chunk'] = 0
            r['row'] = idx_r
            r['split'] = 'confirm'
        with tempfile.TemporaryDirectory(prefix=f'clean-sys-{sys_variant}-', dir=scratch_base) as tmp:
            step = max(1, (len(recs) + cpus - 1) // cpus)
            tasks = [(recs[s : s + step], tmp, k, env_ex, field_t) for k, s in enumerate(range(0, len(recs), step))]
            with ProcessPoolExecutor(max_workers=cpus) as pool:
                rfs = list(pool.map(_simulate_subchunk, tasks))
            _convert_subchunks_to_shard(
                rfs, recs, geom, bounds, out_dir, None,
                root / 'sys_shards' / sys_variant / 'truth_meta.parquet',
                include_truth_edep=False,
            )
            write_json(root / 'sys_shards' / sys_variant / 'plan.json', {'geometry': str(root / 'geometry.json')})
        return {'status': 'complete', 'sys_variant': sys_variant, 'events': len(recs), 'seconds': float(time.perf_counter() - t0)}

    done = []
    for sid in range(shard_start, shard_start + shard_count):
        in_pq = root / 'input_shards' / f'input_{sid:03d}.parquet'
        if not in_pq.exists():
            continue
        shard_dir = root / 'data' / 'shards' / f'{sid:03d}'
        if (shard_dir / 'observations.parquet').exists():
            done.append(sid)
            continue
        recs = pq.read_table(in_pq).to_pylist()
        sp = str(recs[0]['split'])
        sc_part = (root / 'truth_sidecar' / 'parts' / f'event_{sid:03d}.parquet') if sp in ('train', 'validation') else None
        sealed_part = (root / 'sealed_truth_meta' / 'parts' / f'meta_{sid:03d}.parquet') if sp not in ('train', 'validation') else None
        val_meta_part = (root / 'validation_meta' / 'parts' / f'meta_{sid:03d}.parquet') if sp == 'validation' else sealed_part

        with tempfile.TemporaryDirectory(prefix=f'clean-s{sid:03d}-', dir=scratch_base) as tmp:
            step = max(1, (len(recs) + cpus - 1) // cpus)
            tasks = [(recs[s : s + step], tmp, k, {}, 0.57) for k, s in enumerate(range(0, len(recs), step))]
            with ProcessPoolExecutor(max_workers=cpus) as pool:
                rfs = list(pool.map(_simulate_subchunk, tasks))
            _convert_subchunks_to_shard(rfs, recs, geom, bounds, shard_dir, sc_part, val_meta_part, include_truth_edep=False)
        done.append(sid)

    return {'status': 'complete', 'shards': done, 'seconds': float(time.perf_counter() - t0)}


SYS_VARIANTS = ('nominal_sys_ref', 'tau_helicity_0', 'w_density_0p98', 'no_magnet_shells', 'field_0p97')


def submit_workers(root: Path, shards_per_job: int = 2, cpus: int = 8) -> dict:
    from analysis.flavour.v2.submit import choose_schedd
    from analysis.flavour.v3.common.submit import _package_source

    root = root.resolve()
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    (root / 'code').mkdir(parents=True, exist_ok=True)
    (root / 'runtime').mkdir(parents=True, exist_ok=True)

    prep = json.loads((root / 'prepare_summary.json').read_text())
    n_shards = int(prep['n_shards'])
    n_shard_tasks = (n_shards + shards_per_job - 1) // shards_per_job
    n_total_tasks = n_shard_tasks + len(SYS_VARIANTS)

    repo = Path(__file__).resolve().parents[5]
    tag_name = root.parent.name if root.parent.name else 'clean_v2'
    archive = root / 'code' / f'{tag_name}_sim.tar.gz'
    _package_source(repo, archive)

    chosen, snapshot, checks = choose_schedd(root)
    condor_dir = Path.home() / f'condor/Pinpoint/{tag_name}_sim'
    condor_dir.mkdir(parents=True, exist_ok=True)

    wrapper = condor_dir / 'run_worker.sh'
    wrapper.write_text('\n'.join([
        '#!/bin/bash',
        'set -o pipefail',
        'which() { command -v "$@"; }; export -f which',
        'source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh >/dev/null 2>&1',
        'set -e',
        'ls /eos/home-x/xcheng/FASER/Pinpoint/results >/dev/null',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1',
        f'export PINPOINT_V3_SCRATCH="${{_CONDOR_SCRATCH_DIR:-/tmp}}/{tag_name}_${{PINPOINT_JOB_ID}}"',
        'mkdir -p "$PINPOINT_V3_SCRATCH/code"',
        f'tar -xzf {shlex.quote(str(archive))} -C "$PINPOINT_V3_SCRATCH/code"',
        'export PYTHONPATH="$PINPOINT_V3_SCRATCH/code/src:${PYTHONPATH:-}"',
        'cd "$PINPOINT_V3_SCRATCH"',
        f'exec python3 -m analysis.flavour.v3.v6_clean_sim.produce_clean_sim worker '
        f'--root {shlex.quote(str(root))} --task-id "$1" --shards-per-job {shards_per_job} --cpus {cpus}',
        '',
    ]))
    wrapper.chmod(0o755)

    sub_path = condor_dir / 'array.sub'
    sub_path.write_text('\n'.join([
        'universe = vanilla',
        f'executable = {wrapper}',
        'arguments = $(Process)',
        f'initialdir = {condor_dir}',
        f'output = {root}/logs/worker.$(ClusterId).$(Process).out',
        f'error = {root}/logs/worker.$(ClusterId).$(Process).err',
        f'log = {root}/logs/worker.$(ClusterId).log',
        'environment = "PINPOINT_JOB_ID=$(ClusterId).$(Process)"',
        f'request_cpus = {cpus}',
        'request_memory = 16GB',
        'request_disk = 16GB',
        'requirements = (OpSysAndVer == "AlmaLinux9")',
        '+JobFlavour = "longlunch"',
        'should_transfer_files = YES',
        'when_to_transfer_output = ON_EXIT',
        'transfer_output_files = ""',
        f'queue {n_total_tasks}',
        '',
    ]))
    resp = subprocess.run(['condor_submit', '-name', chosen, str(sub_path)], capture_output=True, text=True, check=True)
    rec = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'schedd': chosen,
        'compatibility_checks': checks,
        'n_shard_tasks': n_shard_tasks,
        'n_sys_tasks': len(SYS_VARIANTS),
        'n_total_tasks': n_total_tasks,
        'cpus_per_task': cpus,
        'archive': str(archive),
        'submit_file': str(sub_path),
        'stdout': resp.stdout,
    }
    write_json(root / 'runtime' / f'{tag_name}_worker_array.submission.json', rec)
    return rec


def finalize_clean_sim(root: Path) -> dict:
    prep = json.loads((root / 'prepare_summary.json').read_text())
    n_shards = int(prep['n_shards'])
    missing = [sid for sid in range(n_shards) if not (root / 'data' / 'shards' / f'{sid:03d}' / 'observations.parquet').exists()]
    if missing:
        raise RuntimeError(f'Cannot finalize clean_sim; {len(missing)} shards still incomplete: {missing[:10]}')

    # 1. Write data/plan.json & tune_screen.parquet (2,000 validation events)
    write_json(root / 'data' / 'plan.json', {
        'stage': 'Stage321_CleanV2',
        'geometry': str(root / 'geometry.json'),
        'n_shards': n_shards,
        'include_truth_edep': False,
        'Cal_A_Cut_B_Confirm_payload_read': False,
    })

    val_provs = [
        pd.read_parquet(root / 'data' / 'shards' / f'{sid:03d}' / 'provenance.parquet')
        for sid in range(n_shards)
    ]
    all_prov = pd.concat(val_provs, ignore_index=True)
    val_df = all_prov[all_prov['split'] == 'validation'].reset_index(drop=True)
    rng = np.random.default_rng(20261118)
    scr_idx = np.sort(rng.choice(len(val_df), size=min(2000, len(val_df)), replace=False))
    val_df.iloc[scr_idx][KEY + ['label']].to_parquet(root / 'data' / 'tune_screen.parquet', index=False, compression=COMPRESSION)

    # 2. Merge train/validation event_sidecar.parquet cleanly & remove parts
    sc_parts = sorted((root / 'truth_sidecar' / 'parts').glob('event_*.parquet'))
    sc_df = pd.concat([pd.read_parquet(p) for p in sc_parts], ignore_index=True)
    sc_df.to_parquet(root / 'truth_sidecar' / 'event_sidecar.parquet', index=False, compression=COMPRESSION)
    shutil.rmtree(root / 'truth_sidecar' / 'parts', ignore_errors=True)

    # 3. Merge validation_meta (Tune8k) cleanly
    vm_parts = sorted((root / 'validation_meta' / 'parts').glob('meta_*.parquet'))
    if vm_parts:
        vm_df = pd.concat([pd.read_parquet(p) for p in vm_parts], ignore_index=True)
        vm_df.to_parquet(root / 'validation_meta' / 'tune_truth_meta.parquet', index=False, compression=COMPRESSION)
        shutil.rmtree(root / 'validation_meta' / 'parts', ignore_errors=True)

    # 4. Merge sealed_truth_meta per split (kept outside data/ and truth_sidecar/ until Stage 319 one-shot unsealing)
    sm_parts = sorted((root / 'sealed_truth_meta' / 'parts').glob('meta_*.parquet'))
    if sm_parts:
        sm_df = pd.concat([pd.read_parquet(p) for p in sm_parts], ignore_index=True)
        for sp in ('cal_a', 'cut_b', 'confirm'):
            sub = sm_df[sm_df['split'] == sp].sort_values(KEY).reset_index(drop=True)
            sub.to_parquet(root / 'sealed_truth_meta' / f'{sp}_truth_meta.parquet', index=False, compression=COMPRESSION)
        shutil.rmtree(root / 'sealed_truth_meta' / 'parts', ignore_errors=True)

    # Remove transient input_shards to keep EOS compact per AGENTS.md
    shutil.rmtree(root / 'input_shards', ignore_errors=True)

    summary = {
        'status': 'complete',
        'UTC': datetime.now(timezone.utc).isoformat(),
        'total_events': len(all_prov),
        'split_counts': all_prov.groupby('split').size().to_dict(),
        'train_tune_sidecar_rows': len(sc_df),
    }
    write_json(root / 'finalize_summary.json', summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'worker', 'submit-workers', 'finalize'))
    parser.add_argument('--root', type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument('--task-id', type=int, default=None)
    parser.add_argument('--shard-start', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--shards-per-job', type=int, default=3)
    parser.add_argument('--cpus', type=int, default=8)
    parser.add_argument('--sys-variant', type=str, default=None)
    args = parser.parse_args()

    if args.command == 'prepare':
        out = prepare_clean_sim(args.root, workers=args.cpus)
    elif args.command == 'worker':
        if args.task_id is not None:
            prep = json.loads((args.root / 'prepare_summary.json').read_text())
            n_shards = int(prep['n_shards'])
            n_shard_tasks = (n_shards + args.shards_per_job - 1) // args.shards_per_job
            if args.task_id < n_shard_tasks:
                s0 = args.task_id * args.shards_per_job
                scnt = min(args.shards_per_job, n_shards - s0)
                out = run_worker(args.root, s0, scnt, cpus=args.cpus, sys_variant=None)
            else:
                sv = SYS_VARIANTS[args.task_id - n_shard_tasks]
                out = run_worker(args.root, 0, 0, cpus=args.cpus, sys_variant=sv)
        else:
            out = run_worker(args.root, args.shard_start, args.shard_count, cpus=args.cpus, sys_variant=args.sys_variant)
    elif args.command == 'submit-workers':
        out = submit_workers(args.root, shards_per_job=args.shards_per_job, cpus=args.cpus)
    else:
        out = finalize_clean_sim(args.root)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
