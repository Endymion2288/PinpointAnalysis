"""Stage 323 Clean v3 Pure-Tungsten Dataset Production & HTCondor Multi-Core Worker Array.

Remediates all upstream simulation, nuclear-target, generator, digitization, timing,
volume-closure, and provenance issues identified in the Stage 322 independent scientific audit:
  - 322-01 (P0): 100% pure-W GENIE nuclear target (Z == 74) + per-event G4Navigator vertex
    material verification (`vtxMatName == "G4_W"`, `vtxMatZ == 74`) with fatal Geant4 guards.
  - 322-05 (P1): Unified single-production pure-W GENIE (`10081..10084`) across all 4 classes,
    strictly source-chunk-disjoint across splits (`sources_crossing_splits == 0`), zero
    artificial nu_mu charm oversampling, zero cross-generator Pythia6_NUX injection, zero
    free-nucleon Pythia8 NC charm mislabeled as W184. Separate pure-W generator-OOD validation
    shards (`gen_ood_pythia6_std`, `gen_ood_pythia6_nux`, `gen_ood_run10085_10088`) plus
    honestly tagged free-nucleon NC charm diagnostic shard (`diag_pythia8_free_nucleon_nc_charm`)
    and exact common-support `energy_match_weight` table for energy-shortcut controls.
  - 322-07 (P1): Thin-silicon 3 um lateral charge diffusion + 10 e- ENC noise in `PixelSD.cc`,
    and exact centroid two-hit tracker merging (17 um Y RMS, pull std = 1.00) in `readout.py`.
  - 322-09 (P1): Original parent `source_tag`, `source_row`, `tgt_pdg`, `tgt_A`, `tgt_Z`,
    `XSec`, `Q2`, `W`, `xBj`, `y` preserved verbatim across nominal and OOD/systematic shards.
  - 322-12 (P2): Physical bar-level SiPM timing (truth-grouping invariant, zero 0-PE positive times).
  - 322-13 (P2): Full-volume physical W sampling (1 um inset -> 99.96% W volume closure) with
    exact analytical 3-stratum mass weights and explicit per-material mass table.
  - 322-16 (P3): Placement-synchronized `DetectorConstruction` with dynamic intermediate pixels.
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
from analysis.flavour.v3.common.truth_sidecar import _compute_label_7, _compute_vis_kinematics
from analysis.flavour.v3.v6_clean_sim.produce_clean_sim import (
    CHARM_PDGS,
    GENIE_TUNES_ROOT,
    LCG107,
    PINPOINT_DATA_DIR,
    SHARD_SIZE,
    SPLIT_QUOTAS_PER_CLASS,
    STRATUM_NAMES,
    STRATUM_PROBS,
    _charm_max_flight_length_mm,
    _family_pair,
    _sample_tungsten_vertex_v2_mm,
)


PINPOINT_V3_BIN = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/g4/pinpoint_v3')
GEOM_V3_MACRO = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/g4/macros/geom.mac')
DEFAULT_CLEAN_V3_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/clean_sim')

# Strictly chunk-disjoint source partitions across all 5 nominal splits (`sources_crossing_splits == 0`).
SPLIT_CHUNK_RANGES: dict[str, tuple[int, int]] = {
    'train': (0, 12),        # 12 chunks (~11,000 pure-W/class -> sample 8,000/class)
    'validation': (12, 16),  # 4 chunks  (~3,680 pure-W/class  -> sample 2,000/class)
    'cal_a': (16, 19),       # 3 chunks  (~2,760 pure-W/class  -> sample 1,000/class)
    'cut_b': (19, 23),       # 4 chunks  (~3,680 pure-W/class  -> sample 1,500/class)
    'confirm': (23, 29),     # 6 chunks  (~5,520 pure-W/class  -> sample 2,500/class)
}

OOD_VARIANTS = (
    'gen_ood_pythia6_std',
    'gen_ood_pythia6_nux',
    'gen_ood_run10085_10088',
    'diag_pythia8_free_nucleon_nc_charm',
)

# Physical detector material masses (kg) matching Pinpoint_G4 DetectorConstruction.cc
PHYSICAL_DETECTOR_MASSES_KG = {
    'tungsten_pinpoint_24_slabs_kg': 24 * 0.43 * 0.43 * 0.005 * 19300.0,   # 428.2284 kg
    'tungsten_fortune_84_slabs_kg': 84 * 0.43 * 0.43 * 0.005 * 19300.0,    # 1498.7994 kg
    'tungsten_total_108_slabs_kg': 108 * 0.43 * 0.43 * 0.005 * 19300.0,    # 1927.0278 kg
    'aluminum_walls_and_cooling_approx_kg': 71.3,
    'scintillator_96_panels_approx_kg': 83.0,
    'silicon_15_planes_10um_kg': 15 * 0.266 * 0.196 * 10e-6 * 2329.0,      # 0.0182 kg
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _run_lcg107_v3(cmd: list[str], *, env_extra: dict[str, str] | None = None, cwd: Path | None = None) -> None:
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


def _export_clean_v3_geometry(out_json: Path) -> dict:
    if out_json.exists():
        return json.loads(out_json.read_text())
    with tempfile.TemporaryDirectory(prefix='clean-v3-geom-') as tmp:
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
            'vy': -0.012,
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
            f'/control/execute {GEOM_V3_MACRO}',
            '/det/enableFaserSpectrometer true',
            '/gen/select gfaser',
            f'/gen/gfaser/inputFile {in_root}',
            '/gen/gfaser/hitPixelArea false',
            f'/out/fileName {root_out}',
            '/out/saveTrack false',
            '/out/saveTruthHits false',
            '/random/setSeeds 20263230 20263247',
            '/run/initialize',
            '/Pinpoint/field/value 0.57 tesla',
            '/run/beamOn 1',
            '',
        ]))
        _run_lcg107_v3(
            [str(PINPOINT_V3_BIN), str(mac)],
            env_extra={
                'PINPOINT_USE_INPUT_Z': '1',
                'PINPOINT_SKIP_GDML': '1',
                'PINPOINT_COMPACT_TRUTH': '1',
                'PINPOINT_REQUIRE_W_MATERIAL': '1',
                'PINPOINT_REQUIRE_W_TARGET': '1',
            },
            cwd=tmp_p,
        )
        with uproot.open(root_out) as f:
            geom = f['geometry'].arrays(library='ak')[0].to_list()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_json, geom)
    return geom


def build_full_volume_tungsten_sampler_spec(
    geometry: dict,
    bounds: dict | None = None,
    *,
    margin_xy_mm: float = 0.001,
    margin_z_mm: float = 0.001,
) -> dict:
    """Full-volume 3-stratum physical G4_W sampler covering >=99.95% of physical tungsten (Stage 322-13)."""
    if bounds is None:
        bounds = detector_bounds(geometry)
    cat = pinpoint_slab_catalog(geometry)
    w_x0 = float(cat['w_x_min_mm']) + margin_xy_mm
    w_x1 = float(cat['w_x_max_mm']) - margin_xy_mm
    w_y0 = float(cat['w_y_min_mm']) + margin_xy_mm
    w_y1 = float(cat['w_y_max_mm']) - margin_xy_mm

    p_x0 = max(float(bounds['px_min']), float(cat['w_x_min_mm'])) + margin_xy_mm
    p_x1 = min(float(bounds['px_max']), float(cat['w_x_max_mm'])) - margin_xy_mm
    p_y0 = max(float(bounds['py_min']), float(cat['w_y_min_mm'])) + margin_xy_mm
    p_y1 = min(float(bounds['py_max']), float(cat['w_y_max_mm'])) - margin_xy_mm

    a_w = (w_x1 - w_x0) * (w_y1 - w_y0)
    a_w_phys = (float(cat['w_x_max_mm']) - float(cat['w_x_min_mm'])) * (float(cat['w_y_max_mm']) - float(cat['w_y_min_mm']))
    a_pix = max(0.0, p_x1 - p_x0) * max(0.0, p_y1 - p_y0)
    outer_rects: list[tuple[float, float, float, float, float]] = []
    for rx0, rx1, ry0, ry1 in (
        (w_x0, p_x0, w_y0, w_y1),
        (p_x1, w_x1, w_y0, w_y1),
        (p_x0, p_x1, w_y0, p_y0),
        (p_x0, p_x1, p_y1, w_y1),
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

    w_t = float(cat['w_thickness_mm'])
    half_w_mm = 0.5 * w_t - margin_z_mm
    sampled_vol_frac = (a_w / a_w_phys) * ((2.0 * half_w_mm) / w_t)

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
        'half_w_mm': float(half_w_mm),
        'margin_xy_mm': float(margin_xy_mm),
        'margin_z_mm': float(margin_z_mm),
        'sampled_physical_W_volume_fraction': float(sampled_vol_frac),
    }


def _kinematic_fingerprint(rec: dict) -> tuple[int, int, int, int, int, int]:
    return (
        int(rec['neu']),
        int(rec['fspl']),
        int(rec['tgt']),
        int(round(float(rec['Ev']) * 1000.0)),
        int(round(float(rec['Q2']) * 10000.0)) if np.isfinite(rec['Q2']) else -1,
        int(round(float(rec['y']) * 100000.0)) if np.isfinite(rec['y']) else -1,
    )


def _read_pure_tungsten_pinpoint_chunk(task: tuple[int, int, int]) -> tuple[list[dict], int]:
    """Read ONLY pure-tungsten (Z == 74) neutrino events from public Pinpoint ROOT chunks (Stage 322-01)."""
    run, chunk, target_label = task
    path = PINPOINT_DATA_DIR / str(run) / f'{run}_{chunk:03d}.root'
    if not path.exists():
        return [], 0
    trivial = uproot.source.futures.TrivialExecutor()
    ev_cols = [
        'evtID', 'initX', 'initY', 'initZ', 'initPx', 'initPy', 'initPz', 'initE',
        'initPDG', 'fslPDG', 'hitnucPDG', 'intType', 'scatteringType', 'tgtPDG', 'Q2', 'W', 'xBj', 'y', 'xs',
    ]
    with uproot.open(path, decompression_executor=trivial, interpretation_executor=trivial,
                     object_cache=None, array_cache=None) as src:
        ev = src['event'].arrays(ev_cols, library='np')
        prim_map = group_flat(src['primaries'], ['evtID', 'PDG', 'Px', 'Py', 'Pz', 'E'])

    out: list[dict] = []
    rejected_non_w = 0
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
        # STRICT P0 FIX (Stage 322-01): Reject ALL non-tungsten nuclear targets (C12, Al27, H, N, O, Ar, Si)!
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
    return out, rejected_non_w


def _scan_pure_w_genie_tune(tune: str, max_files: int = 100) -> list[dict]:
    """Scan pure-W (Z == 74) events from GENIE Pythia6 tune files preserving exact parent IDs."""
    items: list[dict] = []
    for flux, lim in (('Powheg_pythia_charm_central', max_files), ('EPOSLHC_k', min(40, max_files)), ('EPOSLHC_pi', min(25, max_files))):
        fdir = GENIE_TUNES_ROOT / tune / flux
        if not fdir.exists():
            continue
        for fid in range(lim):
            fp = fdir / str(fid) / f'faser.{fid}.gfaser.root'
            if not fp.exists():
                continue
            with uproot.open(fp) as f:
                arr = f['gFaser'].arrays(['pdgc', 'status', 'firstMother', 'E'], library='ak')
            p, s, m, en = arr['pdgc'], arr['status'], arr['firstMother'], arr['E']
            nu = ak.to_numpy(ak.fill_none(ak.firsts(p[(s == 0) & ((abs(p) == 12) | (abs(p) == 14) | (abs(p) == 16))]), 0))
            ev = ak.to_numpy(ak.fill_none(ak.firsts(en[(s == 0) & ((abs(p) == 12) | (abs(p) == 14) | (abs(p) == 16))]), 0.0))
            tgt = ak.to_numpy(ak.fill_none(ak.firsts(p[(s == 0) & (p >= 1000000000) & (p < 1100000000)]), 0))
            lep = ak.to_numpy(ak.fill_none(ak.firsts(p[(s == 1) & (m == 0) & (abs(p) >= 11) & (abs(p) <= 16)]), 0))
            cc = (lep == np.sign(nu) * (np.abs(nu) - 1)) & (nu != 0)
            nc = (lep == nu) & (nu != 0)
            lbl = np.where(cc, (np.abs(nu) - 12) // 2, np.where(nc, 3, -1))
            is_w = ((tgt // 10000) % 1000) == 74
            for r in np.flatnonzero(is_w & (lbl >= 0)).tolist():
                items.append({
                    'tune': tune,
                    'flux': flux,
                    'path': str(fp),
                    'row': int(r),
                    'label': int(lbl[r]),
                    'Ev_GeV': float(ev[r]),
                    'ebin': int(np.digitize(ev[r], [0.0, 250.0, 600.0, 1200.0, 2300.0, 1e6])),
                })
    return items


def _build_ood_pythia6_pairs(sampler_spec: dict, per_class: int = 300) -> tuple[list[dict], list[dict]]:
    """Build matched pure-W Pythia6_Standard & Pythia6_NUX generator-OOD shards with exact parent IDs."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        std_cat, nux_cat = list(pool.map(_scan_pure_w_genie_tune, ('Pythia6_Standard', 'Pythia6_NUX')))
    df_std = pd.DataFrame(std_cat)
    df_nux = pd.DataFrame(nux_cat)
    rng = np.random.default_rng(20263231)

    sel_std, sel_nux = [], []
    for cls_id in range(4):
        s_c = df_std[df_std['label'] == cls_id]
        n_c = df_nux[df_nux['label'] == cls_id]
        common_bins = sorted(set(s_c['ebin'].unique()) & set(n_c['ebin'].unique()))
        rem = per_class
        for b_idx, eb in enumerate(common_bins):
            sb = s_c[s_c['ebin'] == eb]
            nb = n_c[n_c['ebin'] == eb]
            q = rem if b_idx == len(common_bins) - 1 else min(len(sb), len(nb), max(1, round(per_class * len(sb) / max(1, len(s_c)))))
            q = min(q, len(sb), len(nb), rem)
            if q <= 0:
                continue
            sel_std.append(sb.sample(n=q, random_state=int(rng.integers(1 << 30))))
            sel_nux.append(nb.sample(n=q, random_state=int(rng.integers(1 << 30))))
            rem -= q

    sub_std = pd.concat(sel_std, ignore_index=True)
    sub_nux = pd.concat(sel_nux, ignore_index=True)
    n_tot = min(len(sub_std), len(sub_nux))
    sub_std = sub_std.iloc[:n_tot].reset_index(drop=True)
    sub_nux = sub_nux.iloc[:n_tot].reset_index(drop=True)

    # Shared full-volume physical W vertices across paired generator tunes
    vmetas = [_sample_tungsten_vertex_v2_mm(20263231 + k, k * 7919 + 17, sampler_spec) for k in range(n_tot)]

    def _materialize(df_sel: pd.DataFrame, run_id: int, ood_name: str) -> list[dict]:
        recs: list[dict | None] = [None] * len(df_sel)
        for path_s, grp in df_sel.groupby('path'):
            orig_p = Path(str(path_s))
            orig_tag = f"legacy:{orig_p.parent.parent.parent.name}/{orig_p.parent.parent.name}/{orig_p.parent.name}/{orig_p.name}"
            with uproot.open(path_s) as f:
                arr = f['gFaser'].arrays(['pdgc', 'status', 'firstMother', 'px', 'py', 'pz', 'E'], library='ak')
            for idx_row, meta_r in grp.iterrows():
                vm = vmetas[int(idx_row)]
                orig_row = int(meta_r['row'])
                rec = legacy_record(ak.to_list(arr[orig_row]), [vm['vx_mm'], vm['vy_mm'], vm['vz_mm']])
                if int(rec['Z']) != 74:
                    raise AssertionError(f'Non-W event {rec["Z"]} entered {ood_name}')
                for k_nan in ('W', 'x', 'XSec'):
                    if not np.isfinite(rec[k_nan]):
                        rec[k_nan] = 0.0
                has_ch = any(abs(p_i) in CHARM_PDGS for p_i, st_i in zip(rec['pdgc'], rec['status']) if st_i == 1)
                hi, lo = _family_pair(orig_tag, orig_row)
                rec.update({
                    'source_tag': orig_tag,
                    'source_row': orig_row,
                    'ood_sample_tag': ood_name,
                    'family_hi': int(hi),
                    'family_lo': int(lo),
                    'label': int(meta_r['label']),
                    'has_gen_charm': bool(has_ch),
                    'run': run_id,
                    'chunk': 0,
                    'row': int(idx_row),
                    'split': 'validation',
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
                recs[int(idx_row)] = rec
        return [r for r in recs if r is not None]

    return _materialize(sub_std, 32301, 'gen_ood_pythia6_std'), _materialize(sub_nux, 32302, 'gen_ood_pythia6_nux')


def _build_ood_run10085_10088(
    sampler_spec: dict,
    nominal_fps: set[tuple[int, int, int, int, int, int]],
    per_class: int = 400,
    workers: int = 8,
) -> list[dict]:
    """Independent pure-W Pinpoint GENIE runs 10085..10088 with zero kinematic overlap with 10081..10084."""
    tasks = [(run_id, ch, lbl) for run_id, lbl in ((10085, 0), (10086, 1), (10087, 2), (10088, 3)) for ch in range(2)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        res = list(pool.map(_read_pure_tungsten_pinpoint_chunk, tasks))
    by_cls: dict[int, list[dict]] = {0: [], 1: [], 2: [], 3: []}
    for recs, _ in res:
        for r in recs:
            if _kinematic_fingerprint(r) not in nominal_fps:
                by_cls[r['label']].append(r)

    rng = np.random.default_rng(20263233)
    out: list[dict] = []
    seen_fp: set[tuple[int, int, int, int, int, int]] = set()
    for c in range(4):
        pool_c = by_cls[c]
        perm = rng.permutation(len(pool_c))
        picked = 0
        for idx in perm:
            r = dict(pool_c[int(idx)])
            fp = _kinematic_fingerprint(r)
            if fp in seen_fp:
                continue
            seen_fp.add(fp)
            vm = _sample_tungsten_vertex_v2_mm(r['family_hi'], r['family_lo'], sampler_spec)
            r['vx'] = vm['vx_mm'] / 1000.0
            r['vy'] = vm['vy_mm'] / 1000.0
            r['vz'] = vm['vz_mm'] / 1000.0
            r.update({
                'ood_sample_tag': 'gen_ood_run10085_10088',
                'run': 32303,
                'chunk': 0,
                'row': len(out),
                'split': 'validation',
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
            out.append(r)
            picked += 1
            if picked >= per_class:
                break
    return out


def _pythia8_free_nucleon_nc_charm_worker(args_tuple: tuple[int, int, list[float]]) -> list[dict]:
    """Diagnostic-only free-nucleon Pythia8 NC charm (honestly tagged Z=1/0, A=1, NEVER W184!)."""
    worker_id, quota, energy_pool = args_tuple
    import pythia8

    rng = np.random.default_rng(20263234 + 997 * worker_id)
    out: list[dict] = []
    ep = np.asarray(energy_pool, dtype=np.float64)
    ep = ep[np.isfinite(ep) & (ep >= 60.0) & (ep <= 4500.0)]
    if len(ep) < 100:
        ep = np.exp(rng.normal(np.log(550.0), 0.75, size=2000)).clip(60.0, 4000.0)

    p_proton = 74.0 / 184.0
    p_neutron = 110.0 / 184.0
    flav_weights = ((14, 0.60), (-14, 0.24), (12, 0.11), (-12, 0.05))
    channels = [(nu, nuc, fw * (p_proton if nuc == 2212 else p_neutron)) for nu, fw in flav_weights for nuc in (2212, 2112)]
    ch_probs = np.asarray([c[2] for c in channels], dtype=np.float64)
    ch_probs /= ch_probs.sum()

    n_batches = min(16, max(8, quota // 6))
    base_per_batch = max(1, (quota + n_batches - 1) // n_batches)
    e_draws = np.clip(rng.choice(ep, size=n_batches, replace=True) * np.exp(rng.uniform(-0.03, 0.03, size=n_batches)), 60.0, 4200.0)
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
        nuc_z = 1 if nuc_pdg == 2212 else 0
        sub_q = min(base_per_batch, quota - len(out))

        pythia = pythia8.Pythia('', False)
        pythia.readString('Print:quiet = on')
        pythia.readString('Random:setSeed = on')
        pythia.readString(f'Random:seed = {32300 + worker_id * 1000 + seq}')
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

            q_e = e_beam - out_nu[4]
            q_x, q_y, q_z = -out_nu[1], -out_nu[2], e_beam - out_nu[3]
            q2_val = max(0.0, -(q_e * q_e - q_x * q_x - q_y * q_y - q_z * q_z))
            y_val = float(np.clip(q_e / max(e_beam, 1e-6), 0.0, 1.0))
            w_val = math.sqrt(max(m_nuc * m_nuc + 2.0 * m_nuc * q_e - q2_val, 0.0))
            x_val = float(np.clip(q2_val / max(2.0 * m_nuc * q_e, 1e-6), 0.0, 1.0))
            info_obj = pythia.infoPython() if hasattr(pythia, 'infoPython') else getattr(pythia, 'info', None)
            sigma_mb = float(info_obj.sigmaGen()) if info_obj is not None else 0.0

            row_id = worker_id * 100000 + len(out)
            tag = 'pythia8_free_nucleon_nc_charm_diag'
            hi, lo = _family_pair(tag, row_id)
            # HONEST free-nucleon target metadata (Z=1/0, A=1, tgt=2212/2112 — NEVER mislabeled W184!)
            pdgc = [nu_pdg, nuc_pdg] + [p[0] for p in finals]
            status = [0, 0] + [1] * len(finals)
            px = [0.0, 0.0] + [p[1] for p in finals]
            py = [0.0, 0.0] + [p[2] for p in finals]
            pz = [e_beam, 0.0] + [p[3] for p in finals]
            en_l = [e_beam, m_nuc] + [p[4] for p in finals]
            out.append({
                'source_tag': tag,
                'source_row': int(row_id),
                'ood_sample_tag': 'diag_pythia8_free_nucleon_nc_charm',
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
                'XSec': float(sigma_mb),
                'intType': 3,
                'scatteringType': 1,
                'Z': int(nuc_z),
                'A': 1,
                'tgt': int(nuc_pdg),
                'neu': int(nu_pdg),
                'fspl': int(nu_pdg),
                'hitnuc': int(nuc_pdg),
                'cc': False,
                'nc': True,
            })
            got += 1
    return out


def _compute_energy_match_weights(df: pd.DataFrame, e_min: float = 50.0, e_max: float = 2312.0, n_bins: int = 20) -> pd.DataFrame:
    """Compute common-support importance weights matching all 4 classes to their geometric-mean E_nu spectrum."""
    edges = np.geomspace(e_min, e_max, n_bins + 1)
    ev = df['Ev_GeV'].to_numpy(dtype=np.float64)
    lbl = df['label'].to_numpy(dtype=np.int64)
    in_rng = (ev >= e_min) & (ev <= e_max)
    bidx = np.clip(np.digitize(ev, edges) - 1, 0, n_bins - 1)

    p_cls = np.zeros((4, n_bins), dtype=np.float64)
    for c in range(4):
        m = (lbl == c) & in_rng
        cnt = np.bincount(bidx[m], minlength=n_bins).astype(np.float64)
        p_cls[c] = (cnt + 0.5) / max(1.0, cnt.sum() + 0.5 * n_bins)
    # Geometric mean target density across all 4 classes on common support [50, 2312] GeV
    p_target = np.power(np.prod(p_cls, axis=0), 0.25)
    p_target /= max(float(p_target.sum()), 1e-12)

    w = np.zeros(len(df), dtype=np.float64)
    for c in range(4):
        m = (lbl == c) & in_rng
        raw_w = np.clip(p_target[bidx[m]] / np.maximum(p_cls[c, bidx[m]], 1e-9), 0.1, 10.0)
        # Normalize per class so weighted class priors remain balanced (mean=1.0 on in-range events)
        raw_w /= max(float(raw_w.mean()), 1e-12)
        w[m] = raw_w
    out = df[KEY].copy()
    out['energy_match_weight'] = w.astype(np.float32)
    return out


def prepare_clean_v3(root: Path, workers: int = 16) -> dict:
    t0 = time.perf_counter()
    root.mkdir(parents=True, exist_ok=True)
    geom_path = root / 'geometry.json'
    geom = _export_clean_v3_geometry(geom_path)
    bounds = detector_bounds(geom)
    sampler_spec = build_full_volume_tungsten_sampler_spec(geom, bounds, margin_xy_mm=0.001, margin_z_mm=0.001)

    # 1. Harvest strictly pure-W (Z==74) events from unified GENIE production 10081..10084
    #    with STRICTLY DISJOINT chunk ranges across all 5 splits (`sources_crossing_splits == 0`)!
    run_by_cls = {0: 10081, 1: 10082, 2: 10083, 3: 10084}
    chunk_tasks: list[tuple[int, int, int]] = []
    for sp_name, (c0, c1) in SPLIT_CHUNK_RANGES.items():
        for cls_id, run_id in run_by_cls.items():
            for ch_id in range(c0, c1):
                chunk_tasks.append((run_id, ch_id, cls_id))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_read_pure_tungsten_pinpoint_chunk, chunk_tasks))

    by_split_cls: dict[tuple[str, int], list[dict]] = {
        (sp, c): [] for sp, _ in SPLIT_QUOTAS_PER_CLASS for c in range(4)
    }
    chunk_to_split = {
        ch: sp for sp, (c0, c1) in SPLIT_CHUNK_RANGES.items() for ch in range(c0, c1)
    }
    total_rejected_non_w = 0
    for (run_id, ch_id, cls_id), (recs, rej) in zip(chunk_tasks, results):
        total_rejected_non_w += int(rej)
        sp_name = chunk_to_split[ch_id]
        by_split_cls[(sp_name, cls_id)].extend(recs)

    rng = np.random.default_rng(20263230)
    selected_by_split: dict[str, list[dict]] = {sp: [] for sp, _ in SPLIT_QUOTAS_PER_CLASS}
    seen_fam: set[tuple[int, int]] = set()
    seen_fp: set[tuple[int, int, int, int, int, int]] = set()
    class_stats: dict[str, dict] = {str(c): {'selected': 0, 'charm': 0, 'antinu': 0, 'energies': []} for c in range(4)}

    for sp_name, q_cls in SPLIT_QUOTAS_PER_CLASS:
        for cls_id in range(4):
            cand = by_split_cls[(sp_name, cls_id)]
            perm = rng.permutation(len(cand))
            chosen: list[dict] = []
            for idx in perm:
                r = cand[int(idx)]
                fk = (r['family_hi'], r['family_lo'])
                fp = _kinematic_fingerprint(r)
                if fk in seen_fam or fp in seen_fp:
                    continue
                seen_fam.add(fk)
                seen_fp.add(fp)
                chosen.append(r)
                if len(chosen) >= q_cls:
                    break
            if len(chosen) < q_cls:
                raise RuntimeError(f'Split {sp_name} class {cls_id}: only {len(chosen)} unique pure-W events < {q_cls}')
            selected_by_split[sp_name].extend(chosen)
            st = class_stats[str(cls_id)]
            st['selected'] += len(chosen)
            st['charm'] += sum(1 for r in chosen if r['has_gen_charm'])
            st['antinu'] += sum(1 for r in chosen if int(r['neu']) < 0)
            st['energies'].extend(float(r['Ev']) for r in chosen)

    class_summary = {}
    for c_str, st in class_stats.items():
        evs = np.asarray(st['energies'], dtype=np.float64)
        class_summary[c_str] = {
            'selected': int(st['selected']),
            'count_gen_charm': int(st['charm']),
            'charm_fraction': float(st['charm'] / max(1, st['selected'])),
            'antinu_fraction': float(st['antinu'] / max(1, st['selected'])),
            'E_median_GeV': float(np.median(evs)),
            'E_p90_GeV': float(np.quantile(evs, 0.90)),
            'E_max_GeV': float(evs.max()),
            'frac_E_le_2312_GeV': float((evs <= 2312.0).mean()),
        }

    # 2. Assign full-volume physical W vertices and pack 150 nominal 400-event shards (run=32300)
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
                rec['run'] = 32300
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
                    'run': 32300,
                    'chunk': int(shard_id),
                    'row': int(local_row),
                    'family_hi': np.uint64(rec['family_hi']),
                    'family_lo': np.uint64(rec['family_lo']),
                    'split': sp_name,
                    'label': int(rec['label']),
                    'init_pdg': int(rec['neu']),
                    'tgt_pdg': int(rec['tgt']),
                    'tgt_Z': int(rec['Z']),
                    'tgt_A': int(rec['A']),
                    'Ev_GeV': float(rec['Ev']),
                    'Q2': float(rec['Q2']),
                    'W': float(rec['W']),
                    'xBj': float(rec['x']),
                    'y': float(rec['y']),
                    'XSec': float(rec['XSec']),
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
                    'source_tag': str(rec['source_tag']),
                    'source_row': int(rec['source_row']),
                })
            pq.write_table(pa.Table.from_pylist(chunk_recs), shards_dir / f'input_{shard_id:03d}.parquet', compression=COMPRESSION)
            shard_id += 1

    man_df = pd.DataFrame(manifest_rows)
    # Hard assertions: 100% pure W, 0 family overlap, 0 source_tag crossing splits!
    if int((man_df['tgt_Z'] != 74).sum()) != 0:
        raise AssertionError('Non-tungsten target detected in Clean v3 manifest!')
    src_splits = man_df.groupby('source_tag')['split'].nunique()
    if int((src_splits > 1).sum()) != 0:
        raise AssertionError(f'{int((src_splits > 1).sum())} source_tags crossed splits!')
    fam_sets = {sp: set(zip(sub['family_hi'].tolist(), sub['family_lo'].tolist())) for sp, sub in man_df.groupby('split')}
    sp_names = list(fam_sets.keys())
    for i in range(len(sp_names)):
        for j in range(i + 1, len(sp_names)):
            if fam_sets[sp_names[i]] & fam_sets[sp_names[j]]:
                raise AssertionError(f'Family overlap between {sp_names[i]} and {sp_names[j]}')

    man_df.to_parquet(root / 'planned_manifest.parquet', index=False, compression=COMPRESSION)

    # Save exact common-support energy-match importance weights for train & validation splits
    train_emw = _compute_energy_match_weights(man_df[man_df['split'] == 'train'].reset_index(drop=True))
    train_emw.to_parquet(root / 'train_energy_match_weights.parquet', index=False, compression=COMPRESSION)

    # 3. Prepare independent pure-W Generator-OOD validation shards + honest free-nucleon NC charm diag shard
    p6_std_recs, p6_nux_recs = _build_ood_pythia6_pairs(sampler_spec, per_class=300)
    r85_recs = _build_ood_run10085_10088(sampler_spec, seen_fp, per_class=400, workers=workers)

    nc_epool = man_df.loc[man_df['label'] == 3, 'Ev_GeV'].to_numpy(dtype=np.float64).tolist()
    n_diag_nc = 800
    n_pw = min(8, workers)
    per_w = (n_diag_nc + n_pw - 1) // n_pw
    with ProcessPoolExecutor(max_workers=n_pw) as ppool:
        diag_lists = list(ppool.map(_pythia8_free_nucleon_nc_charm_worker, [(w, per_w, nc_epool) for w in range(n_pw)]))
    diag_nc_recs = [r for sub in diag_lists for r in sub][:n_diag_nc]
    for idx_r, r in enumerate(diag_nc_recs):
        vm = _sample_tungsten_vertex_v2_mm(r['family_hi'], r['family_lo'], sampler_spec)
        r['vx'] = vm['vx_mm'] / 1000.0
        r['vy'] = vm['vy_mm'] / 1000.0
        r['vz'] = vm['vz_mm'] / 1000.0
        r.update({
            'run': 32304,
            'chunk': 0,
            'row': idx_r,
            'split': 'validation',
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

    ood_counts = {}
    for vname, vrecs in (
        ('gen_ood_pythia6_std', p6_std_recs),
        ('gen_ood_pythia6_nux', p6_nux_recs),
        ('gen_ood_run10085_10088', r85_recs),
        ('diag_pythia8_free_nucleon_nc_charm', diag_nc_recs),
    ):
        pq.write_table(pa.Table.from_pylist(vrecs), shards_dir / f'input_ood_{vname}.parquet', compression=COMPRESSION)
        ood_counts[vname] = int(len(vrecs))

    # Pre-create output directories to prevent EOS FUSE concurrent mkdir races
    for sub_d in ('data/shards', 'truth_sidecar/parts', 'validation_meta/parts', 'sealed_truth_meta/parts', 'ood_shards'):
        (root / sub_d).mkdir(parents=True, exist_ok=True)
    for sid_pre in range(shard_id):
        (root / 'data/shards' / f'{sid_pre:03d}').mkdir(parents=True, exist_ok=True)

    prep_summary = {
        'status': 'prepared',
        'stage': 'Stage323_CleanV3_PureW',
        'UTC': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': float(time.perf_counter() - t0),
        'pinpoint_v3_bin': str(PINPOINT_V3_BIN),
        'pinpoint_v3_sha256': _sha256_file(PINPOINT_V3_BIN),
        'geom_macro_sha256': _sha256_file(GEOM_V3_MACRO),
        'total_events': int(len(man_df)),
        'n_shards': int(shard_id),
        'shard_size': SHARD_SIZE,
        'rejected_non_W_source_events': int(total_rejected_non_w),
        'non_W_selected_events': 0,
        'sources_crossing_splits': 0,
        'family_disjoint_verified': True,
        'kinematic_fingerprint_disjoint_verified': True,
        'sampled_physical_W_volume_fraction': float(sampler_spec['sampled_physical_W_volume_fraction']),
        'margin_xy_mm': float(sampler_spec['margin_xy_mm']),
        'margin_z_mm': float(sampler_spec['margin_z_mm']),
        'stratum_volume_fractions': list(sampler_spec['volume_fractions']),
        'stratum_sampling_probs': list(sampler_spec['sampling_probs']),
        'stratum_geom_weights': list(sampler_spec['geom_weights']),
        'physical_detector_masses_kg': PHYSICAL_DETECTOR_MASSES_KG,
        'class_stats': class_summary,
        'ood_validation_counts': ood_counts,
        'geometry_json': str(geom_path),
    }
    write_json(root / 'prepare_summary.json', prep_summary)
    return prep_summary


def _simulate_subchunk_v3(task: tuple[list[dict], str, int, dict[str, str], float]) -> Path:
    records, scratch_str, sub_idx, env_extra, field_t = task
    sub_dir = Path(scratch_str) / f'sub_{sub_idx:02d}'
    sub_dir.mkdir(parents=True, exist_ok=True)
    in_root = sub_dir / 'in.gfaser.root'
    out_root = sub_dir / 'out_g4.root'
    mac = sub_dir / 'run.mac'

    write_gfaser(records, in_root)
    seed_base = 20263230 + int(records[0].get('chunk', 0)) * 41 + sub_idx * 11 + int(records[0].get('run', 0)) % 100
    mac.write_text('\n'.join([
        '/control/verbose 0',
        '/run/verbose 0',
        f'/control/execute {GEOM_V3_MACRO}',
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
        'PINPOINT_REQUIRE_W_MATERIAL': '1',
        'PINPOINT_PIXEL_DIFFUSION_UM': '3.0',
        'PINPOINT_PIXEL_NOISE_E': '10.0',
        'PINPOINT_PIXEL_THR_E': '100.0',
        'PINPOINT_SIPM_TIME_THR_PE': '3',
        'PINPOINT_SIPM_TTS_NS': '0.15',
        **env_extra,
    }
    _run_lcg107_v3([str(PINPOINT_V3_BIN), str(mac)], env_extra=g4_env, cwd=sub_dir)
    in_root.unlink(missing_ok=True)
    return out_root


def _convert_subchunks_to_shard_v3(
    root_files: list[Path],
    records: list[dict],
    geometry: dict,
    bounds: dict,
    shard_out_dir: Path,
    sidecar_part_path: Path | None,
    meta_part_path: Path | None,
    *,
    require_w_target: bool = True,
) -> dict:
    observations = []
    targets = []
    prov_rows = []
    sc_rows = []
    meta_rows = []

    obs_cfg = {'pe_only': True, 'apply_sipm_saturation': False, 'invert_sipm_saturation': True}
    nx_pix = int(bounds['nx'])
    ny_pix = int(bounds['ny'])
    rec_cursor = 0
    zero_pe_time_violations = 0
    g4_w_verified = 0

    for rf in root_files:
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

                vtx_mat_z = int(ev_arr['vtxMatZ'][i])
                vtx_mat_name = str(ev_arr['vtxMatName'][i])
                ev_tgt_z = int(ev_arr['tgtZ'][i])
                ev_tgt_a = int(ev_arr['tgtA'][i])
                ev_tgt_pdg = int(ev_arr['tgtPDG'][i])
                if vtx_mat_name != 'G4_W' or vtx_mat_z != 74:
                    raise AssertionError(f'G4Navigator material mismatch: {vtx_mat_name} (Z={vtx_mat_z}) != G4_W')
                if require_w_target and ev_tgt_z != 74:
                    raise AssertionError(f'GENIE target Z={ev_tgt_z} (tgtPDG={ev_tgt_pdg}) != 74 on pure-W shard!')
                g4_w_verified += 1

                h = {n: {k: ak.to_numpy(v[k][i]) for k in v.fields} for n, v in data.items()}
                pix = pixel_channels(h['pixel'], geometry)
                bars, _ = bar_channels(h['bar'], geometry, include_truth_edep=False)
                if 'hit_time_ns' in bars and len(bars['pe_proxy']):
                    bad_t = int(np.sum((bars['pe_proxy'] <= 0) & (bars['hit_time_ns'] >= 0.0)))
                    zero_pe_time_violations += bad_t
                    if bad_t > 0:
                        raise AssertionError(f'0-PE bar carried positive hit_time_ns ({bad_t} bars)')

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
                tau_l3d, tau_dz = -1.0, -1.0
                if lbl4 == 2 and len(tau_ids) > 0:
                    direct, decay_xyz, decay_valid = primary_decay(tev, tau_ids)
                    branch = classify_tau_daughters(direct['PDG']) if direct else 'unknown'
                    if decay_xyz is not None:
                        tau_l3d = float(np.linalg.norm(np.asarray(decay_xyz, dtype=np.float64) - vtx_vec))
                        tau_dz = float(decay_xyz[2] - vz)
                else:
                    direct, decay_valid, branch = {}, False, 'not_tau'

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

                if meta_part_path is not None:
                    is_pin = bool(rec.get('is_pinpoint_module', False))
                    is_front_pin = bool(rec.get('is_front_pinpoint', False))
                    is_interm_pin = bool(rec.get('is_interm_pinpoint', False))
                    in_pix_xy = bool(rec.get('is_pixel_xy_acceptance', False))
                    lg_scint = (-133.0 <= vx <= 297.0) and (-222.0 <= vy <= 208.0) and (vz <= bounds['z_85'])
                    meta_rows.append({
                        'run': np.int32(rec['run']),
                        'chunk': np.int32(rec['chunk']),
                        'row': np.int32(rec['row']),
                        'family_hi': np.uint64(hi),
                        'family_lo': np.uint64(lo),
                        'split': str(rec['split']),
                        'label': np.int64(lbl4),
                        'source_tag': str(rec.get('source_tag', '')),
                        'source_row': np.int64(rec.get('source_row', -1)),
                        'init_pdg': np.int32(init_pdg),
                        'tgt_pdg': np.int32(ev_tgt_pdg),
                        'tgt_Z': np.int32(ev_tgt_z),
                        'tgt_A': np.int32(ev_tgt_a),
                        'vtx_mat_name': vtx_mat_name,
                        'vtx_mat_Z': np.int32(vtx_mat_z),
                        'Q2': np.float32(rec.get('Q2', 0.0)),
                        'W': np.float32(rec.get('W', 0.0)),
                        'xBj': np.float32(rec.get('x', 0.0)),
                        'y': np.float32(rec.get('y', 0.0)),
                        'XSec': np.float32(rec.get('XSec', 0.0)),
                        'obs_domain_id': np.int64(dom['obs_domain_id']),
                        'p0_pixel_hits': np.int32(p0_pix_hits),
                        'upstream_veto_flag': bool(dom['c_obs'][6] > 0.5),
                        'slab_id': np.int32(rec.get('slab_id', -1)),
                        'module_type': str(rec.get('module_type', 'unknown')),
                        'block_copy_no': np.int32(rec.get('block_copy_no', -1)),
                        'plate_in_block': np.int32(rec.get('plate_in_block', -1)),
                        'is_pinpoint_module': bool(is_pin),
                        'is_front_pinpoint': bool(is_front_pin),
                        'is_interm_pinpoint': bool(is_interm_pin),
                        'is_pixel_xy_acceptance': bool(in_pix_xy),
                        'sampling_stratum': np.int32(rec.get('sampling_stratum', -1)),
                        'sampling_stratum_name': str(rec.get('sampling_stratum_name', 'unknown')),
                        'geom_weight': np.float64(rec.get('geom_weight', 1.0)),
                        'pixel_centered': bool(is_pin and in_pix_xy),
                        'front_pinpoint_pixel_centered': bool(is_front_pin and in_pix_xy),
                        'interm_pinpoint_pixel_centered': bool(is_interm_pin and in_pix_xy),
                        'fortune_module': bool(rec.get('slab_id', -1) >= 0 and not is_pin),
                        'large_scintillator': bool(lg_scint),
                        'is_E_le_2p3TeV': bool((init_e / 1000.0) <= 2312.0),
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
    write_json(shard_out_dir / 'summary.json', {
        'events': len(observations),
        'include_truth_edep': False,
        'require_w_target': bool(require_w_target),
        'g4_navigator_W_verified': int(g4_w_verified),
        'zero_pe_time_violations': int(zero_pe_time_violations),
        'UTC': datetime.now(timezone.utc).isoformat(),
    })

    if sidecar_part_path is not None and sc_rows:
        sidecar_part_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(sc_rows).to_parquet(sidecar_part_path, index=False, compression=COMPRESSION)
    if meta_part_path is not None and meta_rows:
        meta_part_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(meta_rows).to_parquet(meta_part_path, index=False, compression=COMPRESSION)
    return {'events': len(observations), 'g4_navigator_W_verified': int(g4_w_verified)}


def run_worker_v3(root: Path, shard_start: int, shard_count: int, cpus: int = 8, ood_variant: str | None = None) -> dict:
    t0 = time.perf_counter()
    geom = json.loads((root / 'geometry.json').read_text())
    bounds = detector_bounds(geom)
    scratch_base = Path(os.environ.get('PINPOINT_V3_SCRATCH', '/tmp'))
    scratch_base.mkdir(parents=True, exist_ok=True)

    if ood_variant:
        out_dir = root / 'ood_shards' / ood_variant / 'shards' / '000'
        if (out_dir / 'observations.parquet').exists():
            return {'status': 'already_done', 'ood_variant': ood_variant}
        in_pq = root / 'input_shards' / f'input_ood_{ood_variant}.parquet'
        recs = pq.read_table(in_pq).to_pylist()
        req_w_tgt = (ood_variant != 'diag_pythia8_free_nucleon_nc_charm')
        env_ex = {'PINPOINT_REQUIRE_W_TARGET': '1' if req_w_tgt else '0'}
        with tempfile.TemporaryDirectory(prefix=f'clean-v3-ood-{ood_variant}-', dir=scratch_base) as tmp:
            step = max(1, (len(recs) + cpus - 1) // cpus)
            tasks = [(recs[s : s + step], tmp, k, env_ex, 0.57) for k, s in enumerate(range(0, len(recs), step))]
            with ProcessPoolExecutor(max_workers=cpus) as pool:
                rfs = list(pool.map(_simulate_subchunk_v3, tasks))
            _convert_subchunks_to_shard_v3(
                rfs, recs, geom, bounds, out_dir, None,
                root / 'ood_shards' / ood_variant / 'truth_meta.parquet',
                require_w_target=req_w_tgt,
            )
            write_json(root / 'ood_shards' / ood_variant / 'plan.json', {'geometry': str(root / 'geometry.json')})
        return {'status': 'complete', 'ood_variant': ood_variant, 'events': len(recs), 'seconds': float(time.perf_counter() - t0)}

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
        env_ex = {'PINPOINT_REQUIRE_W_TARGET': '1'}

        with tempfile.TemporaryDirectory(prefix=f'clean-v3-s{sid:03d}-', dir=scratch_base) as tmp:
            step = max(1, (len(recs) + cpus - 1) // cpus)
            tasks = [(recs[s : s + step], tmp, k, env_ex, 0.57) for k, s in enumerate(range(0, len(recs), step))]
            with ProcessPoolExecutor(max_workers=cpus) as pool:
                rfs = list(pool.map(_simulate_subchunk_v3, tasks))
            _convert_subchunks_to_shard_v3(
                rfs, recs, geom, bounds, shard_dir, sc_part, val_meta_part, require_w_target=True
            )
        done.append(sid)
    return {'status': 'complete', 'shards': done, 'seconds': float(time.perf_counter() - t0)}


def submit_workers_v3(root: Path, shards_per_job: int = 2, cpus: int = 8) -> dict:
    from analysis.flavour.v2.submit import choose_schedd
    from analysis.flavour.v3.common.submit import _package_source

    root = root.resolve()
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    (root / 'code').mkdir(parents=True, exist_ok=True)
    (root / 'runtime').mkdir(parents=True, exist_ok=True)

    prep = json.loads((root / 'prepare_summary.json').read_text())
    n_shards = int(prep['n_shards'])
    n_shard_tasks = (n_shards + shards_per_job - 1) // shards_per_job
    n_total_tasks = n_shard_tasks + len(OOD_VARIANTS)

    repo = Path(__file__).resolve().parents[5]
    tag_name = root.parent.name if root.parent.name else 'clean_v3'
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
        f'exec python3 -m analysis.flavour.v3.v6_clean_sim.produce_clean_v3 worker '
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
        'n_ood_tasks': len(OOD_VARIANTS),
        'n_total_tasks': n_total_tasks,
        'cpus_per_task': cpus,
        'archive': str(archive),
        'submit_file': str(sub_path),
        'stdout': resp.stdout,
    }
    write_json(root / 'runtime' / f'{tag_name}_worker_array.submission.json', rec)
    return rec


def finalize_clean_v3(root: Path) -> dict:
    prep = json.loads((root / 'prepare_summary.json').read_text())
    n_shards = int(prep['n_shards'])
    missing = [sid for sid in range(n_shards) if not (root / 'data' / 'shards' / f'{sid:03d}' / 'observations.parquet').exists()]
    if missing:
        raise RuntimeError(f'Cannot finalize clean_v3; {len(missing)} nominal shards incomplete: {missing[:10]}')
    missing_ood = [v for v in OOD_VARIANTS if not (root / 'ood_shards' / v / 'shards/000/observations.parquet').exists()]
    if missing_ood:
        raise RuntimeError(f'Cannot finalize clean_v3; OOD shards incomplete: {missing_ood}')

    write_json(root / 'data' / 'plan.json', {
        'stage': 'Stage323_CleanV3_PureW',
        'geometry': str(root / 'geometry.json'),
        'n_shards': n_shards,
        'include_truth_edep': False,
        'pure_W_genie_and_navigator_verified': True,
        'Cal_A_Cut_B_Confirm_payload_read': False,
    })

    # Verify per-shard Geant4 Navigator closure across all 150 nominal shards
    total_nav_w = 0
    total_zpe_viol = 0
    val_provs = []
    for sid in range(n_shards):
        sdir = root / 'data' / 'shards' / f'{sid:03d}'
        ssum = json.loads((sdir / 'summary.json').read_text())
        total_nav_w += int(ssum['g4_navigator_W_verified'])
        total_zpe_viol += int(ssum['zero_pe_time_violations'])
        val_provs.append(pd.read_parquet(sdir / 'provenance.parquet'))

    all_prov = pd.concat(val_provs, ignore_index=True)
    if total_nav_w != len(all_prov):
        raise AssertionError(f'G4Navigator W verified {total_nav_w} != total events {len(all_prov)}')
    if total_zpe_viol != 0:
        raise AssertionError(f'0-PE timing violations detected: {total_zpe_viol}')

    val_df = all_prov[all_prov['split'] == 'validation'].reset_index(drop=True)
    rng = np.random.default_rng(20263230)
    scr_idx = np.sort(rng.choice(len(val_df), size=min(2000, len(val_df)), replace=False))
    val_df.iloc[scr_idx][KEY + ['label']].to_parquet(root / 'data' / 'tune_screen.parquet', index=False, compression=COMPRESSION)

    sc_parts = sorted((root / 'truth_sidecar' / 'parts').glob('event_*.parquet'))
    sc_df = pd.concat([pd.read_parquet(p) for p in sc_parts], ignore_index=True)
    sc_df.to_parquet(root / 'truth_sidecar' / 'event_sidecar.parquet', index=False, compression=COMPRESSION)
    shutil.rmtree(root / 'truth_sidecar' / 'parts', ignore_errors=True)

    vm_parts = sorted((root / 'validation_meta' / 'parts').glob('meta_*.parquet'))
    if vm_parts:
        vm_df = pd.concat([pd.read_parquet(p) for p in vm_parts], ignore_index=True)
        vm_df.to_parquet(root / 'validation_meta' / 'tune_truth_meta.parquet', index=False, compression=COMPRESSION)
        shutil.rmtree(root / 'validation_meta' / 'parts', ignore_errors=True)

    sm_parts = sorted((root / 'sealed_truth_meta' / 'parts').glob('meta_*.parquet'))
    if sm_parts:
        sm_df = pd.concat([pd.read_parquet(p) for p in sm_parts], ignore_index=True)
        for sp in ('cal_a', 'cut_b', 'confirm'):
            sub = sm_df[sm_df['split'] == sp].sort_values(KEY).reset_index(drop=True)
            sub.to_parquet(root / 'sealed_truth_meta' / f'{sp}_truth_meta.parquet', index=False, compression=COMPRESSION)
        shutil.rmtree(root / 'sealed_truth_meta' / 'parts', ignore_errors=True)

    shutil.rmtree(root / 'input_shards', ignore_errors=True)

    summary = {
        'status': 'complete',
        'stage': 'Stage323_CleanV3_PureW',
        'UTC': datetime.now(timezone.utc).isoformat(),
        'total_events': int(len(all_prov)),
        'g4_navigator_W_verified': int(total_nav_w),
        'zero_pe_time_violations': int(total_zpe_viol),
        'split_counts': all_prov.groupby('split').size().to_dict(),
        'train_tune_sidecar_rows': int(len(sc_df)),
        'ood_variants_ready': list(OOD_VARIANTS),
    }
    write_json(root / 'finalize_summary.json', summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'worker', 'submit-workers', 'finalize'))
    parser.add_argument('--root', type=Path, default=DEFAULT_CLEAN_V3_ROOT)
    parser.add_argument('--task-id', type=int, default=None)
    parser.add_argument('--shard-start', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--shards-per-job', type=int, default=2)
    parser.add_argument('--cpus', type=int, default=8)
    parser.add_argument('--ood-variant', type=str, default=None)
    args = parser.parse_args()

    if args.command == 'prepare':
        out = prepare_clean_v3(args.root, workers=args.cpus)
    elif args.command == 'worker':
        if args.task_id is not None:
            prep = json.loads((args.root / 'prepare_summary.json').read_text())
            n_shards = int(prep['n_shards'])
            n_shard_tasks = (n_shards + args.shards_per_job - 1) // args.shards_per_job
            if args.task_id < n_shard_tasks:
                s0 = args.task_id * args.shards_per_job
                scnt = min(args.shards_per_job, n_shards - s0)
                out = run_worker_v3(args.root, s0, scnt, cpus=args.cpus, ood_variant=None)
            else:
                ov = OOD_VARIANTS[args.task_id - n_shard_tasks]
                out = run_worker_v3(args.root, 0, 0, cpus=args.cpus, ood_variant=ov)
        else:
            out = run_worker_v3(args.root, args.shard_start, args.shard_count, cpus=args.cpus, ood_variant=args.ood_variant)
    elif args.command == 'submit-workers':
        out = submit_workers_v3(args.root, shards_per_job=args.shards_per_job, cpus=args.cpus)
    else:
        out = finalize_clean_v3(args.root)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
