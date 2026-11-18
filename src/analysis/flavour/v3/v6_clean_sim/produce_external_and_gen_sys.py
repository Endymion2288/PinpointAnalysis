"""Stage 318b: External Rock-Muon / Neutral-Hadron Punch-Through & Pythia6_NUX Generator Systematics.

Simulates two additional physical validation & systematic samples in Stage 315 `pinpoint_fixed` Geant4:
  1. `external_punchthrough` (600 events: 300 upstream rock muons PDG=+-13 at z=-3005 mm +
     300 upstream neutral hadrons n [2112] & K_L^0 [130] at z=-3005 mm entering unvetoed and
     showering inside tungsten) to quantify upstream veto rejection & neutral-hadron tau-tail leakage.
  2. Cell-matched `pythia6_std_ref` vs `pythia6_nux` (600 tungsten events each, identical vertex
     positions `(vx, vy, vz)` per cell across all 4 neutrino classes) to quantify GENIE hadronization
     systematic uncertainty in the Stage 319 27-bin profile likelihood.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import time

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from analysis.flavour.inventory import write_json
from analysis.flavour.s6.gfaser import legacy_record
from analysis.flavour.v3.common.domains import detector_bounds
from analysis.flavour.v3.v6_clean_sim.produce_clean_sim import (
    CHARM_PDGS,
    DEFAULT_CLEAN_ROOT,
    GENIE_TUNES_ROOT,
    _convert_subchunks_to_shard,
    _family_pair,
    _sample_tungsten_vertex_mm,
    _simulate_subchunk,
)


def _build_external_punchthrough_records(n_mu: int = 300, n_nhad: int = 300) -> list[dict]:
    rng = np.random.default_rng(20263185)
    recs = []
    # 1. Upstream Rock Muons (z = -3005 mm, entering upstream veto face)
    for i in range(n_mu):
        pdg = 13 if (i % 2 == 0) else -13
        e_gev = float(np.exp(rng.uniform(np.log(50.0), np.log(2000.0))))
        mass = 0.105658
        p_tot = math.sqrt(max(e_gev * e_gev - mass * mass, 1.0))
        tx = float(rng.normal(0.0, 0.003))
        ty = float(rng.normal(0.0, 0.003))
        pz = p_tot / math.sqrt(1.0 + tx * tx + ty * ty)
        px, py = pz * tx, pz * ty
        vx = float(rng.uniform(-125.0, 125.0))
        vy = float(rng.uniform(-98.0, 74.0))
        vz = -3005.0
        hi, lo = _family_pair('ext_rock_muon', i)
        recs.append({
            'source_tag': 'ext_rock_muon',
            'source_row': i,
            'ext_subtype': 'rock_muon',
            'family_hi': hi,
            'family_lo': lo,
            'label': 1,  # background non-tau (muon-like)
            'has_gen_charm': False,
            'run': 31905,
            'chunk': 0,
            'row': i,
            'split': 'confirm',
            'n': 3,
            'pdgc': [14 if pdg > 0 else -14, 1000741840, pdg],
            'status': [0, 0, 1],
            'name': ['', '', ''],
            'px': [0.0, 0.0, px],
            'py': [0.0, 0.0, py],
            'pz': [e_gev, 0.0, pz],
            'E': [e_gev, 171.4, e_gev],
            'vx': vx / 1000.0,
            'vy': vy / 1000.0,
            'vz': vz / 1000.0,
            'pxv': 0.0,
            'pyv': 0.0,
            'pzv': e_gev,
            'Ev': e_gev,
            'Q2': 0.0,
            'W': 0.0,
            'x': 0.0,
            'y': 0.0,
            'XSec': 0.0,
            'intType': 2,
            'scatteringType': 0,
            'Z': 74,
            'A': 184,
            'tgt': 1000741840,
            'neu': 14 if pdg > 0 else -14,
            'fspl': pdg,
            'hitnuc': 2112,
            'cc': False,
            'nc': True,
        })

    # 2. Upstream Neutral Hadrons (n=2112 [60%], K_L^0=130 [40%], z = -3005 mm, Q=0 bypasses charged veto!)
    for j in range(n_nhad):
        idx = n_mu + j
        is_n = (j % 5) < 3
        pdg = 2112 if is_n else 130
        mass = 0.939565 if is_n else 0.497611
        e_gev = float(np.exp(rng.uniform(np.log(25.0), np.log(600.0))))
        p_tot = math.sqrt(max(e_gev * e_gev - mass * mass, 1.0))
        tx = float(rng.normal(0.0, 0.005))
        ty = float(rng.normal(0.0, 0.005))
        pz = p_tot / math.sqrt(1.0 + tx * tx + ty * ty)
        px, py = pz * tx, pz * ty
        vx = float(rng.uniform(-120.0, 120.0))
        vy = float(rng.uniform(-92.0, 70.0))
        vz = -3005.0
        hi, lo = _family_pair('ext_neutral_hadron', j)
        recs.append({
            'source_tag': 'ext_neutral_hadron',
            'source_row': j,
            'ext_subtype': 'neutron' if is_n else 'k0l',
            'family_hi': hi,
            'family_lo': lo,
            'label': 3,  # background non-tau (neutral hadron shower)
            'has_gen_charm': False,
            'run': 31905,
            'chunk': 0,
            'row': idx,
            'split': 'confirm',
            'n': 3,
            'pdgc': [14, 1000741840, pdg],
            'status': [0, 0, 1],
            'name': ['', '', ''],
            'px': [0.0, 0.0, px],
            'py': [0.0, 0.0, py],
            'pz': [e_gev, 0.0, pz],
            'E': [e_gev, 171.4, e_gev],
            'vx': vx / 1000.0,
            'vy': vy / 1000.0,
            'vz': vz / 1000.0,
            'pxv': 0.0,
            'pyv': 0.0,
            'pzv': e_gev,
            'Ev': e_gev,
            'Q2': 0.0,
            'W': 0.0,
            'x': 0.0,
            'y': 0.0,
            'XSec': 0.0,
            'intType': 3,
            'scatteringType': 0,
            'Z': 74,
            'A': 184,
            'tgt': 1000741840,
            'neu': 14,
            'fspl': 14,
            'hitnuc': 2112,
            'cc': False,
            'nc': True,
        })
    return recs


def _scan_tune_tungsten(tune: str, max_files: int = 35) -> list[dict]:
    items = []
    for flux in ('Powheg_pythia_charm_central', 'EPOSLHC_k', 'EPOSLHC_pi'):
        fdir = GENIE_TUNES_ROOT / tune / flux
        if not fdir.exists():
            continue
        for fid in range(max_files):
            fp = fdir / str(fid) / f'faser.{fid}.gfaser.root'
            if not fp.exists():
                continue
            with uproot.open(fp) as f:
                arr = f['gFaser'].arrays(['pdgc', 'status', 'firstMother', 'px', 'py', 'pz', 'E'], library='ak')
            p, s, m, en = arr['pdgc'], arr['status'], arr['firstMother'], arr['E']
            nu = ak.to_numpy(ak.fill_none(ak.firsts(p[(s == 0) & ((abs(p) == 12) | (abs(p) == 14) | (abs(p) == 16))]), 0))
            ev = ak.to_numpy(ak.fill_none(ak.firsts(en[(s == 0) & ((abs(p) == 12) | (abs(p) == 14) | (abs(p) == 16))]), 0.0))
            tgt = ak.to_numpy(ak.fill_none(ak.firsts(p[(s == 0) & (p >= 1000000000) & (p < 1100000000)]), 0))
            lep = ak.to_numpy(ak.fill_none(ak.firsts(p[(s == 1) & (m == 0) & (abs(p) >= 11) & (abs(p) <= 16)]), 0))
            cc = lep == np.sign(nu) * (np.abs(nu) - 1)
            nc = lep == nu
            lbl = np.where(cc, (np.abs(nu) - 12) // 2, np.where(nc, 3, -1))
            keep = (((tgt // 10000) % 1000) == 74) & (lbl >= 0)
            for r in np.flatnonzero(keep).tolist():
                items.append({
                    'tune': tune,
                    'flux': flux,
                    'path': str(fp),
                    'row': int(r),
                    'label': int(lbl[r]),
                    'ebin': int(np.digitize(ev[r], [0.0, 250.0, 600.0, 1200.0, 2300.0, 1e6])),
                })
    return items


def _build_matched_generator_records(w_z: np.ndarray, w_t: float, per_class: int = 150) -> tuple[list[dict], list[dict]]:
    with ThreadPoolExecutor(max_workers=2) as pool:
        std_cat, nux_cat = list(pool.map(_scan_tune_tungsten, ('Pythia6_Standard', 'Pythia6_NUX')))
    df_std = pd.DataFrame(std_cat)
    df_nux = pd.DataFrame(nux_cat)
    rng = np.random.default_rng(20263186)

    sel_std, sel_nux = [], []
    for cls_id in range(4):
        s_c = df_std[df_std['label'] == cls_id]
        n_c = df_nux[df_nux['label'] == cls_id]
        common_bins = sorted(set(s_c['ebin'].unique()) & set(n_c['ebin'].unique()))
        rem = per_class
        for b_idx, eb in enumerate(common_bins):
            sb = s_c[s_c['ebin'] == eb]
            nb = n_c[n_c['ebin'] == eb]
            q = rem if b_idx == len(common_bins) - 1 else min(len(sb), len(nb), max(1, round(per_class * len(sb) / len(s_c))))
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

    # Shared cell-matched vertices in tungsten!
    vtxs = [_sample_tungsten_vertex_mm(20263186 + k, k * 7919, w_z, w_t) for k in range(n_tot)]

    def _materialize(df_sel: pd.DataFrame, run_id: int, tag_prefix: str) -> list[dict]:
        recs: list[dict | None] = [None] * len(df_sel)
        for path_s, grp in df_sel.groupby('path'):
            with uproot.open(path_s) as f:
                arr = f['gFaser'].arrays(['pdgc', 'status', 'firstMother', 'px', 'py', 'pz', 'E'], library='ak')
            for idx_row, meta_r in grp.iterrows():
                vx, vy, vz = vtxs[int(idx_row)]
                rec = legacy_record(ak.to_list(arr[int(meta_r['row'])]), [vx, vy, vz])
                for k_nan in ('W', 'x', 'XSec'):
                    if not np.isfinite(rec[k_nan]):
                        rec[k_nan] = 0.0
                has_ch = any(abs(p_i) in CHARM_PDGS for p_i, st_i in zip(rec['pdgc'], rec['status']) if st_i == 1)
                hi, lo = _family_pair(tag_prefix, int(idx_row))
                rec.update({
                    'source_tag': tag_prefix,
                    'source_row': int(idx_row),
                    'family_hi': hi,
                    'family_lo': lo,
                    'label': int(meta_r['label']),
                    'has_gen_charm': bool(has_ch),
                    'run': run_id,
                    'chunk': 0,
                    'row': int(idx_row),
                    'split': 'confirm',
                })
                recs[int(idx_row)] = rec
        return [r for r in recs if r is not None]

    return _materialize(sub_std, 31907, 'gen_pythia6_std'), _materialize(sub_nux, 31906, 'gen_pythia6_nux')


def run_all_extra_sys(clean_root: Path, cpus: int = 16) -> dict:
    t0 = time.perf_counter()
    geom = json.loads((clean_root / 'geometry.json').read_text())
    bounds = detector_bounds(geom)
    w_z = np.asarray(geom['tungsten_Zpos'], dtype=np.float64)
    w_t = float(geom['tungsten_thickness'])

    ext_recs = _build_external_punchthrough_records(300, 300)
    std_recs, nux_recs = _build_matched_generator_records(w_z, w_t, per_class=150)

    summary = {}
    for vname, recs in (
        ('external_punchthrough', ext_recs),
        ('pythia6_std_ref', std_recs),
        ('pythia6_nux', nux_recs),
    ):
        out_dir = clean_root / 'sys_shards' / vname / 'shards' / '000'
        if (out_dir / 'observations.parquet').exists():
            summary[vname] = {'status': 'already_done', 'events': len(recs)}
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f's318b-{vname}-', dir='/tmp') as tmp:
            step = max(1, (len(recs) + cpus - 1) // cpus)
            tasks = [(recs[s : s + step], tmp, k, {}, 0.57) for k, s in enumerate(range(0, len(recs), step))]
            with ProcessPoolExecutor(max_workers=cpus) as pool:
                rfs = list(pool.map(_simulate_subchunk, tasks))
            meta_out = clean_root / 'sys_shards' / vname / 'truth_meta.parquet'
            _convert_subchunks_to_shard(rfs, recs, geom, bounds, out_dir, None, meta_out)
            if vname == 'external_punchthrough':
                mdf = pd.read_parquet(meta_out)
                mdf['ext_subtype'] = [r['ext_subtype'] for r in recs]
                mdf.to_parquet(meta_out, index=False)
            write_json(clean_root / 'sys_shards' / vname / 'plan.json', {'geometry': str(clean_root / 'geometry.json')})
        summary[vname] = {'status': 'complete', 'events': len(recs)}

    summary['wall_seconds'] = float(time.perf_counter() - t0)
    summary['UTC'] = datetime.now(timezone.utc).isoformat()
    write_json(clean_root / 'extra_sys_summary.json', summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument('--cpus', type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(run_all_extra_sys(args.root, cpus=args.cpus), indent=2))


if __name__ == '__main__':
    main()
