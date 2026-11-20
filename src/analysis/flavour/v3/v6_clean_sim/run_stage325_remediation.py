"""Stage 325: End-to-End Remediation of Stage 324 Independent Clean v3 Scientific Audit.

Closes all model-training and performance-affecting issues identified in Stage 324 while
maintaining strict physical seal isolation on Cal-A, Cut-B, and Confirm:
  - 324-01: Hard path-level split_shards.json allowlist + runtime sys.addaudithook asserting
            ZERO file opens/stats on sealed shards (100..149) during training & freeze.
  - 324-02: Exact full-window weighted numerator/denominator microbatch gradient accumulation
            verified on real 128-event PinpointFormerMultitask & PinpointFormerV4 batches (<1e-4).
  - 324-03: Strict equal-class-weight, zero-weight-filtered, all-4-tasks-weighted common-support
            [50, 2312] GeV energy-matched training control + common-(E_nu, nu/nubar) evaluation.
  - 324-04: Genuine Tungsten-184 nuclear-PDF (LHAPDF6:EPPS21nlo_CT18Anlo_W184 + 260 MeV/c Fermi
            motion) NC inclusive & NC charm validation shards simulated through pinpoint_v3_1.
  - 324-05: Frozen per-event BDT-70d / LR / Occupancy / Energy baselines + screen-excluded Tune6k
            disjoint split-half locked 1%/0.1% FPR operating points on observable domains D1/D2/D3.
  - 324-06: Bore-aware Magnet1+2 aperture quality control in spectrometer.py / nomad_kinematics.py
            plus live on-axis vs shell-scattering Geant4 muon closure and Train/Tune bore audit.
  - 324-07: Full-target module-aware C++17 micro-vertex kernel linking BOTH 30mm front and
            150.01mm intermediate Pinpoint blocks + region-stratified object-level PV/SV audit.
  - 324-08: Fixed-clustering 1mm coordinate-only ablation on BOTH PF-20um and PF-Hybrid v2
            (preserving exact cluster/token/patch counts & IPT) + true 6T observable-domain eval.
  - 324-09/14: Deterministic FP32 Tune prediction export (<1e-5 match to observation-only replay),
               single-commit code tarball SHA-256 provenance, and exact entry/exit UTC timestamps.
  - 324-10/11/12/13/15: Live pinpoint_v3_1 strict ion-PDG guard matrix, tau V-A helicity probe,
                        detector-response degradation stress tests, regularized Hybrid v2 control,
                        and PV pull / per-class auxiliary calibration audit.
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
import pickle
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

import awkward as ak
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import beta as scipy_beta
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
import uproot

from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.v2.partitions import FAMILY, KEY
from analysis.flavour.v2.profile import device_batch, init_worker, shutdown_loader
from analysis.flavour.v2.r1.tasks import probabilities
from analysis.flavour.v2.r2.readout import grouped_folds
from analysis.flavour.v3.common.datasets import V3EvalDataset, V3ObservationDataset, V3TrainDataset
from analysis.flavour.v3.common.domains import (
    OBS_DOMAIN_NAMES,
    compute_observable_containment,
    detector_bounds,
)
from analysis.flavour.v3.common.observation_inference import infer_observation_only
from analysis.flavour.v3.common.submit import _package_source, submit_job
from analysis.flavour.v3.v1_b1pp.models import ZAlignedStereoNet
from analysis.flavour.v3.v1_b1pp.representation import (
    collate_b1pp,
    extract_b1pp_item,
    static_plane_layout,
)
from analysis.flavour.v3.v1_b1pp.spectrometer import (
    TRACKER_Z_MM,
    extract_spectrometer_kinematics,
)
from analysis.flavour.v3.v2_pinpointformer.pixel_tokenizer import collate_pf, extract_pf_item
from analysis.flavour.v3.v2_pinpointformer.train import move_batch_to_device
from analysis.flavour.v3.v3_multitask.losses import compute_window_task_denominators
from analysis.flavour.v3.v3_multitask.models import PinpointFormerMultitask
from analysis.flavour.v3.v3_multitask.train import _pf_v3_loss
from analysis.flavour.v3.v4_tau.ensemble import fit_oof_domain_ensemble
from analysis.flavour.v3.v4_tau.models import PinpointFormerV4
from analysis.flavour.v3.v4_tau.nomad_kinematics import NomadTransverseExtractor
from analysis.flavour.v3.v6_clean_sim.evaluate_stage317 import (
    SEEDS,
    _compute_vertex_audit,
    _select_seed_prediction_df,
    paired_family_bootstrap_delta,
)
from analysis.flavour.s6.gfaser import write_gfaser
from analysis.flavour.v3.v6_clean_sim.produce_clean_v3 import (
    CHARM_PDGS,
    DEFAULT_CLEAN_V3_ROOT,
    OOD_VARIANTS as CLEAN_V3_OOD_VARIANTS,
    _convert_subchunks_to_shard_v3,
    _family_pair,
    _sample_tungsten_vertex_v2_mm,
    build_full_volume_tungsten_sampler_spec,
)


REPO_ROOT = Path(__file__).resolve().parents[5]
G4_REPO_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/Pinpoint_G4')
STAGE325_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v7_stage325_2026-11-20')
PINPOINT_V3_1_BIN = STAGE325_ROOT / 'g4/pinpoint_v3_1'
GEOM_V3_1_MACRO = STAGE325_ROOT / 'g4/macros/geom.mac'
ROADMAP_FREEZE_V9 = REPO_ROOT / 'roadmap/v3/v9_stage325_freeze_manifest.json'
W184_EPPS21_OOD_VARIANTS = ('gen_ood_w184_epps21_nc_incl', 'gen_ood_w184_epps21_nc_charm')
ALL_STAGE325_OOD = tuple(CLEAN_V3_OOD_VARIANTS) + W184_EPPS21_OOD_VARIANTS


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _clopper_pearson_ci95(k: int, n: int) -> list[float]:
    if n <= 0:
        return [0.0, 1.0]
    lo = 0.0 if k <= 0 else float(scipy_beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k >= n else float(scipy_beta.ppf(0.975, k + 1, n - k))
    return [lo, hi]


def _ba4_and_nll(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    pred = probs.argmax(axis=1)
    recalls = [float((pred[labels == c] == c).mean()) for c in range(4) if (labels == c).any()]
    p_clip = np.clip(probs.astype(np.float64), 1e-7, 1.0)
    p_clip /= p_clip.sum(axis=1, keepdims=True)
    nll = float(-np.mean(np.log(p_clip[np.arange(len(labels)), labels])))
    non_tau = labels != 2
    cond_ba3 = 0.0
    if non_tau.any():
        p3 = p_clip[non_tau][:, [0, 1, 3]]
        y3 = np.where(labels[non_tau] == 3, 2, labels[non_tau])
        pr3 = p3.argmax(axis=1)
        r3 = [float((pr3[y3 == c] == c).mean()) for c in range(3) if (y3 == c).any()]
        cond_ba3 = float(np.mean(r3)) if r3 else 0.0
    cc_m = labels != 3
    cc_ba3 = 0.0
    if cc_m.any():
        pc3 = p_clip[cc_m][:, [0, 1, 2]]
        prc3 = pc3.argmax(axis=1)
        rc3 = [float((prc3[labels[cc_m] == c] == c).mean()) for c in range(3) if (labels[cc_m] == c).any()]
        cc_ba3 = float(np.mean(rc3)) if rc3 else 0.0
    return {'BA4': float(np.mean(recalls)) if recalls else 0.0, 'cond_BA3': cond_ba3, 'cc_BA3': cc_ba3, 'NLL': nll}


def _weighted_ba4(labels: np.ndarray, probs: np.ndarray, weights: np.ndarray) -> float:
    pred = probs.argmax(axis=1)
    recalls = []
    for c in range(4):
        m = (labels == c) & (weights > 0.0)
        if m.any():
            recalls.append(float(weights[m & (pred == c)].sum() / max(float(weights[m].sum()), 1e-12)))
    return float(np.mean(recalls)) if recalls else 0.0


def _split_half_tail_eval(
    scores: np.ndarray,
    labels: np.ndarray,
    keys_df: pd.DataFrame,
    mask: np.ndarray,
    fpr_targets: tuple[float, ...] = (0.01, 0.001),
) -> dict:
    """Strictly disjoint KEY-hashed split-half threshold lock (Stage 324-05)."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return {}
    # Deterministic KEY hash split (half A locks threshold, half B evaluates strictly held-out counts)
    hvals = np.array([
        int(hashlib.blake2b(f"{int(r)}:{int(c)}:{int(rw)}".encode(), digest_size=8).hexdigest(), 16) & 1
        for r, c, rw in zip(keys_df['run'].iloc[idx], keys_df['chunk'].iloc[idx], keys_df['row'].iloc[idx])
    ], dtype=np.int64)
    lock_m = idx[hvals == 0]
    held_m = idx[hvals == 1]

    s_lock, y_lock = scores[lock_m], labels[lock_m]
    s_held, y_held = scores[held_m], labels[held_m]
    bg_lock = s_lock[y_lock != 2]
    bg_held = s_held[y_held != 2]
    sg_held = s_held[y_held == 2]
    nb_lock = int(len(bg_lock))
    nb_held = int(len(bg_held))
    ns_held = int(len(sg_held))
    bg_lock_sorted = np.sort(bg_lock)[::-1]

    out = {'n_lock_bg': nb_lock, 'n_held_bg': nb_held, 'n_held_tau': ns_held}
    for fpr in fpr_targets:
        tag = f'fpr_{fpr:g}'
        if nb_lock == 0 or nb_held == 0 or ns_held == 0:
            out[tag] = {'valid': False}
            continue
        k_allow = int(math.floor(fpr * nb_lock))
        thr = float(bg_lock_sorted[k_allow]) if k_allow < nb_lock else float(-1e9)
        fp = int(np.sum(bg_held > thr))
        tp = int(np.sum(sg_held > thr))
        out[tag] = {
            'threshold_locked_on_halfA': thr,
            'lock_allowed_bg': k_allow,
            'held_FP': fp,
            'held_B': nb_held,
            'held_FPR': float(fp / nb_held),
            'held_FPR_ci95': _clopper_pearson_ci95(fp, nb_held),
            'held_TP': tp,
            'held_tau': ns_held,
            'held_tau_eff': float(tp / ns_held),
            'held_tau_eff_ci95': _clopper_pearson_ci95(tp, ns_held),
            'low_bg_warning': bool(k_allow < 10),
        }
    return out


# ---------------------------------------------------------------------------
# 1. Stage 325 Preparation: split_shards.json, Equal-Weight EMatch v2,
#    Live G4 Probes (324-15/10/06/13), and W-184 EPPS21 NC Charm OOD Shards (324-04)
# ---------------------------------------------------------------------------
def _compute_equal_class_energy_match_weights(
    man_train: pd.DataFrame,
    e_min: float = 50.0,
    e_max: float = 2312.0,
    n_bins: int = 20,
) -> tuple[pd.DataFrame, dict]:
    """Strict common-support [50, 2312] GeV weights with EXACT equal total weight sum per class (324-03)."""
    edges = np.geomspace(e_min, e_max, n_bins + 1)
    ev = man_train['Ev_GeV'].to_numpy(dtype=np.float64)
    lbl = man_train['label'].to_numpy(dtype=np.int64)
    in_rng = (ev >= e_min) & (ev <= e_max)
    bidx = np.clip(np.digitize(ev, edges) - 1, 0, n_bins - 1)

    p_cls = np.zeros((4, n_bins), dtype=np.float64)
    for c in range(4):
        m = (lbl == c) & in_rng
        cnt = np.bincount(bidx[m], minlength=n_bins).astype(np.float64)
        p_cls[c] = (cnt + 0.5) / max(1.0, cnt.sum() + 0.5 * n_bins)
    p_target = np.power(np.prod(p_cls, axis=0), 0.25)
    p_target /= max(float(p_target.sum()), 1e-12)

    # Exact target total weight per class = mean number of in-range events across all 4 classes
    n_in_cls = [int(((lbl == c) & in_rng).sum()) for c in range(4)]
    target_sum_per_class = float(np.mean(n_in_cls))
    w = np.zeros(len(man_train), dtype=np.float64)
    diag = {'e_min_GeV': e_min, 'e_max_GeV': e_max, 'target_sum_per_class': target_sum_per_class, 'per_class': {}}
    for c in range(4):
        m = (lbl == c) & in_rng
        raw_w = np.clip(p_target[bidx[m]] / np.maximum(p_cls[c, bidx[m]], 1e-9), 0.1, 10.0)
        # Scale so sum(w[m]) == target_sum_per_class EXACTLY for every class c in {0,1,2,3}!
        raw_w *= (target_sum_per_class / max(float(raw_w.sum()), 1e-12))
        w[m] = raw_w
        ess = float((raw_w.sum() ** 2) / max(float(np.square(raw_w).sum()), 1e-12))
        diag['per_class'][str(c)] = {
            'in_range_count': int(m.sum()),
            'zero_weight_excluded_count': int(((lbl == c) & (~in_rng)).sum()),
            'weight_sum': float(w[lbl == c].sum()),
            'effective_sample_size': ess,
            'weighted_E_median_GeV': float(np.interp(0.5, np.cumsum(raw_w[np.argsort(ev[m])]) / raw_w.sum(), np.sort(ev[m]))),
        }
    out = man_train[KEY].copy()
    out['energy_match_weight'] = w.astype(np.float32)
    return out, diag


def _run_g4_single(
    macro_text: str,
    workdir: Path,
    env_extra: dict[str, str] | None = None,
    timeout_s: int = 180,
) -> subprocess.CompletedProcess:
    mac_p = workdir / 'run.mac'
    mac_p.write_text(macro_text)
    env = dict(os.environ)
    env.update({
        'PINPOINT_USE_INPUT_Z': '1',
        'PINPOINT_SKIP_GDML': '1',
        'PINPOINT_COMPACT_TRUTH': '1',
        'PINPOINT_REQUIRE_W_MATERIAL': '0',
        'PINPOINT_REQUIRE_W_TARGET': '0',
        'PINPOINT_PIXEL_DIFFUSION_UM': '3.0',
        'PINPOINT_PIXEL_NOISE_E': '10.0',
        'PINPOINT_PIXEL_THR_E': '100.0',
        'PINPOINT_SIPM_TIME_THR_PE': '3',
        'PINPOINT_SIPM_TTS_NS': '0.15',
    })
    if env_extra:
        env.update(env_extra)
    # Run inside LCG_107 gcc11 environment matching pinpoint_v3_1 (macro is positional arg!)
    cmd = (
        'source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc11-opt/setup.sh >/dev/null 2>&1 && '
        f'{shlex.quote(str(PINPOINT_V3_1_BIN))} {shlex.quote(str(mac_p))}'
    )
    return subprocess.run(['bash', '-c', cmd], cwd=workdir, env=env, capture_output=True, text=True, timeout=timeout_s)


def run_live_g4_v3_1_probes(clean_root: Path) -> dict:
    """Execute live pinpoint_v3_1 probes for 324-15 (guards), 324-10 (dark rate), 324-06 (bore), 324-13 (tau helicity)."""
    t0 = time.perf_counter()
    geom = json.loads((clean_root / 'geometry.json').read_text())
    bounds = detector_bounds(geom)
    res: dict = {'pinpoint_v3_1_bin': str(PINPOINT_V3_1_BIN), 'pinpoint_v3_1_sha256': _sha256_file(PINPOINT_V3_1_BIN)}

    with tempfile.TemporaryDirectory(prefix='s325-g4-probes-') as td:
        tmp = Path(td)
        # 1. Stage 324-15: 7-case strict W target & material guard matrix
        guard_cases = [
            ('W184_in_W', 1000741840, 74, 184, 0.0, 0.0, -2.98449, True),
            ('W186_in_W', 1000741860, 74, 186, 0.0, 0.0, -2.98449, True),
            ('C12_in_W', 1000060120, 6, 12, 0.0, 0.0, -2.98449, False),
            ('proton2212_fakeZ74', 2212, 74, 1, 0.0, 0.0, -2.98449, False),
            ('W184_mismatchA186', 1000741840, 74, 186, 0.0, 0.0, -2.98449, False),
            ('unknown_pdg0_fakeZ74', 0, 74, 184, 0.0, 0.0, -2.98449, False),
            ('W184_in_Air', 1000741840, 74, 184, 0.0, 0.0, -2.99600, False),
        ]
        guard_out = {}
        for cname, tpdg, tz, ta, vx, vy, vz, exp_pass in guard_cases:
            cdir = tmp / f'guard_{cname}'
            cdir.mkdir()
            in_root = cdir / 'in.root'
            rec = [{
                'n': 3, 'pdgc': [14, tpdg, 13], 'status': [0, 0, 1], 'name': ['', '', 'mu-'],
                'px': [0.0, 0.0, 0.0], 'py': [0.0, 0.0, 0.0], 'pz': [100.0, 0.0, 100.0],
                'E': [100.0, 171.3, 100.0001], 'pxv': 0.0, 'pyv': 0.0, 'pzv': 100.0, 'Ev': 100.0,
                'Q2': 10.0, 'W': 5.0, 'x': 0.2, 'y': 0.3, 'XSec': 1.0, 'intType': 2, 'scatteringType': 1,
                'Z': tz, 'A': ta, 'tgt': tpdg, 'neu': 14, 'fspl': 13, 'hitnuc': 2112,
                'cc': True, 'nc': False, 'vx': vx, 'vy': vy, 'vz': vz,
            }]
            write_gfaser(rec, in_root)
            mac = (
                '/control/verbose 0\n'
                '/run/verbose 0\n'
                f'/control/execute {GEOM_V3_1_MACRO}\n'
                '/det/enableFaserSpectrometer true\n'
                '/gen/select gfaser\n'
                f'/gen/gfaser/inputFile {in_root}\n'
                '/gen/gfaser/hitPixelArea false\n'
                f'/out/fileName {cdir / "out.root"}\n'
                '/out/saveTrack false\n'
                '/out/saveTruthHits false\n'
                '/run/initialize\n'
                '/Pinpoint/field/value 0.57 tesla\n'
                '/run/beamOn 1\n'
            )
            cp = _run_g4_single(mac, cdir, {'PINPOINT_REQUIRE_W_MATERIAL': '1', 'PINPOINT_REQUIRE_W_TARGET': '1'})
            passed = (cp.returncode == 0)
            guard_out[cname] = {'expected_pass': exp_pass, 'actual_pass': passed, 'returncode': int(cp.returncode), 'ok': bool(passed == exp_pass)}
            if passed != exp_pass:
                raise AssertionError(f'Stage 324-15 guard test failed for {cname}: expected={exp_pass}, rc={cp.returncode}\n{cp.stderr[-400:]}\n{cp.stdout[-400:]}')
        res['guard_matrix_324_15'] = guard_out

        # 2. Stage 324-10: Pixel dark noise injection verification
        dark_out = {}
        for rate_str in ('0.0', '1e-5'):
            ddir = tmp / f'dark_{rate_str}'
            ddir.mkdir()
            mac = (
                '/control/verbose 0\n'
                '/run/verbose 0\n'
                f'/control/execute {GEOM_V3_1_MACRO}\n'
                '/det/enableFaserSpectrometer true\n'
                '/gen/select gun\n'
                f'/out/fileName {ddir / "out.root"}\n'
                '/out/saveTrack false\n'
                '/out/saveTruthHits false\n'
                '/run/initialize\n'
                '/Pinpoint/field/value 0.57 tesla\n'
                '/gps/particle mu-\n'
                '/gps/energy 50 GeV\n'
                '/gps/position 0 -12 -3005 mm\n'
                '/gps/direction 0 0 1\n'
                '/run/beamOn 20\n'
            )
            cp = _run_g4_single(mac, ddir, {'PINPOINT_PIXEL_DARK_RATE': rate_str})
            if cp.returncode != 0:
                raise RuntimeError(f'Dark noise probe failed for {rate_str}: {cp.stderr[-400:]}\n{cp.stdout[-400:]}')
            with uproot.open(ddir / 'out.root') as rf:
                ph = rf['Hits/pixelHits'].arrays(['event_id', 'trackID'], library='ak')
            tids = ak.to_numpy(ak.flatten(ph['trackID'])) if len(ph['trackID']) else np.empty(0, dtype=np.int64)
            n_tot = int(len(tids))
            n_dark = int(np.sum(tids == -1))
            dark_out[f'rate_{rate_str}'] = {
                'events': 20,
                'mean_total_pixel_hits_per_event': float(n_tot / 20.0),
                'mean_dark_hits_per_event': float(n_dark / 20.0),
            }
        res['pixel_dark_noise_324_10'] = dark_out

        # 3. Stage 324-06: On-axis in-bore vs off-axis shell 100 GeV mu+ spectrometer closure
        spec_out = {}
        for tag, x_mm, y_mm in (('on_axis_x0', 0.0, -10.0), ('off_axis_shell_x112', 112.0, -10.0)):
            sdir = tmp / f'spec_{tag}'
            sdir.mkdir()
            mac = (
                '/control/verbose 0\n'
                '/run/verbose 0\n'
                f'/control/execute {GEOM_V3_1_MACRO}\n'
                '/det/enableFaserSpectrometer true\n'
                '/gen/select gun\n'
                f'/out/fileName {sdir / "out.root"}\n'
                '/out/saveTrack false\n'
                '/out/saveTruthHits false\n'
                '/run/initialize\n'
                '/Pinpoint/field/value 0.57 tesla\n'
                '/gps/particle mu+\n'
                '/gps/energy 100 GeV\n'
                f'/gps/position {x_mm} {y_mm} -1640 mm\n'
                '/gps/direction 0 0 1\n'
                '/run/beamOn 60\n'
            )
            cp = _run_g4_single(mac, sdir)
            if cp.returncode != 0:
                raise RuntimeError(f'Spec probe failed for {tag}: {cp.stderr[-400:]}\n{cp.stdout[-400:]}')
            with uproot.open(sdir / 'out.root') as rf:
                fh = rf['Hits/faserHits'].arrays(['event_id', 'trackerID', 'x', 'y', 'z'], library='ak')
            n_3st_raw = 0
            n_in_bore = 0
            n_shell_flagged = 0
            raw_sys = []
            in_bore_sys = []
            for eid in range(len(fh['event_id'])):
                sts = ak.to_numpy(fh['trackerID'][eid])
                if len(sts) == 0:
                    continue
                xs = ak.to_numpy(fh['x'][eid])
                ys = ak.to_numpy(fh['y'][eid])
                st_xy = {}
                for s in (0, 1, 2):
                    ms = sts == s
                    if ms.any():
                        st_xy[s] = (float(np.mean(xs[ms])), float(np.mean(ys[ms])))
                if len(st_xy) == 3:
                    n_3st_raw += 1
                    sy_raw = st_xy[1][1] - 0.5 * (st_xy[0][1] + st_xy[2][1])
                    raw_sys.append(sy_raw)
                    trk_obs = {
                        'station': [0, 1, 2],
                        'xy_mm': [list(st_xy[0]), list(st_xy[1]), list(st_xy[2])],
                    }
                    sk = extract_spectrometer_kinematics(trk_obs, (-1640.0, x_mm, y_mm, True), bounds)
                    if bool(sk['cand_mask'][0]):
                        if float(sk['cand'][0, 0]) > 0.75:
                            n_in_bore += 1
                            in_bore_sys.append(sy_raw)
                        else:
                            n_shell_flagged += 1
            spec_out[tag] = {
                'events': 60,
                'n_3station_raw': n_3st_raw,
                'n_accepted_in_bore': n_in_bore,
                'n_flagged_out_of_bore': n_shell_flagged,
                'raw_sagitta_mean_mm': float(np.mean(raw_sys)) if raw_sys else 0.0,
                'raw_sagitta_min_mm': float(np.min(raw_sys)) if raw_sys else 0.0,
                'raw_sagitta_max_mm': float(np.max(raw_sys)) if raw_sys else 0.0,
                'n_wrong_sign_raw': int(sum(1 for v in raw_sys if v > 0.0)),
                'n_wrong_sign_accepted_in_bore': int(sum(1 for v in in_bore_sys if v > 0.0)),
            }
        res['spectrometer_bore_closure_324_06'] = spec_out

        # 4. Stage 324-13: Tau V-A helicity polarization closure via Py8Decayer (tau tree daughters!)
        # For tau- -> pi- nu_tau: analytical V-A gives <z_pi> = 1/3 (h=-1), 1/2 (h=0), 2/3 (h=+1).
        tau_hel = {}
        for h_label, hel_env in (('VA_left_h_minus1', '-1.0'), ('unpolarized_h_0', '0.0'), ('right_h_plus1', '1.0')):
            tdir = tmp / f'tau_{h_label}'
            tdir.mkdir()
            in_r = tdir / 'in.root'
            recs = [{
                'n': 3, 'pdgc': [16, 1000741840, 15], 'status': [0, 0, 1], 'name': ['', '', 'tau-'],
                'px': [0.0, 0.0, 0.0], 'py': [0.0, 0.0, 0.0], 'pz': [500.0, 0.0, 500.0],
                'E': [500.0, 171.3, math.hypot(500.0, 1.77686)], 'pxv': 0.0, 'pyv': 0.0, 'pzv': 500.0, 'Ev': 500.0,
                'Q2': 20.0, 'W': 10.0, 'x': 0.2, 'y': 0.1, 'XSec': 1.0, 'intType': 2, 'scatteringType': 1,
                'Z': 74, 'A': 184, 'tgt': 1000741840, 'neu': 16, 'fspl': 15, 'hitnuc': 2112,
                'cc': True, 'nc': False, 'vx': 0.0, 'vy': 0.0, 'vz': -2.996,
            } for _ in range(400)]
            write_gfaser(recs, in_r)
            mac = (
                '/control/verbose 0\n'
                '/run/verbose 0\n'
                f'/control/execute {GEOM_V3_1_MACRO}\n'
                '/det/enableFaserSpectrometer true\n'
                '/gen/select gfaser\n'
                f'/gen/gfaser/inputFile {in_r}\n'
                '/gen/gfaser/hitPixelArea false\n'
                f'/out/fileName {tdir / "out.root"}\n'
                '/out/saveTrack false\n'
                '/out/saveTruthHits false\n'
                '/random/setSeeds 32513 32519\n'
                '/run/initialize\n'
                '/Pinpoint/field/value 0.57 tesla\n'
                '/run/beamOn 400\n'
            )
            cp = _run_g4_single(mac, tdir, {'PINPOINT_TAU_HELICITY': hel_env})
            if cp.returncode != 0:
                raise RuntimeError(f'Tau helicity probe failed for {h_label}: {cp.stderr[-400:]}\n{cp.stdout[-400:]}')
            with uproot.open(tdir / 'out.root') as rf:
                tt = rf['tau'].arrays(['evtID', 'PDG', 'Px', 'Py', 'Pz'], library='np')
            by_evt: dict[int, list[tuple[int, float]]] = {}
            for eid, pdg, px, py_p, pz in zip(tt['evtID'], tt['PDG'], tt['Px'], tt['Py'], tt['Pz']):
                # Geant4 internal momentum unit is MeV/c; 500 GeV = 500,000 MeV/c
                pmag_gev = math.sqrt(float(px) ** 2 + float(py_p) ** 2 + float(pz) ** 2) / 1000.0
                by_evt.setdefault(int(eid), []).append((int(pdg), pmag_gev))
            z_vis_all = []
            z_pi_1p = []
            z_lep_all = []
            for eid, plist in by_evt.items():
                vis = [(p, e) for p, e in plist if abs(p) not in (12, 14, 16)]
                if not vis:
                    continue
                z_v = sum(e for _, e in vis) / 500.0
                if 0.005 < z_v <= 1.05:
                    z_vis_all.append(z_v)
                if len(vis) == 1 and abs(vis[0][0]) == 211:
                    z_pi_1p.append(vis[0][1] / 500.0)
                if len(vis) == 1 and abs(vis[0][0]) in (11, 13):
                    z_lep_all.append(vis[0][1] / 500.0)
            tau_hel[h_label] = {
                'n_events_decayed': len(z_vis_all),
                'mean_all_vis_p_frac': float(np.mean(z_vis_all)) if z_vis_all else 0.0,
                'n_pi_1prong': len(z_pi_1p),
                'mean_z_pi_1prong': float(np.mean(z_pi_1p)) if z_pi_1p else None,
                'n_leptonic_1prong': len(z_lep_all),
                'mean_z_lepton': float(np.mean(z_lep_all)) if z_lep_all else None,
            }
        res['tau_helicity_closure_324_13'] = tau_hel

    res['wall_seconds'] = float(time.perf_counter() - t0)
    out_p = STAGE325_ROOT / 'g4/live_g4_v3_1_closure.json'
    write_json(out_p, res)
    return res


def _pythia8_w184_epps21_nc_worker(args: tuple) -> tuple[list[dict], dict]:
    """Generate Tungsten-184 nuclear-PDF (EPPS21nlo_CT18Anlo_W184) + 260 MeV/c Fermi-motion NC DIS events."""
    os.environ['LHAPDF_DATA_PATH'] = f"/cvmfs/sft.cern.ch/lcg/external/lhapdfsets/current:{os.environ.get('LHAPDF_DATA_PATH', '')}"
    import pythia8

    worker_id, n_target, charm_only, e_pool = args
    rng = np.random.default_rng(20263250 + worker_id * 1009 + (50000 if charm_only else 0))
    m_nuc = 0.938918  # average nucleon mass GeV
    p_fermi = 0.260   # tungsten nuclear Fermi momentum GeV/c

    out: list[dict] = []
    n_tried = 0
    n_charm_seen = 0
    tag = 'pythia8_w184_epps21_nc_charm' if charm_only else 'pythia8_w184_epps21_nc_incl'
    ood_tag = 'gen_ood_w184_epps21_nc_charm' if charm_only else 'gen_ood_w184_epps21_nc_incl'
    e_bins = [120.0, 250.0, 500.0, 900.0, 1500.0, 2500.0]
    bin_probs = np.array([0.12, 0.22, 0.28, 0.20, 0.12, 0.06], dtype=np.float64)

    # Loop over discrete beam energy bins matching nominal Clean v3 NC spectrum
    for b_idx, e_beam in enumerate(e_bins):
        n_want_bin = max(1, int(round(n_target * bin_probs[b_idx]))) if b_idx < len(e_bins) - 1 else max(1, n_target - len(out))
        py_b = pythia8.Pythia('', False)
        py_b.readString('Beams:frameType = 2')
        nu_pdg = int(rng.choice([12, 14, -12, -14], p=[0.20, 0.52, 0.08, 0.20]))
        py_b.readString(f'Beams:idA = {nu_pdg}')
        py_b.readString('Beams:idB = 2212')
        py_b.readString(f'Beams:eA = {e_beam:.3f}')
        py_b.readString(f'Beams:eB = {m_nuc:.6f}')
        py_b.readString('PDF:lepton = off')
        py_b.readString('PDF:pSetB = LHAPDF6:EPPS21nlo_CT18Anlo_W184/0')
        py_b.readString('WeakBosonExchange:ff2ff(t:gmZ) = on')
        py_b.readString('PhaseSpace:Q2Min = 1.0')
        py_b.readString('Print:quiet = on')
        py_b.readString('Random:setSeed = on')
        py_b.readString(f'Random:seed = {325000 + worker_id * 100 + b_idx * 7 + (5000 if charm_only else 0)}')
        if not py_b.init():
            continue
        got_bin = 0
        max_tries = n_want_bin * (1500 if charm_only else 5)
        for _ in range(max_tries):
            if got_bin >= n_want_bin or len(out) >= n_target:
                break
            if not py_b.next():
                continue
            n_tried += 1
            ev = py_b.event
            has_c = any(abs(int(pt.id())) in CHARM_PDGS for pt in ev)
            if has_c:
                n_charm_seen += 1
            if charm_only and not has_c:
                continue

            # Apply isotropic tungsten nuclear Fermi momentum boost (|p_F| <= 260 MeV/c, p^2 dp distribution)
            pf_mag = p_fermi * (float(rng.random()) ** (1.0 / 3.0))
            cos_t = float(rng.uniform(-1.0, 1.0))
            sin_t = math.sqrt(max(0.0, 1.0 - cos_t * cos_t))
            phi = float(rng.uniform(0.0, 2.0 * math.pi))
            bx = (pf_mag * sin_t * math.cos(phi)) / math.hypot(m_nuc, pf_mag)
            by = (pf_mag * sin_t * math.sin(phi)) / math.hypot(m_nuc, pf_mag)
            bz = (pf_mag * cos_t) / math.hypot(m_nuc, pf_mag)
            b2 = bx * bx + by * by + bz * bz
            gam = 1.0 / math.sqrt(max(1.0 - b2, 1e-9))

            finals = []
            for pt in ev:
                if not pt.isFinal():
                    continue
                pid = int(pt.id())
                if abs(pid) in (12, 14, 16):
                    continue
                px, py_p, pz, pe = float(pt.px()), float(pt.py()), float(pt.pz()), float(pt.e())
                bp = bx * px + by * py_p + bz * pz
                g2 = (gam - 1.0) / max(b2, 1e-12)
                px_b = px + g2 * bp * bx + gam * bx * pe
                py_b_val = py_p + g2 * bp * by + gam * by * pe
                pz_b = pz + g2 * bp * bz + gam * bz * pe
                pe_b = gam * (pe + bp)
                finals.append((pid, px_b, py_b_val, pz_b, pe_b))
            if not finals:
                continue

            q2_val = float(py_b.infoPython().Q2Fac())
            x_val = float(py_b.infoPython().x2())
            y_val = float(np.clip(q2_val / max(2.0 * m_nuc * e_beam * max(x_val, 1e-5), 1e-4), 0.0, 1.0))
            w_val = math.sqrt(max(m_nuc * m_nuc + q2_val * (1.0 / max(x_val, 1e-5) - 1.0), 1.0))
            sigma_mb = float(py_b.infoPython().sigmaGen())

            row_id = worker_id * 100000 + len(out)
            hi, lo = _family_pair(tag, row_id)
            pdgc = [nu_pdg, 1000741840] + [p[0] for p in finals]
            status = [0, 0] + [1] * len(finals)
            px_l = [0.0, 0.0] + [p[1] for p in finals]
            py_l = [0.0, 0.0] + [p[2] for p in finals]
            pz_l = [e_beam, 0.0] + [p[3] for p in finals]
            en_l = [e_beam, 171.3] + [p[4] for p in finals]
            out.append({
                'source_tag': tag,
                'source_row': int(row_id),
                'ood_sample_tag': ood_tag,
                'family_hi': int(hi),
                'family_lo': int(lo),
                'label': 3,
                'has_gen_charm': bool(has_c),
                'n': len(pdgc),
                'pdgc': pdgc,
                'status': status,
                'name': [''] * len(pdgc),
                'px': px_l, 'py': py_l, 'pz': pz_l, 'E': en_l,
                'pxv': 0.0, 'pyv': 0.0, 'pzv': float(e_beam), 'Ev': float(e_beam),
                'Q2': q2_val, 'W': w_val, 'x': x_val, 'y': y_val, 'XSec': sigma_mb,
                'intType': 3, 'scatteringType': 1,
                'Z': 74, 'A': 184, 'tgt': 1000741840,
                'neu': int(nu_pdg), 'fspl': int(nu_pdg), 'hitnuc': 2212,
                'cc': False, 'nc': True,
            })
            got_bin += 1
    return out, {'tried': n_tried, 'charm_seen': n_charm_seen, 'selected': len(out)}


def _simulate_subchunk_v3_1(args: tuple) -> str:
    recs, tmp_dir, sub_idx, env_extra, mag_t = args
    sub_dir = Path(tmp_dir) / f'sub_{sub_idx:02d}'
    sub_dir.mkdir(parents=True, exist_ok=True)
    in_root = sub_dir / 'input.root'
    out_root = sub_dir / 'output.root'
    write_gfaser(recs, in_root)
    seed_base = 20263250 + sub_idx * 17 + int(recs[0].get('run', 0)) % 100
    mac = (
        '/control/verbose 0\n'
        '/run/verbose 0\n'
        f'/control/execute {GEOM_V3_1_MACRO}\n'
        '/det/enableFaserSpectrometer true\n'
        '/gen/select gfaser\n'
        f'/gen/gfaser/inputFile {in_root}\n'
        '/gen/gfaser/hitPixelArea false\n'
        f'/out/fileName {out_root}\n'
        '/out/saveTrack false\n'
        '/out/saveTruthHits false\n'
        f'/random/setSeeds {seed_base} {seed_base + 19}\n'
        '/run/initialize\n'
        f'/Pinpoint/field/value {mag_t:.3f} tesla\n'
        f'/run/beamOn {len(recs)}\n'
    )
    cp = _run_g4_single(mac, sub_dir, env_extra, timeout_s=3600)
    if cp.returncode != 0 or not out_root.is_file():
        raise RuntimeError(f'pinpoint_v3_1 subchunk {sub_idx} failed rc={cp.returncode}: {cp.stderr[-600:]}\n{cp.stdout[-400:]}')
    return str(out_root)


def run_w184_ood_worker(clean_root: Path, variant: str, cpus: int = 8) -> dict:
    t0 = time.perf_counter()
    geom = json.loads((clean_root / 'geometry.json').read_text())
    bounds = detector_bounds(geom)
    ood_dir = STAGE325_ROOT / 'ood_shards' / variant
    recs = pq.read_table(ood_dir / 'input.parquet').to_pylist()
    out_shard = ood_dir / 'shards/000'
    scratch_base = os.environ.get('PINPOINT_V3_SCRATCH', '/tmp')
    Path(scratch_base).mkdir(parents=True, exist_ok=True)
    env_ex = {'PINPOINT_REQUIRE_W_MATERIAL': '1', 'PINPOINT_REQUIRE_W_TARGET': '1'}
    with tempfile.TemporaryDirectory(prefix=f's325-w184-{variant}-', dir=scratch_base) as tmp:
        step = max(1, (len(recs) + cpus - 1) // cpus)
        tasks = [(recs[s : s + step], tmp, k, env_ex, 0.57) for k, s in enumerate(range(0, len(recs), step))]
        with ProcessPoolExecutor(max_workers=cpus) as pool:
            rfs = [Path(p) for p in pool.map(_simulate_subchunk_v3_1, tasks)]
        _convert_subchunks_to_shard_v3(
            rfs, recs, geom, bounds, out_shard, None,
            ood_dir / 'truth_meta.parquet',
            require_w_target=True,
        )
        write_json(ood_dir / 'plan.json', {'geometry': str(clean_root / 'geometry.json')})
    return {'status': 'complete', 'variant': variant, 'events': len(recs), 'seconds': float(time.perf_counter() - t0)}


def prepare_stage325_upgrades(clean_root: Path, submit_w184_condor: bool = True) -> dict:
    """Create split_shards.json, equal-weight EMatch v2, live G4 v3.1 probes, and W184 EPPS21 NC shards."""
    STAGE325_ROOT.mkdir(parents=True, exist_ok=True)
    geom = json.loads((clean_root / 'geometry.json').read_text())
    bounds = detector_bounds(geom)
    sampler_spec = build_full_volume_tungsten_sampler_spec(geom, bounds, margin_xy_mm=0.001, margin_z_mm=0.001)

    # 1. Stage 324-01: write split_shards.json strictly from planned_manifest.parquet (ZERO sealed shard opens!)
    man_cols = pd.read_parquet(clean_root / 'planned_manifest.parquet', columns=['chunk', 'split'])
    split_shards: dict[str, list[str]] = {}
    for sp_name, sub in man_cols.groupby('split'):
        split_shards[str(sp_name)] = sorted(f'{int(c):03d}' for c in sub['chunk'].unique())
    write_json(clean_root / 'data/split_shards.json', split_shards)

    # 2. Stage 324-03: strict common-support equal-class-weight EMatch v2 weights
    man_tr = pd.read_parquet(clean_root / 'planned_manifest.parquet', columns=KEY + ['split', 'label', 'Ev_GeV'])
    man_tr = man_tr[man_tr['split'] == 'train'].reset_index(drop=True)
    emw_df, emw_diag = _compute_equal_class_energy_match_weights(man_tr)
    emw_path = STAGE325_ROOT / 'train_energy_match_weights_v2.parquet'
    emw_df.to_parquet(emw_path, index=False, compression=COMPRESSION)
    write_json(STAGE325_ROOT / 'train_energy_match_weights_v2_audit.json', emw_diag)

    # 3. Live pinpoint_v3_1 closure probes (324-15, 324-10, 324-06, 324-13)
    g4_probes = run_live_g4_v3_1_probes(clean_root)

    # 4. Stage 324-04: prepare genuine Tungsten-184 EPPS21 nPDF NC inclusive & NC charm inputs (1,200 each)
    epps_summary = {}
    for vname, charm_only, run_id in (
        ('gen_ood_w184_epps21_nc_incl', False, 32501),
        ('gen_ood_w184_epps21_nc_charm', True, 32502),
    ):
        vdir = STAGE325_ROOT / 'ood_shards' / vname
        vdir.mkdir(parents=True, exist_ok=True)
        n_w = 8
        per_w = 150
        with ProcessPoolExecutor(max_workers=n_w) as pool:
            parts = list(pool.map(_pythia8_w184_epps21_nc_worker, [(w, per_w, charm_only, None) for w in range(n_w)]))
        recs = [r for sub, _ in parts for r in sub][:1200]
        tried = sum(st['tried'] for _, st in parts)
        cseen = sum(st['charm_seen'] for _, st in parts)
        for idx_r, r in enumerate(recs):
            vm = _sample_tungsten_vertex_v2_mm(r['family_hi'], r['family_lo'], sampler_spec)
            r['vx'] = vm['vx_mm'] / 1000.0
            r['vy'] = vm['vy_mm'] / 1000.0
            r['vz'] = vm['vz_mm'] / 1000.0
            r.update({
                'run': run_id, 'chunk': 0, 'row': idx_r, 'split': 'validation',
                'slab_id': int(vm['slab_id']), 'module_type': str(vm['module_type']),
                'block_copy_no': int(vm['block_copy_no']), 'plate_in_block': int(vm['plate_in_block']),
                'is_pinpoint_module': bool(vm['is_pinpoint_module']),
                'is_front_pinpoint': bool(vm['is_front_pinpoint']),
                'is_interm_pinpoint': bool(vm['is_interm_pinpoint']),
                'is_pixel_xy_acceptance': bool(vm['is_pixel_xy_acceptance']),
                'sampling_stratum': int(vm['sampling_stratum']),
                'sampling_stratum_name': str(vm['sampling_stratum_name']),
                'geom_weight': float(vm['geom_weight']),
            })
        pq.write_table(pa.Table.from_pylist(recs), vdir / 'input.parquet', compression=COMPRESSION)
        epps_summary[vname] = {
            'generator': 'Pythia8.315 + LHAPDF6:EPPS21nlo_CT18Anlo_W184/0 + Fermi260MeV (no intranuclear FSI cascade)',
            'selected_events': len(recs),
            'total_tried': tried,
            'charm_seen_in_stream': cseen,
            'natural_charm_fraction': float(cseen / max(tried, 1)),
            'charm_in_selected': int(sum(1 for r in recs if r['has_gen_charm'])),
        }
    write_json(STAGE325_ROOT / 'ood_shards/w184_epps21_generation_summary.json', epps_summary)

    condor_rec = None
    if submit_w184_condor:
        from analysis.flavour.v2.submit import choose_schedd
        (STAGE325_ROOT / 'logs').mkdir(parents=True, exist_ok=True)
        (STAGE325_ROOT / 'code').mkdir(parents=True, exist_ok=True)
        (STAGE325_ROOT / 'runtime').mkdir(parents=True, exist_ok=True)
        archive = STAGE325_ROOT / 'code/s325_w184_ood.tar.gz'
        _package_source(REPO_ROOT, archive)
        chosen, _, checks = choose_schedd(STAGE325_ROOT)
        cdir = Path.home() / 'condor/Pinpoint/stage325_w184_ood'
        cdir.mkdir(parents=True, exist_ok=True)
        wrapper = cdir / 'run_w184.sh'
        wrapper.write_text('\n'.join([
            '#!/bin/bash',
            'set -o pipefail',
            'which() { command -v "$@"; }; export -f which',
            'source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh >/dev/null 2>&1',
            'set -e',
            'ls /eos/home-x/xcheng/FASER/Pinpoint/results >/dev/null',
            'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
            'export PINPOINT_V3_SCRATCH="${_CONDOR_SCRATCH_DIR:-/tmp}/s325_w184_${PINPOINT_JOB_ID}"',
            'mkdir -p "$PINPOINT_V3_SCRATCH/code"',
            f'tar -xzf {shlex.quote(str(archive))} -C "$PINPOINT_V3_SCRATCH/code"',
            'export PYTHONPATH="$PINPOINT_V3_SCRATCH/code/src:${PYTHONPATH:-}"',
            'cd "$PINPOINT_V3_SCRATCH"',
            'VARIANTS=(gen_ood_w184_epps21_nc_incl gen_ood_w184_epps21_nc_charm)',
            'VNAME="${VARIANTS[$1]}"',
            f'exec python3 -u -m analysis.flavour.v3.v6_clean_sim.run_stage325_remediation w184-ood-worker --clean-root {shlex.quote(str(clean_root))} --variant "$VNAME" --cpus 8',
            '',
        ]))
        wrapper.chmod(0o755)
        sub_p = cdir / 'w184.sub'
        sub_p.write_text('\n'.join([
            'universe = vanilla',
            f'executable = {wrapper}',
            'arguments = $(Process)',
            f'initialdir = {cdir}',
            f'output = {STAGE325_ROOT}/logs/w184.$(ClusterId).$(Process).out',
            f'error = {STAGE325_ROOT}/logs/w184.$(ClusterId).$(Process).err',
            f'log = {STAGE325_ROOT}/logs/w184.$(ClusterId).log',
            'environment = "PINPOINT_JOB_ID=$(ClusterId).$(Process)"',
            'request_cpus = 8',
            'request_memory = 16GB',
            'request_disk = 16GB',
            'requirements = (OpSysAndVer == "AlmaLinux9")',
            '+JobFlavour = "longlunch"',
            'should_transfer_files = YES',
            'when_to_transfer_output = ON_EXIT',
            'transfer_output_files = ""',
            'queue 2',
            '',
        ]))
        resp = subprocess.run(['condor_submit', '-name', chosen, str(sub_p)], capture_output=True, text=True, check=True)
        condor_rec = {'schedd': chosen, 'checks': checks, 'stdout': resp.stdout}
        write_json(STAGE325_ROOT / 'runtime/w184_epps21_ood.submission.json', condor_rec)

    summary = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'split_shards_counts': {k: len(v) for k, v in split_shards.items()},
        'ematch_v2_audit': emw_diag,
        'live_g4_v3_1_probes': g4_probes,
        'w184_epps21_summary': epps_summary,
        'w184_condor_submission': condor_rec,
    }
    write_json(STAGE325_ROOT / 'stage325_prepare_summary.json', summary)
    return summary


# ---------------------------------------------------------------------------
# 2. GPU Equivalence Gate (324-02), Bore Audit (324-06) & Object-Level Micro-Vertex Audit (324-07)
# ---------------------------------------------------------------------------
def _identity_collate(xs: list[dict]) -> list[dict]:
    return xs


def run_verify_gpu_gates(clean_root: Path) -> dict:
    t0 = time.perf_counter()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    geom = json.loads((clean_root / 'geometry.json').read_text())
    cfg_base = {
        'pe_only': True, 'bar_threshold_pe': 2.5, 'sipm_npix': 14400.0,
        'apply_sipm_saturation': False, 'invert_sipm_saturation': True,
    }
    tr_ds = V3TrainDataset(clean_root / 'data', clean_root / 'truth_sidecar', split='train', config=cfg_base, item_transform=extract_pf_item)
    items128 = [tr_ds[i] for i in range(128)]
    tr_ds.native.close()
    # Attach non-uniform sample weights & verify full 128 vs 64x2 gradient & 1-step AdamW equivalence in FP32
    rng = np.random.default_rng(32502)
    for it in items128:
        it['sample_weight'] = float(rng.uniform(0.25, 3.5))

    # Build 2 microbatches with IDENTICAL tensor shapes (fixed padding!) as full batch for exact FP32 comparison
    b_full = collate_pf(items128, mirror_rng=None)
    mb0 = collate_pf(items128[:64], mirror_rng=None)
    mb1 = collate_pf(items128[64:], mirror_rng=None)
    # Match max sequence length across mb0, mb1, b_full so masking is identical
    def _pad_match(mb: dict, ref: dict) -> dict:
        out = dict(mb)
        for k, v in mb.items():
            if not isinstance(v, torch.Tensor) or k not in ref:
                continue
            rt = ref[k]
            if v.ndim >= 2 and v.shape[1] < rt.shape[1]:
                pad_shape = list(v.shape)
                pad_shape[1] = rt.shape[1] - v.shape[1]
                pad = torch.zeros(pad_shape, dtype=v.dtype)
                out[k] = torch.cat([v, pad], dim=1)
        return out
    mb0 = _pad_match(mb0, b_full)
    mb1 = _pad_match(mb1, b_full)

    equiv_res = {}
    for mname, cls_m, extra_cfg in (
        ('PF_Multitask_weightAll', PinpointFormerMultitask, {'enable_micro_vertex': True, 'weight_all_tasks': True}),
        ('PF_V4_NOMAD_procWeight', PinpointFormerV4, {'enable_micro_vertex': True, 'model_family': 'pf_v4', 'enable_nomad': True, 'legacy_nomad_scale': False, 'enable_pauc': False, 'nomad_init_scale': 1.0, 'weight_all_tasks': False}),
    ):
        mc = {
            **cfg_base, 'task': 'T4', 'dim': 192, 'heads': 8, 'n_local_layers': 4, 'n_global_layers': 4,
            'drop_path': 0.0, 'head_dropout': 0.0, 'enable_dense_gn2': False, 'use_fixed_event_loss': True,
            **extra_cfg,
        }
        torch.manual_seed(325)
        m_full = cls_m(geom, mc).to(device)
        torch.manual_seed(325)
        m_acc = cls_m(geom, mc).to(device)
        m_full.train()
        m_acc.train()
        o_full = torch.optim.AdamW(m_full.build_param_groups(8e-4, 0.03))
        o_acc = torch.optim.AdamW(m_acc.build_param_groups(8e-4, 0.03))

        # Full batch 128 on CPU/GPU in FP32
        o_full.zero_grad(set_to_none=True)
        df_dev = move_batch_to_device(b_full, device)
        lf, _ = _pf_v3_loss(m_full, df_dev, {'grad_blend': 1.0})
        lf.backward()
        g_full = {n: p.grad.detach().float().cpu().clone() for n, p in m_full.named_parameters() if p.grad is not None}
        o_full.step()

        # 64 x 2 exact full-window denominator accumulation
        o_acc.zero_grad(set_to_none=True)
        fd = compute_window_task_denominators([mb0, mb1], pos_weight=3.0, weight_all_tasks=bool(extra_cfg['weight_all_tasks']))
        l_acc_sum = 0.0
        for mb in (mb0, mb1):
            dm = move_batch_to_device(mb, device)
            lm, _ = _pf_v3_loss(m_acc, dm, {'grad_blend': 1.0, 'fixed_denoms': fd})
            lm.backward()
            l_acc_sum += float(lm.detach().item())
        g_acc = {n: p.grad.detach().float().cpu().clone() for n, p in m_acc.named_parameters() if p.grad is not None}
        o_acc.step()

        max_abs_g = 0.0
        max_rel_g = 0.0
        g_norm_full = math.sqrt(sum(float(v.square().sum().item()) for v in g_full.values()))
        g_norm_diff = math.sqrt(sum(float((g_full[k] - g_acc[k]).square().sum().item()) for k in g_full))
        for k in g_full:
            d = float((g_full[k] - g_acc[k]).abs().max().item())
            max_abs_g = max(max_abs_g, d)
            max_rel_g = max(max_rel_g, d / max(float(g_full[k].abs().max().item()), 1e-4))
        max_p_diff = max(
            float((p1.detach().float().cpu() - p2.detach().float().cpu()).abs().max().item())
            for p1, p2 in zip(m_full.parameters(), m_acc.parameters())
        )
        passed = bool(abs(float(lf.item()) - l_acc_sum) < 1e-4 and (g_norm_diff / max(g_norm_full, 1e-8)) < 1e-4 and max_p_diff < 1e-5)
        equiv_res[mname] = {
            'loss_full_128': float(lf.item()),
            'loss_accum_64x2': float(l_acc_sum),
            'loss_abs_diff': abs(float(lf.item()) - l_acc_sum),
            'global_grad_l2_rel_diff': float(g_norm_diff / max(g_norm_full, 1e-8)),
            'max_param_abs_grad_diff': max_abs_g,
            'max_1step_adamw_param_diff': max_p_diff,
            'passed': passed,
        }
        del m_full, m_acc, o_full, o_acc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not passed:
            raise AssertionError(f'324-02 real-model equivalence failed for {mname}: {equiv_res[mname]}')

    # 2. Stage 324-06 Bore audit + Stage 324-07 Region-stratified object-level micro-vertex PV/SV audit on CleanTune8k
    from torch.utils.data import BatchSampler, DataLoader, SequentialSampler
    val_ds = V3EvalDataset(clean_root / 'data', split='validation', config=cfg_base, item_transform=extract_pf_item)
    tune_meta = pd.read_parquet(clean_root / 'validation_meta/tune_truth_meta.parquet')
    sc_df = pd.read_parquet(clean_root / 'truth_sidecar/event_sidecar.parquet')
    sc_pos = pd.MultiIndex.from_frame(sc_df[KEY]).get_indexer(pd.MultiIndex.from_frame(tune_meta[KEY]))
    sc_tune = sc_df.iloc[sc_pos].reset_index(drop=True)

    loader = DataLoader(
        val_ds,
        batch_sampler=BatchSampler(SequentialSampler(val_ds), batch_size=128, drop_last=False),
        num_workers=8,
        collate_fn=_identity_collate,
        multiprocessing_context='spawn',
        worker_init_fn=init_worker,
        prefetch_factor=2,
    )
    n_val = len(val_ds)
    avr_pv = np.zeros((n_val, 4), dtype=np.float32)
    mv_feats = np.zeros((n_val, 16), dtype=np.float32)
    trk_lead_flag = np.zeros(n_val, dtype=np.float32)
    trk_n_in_bore = 0
    trk_n_out_bore = 0
    trk_n_2st = 0
    cursor = 0
    try:
        for items in loader:
            for loc, it in enumerate(items):
                gi = cursor + loc
                avr_pv[gi] = it['avr_pv_mm']
                mv_feats[gi] = it['micro_vtx_feat']
                cm = it['trk_cand_mask']
                tc = it['trk_cand']
                if cm.any():
                    trk_lead_flag[gi] = float(tc[0, 0])
                    for si in np.flatnonzero(cm):
                        fl = float(tc[si, 0])
                        if fl > 0.75:
                            trk_n_in_bore += 1
                        elif fl > 0.60:
                            trk_n_out_bore += 1
                        else:
                            trk_n_2st += 1
            cursor += len(items)
    finally:
        shutdown_loader(loader)
    val_ds.native.close()

    bore_audit = {
        'tune8k_total_candidates': int(trk_n_in_bore + trk_n_out_bore + trk_n_2st),
        'in_bore_3station_candidates': int(trk_n_in_bore),
        'out_of_bore_shell_3station_candidates': int(trk_n_out_bore),
        'two_station_candidates': int(trk_n_2st),
        'out_of_bore_fraction_of_3st': float(trk_n_out_bore / max(1, trk_n_in_bore + trk_n_out_bore)),
        'events_with_in_bore_lead': int((trk_lead_flag > 0.75).sum()),
        'events_with_shell_lead': int(((trk_lead_flag > 0.60) & (trk_lead_flag <= 0.75)).sum()),
    }

    # Region-stratified object-level micro-vertex PV & displaced-SV diagnostic (Front 30mm vs Interm 150.01mm!)
    v_true = np.column_stack([
        sc_tune['vertex_x_mm'].to_numpy(dtype=np.float64),
        sc_tune['vertex_y_mm'].to_numpy(dtype=np.float64),
        sc_tune['vertex_z_mm'].to_numpy(dtype=np.float64),
    ])
    front_m = tune_meta['front_pinpoint_pixel_centered'].to_numpy(dtype=bool)
    pin_m = tune_meta['pixel_centered'].to_numpy(dtype=bool)
    interm_m = pin_m & (~front_m)
    lbl7 = sc_tune['label_7'].to_numpy(dtype=np.int64)
    has_charm = sc_tune['charm_flag'].to_numpy(dtype=np.int64) > 0

    mv_audit = {}
    for rname, rmask in (('FrontPinpoint_30mm', front_m), ('IntermPinpoint_150mm', interm_m), ('AllPinpoint_PixXY', pin_m)):
        ok = rmask & (avr_pv[:, 3] > 0.5)
        n_reg = int(rmask.sum())
        dz = np.abs(avr_pv[ok, 2] - v_true[ok, 2]) if ok.any() else np.zeros(0)
        d3 = np.linalg.norm(avr_pv[ok, :3] - v_true[ok, :3], axis=1) if ok.any() else np.zeros(0)
        n_trk_ev = np.expm1(mv_feats[rmask, 0] * 3.5)
        n_3h_ev = np.expm1(mv_feats[rmask, 1] * 3.0)
        by_proc = {}
        proc_defs = [
            ('e_CC_noCharm', (lbl7 == 0) & (~has_charm)),
            ('mu_CC_noCharm', ((lbl7 == 1) | (lbl7 == 2)) & (~has_charm)),
            ('CC_charm', has_charm & (lbl7 != 6)),
            ('tau_lep', (lbl7 == 3) | (lbl7 == 4)),
            ('tau_had', lbl7 == 5),
            ('NC', lbl7 == 6),
        ]
        for pname, pm in proc_defs:
            m_p = rmask & pm
            if not m_p.any():
                continue
            by_proc[pname] = {
                'count': int(m_p.sum()),
                'avr_pv_valid_rate': float((avr_pv[m_p, 3] > 0.5).mean()),
                'sv_2v_valid_rate': float((mv_feats[m_p, 11] > 0.5).mean()),
                'mean_max_ip_um': float((50.0 * np.sinh(mv_feats[m_p, 4] * 3.0)).mean()),
                'mean_n_displaced_gt2p5': float(np.expm1(mv_feats[m_p, 8] * 3.0).mean()),
            }
        mv_audit[rname] = {
            'events': n_reg,
            'avr_pv_valid_rate': float(ok.sum() / max(1, n_reg)),
            'mean_n_tracklets': float(n_trk_ev.mean()) if n_reg else 0.0,
            'mean_n_3hit_tracklets': float(n_3h_ev.mean()) if n_reg else 0.0,
            'avr_pv_median_abs_dz_mm': float(np.median(dz)) if len(dz) else None,
            'avr_pv_median_3d_mm': float(np.median(d3)) if len(d3) else None,
            'by_process': by_proc,
        }

    rep = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': float(time.perf_counter() - t0),
        'exact_grad_accum_324_02': equiv_res,
        'spectrometer_bore_audit_324_06': bore_audit,
        'micro_vertex_object_audit_324_07': mv_audit,
    }
    write_json(STAGE325_ROOT / 'stage325_gpu_gates_and_object_audit.json', rep)
    return rep


# ---------------------------------------------------------------------------
# 3. Frozen Non-NN Baselines & Negative Controls (Saving Models + Per-Event Predictions!)
# ---------------------------------------------------------------------------
def _extract_tabular_and_occupancy_split(data_root: Path, split: str, geometry: dict, cfg: dict, workers: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Extract 70-dim physics features + 10-dim pure-occupancy counts in parallel across CPU workers."""
    from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

    ds = V3EvalDataset(data_root, split=split, config=cfg, item_transform=extract_pf_item)
    nomad_ext = NomadTransverseExtractor(geometry, phys_calibrated=True, pe_per_mip_mev=40.0)
    n = len(ds)
    feats70 = np.zeros((n, 70), dtype=np.float32)
    occ10 = np.zeros((n, 10), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    doms = np.zeros(n, dtype=np.int64)

    eff_w = min(workers, 8) if n >= 512 else 0
    lkw: dict = {
        'batch_sampler': BatchSampler(SequentialSampler(ds), batch_size=128, drop_last=False),
        'num_workers': eff_w,
        'collate_fn': _identity_collate,
    }
    if eff_w > 0:
        lkw.update(multiprocessing_context='spawn', worker_init_fn=init_worker, prefetch_factor=2)
    loader = DataLoader(ds, **lkw)
    cursor = 0
    try:
        for items in loader:
            batch = collate_pf(items, mirror_rng=None)
            with torch.inference_mode():
                nomad_t = nomad_ext.from_pf(batch).numpy()
            for loc, it in enumerate(items):
                gi = cursor + loc
                c_obs = it['c_obs']                                  # 16
                mv = it['micro_vtx_feat']                            # 16
                nm = nomad_t[loc]                                    # 10
                cand_m = it['trk_cand_mask']
                n_cand = float(cand_m.sum())
                lead_c = it['trk_cand'][0] if len(it['trk_cand']) else np.zeros(11, dtype=np.float32)
                ps = it['pix_plane_summary']  # (15, 12)
                tgt_ps = ps[:12]
                ipt_ps = ps[12:]
                ps_agg = np.array([
                    float(tgt_ps[:, 0].sum()) / 12.0,
                    float(tgt_ps[:, 1].sum()),
                    float(tgt_ps[:, 2].sum()),
                    float(tgt_ps[:, 8].sum()),
                    float(tgt_ps[:, 9].sum()),
                    float(tgt_ps[:, 10].mean()),
                    float(tgt_ps[:, 11].mean()),
                    float(tgt_ps[0, 1]),
                    float(ipt_ps[:, 0].sum()) / 3.0,
                    float(ipt_ps[:, 1].sum()),
                    float(ipt_ps[:, 2].sum()),
                    float(ipt_ps[:, 8].sum()),
                    float(ipt_ps[:, 9].sum()),
                    float(ipt_ps[:, 10].mean()),
                    float(ipt_ps[:, 11].mean()),
                    float(it['avr_pv_mm'][3]),
                ], dtype=np.float32)
                row_f = np.concatenate([c_obs, mv, nm, np.array([n_cand], dtype=np.float32), lead_c[:11], ps_agg])
                feats70[gi] = np.nan_to_num(row_f, nan=0.0, posinf=0.0, neginf=0.0)
                # Pure 10-dim occupancy-only vector (counts/active-planes/bars/tracker-stations/IPT/front-veto)
                occ10[gi] = np.array([
                    c_obs[0], c_obs[1], c_obs[2], c_obs[3], c_obs[13], c_obs[14],
                    float(tgt_ps[:, 0].sum()), float(ipt_ps[:, 0].sum()), float(tgt_ps[0, 0]), n_cand,
                ], dtype=np.float32)
                labels[gi] = int(it['label'])
                doms[gi] = int(it['obs_domain_id'])
            cursor += len(items)
    finally:
        shutdown_loader(loader)
    meta = ds.metadata[KEY + FAMILY + ['label']].copy().reset_index(drop=True)
    ds.native.close()
    return feats70, occ10, labels, doms, meta


def _ood_root_for(clean_root: Path, ov: str) -> Path:
    if ov in W184_EPPS21_OOD_VARIANTS:
        return STAGE325_ROOT / 'ood_shards' / ov
    return clean_root / 'ood_shards' / ov


def run_frozen_baselines_stage325(clean_root: Path) -> dict:
    """Train & cryptographically freeze BDT-70d, LR-6d/16d/70d, Occupancy-10d, Energy-only (324-05/12)."""
    t0 = time.perf_counter()
    geom = json.loads((clean_root / 'geometry.json').read_text())
    cfg = {
        'pe_only': True, 'bar_threshold_pe': 2.5, 'sipm_npix': 14400.0,
        'apply_sipm_saturation': False, 'invert_sipm_saturation': True,
    }
    X_tr, O_tr, y_tr, _, tr_meta = _extract_tabular_and_occupancy_split(clean_root / 'data', 'train', geom, cfg)
    X_val, O_val, y_val, d_val, val_meta = _extract_tabular_and_occupancy_split(clean_root / 'data', 'validation', geom, cfg)

    # Energy-only negative control (true E_nu from planned_manifest for Train & tune_truth_meta for Tune)
    man_tr = pd.read_parquet(clean_root / 'planned_manifest.parquet', columns=KEY + ['Ev_GeV'])
    pos_tr = pd.MultiIndex.from_frame(man_tr[KEY]).get_indexer(pd.MultiIndex.from_frame(tr_meta[KEY]))
    E_tr = np.log1p(man_tr['Ev_GeV'].to_numpy(dtype=np.float32)[pos_tr])[:, None]

    tune_meta = pd.read_parquet(clean_root / 'validation_meta/tune_truth_meta.parquet')
    pos_va = pd.MultiIndex.from_frame(tune_meta[KEY]).get_indexer(pd.MultiIndex.from_frame(val_meta[KEY]))
    tune_al = tune_meta.iloc[pos_va].reset_index(drop=True)
    E_val = np.log1p(tune_al['energy_nu_GeV'].to_numpy(dtype=np.float32))[:, None]
    gw_val = tune_al['geom_weight'].to_numpy(dtype=np.float64)

    # Screen-excluded Tune6k mask
    scr_df = pd.read_parquet(clean_root / 'data/tune_screen.parquet', columns=KEY)
    in_scr = pd.MultiIndex.from_frame(val_meta[KEY]).isin(pd.MultiIndex.from_frame(scr_df[KEY]))
    tune6k_m = ~in_scr

    ood_data = {}
    for ov in ALL_STAGE325_OOD:
        oroot = _ood_root_for(clean_root, ov)
        if not (oroot / 'shards/000/observations.parquet').is_file():
            continue
        Xo, Oo, yo, do, ometa = _extract_tabular_and_occupancy_split(oroot, 'validation', geom, cfg)
        otm = pd.read_parquet(oroot / 'truth_meta.parquet')
        Eo = np.log1p(otm['energy_nu_GeV'].to_numpy(dtype=np.float32))[:, None]
        ood_data[ov] = (Xo, Oo, Eo, yo, do, ometa)

    bl_dir = STAGE325_ROOT / 'frozen_baselines'
    bl_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ('ctrl_energy_only_1d', 'energy', None, 'bdt'),
        ('ctrl_occupancy_only_10d', 'occ', list(range(10)), 'bdt'),
        ('lr_presence_counts_6d', 'x70', [0, 1, 2, 3, 13, 14], 'lr'),
        ('lr_c_obs_16d', 'x70', list(range(16)), 'lr'),
        ('lr_full_obs_physics_70d', 'x70', list(range(70)), 'lr'),
        ('bdt_histgb_full_obs_70d', 'x70', list(range(70)), 'bdt'),
    ]

    summary = {}
    for name, src_kind, cols, model_kind in specs:
        def _sel(X, O, E):
            if src_kind == 'energy':
                return E
            if src_kind == 'occ':
                return O[:, cols]
            return X[:, cols]

        A_tr = _sel(X_tr, O_tr, E_tr)
        A_va = _sel(X_val, O_val, E_val)
        if model_kind == 'lr':
            scaler = StandardScaler()
            clf = LogisticRegression(C=1.0, max_iter=1000, random_state=325)
            clf.fit(scaler.fit_transform(A_tr), y_tr)
            artifact = {'scaler': scaler, 'clf': clf, 'src_kind': src_kind, 'cols': cols}
            pred_fn = lambda A: clf.predict_proba(scaler.transform(A))
        else:
            clf = HistGradientBoostingClassifier(
                max_iter=350, max_depth=8, learning_rate=0.06, min_samples_leaf=30, l2_regularization=1.0, random_state=325
            )
            clf.fit(A_tr, y_tr)
            artifact = {'clf': clf, 'src_kind': src_kind, 'cols': cols}
            pred_fn = lambda A: clf.predict_proba(A)

        p_val = pred_fn(A_va)
        mdl_p = bl_dir / f'{name}.pkl'
        with mdl_p.open('wb') as f:
            pickle.dump(artifact, f, protocol=5)
        pred_df = val_meta.copy()
        pred_df['obs_domain_id'] = d_val
        for c in range(4):
            pred_df[f'p{c}'] = p_val[:, c].astype(np.float32)
        pred_p = bl_dir / f'{name}_tune8k_predictions.parquet'
        pred_df.to_parquet(pred_p, index=False, compression=COMPRESSION)

        ood_res = {}
        for ov, (Xo, Oo, Eo, yo, do, _) in ood_data.items():
            po = pred_fn(_sel(Xo, Oo, Eo))
            if 'nc_' in ov:
                ood_res[ov] = {
                    'n_events': int(len(yo)),
                    'mean_p_tau': float(po[:, 2].mean()),
                    'frac_pred_nc': float((po.argmax(1) == 3).mean()),
                    'frac_pred_tau': float((po.argmax(1) == 2).mean()),
                }
            else:
                ood_res[ov] = _ba4_and_nll(yo, po)

        summary[name] = {
            'model_path': str(mdl_p),
            'model_sha256': _sha256_file(mdl_p),
            'tune8k_pred_path': str(pred_p),
            'tune8k_pred_sha256': _sha256_file(pred_p),
            'tune8k_all': _ba4_and_nll(y_val, p_val),
            'tune6k_screen_excl': _ba4_and_nll(y_val[tune6k_m], p_val[tune6k_m]),
            'tune8k_mw2t_BA4': _weighted_ba4(y_val, p_val, gw_val),
            'tune6k_split_half_all': _split_half_tail_eval(p_val[:, 2], y_val, val_meta, tune6k_m),
            'tune6k_split_half_D1': _split_half_tail_eval(p_val[:, 2], y_val, val_meta, tune6k_m & (d_val == 1)),
            'ood_validation': ood_res,
        }

    out = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': float(time.perf_counter() - t0),
        'train_events': int(len(y_tr)),
        'tune_events': int(len(y_val)),
        'baselines': summary,
    }
    write_json(STAGE325_ROOT / 'stage325_frozen_baselines.json', out)
    return out


# ---------------------------------------------------------------------------
# 4. Stage 325 Matched-Budget, Single-Commit, Exact-Accumulation, FP32-Eval Training Matrix
# ---------------------------------------------------------------------------
def build_stage325_configs(clean_root: Path) -> dict[str, dict]:
    cfg_dir = STAGE325_ROOT / 'configs'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    data_root = str(clean_root / 'data')
    geom_json = str(clean_root / 'geometry.json')
    tune_screen = str(clean_root / 'data/tune_screen.parquet')
    sidecar_root = str(clean_root / 'truth_sidecar')
    emw_v2 = str(STAGE325_ROOT / 'train_energy_match_weights_v2.parquet')
    base_train_mod = 'analysis.flavour.v3.v3_multitask.train'

    def _base_pf(name: str, seed: int) -> dict:
        return {
            'name': name,
            'seed': int(seed),
            'task': 'T4',
            'data_root': data_root,
            'geometry_json': geom_json,
            'tune_screen_path': tune_screen,
            'sidecar_root': sidecar_root,
            'quantize_xy_mm': None,
            'quantize_target_only': True,
            'quantize_target_xy_mm': None,
            'quantize_ipt_xy_mm': None,
            'quantize_cluster_coords_only_mm': None,
            'shuffle_micro_vertex': False,
            'dim': 192,
            'heads': 8,
            'n_local_layers': 4,
            'n_global_layers': 4,
            'drop_path': 0.10,
            'head_dropout': 0.15,
            'passes': 20,
            'batch_size': 128,
            'learning_rate': 8.0e-4,
            'min_lr': 1.0e-5,
            'weight_decay': 0.03,
            'warmup_fraction': 0.06,
            'ema_decay': 0.999,
            'bf16': True,
            'eval_fp32': True,
            'screen_every_passes': 2,
            'cache_workers': 6,
            'loader_workers': 2,
            'pe_only': True,
            'bar_threshold_pe': 2.5,
            'sipm_npix': 14400.0,
            'apply_sipm_saturation': False,
            'invert_sipm_saturation': True,
            'y_mirror': False,
            'enable_dense_gn2': False,
            'load_cluster_sidecar': False,
            'use_fixed_event_loss': True,
            'weight_all_tasks': False,
            'filter_zero_weight_events': False,
            'train_module': base_train_mod,
        }

    specs = [
        ('s325-pf-20um', 'v325_pf_20um', {'enable_micro_vertex': False}, base_train_mod),
        ('s325-pf-20um-coord1mm', 'v325_pf_20um_coord1mm', {'enable_micro_vertex': False, 'quantize_cluster_coords_only_mm': 1.00}, base_train_mod),
        ('s325-pf-hybrid-v2', 'v325_pf_hybrid_v2', {'enable_micro_vertex': True}, base_train_mod),
        ('s325-pf-hybrid-v2-shuffled', 'v325_pf_hybrid_v2_shuffled', {'enable_micro_vertex': True, 'shuffle_micro_vertex': True}, base_train_mod),
        ('s325-pf-hybrid-v2-coord1mm', 'v325_pf_hybrid_v2_coord1mm', {'enable_micro_vertex': True, 'quantize_cluster_coords_only_mm': 1.00}, base_train_mod),
        (
            's325-pf-hybrid-v2-nomad',
            'v325_pf_hybrid_v2_nomad',
            {
                'enable_micro_vertex': True,
                'model_family': 'pf_v4',
                'enable_nomad': True,
                'legacy_nomad_scale': False,
                'enable_pauc': False,
                'nomad_init_scale': 1.0,
                'train_module': 'analysis.flavour.v3.v4_tau.train',
            },
            'analysis.flavour.v3.v4_tau.train',
        ),
        (
            's325-pf-hybrid-v2-reg',
            'v325_pf_hybrid_v2_reg',
            {'enable_micro_vertex': True, 'drop_path': 0.18, 'head_dropout': 0.25, 'weight_decay': 0.06},
            base_train_mod,
        ),
        (
            's325-pf-20um-ematched-v2',
            'v325_pf_20um_ematched_v2',
            {
                'enable_micro_vertex': False,
                'sample_weight_path': emw_v2,
                'weight_all_tasks': True,
                'filter_zero_weight_events': True,
            },
            base_train_mod,
        ),
    ]

    groups: dict[str, dict] = {}
    for gname, prefix, overrides, tmod in specs:
        c_paths = []
        for s in SEEDS:
            c = _base_pf(f'{prefix}_s{s}', s)
            c.update(overrides)
            p = cfg_dir / f"{c['name']}.json"
            write_json(p, c)
            c_paths.append(p)
        groups[gname] = {'prefix': prefix, 'configs': c_paths, 'share': '--share-pf-inputs', 'train_module': tmod}

    b1_paths = []
    for s in SEEDS:
        c = {
            'name': f'v325_b1pp_clean_s{s}',
            'seed': int(s),
            'task': 'T4',
            'data_root': data_root,
            'geometry_json': geom_json,
            'tune_screen_path': tune_screen,
            'dim': 160,
            'n_long_layers': 4,
            'heads': 8,
            'drop_path': 0.10,
            'channel_dropout': 0.03,
            'passes': 20,
            'batch_size': 128,
            'learning_rate': 9.0e-4,
            'min_lr': 1.0e-5,
            'weight_decay': 0.03,
            'warmup_fraction': 0.06,
            'ema_decay': 0.999,
            'bf16': True,
            'eval_fp32': True,
            'screen_every_passes': 2,
            'cache_workers': 4,
            'loader_workers': 2,
            'pe_only': True,
            'bar_threshold_pe': 2.5,
            'sipm_npix': 14400.0,
            'apply_sipm_saturation': False,
            'invert_sipm_saturation': True,
            'y_mirror': False,
            'train_module': 'analysis.flavour.v3.v1_b1pp.train',
        }
        p = cfg_dir / f"{c['name']}.json"
        write_json(p, c)
        b1_paths.append(p)
    groups['s325-b1pp-clean'] = {
        'prefix': 'v325_b1pp_clean',
        'configs': b1_paths,
        'share': '--share-b1pp-inputs',
        'train_module': 'analysis.flavour.v3.v1_b1pp.train',
    }
    return groups


def submit_stage325_group(
    clean_root: Path,
    group_name: str,
    backend: str,
    parallel: int = 3,
    kubeflow_flavor: str = 'h100-94gb',
    extra_groups: list[str] | None = None,
    min_gpu_mem: int = 36000,
) -> dict:
    groups = build_stage325_configs(clean_root)
    spec = groups[group_name]
    all_cfgs = list(spec['configs'])
    job_name = group_name
    if extra_groups:
        for eg in extra_groups:
            all_cfgs.extend(groups[eg]['configs'])
        job_name = f'{group_name}-pack'
    cmd = [
        'analysis.flavour.v3.common.group',
        '--root', str(STAGE325_ROOT),
        '--name', job_name,
        '--parallel', str(parallel),
        spec['share'],
        '--cache-workers', '6',
        '--train-module', spec['train_module'],
        '--configs', *[str(p) for p in all_cfgs],
    ]
    if backend == 'kubeflow':
        return submit_job(
            root=STAGE325_ROOT,
            name=job_name,
            stage='stage325',
            command=cmd,
            backend='kubeflow',
            flavor=kubeflow_flavor,
            gpu=True,
            deadline_seconds=14400,
        )
    return submit_job(
        root=STAGE325_ROOT,
        name=job_name,
        stage='stage325',
        command=cmd,
        backend='condor',
        cpus=6,
        memory='24000MB',
        disk='20GB',
        gpu=True,
        min_gpu_memory_mb=min_gpu_mem,
        min_capability=8.0,
        job_flavour='workday',
        allow_backfill=True,
    )


# ---------------------------------------------------------------------------
# 5. Zero-Seal-Access Freeze, Comprehensive Diagnostics, Stress Tests & Figures
# ---------------------------------------------------------------------------
def _perturb_observation_for_stress(
    obs: dict,
    bounds: dict,
    rng: np.random.Generator,
    *,
    pixel_dead: float = 0.0,
    pixel_dark_rate: float = 0.0,
    tracker_dead: float = 0.0,
    tracker_noise_per_station: float = 0.0,
) -> dict:
    ob = {
        'pixel': {k: list(v) for k, v in obs['pixel'].items()},
        'bar': {k: list(v) for k, v in obs['bar'].items()},
        'tracker_hypothesis': {k: list(v) for k, v in obs['tracker_hypothesis'].items()},
    }
    if pixel_dead > 0.0 and len(ob['pixel']['channel_id']) > 0:
        keep = rng.random(len(ob['pixel']['channel_id'])) >= pixel_dead
        ob['pixel']['channel_id'] = [x for x, m in zip(ob['pixel']['channel_id'], keep) if m]
        ob['pixel']['xyz_mm'] = [x for x, m in zip(ob['pixel']['xyz_mm'], keep) if m]
    if pixel_dark_rate > 0.0:
        nx, ny, nz = int(bounds['nx']), int(bounds['ny']), int(bounds['nz'])
        mean_per_plane = pixel_dark_rate * float(nx * ny)
        px_min, py_min = float(bounds['px_min']), float(bounds['py_min'])
        dx, dy = float(bounds['dx_mm']), float(bounds['dy_mm'])
        pzs = bounds['pixel_Zpos']
        cids = ob['pixel']['channel_id']
        xyzs = ob['pixel']['xyz_mm']
        for p in range(nz):
            nf = int(rng.poisson(mean_per_plane))
            if nf <= 0:
                continue
            cs = rng.integers(0, nx, size=nf)
            rs = rng.integers(0, ny, size=nf)
            for c, r in zip(cs, rs):
                cids.append(int((p * ny + int(r)) * nx + int(c)))
                xyzs.append([px_min + (int(c) + 0.5) * dx, py_min + (int(r) + 0.5) * dy, float(pzs[p])])
    if tracker_dead > 0.0 and len(ob['tracker_hypothesis']['station']) > 0:
        keep = rng.random(len(ob['tracker_hypothesis']['station'])) >= tracker_dead
        for k in ('station', 'xy_mm', 'sigma_xy_mm'):
            if k in ob['tracker_hypothesis']:
                ob['tracker_hypothesis'][k] = [x for x, m in zip(ob['tracker_hypothesis'][k], keep) if m]
    if tracker_noise_per_station > 0.0:
        for st in (0, 1, 2):
            nf = int(rng.poisson(tracker_noise_per_station))
            for _ in range(nf):
                ob['tracker_hypothesis']['station'].append(st)
                ob['tracker_hypothesis']['xy_mm'].append([float(rng.uniform(-125.0, 125.0)), float(rng.uniform(-110.0, 140.0))])
                if 'sigma_xy_mm' in ob['tracker_hypothesis']:
                    ob['tracker_hypothesis']['sigma_xy_mm'].append([0.58, 0.017])
    return ob


def _load_model_fp32(ckpt_p: Path, cfg: dict, geometry: dict, device: torch.device) -> tuple[torch.nn.Module, bool]:
    is_b1 = ('v1_b1pp' in str(cfg.get('train_module', ''))) or (cfg.get('model_family') == 'b1pp')
    state = torch.load(ckpt_p, map_location=device, weights_only=False)
    if is_b1:
        model: torch.nn.Module = ZAlignedStereoNet(static_plane_layout(geometry), cfg).to(device)
    elif cfg.get('model_family') == 'pf_v4':
        model = PinpointFormerV4(geometry, cfg).to(device)
    else:
        model = PinpointFormerMultitask(geometry, cfg).to(device)
    model.load_state_dict(state['model'])
    model.eval()
    return model, is_b1


def _predict_batches_fp32(model: torch.nn.Module, is_b1: bool, batches: list[dict], device: torch.device) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    p_chunks, vh_chunks, lv_chunks = [], [], []
    with torch.inference_mode():
        for b_cpu in batches:
            if is_b1:
                gb = device_batch(b_cpu, device)
                p_chunks.append(probabilities(model(gb).float(), 'T4').cpu().numpy())
            else:
                gb = move_batch_to_device(b_cpu, device)
                out = model(gb, grad_blend=1.0, return_aux=False)
                p_chunks.append(out['probs4'].float().cpu().numpy())
                vh_chunks.append(out['v_hat'].float().cpu().numpy())
                lv_chunks.append(out['log_var_v'].float().cpu().numpy())
    probs = np.concatenate(p_chunks, axis=0)
    vhat = np.concatenate(vh_chunks, axis=0) if vh_chunks else None
    lvar = np.concatenate(lv_chunks, axis=0) if lv_chunks else None
    return probs, vhat, lvar


def freeze_and_validate_stage325(clean_root: Path) -> dict:
    """Zero-seal-access cryptographic freeze + Tune6k/OOD/Stress/PV-pull/Obs-only closure (Stage 325)."""
    t0 = time.perf_counter()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    split_shards = json.loads((clean_root / 'data/split_shards.json').read_text())
    sealed_names = sorted(
        set(split_shards.get('cal_a', [])) | set(split_shards.get('cut_b', [])) | set(split_shards.get('confirm', []))
    )
    sealed_prefixes = tuple(str((clean_root / 'data/shards' / s).resolve()) for s in sealed_names)
    sealed_meta_dir = str((clean_root / 'sealed_truth_meta').resolve())
    sealed_accesses: list[tuple[str, str]] = []

    def _seal_audit_hook(event: str, args: tuple) -> None:
        if event in ('open', 'os.scandir', 'os.listdir', 'os.stat'):
            for a in args:
                if isinstance(a, (str, bytes, os.PathLike)):
                    s = os.fsdecode(a)
                    if sealed_meta_dir in s or any(p in s for p in sealed_prefixes):
                        sealed_accesses.append((event, s))

    sys.addaudithook(_seal_audit_hook)

    groups = build_stage325_configs(clean_root)
    geom = json.loads((clean_root / 'geometry.json').read_text())
    bounds = detector_bounds(geom)
    tune_meta = pd.read_parquet(clean_root / 'validation_meta/tune_truth_meta.parquet')
    sidecar_df = pd.read_parquet(clean_root / 'truth_sidecar/event_sidecar.parquet')
    scr_df = pd.read_parquet(clean_root / 'data/tune_screen.parquet', columns=KEY)
    in_screen = pd.MultiIndex.from_frame(tune_meta[KEY]).isin(pd.MultiIndex.from_frame(scr_df[KEY]))
    tune6k_m = ~in_screen

    labels = tune_meta['label'].to_numpy(dtype=np.int64)
    obs_dom = tune_meta['obs_domain_id'].to_numpy(dtype=np.int64)
    geom_w = tune_meta['geom_weight'].to_numpy(dtype=np.float64)
    ev_tune = tune_meta['energy_nu_GeV'].to_numpy(dtype=np.float64)
    # Common-support [50, 2312] GeV + matched (E_nu, nu/nubar) weights for Stage 324-03/12 validation!
    common_e_m = (ev_tune >= 50.0) & (ev_tune <= 2312.0)
    nu_sign = (tune_meta['init_pdg'].to_numpy(dtype=np.int64) > 0).astype(np.int64) if 'init_pdg' in tune_meta.columns else np.ones(len(tune_meta), dtype=np.int64)
    e_edges = np.geomspace(50.0, 2312.0, 11)
    eb_t = np.clip(np.digitize(ev_tune, e_edges) - 1, 0, 9)
    cell_t = eb_t * 2 + nu_sign
    p_cell = np.zeros((4, 20), dtype=np.float64)
    for c in range(4):
        mc = (labels == c) & common_e_m & tune6k_m
        cnt = np.bincount(cell_t[mc], minlength=20).astype(np.float64)
        p_cell[c] = (cnt + 0.5) / max(1.0, cnt.sum() + 10.0)
    p_cell_tgt = np.power(np.prod(p_cell, axis=0), 0.25)
    p_cell_tgt /= max(float(p_cell_tgt.sum()), 1e-12)
    matched_val_w = np.zeros(len(tune_meta), dtype=np.float64)
    for c in range(4):
        mc = (labels == c) & common_e_m & tune6k_m
        rw = np.clip(p_cell_tgt[cell_t[mc]] / np.maximum(p_cell[c, cell_t[mc]], 1e-9), 0.1, 10.0)
        rw /= max(float(rw.sum()), 1e-12)
        matched_val_w[mc] = rw

    folds, _ = grouped_folds(tune_meta[KEY + FAMILY + ['label']].reset_index(drop=True), folds=5, seed=20263250)
    slices = {
        'All_Tune8k': np.ones(len(tune_meta), dtype=bool),
        'PinpointBlock_PixXY': tune_meta['pixel_centered'].to_numpy(dtype=bool),
        'FrontPinpoint_PixXY': tune_meta['front_pinpoint_pixel_centered'].to_numpy(dtype=bool),
        'D_pix_contained': obs_dom == 1,
    }
    pc_m = slices['PinpointBlock_PixXY']
    fp_m = slices['FrontPinpoint_PixXY']
    interm_m = pc_m & (~fp_m)
    fam_codes = pd.MultiIndex.from_frame(tune_meta[FAMILY]).factorize()[0]
    ref_idx = pd.MultiIndex.from_frame(tune_meta[KEY])

    checkpoints = {}
    tune_summary = {}
    raw_probs_by_group: dict[str, np.ndarray] = {}
    oof_probs_by_group: dict[str, np.ndarray] = {}

    for gname, spec in groups.items():
        prefix = spec['prefix']
        m_frames = {}
        raw_list = []
        for s in SEEDS:
            sname = f'{prefix}_s{s}'
            exp_dir = STAGE325_ROOT / 'experiments' / sname
            df, info = _select_seed_prediction_df(exp_dir)
            m_frames[sname] = df
            p_cols = ['p_0', 'p_1', 'p_2', 'p_3'] if 'p_0' in df.columns else ['p0', 'p1', 'p2', 'p3']
            pos = pd.MultiIndex.from_frame(df[KEY]).get_indexer(ref_idx)
            raw_list.append(df.iloc[pos][p_cols].to_numpy(dtype=np.float64))

            man = json.loads((exp_dir / 'manifest.json').read_text())
            ckpt_file = exp_dir / ('best.pt' if info['chosen_file'] == 'best_predictions.parquet' else 'ema.pt')
            pred_file = exp_dir / info['chosen_file']
            cfg_file = STAGE325_ROOT / 'configs' / f'{sname}.json'
            checkpoints[sname] = {
                'group': gname,
                'path': str(ckpt_file),
                'sha256': _sha256_file(ckpt_file),
                'chosen_prediction': info['chosen_file'],
                'prediction_sha256': _sha256_file(pred_file),
                'config_path': str(cfg_file),
                'config_sha256': _sha256_file(cfg_file),
                'GPU': man.get('GPU'),
                'grad_accum': man.get('grad_accum', 1),
                'eval_precision': man.get('eval_precision', 'fp32'),
                'code_archive_sha256': man.get('code_archive_sha256', ''),
                'UTC_started': man.get('UTC_started'),
                'UTC_finished': man.get('UTC_finished'),
                'best_pass': man.get('best_pass'),
                'train_events': man.get('train_events', man.get('train_rows')),
            }
        p_raw = np.mean(raw_list, axis=0)
        ens = fit_oof_domain_ensemble(m_frames, tune_meta, slices, n_folds=5, folds=folds)
        p_oof = ens['oof_probs']
        raw_probs_by_group[gname] = p_raw
        oof_probs_by_group[gname] = p_oof

        vtx = _compute_vertex_audit(list(m_frames.values()), tune_meta, sidecar_df)
        if vtx is not None:
            vtx.pop('_dz_pix_centered', None)
            # Add Front vs Intermediate Pinpoint PV resolution breakdown (324-07)
            s_pos = pd.MultiIndex.from_frame(sidecar_df[KEY]).get_indexer(ref_idx)
            sc_al = sidecar_df.iloc[s_pos].reset_index(drop=True)
            v_valid = sc_al['vertex_valid'].to_numpy(dtype=int) > 0
            vx_t, vy_t, vz_t = sc_al['vertex_x_mm'].to_numpy(np.float64), sc_al['vertex_y_mm'].to_numpy(np.float64), sc_al['vertex_z_mm'].to_numpy(np.float64)
            vx_h = np.mean([df.iloc[pd.MultiIndex.from_frame(df[KEY]).get_indexer(ref_idx)]['v_hat_x_mm'].to_numpy(np.float64) for df in m_frames.values()], axis=0)
            vy_h = np.mean([df.iloc[pd.MultiIndex.from_frame(df[KEY]).get_indexer(ref_idx)]['v_hat_y_mm'].to_numpy(np.float64) for df in m_frames.values()], axis=0)
            vz_h = np.mean([df.iloc[pd.MultiIndex.from_frame(df[KEY]).get_indexer(ref_idx)]['v_hat_z_mm'].to_numpy(np.float64) for df in m_frames.values()], axis=0)
            for rtag, rmk in (('FrontPinpoint_30mm', fp_m & v_valid), ('IntermPinpoint_150mm', interm_m & v_valid)):
                if rmk.any():
                    r3 = np.sqrt((vx_h[rmk] - vx_t[rmk]) ** 2 + (vy_h[rmk] - vy_t[rmk]) ** 2 + (vz_h[rmk] - vz_t[rmk]) ** 2)
                    vtx[rtag] = {
                        'count': int(rmk.sum()),
                        'median_abs_dz_mm': float(np.median(np.abs(vz_h[rmk] - vz_t[rmk]))),
                        'median_3d_mm': float(np.median(r3)),
                    }

        dom_raw = {}
        for did in (1, 2, 3):
            dm = tune6k_m & (obs_dom == did)
            if dm.any():
                dom_raw[OBS_DOMAIN_NAMES[did]] = _ba4_and_nll(labels[dm], p_raw[dm])

        tune_summary[gname] = {
            'prefix': prefix,
            'raw_tune8k_all': _ba4_and_nll(labels, p_raw),
            'raw_tune6k_screen_excl': _ba4_and_nll(labels[tune6k_m], p_raw[tune6k_m]),
            'raw_tune6k_common_E_nu_matched_BA4': _weighted_ba4(labels, p_raw, matched_val_w),
            'raw_tune6k_pinpoint': _ba4_and_nll(labels[tune6k_m & pc_m], p_raw[tune6k_m & pc_m]),
            'raw_tune6k_front_pinpoint': _ba4_and_nll(labels[tune6k_m & fp_m], p_raw[tune6k_m & fp_m]),
            'raw_tune6k_interm_pinpoint': _ba4_and_nll(labels[tune6k_m & interm_m], p_raw[tune6k_m & interm_m]),
            'raw_tune6k_by_obs_domain': dom_raw,
            'oof_tune8k_all': _ba4_and_nll(labels, p_oof),
            'oof_tune8k_pinpoint': _ba4_and_nll(labels[pc_m], p_oof[pc_m]),
            'mass_weighted_2t_raw_BA4': _weighted_ba4(labels, p_raw, geom_w),
            'tune6k_split_half_all': _split_half_tail_eval(p_raw[:, 2], labels, tune_meta, tune6k_m),
            'tune6k_split_half_D1_obs': _split_half_tail_eval(p_raw[:, 2], labels, tune_meta, tune6k_m & (obs_dom == 1)),
            'tune6k_split_half_pinpoint': _split_half_tail_eval(p_raw[:, 2], labels, tune_meta, tune6k_m & pc_m),
            'vertex_audit': vtx,
        }

    # Include frozen BDT-70d in paired Tune6k bootstrap comparisons!
    bl_json_p = STAGE325_ROOT / 'stage325_frozen_baselines.json'
    bl_doc = json.loads(bl_json_p.read_text()) if bl_json_p.is_file() else None
    if bl_doc and 'bdt_histgb_full_obs_70d' in bl_doc['baselines']:
        bdt_df = pd.read_parquet(bl_doc['baselines']['bdt_histgb_full_obs_70d']['tune8k_pred_path'])
        bpos = pd.MultiIndex.from_frame(bdt_df[KEY]).get_indexer(ref_idx)
        raw_probs_by_group['bdt_histgb_full_obs_70d'] = bdt_df.iloc[bpos][['p0', 'p1', 'p2', 'p3']].to_numpy(dtype=np.float64)

    # Paired 1,000-resample family bootstrap on STRICTLY SCREEN-EXCLUDED Tune6k (raw 3-seed average!)
    pairs = [
        ('PF20um - BDT70d (Tune6k All)', 's325-pf-20um', 'bdt_histgb_full_obs_70d', tune6k_m),
        ('PF20um - B1++ (Tune6k All)', 's325-pf-20um', 's325-b1pp-clean', tune6k_m),
        ('PF20um - PF20um_Coord1mm (Tune6k Pinpoint)', 's325-pf-20um', 's325-pf-20um-coord1mm', tune6k_m & pc_m),
        ('HybridV2 - PF20um (Tune6k All)', 's325-pf-hybrid-v2', 's325-pf-20um', tune6k_m),
        ('HybridV2 - PF20um (Tune6k Pinpoint)', 's325-pf-hybrid-v2', 's325-pf-20um', tune6k_m & pc_m),
        ('HybridV2 - HybridV2_Coord1mm (Tune6k Pinpoint, 324-08)', 's325-pf-hybrid-v2', 's325-pf-hybrid-v2-coord1mm', tune6k_m & pc_m),
        ('HybridV2 - HybridV2_Shuffled (Tune6k All)', 's325-pf-hybrid-v2', 's325-pf-hybrid-v2-shuffled', tune6k_m),
        ('HybridV2 - HybridV2_Shuffled (Tune6k Pinpoint)', 's325-pf-hybrid-v2', 's325-pf-hybrid-v2-shuffled', tune6k_m & pc_m),
        ('HybridV2_NOMAD - HybridV2 (Tune6k All)', 's325-pf-hybrid-v2-nomad', 's325-pf-hybrid-v2', tune6k_m),
        ('HybridV2_Reg - HybridV2 (Tune6k All, 324-11)', 's325-pf-hybrid-v2-reg', 's325-pf-hybrid-v2', tune6k_m),
        ('Natural - EMatchV2 (Tune6k CommonSupp, 324-03)', 's325-pf-20um', 's325-pf-20um-ematched-v2', tune6k_m & common_e_m),
    ]
    paired_boot = {}
    for pname, ga, gb, mk in pairs:
        if ga not in raw_probs_by_group or gb not in raw_probs_by_group:
            continue
        pa_m = raw_probs_by_group[ga]
        pb_m = raw_probs_by_group[gb]
        paired_boot[pname] = paired_family_bootstrap_delta(
            pa_m[:, 2], pb_m[:, 2], pa_m, pb_m, labels, fam_codes, mk, n_boot=1000, seed=20263250
        )

    # Re-evaluate Stage 323 6T model strictly on its OWN true observable 6T domain (Stage 324-08!)
    s323_6t_dir = clean_root / 'stage323/experiments'
    eval_6t_own_dom = None
    if (s323_6t_dir / 'v3_pf_layers_6t_s17/best_predictions.parquet').is_file():
        cfg6t = {'pe_only': True, 'active_pixel_planes': [0, 2, 4, 6, 8, 10, 12, 13, 14]}
        ds6t = V3EvalDataset(clean_root / 'data', split='validation', config=cfg6t, item_transform=None)
        dom6t = np.zeros(len(ds6t), dtype=np.int64)
        for i in range(len(ds6t)):
            dom6t[i] = int(ds6t[i]['obs_domain_id'])
        m6t_idx = pd.MultiIndex.from_frame(ds6t.metadata[KEY]).get_indexer(ref_idx)
        dom6t = dom6t[m6t_idx]
        ds6t.native.close()
        dfs_6t = []
        for s in SEEDS:
            df_s, _ = _select_seed_prediction_df(s323_6t_dir / f'v3_pf_layers_6t_s{s}')
            pos = pd.MultiIndex.from_frame(df_s[KEY]).get_indexer(ref_idx)
            dfs_6t.append(df_s.iloc[pos][['p_0', 'p_1', 'p_2', 'p_3']].to_numpy(dtype=np.float64))
        p6t_raw = np.mean(dfs_6t, axis=0)
        eval_6t_own_dom = {
            'domain_counts_6t_obs': {OBS_DOMAIN_NAMES[d]: int((dom6t == d).sum()) for d in (0, 1, 2, 3)},
            'raw_tune6k_all': _ba4_and_nll(labels[tune6k_m], p6t_raw[tune6k_m]),
            'raw_tune6k_by_own_6t_obs_domain': {
                OBS_DOMAIN_NAMES[d]: _ba4_and_nll(labels[tune6k_m & (dom6t == d)], p6t_raw[tune6k_m & (dom6t == d)])
                for d in (1, 2, 3) if (tune6k_m & (dom6t == d)).any()
            },
        }

    # PV Pull Calibration Audit (324-13) on 512 Tune events via FP32 inference (returns v_hat & log_var_v)
    pv_pull_audit = {}
    s_pos = pd.MultiIndex.from_frame(sidecar_df[KEY]).get_indexer(ref_idx)
    sc_al = sidecar_df.iloc[s_pos].reset_index(drop=True)
    obs_shard080 = pq.read_table(clean_root / 'data/shards/080/observations.parquet').to_pylist()
    prov_shard080 = pd.read_parquet(clean_root / 'data/shards/080/provenance.parquet')
    pos80 = pd.MultiIndex.from_frame(sidecar_df[KEY]).get_indexer(pd.MultiIndex.from_frame(prov_shard080[KEY]))
    sc80 = sidecar_df.iloc[pos80].reset_index(drop=True)
    v_true_80 = np.column_stack([sc80['vertex_x_mm'], sc80['vertex_y_mm'], sc80['vertex_z_mm']]).astype(np.float64)
    v_ok_80 = sc80['vertex_valid'].to_numpy(dtype=int) > 0
    cfg_eval_base = {'pe_only': True, 'bar_threshold_pe': 2.5, 'sipm_npix': 14400.0, 'apply_sipm_saturation': False, 'invert_sipm_saturation': True}
    items80_pf = []
    for ii, ob in enumerate(obs_shard080):
        dm_i = compute_observable_containment(ob, geom, bounds, config=cfg_eval_base)
        items80_pf.append(extract_pf_item(ii, ob, None, dm_i, geom, bounds, cfg_eval_base))
    batches80 = [collate_pf(items80_pf[st : st + 64], mirror_rng=None) for st in range(0, len(items80_pf), 64)]

    for g in ('s325-pf-20um', 's325-pf-hybrid-v2', 's325-pf-hybrid-v2-nomad'):
        sname = f"{groups[g]['prefix']}_s17"
        cfg_s = json.loads(Path(checkpoints[sname]['config_path']).read_text())
        mdl, is_b1 = _load_model_fp32(Path(checkpoints[sname]['path']), cfg_s, geom, device)
        _, vh80, lv80 = _predict_batches_fp32(mdl, is_b1, batches80, device)
        del mdl
        if vh80 is not None and lv80 is not None:
            # VERTEX_AXIS_SCALES_MM = (0.20, 0.20, 2.50); sigma_hat = scale * exp(0.5 * clamp(lv, -6, 6))
            scales = np.array([0.20, 0.20, 2.50], dtype=np.float64)
            sig = scales[None, :] * np.exp(0.5 * np.clip(lv80, -6.0, 6.0))
            pull = (vh80[v_ok_80] - v_true_80[v_ok_80]) / np.maximum(sig[v_ok_80], 1e-6)
            pv_pull_audit[sname] = {
                'n_events': int(v_ok_80.sum()),
                'axis_pull_mean_xyz': [float(np.mean(pull[:, a])) for a in range(3)],
                'axis_pull_std_xyz': [float(np.std(pull[:, a])) for a in range(3)],
                'axis_pull_p68_abs_xyz': [float(np.quantile(np.abs(pull[:, a]), 0.68)) for a in range(3)],
                'axis_median_sigma_mm_xyz': [float(np.median(sig[v_ok_80, a])) for a in range(3)],
            }

    # All OOD / W-184 EPPS21 NC Charm FP32 Evaluation (reusing tokenized caches across seeds & groups)
    ood_eval_groups = list(groups.keys())
    ood_results: dict[str, dict] = {g: {} for g in ood_eval_groups}
    ood_batch_cache: dict[str, list[dict]] = {}
    ood_fingerprints: dict[str, str] = {}
    for ov in ALL_STAGE325_OOD:
        oroot = _ood_root_for(clean_root, ov)
        obs_p = oroot / 'shards/000/observations.parquet'
        if not obs_p.is_file():
            continue
        ood_fingerprints[ov] = _sha256_file(obs_p)
        ometa = pd.read_parquet(oroot / 'truth_meta.parquet')
        yo = ometa['label'].to_numpy(dtype=np.int64)
        for g in ood_eval_groups:
            prefix = groups[g]['prefix']
            cfg17 = json.loads(Path(checkpoints[f'{prefix}_s17']['config_path']).read_text())
            is_b1 = ('v1_b1pp' in str(cfg17.get('train_module', '')))
            ckey = json.dumps({
                'ood': str(oroot), 'is_b1': is_b1,
                'qco': cfg17.get('quantize_cluster_coords_only_mm'),
                'sh': cfg17.get('shuffle_micro_vertex', False),
            }, sort_keys=True)
            if ckey not in ood_batch_cache:
                g_json = oroot / 'shards/000/geometry.json'
                if not g_json.is_file():
                    g_json = clean_root / 'geometry.json'
                ods = V3ObservationDataset(obs_p, g_json, config=cfg17, item_transform=extract_b1pp_item if is_b1 else extract_pf_item)
                oitems = [ods[i] for i in range(len(ods))]
                ood_batch_cache[ckey] = [
                    collate_b1pp(oitems[st : st + 128], mirror_rng=None) if is_b1 else collate_pf(oitems[st : st + 128], mirror_rng=None)
                    for st in range(0, len(oitems), 128)
                ]
            ps_seeds = []
            for s in SEEDS:
                sn = f'{prefix}_s{s}'
                cs = json.loads(Path(checkpoints[sn]['config_path']).read_text())
                mdl, mb1 = _load_model_fp32(Path(checkpoints[sn]['path']), cs, geom, device)
                p_s, _, _ = _predict_batches_fp32(mdl, mb1, ood_batch_cache[ckey], device)
                ps_seeds.append(p_s)
                del mdl
            po = np.mean(ps_seeds, axis=0)
            if 'nc_' in ov:
                ood_results[g][ov] = {
                    'n_events': int(len(yo)),
                    'mean_p_tau': float(po[:, 2].mean()),
                    'frac_pred_nc': float((po.argmax(1) == 3).mean()),
                    'frac_pred_tau': float((po.argmax(1) == 2).mean()),
                    'frac_p_tau_gt_0p5': float((po[:, 2] > 0.5).mean()),
                }
            else:
                ood_results[g][ov] = _ba4_and_nll(yo, po)

    # Stage 324-10/06 Detector-Response Degradation Stress Tests on 400 real Tune events (shard 080)
    y80 = prov_shard080['label'].to_numpy(dtype=np.int64)
    stress_specs = [
        ('nominal', {}, {'pe_only': True, 'bar_threshold_pe': 2.5, 'sipm_npix': 14400.0, 'apply_sipm_saturation': False, 'invert_sipm_saturation': True}),
        ('pixel_dead_2pct', {'pixel_dead': 0.02}, cfg_eval_base),
        ('pixel_dead_5pct', {'pixel_dead': 0.05}, cfg_eval_base),
        ('pixel_dark_1e-5', {'pixel_dark_rate': 1e-5}, cfg_eval_base),
        ('tracker_dead_2pct_noise1', {'tracker_dead': 0.02, 'tracker_noise_per_station': 1.0}, cfg_eval_base),
        ('tracker_dead_5pct_noise2', {'tracker_dead': 0.05, 'tracker_noise_per_station': 2.0}, cfg_eval_base),
        ('sipm_sat_cap_0p95', {}, {**cfg_eval_base, 'sipm_invert_max_occupancy': 0.95, 'sipm_sat_cap': 0.95}),
    ]
    stress_results: dict[str, dict] = {}
    for g in ('s325-b1pp-clean', 's325-pf-20um', 's325-pf-hybrid-v2', 's325-pf-hybrid-v2-nomad'):
        sname = f"{groups[g]['prefix']}_s17"
        cfg_s = json.loads(Path(checkpoints[sname]['config_path']).read_text())
        mdl, is_b1 = _load_model_fp32(Path(checkpoints[sname]['path']), cfg_s, geom, device)
        g_st = {}
        p_nom = None
        for st_name, pkw, c_over in stress_specs:
            rng_st = np.random.default_rng(32510)
            c_use = {**cfg_s, **c_over}
            its_st = []
            for ii, ob in enumerate(obs_shard080):
                ob_p = _perturb_observation_for_stress(ob, bounds, rng_st, **pkw) if pkw else ob
                dm_i = compute_observable_containment(ob_p, geom, bounds, config=c_use)
                fn = extract_b1pp_item if is_b1 else extract_pf_item
                its_st.append(fn(ii, ob_p, None, dm_i, geom, bounds, c_use))
            bs_st = [
                collate_b1pp(its_st[st : st + 64], mirror_rng=None) if is_b1 else collate_pf(its_st[st : st + 64], mirror_rng=None)
                for st in range(0, len(its_st), 64)
            ]
            p_st, _, _ = _predict_batches_fp32(mdl, is_b1, bs_st, device)
            if st_name == 'nominal':
                p_nom = p_st
            tv = float(0.5 * np.abs(p_st - p_nom).sum(axis=1).mean()) if p_nom is not None else 0.0
            m_st = _ba4_and_nll(y80, p_st)
            g_st[st_name] = {'BA4': m_st['BA4'], 'NLL': m_st['NLL'], 'mean_TV_vs_nominal': tv}
        del mdl
        stress_results[g] = g_st

    # Stage 324-09 & 322-11: Direct disk saved FP32 predictions vs label-free observation-only replay closure!
    obs_only_closure = {}
    with tempfile.TemporaryDirectory(prefix='s325-obs-only-') as tmp:
        tmp_p = Path(tmp)
        raw64 = pq.read_table(clean_root / 'data/shards/080/observations.parquet').slice(0, 64)
        prov64 = prov_shard080.iloc[:64].reset_index(drop=True)
        pq.write_table(raw64, tmp_p / 'observations.parquet')
        (tmp_p / 'geometry.json').write_text(json.dumps(geom))
        for g in ('s325-b1pp-clean', 's325-pf-20um', 's325-pf-hybrid-v2', 's325-pf-hybrid-v2-nomad'):
            sname = f"{groups[g]['prefix']}_s17"
            ckpt_p = Path(checkpoints[sname]['path'])
            cfg_s = json.loads(Path(checkpoints[sname]['config_path']).read_text())
            df_obs = infer_observation_only(tmp_p, tmp_p / 'geometry.json', ckpt_p, cfg_s, batch_size=64, device=device)
            p_obs = df_obs[['p0', 'p1', 'p2', 'p3']].to_numpy(dtype=np.float64)

            # Compare DIRECTLY against saved disk best_predictions.parquet (proving FP32 parity, fixing 324-09!)
            saved_df = pd.read_parquet(STAGE325_ROOT / 'experiments' / sname / checkpoints[sname]['chosen_prediction'])
            p_cols = ['p_0', 'p_1', 'p_2', 'p_3'] if 'p_0' in saved_df.columns else ['p0', 'p1', 'p2', 'p3']
            pos64 = pd.MultiIndex.from_frame(saved_df[KEY]).get_indexer(pd.MultiIndex.from_frame(prov64[KEY]))
            p_saved = saved_df.iloc[pos64][p_cols].to_numpy(dtype=np.float64)
            diff_vs_saved = float(np.max(np.abs(p_obs - p_saved)))
            obs_only_closure[sname] = {
                'events': 64,
                'targets_file_present': False,
                'all_finite': bool(np.isfinite(p_obs).all()),
                'max_abs_diff_vs_saved_disk_parquet_fp32': diff_vs_saved,
                'passed': bool(np.isfinite(p_obs).all() and diff_vs_saved <= 1e-5),
            }
            if diff_vs_saved > 1e-5:
                raise AssertionError(f'324-09 FP32 saved-vs-obs parity failed for {sname}: max_diff={diff_vs_saved}')

    # Hash ONLY OPEN authorized shards (train 000..079 + validation 080..099); ZERO access to 100..149!
    open_shard_hashes = {}
    for sp_open in ('train', 'validation'):
        for sname in split_shards[sp_open]:
            sp = clean_root / 'data/shards' / sname / 'observations.parquet'
            open_shard_hashes[sname] = _sha256_file(sp)

    # Assert runtime audit hook recorded ZERO accesses to any sealed shard!
    if sealed_accesses:
        raise AssertionError(f'Stage 324-01 VIOLATION: sealed paths accessed during freeze: {sealed_accesses[:5]}')

    key_files = {
        'pinpoint_v3_1_bin': str(PINPOINT_V3_1_BIN),
        'geom_mac': str(GEOM_V3_1_MACRO),
        'geometry_json': str(clean_root / 'geometry.json'),
        'split_shards_json': str(clean_root / 'data/split_shards.json'),
        'tune_screen': str(clean_root / 'data/tune_screen.parquet'),
        'event_sidecar': str(clean_root / 'truth_sidecar/event_sidecar.parquet'),
        'tune_truth_meta': str(clean_root / 'validation_meta/tune_truth_meta.parquet'),
        'train_energy_match_weights_v2': str(STAGE325_ROOT / 'train_energy_match_weights_v2.parquet'),
    }
    file_hashes = {k: {'path': v, 'sha256': _sha256_file(Path(v))} for k, v in key_files.items() if Path(v).is_file()}
    analysis_sha = subprocess.check_output(['git', '-C', str(REPO_ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    g4_sha = subprocess.check_output(['git', '-C', str(G4_REPO_ROOT), 'rev-parse', 'HEAD'], text=True).strip()

    # Render 300-DPI English Publication Figure 1: Tune6k Comparison, 324-08 Fixed-Cluster Coord Ablation & OOD/W184 NC Charm
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), dpi=300)
    ax0, ax1, ax2 = axes

    # Panel 0: Screen-Excluded Tune6k Raw 3-Seed BA4 Ladder (Negative Controls -> BDT -> B1++ -> PF -> Hybrid v2)
    p0_items = []
    if bl_doc:
        for bk, blbl in [
            ('ctrl_energy_only_1d', 'E-only'),
            ('ctrl_occupancy_only_10d', 'Occ-10d'),
            ('bdt_histgb_full_obs_70d', 'BDT-70d'),
        ]:
            if bk in bl_doc['baselines']:
                p0_items.append((blbl, 100.0 * bl_doc['baselines'][bk]['tune6k_screen_excl']['BA4'], '#999999'))
    for gk, glbl, col in [
        ('s325-b1pp-clean', 'B1++', '#6A8CAF'),
        ('s325-pf-20um', 'PF-20um', '#4C72B0'),
        ('s325-pf-20um-ematched-v2', 'PF-EMv2', '#DD8452'),
        ('s325-pf-hybrid-v2', 'Hyb-v2', '#C44E52'),
        ('s325-pf-hybrid-v2-reg', 'Hyb-Reg', '#8172B3'),
        ('s325-pf-hybrid-v2-nomad', 'Hyb+NOM', '#55A868'),
    ]:
        p0_items.append((glbl, 100.0 * tune_summary[gk]['raw_tune6k_screen_excl']['BA4'], col))
    xp = np.arange(len(p0_items))
    ax0.bar(xp, [x[1] for x in p0_items], color=[x[2] for x in p0_items], width=0.68)
    ax0.set_xticks(xp)
    ax0.set_xticklabels([x[0] for x in p0_items], rotation=32, ha='right', fontsize=8.2)
    ax0.set_ylabel('Screen-Excluded Tune6k Raw BA4 [%]')
    ax0.set_title('Stage 325 Clean v3: Controls, BDT & Neural Models')
    ax0.set_ylim(25, 63)
    ax0.grid(True, axis='y', alpha=0.25)

    # Panel 1: Decisive Stage 324-07/08 Fixed-Cluster Coord-Only Ablation & Front/Interm PV Resolution
    p1_specs = [
        ('PF\n20um', 's325-pf-20um'),
        ('PF\n1mm-C', 's325-pf-20um-coord1mm'),
        ('Hyb-v2\n20um', 's325-pf-hybrid-v2'),
        ('Hyb-v2\n1mm-C', 's325-pf-hybrid-v2-coord1mm'),
        ('Hyb-v2\nShuf', 's325-pf-hybrid-v2-shuffled'),
        ('Hyb-v2\nNOMAD', 's325-pf-hybrid-v2-nomad'),
    ]
    x1 = np.arange(len(p1_specs))
    nll1 = [tune_summary[g]['raw_tune6k_pinpoint']['NLL'] for _, g in p1_specs]
    dz_f = [tune_summary[g]['vertex_audit']['FrontPinpoint_30mm']['median_abs_dz_mm'] for _, g in p1_specs]
    dz_i = [tune_summary[g]['vertex_audit']['IntermPinpoint_150mm']['median_abs_dz_mm'] for _, g in p1_specs]
    ax1.bar(x1 - 0.16, dz_f, width=0.30, color='#4C72B0', label='Front 30mm PV $|\\Delta z|$ [mm]')
    ax1.bar(x1 + 0.16, dz_i, width=0.30, color='#DD8452', label='Interm 150mm PV $|\\Delta z|$ [mm]')
    ax1_r = ax1.twinx()
    ax1_r.plot(x1, nll1, 'ko-', lw=1.8, ms=5.5, label='Tune6k Pinpoint NLL')
    ax1.set_xticks(x1)
    ax1.set_xticklabels([lbl for lbl, _ in p1_specs], fontsize=8.0)
    ax1.set_ylabel('PV Median $|\\Delta z|$ [mm]')
    ax1_r.set_ylabel('Tune6k Pinpoint NLL')
    ax1.set_title('324-07/08: Fixed-Cluster Coord Ablation & PV Depth')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_r.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=7.2)
    ax1.grid(True, axis='y', alpha=0.25)

    # Panel 2: Generator-OOD & Genuine W-184 EPPS21 NC Charm Stress Test (324-04)
    ood_models = [
        ('s325-b1pp-clean', 'B1++', '#8C8C8C'),
        ('s325-pf-20um', 'PF-20um', '#4C72B0'),
        ('s325-pf-hybrid-v2', 'Hyb-v2', '#C44E52'),
        ('s325-pf-hybrid-v2-reg', 'Hyb-Reg', '#8172B3'),
        ('s325-pf-hybrid-v2-nomad', 'Hyb+NOM', '#55A868'),
    ]
    cats = [('Run85-88\nBA4', 'gen_ood_run10085_10088', 'BA4'), ('P6-NUX\nBA4', 'gen_ood_pythia6_nux', 'BA4'),
            ('W184 NC\nPred-NC', 'gen_ood_w184_epps21_nc_incl', 'frac_pred_nc'),
            ('W184 NC-c\nPred-tau', 'gen_ood_w184_epps21_nc_charm', 'frac_pred_tau')]
    cats = [c for c in cats if c[1] in ood_results['s325-pf-20um']]
    x2 = np.arange(len(cats))
    w2 = 0.15
    for gi, (gk, glbl, col) in enumerate(ood_models):
        vals = [100.0 * ood_results[gk][ov][mk] for _, ov, mk in cats]
        ax2.bar(x2 + (gi - 2) * w2, vals, width=w2, color=col, label=glbl)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([c[0] for c in cats], fontsize=8.2)
    ax2.set_ylabel('Rate / Balanced Accuracy [%]')
    ax2.set_title('324-04: Pure-W OOD & EPPS21 W-184 NC Charm')
    ax2.legend(loc='upper right', fontsize=7.4)
    ax2.grid(True, axis='y', alpha=0.25)

    fig.tight_layout()
    fig1_path = STAGE325_ROOT / 'stage325_remediation_summary.png'
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)

    # Figure 2: Detector-Response Degradation Stress Tests (324-10 & 324-06)
    fig2, (bx0, bx1) = plt.subplots(1, 2, figsize=(13.5, 4.6), dpi=300)
    st_keys = [s[0] for s in stress_specs]
    st_lbls = ['Nominal', 'PixDead\n2%', 'PixDead\n5%', 'PixDark\n1e-5', 'TrkD2%\n+N1', 'TrkD5%\n+N2', 'SiPM\nCap95%']
    xs = np.arange(len(st_keys))
    for gk, glbl, col, mk in [
        ('s325-b1pp-clean', 'B1++', '#8C8C8C', 's-'),
        ('s325-pf-20um', 'PF-20um', '#4C72B0', 'o-'),
        ('s325-pf-hybrid-v2', 'Hybrid-v2', '#C44E52', '^-'),
        ('s325-pf-hybrid-v2-nomad', 'Hybrid-v2+NOMAD', '#55A868', 'D-'),
    ]:
        bx0.plot(xs, [100.0 * stress_results[gk][k]['BA4'] for k in st_keys], mk, color=col, lw=1.8, ms=5.5, label=glbl)
        bx1.plot(xs[1:], [stress_results[gk][k]['mean_TV_vs_nominal'] for k in st_keys[1:]], mk, color=col, lw=1.8, ms=5.5, label=glbl)
    bx0.set_xticks(xs)
    bx0.set_xticklabels(st_lbls, fontsize=8.0)
    bx0.set_ylabel('Tune Shard-080 BA4 [%]')
    bx0.set_title('324-10/06: Detector Degradation Robustness (BA4)')
    bx0.legend(loc='lower left', fontsize=8.0)
    bx0.grid(True, alpha=0.25)
    bx1.set_xticks(xs[1:])
    bx1.set_xticklabels(st_lbls[1:], fontsize=8.0)
    bx1.set_ylabel('Mean Total Variation vs Nominal')
    bx1.set_title('Prediction Sensitivity Under Detector Degradation')
    bx1.legend(loc='upper left', fontsize=8.0)
    bx1.grid(True, alpha=0.25)
    fig2.tight_layout()
    fig2_path = STAGE325_ROOT / 'stage325_detector_stress_tests.png'
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)

    freeze_doc = {
        'protocol': 'Stage 325 Remediation of Stage 324 Clean v3 Audit (ZERO Sealed Shard Access Verified)',
        'UTC_frozen': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': float(time.perf_counter() - t0),
        'PinpointAnalysis_git_commit': analysis_sha,
        'Pinpoint_G4_git_commit': g4_sha,
        'Cal_A_Cut_B_Confirm_touched': False,
        'sealed_shards_accessed_during_freeze': len(sealed_accesses),
        'sealed_shards_excluded_from_freeze_hashing': sealed_names,
        'remediated_stage324_items': [
            '324-01', '324-02', '324-03', '324-04', '324-05', '324-06', '324-07',
            '324-08', '324-09', '324-10', '324-11', '324-12', '324-13', '324-14', '324-15',
        ],
        'artifact_fingerprints': file_hashes,
        'open_shard_observations_sha256': open_shard_hashes,
        'ood_shard_observations_sha256': ood_fingerprints,
        'frozen_non_nn_baselines': {
            k: {mk: v[mk] for mk in ('model_path', 'model_sha256', 'tune8k_pred_path', 'tune8k_pred_sha256', 'tune6k_screen_excl')}
            for k, v in (bl_doc['baselines'].items() if bl_doc else [])
        },
        'checkpoints': checkpoints,
        'tune_validation_summary': tune_summary,
        'tune6k_paired_bootstrap': paired_boot,
        'stage323_6t_own_observable_domain_re_eval': eval_6t_own_dom,
        'pv_pull_calibration_324_13': pv_pull_audit,
        'generator_ood_validation': ood_results,
        'detector_degradation_stress_324_10': stress_results,
        'observation_only_fp32_saved_parity_324_09': obs_only_closure,
        'figure_paths': [str(fig1_path), str(fig2_path)],
    }
    write_json(STAGE325_ROOT / 'v9_stage325_freeze_manifest.json', freeze_doc)
    write_json(ROADMAP_FREEZE_V9, freeze_doc)
    return freeze_doc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'command',
        choices=('prepare-v3-upgrades', 'w184-ood-worker', 'verify-gpu-gates', 'frozen-baselines', 'build-configs', 'submit-group', 'freeze-and-validate'),
    )
    parser.add_argument('--clean-root', type=Path, default=DEFAULT_CLEAN_V3_ROOT)
    parser.add_argument('--variant', type=str, default=None)
    parser.add_argument('--cpus', type=int, default=8)
    parser.add_argument('--group', type=str, default=None)
    parser.add_argument('--extra-groups', nargs='*', default=None)
    parser.add_argument('--backend', type=str, default='condor', choices=('condor', 'kubeflow'))
    parser.add_argument('--parallel', type=int, default=3)
    parser.add_argument('--kubeflow-flavor', type=str, default='h100-94gb')
    parser.add_argument('--min-gpu-mem', type=int, default=36000)
    parser.add_argument('--no-submit-w184', action='store_true')
    args = parser.parse_args()

    if args.command == 'prepare-v3-upgrades':
        out = prepare_stage325_upgrades(args.clean_root, submit_w184_condor=not args.no_submit_w184)
    elif args.command == 'w184-ood-worker':
        if not args.variant:
            parser.error('--variant required')
        out = run_w184_ood_worker(args.clean_root, args.variant, cpus=args.cpus)
    elif args.command == 'verify-gpu-gates':
        out = run_verify_gpu_gates(args.clean_root)
    elif args.command == 'frozen-baselines':
        out = run_frozen_baselines_stage325(args.clean_root)
    elif args.command == 'build-configs':
        grps = build_stage325_configs(args.clean_root)
        out = {k: {'prefix': v['prefix'], 'configs': [str(p) for p in v['configs']]} for k, v in grps.items()}
    elif args.command == 'submit-group':
        if not args.group:
            parser.error('--group required')
        out = submit_stage325_group(
            args.clean_root, args.group, args.backend,
            parallel=args.parallel, kubeflow_flavor=args.kubeflow_flavor,
            extra_groups=args.extra_groups, min_gpu_mem=args.min_gpu_mem,
        )
    else:
        out = freeze_and_validate_stage325(args.clean_root)
    print(json.dumps({k: v for k, v in out.items() if k not in ('open_shard_observations_sha256', 'checkpoints')}, indent=2, default=str))


if __name__ == '__main__':
    main()
