"""Python ctypes loader & fallback wrapper for _fast_kernels.cpp native C++17 accelerators."""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np


_LIB: ctypes.CDLL | None = None
_LIB_TRIED = False

MICRO_VTX_FEAT_DIM = 16
MICRO_VTX_FEATURE_NAMES = (
    'log1p_n_tracklets',
    'log1p_n_3hit_tracklets',
    'avr_pv_valid',
    'avr_pv_chi2_ndf_asinh',
    'max_ip_d0_50um_asinh',
    'second_ip_d0_50um_asinh',
    'max_ip_sig_asinh',
    'second_ip_sig_asinh',
    'log1p_n_displaced_gt2p5',
    'frac_displaced_gt2p5',
    'max_thrust_signed_d0_asinh',
    'sv_2v_valid',
    'sv_delta_chi2_log1p',
    'sv_lxy_100um_asinh',
    'sv_lz_2mm_asinh',
    'max_kink_mrad_asinh',
)


def _compile_or_load_lib() -> ctypes.CDLL | None:
    src = Path(__file__).with_name('_fast_kernels.cpp')
    if not src.is_file():
        return None
    digest = hashlib.sha256(src.read_bytes()).hexdigest()[:12]
    so_name = f'libpinpoint_fast_{digest}.so'

    # 1. Check pre-built shared EOS cache or explicit env var first (instant load on Kubeflow/Condor)
    candidates = []
    if env_so := os.environ.get('PINPOINT_FAST_SO'):
        candidates.append(Path(env_so))
    eos_cache_dir = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v4_audit_2026-11-18/native_bin')
    candidates.append(eos_cache_dir / so_name)
    candidates.append(Path(tempfile.gettempdir()) / so_name)

    for cand in candidates:
        if cand.is_file():
            try:
                lib = ctypes.CDLL(str(cand))
                _bind_signatures(lib)
                return lib
            except OSError:
                continue

    # 2. Compile locally with g++ -O3 if available
    cxx = shutil.which('g++') or shutil.which('c++')
    if not cxx:
        return None
    tmp_dir = Path(tempfile.gettempdir())
    out_so = tmp_dir / so_name
    tmp_so = tmp_dir / f'{so_name}.{os.getpid()}.tmp'
    cmd = [cxx, '-O3', '-std=c++17', '-fPIC', '-shared', str(src), '-o', str(tmp_so), '-lm']
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return None
    try:
        os.replace(tmp_so, out_so)
    except OSError:
        out_so = tmp_so
    try:
        if eos_cache_dir.parent.exists():
            eos_cache_dir.mkdir(parents=True, exist_ok=True)
            eos_target = eos_cache_dir / so_name
            if not eos_target.exists():
                shutil.copyfile(out_so, eos_target)
    except OSError:
        pass
    try:
        lib = ctypes.CDLL(str(out_so))
        _bind_signatures(lib)
        return lib
    except OSError:
        return None


def _bind_signatures(lib: ctypes.CDLL) -> None:
    u64_p = np.ctypeslib.ndpointer(dtype=np.uint64, flags='C_CONTIGUOUS')
    i64_p = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
    i32_p = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    i16_p = np.ctypeslib.ndpointer(dtype=np.int16, flags='C_CONTIGUOUS')
    f64_p = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    f32_p = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')

    lib.fast_cluster_pixel_hits.argtypes = [
        u64_p, f64_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_double,
        i32_p, i32_p, i16_p, u64_p, i32_p,
    ]
    lib.fast_cluster_pixel_hits.restype = ctypes.c_int32

    lib.fast_plane_neighborhoods.argtypes = [
        i64_p, f64_p, f64_p, f64_p, ctypes.c_int64,
        f64_p, f64_p, f64_p, f64_p, f64_p, f64_p, f64_p,
    ]
    lib.fast_plane_neighborhoods.restype = None

    lib.fast_rasterize_patches.argtypes = [
        i32_p, i32_p, f64_p, f64_p, f64_p, i64_p, i64_p,
        ctypes.c_int64, ctypes.c_int32, ctypes.c_double, ctypes.c_double, f32_p,
    ]
    lib.fast_rasterize_patches.restype = None

    lib.fast_reconstruct_micron_vertex.argtypes = [
        i64_p, f64_p, f64_p, f64_p, f64_p, f64_p,
        ctypes.c_int64, ctypes.c_int32, ctypes.c_double, ctypes.c_double,
        f64_p, f64_p, f64_p, i32_p, f32_p,
    ]
    lib.fast_reconstruct_micron_vertex.restype = None


def get_fast_lib() -> ctypes.CDLL | None:
    global _LIB, _LIB_TRIED
    if not _LIB_TRIED:
        _LIB = _compile_or_load_lib()
        _LIB_TRIED = True
    return _LIB


def cluster_pixel_hits_fast(
    channel_id: np.ndarray,
    xyz_mm: np.ndarray,
    nx: int,
    ny: int,
    radius_mm: float = 0.06,
) -> dict | None:
    lib = get_fast_lib()
    if lib is None:
        return None
    cid = np.ascontiguousarray(channel_id, dtype=np.uint64)
    xyz = np.ascontiguousarray(xyz_mm, dtype=np.float64).reshape(-1, 3)
    n_hits = len(cid)
    if n_hits == 0:
        return {
            'n_clusters': 0,
            'hit_order': np.empty(0, dtype=np.int32),
            'cluster_offsets': np.zeros(1, dtype=np.int32),
            'cluster_plane': np.empty(0, dtype=np.int16),
            'cluster_first_channel_id': np.empty(0, dtype=np.uint64),
            'hit_to_cluster': np.empty(0, dtype=np.int32),
        }
    hit_order = np.empty(n_hits, dtype=np.int32)
    offsets = np.empty(n_hits + 1, dtype=np.int32)
    cl_plane = np.empty(n_hits, dtype=np.int16)
    cl_fcid = np.empty(n_hits, dtype=np.uint64)
    hit_to_cl = np.empty(n_hits, dtype=np.int32)
    n_cl = int(lib.fast_cluster_pixel_hits(
        cid, xyz, n_hits, int(nx), int(ny), float(radius_mm),
        hit_order, offsets, cl_plane, cl_fcid, hit_to_cl,
    ))
    return {
        'n_clusters': n_cl,
        'hit_order': hit_order,
        'cluster_offsets': offsets[: n_cl + 1].copy(),
        'cluster_plane': cl_plane[:n_cl].copy(),
        'cluster_first_channel_id': cl_fcid[:n_cl].copy(),
        'hit_to_cluster': hit_to_cl,
    }


def plane_neighborhoods_fast(
    cl_plane: np.ndarray,
    x_mean: np.ndarray,
    y_mean: np.ndarray,
    n_hits: np.ndarray,
) -> tuple[np.ndarray, ...] | None:
    lib = get_fast_lib()
    if lib is None:
        return None
    n_cl = len(cl_plane)
    cnt_r015 = np.zeros(n_cl, dtype=np.float64)
    hits_r015 = np.zeros(n_cl, dtype=np.float64)
    cnt_r040 = np.zeros(n_cl, dtype=np.float64)
    hits_r040 = np.zeros(n_cl, dtype=np.float64)
    cnt_r1 = np.zeros(n_cl, dtype=np.float64)
    cnt_r5 = np.zeros(n_cl, dtype=np.float64)
    hits_r5 = np.zeros(n_cl, dtype=np.float64)
    if n_cl <= 1:
        return cnt_r015, hits_r015, cnt_r040, hits_r040, cnt_r1, cnt_r5, hits_r5
    lib.fast_plane_neighborhoods(
        np.ascontiguousarray(cl_plane, dtype=np.int64),
        np.ascontiguousarray(x_mean, dtype=np.float64),
        np.ascontiguousarray(y_mean, dtype=np.float64),
        np.ascontiguousarray(n_hits, dtype=np.float64),
        n_cl,
        cnt_r015, hits_r015, cnt_r040, hits_r040, cnt_r1, cnt_r5, hits_r5,
    )
    return cnt_r015, hits_r015, cnt_r040, hits_r040, cnt_r1, cnt_r5, hits_r5


def rasterize_patches_fast(
    order: np.ndarray,
    offsets: np.ndarray,
    pxyz: np.ndarray,
    x_mean: np.ndarray,
    y_mean: np.ndarray,
    sel_local: np.ndarray,
    keep_idx: np.ndarray,
    patch_size: int,
    pitch_x: float,
    pitch_y: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    lib = get_fast_lib()
    if lib is None:
        return None
    n_p = len(sel_local)
    if n_p == 0:
        return np.empty((0, 2, patch_size, patch_size), dtype=np.float16), np.empty(0, dtype=np.int16)
    patches = np.empty((n_p, 2, patch_size, patch_size), dtype=np.float32)
    lib.fast_rasterize_patches(
        np.ascontiguousarray(order, dtype=np.int32),
        np.ascontiguousarray(offsets, dtype=np.int32),
        np.ascontiguousarray(pxyz, dtype=np.float64).reshape(-1, 3),
        np.ascontiguousarray(x_mean, dtype=np.float64),
        np.ascontiguousarray(y_mean, dtype=np.float64),
        np.ascontiguousarray(sel_local, dtype=np.int64),
        np.ascontiguousarray(keep_idx, dtype=np.int64),
        n_p, int(patch_size), float(pitch_x), float(pitch_y), patches,
    )
    return patches.astype(np.float16), np.asarray(sel_local, dtype=np.int16)


def reconstruct_micron_vertex(
    cl_plane: np.ndarray,
    x_mean: np.ndarray,
    y_mean: np.ndarray,
    z_cl: np.ndarray,
    n_hits: np.ndarray,
    lam2: np.ndarray,
    n_pix_planes: int,
    pitch_x: float,
    pitch_y: float,
    cal_start: tuple[float, float, float, bool] | np.ndarray,
    default_center: tuple[float, float, float] | np.ndarray,
) -> tuple[np.ndarray, bool, np.ndarray]:
    """Reconstruct explicit 3D primary/secondary vertex, d0/sigma_d0, and kink angles."""
    cs = np.ascontiguousarray(
        [float(cal_start[0]), float(cal_start[1]), float(cal_start[2]), 1.0 if bool(cal_start[3]) else 0.0],
        dtype=np.float64,
    )
    dc = np.ascontiguousarray(default_center, dtype=np.float64)
    pv_xyz = np.empty(3, dtype=np.float64)
    pv_valid = np.zeros(1, dtype=np.int32)
    feat = np.zeros(MICRO_VTX_FEAT_DIM, dtype=np.float32)

    lib = get_fast_lib()
    n_cl = len(cl_plane)
    if lib is None or n_cl < 2:
        if cs[3] > 0.5:
            pv_xyz[:] = (cs[1], cs[2], cs[0])
        else:
            pv_xyz[:] = dc[:3]
        return pv_xyz.astype(np.float32), False, feat

    lib.fast_reconstruct_micron_vertex(
        np.ascontiguousarray(cl_plane, dtype=np.int64),
        np.ascontiguousarray(x_mean, dtype=np.float64),
        np.ascontiguousarray(y_mean, dtype=np.float64),
        np.ascontiguousarray(z_cl, dtype=np.float64),
        np.ascontiguousarray(n_hits, dtype=np.float64),
        np.ascontiguousarray(lam2, dtype=np.float64),
        n_cl, int(n_pix_planes), float(pitch_x), float(pitch_y),
        cs, dc, pv_xyz, pv_valid, feat,
    )
    return pv_xyz.astype(np.float32), bool(pv_valid[0] > 0), feat
