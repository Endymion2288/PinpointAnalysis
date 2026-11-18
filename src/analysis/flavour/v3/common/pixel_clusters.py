"""Deterministic single-plane pixel connected-component clustering.

Shared by W3-02 (truth_sidecar) and Phase V2 (pixel_tokenizer) to guarantee
exact 1-to-1 cluster ordering and channel_id alignment without randomness.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from analysis.flavour.v3.common.fast_kernels import cluster_pixel_hits_fast


CLUSTER_RADIUS_MM = 0.06


def cluster_pixel_hits(
    channel_id: np.ndarray,
    xyz_mm: np.ndarray,
    nx: int,
    ny: int,
    radius_mm: float = CLUSTER_RADIUS_MM,
) -> dict:
    """Cluster pixel hits per silicon plane using radius r=0.06 mm connectivity.

    Clusters are ordered ascending by (plane, min(channel_id)) so ordering is
    strictly invariant to input permutation and deterministic across callers.
    """
    cid = np.asarray(channel_id, dtype=np.uint64)
    xyz = np.asarray(xyz_mm, dtype=np.float64).reshape(-1, 3)
    if len(xyz) and not np.isfinite(xyz).all():
        raise ValueError('Non-finite pixel xyz_mm coordinates encountered in cluster_pixel_hits')
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

    fast_out = cluster_pixel_hits_fast(cid, xyz, nx, ny, radius_mm=radius_mm)
    if fast_out is not None:
        return fast_out

    # Canonical sort by channel_id ensures permutation invariance.
    canon = np.argsort(cid, kind='mergesort')
    cid_s = cid[canon]
    xyz_s = xyz[canon]
    stride = np.uint64(nx) * np.uint64(ny)
    planes = (cid_s // stride).astype(np.int64)

    bounds = np.r_[0, np.flatnonzero(np.diff(planes)) + 1, n_hits]
    hit_to_cluster_sorted = np.empty(n_hits, dtype=np.int32)
    cluster_planes: list[np.ndarray] = []
    cluster_first_cids: list[np.ndarray] = []
    cluster_counts: list[np.ndarray] = []
    cluster_offset_id = 0

    for start, stop in zip(bounds[:-1], bounds[1:]):
        m = int(stop - start)
        plane_val = int(planes[start])
        local_cid = cid_s[start:stop]
        if m == 1:
            hit_to_cluster_sorted[start:stop] = cluster_offset_id
            cluster_planes.append(np.array([plane_val], dtype=np.int16))
            cluster_first_cids.append(local_cid[:1].copy())
            cluster_counts.append(np.array([1], dtype=np.int32))
            cluster_offset_id += 1
            continue

        pairs = cKDTree(xyz_s[start:stop, :2]).query_pairs(r=radius_mm, output_type='ndarray')
        if len(pairs) == 0:
            # Every hit is a singleton; already sorted by channel_id.
            hit_to_cluster_sorted[start:stop] = np.arange(cluster_offset_id, cluster_offset_id + m, dtype=np.int32)
            cluster_planes.append(np.full(m, plane_val, dtype=np.int16))
            cluster_first_cids.append(local_cid.copy())
            cluster_counts.append(np.ones(m, dtype=np.int32))
            cluster_offset_id += m
            continue

        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        data = np.ones(len(rows), dtype=np.int8)
        graph = csr_matrix((data, (rows, cols)), shape=(m, m))
        n_comp, labels = connected_components(graph, directed=False, return_labels=True)

        # Since local_cid is strictly ascending, first appearance of each label
        # in labels[0..m-1] IS the component with smallest min(channel_id)!
        first_pos = np.empty(n_comp, dtype=np.int32)
        seen = np.zeros(n_comp, dtype=bool)
        # Vectorized min index per label via reversed assignment
        first_pos[labels[::-1]] = np.arange(m - 1, -1, -1, dtype=np.int32)
        comp_order = np.argsort(first_pos, kind='mergesort')
        remap = np.empty(n_comp, dtype=np.int32)
        remap[comp_order] = np.arange(n_comp, dtype=np.int32)
        relabeled = remap[labels]

        hit_to_cluster_sorted[start:stop] = cluster_offset_id + relabeled
        cluster_planes.append(np.full(n_comp, plane_val, dtype=np.int16))
        cluster_first_cids.append(local_cid[first_pos[comp_order]])
        cluster_counts.append(np.bincount(relabeled, minlength=n_comp).astype(np.int32))
        cluster_offset_id += n_comp

    # Build hit_order grouping hits by cluster_id (preserving sorted channel_id within cluster)
    order_within = np.argsort(hit_to_cluster_sorted, kind='mergesort')
    hit_order = canon[order_within].astype(np.int32)
    counts_all = np.concatenate(cluster_counts)
    offsets = np.r_[0, np.cumsum(counts_all)].astype(np.int32)

    hit_to_cluster_orig = np.empty(n_hits, dtype=np.int32)
    hit_to_cluster_orig[canon] = hit_to_cluster_sorted

    return {
        'n_clusters': int(cluster_offset_id),
        'hit_order': hit_order,
        'cluster_offsets': offsets,
        'cluster_plane': np.concatenate(cluster_planes),
        'cluster_first_channel_id': np.concatenate(cluster_first_cids),
        'hit_to_cluster': hit_to_cluster_orig,
    }
