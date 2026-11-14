"""Observation-only clusters/segments/vertices; truth matching is a separate pass."""
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from analysis.flavour.datasets import NativeShard
from analysis.flavour.s2.train import atomic_output


TRACKER_Z_MM = np.array([47.4, 1237.4, 2427.4])


def clusters(pixel, geometry):
    xyz = np.asarray(pixel['xyz_mm']).reshape(-1, 3)
    ids = np.asarray(pixel['channel_id'], dtype='int64')
    nx, ny = len(geometry['pixel_Xpos']), len(geometry['pixel_Ypos'])
    planes = ids // (nx*ny)
    result = []
    for plane in np.unique(planes):
        indices = np.flatnonzero(planes == plane)
        xy = xyz[indices, :2]
        pairs = cKDTree(xy).query_pairs(.06, output_type='ndarray')
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(xy), len(xy)))
        count, groups = connected_components(graph, directed=False)
        for group in range(count):
            take = indices[groups == group]
            values = xyz[take]
            center = values.mean(0)
            width = np.sqrt(np.mean(np.sum((values[:, :2]-center[:2])**2, axis=1)))
            result.append({'plane': int(plane), 'xyz': center, 'width': float(width), 'n': len(take), 'ids': ids[take]})
    return result


def pixel_segments(items):
    eligible = [i for i, c in enumerate(items) if c['width'] <= .5 and c['n'] <= 64]
    by_plane = {}
    for i in eligible:
        by_plane.setdefault(items[i]['plane'], []).append(i)
    planes = sorted(by_plane)
    links = {}
    previous = {}
    for a, b in zip(planes[:-1], planes[1:]):
        left, right = by_plane[a], by_plane[b]
        u = np.stack([items[i]['xyz'] for i in left])
        v = np.stack([items[i]['xyz'] for i in right])
        distance, nearest = cKDTree(v[:, :2]).query(u[:, :2])
        _, reverse = cKDTree(u[:, :2]).query(v[:, :2])
        dz = float(v[0, 2]-u[0, 2])
        for i, j in enumerate(nearest):
            if reverse[j] == i and distance[i] <= max(.3, .01*dz):
                links[left[i]] = right[j]
                previous[right[j]] = left[i]
    result = []
    for first in eligible:
        if first in previous:
            continue
        chain = [first]
        while chain[-1] in links:
            chain.append(links[chain[-1]])
        if len(chain) < 3:
            continue
        points = np.stack([items[i]['xyz'] for i in chain])
        z0 = float(points[:, 2].mean())
        design = np.column_stack([np.ones(len(points)), points[:, 2]-z0])
        fit = np.linalg.lstsq(design, points[:, :2], rcond=None)[0]
        residual = float(np.sqrt(np.mean(np.sum((points[:, :2]-design@fit)**2, axis=1))))
        result.append({'clusters': chain, 'z0': z0, 'xy': fit[0], 'slope': fit[1],
                       'rms': residual, 'n': len(chain), 'first_z': float(points[:, 2].min())})
    return result


def vertex_candidate(tracks, z_bounds=(-3000., -1600.)):
    good = [t for t in tracks if t['rms'] < .3]
    best = None
    count = 0
    for i, left in enumerate(good):
        for right in good[i+1:]:
            ds = left['slope']-right['slope']
            opening2 = float(ds@ds)
            if opening2 < 1e-8:
                continue
            origin_l = left['xy']-left['slope']*left['z0']
            origin_r = right['xy']-right['slope']*right['z0']
            z = -float((origin_l-origin_r)@ds)/opening2
            if not z_bounds[0] <= z <= min(z_bounds[1], left['first_z']+20, right['first_z']+20):
                continue
            a, b = origin_l+left['slope']*z, origin_r+right['slope']*z
            dca = float(np.linalg.norm(a-b))
            if dca > 1.:
                continue
            sigma_z = np.sqrt(.05**2+left['rms']**2+right['rms']**2)/np.sqrt(opening2)
            quality = dca+sigma_z/100.
            count += 1
            if best is None or quality < best['quality']:
                best = {'xyz': np.r_[(a+b)/2, z], 'dca': dca, 'sigma_z': float(sigma_z), 'quality': float(quality)}
    return best, count


def tracker_associations(tracker):
    station = np.asarray(tracker['station'])
    xy = np.asarray(tracker['xy_mm']).reshape(-1, 2)
    rows = [np.flatnonzero(station == s) for s in range(3)]
    if not all(len(x) for x in rows):
        return []
    # B field is along x, so y bends. Three stations determine a bending
    # quadratic with no independent fit dof; do not call this momentum fitting.
    proposals = []
    last_tree = cKDTree(xy[rows[2]]/[2., 20.])
    for i in rows[0]:
        predicted = 2*xy[rows[1]]-xy[i]
        distances, indices = last_tree.query(predicted/[2., 20.])
        for j, distance, index in zip(rows[1], distances, indices):
            k = rows[2][index]
            dx = xy[k, 0]-predicted[np.flatnonzero(rows[1] == j)[0], 0]
            if abs(dx) < 3.5 and abs(xy[k, 1]-(2*xy[j, 1]-xy[i, 1])) < 20.:
                proposals.append((float(distance), [int(i), int(j), int(k)]))
    used = set()
    accepted = []
    for distance, ids in sorted(proposals):
        if not used.intersection(ids):
            used.update(ids)
            accepted.append({'indices': ids, 'quality': distance, 'slope': (xy[ids[-1]]-xy[ids[0]])/(TRACKER_Z_MM[-1]-TRACKER_Z_MM[0])})
    return accepted


def raw_features(observation):
    """Fixed physical moments/profiles. No normalization is fitted here."""
    features = []
    for name in ['pixel', 'bar', 'tracker_hypothesis']:
        data = observation[name]
        if name == 'pixel':
            xyz = np.asarray(data['xyz_mm']).reshape(-1, 3)
            weight = np.ones(len(xyz))
        elif name == 'bar':
            b = np.asarray(data['bounds_mm']).reshape(-1, 5)
            xyz = np.column_stack([(b[:, 0]+b[:, 1])/2, (b[:, 2]+b[:, 3])/2, b[:, 4]])
            weight = np.asarray(data['edep_proxy_MeV'])
        else:
            xyz = np.column_stack([np.asarray(data['xy_mm']).reshape(-1, 2), TRACKER_Z_MM[np.asarray(data['station'], dtype=int)]])
            weight = np.ones(len(xyz))
        if len(xyz):
            w = weight/max(float(weight.sum()), 1e-30)
            mean = w@xyz
            rms = np.sqrt(w@((xyz-mean)**2))
            extent = np.ptp(xyz, axis=0)
            profile = np.histogram(xyz[:, 2], bins=np.linspace(-3100, 2500, 17), weights=weight)[0]/max(float(weight.sum()), 1e-30)
            features.extend([1., np.log1p(len(xyz)), np.log1p(weight.sum()), *(mean/[210, 210, 3000]), *(rms/[210, 210, 3000]), *(extent/[420, 420, 6000]), *profile])
        else:
            features.extend([0.]*28)
    return np.asarray(features, dtype='float32')


def object_features(items, tracks, tracker, vertex, vertex_count):
    result = [np.log1p(len(items)), np.log1p(len(tracks)), np.log1p(len(tracker)), np.log1p(vertex_count)]
    for values in [[c['width'] for c in items], [c['n'] for c in items], [t['rms'] for t in tracks], [t['n'] for t in tracks],
                   [float(t['slope'][0]) for t in tracks], [float(t['slope'][1]) for t in tracks],
                   [float(t['quality']) for t in tracker]]:
        result.extend(np.quantile(values, [.1, .5, .9]).tolist() if values else [0., 0., 0.])
    if vertex:
        result.extend([1., *(vertex['xyz']/[210, 210, 3000]), np.log1p(vertex['dca']), np.log1p(vertex['sigma_z'])])
    else:
        result.extend([0.]*6)
    return np.asarray(result, dtype='float32')


def reconstruct(observation, geometry):
    items = clusters(observation['pixel'], geometry)
    tracks = pixel_segments(items)
    vertex, n_vertex = vertex_candidate(tracks)
    tracker = tracker_associations(observation['tracker_hypothesis'])
    return {'clusters': items, 'tracks': tracks, 'tracker_tracks': tracker, 'vertex': vertex,
            'vertex_count': n_vertex, 'raw_features': raw_features(observation),
            'object_features': object_features(items, tracks, tracker, vertex, n_vertex)}


def match_truth(reco, truth, observation, geometry):
    """Runs AFTER reconstruction. Cannot change candidates or input features."""
    p = np.stack(truth['pixel_truth_rows']).astype(float)
    nx, ny = len(geometry['pixel_Xpos']), len(geometry['pixel_Ypos'])
    labels = {}
    truth_planes = {}
    if p.shape[1]:
        ids = (p[0].astype('int64')*ny+p[1].astype('int64'))*nx+p[2].astype('int64')
        for channel, tid, plane in zip(ids, p[3].astype(int), p[0].astype(int)):
            labels.setdefault(int(channel), int(tid))
            truth_planes.setdefault(int(tid), set()).add(int(plane))
    purity, matched = [], set()
    for cluster in reco['clusters']:
        tids = np.array([labels.get(int(c), -1) for c in cluster['ids']])
        _, count = np.unique(tids[tids >= 0], return_counts=True)
        purity.append(float(count.max()/len(tids)) if len(count) else 0.)
    n_match = 0
    for track in reco['tracks']:
        ids = np.concatenate([reco['clusters'][i]['ids'] for i in track['clusters']])
        tids = np.array([labels.get(int(c), -1) for c in ids])
        values, counts = np.unique(tids[tids >= 0], return_counts=True)
        if len(counts) and counts.max()/len(ids) >= .75:
            tid = int(values[counts.argmax()])
            n_match += 1
            matched.add(tid)
    reconstructable = {tid for tid, planes in truth_planes.items() if len(planes) >= 3}
    # Reproduce only the accepted digitizer's index/keep map for evaluation.
    raw = np.stack(truth['tracker_truth_rows']).astype(float)
    cell_labels = []
    if raw.shape[1]:
        cells = np.column_stack([raw[0].astype(int), np.floor(raw[2:4].T/.1).astype(int)])
        unique, inverse = np.unique(cells, axis=0, return_inverse=True)
        seed = int(truth['family_hi']) ^ int(truth['family_lo']) ^ 20261111
        keep = np.random.default_rng(seed).random(len(unique)) < .99
        cell_labels = [set(raw[1, inverse == i].astype(int).tolist()) for i in np.flatnonzero(keep)]
        if len(cell_labels) != len(observation['tracker_hypothesis']['station']):
            raise ValueError('tracker truth map differs from accepted observation digitizer')
    tracker_eligible = {}
    stations = np.asarray(observation['tracker_hypothesis']['station'])
    for s, ids in zip(stations, cell_labels):
        for tid in ids:
            tracker_eligible.setdefault(tid, set()).add(int(s))
    tracker_matched = set()
    tracker_good = 0
    for track in reco['tracker_tracks']:
        common = set.intersection(*[cell_labels[i] for i in track['indices']])
        if common:
            tracker_good += 1
            tracker_matched.update(common)
    eligible = {tid for tid, stations in tracker_eligible.items() if len(stations) == 3}
    vertex = reco['vertex']
    primary_residual = vertex['xyz']-np.asarray(truth['vertex_mm']) if vertex else np.full(3, np.nan)
    tau_vertex = truth['tau_decay_vertex_mm']
    tau_residual = vertex['xyz']-np.asarray(tau_vertex) if vertex and tau_vertex is not None else np.full(3, np.nan)
    return {'cluster_count': len(purity), 'cluster_purity_sum': float(sum(purity)),
            'pixel_tracks': len(reco['tracks']), 'pixel_tracks_matched': n_match,
            'pixel_truth_reconstructable': len(reconstructable), 'pixel_truth_recovered': len(reconstructable & matched),
            'tracker_tracks': len(reco['tracker_tracks']), 'tracker_tracks_matched': tracker_good,
            'tracker_truth_reconstructable': len(eligible), 'tracker_truth_recovered': len(eligible & tracker_matched),
            'primary_vertex_reconstructable': len(reconstructable & set(truth['interaction_track_ids'])) >= 2,
            'tau_vertex_reconstructable': len(reconstructable & set(truth['tau_direct_track_ids'])) >= 2,
            'vertex_present': vertex is not None,
            'vertex_good': bool(vertex and vertex['sigma_z'] < 20 and vertex['dca'] < .3),
            'vertex_primary_residual_mm': primary_residual.tolist(), 'vertex_tau_residual_mm': tau_residual.tolist(),
            'vertex_sigma_z_mm': vertex['sigma_z'] if vertex else None}


def create_shard(root, task, limit=None):
    started = time.monotonic()
    if not (root/'truth_summary.json').exists():
        raise RuntimeError('wait for completed truth assembly before object extraction')
    paths = sorted(p for p in (root/'shards').iterdir() if (p/'observations.parquet').exists())
    path = paths[task]
    metadata = pd.read_parquet(path/'provenance.parquet')
    catalog = pd.read_parquet(root/'catalog.parquet')
    keep_keys = set(map(tuple, catalog[['run', 'chunk', 'row']].to_numpy()))
    truth = pd.read_parquet(root/'truth.parquet').set_index(['run', 'chunk', 'row'])
    geometry = json.loads(Path(json.loads((root/'plan.json').read_text())['geometry']).read_text())
    native = NativeShard(path)
    output = []
    for i, meta in enumerate(metadata.to_dict('records')):
        key = tuple(meta[k] for k in ['run', 'chunk', 'row'])
        if key not in keep_keys:
            continue
        observation = native[i]['observations']
        reco = reconstruct(observation, geometry)
        labels = truth.loc[key].to_dict()
        result = {k: int(meta[k]) for k in ['run', 'chunk', 'row']}
        result.update(raw_features=reco['raw_features'].tolist(), object_features=reco['object_features'].tolist(),
                      reconstructed_vertex_mm=reco['vertex']['xyz'].tolist() if reco['vertex'] else None)
        result.update(match_truth(reco, labels, observation, geometry))
        output.append(result)
        if limit and len(output) >= limit:
            break
    dest = root/'object_parts'/f'{task:03d}.parquet' if not limit else root/f'objects_smoke_{task:03d}.parquet'
    atomic_output(dest, lambda stream: pq.write_table(pa.Table.from_pylist(output), stream, compression='gzip'))
    print(json.dumps({'task': task, 'events': len(output), 'seconds': time.monotonic()-started, 'output': str(dest)}), flush=True)


def finalize_objects(root):
    parts = sorted((root/'object_parts').glob('*.parquet'))
    if len(parts) != len(list((root/'shards').iterdir())):
        raise ValueError('object shards incomplete')
    table = pa.concat_tables([pq.read_table(p) for p in parts], promote_options='permissive')
    frame = table.to_pandas()
    metadata = pd.read_parquet(root/'catalog.parquet')
    key = ['run', 'chunk', 'row']
    frame = metadata[key+['label', 'split']].merge(frame, on=key, validate='one_to_one')
    if len(frame) != len(metadata):
        raise ValueError('object key coverage incomplete')
    atomic_output(root/'objects.parquet', lambda stream: frame.to_parquet(stream, index=False, compression='gzip'))
    output = {}
    for (split, label), group in frame.groupby(['split', 'label']):
        row = {'events': len(group), 'cluster_count': int(group.cluster_count.sum()),
               'cluster_mean_purity_first_contributor': float(group.cluster_purity_sum.sum()/max(group.cluster_count.sum(), 1))}
        for kind in ['pixel', 'tracker']:
            candidates, matched = int(group[kind+'_tracks'].sum()), int(group[kind+'_tracks_matched'].sum())
            eligible, recovered = int(group[kind+'_truth_reconstructable'].sum()), int(group[kind+'_truth_recovered'].sum())
            row[kind] = {'candidates': candidates, 'matched': matched, 'fake': candidates-matched,
                         'purity': matched/max(candidates, 1), 'eligible': eligible, 'recovered': recovered, 'efficiency': recovered/max(eligible, 1)}
        row['vertex'] = {'events': len(group), 'primary_reconstructable': int(group.primary_vertex_reconstructable.sum()),
                         'tau_reconstructable': int(group.tau_vertex_reconstructable.sum()),
                         'candidate_events': int(group.vertex_present.sum()), 'quality_events': int(group.vertex_good.sum())}
        residuals = np.stack(group.vertex_primary_residual_mm)
        good = group.vertex_good.to_numpy()
        for name, mask in [('all_candidates', group.vertex_present.to_numpy()), ('quality', good)]:
            values = residuals[mask]
            row['vertex'][name] = {'events': int(mask.sum()), 'coverage': float(mask.mean()),
                                   'absolute_residual_quantiles_mm': np.quantile(np.abs(values), [.5, .68, .95], axis=0).tolist() if len(values) else None,
                                   'within_1mm_xy_10mm_z': int(np.sum((np.linalg.norm(values[:, :2], axis=1) < 1)&(np.abs(values[:, 2]) < 10))) if len(values) else 0}
        output[f'{split}_{label}'] = row
    report = {'groups': output, 'events': len(frame), 'input_only_observations': True,
              'pixel_truth_limitation': 'first contributor; no energy-mixture truth',
              'tracker_association': 'three-station position compatibility with bending-y tolerance; no measured q/p',
              'classification_started': False}
    (root/'objects_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
