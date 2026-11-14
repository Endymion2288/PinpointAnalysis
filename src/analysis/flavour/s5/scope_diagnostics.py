"""Complete S5 target-overlap and tau geometry diagnostics on saved validation.

Truth projections describe geometric separation before scattering, showers and
pattern recognition. They never enter model inputs or determine data splits.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import uproot

from analysis.flavour.inventory import write_json
from analysis.flavour.s3.metrics import MetricCache, bootstrap_weights, interval_report
from analysis.flavour.s5.evaluate import binomial
from analysis.flavour.s5.truth import group_flat, particle_info, primary_decay


KEY = ['run', 'chunk', 'row']
EDGES = [np.array([0, 100, 300, 1000, 2500, 6000])*1000,
         np.array([-1000, -100, 0, 100, 1000]),
         np.array([-1000, -80, 0, 80, 1000]),
         np.array([-4000, -2600, -2200, -1800, 0])]


def target_support_weights(frame):
    """Equal class density per joint E/x/y/z/exact-target-PDG cell; w <= 1."""
    bins = np.column_stack([np.digitize(frame[column], edge)-1
                           for column, edge in zip(['initE', 'initX', 'initY', 'initZ'], EDGES)])
    in_range = np.all((bins >= 0) & (bins < np.array([len(e)-1 for e in EDGES])), axis=1)
    cells = pd.DataFrame(bins, columns=['e', 'x', 'y', 'z'])
    cells['target'] = frame.tgtPDG.to_numpy()
    index = pd.MultiIndex.from_frame(cells)
    cell, unique = pd.factorize(index, sort=True)
    labels = frame.label.to_numpy()
    counts = np.zeros((len(unique), 4), dtype=int)
    np.add.at(counts, (cell[in_range], labels[in_range]), 1)
    minimum = counts.min(axis=1)
    valid = minimum >= 5
    weight = np.zeros(len(frame))
    keep = in_range & valid[cell]
    weight[keep] = minimum[cell[keep]]/counts[cell[keep], labels[keep]]
    targets = []
    for target in sorted(frame.tgtPDG.unique()):
        selected = frame.tgtPDG.to_numpy() == target
        targets.append({'target_PDG': int(target),
            'class_events': [int(np.sum(selected & (labels == c))) for c in range(4)],
            'class_weight_sums': [float(weight[selected & (labels == c)].sum()) for c in range(4)],
            'retained_events': int(np.sum(selected & keep))})
    return weight, {'edges': dict(zip(['energy_MeV', 'x_mm', 'y_mm', 'z_mm'], [e.tolist() for e in EDGES])),
        'target_matching': 'exact tgtPDG isotope, jointly with fixed energy and vertex bins',
        'minimum_events_per_class_per_cell': 5, 'weight_cap': 1.,
        'supported_cells': int(valid.sum()), 'total_populated_cells': int((counts.sum(1) > 0).sum()),
        'retained_events': int(keep.sum()), 'excluded_events': int((~keep).sum()),
        'class_weight_sums': [float(weight[labels == c].sum()) for c in range(4)],
        'Neff': float(weight.sum()**2/np.dot(weight, weight)) if keep.any() else 0.,
        'targets': targets, 'scope': 'fixed overlap diagnostic; no flux, exposure or extrapolation weights'}


def projected_geometry(vertex, parent, momenta, geometry):
    """Project forward charged daughter rays to the first downstream pixel center.

    Distances/pitch use the max absolute axis displacement; >1 is a geometrical
    one-cell scale, not a fitted separation criterion or calibrated uncertainty.
    """
    z = np.sort(np.asarray(geometry['pixel_Zpos']))
    downstream = z[z > vertex[2]]
    result = {'first_pixel_z_mm': None, 'decay_to_first_pixel_mm': None,
              'between_pixel_planes': bool(z[0] <= vertex[2] < z[-1]),
              'pixel_interval_fraction': None, 'forward_charged_daughters': 0,
              'projected_charged_in_pixel': 0, 'parent_projected_in_pixel': False,
              'maximum_kink_rad': None, 'maximum_parent_separation_mm': None,
              'maximum_parent_separation_pitch': None, 'minimum_pair_separation_mm': None,
              'minimum_pair_separation_pitch': None}
    if result['between_pixel_planes']:
        i = np.searchsorted(z, vertex[2], side='right')
        result['pixel_interval_fraction'] = float((vertex[2]-z[i-1])/(z[i]-z[i-1]))
    p = np.asarray(momenta, dtype=float).reshape(-1, 3)
    forward = p[np.isfinite(p).all(1) & (p[:, 2] > 0)]
    result['forward_charged_daughters'] = len(forward)
    parent = np.asarray(parent, dtype=float)
    parent_valid = bool(np.isfinite(parent).all() and parent[2] > 0)
    if parent_valid and len(forward):
        cosine = (forward @ parent)/(np.linalg.norm(forward, axis=1)*np.linalg.norm(parent))
        result['maximum_kink_rad'] = float(np.max(np.arccos(np.clip(cosine, -1., 1.))))
    if not len(downstream):
        return result
    plane = downstream[0]
    dz = plane-vertex[2]
    result.update(first_pixel_z_mm=float(plane), decay_to_first_pixel_mm=float(dz))
    pitch = np.array([np.median(np.diff(geometry[k])) for k in ['pixel_Xpos', 'pixel_Ypos']])
    low = np.array([geometry[k][0] for k in ['pixel_Xpos', 'pixel_Ypos']])-pitch/2
    high = np.array([geometry[k][-1] for k in ['pixel_Xpos', 'pixel_Ypos']])+pitch/2
    xy = vertex[:2]+dz*forward[:, :2]/forward[:, 2:3]
    accepted = xy[np.all((xy >= low) & (xy <= high), axis=1)]
    result['projected_charged_in_pixel'] = len(accepted)
    if parent_valid:
        parent_xy = vertex[:2]+dz*parent[:2]/parent[2]
        result['parent_projected_in_pixel'] = bool(np.all((parent_xy >= low) & (parent_xy <= high)))
        if result['parent_projected_in_pixel'] and len(accepted):
            residual = accepted-parent_xy
            result['maximum_parent_separation_mm'] = float(np.max(np.linalg.norm(residual, axis=1)))
            result['maximum_parent_separation_pitch'] = float(np.max(np.abs(residual)/pitch))
    if len(accepted) >= 2:
        a, b = np.triu_indices(len(accepted), k=1)
        residual = accepted[a]-accepted[b]
        result['minimum_pair_separation_mm'] = float(np.min(np.linalg.norm(residual, axis=1)))
        result['minimum_pair_separation_pitch'] = float(np.min(np.max(np.abs(residual)/pitch, axis=1)))
    return result


def extract_geometry(arguments):
    input_root, geometry, frame = arguments
    run, chunk = int(frame.run.iloc[0]), int(frame.chunk.iloc[0])
    path = Path(input_root)/str(run)/f'{run}_{chunk:03d}.root'
    # Four outer readers share the four requested CPUs; avoid nested default
    # decompression/interpretation pools for each ROOT file.
    with uproot.open(path, decompression_executor=uproot.source.futures.TrivialExecutor(),
                     interpretation_executor=uproot.source.futures.TrivialExecutor()) as source:
        primaries = group_flat(source['primaries'], ['evtID', 'PDG', 'trackID', 'Px', 'Py', 'Pz'])
        daughters = group_flat(source['tau'], ['evtID', 'PID', 'PDG', 'TID', 'Px', 'Py', 'Pz', 'ProdX', 'ProdY', 'ProdZ'])
    rows = []
    for meta in frame.to_dict('records'):
        event = int(meta['evtID'])
        primary = primaries.get(event, {})
        tau_mask = np.abs(primary['PDG']) == 15
        ids = primary['trackID'][tau_mask]
        direct, vertex, valid = primary_decay(daughters.get(event, {}), ids)
        row = {k: meta[k] for k in KEY}
        row.update(geometry_projection_valid=bool(valid and tau_mask.sum() == 1))
        if row['geometry_projection_valid']:
            charged = np.array([np.isfinite(particle_info(int(p))[0]) and particle_info(int(p))[0] != 0
                                for p in direct['PDG']])
            momentum = np.column_stack([direct['P'+c] for c in 'xyz'])[charged]
            parent = np.column_stack([primary['P'+c] for c in 'xyz'])[tau_mask][0]
            row.update(projected_geometry(vertex, parent, momentum, geometry))
        rows.append(row)
    return rows


def geometry_summary(frame, selected, block_weights):
    masks = {'all_tau': np.ones(len(frame), bool), 'geometry_valid': frame.geometry_projection_valid.eq(True).to_numpy(),
             'between_pixel_planes': frame.between_pixel_planes.eq(True).to_numpy(),
             'forward_daughter_in_first_pixel': frame.projected_charged_in_pixel.ge(1).to_numpy(),
             'parent_daughter_above_one_pitch': frame.maximum_parent_separation_pitch.gt(1).to_numpy(),
             'pair_above_one_pitch': frame.minimum_pair_separation_pitch.gt(1).to_numpy(),
             'labelled_pixel_response': frame.tau_visible_pixel.eq(True).to_numpy()}
    result = []
    for branch in ['inclusive', 'electron', 'muon', 'hadronic', 'unknown']:
        branch_mask = np.ones(len(frame), bool) if branch == 'inclusive' else frame.tau_branch.eq(branch).to_numpy()
        for name, mask in masks.items():
            eligible = branch_mask & mask
            n, k = int(eligible.sum()), int(np.sum(eligible & selected))
            denominator = block_weights @ eligible.astype(float)
            rate = np.divide(block_weights @ (eligible & selected).astype(float), denominator,
                             out=np.full(len(block_weights), np.nan), where=denominator > 0)
            finite = rate[np.isfinite(rate)]
            result.append({'branch': branch, 'domain': name, **binomial(k, n),
                'block_CI95': np.quantile(finite, [.025, .975]).tolist() if len(finite) else [None, None],
                'labelled_pixel_response_events': int(np.sum(eligible & frame.tau_visible_pixel.eq(True).to_numpy())),
                'any_daughter_response_events': int(np.sum(eligible & frame.tau_visible_any.eq(True).to_numpy()))})
    quantiles = {}
    for column in ['decay_to_first_pixel_mm', 'pixel_interval_fraction', 'maximum_kink_rad',
                   'maximum_parent_separation_mm', 'maximum_parent_separation_pitch',
                   'minimum_pair_separation_mm', 'minimum_pair_separation_pitch']:
        values = frame[column].dropna().to_numpy()
        quantiles[column] = {'events': len(values), 'q16_50_68_95': np.quantile(values, [.16, .5, .68, .95]).tolist() if len(values) else None}
    return {'operating_points': result, 'quantiles': quantiles,
            'material_location_counts': frame.tau_decay_location.value_counts().to_dict()}


def plot(output, frame, seed_means):
    from analysis.flavour.s5.plots import plt, save
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    colors = {'electron': 'C0', 'muon': 'C1', 'hadronic': 'C2'}
    for branch, color in colors.items():
        for axis, column in zip(axes[:2], ['maximum_parent_separation_pitch', 'minimum_pair_separation_pitch']):
            values = np.sort(frame.loc[frame.tau_branch.eq(branch), column].dropna().to_numpy())
            values = values[values > 0]
            if len(values):
                axis.step(values, np.arange(1, len(values)+1)/len(values), where='post', color=color, label=f'{branch} (n={len(values)})')
    for axis, title in zip(axes[:2], ['Parent / daughter projected displacement', 'Closest charged daughter pair']):
        axis.axvline(1., color='black', linestyle='--', linewidth=.8)
        axis.set(xscale='log', xlabel='Separation / pixel pitch (max axis)', ylabel='Conditional cumulative fraction', title=title)
        axis.legend(fontsize=8)
    for i, row in enumerate(seed_means):
        item = row['metrics']['class_2_AP']
        axes[2].plot(item['CI95'], [i, i], color='C0')
        axes[2].plot(item['delta'], i, 'o', color='C0')
    axes[2].axvline(0., color='black', linestyle='--', linewidth=.8)
    axes[2].set(yticks=range(len(seed_means)), yticklabels=[r['variant'] for r in seed_means],
                xlabel='Tau AP delta relative to direct4', title='Exact-target overlap; three-seed mean')
    axes[2].xaxis.set_major_locator(MaxNLocator(4))
    axes[2].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    fig.suptitle('S5 validation: straight truth-ray geometry and E / vertex / isotope matching')
    save(fig, output, 'target_and_tau_geometry')


def saved_output_readback(root):
    """Independently join saved scores and geometry, then inspect the zero tail."""
    output = root/'evaluation/scope'
    report = json.loads((output/'summary.json').read_text())
    main = json.loads((root/'evaluation/final/summary.json').read_text())
    frame = pd.read_parquet(output/'tau_geometry.parquet')
    model = report['tau_geometry']['model']
    prediction = pd.read_parquet(root/'experiments'/model/'validation_predictions.parquet')
    joined = frame[KEY+['p2']].merge(prediction[KEY+['p2']], on=KEY, validate='one_to_one')
    difference = float(np.max(np.abs(joined.p2_x-joined.p2_y)))
    if len(joined) != 2500 or difference != 0.:
        raise ValueError('independent geometry/score readback differs')
    catalog = pd.read_parquet(root/'catalog.parquet', columns=KEY+['initE', 'initX', 'initY', 'pixel_channels'])
    frame = frame.merge(catalog, on=KEY, how='left', validate='one_to_one')
    groups = []
    for inside in [False, True]:
        group = frame[frame.projected_charged_in_pixel.ge(1).eq(inside)]
        groups.append({'forward_daughter_in_first_pixel': inside, 'events': len(group),
            'tau_score_q0_50_90_99_100': group.p2.quantile([0, .5, .9, .99, 1]).tolist(),
            'energy_GeV_q10_50_90': (group.initE/1000).quantile([.1, .5, .9]).tolist(),
            'pixel_channels_q10_50_90': group.pixel_channels.quantile([.1, .5, .9]).tolist(),
            'mean_initial_xy_mm': group[['initX', 'initY']].mean().tolist()})
    controls = []
    for name in ['s5_confirm8_direct4_n10000_s42', model, 's5_hgb_moments_n10000_s42']:
        scores = pd.read_parquet(root/'experiments'/name/'validation_predictions.parquet')
        scores = scores[scores.label.eq(2)].merge(frame[KEY+['projected_charged_in_pixel']], on=KEY, validate='one_to_one')
        entry = next(r for r in main['experiments'] if r['name'] == name)
        threshold = entry['operating_points']['inclusive']['FPR_0.01']['threshold']
        selected = scores.p2.ge(threshold)
        inside = scores.projected_charged_in_pixel.ge(1)
        controls.append({'name': name, 'threshold': threshold, 'all_tau_selected': int(selected.sum()),
                         'first_pixel_domain': binomial(int((selected & inside).sum()), int(inside.sum()))})
    readback = {'status': 'complete', 'events': len(joined), 'maximum_score_key_replay_difference': difference,
        'groups': groups, 'controls': controls, 'calibration_test_used': False,
        'finding': 'zero high-score tail in projected pixel domain persists after independent key join; scores vary and HGB selects some events',
        'unresolved': 'transverse-acceptance/occupancy associations and learned fine-topology extraction are not separated; no detector information-limit claim'}
    write_json(output/'geometry_crosscheck.json', readback)
    plot(output, frame, report['three_seed_comparisons'])
    return readback


def run(args):
    started = time.monotonic()
    output = args.root/'evaluation/scope'
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.root/'evaluation/final/summary.json').read_text())
    entries = []
    frames = {}
    for row in report['experiments']:
        frame = pd.read_parquet(args.root/'experiments'/row['name']/'validation_predictions.parquet').sort_values(KEY).reset_index(drop=True)
        if entries and not frame[KEY].equals(entries[0][1][KEY]):
            raise ValueError('S5 prediction event keys differ')
        entries.append((row['name'], frame))
        frames[row['name']] = frame
    common, support = target_support_weights(entries[0][1])
    if not common.any():
        raise ValueError('no joint energy/vertex/target overlap')
    weights, blocks = bootstrap_weights(entries[0][1], args.bootstrap)

    def calculate(entry):
        name, frame = entry
        cache = MetricCache(frame.label.to_numpy(), frame[[f'p{c}' for c in range(4)]].to_numpy())
        return name, cache.report(common), cache.compute(common)[0], np.stack([cache.compute(common*w)[0] for w in weights])

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = {r[0]: r for r in executor.map(calculate, entries)}
    comparisons = []
    for pair in report['comparisons']:
        a, b = results[pair['a']], results[pair['b']]
        comparisons.append({'cohort': pair['cohort'], 'a': a[0], 'b': b[0], 'metrics': interval_report(b[2]-a[2], b[3]-a[3])})
    seed_means = []
    for variant in ['six', 'vertex_kinematics']:
        pairs = [(results[f's5_confirm8_direct4_n10000_s{seed}'], results[f's5_confirm8_{variant}_n10000_s{seed}'])
                 for seed in [17, 42, 2026]]
        seed_means.append({'variant': variant, 'seeds': [17, 42, 2026], 'metrics': interval_report(
            np.mean([b[2]-a[2] for a, b in pairs], axis=0), np.mean([b[3]-a[3] for a, b in pairs], axis=0),
            np.stack([b[2]-a[2] for a, b in pairs]))})
    source_plan = json.loads((args.root/'plan.json').read_text())
    geometry = json.loads(Path(source_plan['geometry']).read_text())
    catalog = pd.read_parquet(args.root/'catalog.parquet', filters=[('split', '=', 'validation'), ('label', '=', 2)])
    tasks = [(source_plan['input_root'], geometry, group) for _, group in catalog.groupby(['run', 'chunk'], sort=True)]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        projection = pd.DataFrame([row for part in executor.map(extract_geometry, tasks) for row in part])
    truth = pd.read_parquet(args.root/'truth.parquet', columns=KEY+['tau_branch', 'tau_decay_location', 'tau_visible_pixel', 'tau_visible_any'])
    projection = projection.merge(truth, on=KEY, how='left', validate='one_to_one')
    model = 's5_confirm8_vertex_kinematics_n10000_s42'
    full = frames[model]
    tau_indices = full.index[full.label.eq(2)].to_numpy()
    tau_frame = full.loc[tau_indices, KEY+['p2']].merge(projection, on=KEY, validate='one_to_one')
    if len(tau_frame) != 2500:
        raise ValueError('incomplete accepted validation tau geometry')
    model_report = next(r for r in report['experiments'] if r['name'] == model)
    threshold = model_report['operating_points']['inclusive']['FPR_0.01']['threshold']
    selected = tau_frame.p2.to_numpy() >= threshold
    geometry_report = geometry_summary(tau_frame, selected, weights[:, tau_indices])
    pitch = [float(np.median(np.diff(geometry[k]))) for k in ['pixel_Xpos', 'pixel_Ypos']]
    geometry_report.update(model=model, threshold=threshold, threshold_background_events=7500,
        pixel_pitch_mm=pitch, source_chunks=len(tasks), events=len(tau_frame),
        scope='straight truth rays at first downstream pixel center; material/B-field/scattering/shower/association not modelled; >1 pitch is a geometry scale only',
        visibility_scope='existing stored response flags; not proof of two resolved tracks or a measured kink',
        quantization_sigma_mm=(np.array(pitch)/np.sqrt(12)).tolist(),
        quantization_scope='ideal uniform one-cell quantization reference; not a hardware-calibrated resolution')
    tau_frame.to_parquet(output/'tau_geometry.parquet', index=False, compression='gzip')
    result = {'status': 'complete', 'validation_events': len(full), 'experiments': [{'name': r[0], 'metrics': r[1]} for r in results.values()],
        'comparisons': comparisons, 'three_seed_comparisons': seed_means, 'target_common_support': support,
        'tau_geometry': geometry_report, 'bootstrap': {'replicates': args.bootstrap, 'blocks': blocks,
            'scope': 'fixed matching weights; physical-event blocks; conditional on selected models and reused validation'},
        'calibration_test_used': False, 'elapsed_seconds': time.monotonic()-started}
    write_json(output/'summary.json', result)
    pd.DataFrame([{'name': r[0], **r[1]} for r in results.values()]).to_csv(output/'target_matched_metrics.csv', index=False)
    saved_output_readback(args.root)
    print(json.dumps({'status': result['status'], 'models': len(results), 'pairs': len(comparisons),
                      'target_retained_events': support['retained_events'], 'tau_geometry_events': len(tau_frame),
                      'seconds': time.monotonic()-started}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--bootstrap', type=int, default=400)
    parser.add_argument('--readback-only', action='store_true')
    args = parser.parse_args()
    if args.readback_only:
        result = saved_output_readback(args.root)
        print(json.dumps({'status': result['status'], 'events': result['events'],
                          'maximum_score_key_replay_difference': result['maximum_score_key_replay_difference']}), flush=True)
    else:
        run(args)


if __name__ == '__main__':
    main()
