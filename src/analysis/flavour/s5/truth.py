"""Truth sidecars. None of this module's outputs are model observations.

Positions are mm, energies/momenta GeV. The G4 tau/charm tree contains
direct children, including non-decay secondaries; DecayXYZ is the child's
maximum-z trajectory point, NOT the parent decay vertex.
"""
from functools import lru_cache

import numpy as np

from analysis.flavour.s2.prepare import classify_tau_daughters


NEUTRINOS = {12, 14, 16, 18}
CHARM_PARENTS = {411, 421, 431, 4122, 4112, 4212, 4222, 4132, 4232}


@lru_cache(maxsize=2048)
def particle_info(pdg):
    from particle import Particle
    try:
        p = Particle.from_pdgid(int(pdg))
        return float(p.charge), float(p.mass or 0.) / 1000.
    except Exception:
        return np.nan, np.nan


def charmed(pdg):
    p = abs(int(pdg))
    return 100 < p < 10**9 and 4 in ((p // 10) % 10, (p // 100) % 10, (p // 1000) % 10)


def group_flat(tree, columns=None):
    arrays = tree.arrays(columns, library='np')
    ids = arrays['evtID']
    order = np.argsort(ids, kind='stable')
    bounds = np.r_[0, np.flatnonzero(np.diff(ids[order])) + 1, len(ids)]
    return {int(ids[order[a]]): {k: v[order[a:b]] for k, v in arrays.items()}
            for a, b in zip(bounds[:-1], bounds[1:]) if b > a}


def subset(data, keep):
    return {k: v[keep] for k, v in data.items()}


def positions(data, prefix):
    return np.column_stack([data[prefix + c] for c in 'XYZ'])


def primary_decay(children, parent_ids):
    if not children:
        return {}, None, False
    direct = subset(children, np.isin(children['PID'], parent_ids))
    if not len(direct['PDG']):
        return {}, None, False
    # Forward neutrino trajectories start at the decay. Ionisation electrons
    # along the parent trajectory have different origins and are excluded.
    neutrino = np.isin(np.abs(direct['PDG']), list(NEUTRINOS)) & (direct['Pz'] > 0)
    if not neutrino.any():
        return direct, None, False
    origins = positions(direct, 'Prod')[neutrino]
    vertex = np.median(origins, axis=0)
    agreement = bool(np.max(np.linalg.norm(origins - vertex, axis=1)) < .01)
    same_vertex = np.linalg.norm(positions(direct, 'Prod') - vertex, axis=1) < .01
    return subset(direct, same_vertex), vertex, agreement


def hit_summary(pixel, bars, tracker, direct_ids):
    output = {}
    for name, hit, flag in [('pixel', pixel, 'fromTau'), ('bar', bars, 'fromTau')]:
        marked = np.asarray(hit.get(flag, []), dtype=bool)
        output['tau_visible_' + name] = bool(marked.any())
        output['tau_' + name + '_channels_flagged'] = int(marked.sum())
    tracks = set(np.asarray(tracker.get('trackID', []), dtype=int).tolist())
    output['tau_visible_tracker'] = bool(tracks.intersection(direct_ids))
    seen = set()
    for hit in (pixel, bars, tracker):
        seen.update(np.asarray(hit.get('trackID', []), dtype=int).tolist())
    output['tau_direct_prongs_hit'] = len(seen.intersection(direct_ids))
    output['tau_visible_any'] = any(output['tau_visible_' + n] for n in ['pixel', 'bar', 'tracker'])
    return output


def event_truth(meta, primary, tau, charm, trajectories, pixel, bars, tracker, geometry):
    out = {k: meta[k] for k in ['run', 'chunk', 'row', 'evtID', 'label', 'split', 'family_hi', 'family_lo']}
    label = int(meta['label'])
    primary = primary or {'PDG': np.array([], dtype=int), 'trackID': np.array([], dtype=int)}
    tau_ids = primary['trackID'][np.abs(primary['PDG']) == 15]
    direct, decay, decay_valid = primary_decay(tau, tau_ids)
    branch = classify_tau_daughters(direct['PDG']) if direct else 'unknown'
    if label != 2:
        branch = 'not_tau'
    all_direct_branch = classify_tau_daughters(tau['PDG'][np.isin(tau['PID'], tau_ids)]) if tau else 'unknown'
    # Keep original S4 classification to expose disagreements, never impute.
    out.update(tau_branch=branch, tau_branch_catalog=meta['tau_branch'],
               tau_branch_all_children=all_direct_branch if label == 2 else 'not_tau',
               tau_branch_valid=branch in ['electron', 'muon', 'hadronic'])
    out['six_label'] = ({0: 0, 1: 1, 3: 5}.get(label, {'electron': 2, 'muon': 3, 'hadronic': 4}.get(branch, -1)))
    charged_ids = []
    prong_known = bool(direct)
    if direct:
        for pdg, tid in zip(direct['PDG'], direct['TID']):
            charge, _ = particle_info(int(pdg))
            prong_known &= np.isfinite(charge)
            if np.isfinite(charge) and charge != 0:
                charged_ids.append(int(tid))
    out.update(tau_direct_prongs=len(charged_ids) if prong_known else None,
               tau_prong_valid=bool(label == 2 and decay_valid and prong_known),
               tau_decay_valid=bool(label == 2 and decay_valid))
    vertex = np.array([meta['init' + c] for c in 'XYZ'])
    out.update(vertex_mm=vertex.tolist(), vertex_valid=bool(np.isfinite(vertex).all()))
    if label == 2 and decay_valid:
        x, y, z = decay
        in_pixel = geometry['pixel_Xpos'][0] <= x <= geometry['pixel_Xpos'][-1] and geometry['pixel_Ypos'][0] <= y <= geometry['pixel_Ypos'][-1]
        cx, cy = np.mean(geometry['scint_bar_Xcenter']), np.mean(geometry['scint_bar_Ycenter'])
        in_bar = abs(x-cx) <= geometry['scint_detector_width']/2 and abs(y-cy) <= geometry['scint_detector_height']/2
        in_z = min(geometry['tungsten_Zpos'])-geometry['tungsten_thickness']/2 <= z <= max(geometry['pixel_Zpos'])+geometry['silicon_thickness']/2
        location = 'outside_transverse' if not in_bar else 'upstream' if z < min(geometry['pixel_Zpos']) else 'downstream' if not in_z else 'gap'
        if in_bar and in_z:
            for key, thickness, name in [('tungsten_Zpos', 'tungsten_thickness', 'tungsten'), ('scint_Zpos', 'scint_thickness', 'scintillator'), ('pixel_Zpos', 'silicon_thickness', 'silicon')]:
                if np.min(np.abs(np.asarray(geometry[key])-z)) <= geometry[thickness]/2:
                    location = name
                    break
        out.update(tau_decay_vertex_mm=decay.tolist(), tau_decay_length_mm=float(np.linalg.norm(decay-vertex)),
                   tau_decay_z_mm=float(z), tau_decay_location=location, tau_decay_in_pixel_xy=bool(in_pixel),
                   tau_decay_in_bar_xy=bool(in_bar), tau_decay_contained=bool(in_bar and in_z),
                   tau_planes_after_decay=int(np.sum(np.asarray(geometry['pixel_Zpos']) > z)))
    else:
        out.update(tau_decay_vertex_mm=None, tau_decay_length_mm=None, tau_decay_z_mm=None,
                   tau_decay_location='undefined', tau_decay_in_pixel_xy=None, tau_decay_in_bar_xy=None,
                   tau_decay_contained=None, tau_planes_after_decay=None)
    out.update(hit_summary(pixel, bars, tracker, charged_ids))
    primary_charm = np.array([charmed(p) for p in primary['PDG']], dtype=bool)
    has_charm = bool(primary_charm.any())
    charm_pdgs = np.abs(charm['PDG']) if charm else np.array([])
    charm_parent_ids = np.unique(charm['PID']).tolist() if charm else []
    out.update(charm_presence=has_charm, charm_count=int(primary_charm.sum()),
               charm_label_valid=bool(len(primary['PDG'])), charm_tree_parent_count=len(charm_parent_ids),
               charm_tree_covers_primary=bool(not has_charm or all(abs(int(p)) in CHARM_PARENTS for p in primary['PDG'][primary_charm])),
               charm_e=bool((charm_pdgs == 11).any()), charm_mu=bool((charm_pdgs == 13).any()),
               charm_branch='muon' if (charm_pdgs == 13).any() else 'electron' if (charm_pdgs == 11).any() else 'hadronic' if has_charm else 'no_charm')
    # Visible primary-system proxy: G4 parentID=0, excluding nuclei/neutrinos;
    # replace tau with decay daughters where defined. It is not accepted energy.
    vis_p, vis_e, valid = np.zeros(3), 0., bool(primary and trajectories)
    original_ids = set(trajectories['trackTID'][trajectories['trackPID'] == 0].tolist()) if trajectories else set()
    if 'Px' in primary:
        for i, (pdg, tid) in enumerate(zip(primary['PDG'], primary['trackID'])):
            if int(tid) not in original_ids or abs(int(pdg)) in NEUTRINOS or abs(int(pdg)) >= 10**9:
                continue
            if abs(int(pdg)) == 15:
                if not decay_valid:
                    valid = False
                    continue
                for j, daughter in enumerate(direct['PDG']):
                    if abs(int(daughter)) in NEUTRINOS:
                        continue
                    _, mass = particle_info(int(daughter))
                    momentum = np.array([direct['P'+c][j] for c in 'xyz'])/1000.
                    valid &= bool(np.isfinite(mass))
                    vis_p += momentum
                    vis_e += np.sqrt(momentum @ momentum + mass**2) if np.isfinite(mass) else 0.
            else:
                vis_p += np.array([primary['P'+c][i] for c in 'xyz'])/1000.
                vis_e += float(primary['E'][i])/1000.
    out.update(visible_p3_GeV=vis_p.tolist(), visible_E_GeV=float(vis_e), visible_target_valid=bool(valid),
               visible_target_definition='non_nuclear_G4_initial_primaries_replace_tau_by_direct_decay_exclude_neutrinos')
    energy = np.asarray(bars.get('edep', []), dtype=float)
    pdg = np.abs(np.asarray(bars.get('pdg', []), dtype=int))
    semantic = np.array([energy[np.isin(pdg, [11, 22])].sum(), energy[pdg == 13].sum(), energy[~np.isin(pdg, [11, 22, 13])].sum()])
    out.update(bar_semantic_fractions=(semantic/max(semantic.sum(), 1e-30)).tolist(),
               bar_semantic_valid=bool(semantic.sum() > 0), tracker_truth_track_ids=sorted(set(np.asarray(tracker.get('trackID', []), dtype=int).tolist())),
               tracker_truth_rows=[np.asarray(tracker.get(k, [])).tolist() for k in ['trackerID', 'trackID', 'x', 'y', 'z']])
    # Pixel labels retain the known first-contributor limitation.
    out['pixel_truth_rows'] = [np.asarray(pixel.get(k, [])).tolist() for k in ['layerID', 'rowID', 'colID', 'trackID']]
    return out
