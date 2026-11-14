"""Geometry labels for evaluation: ROOT stores silicon thickness in micrometres."""
import numpy as np


def tau_material_labels(frame, geometry):
    result = frame.copy()
    selected = result.index[result.label.eq(2) & result.tau_decay_valid]
    if not len(selected):
        return result, {'valid_tau_decays': 0, 'material_labels_changed': 0}
    xyz = np.stack(result.loc[selected, 'tau_decay_vertex_mm'])
    z = xyz[:, 2]
    cx, cy = np.mean(geometry['scint_bar_Xcenter']), np.mean(geometry['scint_bar_Ycenter'])
    inside = (np.abs(xyz[:, 0]-cx) <= geometry['scint_detector_width']/2) & (np.abs(xyz[:, 1]-cy) <= geometry['scint_detector_height']/2)
    silicon_mm = geometry['silicon_thickness']*.001
    first_z = min(min(geometry[key])-width/2 for key, width in [
        ('tungsten_Zpos', geometry['tungsten_thickness']), ('scint_Zpos', geometry['scint_thickness']),
        ('pixel_Zpos', silicon_mm)])
    last_z = max(max(geometry[key])+width/2 for key, width in [
        ('tungsten_Zpos', geometry['tungsten_thickness']), ('scint_Zpos', geometry['scint_thickness']),
        ('pixel_Zpos', silicon_mm)])
    in_z = (z >= first_z) & (z <= last_z)
    names = np.full(len(z), 'gap', dtype=object)
    names[~in_z] = 'downstream'
    names[z < first_z] = 'upstream'
    names[~inside] = 'outside_transverse'
    assigned = np.zeros(len(z), dtype=bool)
    for positions, width, name in [('tungsten_Zpos', geometry['tungsten_thickness'], 'tungsten'),
                                   ('scint_Zpos', geometry['scint_thickness'], 'scintillator'),
                                   ('pixel_Zpos', silicon_mm, 'silicon')]:
        matched = (np.abs(z[:, None]-np.asarray(geometry[positions])).min(1) <= width/2) & inside & in_z & ~assigned
        names[matched] = name
        assigned |= matched
    changed = int(np.count_nonzero(result.loc[selected, 'tau_decay_location'].to_numpy() != names))
    result.loc[selected, 'tau_decay_location'] = names
    result.loc[selected, 'tau_decay_contained'] = inside & in_z
    return result, {'valid_tau_decays': len(selected), 'material_labels_changed': changed,
                    'silicon_thickness_header_um': geometry['silicon_thickness'], 'silicon_thickness_mm_used': silicon_mm,
                    'definition': 'S6 evaluation labels use mm consistently; no observation, class label or model change'}
