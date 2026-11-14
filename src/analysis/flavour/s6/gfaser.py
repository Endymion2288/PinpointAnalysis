"""Minimal current-GFaser bridge; units are metres and GeV in the input tree."""
from array import array

import numpy as np


def write_gfaser(records, destination):
    import ROOT
    ROOT.gROOT.SetBatch(True)
    output = ROOT.TFile(str(destination), 'RECREATE')
    tree = ROOT.TTree('gFaser', 'S6 fixed-primary transport input')
    integers = {key: array('i', [0]) for key in ['n', 'intType', 'scatteringType', 'Z', 'A', 'tgt', 'neu', 'fspl', 'hitnuc']}
    doubles = {key: array('d', [0.]) for key in ['vx', 'vy', 'vz', 'pxv', 'pyv', 'pzv', 'Ev', 'Q2', 'W', 'x', 'y', 'XSec']}
    booleans = {key: array('b', [0]) for key in ['cc', 'nc']}
    vectors = {key: ROOT.std.vector(kind)() for kind, keys in [
        ('int', ['pdgc', 'status']), ('double', ['px', 'py', 'pz', 'E']), ('string', ['name'])] for key in keys}
    for mapping, leaf in [(integers, 'I'), (doubles, 'D'), (booleans, 'O')]:
        for key, value in mapping.items():
            tree.Branch(key, value, f'{key}/{leaf}')
    for key, value in vectors.items():
        tree.Branch(key, value)
    for record in records:
        for mapping in [integers, doubles, booleans]:
            for key, value in mapping.items():
                value[0] = record[key]
        for key, value in vectors.items():
            value.clear()
            for item in record[key]:
                value.push_back(item)
        tree.Fill()
    tree.Write()
    output.Close()


def nominal_record(meta, primary):
    """G4 primaries are already status=1; retain float32 recorded momenta."""
    pdg = np.asarray(primary['PDG'])
    tgt = int(meta['tgtPDG'])
    return {'n': len(pdg), 'pdgc': pdg.astype(int).tolist(), 'status': [1]*len(pdg), 'name': ['']*len(pdg),
        **{key: (np.asarray(primary[source], dtype=float)/1000.).tolist()
           for key, source in [('px', 'Px'), ('py', 'Py'), ('pz', 'Pz'), ('E', 'E')]},
        **{key: float(meta[source])/1000. for key, source in [('vx', 'initX'), ('vy', 'initY'), ('vz', 'initZ'),
            ('pxv', 'initPx'), ('pyv', 'initPy'), ('pzv', 'initPz'), ('Ev', 'initE')]},
        'intType': int(meta['intType']), 'scatteringType': int(meta['scatteringType']),
        'Z': (tgt//10000)%1000, 'A': (tgt//10)%1000, 'tgt': tgt, 'neu': int(meta['initPDG']),
        'fspl': int(meta['fslPDG']), 'hitnuc': int(meta['hitnucPDG']),
        'cc': int(meta['intType']) == 2, 'nc': int(meta['intType']) == 3,
        **{key: float(meta[source]) for key, source in [('Q2', 'Q2'), ('W', 'W'), ('x', 'xBj'), ('y', 'y'), ('XSec', 'xs')]}}


def legacy_record(event, vertex_mm):
    """Old GENIE GFaser ancestry identifies the primary outgoing lepton.

    Missing cross section/scattering-mode metadata stays unknown; it is never
    used for flux or physical-yield weighting. All particle four-vectors are
    copied, while a common PINPOINT benchmark vertex replaces the old geometry.
    """
    pdg, status, mother = [np.asarray(event[k], dtype=int) for k in ['pdgc', 'status', 'firstMother']]
    incoming = np.flatnonzero((status == 0) & np.isin(np.abs(pdg), [12, 14, 16]))
    targets = np.flatnonzero((status == 0) & (pdg >= 1000000000) & (pdg < 1100000000))
    if len(incoming) != 1 or len(targets) != 1:
        raise ValueError('ambiguous incoming neutrino/target')
    i, target = int(incoming[0]), int(targets[0])
    outgoing = np.flatnonzero((status == 1) & (mother == i) & np.isin(np.abs(pdg), [11, 12, 13, 14, 15, 16]))
    if len(outgoing) != 1:
        raise ValueError('ambiguous primary outgoing lepton')
    j = int(outgoing[0])
    cc = pdg[j] == np.sign(pdg[i])*(abs(pdg[i])-1)
    nc = pdg[j] == pdg[i]
    if not (cc or nc):
        raise ValueError('unsupported primary lepton relation')
    incoming_p = np.asarray([event[k][i] for k in ['px', 'py', 'pz']])
    outgoing_p = np.asarray([event[k][j] for k in ['px', 'py', 'pz']])
    transfer_p = incoming_p-outgoing_p
    transfer_e = event['E'][i]-event['E'][j]
    q2 = float(transfer_p@transfer_p-transfer_e**2)
    target_pdg = int(pdg[target])
    return {'n': len(pdg), 'pdgc': pdg.tolist(), 'status': status.tolist(), 'name': ['']*len(pdg),
        **{k: list(map(float, event[k])) for k in ['px', 'py', 'pz', 'E']},
        **{k: float(v)/1000. for k, v in zip(['vx', 'vy', 'vz'], vertex_mm)},
        **{k: float(v) for k, v in zip(['pxv', 'pyv', 'pzv'], incoming_p)}, 'Ev': float(event['E'][i]),
        'Q2': q2, 'W': float('nan'), 'x': float('nan'), 'y': float(transfer_e/event['E'][i]), 'XSec': float('nan'),
        'intType': 2 if cc else 3, 'scatteringType': -1, 'cc': bool(cc), 'nc': bool(nc),
        'Z': (target_pdg//10000)%1000, 'A': (target_pdg//10)%1000, 'tgt': target_pdg,
        'neu': int(pdg[i]), 'fspl': int(pdg[j]), 'hitnuc': 0}
