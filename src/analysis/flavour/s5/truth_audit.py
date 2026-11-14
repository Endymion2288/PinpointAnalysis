"""Explain the finite list of S5 versus catalog tau-branch disagreements."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

from analysis.flavour.inventory import write_json
from analysis.flavour.s5.truth import group_flat, positions, primary_decay


def run(root):
    columns = ['run', 'chunk', 'row', 'evtID', 'split', 'tau_branch', 'tau_branch_catalog', 'tau_decay_valid']
    frame = pd.read_parquet(root/'truth.parquet', columns=columns)
    frame = frame[frame.tau_branch.ne(frame.tau_branch_catalog)]
    plan = json.loads((root/'plan.json').read_text())
    rows = []
    for (run, chunk), events in frame.groupby(['run', 'chunk'], sort=True):
        path = Path(plan['input_root'])/str(run)/f'{run}_{chunk:03d}.root'
        with uproot.open(path) as source:
            parents = group_flat(source['primaries'], ['evtID', 'PDG', 'trackID'])
            children = group_flat(source['tau'], ['evtID', 'PID', 'TID', 'PDG', 'ProdX', 'ProdY', 'ProdZ', 'Pz'])
        for meta in events.to_dict('records'):
            primary = parents[meta['evtID']]
            ids = primary['trackID'][np.abs(primary['PDG']) == 15]
            all_children = children.get(meta['evtID'], {})
            _, vertex, valid = primary_decay(all_children, ids)
            keep = np.isin(all_children['PID'], ids) if all_children else np.zeros(0, bool)
            children_rows = []
            if all_children:
                origins = positions(all_children, 'Prod')
                for i in np.flatnonzero(keep):
                    distance = float(np.linalg.norm(origins[i]-vertex)) if vertex is not None else None
                    children_rows.append({'trackID': int(all_children['TID'][i]), 'PDG': int(all_children['PDG'][i]),
                        'Pz_MeV': float(all_children['Pz'][i]), 'stored_min_z_position_mm': origins[i].tolist(),
                        'distance_to_anchor_mm': distance, 'retained_at_anchor': distance is not None and distance < .01})
            rejected_backward = [r for r in children_rows if r['Pz_MeV'] <= 0 and not r['retained_at_anchor']]
            reason = ('no_consistent_forward_neutrino_anchor' if not valid else
                      'backward_daughter_stored_min_z_differs_from_decay_anchor' if rejected_backward else
                      'stored_daughter_origins_do_not_all_match_anchor')
            rows.append({**meta, 'reason': reason, 'anchor_mm': vertex.tolist() if vertex is not None else None,
                         'direct_children': children_rows})
    output = {'events': len(rows), 'reason_counts': pd.Series([r['reason'] for r in rows], dtype='object').value_counts().to_dict(),
              'rows': rows, 'decision': 'retain unknown branch mask; stored minimum-z extrema do not establish particle birth positions',
              'calibration_test_used': False}
    write_json(root/'truth_disagreements.json', output)
    print(json.dumps({k: v for k, v in output.items() if k != 'rows'}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)


if __name__ == '__main__':
    main()
