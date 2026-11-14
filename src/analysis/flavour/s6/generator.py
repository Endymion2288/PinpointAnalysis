"""Bounded public GENIE Standard/NUX inventory and common-support benchmark."""
import argparse
import json
from pathlib import Path
import time

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import uproot

from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.s5.truth import charmed
from analysis.flavour.s6.gfaser import legacy_record


def scan(args):
    started = time.monotonic()
    config = json.loads((args.root/'conditions.json').read_text())['generator']
    metadata, selected_particles = [], {}
    failures, sources = [], []
    for tune in config['hadronization_configs']:
        for flux in config['flux_directories']:
            for file_id in range(args.files_per_flux):
                path = Path(config['source_root'])/tune/flux/str(file_id)/f'faser.{file_id}.gfaser.root'
                if not path.exists():
                    continue
                with uproot.open(path) as file:
                    tree = file['gFaser']
                    arrays = tree.arrays(['pdgc', 'status', 'firstMother', 'E'], library='ak')
                    p, s, mother, energy = [arrays[k] for k in ['pdgc', 'status', 'firstMother', 'E']]
                    neutrino = (abs(p) == 12) | (abs(p) == 14) | (abs(p) == 16)
                    charged_lepton = (abs(p) == 11) | (abs(p) == 13) | (abs(p) == 15)
                    nu = ak.to_numpy(ak.firsts(p[(s == 0) & neutrino]))
                    tgt = ak.to_numpy(ak.firsts(p[(s == 0) & (p >= 1000000000) & (p < 1100000000)]))
                    lepton = ak.to_numpy(ak.firsts(p[(s == 1) & (mother == 0) & (neutrino | charged_lepton)]))
                    ev = ak.to_numpy(ak.firsts(energy[(s == 0) & neutrino]))
                    cc = lepton == np.sign(nu)*(abs(nu)-1)
                    nc = lepton == nu
                    label = np.where(cc, (abs(nu)-12)//2, np.where(nc, 3, -1))
                    keep = ((tgt//10000)%1000 == config['target_Z']) & (label >= 0)
                    rows = np.flatnonzero(keep)
                    sources.append({'tune': tune, 'flux': flux, 'path': str(path), 'events': tree.num_entries,
                        'tungsten_classified': len(rows), 'class_counts': np.bincount(label[keep].astype(int), minlength=4).tolist()})
                    for i in rows:
                        metadata.append({'tune': tune, 'flux': flux, 'source_path': str(path), 'source_row': int(i),
                            'label': int(label[i]), 'initPDG': int(nu[i]), 'tgtPDG': int(tgt[i]), 'energy_GeV': float(ev[i]),
                            'energy_bin': int(np.digitize(ev[i], [0, 100, 300, 1000, 3000, 10000, np.inf])-1)})
                print(json.dumps({'generator_file': str(path), 'classified_W': len(rows)}), flush=True)
    frame = pd.DataFrame(metadata)
    cells = ['label', 'flux', 'initPDG', 'tgtPDG', 'energy_bin']
    counts = frame.groupby(cells+['tune']).size().unstack('tune', fill_value=0).reindex(columns=config['hadronization_configs'], fill_value=0)
    minimum = counts.min(1)
    supported = minimum[minimum > 0]
    rng = np.random.default_rng(20261114)
    benchmarks = pd.read_parquet(args.root/'manifest.parquet', filters=[('split', '=', 'calibration')])
    pieces = []
    for label in range(4):
        subset = supported[supported.index.get_level_values('label') == label]
        cell_multiset = np.repeat(np.arange(len(subset)), subset.to_numpy())
        chosen = rng.choice(cell_multiset, min(config['maximum_per_class_and_tune'], len(cell_multiset)), replace=False)
        quotas = np.bincount(chosen, minlength=len(subset))
        for cell, quota in zip(subset.index, quotas):
            if not quota:
                continue
            same = np.ones(len(frame), dtype=bool)
            for k, v in zip(cells, cell):
                same &= frame[k].eq(v)
            vertex = benchmarks[benchmarks.label.eq(label)].sample(n=int(quota), random_state=int(rng.integers(2**31)))
            for tune in config['hadronization_configs']:
                candidates = frame[same & frame.tune.eq(tune)].sample(n=int(quota), random_state=int(rng.integers(2**31))).copy()
                for a in 'XYZ':
                    candidates['init'+a] = vertex['init'+a].to_numpy()
                candidates['benchmark_cell'] = str(cell)
                candidates['benchmark_slot'] = np.arange(len(candidates))
                pieces.append(candidates)
    selected = pd.concat(pieces, ignore_index=True).sort_values(['tune', 'label', 'source_path', 'source_row']).reset_index(drop=True)
    records = [None]*len(selected)
    for path, group in selected.groupby('source_path'):
        with uproot.open(path) as file:
            tree = file['gFaser']
            arrays = tree.arrays(['pdgc', 'status', 'firstMother', 'px', 'py', 'pz', 'E'], library='ak')
            for i, row in group.iterrows():
                event = ak.to_list(arrays[int(row.source_row)])
                record = legacy_record(event, [row.initX, row.initY, row.initZ])
                if record['neu'] != row.initPDG or record['tgt'] != row.tgtPDG:
                    raise ValueError('legacy bridge metadata mismatch')
                records[i] = record
                selected.loc[i, 'charm_primary'] = any(charmed(p) for p, s in zip(record['pdgc'], record['status']) if s == 1)
    destination = args.root/'generator'
    destination.mkdir(exist_ok=True)
    selected['run'] = np.where(selected.tune.eq(config['hadronization_configs'][0]), 20001, 20002)
    selected['row'] = selected.groupby('tune').cumcount()
    selected['chunk'] = selected['row']//16
    selected['row'] %= 16
    selected.to_parquet(destination/'manifest.parquet', index=False, compression=COMPRESSION)
    pq.write_table(pa.Table.from_pylist(records), destination/'particles.parquet', compression=COMPRESSION)
    counts.reset_index().to_parquet(destination/'support_cells.parquet', index=False, compression=COMPRESSION)
    write_json(destination/'inventory.json', {'status': 'complete', 'files': sources, 'seconds': time.monotonic()-started,
        'selected_counts': selected.groupby(['tune', 'label']).size().rename('events').reset_index().to_dict('records'),
        'matched_columns': cells, 'vertices': 'identical benchmark vertex slots in each matched cell, sampled from accepted calibration metadata',
        'original_generator_events_paired': False, 'comparison': 'GENIE Pythia6 Standard versus NUX hadronization settings, not NuWro',
        'event_weights': 'unit within exactly matched coarse cells; Neff equals selected event count',
        'missing_generator_fields': ['XSec', 'x', 'W', 'scatteringType', 'hitnuc'],
        'generator_truth_unknowns_not_used_as_model_input': True, 'test_response_used': False})
    print(json.dumps({'selected_events': len(selected), 'seconds': time.monotonic()-started}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['scan'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--files-per-flux', type=int, default=20)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
