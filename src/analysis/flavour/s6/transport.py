"""Isolated G4 transport: fixed input vertex, W density and tau-helicity proxies."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import time

from analysis.flavour.inventory import write_json


LCG107 = '/cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc11-opt'
G4_SOURCE = Path('/eos/home-x/xcheng/FASER/Pinpoint/Pinpoint_G4/Pinpoint')


def replace_once(path, original, changed):
    text = path.read_text()
    if text.count(original) != 1:
        raise ValueError(f'G4 source patch does not match {path.name}')
    path.write_text(text.replace(original, changed))


def lcg107(command, **kwargs):
    script = '\n'.join(['which() { command -v "$@"; }', f'source {LCG107}/setup.sh', 'set -e',
                         'export QT_QPA_PLATFORM=offscreen', f'export PYTHIA8DATA={LCG107}/share/Pythia8/xmldoc',
                         'exec '+shlex.join(list(map(str, command)))])
    return subprocess.run(['/bin/bash', '-c', script], check=True, **kwargs)


def build(args):
    started = time.monotonic()
    destination = args.root/'g4'
    destination.mkdir(exist_ok=True)
    scratch = Path(os.environ.get('PINPOINT_S6_SCRATCH', '/tmp'))/'g4_build'
    source = scratch/'source'
    source.mkdir(parents=True, exist_ok=True)
    for name in ['include', 'src', 'macros']:
        shutil.copytree(args.g4_source/name, source/name, dirs_exist_ok=True)
    for name in ['Pinpoint.cc', 'CMakeLists.txt']:
        shutil.copyfile(args.g4_source/name, source/name)
    generator = source/'src/generators/GFaserGenerator.cc'
    replace_once(generator, '#include <string>', '#include <string>\n#include <cstdlib>')
    replace_once(generator, '  fVz = GenerateRandomZVertex();',
        '  const char* s6InputZ = std::getenv("PINPOINT_USE_INPUT_Z");\n'
        '  if (!s6InputZ || std::string(s6InputZ) != "1") fVz = GenerateRandomZVertex();')
    detector = source/'src/DetectorConstruction.cc'
    replace_once(detector, '#include <iomanip>', '#include <iomanip>\n#include <cstdlib>\n#include "G4Material.hh"')
    replace_once(detector, '  G4Material* tungstenMaterial = nist->FindOrBuildMaterial("G4_W");',
        '  G4Material* tungstenMaterial = nist->FindOrBuildMaterial("G4_W");\n'
        '  if (const char* s6ScaleText = std::getenv("PINPOINT_W_DENSITY_SCALE")) {\n'
        '    const double s6Scale = std::atof(s6ScaleText);\n'
        '    tungstenMaterial = new G4Material("S6_W_density", tungstenMaterial->GetDensity()*s6Scale, tungstenMaterial);\n'
        '    G4cout << "S6 tungsten density scale = " << s6Scale << G4endl;\n'
        '  }')
    decayer = source/'src/Py8Decayer.cc'
    replace_once(decayer, '#include "Py8Decayer.hh"', '#include "Py8Decayer.hh"\n#include <cstdlib>')
    replace_once(decayer,
        '  fDecayer->event.back().pol(\n    round(std::cos(track.GetPolarization().angle(track.GetMomentumDirection()))));',
        '  double s6Polarization = round(std::cos(track.GetPolarization().angle(track.GetMomentumDirection())));\n'
        '  if (std::abs(pdgid) == 15) {\n'
        '    if (const char* s6Helicity = std::getenv("PINPOINT_TAU_HELICITY"))\n'
        '      s6Polarization = std::atof(s6Helicity)*(pdgid > 0 ? 1. : -1.);\n'
        '  }\n'
        '  fDecayer->event.back().pol(s6Polarization);')
    xml = Path(LCG107)/'share/Pythia8/xmldoc'
    replace_once(decayer, 'new Pythia("../share/Pythia8/xmldoc", false)', f'new Pythia("{xml}", false)')
    archive = destination/'source.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        stream.add(source, arcname='source')
    with (destination/'build.log').open('w') as log:
        lcg107(['cmake', '-S', source, '-B', scratch/'build'], stdout=log, stderr=subprocess.STDOUT)
        lcg107(['cmake', '--build', scratch/'build', '--parallel', args.cpus], stdout=log, stderr=subprocess.STDOUT)
    shutil.copyfile(scratch/'build/pinpoint', destination/'pinpoint_s6')
    (destination/'pinpoint_s6').chmod(0o755)
    shutil.copytree(source/'macros', destination/'macros', dirs_exist_ok=True)
    commit = subprocess.check_output(['git', '-C', str(args.g4_source), 'rev-parse', 'HEAD'], text=True).strip()
    write_json(destination/'build.json', {'status': 'complete', 'source': str(args.g4_source), 'source_commit': commit,
        'LCG': LCG107, 'seconds': time.monotonic()-started, 'cpus': args.cpus,
        'patches': ['optional fixed input z; original default preserved', 'optional W material density; geometry unchanged',
                    'optional charge-conjugate tau helicity, no charm override', 'absolute matching Pythia XML directory'],
        'tau_helicity_convention': 'parameter is tau-minus helicity; tau-plus gets opposite helicity',
        'original_source_modified': False, 'UTC': datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'build': 'complete', 'seconds': time.monotonic()-started}), flush=True)


def macro_text(input_file, output_file, geometry_macro, events, field=.57, seed=20261114):
    return '\n'.join(['/control/verbose 0', '/run/verbose 0', f'/control/execute {geometry_macro}',
        '/det/enableFaserSpectrometer true', '/gen/select gfaser', f'/gen/gfaser/inputFile {input_file}',
        '/gen/gfaser/hitPixelArea false', f'/out/fileName {output_file}', '/out/saveTrack false',
        '/out/saveTruthHits false', f'/random/setSeeds {seed} {seed+17}', '/run/initialize',
        f'/Pinpoint/field/value {field} tesla', f'/run/beamOn {events}'])+'\n'


def smoke(args):
    import uproot
    import numpy as np
    scratch = Path(os.environ.get('PINPOINT_S6_SCRATCH', '/tmp'))/'g4_smoke'
    scratch.mkdir(parents=True, exist_ok=True)
    output = scratch/'smoke.root'
    macro = scratch/'smoke.mac'
    macro.write_text(macro_text('/eos/project-f/fasersim-bonn/public/pinpoint/2025-11-26-genie/nue.gfaser.root',
        output, args.root/'g4/macros/geom.mac', 1))
    env = dict(os.environ, PINPOINT_USE_INPUT_Z='0')
    with (args.root/'g4/smoke.log').open('w') as log:
        lcg107([args.root/'g4/pinpoint_s6', macro], cwd=scratch, env=env, stdout=log, stderr=subprocess.STDOUT)
    expected = json.loads(Path(json.loads((args.root/'plan.json').read_text())['geometry']).read_text())
    with uproot.open(output) as file:
        actual = file['geometry'].arrays(library='ak')[0].to_list()
        comparisons = {key: bool(np.allclose(actual[key], expected[key], rtol=2e-6, atol=2e-4))
            if np.asarray(actual[key]).shape == np.asarray(expected[key]).shape else False for key in expected}
        result = {'events': file['event'].num_entries, 'geometry_comparisons': comparisons,
            'status': 'passed' if all(comparisons.values()) else 'geometry_review_required', 'test_used': False,
            'definition': 'interface geometry compatibility for resimulation, not re-certification of accepted samples'}
    write_json(args.root/'g4/smoke.json', result)
    output.unlink()
    print(json.dumps(result), flush=True)
    if result['status'] != 'passed':
        raise RuntimeError('resimulation geometry must match the supplied observation geometry')


def prepare(args):
    """Reconstruct status-one primaries from accepted ROOT records, at fixed z."""
    import pandas as pd
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import uproot
    from analysis.flavour.s6.prepare import require_freeze
    from analysis.flavour.s6.gfaser import nominal_record
    from analysis.flavour.s5.truth import group_flat
    from analysis.flavour.inventory import COMPRESSION
    require_freeze(args.root, 'test')
    plan = json.loads((args.root/'plan.json').read_text())
    manifest = pd.read_parquet(args.root/'transport_manifest.parquet').reset_index(drop=True)
    records = [None]*len(manifest)
    for (run, chunk), group in manifest.groupby(['run', 'chunk'], sort=True):
        path = Path(plan['input_root'])/str(run)/f'{run}_{chunk:03d}.root'
        with uproot.open(path) as source:
            metadata = source['event'].arrays(library='np')
            primary = group_flat(source['primaries'], ['evtID', 'PDG', 'Px', 'Py', 'Pz', 'E'])
            for index, row in group.iterrows():
                event = {k: v[int(row.row)] for k, v in metadata.items()}
                records[index] = nominal_record(event, primary[int(event['evtID'])])
    manifest['transport_chunk'] = np.arange(len(manifest))//args.events_per_task
    destination = args.root/'transport'
    destination.mkdir(exist_ok=True)
    manifest.to_parquet(destination/'manifest.parquet', index=False, compression=COMPRESSION)
    pq.write_table(pa.Table.from_pylist(records), destination/'particles.parquet', compression=COMPRESSION)
    write_json(destination/'inputs.json', {'status': 'complete', 'events': len(manifest),
        'chunks': manifest.transport_chunk.nunique(), 'source': 'accepted ROOT primaries and event metadata',
        'four_vector_precision': 'original primaries tree float32; no generator re-generation', 'vertex': 'original event double mm converted to GFaser metres',
        'events_per_task': args.events_per_task, 'test_used': True})


def transport_variants(config):
    t = config['transport']
    return [{'name': 'nominal', 'field_T': t['nominal_field_T'], 'density': 1., 'helicity': None}]+[
        {'name': f'field_{s:g}', 'field_T': t['nominal_field_T']*s, 'density': 1., 'helicity': None}
        for s in t['field_scales']]+[
        {'name': f'density_{s:g}', 'field_T': t['nominal_field_T'], 'density': s, 'helicity': None}
        for s in t['tungsten_density_scales']]+[
        {'name': f'tau_helicity_{s:g}', 'field_T': t['nominal_field_T'], 'density': 1., 'helicity': s}
        for s in t['tau_helicity_proxies']]


def convert_response(output, destination, metadata, geometry):
    import awkward as ak
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import uproot
    from analysis.flavour.create_data import BRANCHES, ROOT_TREES, lists
    from analysis.flavour.inventory import COMPRESSION
    from analysis.flavour.readout import pixel_channels, bar_channels, tracker_measurements, modality_state
    with uproot.open(output) as source:
        if source['event'].num_entries != len(metadata):
            raise ValueError('transport event coverage differs')
        event = source['event'].arrays(['evtID', 'initX', 'initY', 'initZ'], library='np')
        expected = metadata[['initX', 'initY', 'initZ']].to_numpy()
        actual = np.column_stack([event['init'+a] for a in 'XYZ'])
        if not np.allclose(actual, expected, atol=2e-4, rtol=0):
            raise ValueError('transport changed fixed input vertices')
        data = {n: source[ROOT_TREES[n]].arrays(k, library='ak') for n, k in BRANCHES.items()}
        observations, ideals, rows = [], [], []
        for i, meta in enumerate(metadata.to_dict('records')):
            h = {n: {k: ak.to_numpy(v[k][i]) for k in v.fields} for n, v in data.items()}
            pixel = pixel_channels(h['pixel'], geometry)
            bars, _ = bar_channels(h['bar'], geometry)
            seed = int(meta['family_hi']) ^ int(meta['family_lo']) ^ 20261111
            ideal, tracker = tracker_measurements(h['tracker'], seed)
            state = modality_state(pixel, bars, tracker)
            observations.append({'pixel': lists(pixel), 'bar': lists(bars), 'tracker_hypothesis': lists(tracker),
                'has_response': state['has_response'].tolist(), 'outside_acceptance': [-1, -1, -1],
                'fit_attempted': False, 'momentum_censored': -1})
            ideals.append(lists(ideal))
            rows.append({**meta, 'pixel_channels': len(pixel['channel_id']), 'bar_channels': len(bars['channel_id']),
                'tracker_channels': len(tracker['station']), 'all_empty': not state['has_response'].any()})
    destination.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(observations), destination/'observations.parquet', compression=COMPRESSION)
    pq.write_table(pa.Table.from_pylist([{'label': int(m['label'])} for m in rows]), destination/'targets.parquet', compression=COMPRESSION)
    pq.write_table(pa.Table.from_pylist(ideals), destination/'ideal_tracker.parquet', compression=COMPRESSION)
    pd.DataFrame(rows).to_parquet(destination/'provenance.parquet', index=False, compression=COMPRESSION)


def run(args):
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from analysis.flavour.s6.gfaser import write_gfaser
    from analysis.flavour.s6.prepare import require_freeze
    if args.scope == 'transport':
        require_freeze(args.root, 'test')
    config = json.loads((args.root/'conditions.json').read_text())
    plan = json.loads((args.root/'plan.json').read_text())
    geometry = json.loads(Path(plan['geometry']).read_text())
    base = args.root/args.scope
    manifest = pd.read_parquet(base/'manifest.parquet')
    if args.scope == 'transport':
        chosen = manifest.transport_chunk.eq(args.task)
    else:
        groups = manifest[['run', 'chunk']].drop_duplicates().reset_index(drop=True)
        spec = groups.iloc[args.task]
        chosen = manifest.run.eq(spec.run) & manifest.chunk.eq(spec.chunk)
    metadata = manifest[chosen].copy()
    records = pq.read_table(base/'particles.parquet').take(np.flatnonzero(chosen)).to_pylist()
    if args.scope == 'generator':
        import hashlib
        family = [int.from_bytes(hashlib.blake2b((p+':'+str(r)).encode(), digest_size=8).digest(), 'little')
                  for p, r in zip(metadata.source_path, metadata.source_row)]
        metadata['family_hi'], metadata['family_lo'] = np.asarray(family, dtype='uint64'), np.uint64(20261114)
    scratch = Path(os.environ.get('PINPOINT_S6_SCRATCH', '/tmp'))/f'{args.scope}_{args.task:04d}'
    scratch.mkdir(parents=True, exist_ok=True)
    # Write ROOT vectors using matching LCG107 PyROOT in a separate process.
    particles = scratch/'particles.json'
    particles.write_text(json.dumps(records))
    input_file = scratch/'input.root'
    lcg107(['python3', '-c', 'import json,sys; from analysis.flavour.s6.gfaser import write_gfaser; '
        'write_gfaser(json.load(open(sys.argv[1])),sys.argv[2])', particles, input_file])
    variants = transport_variants(config) if args.scope == 'transport' else [transport_variants(config)[0]]
    for variant in variants:
        if variant['helicity'] is not None and not metadata.label.eq(2).any():
            continue
        destination = base/'shards'/variant['name']/f'{args.task:04d}'
        if (destination/'summary.json').exists():
            continue
        tick = time.monotonic()
        output = scratch/(variant['name']+'.root')
        macro = scratch/(variant['name']+'.mac')
        macro.write_text(macro_text(input_file, output, args.root/'g4/macros/geom.mac', len(metadata),
            field=variant['field_T'], seed=20261114+args.task*31))
        env = dict(os.environ, PINPOINT_USE_INPUT_Z='1', PINPOINT_W_DENSITY_SCALE=str(variant['density']))
        if variant['helicity'] is not None:
            env['PINPOINT_TAU_HELICITY'] = str(variant['helicity'])
        log_path = args.root/'logs'/f'{args.scope}_{args.task:04d}_{variant["name"]}.g4.log'
        with log_path.open('w') as log:
            lcg107([args.root/'g4/pinpoint_s6', macro], cwd=scratch, env=env, stdout=log, stderr=subprocess.STDOUT)
        convert_response(output, destination, metadata, geometry)
        output_bytes = output.stat().st_size
        output.unlink()
        import gzip
        with log_path.open('rb') as original, gzip.open(str(log_path)+'.gz', 'wb') as compressed:
            shutil.copyfileobj(original, compressed)
        log_path.unlink()
        write_json(destination/'summary.json', {'status': 'complete', 'task': args.task, 'events': len(metadata),
            'variant': variant, 'seconds': time.monotonic()-tick, 'transient_ROOT_bytes': output_bytes,
            'ROOT_retained': False, 'fixed_input_vertices': True})
        print(json.dumps({'scope': args.scope, 'task': args.task, 'variant': variant['name'],
            'events': len(metadata), 'seconds': time.monotonic()-tick}), flush=True)
    shutil.rmtree(scratch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['build', 'smoke', 'prepare', 'run'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--g4-source', type=Path, default=G4_SOURCE)
    parser.add_argument('--cpus', type=int, default=4)
    parser.add_argument('--events-per-task', type=int, default=16)
    parser.add_argument('--task', type=int, default=0)
    parser.add_argument('--scope', choices=['transport', 'generator'], default='transport')
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
